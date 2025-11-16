"""
ONNX Model Conversion Utilities
Converts PyTorch models to ONNX for faster inference
"""

import torch
import os

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("❌ ONNX not installed. Install with: pip install onnx onnxruntime")
    ONNX_AVAILABLE = False
    exit(1)


def convert_pytorch_to_onnx(
    pytorch_model_path: str,
    onnx_output_path: str,
    input_shape: tuple,
    model_class=None,
    dynamic_axes=None
):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        pytorch_model_path: Path to saved PyTorch model (.pt file)
        onnx_output_path: Where to save ONNX model (.onnx file)
        input_shape: Shape of input tensor (e.g., (1, 7, 4) for LSTM)
        model_class: PyTorch model class (if custom model)
        dynamic_axes: Dict of dynamic axes for variable batch size
    
    Example:
        convert_pytorch_to_onnx(
            "deterioration_lstm.pt",
            "deterioration_lstm.onnx",
            input_shape=(1, 7, 4),  # (batch, sequence_length, features)
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    """
    print(f"🔄 Converting {pytorch_model_path} to ONNX...")
    
    # Load PyTorch model
    checkpoint = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
    
    if model_class is None:
        # Try to load from checkpoint
        if 'model_class' in checkpoint:
            model_class = checkpoint['model_class']
        else:
            raise ValueError("model_class must be provided if not in checkpoint")
    
    # Initialize model
    if 'model_state_dict' in checkpoint:
        # Load from state dict
        model = model_class(**checkpoint.get('model_config', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume checkpoint IS the model
        model = checkpoint
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes or {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model saved to {onnx_output_path}")
    
    # Verify ONNX model
    verify_onnx_model(onnx_output_path, dummy_input.numpy())


def verify_onnx_model(onnx_path: str, test_input):
    """
    Verify ONNX model loads and runs correctly
    
    Args:
        onnx_path: Path to ONNX model
        test_input: Numpy array for testing
    """
    print("\n🔍 Verifying ONNX model...")
    
    # Check model validity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")
    
    # Test inference
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input/output names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # Run inference
    outputs = ort_session.run([output_name], {input_name: test_input})
    
    print(f"✅ ONNX inference successful")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {outputs[0].shape}")
    print(f"   Sample output: {outputs[0][0]}")
    
    # Compare file sizes
    import os
    pytorch_size = os.path.getsize(onnx_path.replace('.onnx', '.pt')) / 1024 / 1024
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    
    print(f"\n📊 Model Comparison:")
    print(f"   ONNX size: {onnx_size:.2f} MB")
    print(f"   Compression: {(1 - onnx_size/pytorch_size)*100:.1f}% smaller" 
          if pytorch_size > onnx_size else "")


def benchmark_inference_speed(onnx_path: str, pytorch_model_path: str, input_shape: tuple, n_runs=100):
    """
    Compare inference speed between PyTorch and ONNX
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model_path: Path to PyTorch model
        input_shape: Input tensor shape
        n_runs: Number of inference runs for benchmarking
    """
    import time
    import numpy as np
    
    print(f"\n⚡ Benchmarking inference speed ({n_runs} runs)...")
    
    # Prepare test data
    test_input_pt = torch.randn(*input_shape)
    test_input_np = test_input_pt.numpy()
    
    # Benchmark PyTorch
    pt_model = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
    if isinstance(pt_model, dict) and 'model_state_dict' in pt_model:
        # Need to reconstruct model - skip PyTorch benchmark
        print("⚠️  PyTorch model requires reconstruction - skipping PyTorch benchmark")
        pt_times = [0]
    else:
        pt_model.eval()
        pt_times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.time()
                _ = pt_model(test_input_pt)
                pt_times.append(time.time() - start)
    
    # Benchmark ONNX
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    onnx_times = []
    
    for _ in range(n_runs):
        start = time.time()
        _ = ort_session.run(None, {input_name: test_input_np})
        onnx_times.append(time.time() - start)
    
    # Calculate statistics
    pt_avg = np.mean(pt_times) * 1000  # Convert to ms
    onnx_avg = np.mean(onnx_times) * 1000
    
    print(f"\n📊 Benchmark Results:")
    print(f"   PyTorch: {pt_avg:.2f} ms/inference")
    print(f"   ONNX: {onnx_avg:.2f} ms/inference")
    if pt_avg > 0:
        speedup = pt_avg / onnx_avg
        print(f"   Speedup: {speedup:.2f}x faster with ONNX")


def main():
    """Example usage"""
    print("🚀 ONNX Conversion Utility\n")
    
    # Example: Convert deterioration LSTM model
    pytorch_path = "./ml_models/deterioration_lstm.pt"
    onnx_path = "./ml_models/deterioration_lstm.onnx"
    
    if not os.path.exists(pytorch_path):
        print(f"❌ Model not found: {pytorch_path}")
        print("   Train the model first using train_deterioration_model.py")
        return
    
    # Import model class
    import sys
    sys.path.append('..')
    from ml_scripts.train_deterioration_model import DeteriorationLSTM
    
    # Convert
    convert_pytorch_to_onnx(
        pytorch_path,
        onnx_path,
        input_shape=(1, 7, 4),  # (batch, sequence_length, features)
        model_class=DeteriorationLSTM
    )
    
    # Benchmark
    benchmark_inference_speed(onnx_path, pytorch_path, (1, 7, 4))
    
    print("\n✅ Conversion complete!")
    print(f"   Use {onnx_path} for production inference")


if __name__ == "__main__":
    if not ONNX_AVAILABLE:
        print("ONNX not available. Exiting.")
        exit(1)
    main()
