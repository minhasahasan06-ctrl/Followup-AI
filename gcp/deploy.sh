#!/bin/bash
# Followup AI Unified Backend - GCP Deployment Script
#
# Deploys the complete backend (Express + FastAPI) to Google Cloud Run.
# Defaults to CPU-only with scale-to-zero for minimal cost.
# Add --gpu flag to enable GPU acceleration for production workloads.
#
# Prerequisites:
# 1. gcloud CLI installed and authenticated
# 2. Docker installed (for local builds)
# 3. Project configured with required APIs enabled
#
# Usage:
#   # CPU-only deployment (scale-to-zero, ~$0 when idle)
#   ./gcp/deploy.sh PROJECT_ID us-central1
#
#   # GPU deployment (for paid customers)
#   ./gcp/deploy.sh PROJECT_ID us-central1 --gpu
#
# Examples:
#   ./gcp/deploy.sh my-project-id us-central1
#   ./gcp/deploy.sh my-project-id us-central1 --gpu

set -euo pipefail

# Parse arguments
ENABLE_GPU=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            ENABLE_GPU=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./gcp/deploy.sh [PROJECT_ID] [REGION] [--gpu]"
            echo ""
            echo "Arguments:"
            echo "  PROJECT_ID  GCP project ID (optional, uses current gcloud config)"
            echo "  REGION      GCP region (optional, defaults to us-central1)"
            echo ""
            echo "Flags:"
            echo "  --gpu       Enable GPU acceleration (NVIDIA L4)"
            echo "  --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./gcp/deploy.sh                           # CPU-only, current project"
            echo "  ./gcp/deploy.sh my-project us-central1    # CPU-only, specific project"
            echo "  ./gcp/deploy.sh my-project us-central1 --gpu  # With GPU"
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

# Configuration
PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null || echo "")}"
REGION="${2:-us-central1}"
SERVICE_NAME="followup-backend"
REPOSITORY="followup-ai"
IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")

# Resource configuration based on GPU mode
if [ "$ENABLE_GPU" = true ]; then
    MEMORY="8Gi"
    CPU="4"
    MIN_INSTANCES="1"
    WORKERS="4"
    GPU_FLAGS="--gpu=1 --gpu-type=nvidia-l4"
else
    MEMORY="4Gi"
    CPU="2"
    MIN_INSTANCES="0"
    WORKERS="2"
    GPU_FLAGS=""
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Validate prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    if [ -z "$PROJECT_ID" ]; then
        log_error "PROJECT_ID not set. Please provide as argument or set with: gcloud config set project PROJECT_ID"
        exit 1
    fi
    
    echo ""
    log_info "Configuration:"
    log_info "  Project:       $PROJECT_ID"
    log_info "  Region:        $REGION"
    log_info "  Service:       $SERVICE_NAME"
    log_info "  Image tag:     $IMAGE_TAG"
    log_info "  GPU enabled:   $ENABLE_GPU"
    log_info "  Memory:        $MEMORY"
    log_info "  CPU:           $CPU"
    log_info "  Min instances: $MIN_INSTANCES"
    echo ""
}

# Enable required APIs
enable_apis() {
    log_step "Enabling required APIs..."
    
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        secretmanager.googleapis.com \
        --project="$PROJECT_ID" \
        --quiet
    
    log_info "APIs enabled successfully"
}

# Create Artifact Registry repository
create_repository() {
    log_step "Creating Artifact Registry repository..."
    
    if ! gcloud artifacts repositories describe "$REPOSITORY" \
        --location="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        
        gcloud artifacts repositories create "$REPOSITORY" \
            --repository-format=docker \
            --location="$REGION" \
            --description="Followup AI Docker images" \
            --project="$PROJECT_ID"
        
        log_info "Repository created: $REPOSITORY"
    else
        log_info "Repository already exists: $REPOSITORY"
    fi
}

# Create service account with required permissions
create_service_account() {
    local sa_name="${SERVICE_NAME}"
    local sa_email="${sa_name}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    log_step "Setting up service account..."
    
    if ! gcloud iam service-accounts describe "$sa_email" \
        --project="$PROJECT_ID" &> /dev/null; then
        
        gcloud iam service-accounts create "$sa_name" \
            --display-name="Followup Backend Service Account" \
            --description="Service account for Followup AI unified backend" \
            --project="$PROJECT_ID"
        
        log_info "Service account created: $sa_email"
    else
        log_info "Service account already exists: $sa_email"
    fi
    
    log_info "Granting IAM roles..."
    
    for role in \
        "roles/secretmanager.secretAccessor" \
        "roles/cloudsql.client" \
        "roles/storage.objectViewer"; do
        
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$sa_email" \
            --role="$role" \
            --condition=None \
            --quiet 2>/dev/null || true
    done
    
    log_info "IAM roles granted"
}

# Create secrets in Secret Manager
setup_secrets() {
    log_step "Setting up secrets..."
    
    local secrets=(
        "DATABASE_URL"
        "STYTCH_PROJECT_ID"
        "STYTCH_SECRET"
        "OPENAI_API_KEY"
        "OPENAI_BAA"
        "OPENAI_ZDR"
    )
    
    local missing_secrets=()
    
    for secret in "${secrets[@]}"; do
        if ! gcloud secrets describe "$secret" --project="$PROJECT_ID" &> /dev/null; then
            missing_secrets+=("$secret")
        else
            log_info "Secret exists: $secret"
        fi
    done
    
    if [ ${#missing_secrets[@]} -gt 0 ]; then
        log_warn "Missing secrets: ${missing_secrets[*]}"
        log_warn "Please create these secrets manually with:"
        for secret in "${missing_secrets[@]}"; do
            echo "  echo -n 'YOUR_VALUE' | gcloud secrets create $secret --data-file=- --project=$PROJECT_ID"
        done
        echo ""
        read -p "Press Enter to continue after creating secrets, or Ctrl+C to abort..."
    fi
}

# Build and push Docker image
build_and_push() {
    log_step "Building and pushing Docker image..."
    
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}"
    
    # Configure Docker for Artifact Registry
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    
    # Build unified image with CUDA support
    log_info "Building image (this may take 10-15 minutes for first build)..."
    docker build \
        -t "${image_uri}:${IMAGE_TAG}" \
        -t "${image_uri}:latest" \
        -f gcp/Dockerfile.unified \
        .
    
    # Push image
    log_info "Pushing image to Artifact Registry..."
    docker push "${image_uri}:${IMAGE_TAG}"
    docker push "${image_uri}:latest"
    
    log_info "Image pushed: ${image_uri}:${IMAGE_TAG}"
}

# Deploy to Cloud Run
deploy_service() {
    log_step "Deploying to Cloud Run..."
    
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG}"
    local sa_email="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    # Build CORS origins
    local cors_origins=""
    local existing_url=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format='value(status.url)' 2>/dev/null || echo "")
    
    if [ -n "$existing_url" ]; then
        cors_origins="$existing_url"
    fi
    
    if [ -n "${REPLIT_FRONTEND_URL:-}" ]; then
        cors_origins="${cors_origins:+$cors_origins,}$REPLIT_FRONTEND_URL"
    else
        cors_origins="${cors_origins:+$cors_origins,}https://followup-ai.replit.app"
    fi
    
    # Deploy with appropriate configuration
    if [ "$ENABLE_GPU" = true ]; then
        log_info "Deploying with GPU acceleration (NVIDIA L4)..."
        gcloud run deploy "$SERVICE_NAME" \
            --image="$image_uri" \
            --region="$REGION" \
            --platform=managed \
            --allow-unauthenticated \
            --min-instances="$MIN_INSTANCES" \
            --max-instances=10 \
            --memory="$MEMORY" \
            --cpu="$CPU" \
            --gpu=1 \
            --gpu-type=nvidia-l4 \
            --timeout=300 \
            --concurrency=80 \
            --port=5000 \
            --set-env-vars="ENV=production,NODE_ENV=production,LOG_LEVEL=INFO,CORS_ALLOWED_ORIGINS=$cors_origins,WORKERS=$WORKERS,PYTHON_BACKEND_URL=http://127.0.0.1:8000" \
            --set-secrets="DATABASE_URL=DATABASE_URL:latest,STYTCH_PROJECT_ID=STYTCH_PROJECT_ID:latest,STYTCH_SECRET=STYTCH_SECRET:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,OPENAI_BAA=OPENAI_BAA:latest,OPENAI_ZDR=OPENAI_ZDR:latest" \
            --service-account="$sa_email" \
            --project="$PROJECT_ID"
    else
        log_info "Deploying CPU-only (scale-to-zero enabled)..."
        gcloud run deploy "$SERVICE_NAME" \
            --image="$image_uri" \
            --region="$REGION" \
            --platform=managed \
            --allow-unauthenticated \
            --min-instances="$MIN_INSTANCES" \
            --max-instances=10 \
            --memory="$MEMORY" \
            --cpu="$CPU" \
            --timeout=300 \
            --concurrency=80 \
            --port=5000 \
            --set-env-vars="ENV=production,NODE_ENV=production,LOG_LEVEL=INFO,CORS_ALLOWED_ORIGINS=$cors_origins,WORKERS=$WORKERS,PYTHON_BACKEND_URL=http://127.0.0.1:8000" \
            --set-secrets="DATABASE_URL=DATABASE_URL:latest,STYTCH_PROJECT_ID=STYTCH_PROJECT_ID:latest,STYTCH_SECRET=STYTCH_SECRET:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,OPENAI_BAA=OPENAI_BAA:latest,OPENAI_ZDR=OPENAI_ZDR:latest" \
            --service-account="$sa_email" \
            --project="$PROJECT_ID"
    fi
    
    log_info "Deployment complete!"
}

# Display service URL and next steps
show_completion() {
    local url=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format='value(status.url)')
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Deployment Successful!${NC}"
    echo "=========================================="
    echo ""
    echo "Service URL: $url"
    echo ""
    echo "Mode: $([ "$ENABLE_GPU" = true ] && echo "GPU (NVIDIA L4)" || echo "CPU-only (scale-to-zero)")"
    echo ""
    echo "Next steps:"
    echo "1. Update your Replit frontend secrets:"
    echo "   VITE_API_URL=$url"
    echo ""
    echo "2. Restart your Replit application"
    echo ""
    if [ "$ENABLE_GPU" = false ]; then
        echo "To enable GPU later, run:"
        echo "   ./gcp/deploy.sh $PROJECT_ID $REGION --gpu"
        echo ""
    fi
    echo "=========================================="
}

# Main execution
main() {
    echo ""
    echo "=========================================="
    echo "Followup AI - Unified Backend Deployment"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    enable_apis
    create_repository
    create_service_account
    setup_secrets
    build_and_push
    deploy_service
    show_completion
}

# Run main function
main "$@"
