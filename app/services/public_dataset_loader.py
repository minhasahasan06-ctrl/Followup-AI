"""
Public Dataset Loader Service
Integration with public health datasets for ML model training:
- MIMIC-III (PhysioNet) - ICU patient data
- PhysioNet Challenge datasets - Various health monitoring data
- Synthetic data generation for augmentation
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a public dataset"""
    dataset_name: str
    source: str
    version: str
    record_count: int
    patient_count: int
    data_types: List[str]
    file_formats: List[str]
    requires_credentials: bool
    license: str
    citation: str
    description: str = ""
    access_url: str = ""
    documentation_url: str = ""
    total_size_gb: float = 0.0


@dataclass
class DatasetRecord:
    """A single record from a public dataset"""
    record_id: str
    dataset_source: str
    features: Dict[str, Any]
    label: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


SUPPORTED_DATASETS = {
    "mimic_iii": DatasetMetadata(
        dataset_name="MIMIC-III",
        source="physionet",
        version="1.4",
        record_count=58976,
        patient_count=46520,
        data_types=["vitals", "lab_results", "medications", "diagnoses", "procedures"],
        file_formats=["csv", "gz"],
        requires_credentials=True,
        license="PhysioNet Credentialed Health Data License",
        citation="Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database.",
        description="Critical care database containing de-identified health data for ~60,000 ICU stays",
        access_url="https://physionet.org/content/mimiciii/1.4/",
        documentation_url="https://mimic.mit.edu/docs/iii/",
        total_size_gb=6.2
    ),
    "mimic_iv": DatasetMetadata(
        dataset_name="MIMIC-IV",
        source="physionet",
        version="2.2",
        record_count=180733,
        patient_count=299712,
        data_types=["vitals", "lab_results", "medications", "diagnoses", "ed_visits", "icu_stays"],
        file_formats=["csv", "gz"],
        requires_credentials=True,
        license="PhysioNet Credentialed Health Data License",
        citation="Johnson, A., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV.",
        description="Updated critical care database with expanded ED and hospital data",
        access_url="https://physionet.org/content/mimiciv/2.2/",
        documentation_url="https://mimic.mit.edu/docs/iv/",
        total_size_gb=7.8
    ),
    "eicu": DatasetMetadata(
        dataset_name="eICU Collaborative Research Database",
        source="physionet",
        version="2.0",
        record_count=200859,
        patient_count=139367,
        data_types=["vitals", "lab_results", "medications", "diagnoses", "nurse_assessments"],
        file_formats=["csv", "gz"],
        requires_credentials=True,
        license="PhysioNet Credentialed Health Data License",
        citation="Pollard, T., Johnson, A., et al. (2018). eICU Collaborative Research Database.",
        description="Multi-center ICU database from 208 US hospitals",
        access_url="https://physionet.org/content/eicu-crd/2.0/",
        documentation_url="https://eicu-crd.mit.edu/",
        total_size_gb=3.1
    ),
    "physionet_challenge_2019": DatasetMetadata(
        dataset_name="PhysioNet Challenge 2019 - Sepsis",
        source="physionet",
        version="1.0.0",
        record_count=40336,
        patient_count=40336,
        data_types=["vitals", "lab_results", "sepsis_labels"],
        file_formats=["psv"],
        requires_credentials=False,
        license="Open Data Commons ODC-By",
        citation="Reyna, M., Josef, C., Jeter, R., et al. (2020). Early Prediction of Sepsis.",
        description="ICU patient data with hourly labels for sepsis prediction",
        access_url="https://physionet.org/content/challenge-2019/1.0.0/",
        documentation_url="https://physionet.org/content/challenge-2019/",
        total_size_gb=0.5
    ),
    "physionet_wfdb": DatasetMetadata(
        dataset_name="MIT-BIH Arrhythmia Database",
        source="physionet",
        version="1.0.0",
        record_count=109446,
        patient_count=47,
        data_types=["ecg", "annotations", "arrhythmia_labels"],
        file_formats=["dat", "atr", "hea"],
        requires_credentials=False,
        license="Open Data Commons Attribution License",
        citation="Moody, G. & Mark, R. (2001). MIT-BIH Arrhythmia Database.",
        description="Two-lead ECG recordings with annotated arrhythmia events",
        access_url="https://physionet.org/content/mitdb/1.0.0/",
        documentation_url="https://physionet.org/content/mitdb/",
        total_size_gb=0.1
    )
}


class PhysioNetCredentialManager:
    """Manage PhysioNet credentials for data access"""
    
    def __init__(self):
        self.username = os.environ.get("PHYSIONET_USERNAME")
        self.password = os.environ.get("PHYSIONET_PASSWORD")
    
    @property
    def is_configured(self) -> bool:
        """Check if credentials are configured"""
        return bool(self.username and self.password)
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for PhysioNet API"""
        if not self.is_configured:
            return {}
        
        import base64
        credentials = base64.b64encode(
            f"{self.username}:{self.password}".encode()
        ).decode()
        
        return {"Authorization": f"Basic {credentials}"}


class MIMICDataLoader:
    """Load and preprocess MIMIC-III/IV data"""
    
    def __init__(self, data_dir: str = "./data/mimic"):
        self.data_dir = Path(data_dir)
        self.credential_manager = PhysioNetCredentialManager()
    
    def is_available(self) -> bool:
        """Check if MIMIC data is available locally"""
        return (self.data_dir / "patients.csv").exists()
    
    async def load_vitals_chartevents(
        self,
        limit: Optional[int] = None
    ) -> List[DatasetRecord]:
        """Load vital signs from CHARTEVENTS table"""
        
        import pandas as pd
        
        vitals_file = self.data_dir / "CHARTEVENTS.csv.gz"
        if not vitals_file.exists():
            vitals_file = self.data_dir / "chartevents.csv"
        
        if not vitals_file.exists():
            logger.warning(f"MIMIC vitals file not found: {vitals_file}")
            return []
        
        vital_itemids = {
            220045: 'heart_rate',
            220050: 'bp_systolic',
            220051: 'bp_diastolic',
            220052: 'bp_mean',
            220179: 'bp_systolic_nb',
            220180: 'bp_diastolic_nb',
            220210: 'respiratory_rate',
            220277: 'spo2',
            223761: 'temperature',
            223762: 'temperature_c'
        }
        
        try:
            chunks = pd.read_csv(
                vitals_file,
                chunksize=100000,
                usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM']
            )
            
            records = []
            total_loaded = 0
            
            for chunk in chunks:
                vital_chunk = chunk[chunk['ITEMID'].isin(vital_itemids.keys())]
                
                for _, row in vital_chunk.iterrows():
                    if pd.isna(row['VALUENUM']):
                        continue
                    
                    vital_type = vital_itemids.get(row['ITEMID'], 'unknown')
                    
                    record = DatasetRecord(
                        record_id=f"mimic_{row['SUBJECT_ID']}_{row['CHARTTIME']}_{row['ITEMID']}",
                        dataset_source="mimic_iii",
                        features={
                            'vital_type': vital_type,
                            'value': float(row['VALUENUM']),
                            'timestamp': str(row['CHARTTIME'])
                        },
                        metadata={
                            'subject_id': int(row['SUBJECT_ID']),
                            'hadm_id': int(row['HADM_ID']) if not pd.isna(row['HADM_ID']) else None,
                            'icustay_id': int(row['ICUSTAY_ID']) if not pd.isna(row['ICUSTAY_ID']) else None
                        }
                    )
                    records.append(record)
                    total_loaded += 1
                    
                    if limit and total_loaded >= limit:
                        return records
            
            return records
            
        except Exception as e:
            logger.error(f"Error loading MIMIC vitals: {e}")
            return []
    
    async def load_lab_results(
        self,
        limit: Optional[int] = None
    ) -> List[DatasetRecord]:
        """Load lab results from LABEVENTS table"""
        
        import pandas as pd
        
        labs_file = self.data_dir / "LABEVENTS.csv.gz"
        if not labs_file.exists():
            labs_file = self.data_dir / "labevents.csv"
        
        if not labs_file.exists():
            logger.warning(f"MIMIC labs file not found: {labs_file}")
            return []
        
        common_labs = {
            50912: 'creatinine',
            50971: 'potassium',
            50983: 'sodium',
            51006: 'urea_nitrogen',
            51221: 'hematocrit',
            51222: 'hemoglobin',
            51279: 'rbc',
            51301: 'wbc',
            50820: 'ph',
            50821: 'po2',
            50818: 'pco2',
            51265: 'platelets'
        }
        
        try:
            chunks = pd.read_csv(
                labs_file,
                chunksize=100000,
                usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'FLAG']
            )
            
            records = []
            total_loaded = 0
            
            for chunk in chunks:
                lab_chunk = chunk[chunk['ITEMID'].isin(common_labs.keys())]
                
                for _, row in lab_chunk.iterrows():
                    if pd.isna(row['VALUENUM']):
                        continue
                    
                    lab_type = common_labs.get(row['ITEMID'], 'unknown')
                    
                    record = DatasetRecord(
                        record_id=f"mimic_lab_{row['SUBJECT_ID']}_{row['CHARTTIME']}_{row['ITEMID']}",
                        dataset_source="mimic_iii",
                        features={
                            'lab_type': lab_type,
                            'value': float(row['VALUENUM']),
                            'abnormal_flag': str(row['FLAG']) if not pd.isna(row['FLAG']) else None,
                            'timestamp': str(row['CHARTTIME'])
                        },
                        metadata={
                            'subject_id': int(row['SUBJECT_ID']),
                            'hadm_id': int(row['HADM_ID']) if not pd.isna(row['HADM_ID']) else None
                        }
                    )
                    records.append(record)
                    total_loaded += 1
                    
                    if limit and total_loaded >= limit:
                        return records
            
            return records
            
        except Exception as e:
            logger.error(f"Error loading MIMIC labs: {e}")
            return []


class SepsisChallenge2019Loader:
    """Load PhysioNet Challenge 2019 Sepsis data"""
    
    def __init__(self, data_dir: str = "./data/sepsis_challenge"):
        self.data_dir = Path(data_dir)
        
        self.vital_columns = [
            'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp'
        ]
        self.lab_columns = [
            'BUN', 'Calcium', 'Chloride', 'Creatinine', 'Glucose',
            'Magnesium', 'Phosphate', 'Potassium', 'Hct', 'Hgb',
            'WBC', 'Platelets'
        ]
        self.demographic_columns = [
            'Age', 'Gender', 'HospAdmTime', 'ICULOS'
        ]
    
    def is_available(self) -> bool:
        """Check if sepsis challenge data is available"""
        training_a = self.data_dir / "training" / "training_setA"
        training_b = self.data_dir / "training" / "training_setB"
        return training_a.exists() or training_b.exists()
    
    async def load_patient_data(
        self,
        limit: Optional[int] = None
    ) -> List[DatasetRecord]:
        """Load patient time-series data with sepsis labels"""
        
        import pandas as pd
        
        records = []
        total_loaded = 0
        
        training_dirs = [
            self.data_dir / "training" / "training_setA",
            self.data_dir / "training" / "training_setB"
        ]
        
        for training_dir in training_dirs:
            if not training_dir.exists():
                continue
            
            psv_files = list(training_dir.glob("*.psv"))
            
            for psv_file in psv_files:
                try:
                    df = pd.read_csv(psv_file, sep='|')
                    
                    patient_id = psv_file.stem
                    
                    vital_means = {}
                    for col in self.vital_columns:
                        if col in df.columns:
                            values = df[col].dropna()
                            if len(values) > 0:
                                vital_means[col.lower()] = float(values.mean())
                    
                    lab_means = {}
                    for col in self.lab_columns:
                        if col in df.columns:
                            values = df[col].dropna()
                            if len(values) > 0:
                                lab_means[col.lower()] = float(values.mean())
                    
                    sepsis_label = 0
                    if 'SepsisLabel' in df.columns:
                        sepsis_label = 1 if df['SepsisLabel'].max() > 0 else 0
                    
                    age = None
                    if 'Age' in df.columns:
                        age = float(df['Age'].iloc[0]) if not pd.isna(df['Age'].iloc[0]) else None
                    
                    record = DatasetRecord(
                        record_id=f"sepsis2019_{patient_id}",
                        dataset_source="physionet_challenge_2019",
                        features={
                            'vitals': vital_means,
                            'labs': lab_means,
                            'sequence_length': len(df),
                            'age': age
                        },
                        label=sepsis_label,
                        metadata={
                            'patient_id': patient_id,
                            'file_path': str(psv_file)
                        }
                    )
                    records.append(record)
                    total_loaded += 1
                    
                    if limit and total_loaded >= limit:
                        return records
                        
                except Exception as e:
                    logger.error(f"Error loading patient file {psv_file}: {e}")
                    continue
        
        return records


class SyntheticDataGenerator:
    """Generate synthetic health data for training augmentation"""
    
    NORMAL_RANGES = {
        'heart_rate': (60, 100),
        'bp_systolic': (90, 120),
        'bp_diastolic': (60, 80),
        'temperature': (36.1, 37.2),
        'respiratory_rate': (12, 20),
        'spo2': (95, 100),
        'sodium': (136, 145),
        'potassium': (3.5, 5.0),
        'creatinine': (0.7, 1.3),
        'hemoglobin': (12.0, 17.5)
    }
    
    ABNORMAL_RANGES = {
        'high': {
            'heart_rate': (100, 180),
            'bp_systolic': (140, 200),
            'bp_diastolic': (90, 120),
            'temperature': (38.0, 41.0),
            'respiratory_rate': (25, 40),
            'sodium': (150, 165),
            'potassium': (5.5, 7.0),
            'creatinine': (2.0, 8.0)
        },
        'low': {
            'heart_rate': (30, 55),
            'bp_systolic': (60, 85),
            'bp_diastolic': (40, 55),
            'temperature': (33.0, 36.0),
            'spo2': (80, 92),
            'sodium': (120, 134),
            'potassium': (2.5, 3.3),
            'hemoglobin': (6.0, 11.0)
        }
    }
    
    def generate_normal_patient(
        self,
        patient_id: str,
        num_timepoints: int = 24
    ) -> DatasetRecord:
        """Generate synthetic data for a healthy patient"""
        
        vitals_sequence = []
        
        for t in range(num_timepoints):
            timepoint = {}
            for vital, (low, high) in self.NORMAL_RANGES.items():
                base_value = np.random.uniform(low, high)
                noise = np.random.normal(0, (high - low) * 0.05)
                timepoint[vital] = max(low * 0.9, min(high * 1.1, base_value + noise))
            vitals_sequence.append(timepoint)
        
        aggregated = {}
        for vital in self.NORMAL_RANGES.keys():
            values = [tp.get(vital, 0) for tp in vitals_sequence if vital in tp]
            if values:
                aggregated[f'{vital}_mean'] = np.mean(values)
                aggregated[f'{vital}_std'] = np.std(values)
                aggregated[f'{vital}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
        
        return DatasetRecord(
            record_id=f"synthetic_normal_{patient_id}",
            dataset_source="synthetic",
            features=aggregated,
            label=0,
            metadata={
                'synthetic': True,
                'condition': 'normal',
                'num_timepoints': num_timepoints
            }
        )
    
    def generate_deteriorating_patient(
        self,
        patient_id: str,
        num_timepoints: int = 24,
        deterioration_start: int = 12
    ) -> DatasetRecord:
        """Generate synthetic data for a deteriorating patient"""
        
        vitals_sequence = []
        
        for t in range(num_timepoints):
            timepoint = {}
            
            if t < deterioration_start:
                for vital, (low, high) in self.NORMAL_RANGES.items():
                    base_value = np.random.uniform(low, high)
                    noise = np.random.normal(0, (high - low) * 0.05)
                    timepoint[vital] = base_value + noise
            else:
                progress = (t - deterioration_start) / (num_timepoints - deterioration_start)
                
                for vital in self.NORMAL_RANGES.keys():
                    normal_low, normal_high = self.NORMAL_RANGES[vital]
                    normal_val = np.random.uniform(normal_low, normal_high)
                    
                    if vital in self.ABNORMAL_RANGES['high']:
                        abn_low, abn_high = self.ABNORMAL_RANGES['high'][vital]
                        target = np.random.uniform(abn_low, abn_high)
                    elif vital in self.ABNORMAL_RANGES['low']:
                        abn_low, abn_high = self.ABNORMAL_RANGES['low'][vital]
                        target = np.random.uniform(abn_low, abn_high)
                    else:
                        target = normal_val
                    
                    interpolated = normal_val + (target - normal_val) * progress
                    noise = np.random.normal(0, abs(target - normal_val) * 0.1)
                    timepoint[vital] = interpolated + noise
            
            vitals_sequence.append(timepoint)
        
        aggregated = {}
        for vital in self.NORMAL_RANGES.keys():
            values = [tp.get(vital, 0) for tp in vitals_sequence if vital in tp]
            if values:
                aggregated[f'{vital}_mean'] = np.mean(values)
                aggregated[f'{vital}_std'] = np.std(values)
                aggregated[f'{vital}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
        
        return DatasetRecord(
            record_id=f"synthetic_deteriorating_{patient_id}",
            dataset_source="synthetic",
            features=aggregated,
            label=1,
            metadata={
                'synthetic': True,
                'condition': 'deteriorating',
                'num_timepoints': num_timepoints,
                'deterioration_start': deterioration_start
            }
        )
    
    def generate_training_batch(
        self,
        normal_count: int = 500,
        deteriorating_count: int = 500
    ) -> List[DatasetRecord]:
        """Generate a balanced batch of synthetic training data"""
        
        records = []
        
        for i in range(normal_count):
            record = self.generate_normal_patient(f"norm_{i:05d}")
            records.append(record)
        
        for i in range(deteriorating_count):
            det_start = np.random.randint(6, 18)
            record = self.generate_deteriorating_patient(
                f"det_{i:05d}",
                deterioration_start=det_start
            )
            records.append(record)
        
        np.random.shuffle(records)
        
        return records


class PublicDatasetManager:
    """Manage public dataset registration and loading"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.mimic_loader = MIMICDataLoader()
        self.sepsis_loader = SepsisChallenge2019Loader()
        self.synthetic_generator = SyntheticDataGenerator()
    
    async def get_registered_datasets(self) -> List[Dict[str, Any]]:
        """Get list of registered public datasets"""
        
        query = text("""
            SELECT 
                id, dataset_name, source, version,
                requires_credentials, credentials_configured,
                download_status, download_progress,
                record_count, patient_count, total_size_gb,
                last_used_at
            FROM public_dataset_registry
            ORDER BY dataset_name
        """)
        
        result = await self.db.execute(query)
        rows = result.fetchall()
        
        return [
            {
                'id': row.id,
                'name': row.dataset_name,
                'source': row.source,
                'version': row.version,
                'requires_credentials': row.requires_credentials,
                'credentials_configured': row.credentials_configured,
                'download_status': row.download_status,
                'download_progress': row.download_progress,
                'record_count': row.record_count,
                'patient_count': row.patient_count,
                'size_gb': float(row.total_size_gb) if row.total_size_gb else 0,
                'last_used': row.last_used_at.isoformat() if row.last_used_at else None
            }
            for row in rows
        ]
    
    async def register_dataset(
        self,
        dataset_key: str
    ) -> Optional[str]:
        """Register a public dataset in the database"""
        
        if dataset_key not in SUPPORTED_DATASETS:
            logger.error(f"Unknown dataset: {dataset_key}")
            return None
        
        metadata = SUPPORTED_DATASETS[dataset_key]
        
        existing = await self.db.execute(
            text("SELECT id FROM public_dataset_registry WHERE dataset_name = :name"),
            {"name": metadata.dataset_name}
        )
        if existing.fetchone():
            logger.info(f"Dataset {metadata.dataset_name} already registered")
            return None
        
        query = text("""
            INSERT INTO public_dataset_registry (
                id, dataset_name, source, version,
                requires_credentials, credentials_configured,
                access_url, documentation_url, description,
                record_count, patient_count, total_size_gb,
                data_types, file_formats,
                download_status, license, citation_required, citation
            ) VALUES (
                gen_random_uuid()::text, :name, :source, :version,
                :requires_creds, :creds_configured,
                :access_url, :doc_url, :description,
                :record_count, :patient_count, :size_gb,
                :data_types, :file_formats,
                'not_started', :license, true, :citation
            )
            RETURNING id
        """)
        
        creds_configured = False
        if metadata.requires_credentials:
            creds_configured = PhysioNetCredentialManager().is_configured
        
        result = await self.db.execute(query, {
            "name": metadata.dataset_name,
            "source": metadata.source,
            "version": metadata.version,
            "requires_creds": metadata.requires_credentials,
            "creds_configured": creds_configured,
            "access_url": metadata.access_url,
            "doc_url": metadata.documentation_url,
            "description": metadata.description,
            "record_count": metadata.record_count,
            "patient_count": metadata.patient_count,
            "size_gb": metadata.total_size_gb,
            "data_types": json.dumps(metadata.data_types),
            "file_formats": json.dumps(metadata.file_formats),
            "license": metadata.license,
            "citation": metadata.citation
        })
        
        await self.db.commit()
        row = result.fetchone()
        
        return row.id if row else None
    
    async def load_dataset_records(
        self,
        dataset_name: str,
        limit: Optional[int] = None
    ) -> List[DatasetRecord]:
        """Load records from a registered dataset"""
        
        if "mimic" in dataset_name.lower():
            vitals = await self.mimic_loader.load_vitals_chartevents(limit=limit)
            labs = await self.mimic_loader.load_lab_results(limit=limit)
            return vitals + labs
        
        elif "sepsis" in dataset_name.lower() or "challenge" in dataset_name.lower():
            return await self.sepsis_loader.load_patient_data(limit=limit)
        
        elif dataset_name.lower() == "synthetic":
            return self.synthetic_generator.generate_training_batch()
        
        else:
            logger.warning(f"No loader available for dataset: {dataset_name}")
            return []
    
    async def update_dataset_usage(self, dataset_id: str):
        """Update last used timestamp for a dataset"""
        
        await self.db.execute(
            text("UPDATE public_dataset_registry SET last_used_at = NOW() WHERE id = :id"),
            {"id": dataset_id}
        )
        await self.db.commit()
    
    def get_available_datasets(self) -> List[DatasetMetadata]:
        """Get list of all supported datasets"""
        return list(SUPPORTED_DATASETS.values())
    
    def generate_synthetic_data(
        self,
        normal_count: int = 500,
        deteriorating_count: int = 500
    ) -> List[DatasetRecord]:
        """Generate synthetic training data"""
        return self.synthetic_generator.generate_training_batch(
            normal_count=normal_count,
            deteriorating_count=deteriorating_count
        )


def create_dataset_manager(db_session: AsyncSession) -> PublicDatasetManager:
    """Factory function to create a dataset manager"""
    return PublicDatasetManager(db_session)
