"""
Drug Normalization Service - RxNorm Integration
Maps medication names to standardized RxNorm drug concepts (RxCUI)

Uses free RxNorm API from NLM: https://rxnav.nlm.nih.gov/REST/
No API key required.
"""

import logging
import requests
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime

logger = logging.getLogger(__name__)

# RxNorm API Base URL
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


class DrugNormalizationService:
    """Service for normalizing medication names to RxNorm concepts"""

    def __init__(self, db: Session):
        self.db = db

    def normalize_medication(
        self, 
        medication_name: str,
        medication_id: Optional[str] = None
    ) -> Dict:
        """
        Normalize a medication name to RxNorm concept
        
        Returns:
            {
                'rxcui': str,  # RxNorm Concept Unique Identifier
                'name': str,   # Standardized drug name
                'generic_name': str,
                'confidence': float,  # 0.0-1.0
                'match_source': str,  # 'rxnorm_exact', 'rxnorm_approximate'
                'candidates': List[Dict]  # Alternative matches
            }
        """
        try:
            # Step 1: Try exact spelling search
            exact_match = self._search_exact_term(medication_name)
            if exact_match:
                logger.info(f"Found exact RxNorm match for '{medication_name}': RxCUI {exact_match['rxcui']}")
                return {
                    **exact_match,
                    'confidence': 1.0,
                    'match_source': 'rxnorm_exact',
                    'candidates': []
                }

            # Step 2: Try approximate spelling search
            approximate_matches = self._search_approximate_term(medication_name)
            if approximate_matches:
                best_match = approximate_matches[0]
                logger.info(f"Found approximate RxNorm match for '{medication_name}': RxCUI {best_match['rxcui']}")
                return {
                    **best_match,
                    'confidence': best_match.get('score', 0.8),
                    'match_source': 'rxnorm_approximate',
                    'candidates': approximate_matches[1:5]  # Top 5 alternatives
                }

            # No match found
            logger.warning(f"No RxNorm match found for medication: '{medication_name}'")
            return {
                'rxcui': None,
                'name': medication_name,
                'generic_name': None,
                'confidence': 0.0,
                'match_source': 'none',
                'candidates': []
            }

        except Exception as e:
            logger.error(f"Error normalizing medication '{medication_name}': {str(e)}")
            return {
                'rxcui': None,
                'name': medication_name,
                'generic_name': None,
                'confidence': 0.0,
                'match_source': 'error',
                'candidates': [],
                'error': str(e)
            }

    def _search_exact_term(self, term: str) -> Optional[Dict]:
        """Search for exact term match in RxNorm"""
        try:
            url = f"{RXNORM_BASE_URL}/rxcui.json"
            params = {'name': term.strip()}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            rxcui_list = data.get('idGroup', {}).get('rxnormId', [])
            
            if rxcui_list:
                rxcui = rxcui_list[0]  # Take first match
                # Get drug properties
                props = self._get_drug_properties(rxcui)
                return props
            
            return None

        except Exception as e:
            logger.error(f"Error in exact term search: {str(e)}")
            return None

    def _search_approximate_term(self, term: str) -> List[Dict]:
        """Search for approximate matches in RxNorm"""
        try:
            url = f"{RXNORM_BASE_URL}/approximateTerm.json"
            params = {
                'term': term.strip(),
                'maxEntries': 10  # Get top 10 matches
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            candidates = data.get('approximateGroup', {}).get('candidate', [])
            
            if not candidates:
                return []
            
            # Get properties for each candidate
            results = []
            for candidate in candidates[:5]:  # Limit to top 5
                rxcui = candidate.get('rxcui')
                score = candidate.get('score', 50) / 100.0  # Normalize to 0-1
                
                if rxcui:
                    props = self._get_drug_properties(rxcui)
                    if props:
                        props['score'] = score
                        results.append(props)
            
            return results

        except Exception as e:
            logger.error(f"Error in approximate term search: {str(e)}")
            return []

    def _get_drug_properties(self, rxcui: str) -> Optional[Dict]:
        """Get drug properties from RxNorm"""
        try:
            url = f"{RXNORM_BASE_URL}/rxcui/{rxcui}/properties.json"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            props = data.get('properties', {})
            
            return {
                'rxcui': rxcui,
                'name': props.get('name', ''),
                'generic_name': props.get('synonym', ''),
                'tty': props.get('tty', ''),  # Term type (e.g., SBD, GPCK)
            }

        except Exception as e:
            logger.error(f"Error getting drug properties for RxCUI {rxcui}: {str(e)}")
            return None

    def store_medication_drug_match(
        self,
        medication_id: str,
        drug_id: str,
        match_source: str,
        confidence_score: float,
        match_metadata: Optional[Dict] = None,
        matched_by: Optional[str] = None
    ) -> bool:
        """Store a medication-drug match in the database"""
        try:
            query = text("""
                INSERT INTO medication_drug_matches (
                    medication_id, drug_id, match_source, confidence_score,
                    match_metadata, matched_by, is_active
                ) VALUES (
                    :medication_id, :drug_id, :match_source, :confidence_score,
                    :match_metadata, :matched_by, true
                )
            """)
            
            self.db.execute(query, {
                "medication_id": medication_id,
                "drug_id": drug_id,
                "match_source": match_source,
                "confidence_score": str(confidence_score),
                "match_metadata": match_metadata,
                "matched_by": matched_by
            })
            self.db.commit()
            
            logger.info(f"Stored medication-drug match: {medication_id} -> {drug_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing medication-drug match: {str(e)}")
            self.db.rollback()
            return False

    def get_or_create_drug_from_rxcui(self, rxcui: str) -> Optional[str]:
        """
        Get drug_id for an RxCUI, or create a new drug record if it doesn't exist
        
        Returns: drug_id or None
        """
        try:
            # Check if drug already exists
            query = text("SELECT id FROM drugs WHERE rxcui = :rxcui")
            result = self.db.execute(query, {"rxcui": rxcui}).fetchone()
            
            if result:
                return result[0]  # Return existing drug_id
            
            # Create new drug record from RxNorm data
            props = self._get_drug_properties(rxcui)
            if not props:
                return None
            
            # Get drug class information
            drug_class = self._get_drug_class(rxcui)
            
            # Insert new drug
            insert_query = text("""
                INSERT INTO drugs (
                    rxcui, name, generic_name, drug_class,
                    data_source, data_version
                )
                VALUES (
                    :rxcui, :name, :generic_name, :drug_class,
                    'rxnorm', :data_version
                )
                RETURNING id
            """)
            
            result = self.db.execute(insert_query, {
                "rxcui": rxcui,
                "name": props.get('name', ''),
                "generic_name": props.get('generic_name', ''),
                "drug_class": drug_class,
                "data_version": datetime.now().strftime('%Y%m%d')
            }).fetchone()
            
            self.db.commit()
            
            drug_id = result[0] if result else None
            logger.info(f"Created new drug record for RxCUI {rxcui}: drug_id={drug_id}")
            return drug_id

        except Exception as e:
            logger.error(f"Error getting/creating drug for RxCUI {rxcui}: {str(e)}")
            self.db.rollback()
            return None

    def _get_drug_class(self, rxcui: str) -> Optional[str]:
        """Get drug class from RxClass API"""
        try:
            url = f"{RXNORM_BASE_URL}/rxclass/class/byRxcui.json"
            params = {'rxcui': rxcui}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            class_list = data.get('rxclassDrugInfoList', {}).get('rxclassDrugInfo', [])
            
            if class_list:
                # Return first class name (usually ATC or MeSH)
                return class_list[0].get('rxclassMinConceptItem', {}).get('className', '')
            
            return None

        except Exception as e:
            logger.debug(f"Could not get drug class for RxCUI {rxcui}: {str(e)}")
            return None
