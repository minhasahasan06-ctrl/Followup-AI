"""
Clinical Automation Service for Assistant Lysa

Production-grade clinical support automation with:
- SOAP note generation
- ICD-10 code suggestions
- Differential diagnosis generation
- Prescription assistance
- Daily/weekly report generation
- HIPAA-compliant AI processing
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

import openai

from app.models.automation_models import ClinicalAutomationConfig

logger = logging.getLogger(__name__)

openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ClinicalAutomationService:
    """Handles all clinical automation tasks"""
    
    @staticmethod
    async def generate_summary(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a clinical summary for a patient encounter.
        
        Input data:
        - chief_complaint: Main reason for visit
        - symptoms: List of symptoms
        - vitals: Vital signs
        - history: Relevant medical history
        """
        if not patient_id:
            patient_id = input_data.get("patient_id")
        
        chief_complaint = input_data.get("chief_complaint", "")
        symptoms = input_data.get("symptoms", [])
        vitals = input_data.get("vitals", {})
        history = input_data.get("history", "")
        
        prompt = f"""Generate a concise clinical summary for this patient encounter.

Chief Complaint: {chief_complaint}
Symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}
Vitals: {json.dumps(vitals) if vitals else 'Not recorded'}
Relevant History: {history or 'Not provided'}

Generate a professional clinical summary in 2-3 paragraphs:
1. Present illness description
2. Key findings and observations
3. Initial assessment

Keep the summary factual and objective."""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical documentation assistant. Generate accurate, professional clinical summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            summary = response.choices[0].message.content.strip()
            
            logger.info(f"Generated clinical summary for patient {patient_id}")
            
            return {
                "success": True,
                "patient_id": patient_id,
                "summary": summary,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Clinical summary generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def generate_soap_note(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a SOAP note for a patient encounter.
        
        SOAP = Subjective, Objective, Assessment, Plan
        """
        if not patient_id:
            patient_id = input_data.get("patient_id")
        
        config = db.query(ClinicalAutomationConfig).filter(
            ClinicalAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.auto_soap_notes:
            return {
                "success": False,
                "error": "SOAP note automation not enabled"
            }
        
        chief_complaint = input_data.get("chief_complaint", "")
        patient_statement = input_data.get("patient_statement", "")
        symptoms = input_data.get("symptoms", [])
        duration = input_data.get("duration", "")
        vitals = input_data.get("vitals", {})
        physical_exam = input_data.get("physical_exam", "")
        lab_results = input_data.get("lab_results", "")
        
        prompt = f"""Generate a complete SOAP note for this patient encounter.

SUBJECTIVE DATA:
- Chief Complaint: {chief_complaint}
- Patient Statement: {patient_statement}
- Symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}
- Duration: {duration}

OBJECTIVE DATA:
- Vitals: {json.dumps(vitals) if vitals else 'Not recorded'}
- Physical Exam: {physical_exam or 'Pending'}
- Lab Results: {lab_results or 'Pending'}

Generate a structured SOAP note with:
S (Subjective): Patient's description of symptoms
O (Objective): Measurable/observable findings
A (Assessment): Clinical assessment and likely diagnosis
P (Plan): Treatment plan and follow-up

Format as JSON:
{{"subjective": "...", "objective": "...", "assessment": "...", "plan": "..."}}"""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical documentation specialist. Generate accurate, complete SOAP notes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            soap_note = json.loads(response.choices[0].message.content)
            
            logger.info(f"Generated SOAP note for patient {patient_id}")
            
            return {
                "success": True,
                "patient_id": patient_id,
                "soap_note": soap_note,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"SOAP note generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def suggest_icd10(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest ICD-10 diagnosis codes based on clinical information.
        
        Doctor must review and approve all suggestions.
        """
        config = db.query(ClinicalAutomationConfig).filter(
            ClinicalAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.auto_icd10_suggest:
            return {
                "success": False,
                "error": "ICD-10 suggestion not enabled"
            }
        
        diagnosis = input_data.get("diagnosis", "")
        symptoms = input_data.get("symptoms", [])
        chief_complaint = input_data.get("chief_complaint", "")
        
        if not diagnosis and not symptoms and not chief_complaint:
            return {
                "success": False,
                "error": "diagnosis, symptoms, or chief_complaint required"
            }
        
        prompt = f"""Suggest appropriate ICD-10 diagnosis codes for this clinical scenario.

Diagnosis/Assessment: {diagnosis}
Chief Complaint: {chief_complaint}
Symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}

Provide up to 5 most relevant ICD-10 codes with:
- Code
- Description
- Confidence level (high/medium/low)
- Rationale

Format as JSON array:
[{{"code": "X00.0", "description": "...", "confidence": "high", "rationale": "..."}}]

NOTE: These are suggestions only. Doctor must verify accuracy."""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical coding specialist. Suggest accurate ICD-10 codes based on clinical information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            suggestions = result if isinstance(result, list) else result.get("suggestions", [])
            
            logger.info(f"Generated ICD-10 suggestions for patient {patient_id}")
            
            return {
                "success": True,
                "patient_id": patient_id,
                "suggestions": suggestions,
                "requires_doctor_approval": True,
                "disclaimer": "These are AI-generated suggestions. Doctor must verify accuracy before use.",
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"ICD-10 suggestion error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def generate_differential(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate differential diagnosis suggestions.
        
        Doctor must review and make final determination.
        """
        config = db.query(ClinicalAutomationConfig).filter(
            ClinicalAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.auto_differential_diagnosis:
            return {
                "success": False,
                "error": "Differential diagnosis not enabled"
            }
        
        chief_complaint = input_data.get("chief_complaint", "")
        symptoms = input_data.get("symptoms", [])
        age = input_data.get("age")
        sex = input_data.get("sex")
        vitals = input_data.get("vitals", {})
        history = input_data.get("medical_history", "")
        
        prompt = f"""Generate a differential diagnosis list for this patient presentation.

Patient Demographics:
- Age: {age or 'Not specified'}
- Sex: {sex or 'Not specified'}

Chief Complaint: {chief_complaint}
Symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}
Vitals: {json.dumps(vitals) if vitals else 'Not recorded'}
Medical History: {history or 'Not provided'}

Generate differential diagnoses ranked by likelihood:
1. Most likely diagnosis with key supporting findings
2. Second most likely with findings
3-5. Other considerations

Format as JSON:
{{"differentials": [{{"diagnosis": "...", "likelihood": "high/medium/low", "supporting_findings": [...], "rule_out_tests": [...]}}]}}

NOTE: This is clinical decision support only. Doctor must make final diagnosis."""

        try:
            use_o1_model = os.getenv("USE_O1_FOR_CLINICAL_REASONING", "true").lower() == "true"
            baa_signed = os.getenv("OPENAI_BAA_SIGNED", "").lower() == "true"
            enterprise = os.getenv("OPENAI_ENTERPRISE", "").lower() == "true"
            
            if use_o1_model and not (baa_signed and enterprise):
                logger.warning("o1 model requested but BAA/Enterprise not verified - falling back to gpt-4o")
                use_o1_model = False
            
            if use_o1_model:
                logger.info(f"Using o1 model for differential diagnosis (patient: {patient_id}, doctor: {doctor_id})")
                o1_prompt = f"""You are a clinical decision support system providing differential diagnosis analysis.

{prompt}

IMPORTANT: You MUST respond with ONLY valid JSON in this exact format (no markdown, no explanation, just JSON):
{{"differentials": [{{"diagnosis": "...", "likelihood": "high/medium/low", "supporting_findings": [...], "rule_out_tests": [...]}}]}}

NOTE: This is clinical decision support only. Doctor must make final diagnosis."""
                
                try:
                    response = await openai_client.chat.completions.create(
                        model="o1",
                        messages=[
                            {"role": "user", "content": o1_prompt}
                        ],
                        max_completion_tokens=2000
                    )
                except Exception as o1_error:
                    logger.warning(f"o1 model failed: {o1_error}, falling back to gpt-4o")
                    response = await openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a clinical decision support system. Provide thorough differential diagnosis analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=800,
                        response_format={"type": "json_object"}
                    )
                    use_o1_model = False
            else:
                response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a clinical decision support system. Provide thorough differential diagnosis analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800,
                    response_format={"type": "json_object"}
                )
            
            response_content = response.choices[0].message.content
            if response_content.startswith("```"):
                lines = response_content.split("\n")
                response_content = "\n".join(lines[1:-1])
            result = json.loads(response_content)
            
            logger.info(f"Generated differential diagnosis for patient {patient_id} using {'o1' if use_o1_model else 'gpt-4o'}")
            
            return {
                "success": True,
                "patient_id": patient_id,
                "differentials": result.get("differentials", []),
                "requires_doctor_review": True,
                "disclaimer": "AI-generated clinical decision support. Doctor must verify and make final diagnosis.",
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Differential diagnosis error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def generate_daily_report(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a daily activity report for the doctor.
        
        Includes:
        - Appointments summary
        - Patient communications
        - Pending tasks
        - Automation actions taken
        """
        report_date = input_data.get("date")
        if report_date:
            try:
                report_date = datetime.strptime(report_date, "%Y-%m-%d").date()
            except:
                report_date = datetime.utcnow().date() - timedelta(days=1)
        else:
            report_date = datetime.utcnow().date() - timedelta(days=1)
        
        from app.models.appointment import Appointment
        from app.models.automation_models import AutomationJob, AutomationLog
        
        start_of_day = datetime.combine(report_date, datetime.min.time())
        end_of_day = datetime.combine(report_date, datetime.max.time())
        
        appointments = db.query(Appointment).filter(
            and_(
                Appointment.doctor_id == doctor_id,
                Appointment.start_time >= start_of_day,
                Appointment.start_time <= end_of_day
            )
        ).all()
        
        jobs = db.query(AutomationJob).filter(
            and_(
                AutomationJob.doctor_id == doctor_id,
                AutomationJob.created_at >= start_of_day,
                AutomationJob.created_at <= end_of_day
            )
        ).all()
        
        report = {
            "report_date": report_date.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),
            "appointments": {
                "total": len(appointments),
                "completed": sum(1 for a in appointments if a.status == "completed"),
                "no_shows": sum(1 for a in appointments if a.status == "no_show"),
                "cancelled": sum(1 for a in appointments if a.status == "cancelled")
            },
            "automation": {
                "total_jobs": len(jobs),
                "completed": sum(1 for j in jobs if j.status == "completed"),
                "failed": sum(1 for j in jobs if j.status == "failed"),
                "by_type": {}
            },
            "highlights": [],
            "pending_actions": []
        }
        
        for job in jobs:
            job_type = job.job_type
            if job_type not in report["automation"]["by_type"]:
                report["automation"]["by_type"][job_type] = 0
            report["automation"]["by_type"][job_type] += 1
        
        if report["appointments"]["no_shows"] > 0:
            report["highlights"].append(
                f"{report['appointments']['no_shows']} patient(s) missed their appointment"
            )
        
        if report["automation"]["failed"] > 0:
            report["highlights"].append(
                f"{report['automation']['failed']} automation task(s) failed"
            )
        
        logger.info(f"Generated daily report for doctor {doctor_id}")
        
        return {
            "success": True,
            "report": report
        }
    
    @staticmethod
    async def generate_prescription_draft(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a prescription draft for doctor review.
        
        IMPORTANT: Doctor MUST review and approve before use.
        This is assistance only, not a replacement for clinical judgment.
        """
        config = db.query(ClinicalAutomationConfig).filter(
            ClinicalAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.prescription_assist_enabled:
            return {
                "success": False,
                "error": "Prescription assistance not enabled"
            }
        
        diagnosis = input_data.get("diagnosis", "")
        allergies = input_data.get("allergies", [])
        current_medications = input_data.get("current_medications", [])
        age = input_data.get("age")
        weight = input_data.get("weight")
        
        prompt = f"""Suggest medication options for this clinical scenario.

Diagnosis: {diagnosis}
Patient Allergies: {', '.join(allergies) if allergies else 'NKDA'}
Current Medications: {', '.join(current_medications) if current_medications else 'None'}
Age: {age or 'Not specified'}
Weight: {weight or 'Not specified'}

Suggest 2-3 medication options with:
- Drug name (generic)
- Typical dosage range
- Frequency
- Duration
- Key contraindications
- Drug interactions to check

Format as JSON:
{{"suggestions": [{{"drug_name": "...", "dosage": "...", "frequency": "...", "duration": "...", "contraindications": [...], "interactions": [...]}}]}}

CRITICAL: These are suggestions only. Doctor must verify appropriateness."""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a clinical pharmacology assistant. Suggest appropriate medications with safety information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            logger.info(f"Generated prescription suggestions for patient {patient_id}")
            
            return {
                "success": True,
                "patient_id": patient_id,
                "suggestions": result.get("suggestions", []),
                "requires_doctor_approval": True,
                "disclaimer": "AI-generated medication suggestions. Doctor must verify appropriateness, check for interactions, and approve before prescribing.",
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prescription draft error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
