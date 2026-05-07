"""
RAGChainMed API - Clinical Decision Support System
===================================================
FastAPI-based REST API for the RAGChainMed system.

Features:
- Patient data management
- Clinical predictions with RAG
- Audit logging and access control
- Clinical decision support
- Medical knowledge retrieval

Author: RAGChainMed
Date: May 2026
"""

from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from app.rag.rag_pipeline import load_knowledge, retrieve_context
from app.rag.enhanced_rag_pipeline import MedicalRAGPipeline
from app.rag.llm_explainer import generate_explanation

from app.blockchain.audit_log import BlockchainAuditLog, AuditRecord
from app.blockchain.access_control import AccessControlManager, Permission, User, UserRole

from app.clinical.clinical_data_manager import (
    ClinicalDataManager, PatientRecord, VitalSigns, ClinicalNote, DataClassification
)
from app.clinical.clinical_decision_support import ClinicalDecisionSupportEngine

from app.config import (
    TRAINED_MODEL_FILE,
    SCALER_FILE,
    FEATURE_COLUMNS_FILE,
    CLASS_LABELS
)

# ============================================================
# INITIALIZATION
# ============================================================

app = FastAPI(
    title="RAGChainMed API",
    description="AI-based Clinical Decision Support System",
    version="1.0.0"
)

print("🔄 Initializing RAGChainMed...")

# Load model
print("Loading trained model...")
model = joblib.load(TRAINED_MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
print("✓ Model loaded")

# Initialize RAG pipeline
print("Initializing RAG pipeline...")
medical_rag = MedicalRAGPipeline()
db = load_knowledge()
print("✓ RAG pipeline initialized")

# Initialize blockchain audit log
print("Initializing blockchain audit log...")
audit_log = BlockchainAuditLog()
print("✓ Audit log initialized")

# Initialize access control
print("Initializing access control...")
access_control = AccessControlManager()
print("✓ Access control initialized")

# Initialize clinical data manager
print("Initializing clinical data manager...")
data_manager = ClinicalDataManager()
print("✓ Clinical data manager initialized")

# Initialize clinical decision support
print("Initializing clinical decision support...")
cds_engine = ClinicalDecisionSupportEngine()
print("✓ Clinical decision support initialized")

print("✅ RAGChainMed initialization complete\n")

# ============================================================
# INPUT SCHEMAS
# ============================================================

class PatientDataInput(BaseModel):
    """Patient clinical data for prediction"""
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class ClinicalNoteInput(BaseModel):
    """Clinical note submission"""
    patient_id: str
    note_type: str = Field(..., description="e.g., SOAP, Consultation, Progress")
    content: str
    author: str


class VitalSignsInput(BaseModel):
    """Vital signs recording"""
    patient_id: str
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    temperature: float
    respiratory_rate: float
    oxygen_saturation: float
    notes: Optional[str] = None


class MedicalQueryInput(BaseModel):
    """Medical knowledge query"""
    query: str
    search_depth: int = Field(3, ge=1, le=10, description="Number of context documents to retrieve")


class PatientRegistrationInput(BaseModel):
    """Patient registration"""
    first_name: str
    last_name: str
    date_of_birth: str
    gender: str
    blood_type: Optional[str] = None
    contact_number: Optional[str] = None
    email: Optional[str] = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def verify_user_permission(user_id: str, permission: Permission) -> bool:
    """Verify user has required permission"""
    if not access_control.check_permission(user_id, permission):
        raise HTTPException(
            status_code=403,
            detail=f"Permission denied: {permission.value} required"
        )
    return True

def log_audit_event(user_id: str, action: str, data_type: str, 
                   details: Dict[str, Any], patient_id: Optional[str] = None,
                   status: str = "success", error_msg: Optional[str] = None):
    """Log an event to the blockchain audit log"""
    audit_log.add_record(
        user_id=user_id,
        action=action,
        data_type=data_type,
        details=details,
        patient_id=patient_id,
        status=status,
        error_message=error_msg
    )

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():
    """API health check and system info"""
    return {
        "message": "RAGChainMed API Running 🚀",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Clinical predictions with RAG",
            "Patient data management",
            "Blockchain audit logging",
            "Clinical decision support",
            "Access control"
        ]
    }

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "audit_chain_integrity": audit_log.verify_chain(),
        "audit_log_entries": len(audit_log)
    }

# ============================================================
# CLINICAL PREDICTION ENDPOINT
# ============================================================

@app.post("/api/v1/predict")
def predict_health_condition(
    data: PatientDataInput,
    user_id: str = Header(..., description="User ID for access control"),
    patient_id: Optional[str] = Header(None, description="Associated patient ID")
):
    """
    Predict cardiovascular disease severity using RAG-enhanced model.
    
    Returns prediction with RAG-based explanation and clinical recommendations.
    """
    try:
        # Verify permissions
        verify_user_permission(user_id, Permission.REQUEST_PREDICTION)
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # Ensure all required features exist
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Keep only required features in correct order
        input_df = input_df[feature_columns]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        severity = CLASS_LABELS.get(int(prediction), "Unknown")

        # RAG context retrieval
        query = f"Patient with age {data.age}, cholesterol {data.chol}, chest pain {data.cp}"
        context = retrieve_context(db, query)

        # LLM explanation
        try:
            explanation = generate_explanation(
                prediction,
                severity,
                context,
                data.model_dump()
            )
        except Exception as e:
            explanation = f"Base prediction: {severity}. Medical context retrieval successful."
            print(f"Warning: Could not generate LLM explanation: {e}")

        # Clinical Decision Support Assessment
        risk_assessment = cds_engine.assess_risk(
            patient_id=patient_id or "unknown",
            patient_data=data.model_dump(),
            model_prediction=severity,
            rag_context=context
        )

        # Log audit event
        log_audit_event(
            user_id=user_id,
            action="prediction",
            data_type="model_prediction",
            details={
                "prediction": int(prediction),
                "severity": severity,
                "confidence": float(max(prediction_proba))
            },
            patient_id=patient_id,
            status="success"
        )

        return {
            "prediction": int(prediction),
            "severity": severity,
            "confidence": float(max(prediction_proba)),
            "class_probabilities": {
                label: float(prob) 
                for label, prob in zip(CLASS_LABELS.values(), prediction_proba)
            },
            "explanation": explanation,
            "rag_context_sources": 2,  # Number of retrieved documents
            "risk_assessment": {
                "level": risk_assessment.risk_level.value,
                "score": risk_assessment.risk_score,
                "contributing_factors": risk_assessment.contributing_factors,
                "recommendations": [
                    {
                        "type": rec.type.value,
                        "description": rec.description,
                        "priority": rec.priority
                    }
                    for rec in risk_assessment.recommendations
                ]
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        log_audit_event(
            user_id=user_id,
            action="prediction",
            data_type="model_prediction",
            details={"error": str(e)},
            patient_id=patient_id,
            status="failure",
            error_msg=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# PATIENT DATA MANAGEMENT ENDPOINTS
# ============================================================

@app.post("/api/v1/patients/register")
def register_patient(
    patient_data: PatientRegistrationInput,
    user_id: str = Header(..., description="User ID")
):
    """Register a new patient"""
    try:
        verify_user_permission(user_id, Permission.EDIT_PATIENT_DATA)
        
        # Create patient record
        from uuid import uuid4
        patient_id = str(uuid4())[:8]
        
        patient = PatientRecord(
            patient_id=patient_id,
            first_name=patient_data.first_name,
            last_name=patient_data.last_name,
            date_of_birth=patient_data.date_of_birth,
            gender=patient_data.gender,
            blood_type=patient_data.blood_type,
            contact_number=patient_data.contact_number,
            email=patient_data.email,
            created_by=user_id
        )
        
        # Store patient
        data_manager.add_patient(patient)
        
        # Log event
        log_audit_event(
            user_id=user_id,
            action="patient_registration",
            data_type="patient_demographics",
            details={"patient_id": patient_id},
            patient_id=patient_id,
            status="success"
        )
        
        return {
            "patient_id": patient_id,
            "status": "registered",
            "message": f"Patient {patient.first_name} {patient.last_name} registered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/patients/{patient_id}")
def get_patient(
    patient_id: str,
    user_id: str = Header(..., description="User ID")
):
    """Retrieve patient data"""
    try:
        verify_user_permission(user_id, Permission.VIEW_PATIENT_DATA)
        
        patient = data_manager.get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        log_audit_event(
            user_id=user_id,
            action="patient_view",
            data_type="patient_demographics",
            details={},
            patient_id=patient_id,
            status="success"
        )
        
        return patient.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# VITAL SIGNS ENDPOINTS
# ============================================================

@app.post("/api/v1/patients/{patient_id}/vital-signs")
def record_vital_signs(
    patient_id: str,
    vital_data: VitalSignsInput,
    user_id: str = Header(..., description="User ID")
):
    """Record vital signs for a patient"""
    try:
        verify_user_permission(user_id, Permission.EDIT_PATIENT_DATA)
        
        vital_signs = VitalSigns(
            patient_id=patient_id,
            timestamp=datetime.utcnow().isoformat(),
            heart_rate=vital_data.heart_rate,
            systolic_bp=vital_data.systolic_bp,
            diastolic_bp=vital_data.diastolic_bp,
            temperature=vital_data.temperature,
            respiratory_rate=vital_data.respiratory_rate,
            oxygen_saturation=vital_data.oxygen_saturation,
            recorded_by=user_id,
            notes=vital_data.notes
        )
        
        data_manager.record_vital_signs(vital_signs)
        
        log_audit_event(
            user_id=user_id,
            action="vital_signs_record",
            data_type="vital_signs",
            details=vital_signs.to_dict(),
            patient_id=patient_id,
            status="success"
        )
        
        return {
            "status": "recorded",
            "patient_id": patient_id,
            "timestamp": vital_signs.timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/patients/{patient_id}/vital-signs")
def get_vital_signs(
    patient_id: str,
    user_id: str = Header(..., description="User ID"),
    limit: int = Query(10, ge=1, le=100)
):
    """Get vital signs history for a patient"""
    try:
        verify_user_permission(user_id, Permission.VIEW_PATIENT_DATA)
        
        vital_signs_list = data_manager.get_vital_signs_history(patient_id)
        
        log_audit_event(
            user_id=user_id,
            action="vital_signs_view",
            data_type="vital_signs",
            details={"records_returned": len(vital_signs_list[-limit:])},
            patient_id=patient_id,
            status="success"
        )
        
        return {
            "patient_id": patient_id,
            "vital_signs": [v.to_dict() for v in vital_signs_list[-limit:]],
            "total_records": len(vital_signs_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# CLINICAL NOTES ENDPOINTS
# ============================================================

@app.post("/api/v1/patients/{patient_id}/clinical-notes")
def add_clinical_note(
    patient_id: str,
    note_data: ClinicalNoteInput,
    user_id: str = Header(..., description="User ID")
):
    """Add a clinical note for a patient"""
    try:
        verify_user_permission(user_id, Permission.EDIT_PATIENT_DATA)
        
        from uuid import uuid4
        note_id = str(uuid4())[:12]
        
        clinical_note = ClinicalNote(
            note_id=note_id,
            patient_id=patient_id,
            note_type=note_data.note_type,
            content=note_data.content,
            author=user_id,
            timestamp=datetime.utcnow().isoformat(),
            classification=DataClassification.CONFIDENTIAL
        )
        
        data_manager.add_clinical_note(clinical_note)
        
        log_audit_event(
            user_id=user_id,
            action="clinical_note_added",
            data_type="clinical_notes",
            details={"note_id": note_id, "note_type": note_data.note_type},
            patient_id=patient_id,
            status="success"
        )
        
        return {
            "note_id": note_id,
            "status": "created",
            "timestamp": clinical_note.timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/patients/{patient_id}/clinical-notes")
def get_clinical_notes(
    patient_id: str,
    user_id: str = Header(..., description="User ID"),
    limit: int = Query(20, ge=1, le=100)
):
    """Get clinical notes for a patient"""
    try:
        verify_user_permission(user_id, Permission.VIEW_PATIENT_DATA)
        
        notes_list = data_manager.get_clinical_notes(patient_id)
        
        log_audit_event(
            user_id=user_id,
            action="clinical_notes_view",
            data_type="clinical_notes",
            details={"records_returned": len(notes_list[-limit:])},
            patient_id=patient_id,
            status="success"
        )
        
        return {
            "patient_id": patient_id,
            "clinical_notes": [n.to_dict() for n in notes_list[-limit:]],
            "total_records": len(notes_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# MEDICAL KNOWLEDGE & RAG ENDPOINTS
# ============================================================

@app.post("/api/v1/knowledge/query")
def query_medical_knowledge(
    query_data: MedicalQueryInput,
    user_id: str = Header(..., description="User ID")
):
    """Query medical knowledge base using RAG"""
    try:
        verify_user_permission(user_id, Permission.QUERY_KNOWLEDGE_BASE)
        
        # Retrieve context
        context_results = medical_rag.retrieve_context(query_data.query, k=query_data.search_depth)
        formatted_context = medical_rag.format_context(context_results)
        
        log_audit_event(
            user_id=user_id,
            action="knowledge_query",
            data_type="medical_knowledge",
            details={"query": query_data.query, "results": len(context_results)},
            status="success"
        )
        
        return {
            "query": query_data.query,
            "context": formatted_context,
            "sources": [m['source'] for _, m in context_results],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# AUDIT LOG ENDPOINTS
# ============================================================

@app.get("/api/v1/audit/logs")
def get_audit_logs(
    user_id: str = Header(..., description="User ID"),
    limit: int = Query(100, ge=1, le=1000),
    patient_id: Optional[str] = None,
    action_type: Optional[str] = None
):
    """Retrieve audit logs"""
    try:
        verify_user_permission(user_id, Permission.VIEW_AUDIT_LOGS)
        
        if patient_id:
            records = audit_log.get_records_by_patient(patient_id)
        elif action_type:
            records = audit_log.get_records_by_action(action_type)
        else:
            records = audit_log.get_recent_records(limit)
        
        return {
            "total_records": len(records),
            "chain_integrity_verified": audit_log.verify_chain(),
            "records": records[-limit:],
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/audit/report")
def get_audit_report(
    user_id: str = Header(..., description="User ID"),
    patient_id: Optional[str] = None
):
    """Generate audit report"""
    try:
        verify_user_permission(user_id, Permission.VIEW_AUDIT_LOGS)
        
        report = audit_log.generate_audit_report(patient_id=patient_id)
        
        return report
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# PATIENT SUMMARY ENDPOINT
# ============================================================

@app.get("/api/v1/patients/{patient_id}/summary")
def get_patient_summary(
    patient_id: str,
    user_id: str = Header(..., description="User ID")
):
    """Get comprehensive patient summary"""
    try:
        verify_user_permission(user_id, Permission.VIEW_PATIENT_DATA)
        
        summary = data_manager.generate_patient_summary(patient_id)
        
        log_audit_event(
            user_id=user_id,
            action="patient_summary_view",
            data_type="patient_record",
            details={},
            patient_id=patient_id,
            status="success"
        )
        
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# SYSTEM INFO ENDPOINTS
# ============================================================

@app.get("/api/v1/system/stats")
def get_system_stats(
    user_id: str = Header(..., description="User ID")
):
    """Get system statistics"""
    try:
        verify_user_permission(user_id, Permission.VIEW_AUDIT_LOGS)
        
        return {
            "audit_log_entries": len(audit_log),
            "chain_integrity": audit_log.verify_chain(),
            "clinical_data": data_manager.get_data_statistics(),
            "access_control": access_control.get_user_statistics(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))