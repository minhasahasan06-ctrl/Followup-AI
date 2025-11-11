/**
 * Symptom Triage Service
 * 
 * AI-powered urgency assessment for appointment scheduling
 * 
 * Features:
 * - OpenAI GPT-4 analysis of symptoms with medical knowledge
 * - Rule-based fallback for when AI unavailable
 * - Immunocompromised-specific risk assessment
 * - Red flag detection (sepsis, neutropenic fever, etc.)
 * - HIPAA-compliant with PHI redaction
 * - Audit logging for all assessments
 */

import OpenAI from "openai";
import { storage } from "./storage";
import { sanitizeEmailForAI } from "./phiRedactionService";
import type { PatientProfile, InsertAppointmentTriageLog } from "@shared/schema";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export interface TriageResult {
  urgencyLevel: 'emergency' | 'urgent' | 'routine' | 'followup';
  urgencyScore: number; // 0-100
  recommendedTimeframe: string;
  redFlags: string[];
  recommendations: string[]; // Actionable next steps for patient/doctor
  confidence: number; // 0-1
  assessmentMethod: 'ai' | 'rule-based' | 'hybrid';
  reasoning?: string;
}

// Emergency keywords requiring immediate medical attention
const EMERGENCY_KEYWORDS = [
  'chest pain', 'difficulty breathing', 'can\'t breathe', 'shortness of breath',
  'severe bleeding', 'unconscious', 'unresponsive', 'seizure', 'convulsion',
  'suicide', 'self-harm', 'overdose', 'poisoning',
  'stroke symptoms', 'slurred speech', 'face drooping',
  'severe head injury', 'broken bone', 'severe burn',
  'anaphylaxis', 'allergic reaction severe', 'throat swelling'
];

// Urgent keywords for immunocompromised patients
const URGENT_KEYWORDS_IMMUNOCOMPROMISED = [
  'fever', 'temperature', 'chills', 'shaking',
  'neutropenic fever', 'infection', 'infected wound',
  'severe pain', 'vomiting', 'diarrhea persistent',
  'cough new', 'cough worsening', 'respiratory symptoms',
  'exposure to sick', 'exposed to illness',
  'rash spreading', 'unusual bleeding', 'bruising easy'
];

// Routine keywords
const ROUTINE_KEYWORDS = [
  'checkup', 'follow-up', 'followup', 'routine visit',
  'refill prescription', 'medication adjustment',
  'mild symptoms', 'questions about', 'advice on'
];

/**
 * Assess symptom urgency using AI or rule-based logic
 */
export async function assessSymptomUrgency(
  symptoms: string,
  patientProfile?: PatientProfile,
  patientId?: string
): Promise<TriageResult> {
  const startTime = Date.now();
  
  try {
    // Check if OpenAI is available and BAA is signed
    if (process.env.OPENAI_API_KEY && process.env.OPENAI_BAA_SIGNED === 'true') {
      // Use AI assessment
      const aiResult = await assessWithOpenAI(symptoms, patientProfile);
      const processingTimeMs = Date.now() - startTime;
      
      // Log the assessment
      if (patientId) {
        await logTriageAssessment(
          patientId,
          symptoms,
          aiResult,
          processingTimeMs,
          'gpt-4o'
        );
      }
      
      return aiResult;
    } else {
      // Fall back to rule-based assessment
      const ruleBasedResult = assessWithRules(symptoms, patientProfile);
      const processingTimeMs = Date.now() - startTime;
      
      // Log the assessment
      if (patientId) {
        await logTriageAssessment(
          patientId,
          symptoms,
          ruleBasedResult,
          processingTimeMs,
          'rule-based'
        );
      }
      
      return ruleBasedResult;
    }
  } catch (error) {
    console.error('[SymptomTriage] Error during assessment:', error);
    
    // Fall back to rule-based on error
    const fallbackResult = assessWithRules(symptoms, patientProfile);
    const processingTimeMs = Date.now() - startTime;
    
    if (patientId) {
      await logTriageAssessment(
        patientId,
        symptoms,
        fallbackResult,
        processingTimeMs,
        'rule-based-fallback'
      );
    }
    
    return fallbackResult;
  }
}

/**
 * AI-powered assessment using OpenAI GPT-4
 */
async function assessWithOpenAI(
  symptoms: string,
  patientProfile?: PatientProfile
): Promise<TriageResult> {
  // Sanitize symptoms to remove any remaining PHI
  const sanitizedSymptoms = await sanitizeEmailForAI(symptoms);
  
  // Build context about patient if available
  const patientContext = patientProfile ? `
Patient context:
- Immunocompromised condition: ${patientProfile.immunocompromisedCondition || 'Not specified'}
- Comorbidities: ${patientProfile.comorbidities?.join(', ') || 'None listed'}
- Allergies: ${patientProfile.allergies?.join(', ') || 'None listed'}
` : 'No patient profile available.';

  const systemPrompt = `You are a medical triage AI assistant helping to assess the urgency of patient symptoms for appointment scheduling.

Your role is to:
1. Analyze reported symptoms
2. Consider immunocompromised patient vulnerabilities
3. Identify red flags requiring immediate attention
4. Recommend appropriate timeframes for care

Urgency Levels:
- EMERGENCY (90-100): Life-threatening, seek immediate care (911/ER)
- URGENT (60-89): Same-day or next-day care needed
- ROUTINE (30-59): Schedule within 1-2 weeks
- FOLLOWUP (0-29): Routine follow-up, flexible scheduling

Special Considerations for Immunocompromised Patients:
- Fever >100.4°F (38°C) = URGENT (neutropenic fever risk)
- New cough + fever = URGENT (infection risk high)
- Exposure to sick person = URGENT
- Any infection signs = URGENT
- Unexplained fatigue/weakness = ROUTINE but prioritized

Respond in JSON format:
{
  "urgencyLevel": "emergency" | "urgent" | "routine" | "followup",
  "urgencyScore": 0-100,
  "recommendedTimeframe": "string describing when to seek care",
  "redFlags": ["list of concerning symptoms detected"],
  "recommendations": ["list of actionable next steps for patient/doctor"],
  "confidence": 0-1,
  "reasoning": "brief explanation of assessment"
}`;

  const userPrompt = `${patientContext}

Reported symptoms:
${sanitizedSymptoms}

Assess urgency and provide triage recommendation.`;

  const completion = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt }
    ],
    response_format: { type: "json_object" },
    temperature: 0.3,
  });

  const responseText = completion.choices[0]?.message?.content;
  if (!responseText) {
    throw new Error('No response from OpenAI');
  }

  const assessment = JSON.parse(responseText);
  
  return {
    urgencyLevel: assessment.urgencyLevel,
    urgencyScore: assessment.urgencyScore,
    recommendedTimeframe: assessment.recommendedTimeframe,
    redFlags: assessment.redFlags || [],
    recommendations: assessment.recommendations || [],
    confidence: assessment.confidence,
    assessmentMethod: 'ai',
    reasoning: assessment.reasoning
  };
}

/**
 * Rule-based assessment (fallback when AI unavailable)
 */
function assessWithRules(
  symptoms: string,
  patientProfile?: PatientProfile
): TriageResult {
  const lowerSymptoms = symptoms.toLowerCase();
  const redFlags: string[] = [];
  
  // Check for emergency keywords
  for (const keyword of EMERGENCY_KEYWORDS) {
    if (lowerSymptoms.includes(keyword.toLowerCase())) {
      redFlags.push(`Emergency symptom detected: ${keyword}`);
    }
  }
  
  if (redFlags.length > 0) {
    return {
      urgencyLevel: 'emergency',
      urgencyScore: 95,
      recommendedTimeframe: 'Seek immediate emergency care (call 911 or go to ER)',
      redFlags,
      recommendations: ['Call 911 immediately', 'Do not wait for appointment', 'Go to nearest emergency room'],
      confidence: 0.9,
      assessmentMethod: 'rule-based'
    };
  }
  
  // Check for urgent keywords (especially for immunocompromised)
  const isImmunocompromised = patientProfile?.immunocompromisedCondition ? true : false;
  
  for (const keyword of URGENT_KEYWORDS_IMMUNOCOMPROMISED) {
    if (lowerSymptoms.includes(keyword.toLowerCase())) {
      redFlags.push(`Urgent symptom for immunocompromised: ${keyword}`);
    }
  }
  
  if (redFlags.length > 0 || isImmunocompromised) {
    // Fever is especially urgent for immunocompromised patients
    const hasFever = /fever|temperature|chills/i.test(symptoms);
    const urgencyScore = hasFever && isImmunocompromised ? 85 : 70;
    
    return {
      urgencyLevel: 'urgent',
      urgencyScore,
      recommendedTimeframe: hasFever && isImmunocompromised 
        ? 'Seek same-day care (neutropenic fever risk)'
        : 'Schedule within 24-48 hours',
      redFlags,
      recommendations: hasFever && isImmunocompromised
        ? ['Contact doctor immediately', 'Monitor temperature closely', 'Prepare to go to hospital if worsens']
        : ['Schedule appointment within 24-48 hours', 'Monitor symptoms', 'Call if symptoms worsen'],
      confidence: 0.75,
      assessmentMethod: 'rule-based'
    };
  }
  
  // Check for routine keywords
  for (const keyword of ROUTINE_KEYWORDS) {
    if (lowerSymptoms.includes(keyword.toLowerCase())) {
      return {
        urgencyLevel: 'routine',
        urgencyScore: 40,
        recommendedTimeframe: 'Schedule within 1-2 weeks',
        redFlags: [],
        recommendations: ['Schedule routine appointment', 'Prepare list of questions for doctor', 'Gather relevant medical records'],
        confidence: 0.7,
        assessmentMethod: 'rule-based'
      };
    }
  }
  
  // Default to routine if no clear indicators
  return {
    urgencyLevel: 'routine',
    urgencyScore: 50,
    recommendedTimeframe: 'Schedule at your convenience within 1-2 weeks',
    redFlags: [],
    recommendations: ['Schedule appointment when convenient', 'Document symptoms', 'Note any patterns or triggers'],
    confidence: 0.6,
    assessmentMethod: 'rule-based'
  };
}

/**
 * Map urgency level to timeframe string
 */
export function getRecommendedTimeframe(urgencyLevel: string): string {
  switch (urgencyLevel) {
    case 'emergency':
      return 'Seek immediate emergency care (call 911 or go to ER)';
    case 'urgent':
      return 'Seek care within 24 hours';
    case 'routine':
      return 'Schedule within 1-2 weeks';
    case 'followup':
      return 'Schedule at your convenience';
    default:
      return 'Schedule as needed';
  }
}

/**
 * Log triage assessment to database for audit trail
 */
async function logTriageAssessment(
  patientId: string,
  symptoms: string,
  result: TriageResult,
  processingTimeMs: number,
  modelVersion: string
): Promise<void> {
  try {
    const logData: InsertAppointmentTriageLog = {
      patientId,
      symptoms,
      urgencyLevel: result.urgencyLevel,
      urgencyScore: result.urgencyScore,
      recommendedTimeframe: result.recommendedTimeframe,
      redFlags: result.redFlags,
      confidence: result.confidence.toString(),
      assessmentMethod: result.assessmentMethod,
      processingTimeMs,
      modelVersion,
    };
    
    await storage.createAppointmentTriageLog(logData);
  } catch (error) {
    console.error('[SymptomTriage] Error logging assessment:', error);
    // Don't throw - logging failure shouldn't break triage
  }
}

/**
 * Create risk alert for urgent/emergency assessments
 */
export async function escalateToRiskAlert(
  patientId: string,
  triageResult: TriageResult,
  symptoms: string
): Promise<string | null> {
  if (triageResult.urgencyLevel !== 'emergency' && triageResult.urgencyLevel !== 'urgent') {
    return null; // Only escalate urgent/emergency
  }
  
  try {
    const severity = triageResult.urgencyLevel === 'emergency' ? 'critical' : 'high';
    const priority = triageResult.urgencyLevel === 'emergency' ? 'critical' : 'high';
    
    const riskAlert = await storage.createRiskAlert({
      userId: patientId,
      type: 'symptom_triage',
      severity,
      priority,
      title: `${triageResult.urgencyLevel.toUpperCase()}: Medical attention needed`,
      message: `Symptoms reported: ${symptoms}\n\nRecommendation: ${triageResult.recommendedTimeframe}`,
      detectedValue: triageResult.urgencyScore.toString(),
      threshold: triageResult.urgencyLevel === 'emergency' ? '90' : '60',
      aiExplanation: triageResult.reasoning || `Automated triage assessment detected ${triageResult.redFlags.length} red flags.`,
      recommendations: triageResult.redFlags.map(flag => ({
        action: flag,
        urgency: triageResult.urgencyLevel === 'emergency' ? 'immediate' as const : 'today' as const,
        category: 'medical' as const
      }))
    });
    
    return riskAlert.id;
  } catch (error) {
    console.error('[SymptomTriage] Error creating risk alert:', error);
    return null;
  }
}
