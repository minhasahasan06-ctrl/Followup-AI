/**
 * PHI Redaction Service - HIPAA Compliance Layer
 * 
 * Now uses OpenAI GPT-4o (via Python backend) for PHI detection and redaction
 * instead of AWS Comprehend Medical, providing superior accuracy and flexibility.
 * 
 * CRITICAL REQUIREMENTS:
 * 1. OpenAI GPT-4o API must be configured
 * 2. Business Associate Agreement (BAA) with OpenAI must be signed
 * 3. OpenAI Enterprise with Zero Data Retention (ZDR) must be enabled
 * 
 * This service provides defense-in-depth by:
 * - Using GPT-4o to detect all PHI (names, dates, locations, IDs)
 * - Blocking AI calls if HIPAA compliance requirements are not met
 * - Logging all redaction actions for audit trail
 * 
 * For production deployments, set these environment variables:
 * - OPENAI_BAA_SIGNED=true (after signing BAA with OpenAI)
 * - OPENAI_ZDR_ENABLED=true (Zero Data Retention enabled)
 * - OPENAI_ENTERPRISE=true (OpenAI Enterprise plan active)
 */

import axios from "axios";

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || "http://localhost:8000";

interface RedactionResult {
  redactedText: string;
  redactions: Array<{
    type: 'name' | 'phone' | 'email' | 'ssn' | 'date' | 'address' | 'mrn' | 'age' | 'id' | 'other';
    original: string;
    placeholder: string;
  }>;
}

interface PHIEntity {
  text: string;
  category: string;
  start_offset: number;
  end_offset: number;
  confidence: number;
  placeholder: string;
}

interface PHIDetectionResponse {
  original_text: string;
  redacted_text: string;
  phi_detected: boolean;
  phi_entities: PHIEntity[];
  redaction_count: number;
  processing_time_ms: number;
}

/**
 * Redact PHI from text using regex patterns (local fallback)
 * Uses regex patterns to identify and replace common PHI elements
 */
export function redactPHI(text: string, options?: {
  preserveEmailDomains?: boolean;
  preservePartialNames?: boolean;
}): RedactionResult {
  let redactedText = text;
  const redactions: RedactionResult['redactions'] = [];

  // 1. Redact email addresses
  const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;
  redactedText = redactedText.replace(emailRegex, (match) => {
    const placeholder = options?.preserveEmailDomains 
      ? `[EMAIL@${match.split('@')[1]}]`
      : '[EMAIL_REDACTED]';
    redactions.push({ type: 'email', original: match, placeholder });
    return placeholder;
  });

  // 2. Redact phone numbers (various formats)
  const phoneRegex = /(\+?1?\s*)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}/g;
  redactedText = redactedText.replace(phoneRegex, (match) => {
    if (match.length < 10) return match; // Skip short number sequences
    const placeholder = '[PHONE_REDACTED]';
    redactions.push({ type: 'phone', original: match, placeholder });
    return placeholder;
  });

  // 3. Redact SSN patterns
  const ssnRegex = /\b\d{3}-\d{2}-\d{4}\b/g;
  redactedText = redactedText.replace(ssnRegex, (match) => {
    const placeholder = '[SSN_REDACTED]';
    redactions.push({ type: 'ssn', original: match, placeholder });
    return placeholder;
  });

  // 4. Redact Medical Record Numbers (MRN) - common patterns
  const mrnRegex = /\b(MRN|Medical Record|Patient ID|Chart)[:\s#]*[A-Z0-9]{6,12}\b/gi;
  redactedText = redactedText.replace(mrnRegex, (match) => {
    const placeholder = '[MRN_REDACTED]';
    redactions.push({ type: 'mrn', original: match, placeholder });
    return placeholder;
  });

  // 5. Redact dates (MM/DD/YYYY, MM-DD-YYYY, etc.)
  const dateRegex = /\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])[\/\-](19|20)\d{2}\b/g;
  redactedText = redactedText.replace(dateRegex, (match) => {
    const placeholder = '[DATE_REDACTED]';
    redactions.push({ type: 'date', original: match, placeholder });
    return placeholder;
  });

  // 6. Redact street addresses (basic pattern)
  const addressRegex = /\b\d{1,5}\s+[A-Za-z0-9\s,]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b/gi;
  redactedText = redactedText.replace(addressRegex, (match) => {
    const placeholder = '[ADDRESS_REDACTED]';
    redactions.push({ type: 'address', original: match, placeholder });
    return placeholder;
  });

  return {
    redactedText,
    redactions
  };
}

/**
 * Redact patient names from text
 * This requires knowing the patient name beforehand
 */
export function redactPatientName(text: string, patientName?: string): string {
  if (!patientName) return text;

  // Split into first and last name
  const nameParts = patientName.split(' ').filter(part => part.length > 1);
  
  let redacted = text;
  nameParts.forEach(name => {
    // Replace full name and partial matches
    const nameRegex = new RegExp(`\\b${name}\\b`, 'gi');
    redacted = redacted.replace(nameRegex, '[PATIENT_NAME]');
  });

  return redacted;
}

/**
 * Use OpenAI GPT-4o (via Python backend) to detect and redact PHI
 * This replaces AWS Comprehend Medical with superior accuracy
 */
export async function redactPHIWithComprehendMedical(text: string): Promise<{
  redactedText: string;
  phiDetected: boolean;
  phiEntities: Array<{ text: string; type: string; category: string }>;
}> {
  try {
    // Call Python backend for GPT-4o based PHI detection
    const response = await axios.post<PHIDetectionResponse>(
      `${PYTHON_BACKEND_URL}/api/v1/medical-nlp/detect-phi`,
      { text },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000 // 30 second timeout
      }
    );

    const result = response.data;

    return {
      redactedText: result.redacted_text,
      phiDetected: result.phi_detected,
      phiEntities: result.phi_entities.map(entity => ({
        text: entity.text,
        type: entity.category,
        category: entity.category
      }))
    };
  } catch (error) {
    console.error('OpenAI GPT-4o PHI detection failed:', error);
    // Fallback to regex-based redaction
    const fallback = redactPHI(text);
    return {
      redactedText: fallback.redactedText,
      phiDetected: fallback.redactions.length > 0,
      phiEntities: fallback.redactions.map(r => ({
        text: r.original,
        type: r.type,
        category: r.type
      }))
    };
  }
}

/**
 * Sanitize email content for AI processing
 * Uses OpenAI GPT-4o (via Python backend) for PHI detection
 */
export async function sanitizeEmailForAI(email: {
  subject: string;
  content: string;
  patientName?: string;
  senderName?: string;
}): Promise<{
  subject: string;
  content: string;
  wasRedacted: boolean;
  redactionCount: number;
}> {
  try {
    // Call Python backend for email sanitization
    const response = await axios.post(
      `${PYTHON_BACKEND_URL}/api/v1/medical-nlp/sanitize-email`,
      {
        subject: email.subject,
        content: email.content,
        patient_name: email.patientName,
        sender_name: email.senderName
      },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );

    return {
      subject: response.data.subject,
      content: response.data.content,
      wasRedacted: response.data.was_redacted,
      redactionCount: response.data.redaction_count
    };
  } catch (error) {
    console.error('Email sanitization via Python backend failed:', error);
    // Fallback to local regex-based redaction
    const subjectResult = redactPHI(email.subject);
    const contentResult = redactPHI(email.content);

    let sanitizedSubject = subjectResult.redactedText;
    let sanitizedContent = contentResult.redactedText;

    // Also redact known patient/sender names
    if (email.patientName) {
      sanitizedSubject = redactPatientName(sanitizedSubject, email.patientName);
      sanitizedContent = redactPatientName(sanitizedContent, email.patientName);
    }
    if (email.senderName) {
      sanitizedSubject = redactPatientName(sanitizedSubject, email.senderName);
      sanitizedContent = redactPatientName(sanitizedContent, email.senderName);
    }

    const totalRedactions = subjectResult.redactions.length + contentResult.redactions.length;

    return {
      subject: sanitizedSubject,
      content: sanitizedContent,
      wasRedacted: totalRedactions > 0,
      redactionCount: totalRedactions
    };
  }
}

/**
 * Sanitize text for AI processing
 * Uses OpenAI GPT-4o (via Python backend) for PHI detection
 */
export async function sanitizeForAI(text: string, patientName?: string): Promise<{
  sanitizedText: string;
  wasRedacted: boolean;
  redactionCount: number;
  phiCategories: string[];
  processingTimeMs: number;
}> {
  try {
    const response = await axios.post(
      `${PYTHON_BACKEND_URL}/api/v1/medical-nlp/sanitize-for-ai`,
      {
        text,
        patient_name: patientName
      },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );

    return {
      sanitizedText: response.data.sanitized_text,
      wasRedacted: response.data.was_redacted,
      redactionCount: response.data.redaction_count,
      phiCategories: response.data.phi_categories,
      processingTimeMs: response.data.processing_time_ms
    };
  } catch (error) {
    console.error('Sanitization via Python backend failed:', error);
    // Fallback to local regex-based redaction
    const startTime = Date.now();
    const result = redactPHI(text);
    let sanitizedText = result.redactedText;
    
    if (patientName) {
      sanitizedText = redactPatientName(sanitizedText, patientName);
    }

    return {
      sanitizedText,
      wasRedacted: result.redactions.length > 0,
      redactionCount: result.redactions.length,
      phiCategories: [...new Set(result.redactions.map(r => r.type))],
      processingTimeMs: Date.now() - startTime
    };
  }
}

/**
 * Check if OpenAI is configured for HIPAA compliance
 * CRITICAL: Blocks AI features until BAA is signed
 */
export function validateHIPAACompliance(): {
  isConfigured: boolean;
  warnings: string[];
  canUseAI: boolean;
} {
  const warnings: string[] = [];
  let canUseAI = true;

  if (!process.env.OPENAI_API_KEY) {
    warnings.push('OpenAI API key not configured');
    canUseAI = false;
  }

  // CRITICAL: Block AI calls until BAA is signed
  if (!process.env.OPENAI_BAA_SIGNED || process.env.OPENAI_BAA_SIGNED !== 'true') {
    warnings.push('CRITICAL: Business Associate Agreement (BAA) with OpenAI NOT signed. AI features BLOCKED. Set OPENAI_BAA_SIGNED=true after signing BAA.');
    canUseAI = false;
  }

  // IMPORTANT: ZDR should be enabled but doesn't block
  if (!process.env.OPENAI_ZDR_ENABLED || process.env.OPENAI_ZDR_ENABLED !== 'true') {
    warnings.push('IMPORTANT: Zero Data Retention (ZDR) not enabled. Set OPENAI_ZDR_ENABLED=true for HIPAA compliance.');
  }

  // NOTICE: Enterprise recommended but doesn't block
  if (!process.env.OPENAI_ENTERPRISE || process.env.OPENAI_ENTERPRISE !== 'true') {
    warnings.push('NOTICE: OpenAI Enterprise plan recommended for HIPAA compliance. Set OPENAI_ENTERPRISE=true.');
  }

  return {
    isConfigured: warnings.length === 0,
    warnings,
    canUseAI
  };
}

/**
 * Log HIPAA compliance warnings on service startup
 */
export function logHIPAAStatus(): void {
  const { isConfigured, warnings, canUseAI } = validateHIPAACompliance();

  if (!isConfigured) {
    console.warn('⚠️  HIPAA COMPLIANCE WARNINGS:');
    warnings.forEach(warning => console.warn(`   - ${warning}`));
    console.warn('');
    if (!canUseAI) {
      console.warn('   🚫 AI FEATURES BLOCKED until BAA is signed.');
      console.warn('   Emails will use local categorization only.');
    } else {
      console.warn('   OpenAI GPT-4o PHI detection is active (via Python backend).');
      console.warn('   All PHI is automatically redacted before AI processing.');
    }
    console.warn('');
    console.warn('   To enable full AI features:');
    console.warn('   1. Sign BAA with OpenAI Enterprise');
    console.warn('   2. Set OPENAI_BAA_SIGNED=true');
    console.warn('   3. Set OPENAI_ZDR_ENABLED=true');
    console.warn('   Visit: https://openai.com/enterprise');
    console.warn('');
  } else {
    console.log('✅ HIPAA compliance checks passed for OpenAI integration');
    console.log('✅ OpenAI GPT-4o PHI detection active (via Python backend)');
  }
}

/**
 * Extract medical entities from text using GPT-4o
 * Replaces AWS Comprehend Medical entity extraction
 */
export async function extractMedicalEntities(text: string): Promise<{
  entities: Array<{
    text: string;
    category: string;
    confidence: number;
    traits?: Array<{ name: string; score: number }>;
    attributes?: Array<{ text: string; type: string; score: number }>;
  }>;
  icd10Codes: Array<{ code: string; description: string; score: number }>;
  rxnormConcepts: Array<{ code: string; description: string; score: number }>;
  snomedConcepts: Array<{ code: string; description: string; score: number }>;
}> {
  try {
    const response = await axios.post(
      `${PYTHON_BACKEND_URL}/api/v1/medical-nlp/extract-entities`,
      { text, include_phi_check: false },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 60000 // 60 second timeout for entity extraction
      }
    );

    return {
      entities: response.data.entities,
      icd10Codes: response.data.icd10_codes,
      rxnormConcepts: response.data.rxnorm_concepts,
      snomedConcepts: response.data.snomed_concepts
    };
  } catch (error) {
    console.error('Medical entity extraction via Python backend failed:', error);
    return {
      entities: [],
      icd10Codes: [],
      rxnormConcepts: [],
      snomedConcepts: []
    };
  }
}

/**
 * Infer ICD-10-CM codes from clinical text using GPT-4o
 * Replaces AWS Comprehend Medical ICD-10 inference
 */
export async function inferICD10Codes(text: string): Promise<Array<{
  code: string;
  description: string;
  score: number;
  category?: string;
}>> {
  try {
    const response = await axios.post(
      `${PYTHON_BACKEND_URL}/api/v1/medical-nlp/infer-icd10`,
      { text },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );

    return response.data.codes;
  } catch (error) {
    console.error('ICD-10 inference via Python backend failed:', error);
    return [];
  }
}

/**
 * Infer RxNorm concepts from medication text using GPT-4o
 * Replaces AWS Comprehend Medical RxNorm inference
 */
export async function inferRxNormConcepts(text: string): Promise<Array<{
  code: string;
  description: string;
  score: number;
  drugClass?: string;
}>> {
  try {
    const response = await axios.post(
      `${PYTHON_BACKEND_URL}/api/v1/medical-nlp/infer-rxnorm`,
      { text },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );

    return response.data.concepts;
  } catch (error) {
    console.error('RxNorm inference via Python backend failed:', error);
    return [];
  }
}

/**
 * Infer SNOMED-CT concepts from clinical text using GPT-4o
 * Replaces AWS Comprehend Medical SNOMED inference
 */
export async function inferSNOMEDConcepts(text: string): Promise<Array<{
  code: string;
  description: string;
  score: number;
  hierarchy?: string;
}>> {
  try {
    const response = await axios.post(
      `${PYTHON_BACKEND_URL}/api/v1/medical-nlp/infer-snomed`,
      { text },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );

    return response.data.concepts;
  } catch (error) {
    console.error('SNOMED inference via Python backend failed:', error);
    return [];
  }
}

/**
 * Check HIPAA compliance status from Python backend
 */
export async function checkHIPAAStatusFromBackend(): Promise<{
  isConfigured: boolean;
  canUseAI: boolean;
  warnings: string[];
  baaSignned: boolean;
  zdrEnabled: boolean;
  enterprise: boolean;
}> {
  try {
    const response = await axios.get(
      `${PYTHON_BACKEND_URL}/api/v1/medical-nlp/hipaa-status`,
      { timeout: 5000 }
    );

    return {
      isConfigured: response.data.is_configured,
      canUseAI: response.data.can_use_ai,
      warnings: response.data.warnings,
      baaSignned: response.data.baa_signed,
      zdrEnabled: response.data.zdr_enabled,
      enterprise: response.data.enterprise
    };
  } catch (error) {
    console.error('Failed to check HIPAA status from Python backend:', error);
    // Fall back to local check
    const localCheck = validateHIPAACompliance();
    return {
      isConfigured: localCheck.isConfigured,
      canUseAI: localCheck.canUseAI,
      warnings: localCheck.warnings,
      baaSignned: process.env.OPENAI_BAA_SIGNED === 'true',
      zdrEnabled: process.env.OPENAI_ZDR_ENABLED === 'true',
      enterprise: process.env.OPENAI_ENTERPRISE === 'true'
    };
  }
}
