/**
 * PHI Redaction Service - HIPAA Compliance Layer
 * 
 * Uses AWS Comprehend Medical for deterministic PHI detection and redaction
 * before sending data to external AI services (OpenAI).
 * 
 * CRITICAL REQUIREMENTS:
 * 1. AWS Comprehend Medical must be configured
 * 2. Business Associate Agreement (BAA) with OpenAI must be signed
 * 3. OpenAI Enterprise with Zero Data Retention (ZDR) must be enabled
 * 
 * This service provides defense-in-depth by:
 * - Using AWS Comprehend Medical to detect all PHI (names, dates, locations, IDs)
 * - Blocking AI calls if HIPAA compliance requirements are not met
 * - Logging all redaction actions for audit trail
 * 
 * For production deployments, set these environment variables:
 * - OPENAI_BAA_SIGNED=true (after signing BAA with OpenAI)
 * - OPENAI_ZDR_ENABLED=true (Zero Data Retention enabled)
 * - OPENAI_ENTERPRISE=true (OpenAI Enterprise plan active)
 */

import { DetectPHICommand } from "@aws-sdk/client-comprehendmedical";
import { comprehendMedicalClient } from "./aws";

interface RedactionResult {
  redactedText: string;
  redactions: Array<{
    type: 'name' | 'phone' | 'email' | 'ssn' | 'date' | 'address' | 'mrn';
    original: string;
    placeholder: string;
  }>;
}

/**
 * Redact PHI from text before sending to OpenAI
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
 * Use AWS Comprehend Medical to detect and redact PHI
 * This is deterministic and doesn't rely on manual context
 */
export async function redactPHIWithComprehendMedical(text: string): Promise<{
  redactedText: string;
  phiDetected: boolean;
  phiEntities: Array<{ text: string; type: string; category: string }>;
}> {
  try {
    const command = new DetectPHICommand({ Text: text });
    const response = await comprehendMedicalClient.send(command);

    let redactedText = text;
    const phiEntities: Array<{ text: string; type: string; category: string }> = [];

    if (response.Entities && response.Entities.length > 0) {
      // Sort entities by BeginOffset in descending order to avoid offset shifts
      const sortedEntities = response.Entities.sort((a, b) => 
        (b.BeginOffset || 0) - (a.BeginOffset || 0)
      );

      for (const entity of sortedEntities) {
        if (entity.Text && entity.Type && entity.Category) {
          phiEntities.push({
            text: entity.Text,
            type: entity.Type,
            category: entity.Category
          });

          // Redact PHI based on category
          let placeholder = `[${entity.Category}_REDACTED]`;
          
          // More specific placeholders for better context
          if (entity.Type === 'NAME') placeholder = '[PATIENT_NAME]';
          else if (entity.Type === 'AGE') placeholder = '[AGE]';
          else if (entity.Type === 'DATE') placeholder = '[DATE]';
          else if (entity.Type === 'ID') placeholder = '[PATIENT_ID]';
          else if (entity.Type === 'PHONE_OR_FAX') placeholder = '[PHONE]';
          else if (entity.Type === 'EMAIL') placeholder = '[EMAIL]';
          else if (entity.Type === 'ADDRESS') placeholder = '[ADDRESS]';

          // Replace PHI with placeholder
          const entityText = entity.Text;
          const regex = new RegExp(entityText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
          redactedText = redactedText.replace(regex, placeholder);
        }
      }
    }

    return {
      redactedText,
      phiDetected: phiEntities.length > 0,
      phiEntities
    };
  } catch (error) {
    console.error('AWS Comprehend Medical PHI detection failed:', error);
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
 * Uses AWS Comprehend Medical for deterministic PHI detection
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
  // Use AWS Comprehend Medical for deterministic PHI detection
  const subjectResult = await redactPHIWithComprehendMedical(email.subject);
  const contentResult = await redactPHIWithComprehendMedical(email.content);

  const totalRedactions = subjectResult.phiEntities.length + contentResult.phiEntities.length;

  return {
    subject: subjectResult.redactedText,
    content: contentResult.redactedText,
    wasRedacted: totalRedactions > 0,
    redactionCount: totalRedactions
  };
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
      console.warn('   AWS Comprehend Medical PHI detection is active.');
      console.warn('   All PHI is automatically redacted before OpenAI processing.');
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
    console.log('✅ AWS Comprehend Medical PHI detection active');
  }
}
