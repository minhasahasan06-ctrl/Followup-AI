import OpenAI from "openai";
import type { EmailThread, EmailMessage } from "@shared/schema";
import { sanitizeEmailForAI, logHIPAAStatus, validateHIPAACompliance } from "./phiRedactionService";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Log HIPAA compliance status on service initialization
logHIPAAStatus();

interface EmailCategorizationResult {
  category: 'urgent' | 'routine' | 'spam' | 'appointment' | 'prescription' | 'general';
  priority: 'urgent' | 'high' | 'normal' | 'low';
  suggestedResponse?: string;
  requiresImmediateAttention: boolean;
  confidenceScore: number;
  reasoning: string;
}

interface EmailReplyResult {
  suggestedReply: string;
  tone: 'professional' | 'warm' | 'empathetic' | 'concise';
  keyPoints: string[];
  alternativeResponses?: string[];
}

/**
 * AI-Powered Email Categorization
 * Analyzes email content to automatically categorize and prioritize
 */
export async function categorizeEmail(
  subject: string,
  content: string,
  senderContext?: {
    isPatient?: boolean;
    patientName?: string;
    previousEmails?: number;
  }
): Promise<EmailCategorizationResult> {
  if (!process.env.OPENAI_API_KEY) {
    console.warn("OpenAI API key not configured. Using fallback categorization.");
    return fallbackCategorization(subject, content);
  }

  // HIPAA COMPLIANCE: Validate configuration
  const { isConfigured, warnings } = validateHIPAACompliance();
  if (!isConfigured) {
    console.warn("HIPAA compliance warnings detected. Falling back to local categorization.");
    console.warn(warnings.join('\n'));
    return fallbackCategorization(subject, content);
  }

  // HIPAA COMPLIANCE: Redact PHI before sending to OpenAI
  const sanitized = await sanitizeEmailForAI({
    subject,
    content,
    patientName: senderContext?.patientName,
  });

  if (sanitized.wasRedacted) {
    console.log(`[PHI Redaction] Removed ${sanitized.redactionCount} PHI elements before AI processing`);
  }

  try {
    const systemPrompt = `You are an AI assistant for a medical practice that helps categorize and prioritize incoming emails. Your job is to analyze emails and determine:
1. Category (urgent, routine, spam, appointment, prescription, general)
2. Priority (urgent, high, normal, low)
3. Whether it requires immediate attention
4. A brief suggested response if appropriate

Consider medical context and urgency indicators like:
- Symptoms requiring immediate attention
- Appointment requests
- Prescription refills
- Administrative questions
- Spam or irrelevant content

Return a JSON object with: category, priority, requiresImmediateAttention (boolean), confidenceScore (0-100), reasoning (brief explanation), suggestedResponse (optional).`;

    const userPrompt = `Analyze this email (PHI has been redacted for privacy):

Subject: ${sanitized.subject}
Content: ${sanitized.content}

${senderContext?.isPatient ? 'Sender: Patient' : 'Sender: Unknown'}
${senderContext?.previousEmails ? `Previous emails from this sender: ${senderContext.previousEmails}` : ''}

Note: Names, contact info, and dates have been redacted. Categorize based on intent and urgency markers.`;

    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      response_format: { type: "json_object" },
      temperature: 0.3,
    });

    const result = JSON.parse(completion.choices[0].message.content || '{}');

    return {
      category: result.category || 'general',
      priority: result.priority || 'normal',
      suggestedResponse: result.suggestedResponse,
      requiresImmediateAttention: result.requiresImmediateAttention || false,
      confidenceScore: result.confidenceScore || 70,
      reasoning: result.reasoning || 'Automated categorization',
    };
  } catch (error) {
    console.error("Error categorizing email with AI:", error);
    return fallbackCategorization(subject, content);
  }
}

/**
 * AI-Powered Email Reply Generation
 * Generates contextually appropriate email responses
 */
export async function generateEmailReply(
  thread: EmailThread,
  messages: EmailMessage[],
  doctorContext?: {
    doctorName?: string;
    specialty?: string;
    clinicName?: string;
  }
): Promise<EmailReplyResult> {
  if (!process.env.OPENAI_API_KEY) {
    console.warn("OpenAI API key not configured. Using fallback reply.");
    return fallbackReplyGeneration(thread, messages);
  }

  // HIPAA COMPLIANCE: Validate configuration
  const { isConfigured, warnings } = validateHIPAACompliance();
  if (!isConfigured) {
    console.warn("HIPAA compliance warnings detected. Falling back to template reply.");
    console.warn(warnings.join('\n'));
    return fallbackReplyGeneration(thread, messages);
  }

  // HIPAA COMPLIANCE: Redact thread subject
  const sanitizedSubject = await sanitizeEmailForAI({
    subject: thread.subject,
    content: '',
  });

  // HIPAA COMPLIANCE: Redact all messages (parallel processing with Promise.all)
  const sanitizedMessages = await Promise.all(messages.map(async (msg) => {
    const sanitized = await sanitizeEmailForAI({
      subject: '',
      content: msg.content,
      senderName: msg.sender,
    });
    return {
      fromType: msg.fromType,
      content: sanitized.content
    };
  }));

  try {
    const systemPrompt = `You are Assistant Lysa, an AI-powered medical receptionist assistant. Generate professional, empathetic email responses for a medical practice.

IMPORTANT: You are receiving de-identified content where PHI (names, dates, contact info) has been redacted. Your responses should use placeholders like [Doctor Name], [Clinic Name], [Patient], etc.

Your responses should:
- Be warm and professional
- Address patient concerns directly
- Provide clear next steps
- Use empathetic language for health concerns
- Be concise but thorough
- Include appropriate medical disclaimers when needed
- Maintain HIPAA compliance (no PHI in examples)

Tone guidelines:
- Appointment requests: Professional and helpful
- Urgent medical concerns: Empathetic and actionable
- Administrative questions: Clear and concise
- Prescription refills: Professional and process-oriented

Return a JSON object with: suggestedReply (complete email text), tone, keyPoints (array of addressed topics), alternativeResponses (array of 1-2 alternative phrasings).`;

    const conversationHistory = sanitizedMessages
      .map(msg => `${msg.fromType === 'patient' ? '[Patient]' : '[Doctor]'}: ${msg.content}`)
      .join('\n\n');

    const userPrompt = `Generate a reply for this email thread (PHI has been redacted):

Subject: ${sanitizedSubject.subject}
Category: ${thread.category}
Priority: ${thread.priority}

Conversation History (de-identified):
${conversationHistory || '[No previous messages]'}

Generate a professional reply using placeholders:
- Doctor Name: ${doctorContext?.doctorName ? '[Doctor Name]' : 'Dr. [Name]'}
- Specialty: ${doctorContext?.specialty || '[Specialty]'}
- Clinic: ${doctorContext?.clinicName || '[Clinic Name]'}

The doctor will replace placeholders with actual information before sending.`;

    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      response_format: { type: "json_object" },
      temperature: 0.7,
    });

    const result = JSON.parse(completion.choices[0].message.content || '{}');

    return {
      suggestedReply: result.suggestedReply || 'Thank you for your email. We will respond shortly.',
      tone: result.tone || 'professional',
      keyPoints: result.keyPoints || [],
      alternativeResponses: result.alternativeResponses || [],
    };
  } catch (error) {
    console.error("Error generating email reply with AI:", error);
    return fallbackReplyGeneration(thread, messages);
  }
}

/**
 * Batch Email Categorization
 * Processes multiple emails efficiently with async PHI redaction
 */
export async function batchCategorizeEmails(
  emails: Array<{ id: string; subject: string; content: string }>
): Promise<Map<string, EmailCategorizationResult>> {
  const results = new Map<string, EmailCategorizationResult>();

  // Process in batches of 5 to avoid rate limits and AWS Comprehend Medical throttling
  const batchSize = 5;
  for (let i = 0; i < emails.length; i += batchSize) {
    const batch = emails.slice(i, i + batchSize);
    
    // categorizeEmail is already async and awaits sanitization internally
    const promises = batch.map(email => 
      categorizeEmail(email.subject, email.content)
        .then(result => ({ id: email.id, result }))
    );

    const batchResults = await Promise.all(promises);
    batchResults.forEach(({ id, result }) => {
      results.set(id, result);
    });

    // Small delay between batches to avoid throttling
    if (i + batchSize < emails.length) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  return results;
}

/**
 * Fallback categorization when OpenAI is unavailable
 */
function fallbackCategorization(subject: string, content: string): EmailCategorizationResult {
  const subjectLower = subject.toLowerCase();
  const contentLower = content.toLowerCase();
  const combined = subjectLower + ' ' + contentLower;

  // Keyword-based categorization
  let category: EmailCategorizationResult['category'] = 'general';
  let priority: EmailCategorizationResult['priority'] = 'normal';
  let requiresImmediateAttention = false;

  // Urgent keywords
  const urgentKeywords = ['urgent', 'emergency', 'asap', 'severe', 'pain', 'bleeding', 'fever', 'chest pain'];
  if (urgentKeywords.some(keyword => combined.includes(keyword))) {
    category = 'urgent';
    priority = 'urgent';
    requiresImmediateAttention = true;
  }

  // Appointment keywords
  const appointmentKeywords = ['appointment', 'schedule', 'booking', 'reschedule', 'cancel', 'availability'];
  if (appointmentKeywords.some(keyword => combined.includes(keyword))) {
    category = 'appointment';
    priority = priority === 'urgent' ? 'urgent' : 'high';
  }

  // Prescription keywords
  const prescriptionKeywords = ['prescription', 'medication', 'refill', 'pharmacy', 'dosage'];
  if (prescriptionKeywords.some(keyword => combined.includes(keyword))) {
    category = 'prescription';
    priority = priority === 'urgent' ? 'urgent' : 'high';
  }

  // Spam indicators
  const spamKeywords = ['unsubscribe', 'click here', 'limited time', 'winner', 'congratulations', 'viagra'];
  if (spamKeywords.some(keyword => combined.includes(keyword))) {
    category = 'spam';
    priority = 'low';
  }

  return {
    category,
    priority,
    requiresImmediateAttention,
    confidenceScore: 60,
    reasoning: 'Keyword-based fallback categorization',
  };
}

/**
 * Fallback reply generation when OpenAI is unavailable
 */
function fallbackReplyGeneration(thread: EmailThread, messages: EmailMessage[]): EmailReplyResult {
  const templates: Record<string, string> = {
    appointment: `Thank you for your email regarding ${thread.subject}.\n\nWe have received your request and will get back to you shortly with available appointment times.\n\nBest regards,\n[Doctor Name]\n[Clinic Name]`,
    prescription: `Thank you for contacting us about your prescription.\n\nWe have received your request and will process it within 1-2 business days. We will notify you once it's ready for pickup.\n\nBest regards,\n[Doctor Name]\n[Clinic Name]`,
    urgent: `Thank you for reaching out.\n\nWe understand your concern requires immediate attention. Please call our office at [phone number] for urgent matters, or visit the emergency room if this is a medical emergency.\n\nBest regards,\n[Doctor Name]\n[Clinic Name]`,
    general: `Thank you for your email.\n\nWe have received your message and will respond within 24-48 hours.\n\nBest regards,\n[Doctor Name]\n[Clinic Name]`,
  };

  const suggestedReply = templates[thread.category] || templates.general;

  return {
    suggestedReply,
    tone: 'professional',
    keyPoints: ['Acknowledged receipt', 'Provided next steps'],
  };
}

/**
 * Extract action items from email content
 */
export async function extractActionItems(
  content: string
): Promise<Array<{ action: string; priority: string; deadline?: string }>> {
  if (!process.env.OPENAI_API_KEY) {
    return [];
  }

  // HIPAA COMPLIANCE: Validate configuration
  const { isConfigured } = validateHIPAACompliance();
  if (!isConfigured) {
    return [];
  }

  // HIPAA COMPLIANCE: Redact PHI before sending
  const sanitized = await sanitizeEmailForAI({
    subject: '',
    content,
  });

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "Extract action items from de-identified emails. Return a JSON object with actionItems array. Each item has: action (brief description), priority (urgent/high/normal/low), deadline (if mentioned)."
        },
        {
          role: "user",
          content: `Extract action items from this email (PHI redacted):\n\n${sanitized.content}`
        }
      ],
      response_format: { type: "json_object" },
      temperature: 0.3,
    });

    const result = JSON.parse(completion.choices[0].message.content || '{"actionItems":[]}');
    return result.actionItems || [];
  } catch (error) {
    console.error("Error extracting action items:", error);
    return [];
  }
}
