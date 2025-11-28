/**
 * Chat Receptionist Service
 * Enables Assistant Lysa to book appointments through natural conversation
 * 
 * Features:
 * - Intent detection for appointment booking requests
 * - Natural language parsing for date, time, patient, and reason
 * - Real-time availability checking
 * - Appointment creation and confirmation
 * - Conflict resolution and rescheduling
 * - Patient record search and lookup
 */

import OpenAI from "openai";
import { storage } from "./storage";
import type { Appointment, User, PatientProfile } from "@shared/schema";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export interface AppointmentIntent {
  isAppointmentRequest: boolean;
  intentType: 'book' | 'cancel' | 'reschedule' | 'check_availability' | 'none';
  confidence: number;
  extractedInfo: {
    patientId?: string;
    patientName?: string;
    date?: string;
    time?: string;
    reason?: string;
    appointmentId?: string;
  };
  missingFields: string[];
  suggestions: string[];
}

export interface AvailabilitySlot {
  date: string;
  time: string;
  available: boolean;
  conflictingAppointment?: Appointment;
}

/**
 * Detect if the user message contains an appointment booking intent
 * HIPAA COMPLIANT: Uses pattern matching instead of sending PHI to OpenAI
 */
export async function detectAppointmentIntent(
  message: string,
  doctorId: string,
  conversationHistory?: Array<{ role: string; content: string }>
): Promise<AppointmentIntent> {
  // HIPAA COMPLIANCE: Use local pattern matching instead of sending to OpenAI
  // This avoids transmitting patient names and details to external AI
  return detectAppointmentIntentLocal(message, conversationHistory);
}

/**
 * Local pattern-based intent detection (no external AI calls)
 * HIPAA-safe alternative to GPT-4 intent detection
 */
function detectAppointmentIntentLocal(
  message: string,
  conversationHistory?: Array<{ role: string; content: string }>
): AppointmentIntent {
  const messageLower = message.toLowerCase();
  
  // Detect intent type
  let intentType: AppointmentIntent['intentType'] = 'none';
  let confidence = 0;
  
  // Book intent patterns
  const bookPatterns = ['book', 'schedule', 'set up', 'create appointment', 'make appointment'];
  if (bookPatterns.some(p => messageLower.includes(p))) {
    intentType = 'book';
    confidence = 80;
  }
  
  // Cancel intent patterns
  const cancelPatterns = ['cancel', 'delete', 'remove appointment'];
  if (cancelPatterns.some(p => messageLower.includes(p))) {
    intentType = 'cancel';
    confidence = 80;
  }
  
  // Availability check patterns
  const availabilityPatterns = ['availability', 'available', 'free slots', 'open times', 'what times'];
  if (availabilityPatterns.some(p => messageLower.includes(p))) {
    intentType = 'check_availability';
    confidence = 85;
  }
  
  // Extract information using regex patterns
  const extractedInfo: AppointmentIntent['extractedInfo'] = {};
  
  // Extract date patterns
  const datePatterns = {
    'today': /\btoday\b/i,
    'tomorrow': /\btomorrow\b/i,
    'monday': /\b(?:next\s+)?monday\b/i,
    'tuesday': /\b(?:next\s+)?tuesday\b/i,
    'wednesday': /\b(?:next\s+)?wednesday\b/i,
    'thursday': /\b(?:next\s+)?thursday\b/i,
    'friday': /\b(?:next\s+)?friday\b/i,
  };
  
  for (const [dateExpr, pattern] of Object.entries(datePatterns)) {
    if (pattern.test(message)) {
      extractedInfo.date = dateExpr;
      break;
    }
  }
  
  // Extract time patterns
  const timeMatch = message.match(/\b(\d{1,2})\s*(am|pm|:00)\b/i);
  if (timeMatch) {
    extractedInfo.time = timeMatch[0];
  } else if (/\bmorning\b/i.test(message)) {
    extractedInfo.time = 'morning';
  } else if (/\bafternoon\b/i.test(message)) {
    extractedInfo.time = 'afternoon';
  }
  
  // Extract reason patterns
  const reasonMatch = message.match(/\bfor\s+([a-z\s]{3,30})/i);
  if (reasonMatch) {
    extractedInfo.reason = reasonMatch[1].trim();
  }
  
  // Extract patient name using contextual patterns
  // IMPORTANT: This is extracted locally, not sent to OpenAI
  if (intentType === 'book') {
    // Comprehensive list of words to exclude from patient names
    const excludeWords = [
      // Days
      'tomorrow', 'today', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
      // Times
      'morning', 'afternoon', 'evening', 'night',
      // Common verbs and prepositions
      'book', 'schedule', 'set', 'create', 'for', 'the', 'a', 'an', 'at', 'on', 'in',
      'need', 'want', 'please', 'can', 'could', 'should', 'would', 'like', 'have', 'get',
      // Medical terms
      'checkup', 'appointment', 'consultation', 'exam', 'visit', 'followup', 'follow-up'
    ];
    
    // Try contextual patterns only - NO FALLBACK
    // IMPORTANT: Verbs are case-insensitive, but name capture requires capitalization
    // If none of these match, patient name will be undefined (ask doctor for clarification)
    const patterns = [
      // Pattern 1: "book/schedule <Name> for" - name comes between verb and "for"
      /\b(?:[Bb]ook|[Ss]chedule|[Ss]et\s+[Uu]p|[Cc]reate\s+[Aa]ppointment\s+for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+for\b/,
      
      // Pattern 2: "patient <Name>"
      /\b(?:[Pp]atient)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b/,
      
      // Pattern 3: "for <Name>" but ONLY if Name is capitalized (not "for tomorrow")
      /\b[Ff]or\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b/,
      
      // Pattern 4: "<Name> tomorrow/today/day" - name before time reference
      /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:[Tt]omorrow|[Tt]oday|[Mm]onday|[Tt]uesday|[Ww]ednesday|[Tt]hursday|[Ff]riday|[Ss]aturday|[Ss]unday|at|on)\b/,
      
      // NO FALLBACK PATTERN - If none of the above match, patient name will remain undefined
      // This prevents false positives like "Need" or "Want" from being extracted
    ];
    
    for (const pattern of patterns) {
      const match = message.match(pattern);
      if (match && match[1]) {
        const candidateName = match[1].trim();
        const lowerName = candidateName.toLowerCase();
        
        // Validate: not an excluded word, and has reasonable name characteristics
        if (!excludeWords.includes(lowerName) && 
            candidateName.length >= 2 && 
            candidateName.length <= 50 &&
            /^[A-Z]/.test(candidateName)) { // Starts with capital letter
          extractedInfo.patientName = candidateName;
          break; // Use first valid match
        }
      }
    }
  }
  
  // Determine missing fields
  const missingFields: string[] = [];
  const suggestions: string[] = [];
  
  if (intentType === 'book') {
    if (!extractedInfo.patientName) {
      missingFields.push('patient name');
      suggestions.push('Which patient should I schedule?');
    }
    if (!extractedInfo.date) {
      missingFields.push('date');
      suggestions.push('What date works best?');
    }
    if (!extractedInfo.time) {
      missingFields.push('time');
      suggestions.push('What time would you prefer?');
    }
  }
  
  return {
    isAppointmentRequest: intentType !== 'none',
    intentType,
    confidence,
    extractedInfo,
    missingFields,
    suggestions
  };
}

/**
 * LEGACY: OpenAI-based intent detection (requires HIPAA compliance)
 * Only use when BAA/ZDR are properly configured
 */
async function detectAppointmentIntentWithOpenAI(
  message: string,
  doctorId: string,
  conversationHistory?: Array<{ role: string; content: string }>
): Promise<AppointmentIntent> {
  try {
    const systemPrompt = `You are an AI assistant that analyzes doctor requests to detect appointment booking intents. Analyze the message and conversation history to determine if the doctor wants to:

1. BOOK a new appointment
2. CANCEL an existing appointment
3. RESCHEDULE an appointment
4. CHECK availability
5. NONE - not appointment-related

Extract ALL available information from the message:
- Patient name or ID
- Preferred date (today, tomorrow, next Monday, Dec 5, etc.)
- Preferred time (2pm, 14:00, morning, afternoon, etc.)
- Reason for visit
- Appointment ID (if rescheduling/canceling)

Common patterns:
- "Book John for tomorrow at 2pm"
- "Schedule patient Sarah Miller next Tuesday afternoon"
- "I need an appointment slot for Maria on Friday"
- "Cancel the 3pm appointment"
- "What's my availability on Monday?"

Return JSON with:
{
  "isAppointmentRequest": boolean,
  "intentType": "book" | "cancel" | "reschedule" | "check_availability" | "none",
  "confidence": 0-100,
  "extractedInfo": {
    "patientId": string or null,
    "patientName": string or null,
    "date": "YYYY-MM-DD" or relative like "tomorrow" or null,
    "time": "HH:MM" or "morning"/"afternoon" or null,
    "reason": string or null,
    "appointmentId": string or null
  },
  "missingFields": array of field names that are required but missing,
  "suggestions": array of helpful follow-up questions to ask
}`;

    const userPrompt = `Message: "${message}"

${conversationHistory && conversationHistory.length > 0 ? 
  `Recent conversation:\n${conversationHistory.map(m => `${m.role}: ${m.content}`).join('\n')}\n` : 
  ''}

Analyze this message and detect appointment intent.`;

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
    return result as AppointmentIntent;
  } catch (error) {
    console.error('Error detecting appointment intent:', error);
    return {
      isAppointmentRequest: false,
      intentType: 'none',
      confidence: 0,
      extractedInfo: {},
      missingFields: [],
      suggestions: []
    };
  }
}

/**
 * Parse relative date expressions to absolute dates
 */
export function parseRelativeDate(dateExpression: string): Date | null {
  // Get current date at midnight local time to avoid timezone issues
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  
  const expr = dateExpression.toLowerCase().trim();

  // Handle specific dates (YYYY-MM-DD)
  if (/^\d{4}-\d{2}-\d{2}$/.test(expr)) {
    const date = new Date(expr + 'T00:00:00'); // Midnight local time
    return date;
  }

  // Handle common relative expressions
  if (expr === 'today') {
    return today;
  }

  if (expr === 'tomorrow') {
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);
    return tomorrow;
  }

  if (expr.includes('next monday') || expr.includes('monday')) {
    const daysUntilMonday = (8 - today.getDay()) % 7 || 7;
    const nextMonday = new Date(today);
    nextMonday.setDate(today.getDate() + daysUntilMonday);
    return nextMonday;
  }

  if (expr.includes('next tuesday') || expr.includes('tuesday')) {
    const daysUntilTuesday = (9 - today.getDay()) % 7 || 7;
    const nextTuesday = new Date(today);
    nextTuesday.setDate(today.getDate() + daysUntilTuesday);
    return nextTuesday;
  }

  // Add more day-of-week parsing...
  const dayMap: Record<string, number> = {
    'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4,
    'friday': 5, 'saturday': 6, 'sunday': 0
  };

  for (const [dayName, dayNum] of Object.entries(dayMap)) {
    if (expr.includes(dayName)) {
      const daysUntilDay = (dayNum - today.getDay() + 7) % 7 || 7;
      const targetDate = new Date(today);
      targetDate.setDate(today.getDate() + daysUntilDay);
      return targetDate;
    }
  }

  return null;
}

/**
 * Parse time expressions to 24-hour format
 */
export function parseTime(timeExpression: string): string | null {
  const expr = timeExpression.toLowerCase().trim();

  // Handle HH:MM format
  if (/^\d{1,2}:\d{2}$/.test(expr)) {
    return expr.padStart(5, '0');
  }

  // Handle "2pm" style
  const match = expr.match(/(\d{1,2})\s*(am|pm)/);
  if (match) {
    let hour = parseInt(match[1]);
    const period = match[2];
    if (period === 'pm' && hour < 12) hour += 12;
    if (period === 'am' && hour === 12) hour = 0;
    return `${hour.toString().padStart(2, '0')}:00`;
  }

  // Handle "14:00" style
  if (/^\d{1,2}$/.test(expr)) {
    const hour = parseInt(expr);
    if (hour >= 0 && hour < 24) {
      return `${hour.toString().padStart(2, '0')}:00`;
    }
  }

  // Return null for relative times like "morning", "afternoon"
  // These need to be resolved with availability checking
  if (['morning', 'afternoon', 'evening'].includes(expr)) {
    return expr;
  }

  return null;
}

/**
 * Check doctor's availability for a specific date/time range
 */
export async function checkAvailability(
  doctorId: string,
  date: Date,
  timePreference?: string
): Promise<AvailabilitySlot[]> {
  try {
    const dateStr = date.toISOString().split('T')[0];

    // Get doctor's availability for this date
    const availability = await storage.getDoctorAvailability(doctorId, dateStr);
    if (!availability || !availability.isAvailable) {
      return [];
    }

    // Get existing appointments for this date
    const appointments = await storage.getDoctorAppointments(doctorId, dateStr);

    // Parse availability slots
    const slots: AvailabilitySlot[] = [];
    const startTime = availability.startTime;
    const endTime = availability.endTime;

    // Generate 30-minute slots
    let currentTime = startTime;
    while (currentTime < endTime) {
      const conflictingAppointment = appointments.find(apt => 
        apt.startTime === currentTime && apt.status !== 'cancelled'
      );

      // Apply time preference filter
      let matchesPreference = true;
      if (timePreference) {
        const hour = parseInt(currentTime.split(':')[0]);
        if (timePreference === 'morning' && hour >= 12) matchesPreference = false;
        if (timePreference === 'afternoon' && (hour < 12 || hour >= 17)) matchesPreference = false;
        if (timePreference === 'evening' && hour < 17) matchesPreference = false;
        if (timePreference !== 'morning' && timePreference !== 'afternoon' && timePreference !== 'evening' && currentTime !== timePreference) {
          matchesPreference = false;
        }
      }

      if (matchesPreference) {
        slots.push({
          date: dateStr,
          time: currentTime,
          available: !conflictingAppointment,
          conflictingAppointment
        });
      }

      // Increment by 30 minutes
      const [hours, minutes] = currentTime.split(':').map(Number);
      const nextMinutes = (minutes + 30) % 60;
      const nextHours = minutes + 30 >= 60 ? hours + 1 : hours;
      currentTime = `${nextHours.toString().padStart(2, '0')}:${nextMinutes.toString().padStart(2, '0')}`;
    }

    return slots;
  } catch (error) {
    console.error('Error checking availability:', error);
    return [];
  }
}

/**
 * Patient Search Intent Detection
 */
export interface PatientSearchIntent {
  isPatientSearch: boolean;
  intentType: 'search_by_name' | 'search_by_id' | 'get_record' | 'list_patients' | 'none';
  confidence: number;
  searchQuery: string | null;
  patientId: string | null;
}

export interface PatientSearchResult {
  user: User;
  profile: PatientProfile | null;
  recentAppointments?: Appointment[];
  matchScore: number;
}

/**
 * Detect patient search intent from message
 * HIPAA COMPLIANT: Uses local pattern matching
 */
export function detectPatientSearchIntent(message: string): PatientSearchIntent {
  const messageLower = message.toLowerCase();
  
  let intentType: PatientSearchIntent['intentType'] = 'none';
  let confidence = 0;
  let searchQuery: string | null = null;
  let patientId: string | null = null;
  
  // List patients patterns
  const listPatterns = ['list patients', 'show patients', 'all patients', 'my patients', 'patient list'];
  if (listPatterns.some(p => messageLower.includes(p))) {
    intentType = 'list_patients';
    confidence = 90;
  }
  
  // Search by name patterns
  const searchPatterns = ['find patient', 'search patient', 'look up patient', 'lookup patient', 'patient named', 'find record'];
  if (searchPatterns.some(p => messageLower.includes(p))) {
    intentType = 'search_by_name';
    confidence = 85;
    
    // Extract search query
    const namePatterns = [
      /(?:find|search|look\s*up|lookup)\s+(?:patient\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i,
      /patient\s+named\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i,
      /record\s+for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i,
    ];
    
    for (const pattern of namePatterns) {
      const match = message.match(pattern);
      if (match && match[1]) {
        searchQuery = match[1].trim();
        break;
      }
    }
  }
  
  // Search by ID patterns
  const idPatterns = ['patient id', 'followup id', 'patient #', 'record id'];
  if (idPatterns.some(p => messageLower.includes(p))) {
    intentType = 'search_by_id';
    confidence = 85;
    
    // Extract patient ID
    const idMatch = message.match(/(?:patient\s+id|followup\s+id|patient\s*#|record\s+id)[:\s]*([A-Z0-9-]+)/i);
    if (idMatch) {
      patientId = idMatch[1].trim();
    }
  }
  
  // Get record patterns
  const recordPatterns = ['get record', 'show record', 'view record', 'patient record', 'medical record'];
  if (recordPatterns.some(p => messageLower.includes(p))) {
    intentType = 'get_record';
    confidence = 80;
  }
  
  return {
    isPatientSearch: intentType !== 'none',
    intentType,
    confidence,
    searchQuery,
    patientId
  };
}

/**
 * Search patients by name or identifier
 * Returns results sorted by match score
 */
export async function searchPatients(
  query: string,
  doctorId?: string,
  limit: number = 10
): Promise<PatientSearchResult[]> {
  try {
    const normalizedQuery = query.toLowerCase().trim();
    
    // Get all patients (optionally filtered by doctor)
    let patients: Array<User & { profile?: PatientProfile }>;
    
    if (doctorId) {
      patients = await storage.getDoctorPatients(doctorId);
    } else {
      patients = await storage.getAllPatients();
    }
    
    // Score and filter patients by query match
    const results: PatientSearchResult[] = [];
    
    for (const patient of patients) {
      let matchScore = 0;
      
      // Check name match
      const fullName = `${patient.firstName} ${patient.lastName}`.toLowerCase();
      if (fullName.includes(normalizedQuery)) {
        matchScore = 100;
      } else if (patient.firstName?.toLowerCase().includes(normalizedQuery)) {
        matchScore = 80;
      } else if (patient.lastName?.toLowerCase().includes(normalizedQuery)) {
        matchScore = 80;
      }
      
      // Check email match
      if (patient.email?.toLowerCase().includes(normalizedQuery)) {
        matchScore = Math.max(matchScore, 70);
      }
      
      // Check followup ID match
      if (patient.profile?.followupPatientId?.toLowerCase().includes(normalizedQuery)) {
        matchScore = Math.max(matchScore, 90);
      }
      
      // Check phone match
      if (patient.phoneNumber?.includes(normalizedQuery)) {
        matchScore = Math.max(matchScore, 75);
      }
      
      if (matchScore > 0) {
        results.push({
          user: patient,
          profile: patient.profile || null,
          matchScore
        });
      }
    }
    
    // Sort by match score and return top results
    return results
      .sort((a, b) => b.matchScore - a.matchScore)
      .slice(0, limit);
  } catch (error) {
    console.error('Error searching patients:', error);
    return [];
  }
}

/**
 * Get comprehensive patient record
 */
export async function getPatientRecord(
  patientId: string,
  doctorId?: string
): Promise<{
  patient: User;
  profile: PatientProfile | null;
  recentAppointments: Appointment[];
  upcomingAppointments: Appointment[];
} | null> {
  try {
    const patient = await storage.getUser(patientId);
    if (!patient || patient.role !== 'patient') {
      return null;
    }
    
    const profile = await storage.getPatientProfile(patientId);
    
    // Get patient's appointments
    const allAppointments = await storage.getPatientAppointments(patientId);
    const today = new Date().toISOString().split('T')[0];
    
    const recentAppointments = allAppointments
      .filter(apt => apt.date < today)
      .sort((a, b) => b.date.localeCompare(a.date))
      .slice(0, 5);
    
    const upcomingAppointments = allAppointments
      .filter(apt => apt.date >= today && apt.status !== 'cancelled')
      .sort((a, b) => a.date.localeCompare(b.date));
    
    return {
      patient,
      profile,
      recentAppointments,
      upcomingAppointments
    };
  } catch (error) {
    console.error('Error getting patient record:', error);
    return null;
  }
}

/**
 * Format patient information for chat response
 * HIPAA COMPLIANT: Only includes necessary information
 */
export function formatPatientSummary(
  patient: User,
  profile: PatientProfile | null,
  includeAppointments: boolean = false,
  appointments?: Appointment[]
): string {
  let summary = `**${patient.firstName} ${patient.lastName}**\n`;
  
  if (profile?.followupPatientId) {
    summary += `Patient ID: ${profile.followupPatientId}\n`;
  }
  
  if (patient.email) {
    summary += `Email: ${patient.email}\n`;
  }
  
  if (patient.phoneNumber) {
    summary += `Phone: ${patient.phoneNumber}\n`;
  }
  
  if (profile) {
    if (profile.dateOfBirth) {
      const age = calculateAge(profile.dateOfBirth);
      summary += `Age: ${age} years\n`;
    }
    
    if (profile.bloodType) {
      summary += `Blood Type: ${profile.bloodType}\n`;
    }
    
    if (profile.allergies && profile.allergies.length > 0) {
      summary += `Allergies: ${profile.allergies.join(', ')}\n`;
    }
    
    if (profile.medicalConditions && profile.medicalConditions.length > 0) {
      summary += `Conditions: ${profile.medicalConditions.join(', ')}\n`;
    }
  }
  
  if (includeAppointments && appointments && appointments.length > 0) {
    summary += '\n**Recent Appointments:**\n';
    for (const apt of appointments.slice(0, 3)) {
      summary += `- ${apt.date} at ${apt.startTime}: ${apt.reason || 'General visit'} (${apt.status})\n`;
    }
  }
  
  return summary;
}

function calculateAge(dateOfBirth: string): number {
  const birthDate = new Date(dateOfBirth);
  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  return age;
}

/**
 * Book an appointment through chat conversation
 */
export async function bookAppointmentFromChat(
  doctorId: string,
  patientId: string,
  date: Date,
  time: string,
  reason: string,
  duration: number = 30
): Promise<{ success: boolean; appointment?: Appointment; error?: string }> {
  try {
    const dateStr = date.toISOString().split('T')[0];

    // Verify availability
    const slots = await checkAvailability(doctorId, date, time);
    const targetSlot = slots.find(s => s.time === time && s.available);

    if (!targetSlot) {
      return {
        success: false,
        error: 'This time slot is not available. Would you like me to suggest alternative times?'
      };
    }

    // Calculate end time
    const [hours, minutes] = time.split(':').map(Number);
    const endMinutes = (minutes + duration) % 60;
    const endHours = minutes + duration >= 60 ? hours + 1 : hours;
    const endTime = `${endHours.toString().padStart(2, '0')}:${endMinutes.toString().padStart(2, '0')}`;

    // Create appointment
    const appointment = await storage.createAppointment({
      doctorId,
      patientId,
      date: dateStr,
      startTime: time,
      endTime,
      reason,
      status: 'confirmed',
      type: 'in-person',
      notes: 'Booked via Assistant Lysa chat'
    });

    return {
      success: true,
      appointment
    };
  } catch (error) {
    console.error('Error booking appointment:', error);
    return {
      success: false,
      error: 'Failed to book appointment. Please try again.'
    };
  }
}
