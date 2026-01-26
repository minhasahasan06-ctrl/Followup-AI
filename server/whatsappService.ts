import type { Storage } from './storage';
import OpenAI from 'openai';
import { format, addDays, parseISO, startOfDay, endOfDay } from 'date-fns';

const twilioPhoneNumber = process.env.TWILIO_PHONE_NUMBER;

// Twilio integration disabled - using stub
const twilioClient = {
  messages: {
    create: async (opts: any) => {
      console.warn('[TWILIO] WhatsApp messaging disabled - Twilio removed from dependencies');
      console.log('[TWILIO] Would send WhatsApp message:', { to: opts.to, body: opts.body?.substring(0, 50) + '...' });
      return { sid: 'mock-sid-' + Date.now(), status: 'stub' };
    }
  }
};

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface WhatsAppMessage {
  from: string;
  to: string;
  body: string;
  messageSid?: string;
}

interface AppointmentIntent {
  action: 'book' | 'reschedule' | 'cancel' | 'check' | 'unknown';
  patientName?: string;
  doctorName?: string;
  date?: string;
  time?: string;
  reason?: string;
  appointmentId?: string;
  confidence: number;
}

interface WhatsAppConversation {
  id: string;
  phoneNumber: string;
  patientId?: string;
  doctorId: string;
  messages: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
  }>;
  currentIntent?: AppointmentIntent;
  status: 'active' | 'completed' | 'pending_confirmation';
  createdAt: Date;
  updatedAt: Date;
}

const activeConversations = new Map<string, WhatsAppConversation>();

export class WhatsAppAppointmentService {
  private storage: Storage;

  constructor(storage: Storage) {
    this.storage = storage;
  }

  async sendWhatsAppMessage(to: string, message: string): Promise<{ success: boolean; sid?: string }> {
    if (!twilioClient || !twilioPhoneNumber) {
      console.error('[WHATSAPP] WhatsApp not configured');
      return { success: false };
    }

    try {
      const whatsappFrom = `whatsapp:${twilioPhoneNumber}`;
      const whatsappTo = to.startsWith('whatsapp:') ? to : `whatsapp:${to}`;

      const result = await twilioClient.messages.create({
        body: message,
        from: whatsappFrom,
        to: whatsappTo,
      });

      console.log(`[WHATSAPP] Message sent to ${to}. SID: ${result.sid}`);
      return { success: true, sid: result.sid };
    } catch (error: any) {
      console.error('[WHATSAPP] Error sending message:', error.message);
      return { success: false };
    }
  }

  async processIncomingMessage(message: WhatsAppMessage, doctorId: string): Promise<string> {
    const phoneNumber = message.from.replace('whatsapp:', '');
    const conversationKey = `${doctorId}:${phoneNumber}`;

    let conversation = activeConversations.get(conversationKey);
    if (!conversation) {
      conversation = {
        id: `conv_${Date.now()}`,
        phoneNumber,
        doctorId,
        messages: [],
        status: 'active',
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      activeConversations.set(conversationKey, conversation);
    }

    conversation.messages.push({
      role: 'user',
      content: message.body,
      timestamp: new Date(),
    });
    conversation.updatedAt = new Date();

    const intent = await this.detectAppointmentIntent(message.body, conversation);
    conversation.currentIntent = intent;

    const response = await this.generateResponse(conversation, intent);

    conversation.messages.push({
      role: 'assistant',
      content: response,
      timestamp: new Date(),
    });

    return response;
  }

  private async detectAppointmentIntent(
    message: string,
    conversation: WhatsAppConversation
  ): Promise<AppointmentIntent> {
    try {
      const systemPrompt = `You are an AI assistant that helps detect appointment-related intents from WhatsApp messages.
Analyze the message and extract appointment details.

Previous messages in conversation:
${conversation.messages.slice(-5).map(m => `${m.role}: ${m.content}`).join('\n')}

Current message: "${message}"

Respond in JSON format:
{
  "action": "book" | "reschedule" | "cancel" | "check" | "unknown",
  "patientName": "extracted patient name or null",
  "date": "extracted date in YYYY-MM-DD format or relative like 'tomorrow', 'next week' or null",
  "time": "extracted time in HH:MM format or 'morning', 'afternoon', 'evening' or null",
  "reason": "reason for appointment or null",
  "appointmentId": "if mentioned, null otherwise",
  "confidence": 0.0 to 1.0
}`;

      const completion = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: message },
        ],
        response_format: { type: 'json_object' },
        max_tokens: 200,
      });

      const result = JSON.parse(completion.choices[0].message.content || '{}');
      return {
        action: result.action || 'unknown',
        patientName: result.patientName,
        date: result.date,
        time: result.time,
        reason: result.reason,
        appointmentId: result.appointmentId,
        confidence: result.confidence || 0.5,
      };
    } catch (error) {
      console.error('[WHATSAPP] Error detecting intent:', error);
      return { action: 'unknown', confidence: 0 };
    }
  }

  private async generateResponse(
    conversation: WhatsAppConversation,
    intent: AppointmentIntent
  ): Promise<string> {
    const doctor = await this.storage.getUser(conversation.doctorId);
    const doctorName = doctor ? `Dr. ${doctor.lastName}` : 'the doctor';

    switch (intent.action) {
      case 'book':
        return this.handleBookingIntent(conversation, intent, doctorName);
      case 'reschedule':
        return this.handleRescheduleIntent(conversation, intent, doctorName);
      case 'cancel':
        return this.handleCancelIntent(conversation, intent, doctorName);
      case 'check':
        return this.handleCheckIntent(conversation, intent, doctorName);
      default:
        return this.handleGeneralQuery(conversation, doctorName);
    }
  }

  private async handleBookingIntent(
    conversation: WhatsAppConversation,
    intent: AppointmentIntent,
    doctorName: string
  ): Promise<string> {
    const missing: string[] = [];
    if (!intent.patientName) missing.push('patient name');
    if (!intent.date) missing.push('preferred date');
    if (!intent.time) missing.push('preferred time');

    if (missing.length > 0) {
      return `I'd be happy to help book an appointment with ${doctorName}. Could you please provide the following:\n${missing.map(m => `- ${m}`).join('\n')}`;
    }

    const availableSlots = await this.getAvailableSlots(
      conversation.doctorId,
      intent.date!
    );

    if (availableSlots.length === 0) {
      return `I'm sorry, ${doctorName} doesn't have any available slots on ${intent.date}. Would you like to check another date?`;
    }

    const requestedTime = intent.time!.toLowerCase();
    let matchingSlots = availableSlots;

    if (requestedTime === 'morning') {
      matchingSlots = availableSlots.filter(s => parseInt(s.split(':')[0]) < 12);
    } else if (requestedTime === 'afternoon') {
      matchingSlots = availableSlots.filter(s => {
        const hour = parseInt(s.split(':')[0]);
        return hour >= 12 && hour < 17;
      });
    } else if (requestedTime === 'evening') {
      matchingSlots = availableSlots.filter(s => parseInt(s.split(':')[0]) >= 17);
    }

    if (matchingSlots.length === 0) {
      return `No ${requestedTime} slots available on ${intent.date}. Available times are: ${availableSlots.join(', ')}. Which would you prefer?`;
    }

    conversation.status = 'pending_confirmation';
    const suggestedSlot = matchingSlots[0];

    return `Great! I can book an appointment for ${intent.patientName} with ${doctorName} on ${intent.date} at ${suggestedSlot}.\n\nReason: ${intent.reason || 'General consultation'}\n\nPlease reply "confirm" to proceed or "change" to select a different time.\n\nAvailable times: ${matchingSlots.slice(0, 5).join(', ')}`;
  }

  private async handleRescheduleIntent(
    conversation: WhatsAppConversation,
    intent: AppointmentIntent,
    doctorName: string
  ): Promise<string> {
    if (!intent.appointmentId && !intent.patientName) {
      return `To reschedule an appointment with ${doctorName}, please provide either:\n- The appointment confirmation number\n- The patient's name and original appointment date`;
    }

    return `I'll help you reschedule the appointment. What new date and time would work best for you?\n\nAvailable options:\n- Tomorrow morning\n- Tomorrow afternoon\n- Any specific date`;
  }

  private async handleCancelIntent(
    conversation: WhatsAppConversation,
    intent: AppointmentIntent,
    doctorName: string
  ): Promise<string> {
    if (!intent.appointmentId && !intent.patientName) {
      return `To cancel an appointment with ${doctorName}, please provide:\n- The appointment confirmation number\n- Or the patient's name`;
    }

    conversation.status = 'pending_confirmation';
    return `Are you sure you want to cancel this appointment? Reply "yes" to confirm cancellation or "no" to keep the appointment.`;
  }

  private async handleCheckIntent(
    conversation: WhatsAppConversation,
    intent: AppointmentIntent,
    doctorName: string
  ): Promise<string> {
    if (intent.patientName) {
      const patientAppointments = await this.findPatientAppointments(
        conversation.doctorId,
        intent.patientName
      );

      if (patientAppointments.length === 0) {
        return `No upcoming appointments found for ${intent.patientName} with ${doctorName}.`;
      }

      return `Upcoming appointments for ${intent.patientName} with ${doctorName}:\n${patientAppointments.map((a, i) => `${i + 1}. ${format(new Date(a.startTime), 'MMM d, yyyy')} at ${format(new Date(a.startTime), 'h:mm a')} - ${a.appointmentType}`).join('\n')}`;
    }

    return `Would you like to:\n1. Check a specific patient's appointments\n2. View today's schedule\n3. View available slots for a date\n\nPlease provide more details.`;
  }

  private async handleGeneralQuery(
    conversation: WhatsAppConversation,
    doctorName: string
  ): Promise<string> {
    return `Hello! I'm Assistant Lysa, ${doctorName}'s AI receptionist. I can help you with:\n\n📅 Book an appointment\n🔄 Reschedule an appointment\n❌ Cancel an appointment\n📋 Check appointment status\n\nHow may I assist you today?`;
  }

  private async getAvailableSlots(doctorId: string, dateStr: string): Promise<string[]> {
    let targetDate: Date;

    if (dateStr.toLowerCase() === 'tomorrow') {
      targetDate = addDays(new Date(), 1);
    } else if (dateStr.toLowerCase() === 'today') {
      targetDate = new Date();
    } else {
      targetDate = parseISO(dateStr);
    }

    const dayStart = startOfDay(targetDate);
    const dayEnd = endOfDay(targetDate);

    const existingAppointments = await this.storage.getAppointmentsByDoctor(doctorId);
    const dayAppointments = existingAppointments.filter(a => {
      const aptDate = new Date(a.startTime);
      return aptDate >= dayStart && aptDate <= dayEnd;
    });

    const bookedTimes = new Set(
      dayAppointments.map(a => format(new Date(a.startTime), 'HH:mm'))
    );

    const allSlots = [
      '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
      '14:00', '14:30', '15:00', '15:30', '16:00', '16:30',
      '17:00', '17:30',
    ];

    return allSlots.filter(slot => !bookedTimes.has(slot));
  }

  private async findPatientAppointments(doctorId: string, patientName: string) {
    const appointments = await this.storage.getAppointmentsByDoctor(doctorId);
    const now = new Date();

    return appointments.filter(a => {
      const isUpcoming = new Date(a.startTime) > now;
      const matchesPatient = a.patientName?.toLowerCase().includes(patientName.toLowerCase());
      return isUpcoming && matchesPatient;
    });
  }

  async confirmBooking(
    conversation: WhatsAppConversation,
    patientId: string,
    date: Date,
    time: string,
    reason: string
  ): Promise<{ success: boolean; appointmentId?: string; error?: string }> {
    try {
      const [hours, minutes] = time.split(':').map(Number);
      const startTime = new Date(date);
      startTime.setHours(hours, minutes, 0, 0);

      const endTime = new Date(startTime);
      endTime.setMinutes(endTime.getMinutes() + 30);

      const conflicts = await this.storage.findAppointmentConflicts(
        conversation.doctorId,
        startTime,
        endTime
      );

      if (conflicts.length > 0) {
        return { success: false, error: 'Time slot is no longer available' };
      }

      const appointment = await this.storage.createAppointment({
        patientId,
        doctorId: conversation.doctorId,
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString(),
        appointmentType: 'consultation',
        status: 'confirmed',
        confirmationStatus: 'confirmed',
        notes: `Booked via WhatsApp. Reason: ${reason}`,
        source: 'whatsapp',
      });

      conversation.status = 'completed';
      return { success: true, appointmentId: appointment.id };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async sendAppointmentConfirmation(
    to: string,
    patientName: string,
    doctorName: string,
    date: string,
    time: string,
    appointmentId: string
  ): Promise<boolean> {
    const message = `✅ Appointment Confirmed!\n\nPatient: ${patientName}\nDoctor: ${doctorName}\nDate: ${date}\nTime: ${time}\nConfirmation #: ${appointmentId.slice(0, 8).toUpperCase()}\n\nTo reschedule or cancel, reply to this message.\n\n- Followup AI`;

    const result = await this.sendWhatsAppMessage(to, message);
    return result.success;
  }

  async sendAppointmentReminder(
    to: string,
    patientName: string,
    doctorName: string,
    appointmentTime: Date
  ): Promise<boolean> {
    const date = format(appointmentTime, 'MMMM d, yyyy');
    const time = format(appointmentTime, 'h:mm a');

    const message = `📅 Appointment Reminder\n\nHi ${patientName}!\n\nThis is a reminder of your upcoming appointment:\n\nDoctor: ${doctorName}\nDate: ${date}\nTime: ${time}\n\nPlease reply:\n- "confirm" to confirm attendance\n- "reschedule" to change the time\n- "cancel" to cancel the appointment\n\n- Followup AI`;

    const result = await this.sendWhatsAppMessage(to, message);
    return result.success;
  }

  getActiveConversation(doctorId: string, phoneNumber: string): WhatsAppConversation | undefined {
    return activeConversations.get(`${doctorId}:${phoneNumber}`);
  }

  clearConversation(doctorId: string, phoneNumber: string): void {
    activeConversations.delete(`${doctorId}:${phoneNumber}`);
  }
}

export function createWhatsAppService(storage: Storage): WhatsAppAppointmentService {
  return new WhatsAppAppointmentService(storage);
}
