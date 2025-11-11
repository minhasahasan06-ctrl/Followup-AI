import OpenAI from 'openai';
import type { Storage } from './storage';

function getOpenAIClient(): OpenAI | null {
  const baaSignedEnv = process.env.OPENAI_BAA_SIGNED;
  const isBaaSigned = baaSignedEnv === 'true' || baaSignedEnv === '1';
  
  if (!isBaaSigned || !process.env.OPENAI_API_KEY) {
    return null;
  }
  
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface ChatbotContext {
  doctorId: string;
  clinicName?: string;
  sessionId: string;
  messages: ChatMessage[];
}

class ChatbotService {
  private storage: Storage;
  private sessions: Map<string, ChatbotContext> = new Map();

  constructor(storage: Storage) {
    this.storage = storage;
  }

  async initializeSession(doctorId: string, sessionId: string): Promise<ChatbotContext> {
    const doctor = await this.storage.getUser(doctorId);
    const clinicName = `Dr. ${doctor?.firstName} ${doctor?.lastName}'s Office`;

    const systemPrompt = `You are Assistant Lysa, a helpful and professional AI receptionist for ${clinicName}. Your role is to:
1. Help patients schedule appointments
2. Answer common questions about services, hours, and location
3. Collect patient information (name, contact, reason for visit)
4. Provide general health information (but always recommend consulting the doctor for specific medical advice)
5. Be empathetic, professional, and HIPAA-compliant

NEVER:
- Provide specific medical diagnoses or treatment recommendations
- Share patient information with unauthorized individuals
- Make promises about medical outcomes

Office hours: Monday-Friday, 9 AM - 5 PM
Location: 123 Medical Plaza, Suite 100

If a patient wants to schedule an appointment, collect: name, phone, email, preferred date/time, and reason for visit.`;

    const context: ChatbotContext = {
      doctorId,
      clinicName,
      sessionId,
      messages: [
        {
          role: 'system',
          content: systemPrompt,
        },
        {
          role: 'assistant',
          content: `Hello! I'm Assistant Lysa from ${clinicName}. How can I help you today?`,
        },
      ],
    };

    this.sessions.set(sessionId, context);
    return context;
  }

  async sendMessage(
    sessionId: string,
    message: string,
    doctorId?: string
  ): Promise<{ response: string; intent?: string; extractedData?: any }> {
    try {
      let context = this.sessions.get(sessionId);

      if (!context) {
        if (!doctorId) {
          throw new Error('Doctor ID required for new session');
        }
        context = await this.initializeSession(doctorId, sessionId);
      }

      context.messages.push({
        role: 'user',
        content: message,
      });

      const openai = getOpenAIClient();

      if (!openai) {
        const fallbackResponse = this.generateFallbackResponse(message);
        context.messages.push({
          role: 'assistant',
          content: fallbackResponse,
        });
        return { response: fallbackResponse };
      }

      const completion = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: context.messages as any,
        temperature: 0.7,
        max_tokens: 500,
      });

      const response = completion.choices[0]?.message?.content || 'I apologize, but I encountered an error. Please try again.';

      context.messages.push({
        role: 'assistant',
        content: response,
      });

      const intent = await this.extractIntent(message);
      const extractedData = await this.extractPatientData(context.messages);

      this.sessions.set(sessionId, context);

      return {
        response,
        intent,
        extractedData,
      };
    } catch (error) {
      console.error('Chatbot error:', error);
      return {
        response: 'I apologize, but I encountered an error. Please try again or call our office directly.',
      };
    }
  }

  private generateFallbackResponse(message: string): string {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('appointment') || lowerMessage.includes('schedule')) {
      return "I'd be happy to help you schedule an appointment! To get started, I'll need your name, phone number, email, and the reason for your visit. What works best for you?";
    }

    if (lowerMessage.includes('hours') || lowerMessage.includes('open')) {
      return "Our office hours are Monday through Friday, 9 AM to 5 PM. Is there anything else I can help you with?";
    }

    if (lowerMessage.includes('location') || lowerMessage.includes('address')) {
      return "We're located at 123 Medical Plaza, Suite 100. Is there anything else you'd like to know?";
    }

    return "Thank you for your message. How can I assist you today? I can help with scheduling appointments, answering questions about our services, or providing general information about our office.";
  }

  private async extractIntent(message: string): Promise<string> {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('appointment') || lowerMessage.includes('schedule')) {
      return 'schedule_appointment';
    }
    if (lowerMessage.includes('cancel')) {
      return 'cancel_appointment';
    }
    if (lowerMessage.includes('reschedule')) {
      return 'reschedule_appointment';
    }
    if (lowerMessage.includes('hours') || lowerMessage.includes('when open')) {
      return 'office_hours';
    }
    if (lowerMessage.includes('location') || lowerMessage.includes('address')) {
      return 'office_location';
    }

    return 'general_inquiry';
  }

  private async extractPatientData(messages: ChatMessage[]): Promise<any> {
    const userMessages = messages
      .filter((m) => m.role === 'user')
      .map((m) => m.content)
      .join(' ');

    const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/;
    const phoneRegex = /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/;
    const nameRegex = /my name is ([A-Z][a-z]+ [A-Z][a-z]+)/i;

    const email = userMessages.match(emailRegex)?.[0];
    const phone = userMessages.match(phoneRegex)?.[0];
    const name = userMessages.match(nameRegex)?.[1];

    return {
      name: name || null,
      email: email || null,
      phone: phone || null,
    };
  }

  async getChatHistory(sessionId: string): Promise<ChatMessage[]> {
    const context = this.sessions.get(sessionId);
    if (!context) {
      return [];
    }
    return context.messages.filter((m) => m.role !== 'system');
  }

  async endSession(sessionId: string): Promise<void> {
    this.sessions.delete(sessionId);
  }
}

export let chatbotService: ChatbotService;

export function initChatbotService(storage: Storage) {
  chatbotService = new ChatbotService(storage);
}
