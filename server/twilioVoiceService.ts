import twilio from 'twilio';
import type { Storage } from './storage';

const VoiceResponse = twilio.twiml.VoiceResponse;

interface CallContext {
  from: string;
  to: string;
  callSid: string;
  doctorId?: string;
  patientId?: string;
}

class TwilioVoiceService {
  private client: twilio.Twilio;
  private storage: Storage;

  constructor(storage: Storage) {
    this.storage = storage;
    this.client = twilio(
      process.env.TWILIO_ACCOUNT_SID,
      process.env.TWILIO_AUTH_TOKEN
    );
  }

  generateIVRResponse(doctorId?: string): string {
    const twiml = new VoiceResponse();
    
    const gather = twiml.gather({
      input: ['speech', 'dtmf'],
      timeout: 5,
      numDigits: 1,
      action: '/api/v1/voice/ivr/menu',
      method: 'POST',
    });

    gather.say({
      voice: 'alice',
    }, 'Thank you for calling. Press 1 to schedule an appointment. Press 2 to speak with a receptionist. Press 3 to leave a message.');

    twiml.redirect('/api/v1/voice/ivr/menu');

    return twiml.toString();
  }

  handleMenuSelection(digit: string, speech?: string): string {
    const twiml = new VoiceResponse();

    switch (digit) {
      case '1':
        twiml.say('Connecting you to our appointment scheduling system.');
        twiml.redirect('/api/v1/voice/ivr/schedule');
        break;
      case '2':
        twiml.say('Please hold while we connect you to our receptionist.');
        twiml.dial({
          action: '/api/v1/voice/dial/complete',
        }, process.env.TWILIO_PHONE_NUMBER);
        break;
      case '3':
        twiml.say('Please leave a message after the beep. Press the pound key when finished.');
        twiml.record({
          action: '/api/v1/voice/voicemail/complete',
          maxLength: 180,
          playBeep: true,
          transcribe: true,
          transcribeCallback: '/api/v1/voice/voicemail/transcription',
        });
        break;
      default:
        twiml.say('Invalid selection. Please try again.');
        twiml.redirect('/api/v1/voice/ivr/welcome');
        break;
    }

    return twiml.toString();
  }

  handleSchedulingFlow(): string {
    const twiml = new VoiceResponse();
    
    const gather = twiml.gather({
      input: ['speech'],
      timeout: 10,
      action: '/api/v1/voice/schedule/process',
      method: 'POST',
    });

    gather.say('Please describe the reason for your visit and your preferred date and time.');

    twiml.redirect('/api/v1/voice/schedule/process');

    return twiml.toString();
  }

  async processSchedulingRequest(
    speechResult: string,
    from: string
  ): Promise<string> {
    const twiml = new VoiceResponse();

    try {
      const patient = await this.storage.getUserByPhoneNumber(from);
      
      if (!patient) {
        twiml.say('We could not find your patient record. Please call back and press 2 to speak with our receptionist.');
        twiml.hangup();
        return twiml.toString();
      }

      await this.storage.createCallLog({
        doctorId: '', 
        fromNumber: from,
        toNumber: process.env.TWILIO_PHONE_NUMBER || '',
        direction: 'inbound',
        callType: 'scheduling_request',
        duration: 0,
        recordingUrl: null,
        transcription: speechResult,
        status: 'completed',
        notes: `Automated scheduling request: ${speechResult}`,
      });

      twiml.say('Thank you. We have received your scheduling request and will call you back to confirm your appointment.');
      twiml.hangup();
    } catch (error) {
      console.error('Error processing scheduling request:', error);
      twiml.say('We encountered an error. Please try again later or press 2 to speak with our receptionist.');
      twiml.redirect('/api/v1/voice/ivr/welcome');
    }

    return twiml.toString();
  }

  async processVoicemail(
    callSid: string,
    recordingUrl: string,
    transcription: string,
    from: string
  ): Promise<void> {
    try {
      const patient = await this.storage.getUserByPhoneNumber(from);

      await this.storage.createCallLog({
        doctorId: '', 
        fromNumber: from,
        toNumber: process.env.TWILIO_PHONE_NUMBER || '',
        direction: 'inbound',
        callType: 'voicemail',
        duration: 0,
        recordingUrl,
        transcription,
        status: 'completed',
        notes: 'Voicemail received',
      });
    } catch (error) {
      console.error('Error processing voicemail:', error);
    }
  }

  generateVoicemailCompleteResponse(): string {
    const twiml = new VoiceResponse();
    twiml.say('Thank you for your message. We will get back to you as soon as possible. Goodbye.');
    twiml.hangup();
    return twiml.toString();
  }

  async makeOutboundCall(
    to: string,
    message: string,
    doctorId: string
  ): Promise<{ success: boolean; callSid?: string; error?: string }> {
    try {
      const call = await this.client.calls.create({
        to,
        from: process.env.TWILIO_PHONE_NUMBER,
        twiml: `<Response><Say>${message}</Say></Response>`,
      });

      await this.storage.createCallLog({
        doctorId,
        fromNumber: process.env.TWILIO_PHONE_NUMBER || '',
        toNumber: to,
        direction: 'outbound',
        callType: 'notification',
        duration: 0,
        status: 'initiated',
        notes: message,
      });

      return { success: true, callSid: call.sid };
    } catch (error) {
      console.error('Error making outbound call:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async getCallRecording(recordingSid: string): Promise<string | null> {
    try {
      const recording = await this.client.recordings(recordingSid).fetch();
      return `https://api.twilio.com${recording.uri.replace('.json', '.mp3')}`;
    } catch (error) {
      console.error('Error fetching recording:', error);
      return null;
    }
  }
}

export let twilioVoiceService: TwilioVoiceService;

export function initTwilioVoiceService(storage: Storage) {
  twilioVoiceService = new TwilioVoiceService(storage);
}
