import OpenAI from 'openai';
import fs from 'fs/promises';
import path from 'path';
import type { Storage } from './storage';

function getOpenAIClient(): OpenAI | null {
  const baaSignedEnv = process.env.OPENAI_BAA_SIGNED;
  const isBaaSigned = baaSignedEnv === 'true' || baaSignedEnv === '1';
  
  if (!isBaaSigned || !process.env.OPENAI_API_KEY) {
    return null;
  }
  
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}

interface VoiceTranscription {
  text: string;
  duration: number;
  language: string;
}

interface VoiceResponse {
  text: string;
  audioUrl?: string;
}

class VoiceInterfaceService {
  private storage: Storage;

  constructor(storage: Storage) {
    this.storage = storage;
  }

  async transcribeAudio(audioFilePath: string): Promise<VoiceTranscription> {
    const openai = getOpenAIClient();

    if (!openai) {
      throw new Error('OpenAI BAA not signed. Voice transcription unavailable.');
    }

    try {
      const { createReadStream } = await import('fs');
      const transcription = await openai.audio.transcriptions.create({
        file: createReadStream(audioFilePath) as any,
        model: 'whisper-1',
        language: 'en',
      });

      return {
        text: transcription.text,
        duration: 0,
        language: 'en',
      };
    } catch (error) {
      console.error('Error transcribing audio:', error);
      throw error;
    }
  }

  async generateSpeech(text: string, voice: 'alloy' | 'echo' | 'fable' | 'onyx' | 'nova' | 'shimmer' = 'nova'): Promise<Buffer> {
    const openai = getOpenAIClient();

    if (!openai) {
      throw new Error('OpenAI BAA not signed. Text-to-speech unavailable.');
    }

    try {
      const response = await openai.audio.speech.create({
        model: 'tts-1',
        voice,
        input: text,
      });

      const buffer = Buffer.from(await response.arrayBuffer());
      return buffer;
    } catch (error) {
      console.error('Error generating speech:', error);
      throw error;
    }
  }

  async processVoiceFollowup(
    patientId: string,
    audioFilePath: string
  ): Promise<{ transcription: VoiceTranscription; aiResponse: VoiceResponse }> {
    const transcription = await this.transcribeAudio(audioFilePath);
    
    const openai = getOpenAIClient();

    let aiResponseText = 'Thank you for your update. Your doctor will review it soon.';

    if (openai) {
      const completion = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content:
              'You are Agent Clona, a compassionate AI health assistant. Provide empathetic, supportive responses to patient health updates. Ask relevant follow-up questions but never provide medical diagnoses or treatment advice.',
          },
          {
            role: 'user',
            content: `Patient voice update: ${transcription.text}`,
          },
        ],
        temperature: 0.7,
        max_tokens: 200,
      });

      aiResponseText = completion.choices[0]?.message?.content || aiResponseText;
    }

    return {
      transcription,
      aiResponse: {
        text: aiResponseText,
      },
    };
  }
}

export let voiceInterfaceService: VoiceInterfaceService;

export function initVoiceInterfaceService(storage: Storage) {
  voiceInterfaceService = new VoiceInterfaceService(storage);
}
