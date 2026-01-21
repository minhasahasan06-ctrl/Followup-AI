import OpenAI from "openai";
import { gcsService } from "./services/gcpStorageService";
import fs from "fs/promises";
import path from "path";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

interface VoiceProcessingResult {
  audioFileUrl: string;
  audioFileName: string;
  audioFileSize: number;
  audioDuration: number;
  transcription: string;
  extractedSymptoms: Array<{ symptom: string; severity: string; confidence: number }>;
  extractedMood: string;
  moodScore: number;
  medicationAdherence: Array<{ medication: string; taken: boolean; time?: string }>;
  extractedMetrics: {
    heartRate?: number;
    bloodPressure?: string;
    temperature?: number;
    sleepHours?: number;
    stepsCount?: number;
    waterIntake?: number;
  };
  sentimentScore: number;
  empathyLevel: string;
  concernsRaised: boolean;
  concernsSummary: string | null;
  aiResponse: string;
  conversationSummary: string;
  needsFollowup: boolean;
  followupReason: string | null;
  recommendedActions: string[];
}

export async function processVoiceFollowup(
  audioBuffer: Buffer,
  fileName: string,
  patientId: string,
  mimeType: string = "audio/webm"
): Promise<VoiceProcessingResult> {
  try {
    const timestamp = Date.now();
    const gcsKey = `${timestamp}-${fileName}`;
    
    const uploadResult = await gcsService.uploadFile(
      audioBuffer,
      gcsKey,
      {
        contentType: mimeType,
        folder: `voice-followups/${patientId}`,
        metadata: {
          patientId,
          uploadedAt: new Date().toISOString(),
        },
      }
    );
    
    const audioFileUrl = uploadResult.url;

    const tempFilePath = path.join("/tmp", fileName);
    await fs.writeFile(tempFilePath, audioBuffer);

    const fileStream = (await import("fs")).createReadStream(tempFilePath);
    
    const transcriptionResponse = await openai.audio.transcriptions.create({
      file: fileStream as any,
      model: "whisper-1",
      response_format: "verbose_json",
      language: "en",
    });

    const transcription = transcriptionResponse.text;
    const audioDuration = Math.round(transcriptionResponse.duration || 0);

    await fs.unlink(tempFilePath).catch(() => {});

    const extractionPrompt = `You are Agent Clona, an empathetic AI health companion for immunocompromised patients. A patient just recorded a voice message. Analyze the transcription and extract health information.

Transcription: "${transcription}"

Extract the following information in JSON format:
{
  "symptoms": [{"symptom": string, "severity": "mild|moderate|severe", "confidence": 0-1}],
  "mood": "positive|neutral|anxious|stressed|depressed|frustrated",
  "moodScore": number (-1.0 to 1.0),
  "medicationAdherence": [{"medication": string, "taken": boolean, "time"?: string}],
  "metrics": {
    "heartRate"?: number,
    "bloodPressure"?: string (e.g., "120/80"),
    "temperature"?: number (F),
    "sleepHours"?: number,
    "stepsCount"?: number,
    "waterIntake"?: number (oz)
  },
  "sentimentScore": number (-1.0 to 1.0),
  "concernsRaised": boolean,
  "concernsSummary": string | null,
  "needsFollowup": boolean,
  "followupReason": string | null,
  "recommendedActions": string[]
}

Be thorough but only extract information that is explicitly mentioned or clearly implied.`;

    const extractionResponse = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content:
            "You are Agent Clona, a warm and empathetic AI health companion. Extract health data from patient voice messages accurately. Always respond with valid JSON only, no additional text.",
        },
        { role: "user", content: extractionPrompt },
      ],
      temperature: 0.3,
      response_format: { type: "json_object" },
    });

    let extractedData: any = {};
    try {
      extractedData = JSON.parse(extractionResponse.choices[0].message.content || "{}");
    } catch (e) {
      console.error("Failed to parse extraction response:", e);
      extractedData = {
        symptoms: [],
        mood: "neutral",
        moodScore: 0,
        medicationAdherence: [],
        metrics: {},
        sentimentScore: 0,
        concernsRaised: false,
        concernsSummary: null,
        needsFollowup: false,
        followupReason: null,
        recommendedActions: [],
      };
    }

    const responsePrompt = `You are Agent Clona, a warm and empathetic AI health companion for immunocompromised patients. A patient just shared their daily voice check-in.

Transcription: "${transcription}"

Extracted insights:
- Mood: ${extractedData.mood}
- Symptoms: ${extractedData.symptoms?.map((s: any) => s.symptom).join(", ") || "none"}
- Concerns: ${extractedData.concernsSummary || "none"}

Generate a warm, caring, and supportive response (2-3 sentences) that:
1. Acknowledges what they shared
2. Validates their feelings
3. Provides gentle encouragement
4. Addresses any concerns with compassion

Make it feel like a caring friend checking in, not a medical robot. Use simple, warm language.`;

    const responseGeneration = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content:
            "You are Agent Clona, a warm, empathetic AI health companion. Respond with genuine care and understanding, like a supportive friend who happens to know about health.",
        },
        { role: "user", content: responsePrompt },
      ],
      temperature: 0.8,
      max_tokens: 250,
    });

    const aiResponse = responseGeneration.choices[0].message.content || "Thank you for sharing. I'm here to support you.";

    const summaryPrompt = `Summarize this health check-in in one concise sentence for medical records:
Transcription: "${transcription}"
Extracted: ${JSON.stringify(extractedData)}`;

    const summaryGeneration = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: summaryPrompt }],
      temperature: 0.3,
      max_tokens: 100,
    });

    const conversationSummary = summaryGeneration.choices[0].message.content || "Daily voice check-in completed";

    let empathyLevel = "supportive";
    if (extractedData.concernsRaised || extractedData.moodScore < -0.3) {
      empathyLevel = "empathetic";
    } else if (extractedData.moodScore > 0.3) {
      empathyLevel = "encouraging";
    } else if (extractedData.needsFollowup) {
      empathyLevel = "concerned";
    }

    return {
      audioFileUrl,
      audioFileName: fileName,
      audioFileSize: audioBuffer.length,
      audioDuration,
      transcription,
      extractedSymptoms: extractedData.symptoms || [],
      extractedMood: extractedData.mood || "neutral",
      moodScore: extractedData.moodScore || 0,
      medicationAdherence: extractedData.medicationAdherence || [],
      extractedMetrics: extractedData.metrics || {},
      sentimentScore: extractedData.sentimentScore || 0,
      empathyLevel,
      concernsRaised: extractedData.concernsRaised || false,
      concernsSummary: extractedData.concernsSummary || null,
      aiResponse,
      conversationSummary,
      needsFollowup: extractedData.needsFollowup || false,
      followupReason: extractedData.followupReason || null,
      recommendedActions: extractedData.recommendedActions || [],
    };
  } catch (error) {
    console.error("Voice processing error:", error);
    throw new Error("Failed to process voice followup: " + (error as Error).message);
  }
}
