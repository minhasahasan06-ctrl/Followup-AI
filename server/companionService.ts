import OpenAI from "openai";
import { storage } from "./storage";
import type { InsertCompanionCheckIn } from "@shared/schema";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface CheckInRequest {
  patientId: string;
  checkInType: string;
  userInput: string;
  context?: {
    lastCheckIn?: Date;
    recentMedications?: any[];
    recentMeals?: any[];
    currentStreak?: number;
  };
}

interface CheckInResponse {
  response: string;
  empathyLevel: string;
  extractedData: any;
  conversationSummary: string;
  sentimentScore: number;
  concernsRaised: boolean;
  needsFollowup: boolean;
  followupReason?: string;
}

export async function processCompanionCheckIn(
  request: CheckInRequest
): Promise<{ checkInId: string; response: string; concernsRaised: boolean }> {
  try {
    // 1. Get patient engagement data
    let engagement = await storage.getCompanionEngagement(request.patientId);
    if (!engagement) {
      // Initialize engagement tracking
      engagement = await storage.upsertCompanionEngagement({
        patientId: request.patientId,
        totalCheckIns: 0,
        currentStreak: 0,
        longestStreak: 0,
        totalConcernsRaised: 0,
        companionPersonality: "empathetic",
        preferredTone: "warm",
        notificationPreference: "gentle",
        engagementScore: 50,
      });
    }

    // 2. Process check-in with AI
    const aiResponse = await generateCompanionResponse(request, engagement);

    // 3. Extract structured data from conversation
    const extractedData = await extractHealthData(request.userInput);

    // 4. Create check-in record
    const checkIn = await storage.createCompanionCheckIn({
      patientId: request.patientId,
      checkInType: request.checkInType,
      conversationSummary: aiResponse.conversationSummary,
      naturalLanguageInput: request.userInput,
      extractedData: extractedData,
      empathyLevel: aiResponse.empathyLevel,
      aiResponse: aiResponse.response,
      sentimentScore: aiResponse.sentimentScore.toString(),
      concernsRaised: aiResponse.concernsRaised,
      needsFollowup: aiResponse.needsFollowup,
      followupReason: aiResponse.followupReason,
      interactionCount: 1,
    });

    // 5. Update engagement metrics
    await updateEngagementMetrics(
      request.patientId,
      engagement,
      aiResponse.sentimentScore,
      aiResponse.concernsRaised
    );

    // 6. Process extracted health data
    await processExtractedHealthData(request.patientId, extractedData, checkIn.id);

    console.log(`Processed companion check-in for patient ${request.patientId}`);
    return {
      checkInId: checkIn.id,
      response: aiResponse.response,
      concernsRaised: aiResponse.concernsRaised,
    };
  } catch (error) {
    console.error("Error processing companion check-in:", error);
    throw error;
  }
}

async function generateCompanionResponse(
  request: CheckInRequest,
  engagement: any
): Promise<CheckInResponse> {
  const personality = engagement.companionPersonality || "empathetic";
  const tone = engagement.preferredTone || "warm";

  const systemPrompt = buildSystemPrompt(personality, tone);
  const userPrompt = buildUserPrompt(request);

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: systemPrompt,
        },
        {
          role: "user",
          content: userPrompt,
        },
      ],
      temperature: 0.8,
      response_format: { type: "json_object" },
    });

    const content = completion.choices[0]?.message?.content;
    if (!content) {
      console.error("Empty response from OpenAI for companion check-in");
      // Return fallback response
      return {
        response: "Thank you for checking in! I'm here to support you.",
        empathyLevel: "supportive",
        extractedData: {},
        conversationSummary: "Patient checked in",
        sentimentScore: 0,
        concernsRaised: false,
        needsFollowup: false,
      };
    }

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch (parseError) {
      console.error("Failed to parse OpenAI companion response as JSON:", parseError);
      console.error("Raw response:", content);
      // Return fallback response
      return {
        response: "Thank you for checking in! I'm here to support you.",
        empathyLevel: "supportive",
        extractedData: {},
        conversationSummary: "Patient checked in",
        sentimentScore: 0,
        concernsRaised: false,
        needsFollowup: false,
      };
    }

    return {
      response: parsed.response || "Thank you for checking in!",
      empathyLevel: parsed.empathyLevel || "supportive",
      extractedData: parsed.extractedData || {},
      conversationSummary: parsed.summary || "Patient checked in",
      sentimentScore: Number(parsed.sentimentScore) || 0,
      concernsRaised: Boolean(parsed.concernsRaised),
      needsFollowup: Boolean(parsed.needsFollowup),
      followupReason: parsed.followupReason || undefined,
    };
  } catch (error) {
    console.error("Error generating companion response:", error);
    // Return fallback response instead of throwing
    return {
      response: "Thank you for checking in! I'm here to support you.",
      empathyLevel: "supportive",
      extractedData: {},
      conversationSummary: "Patient checked in",
      sentimentScore: 0,
      concernsRaised: false,
      needsFollowup: false,
    };
  }
}

async function extractHealthData(userInput: string): Promise<any> {
  const prompt = `Extract structured health data from this patient's message:

"${userInput}"

Extract:
- Mood (if mentioned)
- Energy level (1-10 if mentioned)
- Symptoms (array of strings)
- Medications mentioned (with taken: true/false)
- Meals mentioned (type and description)
- Concerns or worries

Return JSON with format:
{
  "mood": "string or null",
  "energy": number or null,
  "symptoms": ["string array"],
  "medications": [{"name": "string", "taken": boolean}],
  "meals": [{"type": "string", "description": "string"}],
  "concerns": ["string array"]
}`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are a medical data extraction specialist. Extract health-related information from patient messages.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 0.3,
      response_format: { type: "json_object" },
    });

    const content = completion.choices[0]?.message?.content;
    if (!content) {
      console.error("Empty response from OpenAI for health data extraction");
      return {};
    }

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch (parseError) {
      console.error("Failed to parse OpenAI health data extraction response as JSON:", parseError);
      console.error("Raw response:", content);
      return {};
    }

    // Validate and sanitize extracted data
    return {
      mood: parsed.mood || null,
      energy: parsed.energy !== null && parsed.energy !== undefined ? Number(parsed.energy) : null,
      symptoms: Array.isArray(parsed.symptoms) ? parsed.symptoms : [],
      medications: Array.isArray(parsed.medications) ? parsed.medications : [],
      meals: Array.isArray(parsed.meals) ? parsed.meals : [],
      concerns: Array.isArray(parsed.concerns) ? parsed.concerns : [],
    };
  } catch (error) {
    console.error("Error extracting health data:", error);
    return {};
  }
}

async function updateEngagementMetrics(
  patientId: string,
  currentEngagement: any,
  sentimentScore: number,
  concernsRaised: boolean
): Promise<void> {
  try {
    // Calculate streak
    const lastCheckIn = currentEngagement.lastCheckInDate;
    const now = new Date();
    let newStreak = currentEngagement.currentStreak || 0;

    if (lastCheckIn) {
      const daysSinceLastCheckIn = Math.floor(
        (now.getTime() - new Date(lastCheckIn).getTime()) / (1000 * 60 * 60 * 24)
      );
      
      if (daysSinceLastCheckIn === 1) {
        newStreak++;
      } else if (daysSinceLastCheckIn > 1) {
        newStreak = 1;
      }
    } else {
      newStreak = 1;
    }

    // Calculate average sentiment
    const totalCheckIns = (currentEngagement.totalCheckIns || 0) + 1;
    const currentAvg = parseFloat(currentEngagement.avgSentimentScore || "0");
    const newAvg = ((currentAvg * (totalCheckIns - 1)) + sentimentScore) / totalCheckIns;

    // Update engagement
    await storage.updateCompanionEngagement(patientId, {
      totalCheckIns,
      currentStreak: newStreak,
      longestStreak: Math.max(newStreak, currentEngagement.longestStreak || 0),
      lastCheckInDate: now,
      avgSentimentScore: newAvg.toString(),
      totalConcernsRaised: (currentEngagement.totalConcernsRaised || 0) + (concernsRaised ? 1 : 0),
      engagementScore: calculateEngagementScore(totalCheckIns, newStreak, newAvg),
    });
  } catch (error) {
    console.error("Error updating engagement metrics:", error);
  }
}

async function processExtractedHealthData(
  patientId: string,
  extractedData: any,
  checkInId: string
): Promise<void> {
  try {
    // Process medication adherence
    if (extractedData.medications && Array.isArray(extractedData.medications)) {
      for (const med of extractedData.medications) {
        try {
          if (!med.name) continue; // Skip if no name
          
          const medications = await storage.getActiveMedications(patientId);
          const medication = medications.find(m => 
            m.name.toLowerCase().includes(med.name.toLowerCase())
          );

          if (medication && med.taken !== undefined) {
            await storage.createMedicationAdherence({
              medicationId: medication.id,
              patientId,
              scheduledTime: new Date(),
              takenAt: med.taken ? new Date() : undefined,
              status: med.taken ? "taken" : "missed",
              loggedBy: "companion",
              companionChecked: true,
            });
          }
        } catch (medError) {
          console.error("Error processing medication adherence:", medError);
          // Continue processing other medications
        }
      }
    }

    // Process meals
    if (extractedData.meals && Array.isArray(extractedData.meals)) {
      for (const mealData of extractedData.meals) {
        try {
          if (!mealData.description) continue; // Skip if no description
          
          await storage.createMeal({
            patientId,
            mealType: mealData.type || "snack",
            mealName: mealData.description.substring(0, 50),
            description: mealData.description,
            actualTime: new Date(),
            status: "eaten",
            companionLogged: true,
          });
        } catch (mealError) {
          console.error("Error processing meal:", mealError);
          // Continue processing other meals
        }
      }
    }

    // If concerns raised, create follow-up task
    if (extractedData.concerns && Array.isArray(extractedData.concerns)) {
      for (const concern of extractedData.concerns) {
        try {
          if (!concern || typeof concern !== 'string') continue; // Skip invalid concerns
          
          await storage.createDynamicTask({
            patientId,
            title: "Follow up on health concern",
            description: concern,
            taskType: "followup",
            generatedBy: "companion",
            completed: false,
          });
        } catch (taskError) {
          console.error("Error creating follow-up task:", taskError);
          // Continue processing other concerns
        }
      }
    }
  } catch (error) {
    console.error("Error processing extracted health data:", error);
    // Don't throw - this is a best-effort operation
  }
}

function buildSystemPrompt(personality: string, tone: string): string {
  const personalityMap: Record<string, string> = {
    empathetic: "You are a caring, empathetic health companion who genuinely cares about the patient's well-being. Show warmth and understanding.",
    motivational: "You are an encouraging, motivational health companion who inspires patients to stay on track. Be uplifting and energizing.",
    clinical: "You are a professional, clinical health companion who provides clear, factual health guidance. Be precise and informative.",
    friend: "You are a friendly, relatable health companion who feels like a trusted friend. Be casual yet supportive.",
  };

  const toneMap: Record<string, string> = {
    warm: "Use warm, caring language that makes the patient feel supported and valued.",
    professional: "Use professional, respectful language while maintaining approachability.",
    casual: "Use casual, conversational language that feels natural and relaxed.",
    cheerful: "Use cheerful, positive language that brightens the patient's day.",
  };

  return `${personalityMap[personality] || personalityMap.empathetic}
${toneMap[tone] || toneMap.warm}

You are checking in with an immunocompromised patient. Your goals:
1. Make them feel heard and supported
2. Extract important health information naturally
3. Identify any concerns that need medical attention
4. Encourage healthy behaviors
5. Maintain continuity between check-ins

CRITICAL RESPONSE FORMAT:
You must return ONLY valid JSON with this exact structure:
{
  "response": "Your empathetic response to the patient (2-4 sentences max, warm and personal)",
  "empathyLevel": "supportive|encouraging|urgent|celebratory",
  "extractedData": {
    "mood": "string or null",
    "energy": "number 1-10 or null",
    "symptoms": ["array of strings"],
    "medications": [{"name": "string", "taken": boolean}],
    "meals": [{"type": "breakfast|lunch|dinner|snack", "description": "string"}],
    "concerns": ["array of worries or concerns"]
  },
  "summary": "Brief 1-sentence summary of this check-in",
  "sentimentScore": number between -1 and 1 (-1=very negative, 0=neutral, 1=very positive),
  "concernsRaised": boolean (true if medical attention might be needed),
  "needsFollowup": boolean,
  "followupReason": "string explaining why followup needed, or null"
}

Keep responses natural, brief (2-4 sentences), and conversational. Do not use clinical jargon.`;
}

function buildUserPrompt(request: CheckInRequest): string {
  let context = "";
  if (request.context?.currentStreak && request.context.currentStreak > 1) {
    context += `\nStreak: ${request.context.currentStreak} days in a row! `;
  }
  
  return `Check-in type: ${request.checkInType}
Patient message: "${request.userInput}"
${context}

Respond naturally and extract health data.`;
}

function calculateEngagementScore(
  totalCheckIns: number,
  currentStreak: number,
  avgSentiment: number
): number {
  // Base score from check-ins (max 40 points)
  const checkInScore = Math.min(40, (totalCheckIns / 30) * 40);
  
  // Streak bonus (max 30 points)
  const streakScore = Math.min(30, (currentStreak / 14) * 30);
  
  // Sentiment score (max 30 points)
  // Convert from -1 to 1 range to 0 to 30
  const sentimentScore = ((avgSentiment + 1) / 2) * 30;
  
  return Math.round(checkInScore + streakScore + sentimentScore);
}

export async function suggestCheckInTime(patientId: string): Promise<string> {
  try {
    // Get recent check-ins to learn best time
    const recentCheckIns = await storage.getRecentCheckIns(patientId, 14);
    
    if (recentCheckIns.length === 0) {
      return "09:00"; // Default morning check-in
    }

    // Analyze when patient is most engaged
    const hourCounts: Record<number, number> = {};
    for (const checkIn of recentCheckIns) {
      const hour = new Date(checkIn.checkedInAt).getHours();
      hourCounts[hour] = (hourCounts[hour] || 0) + 1;
    }

    // Find most common hour
    let bestHour = 9;
    let maxCount = 0;
    for (const [hour, count] of Object.entries(hourCounts)) {
      if (count > maxCount) {
        maxCount = count;
        bestHour = parseInt(hour);
      }
    }

    return `${bestHour.toString().padStart(2, '0')}:00`;
  } catch (error) {
    console.error("Error suggesting check-in time:", error);
    return "09:00";
  }
}

export async function generateCheckInPrompt(
  patientId: string,
  checkInType: string
): Promise<string> {
  const prompts: Record<string, string[]> = {
    morning: [
      "Good morning! How are you feeling today?",
      "Hey there! How did you sleep last night?",
      "Morning! What's your energy level like today?",
      "Hi! Ready to start the day? How are you doing?",
    ],
    midday: [
      "How's your day going so far?",
      "Checking in! How are you feeling this afternoon?",
      "Hey! How's your energy holding up?",
      "Hi there! How are things going today?",
    ],
    evening: [
      "How was your day today?",
      "Evening check-in! How are you feeling?",
      "Hi! How did today go for you?",
      "Hey! Looking back on your day, how are you doing?",
    ],
    medication: [
      "Did you take your medications today?",
      "Quick check - how are you doing with your medications?",
      "Have you had a chance to take your meds today?",
    ],
    mood: [
      "How are you feeling emotionally today?",
      "What's your mood like right now?",
      "How's your mental health today?",
    ],
  };

  const options = prompts[checkInType] || prompts.morning;
  return options[Math.floor(Math.random() * options.length)];
}
