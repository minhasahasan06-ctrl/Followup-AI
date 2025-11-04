import OpenAI from "openai";
import { storage } from "./storage";
import type { InsertCorrelationPattern } from "@shared/schema";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface HealthDataPoint {
  type: 'medication' | 'environment' | 'mood' | 'sleep' | 'biomarker' | 'symptom';
  name: string;
  value: any;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export async function analyzeCorrelations(userId: string): Promise<void> {
  try {
    // 1. Gather all relevant health data for the past 30 days
    const healthData = await gatherHealthData(userId);
    
    if (healthData.length < 10) {
      console.log(`Not enough data for correlation analysis (${healthData.length} points)`);
      return;
    }

    // 2. Use AI to detect patterns and correlations
    const patterns = await detectPatternsWithAI(healthData, userId);
    
    // 3. Save detected patterns to database
    for (const pattern of patterns) {
      await storage.createCorrelationPattern(pattern);
    }
    
    console.log(`Created ${patterns.length} correlation patterns for user ${userId}`);
  } catch (error) {
    console.error("Error analyzing correlations:", error);
    throw error;
  }
}

async function gatherHealthData(userId: string): Promise<HealthDataPoint[]> {
  const data: HealthDataPoint[] = [];
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

  // Get medications
  const medications = await storage.getActiveMedications(userId);
  for (const med of medications) {
    data.push({
      type: 'medication',
      name: med.medicationName,
      value: { dosage: med.dosage, frequency: med.frequency },
      timestamp: med.startDate || new Date(),
      metadata: { medicationId: med.id },
    });
  }

  // Get immune biomarkers (last 30 days)
  const biomarkers = await storage.getImmuneBiomarkers(userId, 50);
  for (const biomarker of biomarkers) {
    if (biomarker.recordedAt && biomarker.recordedAt >= thirtyDaysAgo) {
      if (biomarker.hrv !== null && biomarker.hrv !== undefined) {
        data.push({
          type: 'biomarker',
          name: 'HRV',
          value: biomarker.hrv,
          timestamp: biomarker.recordedAt,
        });
      }
      if (biomarker.sleepQuality !== null && biomarker.sleepQuality !== undefined) {
        data.push({
          type: 'sleep',
          name: 'Sleep Quality',
          value: biomarker.sleepQuality,
          timestamp: biomarker.recordedAt,
        });
      }
      if (biomarker.restingHeartRate !== null && biomarker.restingHeartRate !== undefined) {
        data.push({
          type: 'biomarker',
          name: 'Resting Heart Rate',
          value: biomarker.restingHeartRate,
          timestamp: biomarker.recordedAt,
        });
      }
      if (biomarker.stressLevel !== null && biomarker.stressLevel !== undefined) {
        data.push({
          type: 'mood',
          name: 'Stress Level',
          value: biomarker.stressLevel,
          timestamp: biomarker.recordedAt,
        });
      }
    }
  }

  // Get immune digital twin data (predictions and immune scores)
  const digitalTwins = await storage.getImmuneDigitalTwins(userId, 30);
  for (const twin of digitalTwins) {
    if (twin.analysisDate && twin.analysisDate >= thirtyDaysAgo) {
      if (twin.immuneScore !== null && twin.immuneScore !== undefined) {
        data.push({
          type: 'biomarker',
          name: 'Immune Score',
          value: twin.immuneScore,
          timestamp: twin.analysisDate,
        });
      }
    }
  }

  // Get environmental risk data
  const environmentalData = await storage.getEnvironmentalRiskDataByUser(userId, 30);
  for (const env of environmentalData) {
    if (env.measuredAt && env.measuredAt >= thirtyDaysAgo) {
      if (env.airQualityIndex !== null && env.airQualityIndex !== undefined) {
        data.push({
          type: 'environment',
          name: 'Air Quality Index',
          value: env.airQualityIndex,
          timestamp: env.measuredAt,
        });
      }
      if (env.pathogenRisk) {
        data.push({
          type: 'environment',
          name: 'Pathogen Risk',
          value: env.pathogenRisk,
          timestamp: env.measuredAt,
        });
      }
    }
  }

  // Get daily followups for mood and symptoms
  const followups = await storage.getRecentFollowups(userId, 30);
  for (const followup of followups) {
    if (followup.date && followup.date >= thirtyDaysAgo) {
      if (followup.mood) {
        data.push({
          type: 'mood',
          name: 'Daily Mood',
          value: followup.mood,
          timestamp: followup.date,
        });
      }
      if (followup.symptoms) {
        data.push({
          type: 'symptom',
          name: 'Symptoms',
          value: followup.symptoms,
          timestamp: followup.date,
        });
      }
    }
  }

  return data.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
}

async function detectPatternsWithAI(
  healthData: HealthDataPoint[],
  userId: string
): Promise<InsertCorrelationPattern[]> {
  // Prepare data summary for AI
  const dataSummary = {
    totalDataPoints: healthData.length,
    dateRange: {
      start: healthData[0]?.timestamp,
      end: healthData[healthData.length - 1]?.timestamp,
    },
    dataByType: {
      medication: healthData.filter(d => d.type === 'medication').length,
      environment: healthData.filter(d => d.type === 'environment').length,
      mood: healthData.filter(d => d.type === 'mood').length,
      sleep: healthData.filter(d => d.type === 'sleep').length,
      biomarker: healthData.filter(d => d.type === 'biomarker').length,
      symptom: healthData.filter(d => d.type === 'symptom').length,
    },
    sampleData: healthData.slice(-50).map(d => ({
      type: d.type,
      name: d.name,
      value: d.value,
      date: d.timestamp.toISOString().split('T')[0],
    })),
  };

  const systemPrompt = `You are an advanced health data correlation analyst for immunocompromised patients. Your task is to identify meaningful patterns and correlations across multiple health data types: medications, environmental factors, mood, sleep quality, biomarkers, and symptoms.

For each pattern you identify:
1. Look for temporal relationships (does X precede Y?)
2. Calculate correlation strength (-1.0 to 1.0)
3. Assess confidence level (0.0 to 1.0)
4. Determine pattern type (positive, negative, neutral)
5. Assess severity (low, moderate, high, critical)
6. Provide actionable insights and recommendations

Focus on patterns that are clinically meaningful for immunocompromised patients, such as:
- Medication effects on immune biomarkers
- Environmental factors impact on symptoms
- Sleep quality correlation with immune function
- Stress levels affecting biomarkers
- Multi-factor chains (e.g., poor sleep → high stress → low immune score → symptoms)`;

  const userPrompt = `Analyze the following health data and identify up to 5 most significant correlation patterns:

Data Summary:
${JSON.stringify(dataSummary, null, 2)}

Return a JSON array of patterns with this structure:
[
  {
    "patternName": "Clear, descriptive name (e.g., 'High stress → Poor sleep → Immune decline')",
    "patternType": "positive" | "negative" | "neutral",
    "factors": [
      {"type": "medication|environment|mood|sleep|biomarker|symptom", "name": "Factor name", "value": value}
    ],
    "correlationStrength": -1.0 to 1.0,
    "confidence": 0.0 to 1.0,
    "sampleSize": number,
    "timeWindow": "7 days" | "14 days" | "30 days",
    "insight": "Human-readable explanation of the pattern",
    "recommendation": "Actionable advice based on this pattern",
    "severity": "low" | "moderate" | "high" | "critical"
  }
]`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      response_format: { type: "json_object" },
      temperature: 0.3,
    });

    const response = completion.choices[0]?.message?.content;
    if (!response) {
      throw new Error("No response from OpenAI");
    }

    const parsed = JSON.parse(response);
    const patterns = parsed.patterns || [];

    // Convert AI response to database schema
    const now = new Date();
    return patterns.map((pattern: any) => ({
      userId,
      patternName: pattern.patternName,
      patternType: pattern.patternType,
      factors: pattern.factors,
      correlationStrength: pattern.correlationStrength?.toString() || "0",
      confidence: pattern.confidence?.toString() || "0",
      sampleSize: pattern.sampleSize || healthData.length,
      timeWindow: pattern.timeWindow || "30 days",
      firstObserved: healthData[0]?.timestamp || now,
      lastObserved: healthData[healthData.length - 1]?.timestamp || now,
      frequency: 1,
      insight: pattern.insight,
      recommendation: pattern.recommendation,
      severity: pattern.severity,
    }));
  } catch (error) {
    console.error("Error calling OpenAI for correlation analysis:", error);
    return [];
  }
}

export async function generateCorrelationReport(userId: string): Promise<{
  patterns: any[];
  summary: string;
  recommendations: string[];
}> {
  // Get all correlation patterns
  const patterns = await storage.getCorrelationPatterns(userId, 50);
  
  if (patterns.length === 0) {
    return {
      patterns: [],
      summary: "No correlation patterns detected yet. More data is needed for analysis.",
      recommendations: ["Continue tracking daily health metrics", "Maintain medication adherence", "Monitor environmental factors"],
    };
  }

  // Generate summary with AI
  const systemPrompt = `You are a medical data analyst summarizing health correlation patterns for an immunocompromised patient. Provide a clear, actionable summary.`;
  
  const userPrompt = `Summarize these health correlation patterns and provide top 3 actionable recommendations:

Patterns:
${JSON.stringify(patterns.slice(0, 10).map(p => ({
  name: p.patternName,
  type: p.patternType,
  strength: p.correlationStrength,
  severity: p.severity,
  insight: p.insight,
})), null, 2)}

Return JSON: { "summary": "2-3 sentence summary", "recommendations": ["rec1", "rec2", "rec3"] }`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      response_format: { type: "json_object" },
      temperature: 0.3,
    });

    const response = JSON.parse(completion.choices[0]?.message?.content || "{}");
    
    return {
      patterns,
      summary: response.summary || "Multiple health patterns detected.",
      recommendations: response.recommendations || [],
    };
  } catch (error) {
    console.error("Error generating correlation report:", error);
    return {
      patterns,
      summary: "Multiple correlation patterns detected across your health data.",
      recommendations: ["Review identified patterns", "Discuss findings with your doctor", "Continue monitoring"],
    };
  }
}
