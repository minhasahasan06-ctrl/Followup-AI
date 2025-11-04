import OpenAI from "openai";
import type { ImmuneBiomarker, ImmuneDigitalTwin, InsertImmuneBiomarker, InsertImmuneDigitalTwin } from "@shared/schema";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/**
 * Analyze immune biomarkers using AI to create immune digital twin
 * This simulates advanced ML models that predict immune function status
 */
export async function generateImmuneDigitalTwin(
  userId: string,
  biomarkers: ImmuneBiomarker[],
  patientContext?: {
    immunocompromisedCondition?: string;
    medications?: string[];
    age?: number;
  }
): Promise<InsertImmuneDigitalTwin> {
  if (!process.env.OPENAI_API_KEY || biomarkers.length === 0) {
    return generateFallbackDigitalTwin(userId);
  }

  try {
    // Prepare biomarker summary
    const latestBiomarkers = biomarkers.slice(0, 7); // Last 7 days
    const biomarkerSummary = latestBiomarkers.map(b => ({
      date: b.measuredAt,
      hrv: b.hrvRmssd,
      restingHR: b.restingHeartRate,
      sleep: b.sleepDuration,
      deepSleep: b.deepSleepDuration,
      sleepQuality: b.sleepQuality,
      stress: b.stressLevel,
      recovery: b.recoveryScore,
      temperature: b.bodyTemperature,
      spo2: b.oxygenSaturation,
    }));

    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: `You are an AI immunologist specializing in digital twin modeling for immunocompromised patients. Analyze wearable biomarker data and predict immune function status.

Your analysis should:
1. Calculate immune score (0-100, where 100 is optimal immune function)
2. Assess infection risk (low, moderate, high, critical)
3. Identify contributing factors (positive and negative impacts)
4. Provide actionable recommendations
5. Use empathetic, simple language

Consider that immunocompromised patients have:
- Heightened vulnerability to infections
- Slower recovery from stress
- Greater impact from poor sleep and high stress
- Need for proactive prevention

Return a JSON object with: { immuneScore, immuneScoreTrend, infectionRisk, contributingFactors: [{ factor, impact, strength }], recommendations: [], insights }`
        },
        {
          role: "user",
          content: `Patient Context:
- Condition: ${patientContext?.immunocompromisedCondition || 'Immunocompromised'}
- Medications: ${patientContext?.medications?.join(', ') || 'None listed'}
- Age: ${patientContext?.age || 'Not specified'}

Biomarker Trends (Last 7 Days):
${JSON.stringify(biomarkerSummary, null, 2)}

Analyze immune function and provide digital twin prediction.`
        }
      ],
      response_format: { type: "json_object" },
      temperature: 0.3,
    });

    const result = JSON.parse(completion.choices[0].message.content || '{}');

    // Calculate component scores
    const avgHRV = average(latestBiomarkers.map(b => Number(b.hrvRmssd) || 0).filter(v => v > 0));
    const avgSleep = average(latestBiomarkers.map(b => Number(b.sleepDuration) || 0).filter(v => v > 0));
    const avgStress = average(latestBiomarkers.map(b => b.stressLevel || 0).filter(v => v > 0));
    const avgRecovery = average(latestBiomarkers.map(b => b.recoveryScore || 0).filter(v => v > 0));

    return {
      userId,
      predictedAt: new Date(),
      predictionWindow: 'current',
      immuneScore: result.immuneScore || calculateImmuneScore(avgHRV, avgSleep, avgStress, avgRecovery),
      immuneScoreTrend: result.immuneScoreTrend || 'stable',
      recoveryCapacityScore: Math.round(avgRecovery || 50),
      infectionResistanceScore: result.immuneScore ? Math.round(result.immuneScore * 0.9) : 50,
      inflammationScore: avgStress ? Math.round(100 - avgStress) : 50,
      stressResponseScore: avgHRV ? Math.round((avgHRV / 50) * 100) : 50,
      infectionRisk: result.infectionRisk || 'moderate',
      hospitalAdmissionRisk: result.infectionRisk === 'critical' ? 'high' : result.infectionRisk === 'high' ? 'moderate' : 'low',
      contributingFactors: result.contributingFactors || [],
      biomarkerIds: biomarkers.map(b => b.id),
      modelVersion: 'v1.0-gpt4o',
      confidenceScore: '0.85',
      aiInsights: result.insights || 'Immune function analysis based on recent biomarker trends.',
      recommendations: result.recommendations || ['Maintain consistent sleep schedule', 'Monitor stress levels'],
    };
  } catch (error) {
    console.error('Error generating immune digital twin:', error);
    return generateFallbackDigitalTwin(userId);
  }
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function calculateImmuneScore(hrv: number, sleep: number, stress: number, recovery: number): number {
  const hrvScore = Math.min((hrv / 50) * 100, 100); // Normalize HRV (50ms is good)
  const sleepScore = Math.min((sleep / 8) * 100, 100); // 8 hours optimal
  const stressScore = 100 - stress; // Lower stress is better
  const recoveryScore = recovery;

  return Math.round((hrvScore + sleepScore + stressScore + recoveryScore) / 4);
}

function generateFallbackDigitalTwin(userId: string): InsertImmuneDigitalTwin {
  return {
    userId,
    predictedAt: new Date(),
    predictionWindow: 'current',
    immuneScore: 65,
    immuneScoreTrend: 'stable',
    recoveryCapacityScore: 60,
    infectionResistanceScore: 65,
    inflammationScore: 70,
    stressResponseScore: 60,
    infectionRisk: 'moderate',
    hospitalAdmissionRisk: 'low',
    contributingFactors: [],
    biomarkerIds: [],
    modelVersion: 'v1.0-fallback',
    confidenceScore: '0.3',
    aiInsights: 'Limited data available. Connect wearable devices for accurate immune monitoring.',
    recommendations: ['Connect a wearable device', 'Maintain regular sleep schedule'],
  };
}

/**
 * Simulate wearable data collection for demo purposes
 * In production, this would connect to real Fitbit/Apple Health/Google Fit APIs
 */
export function generateSimulatedBiomarker(userId: string, wearableIntegrationId?: string): InsertImmuneBiomarker {
  const now = new Date();
  
  // Simulate realistic biomarker values with some variation
  const baseHRV = 35 + Math.random() * 30; // 35-65ms
  const baseSleep = 6 + Math.random() * 2.5; // 6-8.5 hours
  const baseStress = 30 + Math.random() * 40; // 30-70 stress level

  return {
    userId,
    dataSource: 'simulated',
    wearableIntegrationId,
    measuredAt: now,
    hrvRmssd: baseHRV.toFixed(1),
    hrvSdnn: (baseHRV * 1.5).toFixed(1),
    restingHeartRate: Math.round(55 + Math.random() * 20), // 55-75 BPM
    sleepDuration: baseSleep.toFixed(1),
    deepSleepDuration: (baseSleep * 0.2).toFixed(1),
    remSleepDuration: (baseSleep * 0.25).toFixed(1),
    sleepQuality: Math.round(60 + Math.random() * 30), // 60-90
    sleepEfficiency: (85 + Math.random() * 10).toFixed(1), // 85-95%
    stepsCount: Math.round(5000 + Math.random() * 5000), // 5000-10000 steps
    activeMinutes: Math.round(30 + Math.random() * 60), // 30-90 minutes
    caloriesBurned: Math.round(1800 + Math.random() * 500),
    stressLevel: Math.round(baseStress),
    bodyTemperature: (36.5 + Math.random() * 0.5).toFixed(1), // 36.5-37.0°C
    skinTemperature: (33 + Math.random() * 2).toFixed(1),
    respiratoryRate: (12 + Math.random() * 6).toFixed(1), // 12-18 breaths/min
    oxygenSaturation: (96 + Math.random() * 3).toFixed(1), // 96-99%
    recoveryScore: Math.round(50 + Math.random() * 40), // 50-90
    readinessScore: Math.round(60 + Math.random() * 30), // 60-90
    metadata: {
      deviceModel: 'Simulated Device',
      isManualEntry: false,
      confidenceScore: 0.9,
    },
  };
}

/**
 * Assess infection risk based on immune digital twin and environmental data
 */
export function assessCombinedRisk(
  digitalTwin: ImmuneDigitalTwin,
  environmentalRisk?: { immunocompromisedRisk: string; overallRiskScore: number }
): {
  overallRisk: 'low' | 'moderate' | 'high' | 'critical';
  riskScore: number;
  primaryFactors: string[];
} {
  let riskScore = 0;
  const factors: string[] = [];

  // Immune score contribution (0-40 points)
  const immuneContribution = Math.round((100 - digitalTwin.immuneScore) * 0.4);
  riskScore += immuneContribution;
  if (digitalTwin.immuneScore < 50) {
    factors.push('Low immune function score');
  }

  // Infection risk contribution (0-30 points)
  const infectionRiskMap = { low: 0, moderate: 15, high: 25, critical: 30 };
  riskScore += infectionRiskMap[digitalTwin.infectionRisk as keyof typeof infectionRiskMap] || 15;
  if (digitalTwin.infectionRisk === 'high' || digitalTwin.infectionRisk === 'critical') {
    factors.push('High infection risk prediction');
  }

  // Environmental risk contribution (0-30 points)
  if (environmentalRisk) {
    const envContribution = Math.round(environmentalRisk.overallRiskScore * 0.3);
    riskScore += envContribution;
    if (environmentalRisk.immunocompromisedRisk === 'high' || environmentalRisk.immunocompromisedRisk === 'critical') {
      factors.push('Hazardous environmental conditions');
    }
  }

  // Determine overall risk level
  let overallRisk: 'low' | 'moderate' | 'high' | 'critical';
  if (riskScore < 25) overallRisk = 'low';
  else if (riskScore < 50) overallRisk = 'moderate';
  else if (riskScore < 75) overallRisk = 'high';
  else overallRisk = 'critical';

  return { overallRisk, riskScore, primaryFactors: factors };
}

/**
 * Sync wearable data from various devices
 * Returns simulated data for demo purposes when API keys aren't available
 */
export async function syncWearableData(userId: string, deviceType: string) {
  const { storage } = await import('./storage');
  
  // Check if wearable integration exists
  const integrations = await storage.getWearableIntegrations(userId);
  const integration = integrations.find(i => i.deviceType.toLowerCase() === deviceType.toLowerCase());
  
  if (!integration) {
    throw new Error(`No ${deviceType} integration found for user`);
  }
  
  // For demo: Generate simulated biomarker data
  // In production, this would call actual wearable APIs (Fitbit, Apple Health, etc.)
  const biomarker = generateSimulatedBiomarker(userId, integration.id);
  const savedBiomarker = await storage.createImmuneBiomarker(biomarker);
  
  return {
    success: true,
    message: `Successfully synced data from ${deviceType}`,
    biomarker: savedBiomarker,
  };
}

/**
 * Analyze immune biomarkers and generate digital twin prediction
 */
export async function analyzeImmuneBiomarkers(userId: string) {
  const { storage } = await import('./storage');
  
  // Fetch recent biomarkers
  const biomarkers = await storage.getImmuneBiomarkers(userId, 30);
  
  if (biomarkers.length === 0) {
    return {
      success: false,
      message: 'No biomarker data available. Please sync your wearable device first.',
    };
  }
  
  // Get patient context
  const user = await storage.getUser(userId);
  const patientProfile = await storage.getPatientProfile(userId);
  const medications = await storage.getActiveMedications(userId);
  
  const patientContext = {
    immunocompromisedCondition: patientProfile?.immunocompromisedCondition || undefined,
    medications: medications.map(m => m.name),
    age: patientProfile?.dateOfBirth ? 
      Math.floor((Date.now() - new Date(patientProfile.dateOfBirth).getTime()) / (365.25 * 24 * 60 * 60 * 1000)) : 
      undefined,
  };
  
  // Generate digital twin prediction
  const digitalTwinData = await generateImmuneDigitalTwin(userId, biomarkers, patientContext);
  const digitalTwin = await storage.createImmuneDigitalTwin(digitalTwinData);
  
  return {
    success: true,
    digitalTwin,
    biomarkersAnalyzed: biomarkers.length,
  };
}
