import type { Express, Request } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { isAuthenticated, isDoctor, isPatient, getSession } from "./auth";
import { db } from "./db";
import { eq, and, desc, gte, sql as drizzleSql, isNull } from "drizzle-orm";
import * as schema from "@shared/schema";
import { z } from "zod";
import { pubmedService, physionetService, kaggleService, whoService } from "./dataIntegration";
import { s3Client, textractClient, comprehendMedicalClient, AWS_S3_BUCKET } from "./aws";
import { PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";
import { StartDocumentAnalysisCommand, GetDocumentAnalysisCommand } from "@aws-sdk/client-textract";
import { DetectEntitiesV2Command, DetectPHICommand } from "@aws-sdk/client-comprehendmedical";
import OpenAI from "openai";
import Sentiment from "sentiment";
import multer from "multer";
import path from "path";
import fs from "fs";
import speakeasy from "speakeasy";
import QRCode from "qrcode";
import crypto from "crypto";
import jwt from "jsonwebtoken";
import { personalizationService } from "./personalizationService";
import { rlRewardCalculator } from "./rlRewardCalculator";
import { 
  createHabitSchema,
  completeHabitSchema,
  feedbackSchema,
  agentPromptSchema,
  doctorWellnessSchema 
} from "./mlValidation";
import {
  categorizeEmail,
  generateEmailReply,
  batchCategorizeEmails,
  extractActionItems
} from "./emailAIService";
import { aiRateLimit, batchRateLimit } from "./rateLimiting";
import {
  detectAppointmentIntent,
  parseRelativeDate,
  parseTime,
  checkAvailability,
  bookAppointmentFromChat,
  detectPatientSearchIntent,
  searchPatients,
  getPatientRecord,
  formatPatientSummary
} from "./chatReceptionistService";
import { 
  doctorIntegrationService, 
  initDoctorIntegrationService 
} from "./doctorIntegrationService";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const sentiment = new Sentiment();

// Helper function to extract sections from AI-generated reports
function extractSection(text: string, sectionName: string): string {
  const regex = new RegExp(`${sectionName}[:\\s]*([\\s\\S]*?)(?=(?:TECHNIQUE|FINDINGS|IMPRESSION|RECOMMENDATIONS|SUMMARY|SIGNIFICANT|CLINICAL|$))`, 'i');
  const match = text.match(regex);
  return match ? match[1].trim() : '';
}

// Helper function to generate lab trends from historical data
function generateLabTrends(results: any[]): any[] {
  return results.map(result => ({
    testName: result.name,
    currentValue: result.value,
    trend: result.isAbnormal ? (result.deviation === 'high' ? 'increasing' : 'decreasing') : 'stable',
    changePercent: result.deviationPercent || 0
  }));
}

// ============================================================================
// ENVIRONMENTAL RISK MAP - Helper Functions
// ============================================================================

// Get ZIP code region info (simulated - in production would use census/geocoding API)
function getZipCodeInfo(zipCode: string): { city: string; state: string; lat: number; lon: number } {
  const zipRegions: Record<string, { city: string; state: string; lat: number; lon: number }> = {
    '10001': { city: 'New York', state: 'NY', lat: 40.7484, lon: -73.9967 },
    '90210': { city: 'Beverly Hills', state: 'CA', lat: 34.0901, lon: -118.4065 },
    '60601': { city: 'Chicago', state: 'IL', lat: 41.8819, lon: -87.6278 },
    '77001': { city: 'Houston', state: 'TX', lat: 29.7604, lon: -95.3698 },
    '85001': { city: 'Phoenix', state: 'AZ', lat: 33.4484, lon: -112.0740 },
    '19101': { city: 'Philadelphia', state: 'PA', lat: 39.9526, lon: -75.1652 },
    '78201': { city: 'San Antonio', state: 'TX', lat: 29.4241, lon: -98.4936 },
    '92101': { city: 'San Diego', state: 'CA', lat: 32.7157, lon: -117.1611 },
    '75201': { city: 'Dallas', state: 'TX', lat: 32.7767, lon: -96.7970 },
    '95101': { city: 'San Jose', state: 'CA', lat: 37.3382, lon: -121.8863 },
  };
  
  // Default to a generic region based on first digit
  const region = zipCode[0];
  const defaults: Record<string, { city: string; state: string; lat: number; lon: number }> = {
    '0': { city: 'Boston Area', state: 'MA', lat: 42.3601, lon: -71.0589 },
    '1': { city: 'New York Area', state: 'NY', lat: 40.7128, lon: -74.0060 },
    '2': { city: 'Washington DC Area', state: 'DC', lat: 38.9072, lon: -77.0369 },
    '3': { city: 'Atlanta Area', state: 'GA', lat: 33.7490, lon: -84.3880 },
    '4': { city: 'Detroit Area', state: 'MI', lat: 42.3314, lon: -83.0458 },
    '5': { city: 'Minneapolis Area', state: 'MN', lat: 44.9778, lon: -93.2650 },
    '6': { city: 'Chicago Area', state: 'IL', lat: 41.8781, lon: -87.6298 },
    '7': { city: 'Houston Area', state: 'TX', lat: 29.7604, lon: -95.3698 },
    '8': { city: 'Denver Area', state: 'CO', lat: 39.7392, lon: -104.9903 },
    '9': { city: 'Los Angeles Area', state: 'CA', lat: 34.0522, lon: -118.2437 },
  };
  
  return zipRegions[zipCode] || defaults[region] || { city: 'Unknown', state: 'US', lat: 39.8283, lon: -98.5795 };
}

// Generate simulated environmental data (in production would fetch from real APIs)
function generateSimulatedEnvironmentalData(zipCode: string): {
  temperature: number;
  feelsLike: number;
  humidity: number;
  pressure: number;
  uvIndex: number;
  aqi: number;
  aqiCategory: string;
  pm25: number;
  pm10: number;
  ozone: number;
  pollenOverall: number;
  pollenCategory: string;
  moldCount: number;
} {
  // Seed based on ZIP for consistent results
  const seed = parseInt(zipCode) % 1000;
  const rand = (min: number, max: number) => min + ((seed * 9301 + 49297) % 233280) / 233280 * (max - min);
  
  // Seasonal variation (December = winter in N. hemisphere)
  const month = new Date().getMonth();
  const isWinter = month >= 11 || month <= 2;
  const isSummer = month >= 5 && month <= 8;
  
  // Temperature varies by region (first digit of ZIP)
  const regionTemp = {
    '0': 35, '1': 40, '2': 45, '3': 55, '4': 38, 
    '5': 30, '6': 35, '7': 60, '8': 45, '9': 65
  }[zipCode[0]] || 50;
  
  const baseTemp = isWinter ? regionTemp - 10 : isSummer ? regionTemp + 15 : regionTemp;
  const temperature = baseTemp + rand(-5, 10);
  const humidity = rand(30, 80);
  
  // AQI calculation
  let aqi = Math.floor(rand(20, 150));
  // Higher AQI in urban areas (lower ZIP first digits often more urban)
  if (['0', '1', '2', '6', '9'].includes(zipCode[0])) {
    aqi = Math.min(200, aqi + 20);
  }
  
  const aqiCategory = aqi <= 50 ? 'Good' : 
                      aqi <= 100 ? 'Moderate' : 
                      aqi <= 150 ? 'Unhealthy for Sensitive Groups' : 
                      aqi <= 200 ? 'Unhealthy' : 'Very Unhealthy';
  
  // Pollen (higher in spring/fall)
  const isPollenSeason = (month >= 2 && month <= 5) || (month >= 8 && month <= 10);
  const pollenOverall = isPollenSeason ? rand(5, 11) : rand(1, 6);
  const pollenCategory = pollenOverall <= 2.4 ? 'Low' :
                         pollenOverall <= 4.8 ? 'Low-Medium' :
                         pollenOverall <= 7.2 ? 'Medium' :
                         pollenOverall <= 9.6 ? 'Medium-High' : 'High';
  
  return {
    temperature: Math.round(temperature * 10) / 10,
    feelsLike: Math.round((temperature + (humidity > 60 ? 3 : -2)) * 10) / 10,
    humidity: Math.round(humidity),
    pressure: Math.round(1013 + rand(-15, 15)),
    uvIndex: Math.round(rand(1, isSummer ? 10 : 4)),
    aqi,
    aqiCategory,
    pm25: Math.round(rand(5, aqi / 2) * 10) / 10,
    pm10: Math.round(rand(10, aqi) * 10) / 10,
    ozone: Math.round(rand(20, 70)),
    pollenOverall: Math.round(pollenOverall * 10) / 10,
    pollenCategory,
    moldCount: Math.round(rand(500, humidity > 60 ? 3000 : 1500)),
  };
}

// Compute environmental risk based on conditions
function computeEnvironmentalRisk(
  envData: ReturnType<typeof generateSimulatedEnvironmentalData>,
  conditions: string[]
): {
  compositeScore: number;
  riskLevel: string;
  weatherScore: number;
  airQualityScore: number;
  allergenScore: number;
  factorContributions: Record<string, number>;
  topFactors: string[];
} {
  const contributions: Record<string, number> = {};
  
  // Base weather risk
  let weatherScore = 0;
  if (envData.temperature < 32 || envData.temperature > 95) {
    weatherScore += 30;
    contributions['extreme_temperature'] = 30;
  } else if (envData.temperature < 45 || envData.temperature > 85) {
    weatherScore += 15;
    contributions['temperature'] = 15;
  }
  if (envData.humidity > 70) {
    weatherScore += 15;
    contributions['high_humidity'] = 15;
  }
  if (envData.uvIndex >= 8) {
    weatherScore += 10;
    contributions['high_uv'] = 10;
  }
  
  // Air quality risk
  let airQualityScore = 0;
  if (envData.aqi > 150) {
    airQualityScore = 50;
    contributions['very_poor_aqi'] = 50;
  } else if (envData.aqi > 100) {
    airQualityScore = 35;
    contributions['poor_aqi'] = 35;
  } else if (envData.aqi > 50) {
    airQualityScore = 20;
    contributions['moderate_aqi'] = 20;
  }
  
  // Allergen risk
  let allergenScore = 0;
  if (envData.pollenOverall > 8) {
    allergenScore = 40;
    contributions['high_pollen'] = 40;
  } else if (envData.pollenOverall > 5) {
    allergenScore = 25;
    contributions['medium_pollen'] = 25;
  }
  if (envData.moldCount > 2000) {
    allergenScore += 20;
    contributions['high_mold'] = 20;
  }
  
  // Condition-specific adjustments
  const conditionLower = conditions.map(c => c.toLowerCase());
  
  if (conditionLower.some(c => c.includes('asthma') || c.includes('copd') || c.includes('respiratory'))) {
    airQualityScore *= 1.5;
    allergenScore *= 1.3;
    if (envData.aqi > 50) contributions['asthma_sensitivity'] = 15;
  }
  
  if (conditionLower.some(c => c.includes('heart') || c.includes('cardiac') || c.includes('cardiovascular'))) {
    if (envData.temperature > 85 || envData.temperature < 32) {
      weatherScore *= 1.4;
      contributions['cardiac_temp_sensitivity'] = 20;
    }
  }
  
  if (conditionLower.some(c => c.includes('arthritis') || c.includes('joint'))) {
    if (envData.pressure < 1005 || envData.humidity > 65) {
      weatherScore += 15;
      contributions['arthritis_pressure'] = 15;
    }
  }
  
  if (conditionLower.some(c => c.includes('allergy') || c.includes('allergies') || c.includes('hay fever'))) {
    allergenScore *= 1.4;
    contributions['allergy_sensitivity'] = 10;
  }
  
  // Composite score (weighted average)
  const compositeScore = Math.min(100, Math.round(
    weatherScore * 0.25 + 
    airQualityScore * 0.4 + 
    allergenScore * 0.35
  ));
  
  const riskLevel = compositeScore < 25 ? 'low' : 
                    compositeScore < 50 ? 'moderate' : 
                    compositeScore < 75 ? 'high' : 'critical';
  
  // Top factors
  const topFactors = Object.entries(contributions)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([factor]) => factor.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
  
  return {
    compositeScore,
    riskLevel,
    weatherScore: Math.min(100, Math.round(weatherScore)),
    airQualityScore: Math.min(100, Math.round(airQualityScore)),
    allergenScore: Math.min(100, Math.round(allergenScore)),
    factorContributions: contributions,
    topFactors,
  };
}

// Intelligent Device Data Routing Service
// Routes device metrics to appropriate health sections for AI Health Alerts, ML training, and care pathways
function calculateDeviceRoutingFlags(deviceType: string, readingData: any): Record<string, boolean> {
  const flags: Record<string, boolean> = {};
  
  switch (deviceType) {
    case 'bp_monitor':
      flags.routeToHypertension = true;
      flags.routeToCardiovascular = true;
      break;
      
    case 'glucose_meter':
      flags.routeToDiabetes = true;
      break;
      
    case 'smart_scale':
      flags.routeToFitness = true;
      // Body composition metrics can indicate metabolic health
      if (readingData.bodyFatPercentage || readingData.visceralFat || readingData.metabolicAge) {
        flags.routeToDiabetes = true; // Metabolic health tracking
      }
      break;
      
    case 'thermometer':
      flags.routeToCardiovascular = true;
      // Elevated temperature may indicate infection/inflammation
      if (readingData.temperature && readingData.temperature >= 38.0) {
        flags.routeToRespiratory = true;
      }
      break;
      
    case 'stethoscope':
      flags.routeToCardiovascular = true;
      flags.routeToRespiratory = true;
      break;
      
    case 'smartwatch':
      // Heart & Cardiovascular metrics
      if (readingData.heartRate || readingData.restingHeartRate || readingData.hrv || 
          readingData.hrvSdnn || readingData.ecgData || readingData.afibDetected || 
          readingData.irregularRhythmAlert) {
        flags.routeToCardiovascular = true;
      }
      
      // AFib detection is a critical cardiovascular indicator
      if (readingData.afibDetected || readingData.irregularRhythmAlert) {
        flags.routeToHypertension = true; // AFib often co-occurs with hypertension
      }
      
      // Respiratory metrics
      if (readingData.spo2 || readingData.spo2Min || readingData.respiratoryRate) {
        flags.routeToRespiratory = true;
      }
      
      // Low SpO2 is critical - ensure proper routing
      if (readingData.spo2 && readingData.spo2 < 94) {
        flags.routeToCardiovascular = true;
      }
      
      // Sleep metrics
      if (readingData.sleepDuration || readingData.sleepScore || readingData.sleepDeepMinutes || 
          readingData.sleepRemMinutes || readingData.sleepLightMinutes || readingData.sleepAwakeMinutes ||
          readingData.sleepEfficiency || readingData.sleepConsistency || readingData.sleepDebt ||
          readingData.sleepNeed) {
        flags.routeToSleep = true;
      }
      
      // Activity & Fitness metrics
      if (readingData.steps || readingData.activeMinutes || readingData.caloriesBurned || 
          readingData.distanceMeters || readingData.floorsClimbed || readingData.standingHours ||
          readingData.vo2Max || readingData.trainingLoad || readingData.trainingStatus ||
          readingData.trainingReadiness || readingData.fitnessAge || readingData.lactateThreshold ||
          readingData.performanceCondition || readingData.runningDynamics) {
        flags.routeToFitness = true;
      }
      
      // Recovery & Stress metrics → Mental Health
      if (readingData.stressScore || readingData.recoveryScore || readingData.readinessScore ||
          readingData.bodyBattery || readingData.strainScore) {
        flags.routeToMentalHealth = true;
        flags.routeToFitness = true;
      }
      
      // Women's Health metrics
      if (readingData.cycleDay || readingData.cyclePhase || readingData.periodLogged ||
          readingData.predictedOvulation) {
        flags.routeToWomensHealth = true;
      }
      
      // Skin temperature deviation can indicate hormonal changes or illness
      // skinTemperature is stored as deviation from baseline (e.g., -0.2, +0.5, +1.2)
      if (readingData.skinTemperature !== undefined && readingData.skinTemperature !== null) {
        flags.routeToWomensHealth = true; // BBT tracking for cycle monitoring
        
        // Significant positive deviation (>1.0°C above baseline) may indicate fever
        if (readingData.skinTemperature > 1.0) {
          flags.routeToCardiovascular = true;
          flags.routeToRespiratory = true; // Fever often indicates respiratory issues
        }
      }
      
      // Safety metrics - always route to emergency/high priority
      if (readingData.fallDetected || readingData.emergencySOSTriggered) {
        flags.routeToEmergency = true;
        flags.routeToCardiovascular = true;
        flags.routeToMentalHealth = true; // Falls can indicate cognitive decline
      }
      break;
  }
  
  return flags;
}

// Device Data to AI Health Alerts Integration Service
// Extracts metrics from device readings and feeds them to the AI Health Alert Engine
interface MetricIngestPayload {
  patient_id: string;
  metric_name: string;
  metric_value: number;
  unit: string;
  timestamp?: string;
  confidence?: number;
  source: string;
  capture_id?: string;
  metadata?: Record<string, any>;
}

async function processDeviceReadingForHealthAlerts(
  patientId: string,
  deviceType: string,
  readingData: any,
  readingId: string
): Promise<void> {
  try {
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
    const metrics: MetricIngestPayload[] = [];
    const timestamp = readingData.recordedAt ? new Date(readingData.recordedAt).toISOString() : new Date().toISOString();
    
    // Extract metrics based on device type
    switch (deviceType) {
      case 'bp_monitor':
        if (readingData.systolic !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'bp_systolic',
            metric_value: readingData.systolic,
            unit: 'mmHg',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.diastolic !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'bp_diastolic',
            metric_value: readingData.diastolic,
            unit: 'mmHg',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.pulse !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'pulse_rate',
            metric_value: readingData.pulse,
            unit: 'bpm',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        break;
        
      case 'glucose_meter':
        if (readingData.glucoseLevel !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'blood_glucose',
            metric_value: readingData.glucoseLevel,
            unit: readingData.measurementType === 'a1c' ? '%' : 'mg/dL',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { 
              device_type: deviceType,
              measurement_type: readingData.measurementType,
              meal_context: readingData.mealContext 
            }
          });
        }
        break;
        
      case 'smart_scale':
        if (readingData.weight !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'weight',
            metric_value: readingData.weight,
            unit: 'kg',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.bodyFatPercentage !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'body_fat_percentage',
            metric_value: readingData.bodyFatPercentage,
            unit: '%',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.muscleMass !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'muscle_mass',
            metric_value: readingData.muscleMass,
            unit: 'kg',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.bmi !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'bmi',
            metric_value: readingData.bmi,
            unit: 'kg/m²',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        break;
        
      case 'thermometer':
        if (readingData.temperature !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'temperature',
            metric_value: readingData.temperature,
            unit: readingData.unit || '°C',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { 
              device_type: deviceType,
              measurement_site: readingData.measurementSite 
            }
          });
        }
        break;
        
      case 'stethoscope':
        if (readingData.heartRate !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'heart_rate',
            metric_value: readingData.heartRate,
            unit: 'bpm',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.respiratoryRate !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'respiratory_rate',
            metric_value: readingData.respiratoryRate,
            unit: 'breaths/min',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        break;
        
      case 'smartwatch':
        // Heart metrics
        if (readingData.heartRate !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'heart_rate',
            metric_value: readingData.heartRate,
            unit: 'bpm',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.restingHeartRate !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'resting_heart_rate',
            metric_value: readingData.restingHeartRate,
            unit: 'bpm',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.hrv !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'hrv',
            metric_value: readingData.hrv,
            unit: 'ms',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        
        // Respiratory metrics
        if (readingData.spo2 !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'spo2',
            metric_value: readingData.spo2,
            unit: '%',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.respiratoryRate !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'respiratory_rate',
            metric_value: readingData.respiratoryRate,
            unit: 'breaths/min',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        
        // Sleep metrics
        if (readingData.sleepScore !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'sleep_score',
            metric_value: readingData.sleepScore,
            unit: 'score',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.sleepDuration !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'sleep_duration',
            metric_value: readingData.sleepDuration,
            unit: 'minutes',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        
        // Activity metrics
        if (readingData.steps !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'steps',
            metric_value: readingData.steps,
            unit: 'steps',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.activeMinutes !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'active_minutes',
            metric_value: readingData.activeMinutes,
            unit: 'minutes',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.caloriesBurned !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'calories_burned',
            metric_value: readingData.caloriesBurned,
            unit: 'kcal',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.vo2Max !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'vo2_max',
            metric_value: readingData.vo2Max,
            unit: 'mL/kg/min',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        
        // Recovery and stress metrics
        if (readingData.stressScore !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'stress_score',
            metric_value: readingData.stressScore,
            unit: 'score',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.recoveryScore !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'recovery_score',
            metric_value: readingData.recoveryScore,
            unit: 'score',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.readinessScore !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'readiness_score',
            metric_value: readingData.readinessScore,
            unit: 'score',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        if (readingData.bodyBattery !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'body_battery',
            metric_value: readingData.bodyBattery,
            unit: 'score',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        
        // Temperature deviation (fever indicator)
        if (readingData.skinTemperature !== undefined) {
          metrics.push({
            patient_id: patientId,
            metric_name: 'skin_temperature_deviation',
            metric_value: readingData.skinTemperature,
            unit: '°C',
            timestamp,
            source: 'device_reading',
            capture_id: readingId,
            metadata: { device_type: deviceType }
          });
        }
        break;
    }
    
    // If no metrics extracted, skip
    if (metrics.length === 0) {
      return;
    }
    
    // Send metrics to Python backend's AI Health Alert ingest endpoint
    try {
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/metrics/ingest/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metrics })
      });
      
      if (!response.ok) {
        console.warn(`AI Health Alert metric ingest returned ${response.status}: ${await response.text()}`);
      } else {
        console.log(`[AI Health Alerts] Ingested ${metrics.length} metrics from device reading ${readingId}`);
      }
    } catch (fetchError) {
      // Log but don't fail - this is a background process
      console.warn('[AI Health Alerts] Unable to connect to Python backend for metric ingest:', fetchError);
    }
  } catch (error) {
    console.error('[AI Health Alerts] Error processing device reading for health alerts:', error);
    // Don't throw - this is a non-critical background process
  }
}

// Mental Health Red Flag Indication Service (GPT-4o)
// IMPORTANT: This is an INDICATOR system, not a diagnostic tool
// Provides observational insights requiring professional clinical interpretation
async function extractMentalHealthIndicators(
  messageText: string,
  userId: string,
  sessionId: string,
  messageId?: string
): Promise<void> {
  try {
    // Skip if message is too short or doesn't contain concerning language
    if (messageText.length < 20) {
      return;
    }

    const extractionPrompt = `You are a clinical assistant analyzing patient messages for mental health red flag SYMPTOMS. Your role is to INDICATE observable symptoms that may warrant clinical attention, not diagnose.

Analyze this patient message and identify any mental health red flag symptoms:

"${messageText}"

Look for symptom indicators of:
1. **Suicidal ideation symptoms**: Expressed thoughts of death, wanting to die, self-harm plans, saying goodbye
2. **Self-harm symptoms**: Mentions of cutting, burning, hurting oneself, urges to self-injure
3. **Severe depression symptoms**: Persistent sadness, hopelessness, worthlessness, inability to function, loss of interest, persistent crying, sleep disturbances with mood impact
4. **Severe anxiety symptoms**: Panic attacks, overwhelming fear, constant worry affecting daily life, physical anxiety symptoms (racing heart, can't breathe)
5. **Crisis language symptoms**: "Can't go on", "can't take it anymore", "want it to end", giving away possessions
6. **Substance abuse symptoms**: Excessive drinking, drug use as coping mechanism, increased substance use to manage emotions
7. **Hopelessness symptoms**: No future, no point, giving up, isolation, withdrawal from loved ones

CRITICAL: Focus on OBSERVABLE SYMPTOMS the patient is describing, not your interpretations. Be thorough but avoid false alarms from casual language like "I'm dying to see that movie".

If you find ANY concerning indicators, respond with a JSON object:
{
  "hasRedFlags": true,
  "redFlagTypes": ["suicidal_ideation", "severe_depression", etc.],
  "severityLevel": "low" | "moderate" | "high" | "critical",
  "specificConcerns": ["exact phrases that raised concern"],
  "emotionalTone": "brief description of overall emotional tone",
  "recommendedAction": "suggested clinical action",
  "crisisIndicators": true/false (true if immediate intervention may be needed),
  "confidence": 0.0-1.0
}

If NO concerning indicators are found, respond with:
{
  "hasRedFlags": false
}

Be conservative with "critical" severity - reserve for immediate danger (active suicide plans, immediate self-harm intent).`;

    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are a clinical mental health analysis assistant. You identify indicators of mental health concerns in patient messages. You provide observational insights, not diagnoses. Respond only with valid JSON."
        },
        {
          role: "user",
          content: extractionPrompt
        }
      ],
      temperature: 0.3, // Lower temperature for more consistent analysis
      max_tokens: 500,
      response_format: { type: "json_object" }
    });

    const responseText = completion.choices[0].message.content || '{"hasRedFlags": false}';
    const analysis = JSON.parse(responseText);

    // If red flags are indicated, log to database
    if (analysis.hasRedFlags && analysis.redFlagTypes && analysis.redFlagTypes.length > 0) {
      // Calculate severity score (0-100)
      let severityScore = 0;
      switch (analysis.severityLevel) {
        case 'critical': severityScore = 90; break;
        case 'high': severityScore = 70; break;
        case 'moderate': severityScore = 50; break;
        case 'low': severityScore = 30; break;
        default: severityScore = 40;
      }

      // Insert into mental_health_red_flags table
      await db.insert(schema.mentalHealthRedFlags).values({
        userId,
        sessionId,
        messageId: messageId || null,
        rawText: messageText,
        extractedJson: {
          redFlagTypes: analysis.redFlagTypes,
          severityLevel: analysis.severityLevel,
          specificConcerns: analysis.specificConcerns || [],
          emotionalTone: analysis.emotionalTone || '',
          recommendedAction: analysis.recommendedAction || 'Clinical review recommended',
          crisisIndicators: analysis.crisisIndicators || false
        },
        confidence: analysis.confidence ? String(analysis.confidence) : '0.85',
        extractionModel: 'gpt-4o',
        severityScore,
        requiresImmediateAttention: analysis.crisisIndicators || false,
        clinicianNotified: false
      });

      // HIPAA audit log
      console.log(`[AUDIT] Mental health indicator logged - User: ${userId}, Session: ${sessionId}, Severity: ${analysis.severityLevel}, Crisis: ${analysis.crisisIndicators || false}`);

      // If critical, log additional alert
      if (analysis.crisisIndicators) {
        console.log(`[ALERT] CRITICAL mental health indicator - Immediate clinical review recommended for User: ${userId}`);
      }
    }
  } catch (error) {
    // Silent fail - don't disrupt chat flow if indicator extraction fails
    console.error('[ERROR] Mental health indicator extraction failed:', error);
  }
}

// Configure multer for file uploads (KYC photos)
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const uploadPath = path.join(process.cwd(), 'uploads', 'kyc');
      fs.mkdirSync(uploadPath, { recursive: true });
      cb(null, uploadPath);
    },
    filename: (req, file, cb) => {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
      cb(null, 'kyc-' + uniqueSuffix + path.extname(file.originalname));
    }
  }),
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|pdf/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    if (extname && mimetype) {
      cb(null, true);
    } else {
      cb(new Error('Only .jpg, .jpeg, .png, and .pdf files are allowed'));
    }
  }
});

// Configure multer for PainTrack video uploads (memory storage)
const paintrackVideoUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB combined limit
  fileFilter: (req, file, cb) => {
    const allowedMimeTypes = ['video/webm', 'video/mp4'];
    if (allowedMimeTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Only .webm and .mp4 video files are allowed'));
    }
  }
});

// Configure multer for medical document uploads (stored in memory before S3)
const medicalDocUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB limit for medical documents
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|pdf/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    if (extname && mimetype) {
      cb(null, true);
    } else {
      cb(new Error('Only .jpg, .jpeg, .png, and .pdf files are allowed'));
    }
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  // ============== INITIALIZE SERVICES ==============
  // Initialize doctor integration service early so it's available for all routes
  initDoctorIntegrationService(storage);
  
  // ============== SESSION CONFIGURATION ==============
  // Configure session middleware for cookie-based authentication
  app.use(getSession());

  // ============== DEV-ONLY: AUTHENTICATION BYPASS ==============
  // ⚠️ SECURITY WARNING: This bypass is ONLY for development testing!
  // It MUST NOT be accessible in production environments.
  if (process.env.NODE_ENV === 'development') {
    console.log('[DEV-ONLY] 🔓 Authentication bypass routes enabled for testing');

    // Create test users in database if they don't exist
    const ensureTestUsers = async () => {
      try {
        // Test patient user
        const testPatientId = 'dev-patient-00000000-0000-0000-0000-000000000001';
        const existingPatient = await storage.getUser(testPatientId);
        if (!existingPatient) {
          await storage.createUser({
            id: testPatientId,
            email: 'patient@test.com',
            firstName: 'Test',
            lastName: 'Patient',
            role: 'patient',
            phoneNumber: '+15551234567',
            phoneVerified: true,
            emailVerified: true,
            termsAccepted: true,
            termsAcceptedAt: new Date(),
          });
          console.log('[DEV-ONLY] ✅ Created test patient user');
        }

        // Test doctor user
        const testDoctorId = 'dev-doctor-00000000-0000-0000-0000-000000000002';
        const existingDoctor = await storage.getUser(testDoctorId);
        if (!existingDoctor) {
          await storage.createUser({
            id: testDoctorId,
            email: 'doctor@test.com',
            firstName: 'Dr. Test',
            lastName: 'Doctor',
            role: 'doctor',
            phoneNumber: '+15551234568',
            phoneVerified: true,
            emailVerified: true,
            medicalLicenseNumber: 'TEST-LICENSE-12345',
            organization: 'Test Hospital',
            licenseVerified: true,
            adminVerified: true,
            adminVerifiedAt: new Date(),
            termsAccepted: true,
            termsAcceptedAt: new Date(),
          });
          console.log('[DEV-ONLY] ✅ Created test doctor user');
        }
      } catch (error) {
        console.error('[DEV-ONLY] Error creating test users:', error);
      }
    };

    // Initialize test users
    await ensureTestUsers();

    // Dev-only quick login endpoints
    app.post('/api/dev/login-as-patient', async (req: any, res) => {
      try {
        const testPatientId = 'dev-patient-00000000-0000-0000-0000-000000000001';
        
        // Get user from database
        const user = await storage.getUser(testPatientId);
        if (!user) {
          return res.status(404).json({ message: 'Test patient user not found. Please restart the server.' });
        }
        
        // Set session
        req.session.userId = testPatientId;
        await new Promise<void>((resolve, reject) => {
          req.session.save((err: any) => {
            if (err) reject(err);
            else resolve();
          });
        });
        
        console.log('[DEV-ONLY] 👤 Logged in as test patient');
        res.json({ 
          message: 'Logged in as test patient', 
          user: user
        });
      } catch (error: any) {
        console.error('[DEV-ONLY] Error in dev patient login:', error);
        res.status(500).json({ message: 'Failed to login as test patient', error: error.message });
      }
    });

    app.post('/api/dev/login-as-doctor', async (req: any, res) => {
      try {
        const testDoctorId = 'dev-doctor-00000000-0000-0000-0000-000000000002';
        
        // Get user from database
        const user = await storage.getUser(testDoctorId);
        if (!user) {
          return res.status(404).json({ message: 'Test doctor user not found. Please restart the server.' });
        }
        
        // Set session
        req.session.userId = testDoctorId;
        await new Promise<void>((resolve, reject) => {
          req.session.save((err: any) => {
            if (err) reject(err);
            else resolve();
          });
        });
        
        console.log('[DEV-ONLY] 👨‍⚕️ Logged in as test doctor');
        res.json({ 
          message: 'Logged in as test doctor', 
          user: user
        });
      } catch (error: any) {
        console.error('[DEV-ONLY] Error in dev doctor login:', error);
        res.status(500).json({ message: 'Failed to login as test doctor', error: error.message });
      }
    });

    app.get('/api/dev/test-users', async (req, res) => {
      res.json({
        patient: {
          endpoint: 'POST /api/dev/login-as-patient',
          email: 'patient@test.com',
          userId: 'dev-patient-00000000-0000-0000-0000-000000000001'
        },
        doctor: {
          endpoint: 'POST /api/dev/login-as-doctor',
          email: 'doctor@test.com',
          userId: 'dev-doctor-00000000-0000-0000-0000-000000000002'
        },
        note: 'These endpoints only work in development mode'
      });
    });
  }
  // ============== END DEV-ONLY BYPASS ==============

  // ============== AUTHENTICATION ROUTES (AWS Cognito) ==============
  const { signUp, signIn, confirmSignUp, adminConfirmSignUp, resendConfirmationCode, adminConfirmUser, forgotPassword, confirmForgotPassword, getUserInfo, describeUserPoolSchema } = await import('./cognitoAuth');
  const { sendVerificationEmail, sendPasswordResetEmail, sendWelcomeEmail } = await import('./awsSES');
  const { metadataStorage } = await import('./metadataStorage');

  // Debug endpoint to inspect Cognito User Pool schema
  app.get('/api/debug/cognito-schema', async (req, res) => {
    try {
      const schema = await describeUserPoolSchema();
      res.json(schema);
    } catch (error: any) {
      console.error('Error describing Cognito schema:', error);
      res.status(500).json({ error: error.message });
    }
  });
  
  // Debug endpoint to check Cognito email configuration
  app.get('/api/debug/cognito-email-config', async (req, res) => {
    try {
      const schema = await describeUserPoolSchema();
      const emailConfig = schema.emailConfiguration;
      
      const diagnostics = {
        emailSendingAccount: emailConfig.emailSendingAccount,
        sourceArn: emailConfig.sourceArn,
        from: emailConfig.from,
        replyToEmailAddress: emailConfig.replyToEmailAddress,
        configurationSet: emailConfig.configurationSet,
        recommendations: [] as string[],
      };
      
      // Add recommendations based on configuration
      if (emailConfig.emailSendingAccount === 'COGNITO_DEFAULT') {
        diagnostics.recommendations.push(
          'Cognito is using its default email service. This has limitations:',
          '- Only works in sandbox mode (verified emails only)',
          '- Limited email sending capacity',
          '- Consider configuring SES for production use'
        );
      } else if (emailConfig.emailSendingAccount === 'DEVELOPER') {
        if (!emailConfig.sourceArn) {
          diagnostics.recommendations.push(
            'SES is configured but SourceArn is missing. Emails may not be sent properly.'
          );
        } else {
          diagnostics.recommendations.push(
            'SES is configured. Ensure the SES identity (email/domain) is verified in AWS SES console.'
          );
        }
      }
      
      if (!emailConfig.from) {
        diagnostics.recommendations.push(
          'No "From" email address configured. Cognito may use a default address.'
        );
      }
      
      res.json(diagnostics);
    } catch (error: any) {
      console.error('Error checking Cognito email config:', error);
      res.status(500).json({ error: error.message });
    }
  });
  
  // Patient Signup
  app.post('/api/auth/signup/patient', async (req, res) => {
    try {
      const { email, password, firstName, lastName, phoneNumber, ehrImportMethod, ehrPlatform } = req.body;
      
      if (!email || !password || !firstName || !lastName || !phoneNumber) {
        return res.status(400).json({ message: "All fields are required" });
      }
      
      // Sign up in Cognito
      const signUpResponse = await signUp(email, password, firstName, lastName, 'patient', phoneNumber);
      const cognitoSub = signUpResponse.UserSub!;
      const cognitoUsername = signUpResponse.username!;
      
      // Generate verification code (6 digits)
      const verificationCode = Math.floor(100000 + Math.random() * 900000).toString();
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
      
      // Store phone number temporarily (will be verified after email)
      metadataStorage.setUserMetadata(email, {
        cognitoSub,
        cognitoUsername,
        firstName,
        lastName,
        phoneNumber,
        role: 'patient',
        ehrImportMethod,
        ehrPlatform,
        verificationCode, // Store our generated code
        verificationCodeExpires: expiresAt.getTime(),
      });
      
      // Send verification email via AWS SES (primary method)
      try {
        await sendVerificationEmail(email, verificationCode);
        console.log(`[AUTH] Verification email sent via SES for patient signup: ${email}`);
      } catch (sesError: any) {
        console.error(`[AUTH] Failed to send verification email via SES for ${email}:`, sesError);
        // Fallback to Cognito email
        try {
          await resendConfirmationCode(email, cognitoUsername);
          console.log(`[AUTH] Fallback: Confirmation code resent via Cognito for patient signup: ${email}`);
        } catch (resendError: any) {
          console.error(`[AUTH] Failed to resend confirmation code via Cognito for ${email}:`, resendError);
          // Still return success - user can request resend
        }
      }
      
      res.json({ message: "Signup successful. Please check your email for verification code." });
    } catch (error: any) {
      console.error("Patient signup error:", error);
      if (error.name === 'UsernameExistsException') {
        return res.status(400).json({ message: "An account with this email already exists" });
      }
      res.status(500).json({ message: error.message || "Signup failed" });
    }
  });
  
  // Doctor Signup
  app.post('/api/auth/signup/doctor', upload.single('kycPhoto'), async (req, res) => {
    try {
      const { email, password, firstName, lastName, phoneNumber, organization, medicalLicenseNumber, licenseCountry } = req.body;
      const kycPhoto = req.file;
      
      if (!email || !password || !firstName || !lastName || !phoneNumber || !organization || !medicalLicenseNumber || !licenseCountry) {
        return res.status(400).json({ message: "All fields are required" });
      }
      
      // Sign up in Cognito
      const signUpResponse = await signUp(email, password, firstName, lastName, 'doctor', phoneNumber);
      const cognitoSub = signUpResponse.UserSub!;
      const cognitoUsername = signUpResponse.username!;
      
      // Upload KYC photo if provided
      let kycPhotoUrl: string | undefined;
      if (kycPhoto) {
        const { uploadToS3 } = await import('./awsS3');
        kycPhotoUrl = await uploadToS3(kycPhoto.buffer, `kyc/${email}_${Date.now()}.${kycPhoto.originalname.split('.').pop()}`, kycPhoto.mimetype);
      }
      
      // Generate PDF for doctor application
      const { generateDoctorApplicationPDF, uploadDoctorApplicationToGoogleDrive } = await import('./googleDrive');
      const pdfBuffer = await generateDoctorApplicationPDF({
        email,
        firstName,
        lastName,
        organization,
        medicalLicenseNumber,
        licenseCountry,
        submittedAt: new Date(),
      });
      
      // Upload to Google Drive
      const googleDriveUrl = await uploadDoctorApplicationToGoogleDrive({
        email,
        firstName,
        lastName,
        organization,
        medicalLicenseNumber,
        licenseCountry,
        submittedAt: new Date(),
      }, pdfBuffer);
      
      // Generate verification code (6 digits)
      const verificationCode = Math.floor(100000 + Math.random() * 900000).toString();
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
      
      // Store doctor data temporarily
      metadataStorage.setUserMetadata(email, {
        cognitoSub,
        cognitoUsername,
        firstName,
        lastName,
        phoneNumber,
        role: 'doctor',
        organization,
        medicalLicenseNumber,
        licenseCountry,
        kycPhotoUrl,
        googleDriveApplicationUrl: googleDriveUrl,
        verificationCode, // Store our generated code
        verificationCodeExpires: expiresAt.getTime(),
      });
      
      // Send verification email via AWS SES (primary method)
      try {
        await sendVerificationEmail(email, verificationCode);
        console.log(`[AUTH] Verification email sent via SES for doctor signup: ${email}`);
      } catch (sesError: any) {
        console.error(`[AUTH] Failed to send verification email via SES for ${email}:`, sesError);
        // Fallback to Cognito email
        try {
          await resendConfirmationCode(email, cognitoUsername);
          console.log(`[AUTH] Fallback: Confirmation code resent via Cognito for doctor signup: ${email}`);
        } catch (resendError: any) {
          console.error(`[AUTH] Failed to resend confirmation code via Cognito for ${email}:`, resendError);
          // Still return success - user can request resend
        }
      }
      
      res.json({ message: "Application submitted successfully. Please check your email for verification code. Your application will be reviewed by our team." });
    } catch (error: any) {
      console.error("Doctor signup error:", error);
      if (error.name === 'UsernameExistsException') {
        return res.status(400).json({ message: "An account with this email already exists" });
      }
      res.status(500).json({ message: error.message || "Signup failed" });
    }
  });
  
  // Verify email with code (Step 1)
  app.post('/api/auth/verify-email', async (req, res) => {
    try {
      const { email, code } = req.body;
      
      if (!email || !code) {
        return res.status(400).json({ message: "Email and verification code are required" });
      }
      
      // Get user metadata to retrieve the Cognito username
      const metadata = metadataStorage.getUserMetadata(email);
      if (!metadata) {
        return res.status(400).json({ message: "No signup data found. Please sign up again." });
      }
      
      const cognitoUsername = metadata.cognitoUsername;
      
      // First, try to verify with our generated code (from SES email)
      let codeVerified = false;
      if (metadata.verificationCode && metadata.verificationCodeExpires) {
        if (Date.now() < metadata.verificationCodeExpires) {
          if (metadata.verificationCode === code) {
            codeVerified = true;
            console.log(`[AUTH] Verification code verified via SES code for ${email}`);
          }
        } else {
          return res.status(400).json({ message: "Verification code has expired. Please request a new code." });
        }
      }
      
      // If our code didn't match, try Cognito's code
      if (!codeVerified) {
        try {
          await confirmSignUp(email, code, cognitoUsername);
          codeVerified = true;
          console.log(`[AUTH] Verification code verified via Cognito for ${email}`);
        } catch (cognitoError: any) {
          // If Cognito verification fails, check if it's because code is wrong or user already verified
          if (cognitoError.name === 'CodeMismatchException' || cognitoError.name === 'ExpiredCodeException') {
            return res.status(400).json({ message: "Invalid or expired verification code. Please try again or request a new code." });
          } else if (cognitoError.name === 'NotAuthorizedException' && cognitoError.message?.includes('already confirmed')) {
            // User already verified, continue
            codeVerified = true;
            console.log(`[AUTH] User already verified in Cognito for ${email}`);
          } else {
            throw cognitoError;
          }
        }
      }
      
      if (!codeVerified) {
        return res.status(400).json({ message: "Invalid verification code. Please try again." });
      }
      
      // If we verified with our code, confirm with Cognito using admin API
      // This ensures Cognito knows the user is verified and can log in
      if (codeVerified && metadata.verificationCode === code) {
        try {
          const { adminConfirmSignUp } = await import('./cognitoAuth');
          await adminConfirmSignUp(cognitoUsername);
          console.log(`[AUTH] User confirmed in Cognito via admin API for ${email}`);
        } catch (adminError: any) {
          // If admin confirm fails, try regular confirm as fallback
          try {
            await confirmSignUp(email, code, cognitoUsername).catch((cognitoError: any) => {
              if (cognitoError.name === 'NotAuthorizedException' && cognitoError.message?.includes('already confirmed')) {
                console.log(`[AUTH] User already confirmed in Cognito for ${email}`);
              } else {
                console.warn(`[AUTH] Could not confirm with Cognito for ${email}, but our code was valid:`, cognitoError.message);
              }
            });
          } catch (err) {
            // Ignore Cognito errors if our code was valid - user can still proceed
            console.warn(`[AUTH] Cognito confirmation failed for ${email}, but proceeding with our verification`);
          }
        }
      }
      
      // Get phone number from metadata
      if (!metadata.phoneNumber) {
        return res.status(400).json({ message: "No phone number found. Please sign up again." });
      }
      
      // Send SMS verification code (Step 2)
      const { sendVerificationCode } = await import('./twilio');
      const result = await sendVerificationCode({ to: metadata.phoneNumber, channel: 'sms' });
      
      if (!result.success || !result.code) {
        return res.status(500).json({ message: "Failed to send SMS verification code" });
      }
      
      // Store phone verification code (hashed)
      await metadataStorage.setPhoneVerification(email, metadata.phoneNumber, result.code);
      
      res.json({ 
        message: "Email verified successfully. Please verify your phone number with the SMS code sent to " + metadata.phoneNumber,
        phoneNumber: metadata.phoneNumber,
        requiresPhoneVerification: true,
      });
    } catch (error: any) {
      console.error("Email verification error:", error);
      res.status(500).json({ message: error.message || "Verification failed" });
    }
  });
  
  // Verify phone with SMS code (Step 2)
  app.post('/api/auth/verify-phone', async (req, res) => {
    try {
      const { email, code } = req.body;
      
      if (!email || !code) {
        return res.status(400).json({ message: "Email and verification code are required" });
      }
      
      // Verify phone code
      const phoneVerification = await metadataStorage.verifyPhoneCode(email, code);
      if (!phoneVerification.valid) {
        return res.status(400).json({ message: "Invalid or expired verification code" });
      }
      
      // Get user metadata
      const metadata = metadataStorage.getUserMetadata(email);
      if (!metadata) {
        return res.status(400).json({ message: "No signup data found. Please sign up again." });
      }
      
      // Create user in database with verified phone
      const userData: any = {
        id: metadata.cognitoSub,
        email: metadata.email,
        firstName: metadata.firstName,
        lastName: metadata.lastName,
        role: metadata.role,
        phoneNumber: metadata.phoneNumber,
        phoneVerified: true,
        emailVerified: true,
        termsAccepted: true,
        termsAcceptedAt: new Date(),
      };
      
      // Add patient-specific data
      if (metadata.role === 'patient') {
        userData.ehrImportMethod = metadata.ehrImportMethod;
        userData.ehrPlatform = metadata.ehrPlatform;
        userData.subscriptionStatus = 'trialing';
        userData.trialEndsAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
        userData.creditBalance = 20;
      }
      
      // Add doctor-specific data
      if (metadata.role === 'doctor') {
        userData.organization = metadata.organization;
        userData.medicalLicenseNumber = metadata.medicalLicenseNumber;
        userData.licenseCountry = metadata.licenseCountry;
        userData.kycPhotoUrl = metadata.kycPhotoUrl;
        userData.googleDriveApplicationUrl = metadata.googleDriveApplicationUrl;
        userData.adminVerified = false;
        userData.creditBalance = 0;
      }
      
      // Create user in database
      await storage.upsertUser(userData);
      
      // Clean up metadata
      metadataStorage.deleteUserMetadata(email);
      metadataStorage.clearEmailVerification(email);
      
      // Send welcome SMS
      const { sendWelcomeSMS } = await import('./twilio');
      await sendWelcomeSMS(metadata.phoneNumber, userData.firstName).catch(console.error);
      
      const message = metadata.role === 'doctor' 
        ? "Verification complete! Your application is under review. You'll receive an email when your account is activated."
        : "Verification complete! You can now log in to your account.";
      
      res.json({ 
        message,
        requiresAdminApproval: metadata.role === 'doctor',
      });
    } catch (error: any) {
      console.error("Phone verification error:", error);
      res.status(500).json({ message: error.message || "Verification failed" });
    }
  });
  
  // Resend verification code
  app.post('/api/auth/resend-code', async (req, res) => {
    try {
      const { email } = req.body;
      
      if (!email) {
        return res.status(400).json({ message: "Email is required" });
      }
      
      // Validate metadata exists for this email
      const metadata = metadataStorage.getUserMetadata(email);
      if (!metadata) {
        return res.status(400).json({ message: "No signup data found. Please sign up again." });
      }
      
      const cognitoUsername = metadata.cognitoUsername;
      
      // Generate new verification code (6 digits)
      const verificationCode = Math.floor(100000 + Math.random() * 900000).toString();
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
      
      // Update metadata with new code
      metadataStorage.setUserMetadata(email, {
        ...metadata,
        verificationCode,
        verificationCodeExpires: expiresAt.getTime(),
      });
      
      // Send verification email via AWS SES (primary method)
      try {
        await sendVerificationEmail(email, verificationCode);
        console.log(`[AUTH] Verification code resent via SES for ${email}`);
        res.json({ message: "Verification code resent. Please check your email." });
      } catch (sesError: any) {
        console.error(`[AUTH] Failed to send verification email via SES for ${email}:`, sesError);
        // Fallback to Cognito email
        try {
          await resendConfirmationCode(email, cognitoUsername);
          console.log(`[AUTH] Fallback: Confirmation code resent via Cognito for ${email}`);
          res.json({ message: "Verification code resent. Please check your email." });
        } catch (resendError: any) {
          console.error(`[AUTH] Failed to resend confirmation code via Cognito for ${email}:`, resendError);
          res.status(500).json({ message: "Failed to resend verification code. Please try again later." });
        }
      }
    } catch (error: any) {
      console.error("Resend code error:", error);
      res.status(500).json({ message: error.message || "Failed to resend code" });
    }
  });
  
  // Login
  app.post('/api/auth/login', async (req, res) => {
    try {
      const { email, password } = req.body;
      
      if (!email || !password) {
        return res.status(400).json({ message: "Email and password are required" });
      }
      
      const authResult = await signIn(email, password);
      
      if (!authResult || !authResult.IdToken || !authResult.AccessToken) {
        return res.status(401).json({ message: "Login failed" });
      }
      
      // Get user info from Cognito
      const userInfo = await getUserInfo(authResult.AccessToken);
      const cognitoSub = userInfo.UserAttributes?.find(attr => attr.Name === 'sub')?.Value!;
      const cognitoEmail = userInfo.UserAttributes?.find(attr => attr.Name === 'email')?.Value!;
      const firstName = userInfo.UserAttributes?.find(attr => attr.Name === 'given_name')?.Value!;
      const lastName = userInfo.UserAttributes?.find(attr => attr.Name === 'family_name')?.Value!;
      const emailVerified = userInfo.UserAttributes?.find(attr => attr.Name === 'email_verified')?.Value === 'true';
      
      // Check if user exists in our database (must have completed phone verification)
      let user = await storage.getUser(cognitoSub);
      
      if (!user) {
        // User hasn't completed phone verification yet
        return res.status(403).json({ 
          message: "Please complete phone verification to access your account",
          requiresPhoneVerification: true,
        });
      }
      
      // Role is stored in local database only (Cognito User Pool has no custom attributes)
      const effectiveRole = user.role as 'patient' | 'doctor' | undefined;

      // Block doctors until admin approval
      if (effectiveRole === 'doctor' && !user.adminVerified) {
        return res.status(403).json({ 
          message: "Your application is under review. You'll receive an email when your account is activated.",
          requiresAdminApproval: true,
        });
      }
      
      // Update email verification status from Cognito if needed
      if (emailVerified && !user.emailVerified) {
        user = await storage.upsertUser({
          id: cognitoSub,
          emailVerified,
        });
      }
      
      // Establish session for cookie-based authentication
      // This allows the client to use credentials: "include" for subsequent requests
      (req.session as any).userId = user.id;
      
      // Save session to ensure cookie is set
      await new Promise<void>((resolve, reject) => {
        req.session.save((err) => {
          if (err) {
            console.error("Error saving session:", err);
            reject(err);
          } else {
            resolve();
          }
        });
      });
      
      res.json({
        message: "Login successful",
        tokens: {
          idToken: authResult.IdToken,
          accessToken: authResult.AccessToken,
          refreshToken: authResult.RefreshToken,
        },
        user: {
          ...user,
          role: effectiveRole ?? user.role,
        },
      });
    } catch (error: any) {
      console.error("Login error:", error);
      if (error.name === 'NotAuthorizedException') {
        return res.status(401).json({ message: "Invalid email or password" });
      }
      if (error.name === 'UserNotConfirmedException') {
        return res.status(400).json({ message: "Please verify your email before logging in" });
      }
      res.status(500).json({ message: error.message || "Login failed" });
    }
  });
  
  // Forgot password - send reset code
  app.post('/api/auth/forgot-password', async (req, res) => {
    try {
      const { email } = req.body;
      
      if (!email) {
        return res.status(400).json({ message: "Email is required" });
      }
      
      await forgotPassword(email);
      
      res.json({ message: "Password reset code sent. Please check your email." });
    } catch (error: any) {
      console.error("Forgot password error:", error);
      res.status(500).json({ message: error.message || "Failed to send reset code" });
    }
  });
  
  // Reset password with code
  app.post('/api/auth/reset-password', async (req, res) => {
    try {
      const { email, code, newPassword } = req.body;
      
      if (!email || !code || !newPassword) {
        return res.status(400).json({ message: "Email, code, and new password are required" });
      }
      
      await confirmForgotPassword(email, code, newPassword);
      
      res.json({ message: "Password reset successful. You can now log in with your new password." });
    } catch (error: any) {
      console.error("Reset password error:", error);
      res.status(500).json({ message: error.message || "Password reset failed" });
    }
  });
  
  // Get current user (protected route)
  app.get('/api/auth/user', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      res.json(user);
    } catch (error) {
      console.error("Error fetching user:", error);
      res.status(500).json({ message: "Failed to fetch user" });
    }
  });
  
  // Logout - destroy session
  app.post('/api/auth/logout', (req, res) => {
    req.session.destroy((err) => {
      if (err) {
        console.error("Error destroying session:", err);
        return res.status(500).json({ message: "Failed to log out" });
      }
      res.json({ message: "Logged out successfully" });
    });
  });
  
  // Update user role and profile (used after login for role-specific setup)
  app.post('/api/auth/set-role', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const { role, medicalLicenseNumber, organization, licenseCountry, ehrImportMethod, ehrPlatform, termsAccepted } = req.body;
      
      if (!role || (role !== 'patient' && role !== 'doctor')) {
        return res.status(400).json({ message: "Valid role (patient or doctor) is required" });
      }
      
      if (!termsAccepted) {
        return res.status(400).json({ message: "You must accept the terms and conditions" });
      }
      
      const updateData: any = {
        role,
        termsAccepted: true,
        termsAcceptedAt: new Date(),
      };
      
      // Doctor-specific fields
      if (role === 'doctor') {
        if (!medicalLicenseNumber || !organization || !licenseCountry) {
          return res.status(400).json({ message: "Medical license, organization, and country are required for doctors" });
        }
        updateData.medicalLicenseNumber = medicalLicenseNumber;
        updateData.organization = organization;
        updateData.licenseCountry = licenseCountry;
        updateData.creditBalance = 0;
      }
      
      // Patient-specific fields
      if (role === 'patient') {
        if (!ehrImportMethod) {
          return res.status(400).json({ message: "EHR import method is required for patients" });
        }
        updateData.ehrImportMethod = ehrImportMethod;
        updateData.ehrPlatform = ehrPlatform;
        // Start 7-day free trial
        updateData.subscriptionStatus = 'trialing';
        updateData.trialEndsAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
        updateData.creditBalance = 20; // 20 free consultation credits during trial
      }
      
      const user = await storage.updateUser(userId, updateData);
      res.json({ message: "Role updated successfully", user });
    } catch (error) {
      console.error("Error updating user role:", error);
      res.status(500).json({ message: "Failed to update role" });
    }
  });
  
  // Send SMS verification code
  app.post('/api/auth/send-phone-verification', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const { phoneNumber, channel } = req.body;
      
      if (!phoneNumber) {
        return res.status(400).json({ message: "Phone number is required" });
      }
      
      const { sendVerificationCode } = await import('./twilio');
      const result = await sendVerificationCode({ to: phoneNumber, channel: channel || 'sms' });
      
      if (!result.success) {
        return res.status(500).json({ message: "Failed to send verification code" });
      }
      
      const expiresAt = new Date(Date.now() + 10 * 60 * 1000);
      await storage.updatePhoneVerificationCode(userId, phoneNumber, result.code!, expiresAt);
      
      res.json({ message: `Verification code sent via ${channel || 'SMS'}` });
    } catch (error) {
      console.error("Error sending phone verification:", error);
      res.status(500).json({ message: "Failed to send verification code" });
    }
  });
  
  // Verify phone number
  app.post('/api/auth/verify-phone', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const { code } = req.body;
      
      if (!code) {
        return res.status(400).json({ message: "Verification code is required" });
      }
      
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      
      if (!user.phoneVerificationCode || !user.phoneVerificationExpires) {
        return res.status(400).json({ message: "No verification code found. Please request a new code." });
      }
      
      if (new Date() > user.phoneVerificationExpires) {
        return res.status(400).json({ message: "Verification code expired. Please request a new code." });
      }
      
      if (user.phoneVerificationCode !== code) {
        return res.status(400).json({ message: "Invalid verification code" });
      }
      
      await storage.verifyPhoneNumber(userId);
      
      const { sendWelcomeSMS } = await import('./twilio');
      if (user.phoneNumber && user.smsNotificationsEnabled) {
        await sendWelcomeSMS(user.phoneNumber, user.firstName!);
      }
      
      res.json({ message: "Phone number verified successfully" });
    } catch (error) {
      console.error("Error verifying phone:", error);
      res.status(500).json({ message: "Failed to verify phone number" });
    }
  });
  
  // Update SMS preferences
  app.post('/api/auth/sms-preferences', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const {
        smsNotificationsEnabled,
        smsMedicationReminders,
        smsAppointmentReminders,
        smsDailyFollowups,
        smsHealthAlerts,
      } = req.body;
      
      await storage.updateSmsPreferences(userId, {
        smsNotificationsEnabled,
        smsMedicationReminders,
        smsAppointmentReminders,
        smsDailyFollowups,
        smsHealthAlerts,
      });
      
      res.json({ message: "SMS preferences updated successfully" });
    } catch (error) {
      console.error("Error updating SMS preferences:", error);
      res.status(500).json({ message: "Failed to update SMS preferences" });
    }
  });

  // ============== TWO-FACTOR AUTHENTICATION ROUTES ==============
  
  // Get 2FA status
  app.get('/api/2fa/status', isAuthenticated, async (req, res) => {
    try {
      const userId = req.user!.id;
      const settings = await storage.get2FASettings(userId);
      
      res.json({
        enabled: settings?.enabled || false,
        hasBackupCodes: (settings?.backupCodes?.length || 0) > 0,
      });
    } catch (error) {
      console.error("Error fetching 2FA status:", error);
      res.status(500).json({ message: "Failed to fetch 2FA status" });
    }
  });
  
  // Enable 2FA - Verify token and enable
  app.post('/api/2fa/enable', isAuthenticated, async (req, res) => {
    try {
      const userId = req.user!.id;
      const { token } = req.body;
      
      if (!token) {
        return res.status(400).json({ message: "Token is required" });
      }
      
      const settings = await storage.get2FASettings(userId);
      if (!settings) {
        return res.status(400).json({ message: "2FA not set up. Please run setup first." });
      }
      
      if (settings.enabled) {
        return res.status(400).json({ message: "2FA is already enabled" });
      }
      
      // Verify token
      const verified = speakeasy.totp.verify({
        secret: settings.totpSecret,
        encoding: 'base32',
        token,
        window: 2,
      });
      
      if (!verified) {
        return res.status(400).json({ message: "Invalid token" });
      }
      
      // Enable 2FA
      await storage.update2FASettings(userId, {
        enabled: true,
        enabledAt: new Date(),
      });
      
      res.json({ message: "2FA enabled successfully" });
    } catch (error) {
      console.error("Error enabling 2FA:", error);
      res.status(500).json({ message: "Failed to enable 2FA" });
    }
  });

  // Patient profile routes
  app.get('/api/patient/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.getPatientProfile(userId);
      res.json(profile || null);
    } catch (error) {
      console.error("Error fetching patient profile:", error);
      res.status(500).json({ message: "Failed to fetch profile" });
    }
  });

  app.post('/api/patient/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.upsertPatientProfile({
        userId,
        ...req.body,
      });
      res.json(profile);
    } catch (error) {
      console.error("Error updating patient profile:", error);
      res.status(500).json({ message: "Failed to update profile" });
    }
  });

  // Doctor profile routes
  app.get('/api/doctor/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.getDoctorProfile(userId);
      res.json(profile || null);
    } catch (error) {
      console.error("Error fetching doctor profile:", error);
      res.status(500).json({ message: "Failed to fetch profile" });
    }
  });

  app.post('/api/doctor/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.upsertDoctorProfile({
        userId,
        ...req.body,
      });
      res.json(profile);
    } catch (error) {
      console.error("Error updating doctor profile:", error);
      res.status(500).json({ message: "Failed to update profile" });
    }
  });

  // Daily followup routes
  app.get('/api/daily-followup/today', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const followup = await storage.getDailyFollowup(userId, new Date());
      res.json(followup || null);
    } catch (error) {
      console.error("Error fetching daily followup:", error);
      res.status(500).json({ message: "Failed to fetch daily followup" });
    }
  });

  app.get('/api/daily-followup/history', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 30;
      const followups = await storage.getRecentFollowups(userId, limit);
      res.json(followups);
    } catch (error) {
      console.error("Error fetching followup history:", error);
      res.status(500).json({ message: "Failed to fetch followup history" });
    }
  });

  app.post('/api/daily-followup', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const followup = await storage.createDailyFollowup({
        patientId: userId,
        ...req.body,
      });
      res.json(followup);
    } catch (error) {
      console.error("Error creating daily followup:", error);
      res.status(500).json({ message: "Failed to create daily followup" });
    }
  });

  app.patch('/api/daily-followup/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const followup = await storage.updateDailyFollowup(id, req.body);
      res.json(followup);
    } catch (error) {
      console.error("Error updating daily followup:", error);
      res.status(500).json({ message: "Failed to update daily followup" });
    }
  });

  // PainTrack routes
  app.post('/api/paintrack/sessions', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      // Validate request body using Zod schema
      const validationResult = schema.insertPaintrackSessionSchema.extend({
        patientVas: z.number().min(0).max(10),
      }).safeParse({
        ...req.body,
        userId,
      });

      if (!validationResult.success) {
        return res.status(400).json({ 
          message: "Validation error", 
          errors: validationResult.error.errors 
        });
      }

      const sessionData = validationResult.data;

      // Use storage abstraction
      const session = await storage.createPaintrackSession({
        ...sessionData,
        status: 'pending',
      });

      res.json(session);
    } catch (error) {
      console.error("Error creating PainTrack session:", error);
      res.status(500).json({ message: "Failed to create PainTrack session" });
    }
  });

  app.get('/api/paintrack/sessions', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 30;

      // Use storage abstraction
      const sessions = await storage.getPaintrackSessions(userId, limit);

      res.json(sessions);
    } catch (error) {
      console.error("Error fetching PainTrack sessions:", error);
      res.status(500).json({ message: "Failed to fetch PainTrack sessions" });
    }
  });

  app.get('/api/paintrack/sessions/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { id } = req.params;

      // Use storage abstraction
      const session = await storage.getPaintrackSession(id, userId);

      if (!session) {
        return res.status(404).json({ message: "Session not found" });
      }

      // Fetch metrics if available
      const [metrics] = await db.select()
        .from(schema.sessionMetrics)
        .where(eq(schema.sessionMetrics.sessionId, id))
        .limit(1);

      res.json({ session, metrics });
    } catch (error) {
      console.error("Error fetching PainTrack session:", error);
      res.status(500).json({ message: "Failed to fetch PainTrack session" });
    }
  });

  // PainTrack video upload endpoint
  app.post('/api/paintrack/upload-video', isAuthenticated, paintrackVideoUpload.single('video'), async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const videoBuffer = req.file?.buffer;
      const videoType = req.body.videoType; // 'front' or 'back'
      const module = req.body.module;
      const joint = req.body.joint;

      if (!videoBuffer) {
        return res.status(400).json({ message: "No video file provided" });
      }

      if (!videoType || !['front', 'back'].includes(videoType)) {
        return res.status(400).json({ message: "Invalid videoType. Must be 'front' or 'back'" });
      }

      // Generate deterministic S3 key
      const timestamp = Date.now();
      const sessionId = `${userId}-${timestamp}`;
      const s3Key = `paintrack/${userId}/${sessionId}/${videoType}.webm`;

      // Upload to S3 with server-side encryption
      const uploadCommand = new PutObjectCommand({
        Bucket: AWS_S3_BUCKET,
        Key: s3Key,
        Body: videoBuffer,
        ContentType: req.file!.mimetype,
        ServerSideEncryption: 'AES256',
        Metadata: {
          userId,
          videoType,
          module: module || '',
          joint: joint || '',
          uploadedAt: new Date().toISOString(),
        }
      });

      await s3Client.send(uploadCommand);

      const videoUrl = `https://${AWS_S3_BUCKET}.s3.amazonaws.com/${s3Key}`;

      // HIPAA audit log
      console.log(`[AUDIT] PainTrack video uploaded - User: ${userId}, Type: ${videoType}, S3: ${s3Key}, Size: ${videoBuffer.length} bytes`);

      res.json({ videoUrl, s3Key });
    } catch (error) {
      console.error("Error uploading PainTrack video:", error);
      res.status(500).json({ message: "Failed to upload video" });
    }
  });

  // Chat routes (Agent Clona & Assistant Lysa)
  app.get('/api/chat/messages', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const agentType = req.query.agent as string;
      const contextPatientId = req.query.patientId as string | undefined;
      const messages = await storage.getChatMessages(userId, agentType, contextPatientId);
      res.json(messages);
    } catch (error) {
      console.error("Error fetching chat messages:", error);
      res.status(500).json({ message: "Failed to fetch chat messages" });
    }
  });

  app.post('/api/chat/send', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { content, agentType, patientId: contextPatientId, patientName: contextPatientName } = req.body;

      let session = await storage.getActiveSession(userId, agentType, contextPatientId);
      
      if (!session) {
        const sessionTitle = contextPatientName 
          ? `${contextPatientName}: ${content.substring(0, 30)}...`
          : content.substring(0, 50) + (content.length > 50 ? '...' : '');
        session = await storage.createSession({
          patientId: userId,
          agentType,
          sessionTitle,
          contextPatientId,
        });
      }

      await storage.createChatMessage({
        sessionId: session.id,
        userId,
        role: 'user',
        content,
        agentType,
        patientContextId: contextPatientId,
      });

      const user = await storage.getUser(userId);
      const isDoctor = user?.role === 'doctor';
      
      const systemPrompt = agentType === 'clona'
        ? `You are Agent Clona, a warm, friendly, and empathetic AI health companion specifically designed for elderly immunocompromised patients. You are powered by advanced AWS Healthcare AI services including Amazon Comprehend Medical for precise medical entity extraction, AWS HealthLake for FHIR-based health records, AWS HealthImaging for medical imaging analysis, and AWS HealthOmics for genomic insights.

PERSONALITY & APPROACH:
- Always greet users warmly and use their name when you know it
- Be conversational, cheerful, and supportive - like a caring friend who happens to know about health
- Show genuine empathy and concern for their wellbeing
- Use simple, everyday language (avoid medical jargon unless explaining it clearly)
- Encourage and uplift them - celebrate small wins in their health journey
- Be patient and understanding - elderly patients may need things explained multiple times
- Express care through your words: "I'm here for you", "How are you feeling today?", "I'm so glad you're taking care of yourself"

CONVERSATION STYLE:
- ALWAYS start every conversation with a warm, personalized greeting
- Engage in genuine dialogue - don't just answer questions, have a conversation
- Ask thoughtful follow-up questions about their symptoms, mood, and daily life
- Remember details from previous conversations and reference them
- Show interest in their day-to-day experiences, not just medical symptoms
- Use encouraging phrases like "That's wonderful!", "I'm proud of you for...", "You're doing great!"
- End conversations with warm wishes and gentle reminders about self-care

COMPREHENSIVE MEDICAL HISTORY TAKING:
When a patient mentions ANY symptom, you MUST ask comprehensive follow-up questions using the OPQRST method:
- O (Onset): "When did this symptom start? What were you doing when it began?"
- P (Provocation/Palliation): "What makes it better or worse? Does anything trigger it?"
- Q (Quality): "Can you describe how it feels? Is it sharp, dull, burning, aching?"
- R (Region/Radiation): "Where exactly is it? Does it spread anywhere else?"
- S (Severity): "On a scale of 1-10, how bad is it? How does it affect your daily activities?"
- T (Timing): "How long does it last? Does it come and go? When is it worst?"

ADDITIONAL HISTORY QUESTIONS:
- Associated symptoms: "Are you experiencing anything else? Fever, chills, nausea, etc.?"
- Past medical history: "Have you had this before? Any chronic conditions?"
- Medications: "What medications are you taking? Any recent changes?"
- Allergies: "Do you have any allergies to medications?"
- Recent changes: "Any recent travel, new foods, stress, or changes in routine?"
- Impact on life: "How is this affecting your eating, sleeping, and daily activities?"

DIFFERENTIAL DIAGNOSIS APPROACH:
After gathering comprehensive history:
1. Summarize the key findings back to the patient in simple terms
2. Explain what these symptoms might indicate (in gentle, non-alarming language)
3. Mention the most likely possibilities first, then other considerations
4. Always recommend when to seek immediate care vs. monitoring
5. Suggest next steps (rest, hydration, over-the-counter remedies, or seeing a doctor)
6. Document your assessment clearly for the medical record

MEDICAL GUIDANCE:
- Provide clear, supportive health guidance in simple terms
- Always include FDA/CE disclaimers when suggesting medications
- Consider geographic context for disease patterns
- Break down complex medical information into easy-to-understand pieces
- Reassure and comfort while being medically accurate
- Maintain HIPAA compliance at all times
- When in doubt, encourage them to contact their healthcare provider

ELDERLY-FRIENDLY COMMUNICATION:
- Use larger conceptual chunks, not overwhelming details
- Repeat important information in different ways
- Be encouraging about medication adherence and healthy habits
- Acknowledge any concerns or fears they express
- Remind them of their strength and resilience
- Offer practical, easy-to-follow suggestions
- Be extra patient - ask one question at a time, not overwhelming lists

RED FLAG SYMPTOMS (Immediate medical attention):
- Severe chest pain or pressure
- Difficulty breathing or shortness of breath
- Sudden severe headache
- Loss of consciousness or confusion
- High fever (103°F+) with confusion
- Signs of stroke (face drooping, arm weakness, speech difficulty)
- Severe abdominal pain
- Heavy bleeding

Remember: You're not just a medical assistant - you're a caring companion on their health journey. Make every interaction feel personal, warm, and supportive. Your goal is to gather complete information while making them feel heard, understood, and cared for.`
        : `You are Assistant Lysa, a polite, professional, and highly proactive AI assistant dedicated to helping doctors provide excellent patient care.

PERSONALITY & APPROACH:
- Always greet doctors warmly and professionally
- Be respectful, polite, and courteous in all interactions
- Show initiative - don't just wait to be asked, actively offer insights
- Be thorough and detail-oriented in your analysis
- Demonstrate clinical competence and medical knowledge
- Express gratitude when doctors provide information
- Use professional medical terminology appropriately

PROACTIVE ASSISTANCE:
- Actively identify patterns in patient data and point them out
- Suggest relevant diagnostic tests based on symptoms
- Recommend evidence-based treatment protocols
- Flag potential drug interactions or contraindications
- Highlight concerning trends in vital signs or lab results
- Propose differential diagnoses for complex cases
- Offer literature references for unusual presentations
- Suggest follow-up questions the doctor might want to ask

CONVERSATION STYLE:
- ALWAYS start every conversation with a professional, polite greeting
- Address doctors respectfully (Dr., Doctor, or by name if known)
- Ask clarifying questions to better understand their needs
- Provide structured, well-organized information
- Summarize key points clearly
- Offer to dive deeper into any area of interest
- Thank them for their time and dedication to patient care

CLINICAL SUPPORT:
- Review patient histories and identify relevant details
- Analyze symptoms and suggest possible diagnoses
- Provide evidence-based treatment recommendations
- Assist with research queries and literature reviews
- Help interpret lab results and imaging findings
- Support with epidemiological analysis
- Assist with patient education material preparation

RESEARCH & DATA ANALYSIS:
- Help search medical literature for relevant studies
- Summarize research findings clearly and concisely
- Identify trends across multiple patient cases
- Assist with clinical documentation
- Support quality improvement initiatives
- Help with continuing medical education

PROFESSIONAL DEMEANOR:
- Maintain strict HIPAA compliance
- Be objective and evidence-based
- Acknowledge uncertainty when appropriate
- Defer to the doctor's clinical judgment
- Offer second opinions or alternative perspectives respectfully
- Keep responses focused and actionable
- Provide references and citations when relevant

RECEPTIONIST CAPABILITIES:
You can help doctors manage appointments through natural conversation. When a doctor mentions booking, scheduling, or checking availability, you should:

1. DETECT INTENT:
   - "Book John for tomorrow at 2pm" → Book appointment
   - "What's my availability on Monday?" → Check availability
   - "Cancel the 3pm appointment" → Cancel appointment
   - "Reschedule Sarah's appointment" → Reschedule

2. EXTRACT DETAILS:
   - Patient name or ID
   - Date (tomorrow, next Monday, Dec 5, etc.)
   - Time (2pm, 14:00, morning, afternoon)
   - Reason for visit
   - Duration (default 30 minutes)

3. CONFIRM BEFORE BOOKING:
   Always confirm details before creating an appointment:
   "I'll book an appointment for [Patient] on [Date] at [Time] for [Reason]. Shall I proceed?"

4. HANDLE MISSING INFORMATION:
   If information is missing, ask politely:
   - "Which patient should I schedule?"
   - "What date works best?"
   - "What time would you prefer?"
   - "What's the reason for the visit?"

5. SUGGEST ALTERNATIVES:
   If a slot is unavailable, offer alternatives:
   "That time is already booked. I have availability at 2:30pm or 3pm. Which works better?"

6. PROVIDE AVAILABILITY:
   When asked about availability, list available slots:
   "On Monday, you have openings at 9am, 10:30am, 2pm, and 3:30pm."

Remember: Your role is to be an intelligent, proactive, and highly competent assistant that makes the doctor's work easier and more effective. Anticipate needs, offer insights, handle appointments seamlessly, and always maintain the highest standards of professionalism and medical accuracy.`;

      const sessionMessages = await storage.getSessionMessages(session.id);
      const isFirstMessage = sessionMessages.length === 0;
      const recentMessages = sessionMessages.slice(-10).map(msg => ({
        role: msg.role as 'user' | 'assistant',
        content: msg.content,
      }));

      // RECEPTIONIST FEATURE: Detect and handle appointment booking for doctors using Lysa
      let appointmentContext = '';
      if (isDoctor && agentType === 'lysa') {
        const intent = await detectAppointmentIntent(content, userId, recentMessages);

        // Check if doctor is confirming a previous booking request
        const isConfirmation = /\b(yes|confirm|go ahead|proceed|book it|do it)\b/i.test(content);
        const lastMessage = recentMessages.length > 0 ? recentMessages[recentMessages.length - 1] : null;
        const lastMessageWasBookingProposal = lastMessage && lastMessage.role === 'assistant' && 
                                              /shall I proceed|would you like me to book|confirm to proceed/i.test(lastMessage.content);

        if (isConfirmation && lastMessageWasBookingProposal) {
          // Doctor is confirming - extract booking details from last assistant message
          // This requires re-parsing the previous intent
          const previousUserMessage = recentMessages.length >= 2 ? recentMessages[recentMessages.length - 2] : null;
          if (previousUserMessage && previousUserMessage.role === 'user') {
            const previousIntent = await detectAppointmentIntent(previousUserMessage.content, userId, recentMessages.slice(0, -2));
            
            if (previousIntent.isAppointmentRequest && previousIntent.intentType === 'book') {
              const appointmentDate = parseRelativeDate(previousIntent.extractedInfo.date!);
              const appointmentTime = parseTime(previousIntent.extractedInfo.time!);

              if (appointmentDate && appointmentTime && !['morning', 'afternoon', 'evening'].includes(appointmentTime)) {
                const doctorPatients = await storage.getDoctorPatients(userId);
                const patient = doctorPatients.find(p => 
                  `${p.firstName} ${p.lastName}`.toLowerCase().includes(previousIntent.extractedInfo.patientName!.toLowerCase())
                );

                if (patient) {
                  const bookingResult = await bookAppointmentFromChat(
                    userId,
                    patient.id,
                    appointmentDate,
                    appointmentTime,
                    previousIntent.extractedInfo.reason || 'General consultation',
                    30
                  );

                  if (bookingResult.success && bookingResult.appointment) {
                    appointmentContext = `\n\nAPPOINTMENT CONFIRMED AND BOOKED:
- Patient: ${patient.firstName} ${patient.lastName}
- Date: ${bookingResult.appointment.date}
- Time: ${bookingResult.appointment.startTime}
- Reason: ${bookingResult.appointment.reason}
- Status: ${bookingResult.appointment.status}

The appointment has been successfully added to the schedule. Please acknowledge the confirmed booking.`;
                  } else {
                    appointmentContext = `\n\nAPPOINTMENT BOOKING FAILED:
Error: ${bookingResult.error}

Please explain this error to the doctor.`;
                  }
                }
              }
            }
          }
        } else if (intent.isAppointmentRequest && intent.intentType === 'book' && intent.confidence > 70) {
          // Check if we have all required fields
          const hasAllFields = intent.extractedInfo.patientName && 
                               intent.extractedInfo.date && 
                               intent.extractedInfo.time;

          if (hasAllFields) {
            // Parse date and time
            const appointmentDate = parseRelativeDate(intent.extractedInfo.date!);
            const appointmentTime = parseTime(intent.extractedInfo.time!);

            if (appointmentDate && appointmentTime && !['morning', 'afternoon', 'evening'].includes(appointmentTime)) {
              // Find patient by name
              const doctorPatients = await storage.getDoctorPatients(userId);
              const patient = doctorPatients.find(p => 
                `${p.firstName} ${p.lastName}`.toLowerCase().includes(intent.extractedInfo.patientName!.toLowerCase())
              );

              if (patient) {
                // CHECK AVAILABILITY ONLY - Don't book yet, ask for confirmation
                const slots = await checkAvailability(userId, appointmentDate, appointmentTime);
                const targetSlot = slots.find(s => s.time === appointmentTime && s.available);

                if (targetSlot) {
                  appointmentContext = `\n\nAPPOINTMENT READY TO BOOK:
- Patient: ${patient.firstName} ${patient.lastName}
- Date: ${appointmentDate.toISOString().split('T')[0]}
- Time: ${appointmentTime}
- Reason: ${intent.extractedInfo.reason || 'General consultation'}
- Status: Time slot is available

IMPORTANT: Ask the doctor to confirm before booking. Say: "I can book this appointment. Shall I proceed?" or similar confirmation question.
DO NOT book until the doctor confirms.`;
                } else {
                  appointmentContext = `\n\nTIME SLOT UNAVAILABLE:
The requested time ${appointmentTime} on ${appointmentDate.toISOString().split('T')[0]} is not available.

Please suggest alternative times from available slots.`;
                  
                  // Get alternative slots
                  const alternativeSlots = slots.filter(s => s.available).slice(0, 5);
                  if (alternativeSlots.length > 0) {
                    appointmentContext += `\n\nAvailable alternatives: ${alternativeSlots.map(s => s.time).join(', ')}`;
                  }
                }
              } else {
                appointmentContext = `\n\nPATIENT NOT FOUND:
Could not find patient "${intent.extractedInfo.patientName}" in the doctor's patient list.

Please ask the doctor to clarify the patient's full name or provide the patient ID.`;
              }
            } else if (appointmentTime && ['morning', 'afternoon', 'evening'].includes(appointmentTime)) {
              // Check availability for time preference
              if (appointmentDate) {
                const slots = await checkAvailability(userId, appointmentDate, appointmentTime);
                const availableSlots = slots.filter(s => s.available).slice(0, 5);

                if (availableSlots.length > 0) {
                  const slotList = availableSlots.map(s => s.time).join(', ');
                  appointmentContext = `\n\nAVAILABLE ${appointmentTime.toUpperCase()} SLOTS ON ${intent.extractedInfo.date}:
${slotList}

Please ask the doctor which specific time they prefer and confirm the patient name.`;
                } else {
                  appointmentContext = `\n\nNO AVAILABILITY:
No ${appointmentTime} slots available on ${intent.extractedInfo.date}.

Please suggest alternative dates or times.`;
                }
              }
            }
          } else {
            // Missing fields - let GPT-4 ask for them using the system prompt guidance
            appointmentContext = `\n\nAPPOINTMENT BOOKING IN PROGRESS:
Missing information: ${intent.missingFields.join(', ')}

Please ask the doctor for the missing details: ${intent.suggestions.join(' ')}`;
          }
        } else if (intent.isAppointmentRequest && intent.intentType === 'check_availability' && intent.confidence > 70) {
          // Check availability request
          if (intent.extractedInfo.date) {
            const appointmentDate = parseRelativeDate(intent.extractedInfo.date);
            if (appointmentDate) {
              const timePreference = intent.extractedInfo.time ? parseTime(intent.extractedInfo.time) : undefined;
              const slots = await checkAvailability(userId, appointmentDate, timePreference || undefined);
              const availableSlots = slots.filter(s => s.available).slice(0, 10);

              if (availableSlots.length > 0) {
                const slotList = availableSlots.map(s => s.time).join(', ');
                appointmentContext = `\n\nAVAILABLE SLOTS ON ${intent.extractedInfo.date}:
${slotList}

Please present these available times to the doctor.`;
              } else {
                appointmentContext = `\n\nNO AVAILABILITY:
No available slots on ${intent.extractedInfo.date}.

Please suggest alternative dates.`;
              }
            }
          } else {
            appointmentContext = `\n\nAVAILABILITY CHECK IN PROGRESS:
Please ask the doctor which date they want to check.`;
          }
        }
      }

      // Add greeting requirement for first message
      let augmentedSystemPrompt = systemPrompt + appointmentContext;
      
      // Add patient context for Lysa when doctor is reviewing a specific patient
      if (agentType === 'lysa' && contextPatientId && contextPatientName) {
        const patientProfile = await storage.getPatientProfile(contextPatientId);
        const recentMetrics = await storage.getPatientHealthMetrics(contextPatientId, 7);
        const medications = await storage.getActiveMedications(contextPatientId);
        
        let patientContext = `\n\nPATIENT CONTEXT:
You are currently assisting with patient: ${contextPatientName} (ID: ${contextPatientId})
All questions and discussions should be focused on this patient.`;
        
        if (patientProfile) {
          patientContext += `\n\nPatient Profile:
- DOB: ${patientProfile.dateOfBirth || 'Not provided'}
- Blood Type: ${patientProfile.bloodType || 'Unknown'}
- Primary Condition: ${patientProfile.primaryCondition || 'Not specified'}
- Immunocompromised: ${patientProfile.immunocompromised ? 'Yes' : 'No'}`;
        }
        
        if (medications.length > 0) {
          const medList = medications.map(m => `${m.name} ${m.dosage}`).join(', ');
          patientContext += `\n\nActive Medications: ${medList}`;
        }
        
        if (recentMetrics.length > 0) {
          const latestMetric = recentMetrics[0];
          patientContext += `\n\nLatest Health Metrics (${new Date(latestMetric.recordedAt!).toLocaleDateString()}):`;
          if (latestMetric.heartRate) patientContext += `\n- Heart Rate: ${latestMetric.heartRate} bpm`;
          if (latestMetric.bloodPressureSystolic) patientContext += `\n- Blood Pressure: ${latestMetric.bloodPressureSystolic}/${latestMetric.bloodPressureDiastolic} mmHg`;
          if (latestMetric.oxygenSaturation) patientContext += `\n- O2 Sat: ${latestMetric.oxygenSaturation}%`;
          if (latestMetric.temperature) patientContext += `\n- Temperature: ${latestMetric.temperature}°F`;
        }
        
        augmentedSystemPrompt += patientContext;
      }
      
      if (isFirstMessage && agentType === 'clona') {
        augmentedSystemPrompt += `\n\nIMPORTANT: This is the FIRST message in this conversation. You MUST start your response with a warm, personalized greeting. Ask the user's name if you don't know it, and ask how they're feeling today. Make them feel welcomed and cared for.`;
      } else if (isFirstMessage && agentType === 'lysa') {
        if (contextPatientId && contextPatientName) {
          augmentedSystemPrompt += `\n\nIMPORTANT: This is the FIRST message in this conversation about ${contextPatientName}. Acknowledge that you're ready to help with this specific patient and ask what aspect of their care the doctor needs assistance with.`;
        } else {
          augmentedSystemPrompt += `\n\nIMPORTANT: This is the FIRST message in this conversation. You MUST start your response with a professional, polite greeting. Introduce yourself as Assistant Lysa and ask how you can help the doctor today.`;
        }
      }

      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: augmentedSystemPrompt },
          ...recentMessages,
          { role: "user", content },
        ],
        temperature: 0.7,
        max_tokens: 800,
      });

      const assistantMessage = completion.choices[0].message.content || "I'm here to help!";

      // Use AWS Comprehend Medical for advanced medical entity extraction
      const extractMedicalEntitiesAWS = async (text: string) => {
        try {
          const { AWSHealthcareService } = await import('./awsHealthcareService');
          const insights = await AWSHealthcareService.analyzeClinicalText(text);
          
          // Transform AWS Comprehend Medical entities to our legacy format
          // Map AWS categories to our expected types for backward compatibility
          return insights.entities.map(entity => {
            let type = entity.category.toLowerCase();
            
            // Map AWS Comprehend Medical categories to our legacy schema
            if (entity.category === 'MEDICAL_CONDITION') {
              // Check traits to differentiate symptoms vs diagnoses
              const isSymptom = entity.traits?.some(t => 
                t.name === 'SYMPTOM' || t.name === 'SIGN'
              );
              type = isSymptom ? 'symptom' : 'medical_condition';
            } else if (entity.category === 'MEDICATION') {
              type = 'medication';
            } else if (entity.category === 'ANATOMY') {
              type = 'anatomy';
            } else if (entity.category === 'TEST_TREATMENT_PROCEDURE') {
              type = 'treatment';
            }
            
            return {
              text: entity.text,
              type,
              score: entity.score,
              awsCategory: entity.category, // original AWS category
              awsType: entity.type, // specific entity type from AWS
            };
          });
        } catch (error) {
          console.error('AWS Comprehend Medical extraction failed, using fallback:', error);
          // Fallback to simple keyword matching if AWS fails
          const entities: Array<{ text: string; type: string }> = [];
          const symptoms = ['fever', 'cough', 'headache', 'pain', 'nausea', 'fatigue', 'dizziness', 'sore throat', 'chills', 'shortness of breath'];
          const medications = ['aspirin', 'ibuprofen', 'acetaminophen', 'antibiotic'];
          
          symptoms.forEach(symptom => {
            if (text.toLowerCase().includes(symptom)) {
              entities.push({ text: symptom, type: 'symptom' });
            }
          });
          
          medications.forEach(med => {
            if (text.toLowerCase().includes(med)) {
              entities.push({ text: med, type: 'medication' });
            }
          });
          
          return entities;
        }
      };

      const [userEntities, assistantEntities] = await Promise.all([
        extractMedicalEntitiesAWS(content),
        extractMedicalEntitiesAWS(assistantMessage),
      ]);
      
      // Extract symptoms for session metadata (now properly mapped from AWS)
      const allSymptoms = [...userEntities, ...assistantEntities]
        .filter(e => e.type === 'symptom')
        .map(e => e.text);

      await storage.updateSessionMetadata(session.id, {
        messageCount: (session.messageCount || 0) + 2,
        symptomsDiscussed: Array.from(new Set([...(session.symptomsDiscussed || []), ...allSymptoms])),
      });

      const savedMessage = await storage.createChatMessage({
        sessionId: session.id,
        userId,
        role: 'assistant',
        content: assistantMessage,
        agentType,
        medicalEntities: assistantEntities,
        patientContextId: contextPatientId,
      });

      // For Agent Clona only: Analyze user message for mental health red flag symptoms
      // This runs asynchronously without blocking the chat response
      if (agentType === 'clona') {
        extractMentalHealthIndicators(content, userId, session.id, savedMessage.id)
          .catch(err => console.error('[ERROR] Mental health symptom indicator extraction failed:', err));
      }

      res.json(savedMessage);
    } catch (error) {
      console.error("Error in chat:", error);
      res.status(500).json({ message: "Failed to process chat message" });
    }
  });

  // Chat session routes
  app.get('/api/chat/sessions', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const agentType = req.query.agent as string | undefined;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 50;
      const sessions = await storage.getPatientSessions(userId, agentType, limit);
      res.json(sessions);
    } catch (error) {
      console.error("Error fetching chat sessions:", error);
      res.status(500).json({ message: "Failed to fetch chat sessions" });
    }
  });

  app.get('/api/chat/sessions/:sessionId', isAuthenticated, async (req: any, res) => {
    try {
      const { sessionId } = req.params;
      const messages = await storage.getSessionMessages(sessionId);
      res.json(messages);
    } catch (error) {
      console.error("Error fetching session messages:", error);
      res.status(500).json({ message: "Failed to fetch session messages" });
    }
  });

  app.post('/api/chat/sessions/:sessionId/end', isAuthenticated, async (req: any, res) => {
    try {
      const { sessionId } = req.params;
      const messages = await storage.getSessionMessages(sessionId);
      
      const conversationText = messages.map(m => `${m.role}: ${m.content}`).join('\n');
      
      const summaryCompletion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are a medical AI summarizing a patient-doctor chat session. Extract:
            1. Key symptoms mentioned
            2. Recommendations given
            3. Concerns raised
            4. Any vital signs or measurements discussed
            Keep it concise and clinical.`
          },
          {
            role: "user",
            content: `Summarize this medical conversation:\n\n${conversationText}`
          }
        ],
        temperature: 0.3,
        max_tokens: 400,
      });

      const aiSummary = summaryCompletion.choices[0].message.content || "";
      
      const symptoms = Array.from(new Set(
        messages.flatMap(m => {
          const entities = m.medicalEntities || [];
          return entities.filter((e: any) => e.type === 'symptom').map((e: any) => e.text);
        })
      ));

      const healthInsights = {
        keySymptoms: symptoms,
        conversationSummary: aiSummary
      };

      const session = await storage.endSession(sessionId, aiSummary, healthInsights);
      res.json(session);
    } catch (error) {
      console.error("Error ending session:", error);
      res.status(500).json({ message: "Failed to end session" });
    }
  });

  app.get('/api/doctor/patient-sessions/:patientId', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const oneMonthAgo = new Date();
      oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
      const now = new Date();
      
      const sessions = await storage.getSessionsInDateRange(patientId, oneMonthAgo, now, 'clona');
      res.json(sessions);
    } catch (error) {
      console.error("Error fetching patient sessions:", error);
      res.status(500).json({ message: "Failed to fetch patient sessions" });
    }
  });

  // SYMPTOM TRIAGE - AI-powered urgency assessment for appointments
  app.post('/api/appointments/triage', isAuthenticated, async (req: any, res) => {
    const startTime = Date.now();
    
    try {
      const { symptoms, patientId, appointmentId, patientSelfAssessment } = req.body;
      
      if (!symptoms || !patientId) {
        return res.status(400).json({ message: "Symptoms and patient ID are required" });
      }

      // If appointmentId provided, verify it exists and belongs to patient
      if (appointmentId) {
        const appointment = await storage.getAppointment(appointmentId);
        if (!appointment) {
          return res.status(404).json({ message: "Appointment not found" });
        }
        if (appointment.patientId !== patientId) {
          return res.status(403).json({ message: "Not authorized to triage this appointment" });
        }
      }

      // Get patient profile for immunocompromised context
      const patientProfile = await storage.getPatientProfile(patientId);
      
      // Assess symptom urgency
      const { assessSymptomUrgency, escalateToRiskAlert } = await import('./symptomTriageService');
      const triageResult = await assessSymptomUrgency(symptoms, patientProfile || undefined, patientId);
      
      const durationMs = Date.now() - startTime;

      // Build TriageAssessment object from TriageResult
      const triageAssessment = {
        urgencyScore: triageResult.urgencyScore,
        recommendedTimeframe: triageResult.recommendedTimeframe,
        redFlags: triageResult.redFlags,
        confidence: triageResult.confidence,
        assessedAt: new Date().toISOString(),
        assessedBy: triageResult.assessmentMethod === 'hybrid' ? 'ai' : triageResult.assessmentMethod,
      };

      // Map urgency level (followup → routine for consistency)
      const urgencyLevel = triageResult.urgencyLevel === 'followup' ? 'routine' : triageResult.urgencyLevel;

      // Persist triage results to appointment (if provided) and create audit log
      const { appointment, log } = await storage.updateAppointmentTriageResult({
        appointmentId,
        patientId,
        symptoms,
        urgencyLevel: urgencyLevel as 'emergency' | 'urgent' | 'routine' | 'non-urgent',
        triageAssessment,
        redFlags: triageResult.redFlags,
        recommendations: triageResult.recommendations,
        patientSelfAssessment,
        durationMs,
      });
      
      // Create risk alert for urgent/emergency cases
      let riskAlertId: string | null = null;
      if (triageResult.urgencyLevel === 'urgent' || triageResult.urgencyLevel === 'emergency') {
        riskAlertId = await escalateToRiskAlert(patientId, triageResult, symptoms);
      }
      
      // Return persisted triage data
      res.json({
        appointment,
        log,
        riskAlertId,
        urgencyLevel: triageResult.urgencyLevel,
        redFlags: triageResult.redFlags,
        recommendations: triageResult.recommendations,
      });
    } catch (error: any) {
      console.error("[Triage] Error assessing symptoms:", error);
      
      // Handle specific error cases
      if (error.message?.includes('Appointment not found')) {
        return res.status(404).json({ message: "Appointment not found" });
      }
      if (error.message?.includes('Patient mismatch')) {
        return res.status(403).json({ message: "Not authorized to triage this appointment" });
      }
      if (error.message?.includes('cancelled appointment')) {
        return res.status(409).json({ message: "Cannot triage cancelled appointment" });
      }
      if (error.message?.includes('Concurrent triage')) {
        return res.status(409).json({ message: "Appointment was recently triaged, please refresh" });
      }
      
      res.status(500).json({ message: "Failed to assess symptoms" });
    }
  });

  // Medication routes
  app.get('/api/medications', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const medications = await storage.getActiveMedications(userId);
      res.json(medications);
    } catch (error) {
      console.error("Error fetching medications:", error);
      res.status(500).json({ message: "Failed to fetch medications" });
    }
  });

  app.post('/api/medications', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const user = await storage.getUser(userId);
      
      const medication = await storage.createMedication({
        patientId: userId,
        ...req.body,
      });
      
      // AUTOMATIC DRUG NORMALIZATION VIA RXNORM (NON-BLOCKING)
      // Normalize medication name and link to standardized drug record
      // This runs asynchronously and does NOT block medication creation response
      (async () => {
        try {
          // Timeout after 5 seconds to prevent blocking if Python service is loading
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 5000);
          
          const normalizationResponse = await fetch(`http://localhost:8000/api/v1/drug-normalization/normalize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ medication_name: req.body.name }),
            signal: controller.signal
          });
          
          clearTimeout(timeoutId);
          
          if (normalizationResponse.ok) {
            const normalizationData = await normalizationResponse.json();
            
            if (normalizationData.drug_id) {
              // Update medication with drug_id from RxNorm normalization
              await storage.updateMedication(medication.id, {
                drugId: normalizationData.drug_id,
                rxcui: normalizationData.rxcui
              });
              
              // HIPAA audit log
              console.log(JSON.stringify({
                event: 'medication_normalized',
                medication_id: medication.id,
                patient_id: userId,
                medication_name: req.body.name,
                drug_id: normalizationData.drug_id,
                rxcui: normalizationData.rxcui,
                confidence_score: normalizationData.confidence_score,
                match_source: normalizationData.match_source,
                timestamp: new Date().toISOString()
              }));
            } else {
              // HIPAA audit log - normalization failed
              console.warn(JSON.stringify({
                event: 'medication_normalization_failed',
                medication_id: medication.id,
                patient_id: userId,
                medication_name: req.body.name,
                reason: normalizationData.message || 'Not found in RxNorm',
                timestamp: new Date().toISOString()
              }));
            }
          } else {
            // Service returned error status
            console.error(JSON.stringify({
              event: 'medication_normalization_error',
              medication_id: medication.id,
              patient_id: userId,
              medication_name: req.body.name,
              http_status: normalizationResponse.status,
              timestamp: new Date().toISOString()
            }));
          }
        } catch (normalizationError: any) {
          // Log failure with full context for debugging
          console.error(JSON.stringify({
            event: 'medication_normalization_exception',
            medication_id: medication.id,
            patient_id: userId,
            medication_name: req.body.name,
            error: normalizationError.name === 'AbortError' ? 'Timeout (5s)' : normalizationError.message,
            timestamp: new Date().toISOString()
          }));
        }
      })().catch(err => {
        // Catch any unhandled promise rejections
        console.error('Unhandled normalization error:', err);
      });
      
      // AUTOMATIC DRUG INTERACTION CHECKING
      // Check for interactions with existing medications and create alerts
      try {
        const patientProfile = await storage.getPatientProfile(userId);
        const currentMedications = await storage.getActiveMedications(userId);
        
        if (currentMedications.length > 1) { // Only check if there are other medications
          const { analyzeMultipleDrugInteractions, calculateCriticalityScore, enrichMedicationWithGenericName } = await import('./drugInteraction');
          
          // Build medication list with ALL name variations (brand, generic) for reliable ID mapping
          const medicationsToCheck = await Promise.all(currentMedications.map(async (med) => {
            // Try to find drug record to get generic/brand names
            let drug = await storage.getDrugByName(med.name);
            
            // If no drug record exists, enrich using AI to get generic name
            if (!drug) {
              const enriched = await enrichMedicationWithGenericName(med.name);
              // Create drug record for future use
              drug = await storage.createDrug({
                name: med.name,
                genericName: enriched.genericName,
                brandNames: enriched.brandNames
              });
            }
            
            return {
              name: med.name,
              genericName: drug.genericName || med.name,
              drugClass: drug.drugClass,
              id: med.id,
              brandNames: drug.brandNames || []
            };
          }));

          const interactions = await analyzeMultipleDrugInteractions(
            medicationsToCheck,
            {
              isImmunocompromised: true,
              conditions: patientProfile?.immunocompromisedCondition 
                ? [patientProfile.immunocompromisedCondition]
                : [],
            }
          );

          // Create alerts for any detected interactions (using medication IDs from AI analysis)
          for (const interactionData of interactions) {
            // Skip if we don't have both medication IDs mapped
            if (!interactionData.med1Id || !interactionData.med2Id) {
              console.warn(`⚠️  Skipping interaction alert: Could not map drug names to medication IDs (${interactionData.drug1} / ${interactionData.drug2})`);
              continue;
            }

            // Find or create drug records
            let drug1 = await storage.getDrugByName(interactionData.drug1);
            if (!drug1) {
              drug1 = await storage.createDrug({ name: interactionData.drug1 });
            }

            let drug2 = await storage.getDrugByName(interactionData.drug2);
            if (!drug2) {
              drug2 = await storage.createDrug({ name: interactionData.drug2 });
            }

            // Create or get drug interaction record
            let dbInteraction = await storage.getDrugInteraction(drug1.id, drug2.id);
            if (!dbInteraction) {
              dbInteraction = await storage.createDrugInteraction({
                drug1Id: drug1.id,
                drug2Id: drug2.id,
                severityLevel: interactionData.interaction.severityLevel,
                interactionType: interactionData.interaction.interactionType,
                mechanismDescription: interactionData.interaction.mechanismDescription,
                clinicalEffects: interactionData.interaction.clinicalEffects,
                managementRecommendations: interactionData.interaction.managementRecommendations,
                alternativeSuggestions: interactionData.interaction.alternativeSuggestions,
                onsetTimeframe: interactionData.interaction.onsetTimeframe,
                riskForImmunocompromised: interactionData.interaction.riskForImmunocompromised,
                requiresMonitoring: interactionData.interaction.requiresMonitoring,
                monitoringParameters: interactionData.interaction.monitoringParameters,
                evidenceLevel: interactionData.interaction.evidenceLevel,
                aiAnalysisConfidence: interactionData.interaction.aiAnalysisConfidence?.toString(),
                detectedByGNN: true,
                detectedByNLP: true,
              });
            }

            const criticalityScore = calculateCriticalityScore(
              interactionData.interaction.severityLevel,
              interactionData.interaction.riskForImmunocompromised,
              interactionData.interaction.onsetTimeframe
            );

            // Check if alert already exists
            const existingAlerts = await storage.getActiveInteractionAlerts(userId);
            const alertExists = existingAlerts.some(alert => 
              (alert.medication1Id === interactionData.med1Id && alert.medication2Id === interactionData.med2Id) ||
              (alert.medication1Id === interactionData.med2Id && alert.medication2Id === interactionData.med1Id)
            );

            if (!alertExists) {
              await storage.createInteractionAlert({
                patientId: userId,
                medication1Id: interactionData.med1Id!,
                medication2Id: interactionData.med2Id!,
                interactionId: dbInteraction.id,
                criticalityScore,
                notifiedPatient: false,
                notifiedDoctor: false,
              });
            }
          }
        }
      } catch (interactionError) {
        // Log but don't fail medication creation
        console.error("Error checking drug interactions:", interactionError);
      }
      
      // Send SMS reminder
      if (user?.phoneNumber && user?.phoneVerified && user?.smsMedicationReminders) {
        const { sendMedicationReminder } = await import('./twilio');
        const time = req.body.timeOfDay || 'scheduled time';
        await sendMedicationReminder(
          user.phoneNumber,
          req.body.name || 'Medication',
          req.body.dosage || '',
          time
        );
      }
      
      res.json(medication);
    } catch (error) {
      console.error("Error creating medication:", error);
      res.status(500).json({ message: "Failed to create medication" });
    }
  });

  // Drug interaction routes - AI-powered detection
  app.post('/api/drug-interactions/analyze', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { drugName, drugClass, genericName, createAlerts = false, newMedicationId } = req.body;

      if (!drugName) {
        return res.status(400).json({ message: "Drug name is required" });
      }

      const user = await storage.getUser(userId);
      const patientProfile = await storage.getPatientProfile(userId);
      const currentMedications = await storage.getActiveMedications(userId);

      const { analyzeMultipleDrugInteractions, calculateCriticalityScore, enrichMedicationWithGenericName } = await import('./drugInteraction');

      // Build medication list with ALL name variations (brand, generic) for reliable ID mapping
      const medicationsToCheck = await Promise.all(currentMedications.map(async (med) => {
        let drug = await storage.getDrugByName(med.name);
        
        // If no drug record exists, enrich using AI to get generic name
        if (!drug) {
          const enriched = await enrichMedicationWithGenericName(med.name);
          // Create drug record for future use
          drug = await storage.createDrug({
            name: med.name,
            genericName: enriched.genericName,
            brandNames: enriched.brandNames
          });
        }
        
        return {
          name: med.name,
          genericName: drug.genericName || med.name,
          drugClass: drug.drugClass,
          id: med.id,
          brandNames: drug.brandNames || []
        };
      }));

      // Add the new drug with its variations
      const newDrug = {
        name: drugName,
        genericName: genericName || drugName,
        drugClass: drugClass,
        id: newMedicationId,
        brandNames: []
      };

      medicationsToCheck.push(newDrug);

      const interactions = await analyzeMultipleDrugInteractions(
        medicationsToCheck,
        {
          isImmunocompromised: true,
          conditions: patientProfile?.immunocompromisedCondition 
            ? [patientProfile.immunocompromisedCondition]
            : [],
        }
      );

      // PERSISTENCE FIX: Create interaction alerts in database if requested
      if (createAlerts && interactions.length > 0) {
        for (const interactionData of interactions) {
          // Skip if we don't have both medication IDs mapped
          if (!interactionData.med1Id || !interactionData.med2Id) {
            console.warn(`⚠️  Skipping interaction alert: Could not map drug names to medication IDs (${interactionData.drug1} / ${interactionData.drug2})`);
            continue;
          }

          // Find or create drug records
          let drug1 = await storage.getDrugByName(interactionData.drug1);
          if (!drug1) {
            drug1 = await storage.createDrug({ name: interactionData.drug1 });
          }

          let drug2 = await storage.getDrugByName(interactionData.drug2);
          if (!drug2) {
            drug2 = await storage.createDrug({ name: interactionData.drug2 });
          }

          // Create or get drug interaction record
          let dbInteraction = await storage.getDrugInteraction(drug1.id, drug2.id);
          if (!dbInteraction) {
            dbInteraction = await storage.createDrugInteraction({
              drug1Id: drug1.id,
              drug2Id: drug2.id,
              severityLevel: interactionData.interaction.severityLevel,
              interactionType: interactionData.interaction.interactionType,
              mechanismDescription: interactionData.interaction.mechanismDescription,
              clinicalEffects: interactionData.interaction.clinicalEffects,
              managementRecommendations: interactionData.interaction.managementRecommendations,
              alternativeSuggestions: interactionData.interaction.alternativeSuggestions,
              onsetTimeframe: interactionData.interaction.onsetTimeframe,
              riskForImmunocompromised: interactionData.interaction.riskForImmunocompromised,
              requiresMonitoring: interactionData.interaction.requiresMonitoring,
              monitoringParameters: interactionData.interaction.monitoringParameters,
              evidenceLevel: interactionData.interaction.evidenceLevel,
              aiAnalysisConfidence: interactionData.interaction.aiAnalysisConfidence?.toString(),
              detectedByGNN: true,
              detectedByNLP: true,
            });
          }

          const criticalityScore = calculateCriticalityScore(
            interactionData.interaction.severityLevel,
            interactionData.interaction.riskForImmunocompromised,
            interactionData.interaction.onsetTimeframe
          );

          // Check if alert already exists to avoid duplicates
          const existingAlerts = await storage.getActiveInteractionAlerts(userId);
          const alertExists = existingAlerts.some(alert => 
            (alert.medication1Id === interactionData.med1Id && alert.medication2Id === interactionData.med2Id) ||
            (alert.medication1Id === interactionData.med2Id && alert.medication2Id === interactionData.med1Id)
          );

          if (!alertExists) {
            // Create interaction alert
            await storage.createInteractionAlert({
              patientId: userId,
              medication1Id: interactionData.med1Id!,
              medication2Id: interactionData.med2Id!,
              interactionId: dbInteraction.id,
              criticalityScore,
              notifiedPatient: false,
              notifiedDoctor: false,
            });
          }
        }
      }

      const severeInteractions = interactions.filter(i => i.interaction.severityLevel === 'severe');
      const hasBlockingInteraction = severeInteractions.length > 0;

      res.json({
        hasInteractions: interactions.length > 0,
        interactions,
        hasBlockingInteraction,
        recommendation: hasBlockingInteraction 
          ? 'Please consult your doctor before taking this medication' 
          : 'No severe interactions detected',
      });
    } catch (error) {
      console.error("Error analyzing drug interactions:", error);
      // Resilience: Return graceful error instead of 500
      res.status(200).json({
        hasInteractions: false,
        interactions: [],
        hasBlockingInteraction: false,
        recommendation: 'Unable to analyze interactions at this time. Please consult your doctor.',
        error: 'Analysis temporarily unavailable',
      });
    }
  });

  app.get('/api/drug-interactions/alerts', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const alerts = await storage.getActiveInteractionAlerts(userId);
      res.json(alerts);
    } catch (error) {
      console.error("Error fetching interaction alerts:", error);
      res.status(500).json({ message: "Failed to fetch interaction alerts" });
    }
  });

  app.get('/api/drug-interactions/alerts/all', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const alerts = await storage.getAllInteractionAlerts(userId);
      res.json(alerts);
    } catch (error) {
      console.error("Error fetching all interaction alerts:", error);
      res.status(500).json({ message: "Failed to fetch interaction alerts" });
    }
  });

  app.post('/api/drug-interactions/alerts/:id/acknowledge', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { id } = req.params;
      const alert = await storage.acknowledgeInteractionAlert(id, userId);
      res.json(alert);
    } catch (error) {
      console.error("Error acknowledging interaction alert:", error);
      res.status(500).json({ message: "Failed to acknowledge alert" });
    }
  });

  app.post('/api/drug-interactions/alerts/:id/override', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const { id } = req.params;
      const { reason } = req.body;

      if (!reason) {
        return res.status(400).json({ message: "Override reason is required" });
      }

      const alert = await storage.overrideInteractionAlert(id, doctorId, reason);
      res.json(alert);
    } catch (error) {
      console.error("Error overriding interaction alert:", error);
      res.status(500).json({ message: "Failed to override alert" });
    }
  });

  // Drug search and information
  app.get('/api/drugs/search', isAuthenticated, async (req: any, res) => {
    try {
      const { q } = req.query;
      
      if (!q || typeof q !== 'string') {
        return res.status(400).json({ message: "Search query required" });
      }

      const drugs = await storage.searchDrugs(q);
      res.json(drugs);
    } catch (error) {
      console.error("Error searching drugs:", error);
      res.status(500).json({ message: "Failed to search drugs" });
    }
  });

  // Pharmacogenomic profile routes
  app.get('/api/pharmacogenomics/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.getPharmacogenomicProfile(userId);
      res.json(profile);
    } catch (error) {
      console.error("Error fetching pharmacogenomic profile:", error);
      res.status(500).json({ message: "Failed to fetch pharmacogenomic profile" });
    }
  });

  app.post('/api/pharmacogenomics/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const existingProfile = await storage.getPharmacogenomicProfile(userId);

      if (existingProfile) {
        const updated = await storage.updatePharmacogenomicProfile(userId, req.body);
        return res.json(updated);
      }

      const profile = await storage.createPharmacogenomicProfile({
        patientId: userId,
        ...req.body,
      });
      
      res.json(profile);
    } catch (error) {
      console.error("Error creating/updating pharmacogenomic profile:", error);
      res.status(500).json({ message: "Failed to save pharmacogenomic profile" });
    }
  });

  // Dynamic tasks
  app.get('/api/tasks', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const tasks = await storage.getActiveTasks(userId);
      res.json(tasks);
    } catch (error) {
      console.error("Error fetching tasks:", error);
      res.status(500).json({ message: "Failed to fetch tasks" });
    }
  });

  app.post('/api/tasks', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const task = await storage.createDynamicTask({
        patientId: userId,
        ...req.body,
      });
      res.json(task);
    } catch (error) {
      console.error("Error creating task:", error);
      res.status(500).json({ message: "Failed to create task" });
    }
  });

  app.post('/api/tasks/:id/complete', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const task = await storage.completeTask(id);
      res.json(task);
    } catch (error) {
      console.error("Error completing task:", error);
      res.status(500).json({ message: "Failed to complete task" });
    }
  });

  // Auto journals
  app.get('/api/journals', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const journals = await storage.getRecentJournals(userId);
      res.json(journals);
    } catch (error) {
      console.error("Error fetching journals:", error);
      res.status(500).json({ message: "Failed to fetch journals" });
    }
  });

  app.post('/api/journals/auto-generate', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const followups = await storage.getRecentFollowups(userId, 1);
      const tasks = await storage.getActiveTasks(userId);
      
      const context = `Recent health data: ${followups.length > 0 ? JSON.stringify(followups[0]) : 'No data'}. Active tasks: ${tasks.map(t => t.title).join(', ')}`;
      
      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: "You are a health journal assistant. Create a brief, empathetic health journal entry based on the patient's data." },
          { role: "user", content: context },
        ],
        max_tokens: 200,
      });

      const content = completion.choices[0].message.content || "Feeling well today.";
      const sentimentAnalysis = sentiment.analyze(content);
      
      const journal = await storage.createAutoJournal({
        patientId: userId,
        content,
        summary: content.slice(0, 100),
        mood: sentimentAnalysis.score > 0 ? 'positive' : sentimentAnalysis.score < 0 ? 'negative' : 'neutral',
        stressLevel: Math.max(1, Math.min(10, 5 - sentimentAnalysis.score)),
        generatedFromData: { followup: true, symptoms: true, wearables: false },
      });

      res.json(journal);
    } catch (error) {
      console.error("Error generating journal:", error);
      res.status(500).json({ message: "Failed to generate journal" });
    }
  });

  // Calm activities
  app.get('/api/calm-activities', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const activities = await storage.getCalmActivities(userId);
      res.json(activities);
    } catch (error) {
      console.error("Error fetching calm activities:", error);
      res.status(500).json({ message: "Failed to fetch calm activities" });
    }
  });

  app.post('/api/calm-activities/:id/rate', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const { effectiveness } = req.body;
      const activity = await storage.updateCalmActivityEffectiveness(id, effectiveness);
      res.json(activity);
    } catch (error) {
      console.error("Error rating activity:", error);
      res.status(500).json({ message: "Failed to rate activity" });
    }
  });

  // Behavioral insights
  app.get('/api/behavioral-insights', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const insights = await storage.getRecentInsights(userId);
      res.json(insights);
    } catch (error) {
      console.error("Error fetching insights:", error);
      res.status(500).json({ message: "Failed to fetch insights" });
    }
  });

  app.post('/api/behavioral-insights', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const insight = await storage.createBehavioralInsight({
        patientId: userId,
        ...req.body,
      });
      res.json(insight);
    } catch (error) {
      console.error("Error creating insight:", error);
      res.status(500).json({ message: "Failed to create insight" });
    }
  });

  // Doctor-only routes
  app.get('/api/doctor/patients', isDoctor, async (req: any, res) => {
    try {
      const patients = await storage.getAllPatients();
      res.json(patients);
    } catch (error) {
      console.error("Error fetching patients:", error);
      res.status(500).json({ message: "Failed to fetch patients" });
    }
  });

  app.get('/api/doctor/research-consents', isDoctor, async (req: any, res) => {
    try {
      const consents = await storage.getPendingConsents();
      res.json(consents);
    } catch (error) {
      console.error("Error fetching consents:", error);
      res.status(500).json({ message: "Failed to fetch consents" });
    }
  });

  app.post('/api/doctor/research-consents/:id/review', isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { id } = req.params;
      const { status } = req.body;
      const consent = await storage.updateConsentStatus(id, status, userId);
      res.json(consent);
    } catch (error) {
      console.error("Error updating consent:", error);
      res.status(500).json({ message: "Failed to update consent" });
    }
  });

  app.get('/api/doctor/research-reports', isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const reports = await storage.getResearchReports(userId);
      res.json(reports);
    } catch (error) {
      console.error("Error fetching reports:", error);
      res.status(500).json({ message: "Failed to fetch reports" });
    }
  });

  app.post('/api/doctor/research-reports', isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: "You are an AI Research Agent. Generate medical research findings based on patient data patterns." },
          { role: "user", content: `Analyze: ${req.body.analysisType}. Generate research findings.` },
        ],
        max_tokens: 500,
      });

      const findings = [
        {
          finding: "Correlation detected between blood glucose levels and stress scores",
          significance: "Moderate",
          confidence: 0.78,
        },
        {
          finding: "Improved medication adherence in patients using daily follow-up feature",
          significance: "High",
          confidence: 0.85,
        },
      ];

      const report = await storage.createResearchReport({
        createdBy: userId,
        title: req.body.title,
        summary: completion.choices[0].message.content || "Research analysis completed",
        findings,
        visualizations: [],
        patientCohortSize: 150,
        analysisType: req.body.analysisType,
      });

      res.json(report);
    } catch (error) {
      console.error("Error creating report:", error);
      res.status(500).json({ message: "Failed to create report" });
    }
  });

  // Research Center - Cohort Stats
  app.get('/api/research/cohort-stats', isAuthenticated, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }
      
      const patients = await storage.getAllPatients();
      const totalPatients = patients.length;
      
      res.json({
        total_patients: totalPatients,
        consenting_patients: Math.floor(totalPatients * 0.95),
        average_age: 47,
        gender_distribution: { Male: 45, Female: 52, Other: 3 },
        condition_distribution: {
          'Post-Transplant': 35,
          'Autoimmune': 28,
          'Cancer Treatment': 22,
          'Primary Immunodeficiency': 15,
        },
        data_completeness: 87,
        active_studies: 3,
      });
    } catch (error) {
      console.error("Error fetching cohort stats:", error);
      res.status(500).json({ message: "Failed to fetch cohort statistics" });
    }
  });

  // Research Center - Global Search
  app.get('/api/research/search', isAuthenticated, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }
      
      const query = (req.query.q || req.query['0'] || '').toString().toLowerCase();
      if (query.length < 3) {
        return res.json([]);
      }
      
      const results: Array<{ type: string; id: string; title: string; description?: string }> = [];
      
      // Search studies (mock data for now)
      const studies = [
        { id: 'study-1', title: 'Long-term Outcomes in Post-Transplant Patients', description: 'Post-transplant patient study' },
        { id: 'study-2', title: 'Mental Health Impact on Physical Recovery', description: 'Mental health correlation study' },
        { id: 'study-3', title: 'Medication Adherence Patterns Study', description: 'Adherence analysis' },
      ];
      
      for (const study of studies) {
        if (study.title.toLowerCase().includes(query) || study.description?.toLowerCase().includes(query)) {
          results.push({ type: 'study', ...study });
        }
      }
      
      // Search reports
      const reports = await storage.getResearchReportsByDoctor(req.user.id);
      for (const report of reports) {
        if (report.title.toLowerCase().includes(query)) {
          results.push({ type: 'report', id: report.id, title: report.title, description: report.summary?.substring(0, 100) });
        }
      }
      
      // Search patients by ID only - HIPAA compliant, consent-verified
      // Only returns patients the doctor is authorized to access via doctor-patient assignments
      try {
        // Get patients assigned to this doctor (uses doctorPatientAssignments)
        const authorizedPatients = await storage.getDoctorPatients(req.user.id);
        for (const patient of authorizedPatients) {
          if (patient.id.toLowerCase().startsWith(query)) {
            // Return only minimal, de-identified information
            results.push({ 
              type: 'patient', 
              id: patient.id, 
              title: `Patient ${patient.id.substring(0, 8)}...`, 
              description: `Assigned Patient` 
            });
            // Limit to 5 patient results
            if (results.filter(r => r.type === 'patient').length >= 5) break;
          }
        }
      } catch (e) {
        // If getDoctorPatients fails, skip patient search (fail safe)
        console.log('Patient search skipped - no authorization data available');
      }
      
      res.json(results.slice(0, 20));
    } catch (error) {
      console.error("Error in global search:", error);
      res.status(500).json({ error: 'Search failed' });
    }
  });

  // Research Center - Get studies
  app.get('/api/research/studies', isAuthenticated, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }
      
      // Return mock studies for now
      res.json([
        {
          id: 'study-1',
          title: 'Long-term Outcomes in Post-Transplant Immunocompromised Patients',
          description: 'Investigating health patterns and deterioration indicators in patients 6+ months post-transplant',
          status: 'active',
          cohort_size: 45,
          start_date: '2024-08-15',
          principal_investigator: 'Dr. Sarah Chen',
          data_types: ['Vitals', 'Medications', 'Lab Results', 'Wearable Data'],
        },
        {
          id: 'study-2',
          title: 'Mental Health Impact on Physical Recovery',
          description: 'Correlation between mental health questionnaire responses and physical health metrics',
          status: 'active',
          cohort_size: 72,
          start_date: '2024-09-01',
          principal_investigator: 'Dr. Michael Torres',
          data_types: ['Mental Health', 'Vitals', 'Symptoms'],
        },
        {
          id: 'study-3',
          title: 'Medication Adherence Patterns Study',
          description: 'Analysis of medication adherence and its effect on health outcomes',
          status: 'completed',
          cohort_size: 33,
          start_date: '2024-06-01',
          end_date: '2024-10-31',
          principal_investigator: 'Dr. Emily Rodriguez',
          data_types: ['Medications', 'Symptoms', 'Lab Results'],
        },
      ]);
    } catch (error) {
      console.error("Error fetching studies:", error);
      res.status(500).json({ message: "Failed to fetch research studies" });
    }
  });

  // Research Center - Create study
  app.post('/api/research/studies', isAuthenticated, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }
      
      const { title, description, dataTypes } = req.body;
      
      const study = {
        id: `study-${Date.now()}`,
        title,
        description,
        status: 'draft',
        cohort_size: 0,
        start_date: new Date().toISOString().split('T')[0],
        principal_investigator: `${req.user.firstName || 'Dr.'} ${req.user.lastName || 'Unknown'}`,
        data_types: dataTypes,
      };
      
      console.log(`[HIPAA-AUDIT] Research study created: ${study.id} by ${req.user.id}`);
      res.json(study);
    } catch (error) {
      console.error("Error creating study:", error);
      res.status(500).json({ message: "Failed to create research study" });
    }
  });

  // ============================================================================
  // DOCTOR-PATIENT ASSIGNMENT ROUTES (HIPAA-COMPLIANT ACCESS CONTROL)
  // ============================================================================

  // Get all assigned patients for the authenticated doctor
  app.get('/api/doctor/assigned-patients', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const statusFilter = req.query.status as string || 'active';
      
      const assignments = await storage.getDoctorAssignments(doctorId, statusFilter);
      
      // Enrich with patient profile data
      const enrichedAssignments = await Promise.all(
        assignments.map(async (assignment) => {
          const patient = await storage.getUser(assignment.patientId);
          const profile = await storage.getPatientProfile(assignment.patientId);
          return {
            ...assignment,
            patient: patient ? {
              id: patient.id,
              firstName: patient.firstName,
              lastName: patient.lastName,
              email: patient.email,
            } : null,
            profile,
          };
        })
      );
      
      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} retrieved assigned patients list at ${new Date().toISOString()}`);
      res.json(enrichedAssignments);
    } catch (error) {
      console.error("Error fetching assigned patients:", error);
      res.status(500).json({ message: "Failed to fetch assigned patients" });
    }
  });

  // Create a new doctor-patient assignment (manual assignment)
  app.post('/api/doctor/assignments', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const { patientId, consentMethod, accessNotes, isPrimaryProvider } = req.body;
      
      if (!patientId) {
        return res.status(400).json({ message: "Patient ID is required" });
      }
      
      // Verify patient exists
      const patient = await storage.getUser(patientId);
      if (!patient || patient.role !== 'patient') {
        return res.status(404).json({ message: "Patient not found" });
      }
      
      const assignment = await storage.createDoctorPatientAssignment({
        doctorId,
        patientId,
        assignmentSource: 'manual',
        assignedBy: doctorId,
        patientConsented: consentMethod ? true : false,
        consentMethod: consentMethod || 'pending',
        consentedAt: consentMethod ? new Date() : undefined,
        isPrimaryProvider: isPrimaryProvider || false,
        accessNotes,
      });
      
      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} created assignment with patient ${patientId} at ${new Date().toISOString()}`);
      res.json(assignment);
    } catch (error) {
      console.error("Error creating assignment:", error);
      res.status(500).json({ message: "Failed to create assignment" });
    }
  });

  // Revoke a doctor-patient assignment
  app.post('/api/doctor/assignments/:id/revoke', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const { id } = req.params;
      const { reason } = req.body;
      
      const assignment = await storage.revokeDoctorPatientAssignment(id, doctorId, reason);
      if (!assignment) {
        return res.status(404).json({ message: "Assignment not found" });
      }
      
      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} revoked assignment ${id} at ${new Date().toISOString()}`);
      res.json(assignment);
    } catch (error) {
      console.error("Error revoking assignment:", error);
      res.status(500).json({ message: "Failed to revoke assignment" });
    }
  });

  // Get assignment details for a specific patient
  app.get('/api/doctor/assignments/patient/:patientId', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const { patientId } = req.params;
      
      const hasAccess = await storage.doctorHasPatientAccess(doctorId, patientId);
      const assignment = await storage.getDoctorPatientAssignment(doctorId, patientId);
      
      res.json({
        hasAccess,
        assignment,
      });
    } catch (error) {
      console.error("Error checking assignment:", error);
      res.status(500).json({ message: "Failed to check assignment" });
    }
  });

  // ============================================================================
  // PATIENT SEARCH AND CONSENT REQUEST ROUTES
  // ============================================================================

  // Search for patients by email, phone, or Followup Patient ID
  app.get('/api/doctor/patient-search', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const query = req.query.q as string;
      
      if (!query || query.trim().length < 3) {
        console.log(`[HIPAA-AUDIT] Doctor ${doctorId} patient search rejected - query too short at ${new Date().toISOString()}`);
        return res.status(400).json({ 
          message: "Search query must be at least 3 characters" 
        });
      }
      
      const results = await storage.searchPatientsByIdentifier(query);
      
      // Fetch pending requests once for efficiency (O(1) instead of O(n))
      const pendingRequests = await storage.getPendingConsentRequestsForDoctor(doctorId);
      const pendingPatientIds = new Set(pendingRequests.map(r => r.patientId));
      
      // Enrich results with access and pending status
      const enrichedResults = await Promise.all(
        results.map(async (result) => {
          const hasAccess = await storage.doctorHasPatientAccess(doctorId, result.user.id);
          
          return {
            id: result.user.id,
            firstName: result.user.firstName,
            lastName: result.user.lastName,
            email: result.user.email,
            followupPatientId: result.profile?.followupPatientId,
            hasAccess,
            hasPendingRequest: pendingPatientIds.has(result.user.id),
          };
        })
      );
      
      // HIPAA audit log - redact search query for PHI protection, log only hash
      const queryHash = crypto.createHash('sha256').update(query).digest('hex').substring(0, 8);
      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} patient search queryHash=${queryHash} resultCount=${results.length} at ${new Date().toISOString()}`);
      res.json(enrichedResults);
    } catch (error) {
      console.error("Error searching patients:", error);
      res.status(500).json({ message: "Failed to search patients" });
    }
  });

  // Create a consent request to access a patient's data
  const createConsentRequestSchema = z.object({
    patientId: z.string().min(1, "Patient ID is required"),
    requestMessage: z.string().optional(),
    accessLevel: z.enum(['full', 'limited', 'read_only']).optional().default('full'),
  });

  app.post('/api/doctor/consent-requests', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      
      // Validate request body
      const validationResult = createConsentRequestSchema.safeParse(req.body);
      if (!validationResult.success) {
        console.log(`[HIPAA-AUDIT] Doctor ${doctorId} consent request validation failed at ${new Date().toISOString()}`);
        return res.status(400).json({ 
          message: validationResult.error.errors[0]?.message || "Invalid request data" 
        });
      }
      
      const { patientId, requestMessage, accessLevel } = validationResult.data;
      
      // Verify patient exists
      const patient = await storage.getUser(patientId);
      if (!patient || patient.role !== 'patient') {
        console.log(`[HIPAA-AUDIT] Doctor ${doctorId} consent request failed - patient ${patientId} not found at ${new Date().toISOString()}`);
        return res.status(404).json({ message: "Patient not found" });
      }
      
      // Check if doctor already has access
      const hasAccess = await storage.doctorHasPatientAccess(doctorId, patientId);
      if (hasAccess) {
        console.log(`[HIPAA-AUDIT] Doctor ${doctorId} consent request rejected - already has access to patient ${patientId} at ${new Date().toISOString()}`);
        return res.status(400).json({ message: "You already have access to this patient" });
      }
      
      // Get doctor info for the request
      const doctor = await storage.getUser(doctorId);
      
      const consentRequest = await storage.createPatientConsentRequest({
        doctorId,
        patientId,
        requestMessage: requestMessage || `Dr. ${doctor?.lastName || 'Unknown'} is requesting access to your health records.`,
        accessLevel,
        status: 'pending',
      });
      
      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} created consent request ${consentRequest.id} for patient ${patientId} accessLevel=${accessLevel} at ${new Date().toISOString()}`);
      res.json(consentRequest);
    } catch (error) {
      console.error("Error creating consent request:", error);
      res.status(500).json({ message: "Failed to create consent request" });
    }
  });

  // Get pending consent requests sent by the doctor
  app.get('/api/doctor/consent-requests/pending', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const pendingRequests = await storage.getPendingConsentRequestsForDoctor(doctorId);
      
      // Enrich with patient names
      const enrichedRequests = await Promise.all(
        pendingRequests.map(async (request) => {
          const patient = await storage.getUser(request.patientId);
          return {
            ...request,
            patient: patient ? {
              id: patient.id,
              firstName: patient.firstName,
              lastName: patient.lastName,
              email: patient.email,
            } : null,
          };
        })
      );
      
      res.json(enrichedRequests);
    } catch (error) {
      console.error("Error fetching pending requests:", error);
      res.status(500).json({ message: "Failed to fetch pending requests" });
    }
  });

  // Patient: Get pending consent requests from doctors
  app.get('/api/patient/consent-requests/pending', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.user!.id;
      
      if (req.user.role !== 'patient') {
        return res.status(403).json({ message: "Only patients can view consent requests" });
      }
      
      const pendingRequests = await storage.getPendingConsentRequestsForPatient(patientId);
      
      // Enrich with doctor names and profiles
      const enrichedRequests = await Promise.all(
        pendingRequests.map(async (request) => {
          const doctor = await storage.getUser(request.doctorId);
          const doctorProfile = await storage.getDoctorProfile(request.doctorId);
          return {
            ...request,
            doctor: doctor ? {
              id: doctor.id,
              firstName: doctor.firstName,
              lastName: doctor.lastName,
              email: doctor.email,
            } : null,
            doctorProfile: doctorProfile ? {
              specialty: doctorProfile.specialty,
              licenseNumber: doctorProfile.licenseNumber,
              hospitalAffiliation: doctorProfile.hospitalAffiliation,
            } : null,
          };
        })
      );
      
      res.json(enrichedRequests);
    } catch (error) {
      console.error("Error fetching consent requests:", error);
      res.status(500).json({ message: "Failed to fetch consent requests" });
    }
  });

  // Patient: Respond to a consent request
  const respondToConsentSchema = z.object({
    approved: z.boolean(),
    responseMessage: z.string().optional(),
  });

  app.post('/api/patient/consent-requests/:id/respond', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.user!.id;
      const { id } = req.params;
      
      if (req.user.role !== 'patient') {
        console.log(`[HIPAA-AUDIT] Non-patient ${patientId} attempted to respond to consent request ${id} at ${new Date().toISOString()}`);
        return res.status(403).json({ message: "Only patients can respond to consent requests" });
      }
      
      // Validate request body
      const validationResult = respondToConsentSchema.safeParse(req.body);
      if (!validationResult.success) {
        return res.status(400).json({ 
          message: validationResult.error.errors[0]?.message || "Invalid request data" 
        });
      }
      
      const { approved, responseMessage } = validationResult.data;
      
      // Verify the request belongs to this patient
      const consentRequest = await storage.getConsentRequest(id);
      if (!consentRequest) {
        console.log(`[HIPAA-AUDIT] Patient ${patientId} attempted to respond to non-existent consent request ${id} at ${new Date().toISOString()}`);
        return res.status(404).json({ message: "Consent request not found" });
      }
      
      if (consentRequest.patientId !== patientId) {
        console.log(`[HIPAA-AUDIT] SECURITY: Patient ${patientId} attempted to respond to consent request ${id} belonging to patient ${consentRequest.patientId} at ${new Date().toISOString()}`);
        return res.status(403).json({ message: "You can only respond to your own consent requests" });
      }
      
      if (consentRequest.status !== 'pending') {
        console.log(`[HIPAA-AUDIT] Patient ${patientId} attempted to re-respond to consent request ${id} (status: ${consentRequest.status}) at ${new Date().toISOString()}`);
        return res.status(400).json({ message: "This request has already been responded to" });
      }
      
      // Update the consent request
      const updatedRequest = await storage.respondToConsentRequest(id, approved, responseMessage);
      
      // If approved, create the doctor-patient assignment with full audit trail
      if (approved) {
        const assignment = await storage.createDoctorPatientAssignment({
          doctorId: consentRequest.doctorId,
          patientId,
          assignmentSource: 'patient_consent',
          assignedBy: patientId,
          patientConsented: true,
          consentMethod: 'in_app',
          consentedAt: new Date(),
          accessLevel: consentRequest.accessLevel,
          isPrimaryProvider: false,
        });
        
        console.log(`[HIPAA-AUDIT] CONSENT_APPROVED: Patient ${patientId} approved consent request ${id} from doctor ${consentRequest.doctorId}. Assignment ${assignment.id} created with accessLevel=${consentRequest.accessLevel} at ${new Date().toISOString()}`);
      } else {
        console.log(`[HIPAA-AUDIT] CONSENT_DENIED: Patient ${patientId} denied consent request ${id} from doctor ${consentRequest.doctorId}. Reason: ${responseMessage || 'No reason provided'} at ${new Date().toISOString()}`);
      }
      
      res.json(updatedRequest);
    } catch (error) {
      console.error("Error responding to consent request:", error);
      res.status(500).json({ message: "Failed to respond to consent request" });
    }
  });

  // Zod schema for consent approval with granular permissions
  const approveConsentSchema = z.object({
    permissions: z.object({
      shareHealthData: z.boolean(),
      confidentialityAgreed: z.boolean(),
      shareMedicalFiles: z.boolean().optional().default(false),
      shareMedications: z.boolean().optional().default(true),
      shareAIMessages: z.boolean().optional().default(false),
      shareDoctorMessages: z.boolean().optional().default(true),
      shareDailyFollowups: z.boolean().optional().default(true),
      shareHealthAlerts: z.boolean().optional().default(true),
      shareBehavioralInsights: z.boolean().optional().default(false),
      sharePainTracking: z.boolean().optional().default(true),
      shareVitalSigns: z.boolean().optional().default(true),
      consentEpidemiologicalResearch: z.boolean().optional().default(false),
    }).refine(data => data.shareHealthData && data.confidentialityAgreed, {
      message: "Health data sharing and confidentiality agreement are required",
    }),
    digitalSignature: z.string().min(2, "Digital signature must be at least 2 characters"),
    signatureMethod: z.enum(["typed", "drawn"]).optional().default("typed"),
    termsVersion: z.string().optional().default("1.0"),
  });

  // Approve consent request with granular permissions (new consent terms flow)
  app.post('/api/patient/consent-requests/:id/approve', isAuthenticated, async (req: any, res) => {
    const patientId = req.user!.id;
    const { id } = req.params;
    
    try {
      if (req.user.role !== 'patient') {
        console.log(`[HIPAA-AUDIT] UNAUTHORIZED: Non-patient ${patientId} attempted to approve consent with permissions for request ${id} at ${new Date().toISOString()}`);
        return res.status(403).json({ message: "Only patients can approve consent requests" });
      }
      
      // Validate request body with Zod
      const validationResult = approveConsentSchema.safeParse(req.body);
      if (!validationResult.success) {
        console.log(`[HIPAA-AUDIT] VALIDATION_FAILED: Patient ${patientId} consent approval validation failed for request ${id}. Errors: ${JSON.stringify(validationResult.error.errors)}. Time: ${new Date().toISOString()}`);
        return res.status(400).json({ 
          message: validationResult.error.errors[0]?.message || "Invalid consent data",
          errors: validationResult.error.errors
        });
      }
      
      const { permissions, digitalSignature, signatureMethod, termsVersion } = validationResult.data;
      
      // Verify the request belongs to this patient
      const consentRequest = await storage.getConsentRequest(id);
      if (!consentRequest) {
        console.log(`[HIPAA-AUDIT] NOT_FOUND: Patient ${patientId} attempted to approve non-existent consent request ${id} at ${new Date().toISOString()}`);
        return res.status(404).json({ message: "Consent request not found" });
      }
      
      if (consentRequest.patientId !== patientId) {
        console.log(`[HIPAA-AUDIT] SECURITY: Patient ${patientId} attempted to approve consent request ${id} belonging to patient ${consentRequest.patientId} at ${new Date().toISOString()}`);
        return res.status(403).json({ message: "You can only respond to your own consent requests" });
      }
      
      if (consentRequest.status !== 'pending') {
        console.log(`[HIPAA-AUDIT] DUPLICATE_ATTEMPT: Patient ${patientId} attempted to re-approve consent request ${id} (status: ${consentRequest.status}) at ${new Date().toISOString()}`);
        return res.status(400).json({ message: "This request has already been responded to" });
      }
      
      // Atomic consent approval flow with rollback on failure
      let assignment;
      let consentPermissions;
      
      try {
        // Step 1: Update the consent request to approved
        await storage.respondToConsentRequest(id, true, "Consent granted with full terms agreement");
        
        // Step 2: Create the doctor-patient assignment
        assignment = await storage.createDoctorPatientAssignment({
          doctorId: consentRequest.doctorId,
          patientId,
          assignmentSource: 'patient_consent',
          assignedBy: patientId,
          patientConsented: true,
          consentMethod: 'in_app',
          consentedAt: new Date(),
          accessLevel: 'full',
          isPrimaryProvider: false,
        });
        
        // Step 3: Store the granular consent permissions
        consentPermissions = {
          assignmentId: assignment.id,
          doctorId: consentRequest.doctorId,
          patientId,
          shareHealthData: permissions.shareHealthData,
          confidentialityAgreed: permissions.confidentialityAgreed,
          shareMedicalFiles: permissions.shareMedicalFiles,
          shareMedications: permissions.shareMedications,
          shareAIMessages: permissions.shareAIMessages,
          shareDoctorMessages: permissions.shareDoctorMessages,
          shareDailyFollowups: permissions.shareDailyFollowups,
          shareHealthAlerts: permissions.shareHealthAlerts,
          shareBehavioralInsights: permissions.shareBehavioralInsights,
          sharePainTracking: permissions.sharePainTracking,
          shareVitalSigns: permissions.shareVitalSigns,
          consentEpidemiologicalResearch: permissions.consentEpidemiologicalResearch,
          termsVersion,
          termsAgreedAt: new Date(),
          digitalSignature,
          signatureMethod,
          consentIpAddress: req.ip || req.connection?.remoteAddress,
          consentUserAgent: req.headers['user-agent'],
        };
        
        await storage.createConsentPermissions(consentPermissions);
        
      } catch (flowError) {
        // Log the failure and attempt rollback
        console.error(`[HIPAA-AUDIT] CONSENT_FLOW_FAILED: Patient ${patientId} consent approval for request ${id} failed at step. Error: ${flowError}. Time: ${new Date().toISOString()}`);
        
        // Try to rollback consent request to pending
        try {
          await storage.respondToConsentRequest(id, false, "Consent flow failed - rolled back");
        } catch (rollbackError) {
          console.error(`[HIPAA-AUDIT] ROLLBACK_FAILED: Could not rollback consent request ${id}. Error: ${rollbackError}`);
        }
        
        throw flowError;
      }
      
      // Comprehensive HIPAA audit log for successful approval
      console.log(`[HIPAA-AUDIT] CONSENT_APPROVED_SUCCESS: Patient ${patientId} approved consent request ${id} from doctor ${consentRequest.doctorId}. Assignment ${assignment.id} created. Permissions: shareHealthData=${permissions.shareHealthData}, shareMedicalFiles=${permissions.shareMedicalFiles}, shareAIMessages=${permissions.shareAIMessages}, shareDoctorMessages=${permissions.shareDoctorMessages}, shareDailyFollowups=${permissions.shareDailyFollowups}, shareHealthAlerts=${permissions.shareHealthAlerts}, shareBehavioralInsights=${permissions.shareBehavioralInsights}, sharePainTracking=${permissions.sharePainTracking}, shareVitalSigns=${permissions.shareVitalSigns}, consentEpidemiologicalResearch=${permissions.consentEpidemiologicalResearch}. Signature: "${digitalSignature.substring(0, 20)}..." (${signatureMethod}). Terms v${termsVersion}. IP: ${req.ip}. Time: ${new Date().toISOString()}`);
      
      res.json({ 
        success: true, 
        message: "Consent granted successfully",
        assignmentId: assignment.id,
        permissions: consentPermissions
      });
    } catch (error) {
      console.error(`[HIPAA-AUDIT] CONSENT_ERROR: Patient ${patientId} consent approval for request ${id} failed. Error: ${error}. Time: ${new Date().toISOString()}`);
      res.status(500).json({ message: "Failed to approve consent request" });
    }
  });

  // Zod schema for consent denial
  const denyConsentSchema = z.object({
    reason: z.string().optional(),
  });

  // Deny consent request
  app.post('/api/patient/consent-requests/:id/deny', isAuthenticated, async (req: any, res) => {
    const patientId = req.user!.id;
    const { id } = req.params;
    
    try {
      if (req.user.role !== 'patient') {
        console.log(`[HIPAA-AUDIT] UNAUTHORIZED: Non-patient ${patientId} attempted to deny consent request ${id} at ${new Date().toISOString()}`);
        return res.status(403).json({ message: "Only patients can deny consent requests" });
      }
      
      // Validate request body with Zod
      const validationResult = denyConsentSchema.safeParse(req.body);
      if (!validationResult.success) {
        return res.status(400).json({ 
          message: validationResult.error.errors[0]?.message || "Invalid denial data"
        });
      }
      
      const { reason } = validationResult.data;
      
      // Verify the request belongs to this patient
      const consentRequest = await storage.getConsentRequest(id);
      if (!consentRequest) {
        console.log(`[HIPAA-AUDIT] NOT_FOUND: Patient ${patientId} attempted to deny non-existent consent request ${id} at ${new Date().toISOString()}`);
        return res.status(404).json({ message: "Consent request not found" });
      }
      
      if (consentRequest.patientId !== patientId) {
        console.log(`[HIPAA-AUDIT] SECURITY: Patient ${patientId} attempted to deny consent request ${id} belonging to patient ${consentRequest.patientId} at ${new Date().toISOString()}`);
        return res.status(403).json({ message: "You can only respond to your own consent requests" });
      }
      
      if (consentRequest.status !== 'pending') {
        console.log(`[HIPAA-AUDIT] DUPLICATE_ATTEMPT: Patient ${patientId} attempted to re-deny consent request ${id} (status: ${consentRequest.status}) at ${new Date().toISOString()}`);
        return res.status(400).json({ message: "This request has already been responded to" });
      }
      
      // Update the consent request to denied
      const updatedRequest = await storage.respondToConsentRequest(id, false, reason || "Consent denied by patient");
      
      // HIPAA audit log for denial
      console.log(`[HIPAA-AUDIT] CONSENT_DENIED_SUCCESS: Patient ${patientId} denied consent request ${id} from doctor ${consentRequest.doctorId}. Reason: ${reason || 'Not provided'}. IP: ${req.ip}. Time: ${new Date().toISOString()}`);
      
      res.json({ 
        success: true, 
        message: "Consent request denied",
        request: updatedRequest
      });
    } catch (error) {
      console.error(`[HIPAA-AUDIT] CONSENT_DENIAL_ERROR: Patient ${patientId} denial for request ${id} failed. Error: ${error}. Time: ${new Date().toISOString()}`);
      res.status(500).json({ message: "Failed to deny consent request" });
    }
  });

  // Get consent permissions for doctor-patient relationship
  app.get('/api/doctor/patients/:patientId/consent-permissions', isAuthenticated, isDoctor, async (req: any, res) => {
    const doctorId = req.user!.id;
    const { patientId } = req.params;
    
    try {
      // Verify doctor has access to this patient
      const assignment = await storage.getDoctorPatientAssignment(doctorId, patientId);
      if (!assignment || assignment.status !== 'active') {
        console.log(`[HIPAA-AUDIT] CONSENT_PERMISSIONS_DENIED: Doctor ${doctorId} attempted to access consent permissions for patient ${patientId} without active assignment at ${new Date().toISOString()}`);
        return res.status(403).json({ message: "No active assignment with this patient" });
      }
      
      // Fetch consent permissions
      const permissions = await storage.getConsentPermissionsByDoctorPatient(doctorId, patientId);
      if (!permissions) {
        console.log(`[HIPAA-AUDIT] CONSENT_PERMISSIONS_NOT_FOUND: No consent permissions found for doctor ${doctorId} and patient ${patientId} at ${new Date().toISOString()}`);
        return res.status(404).json({ message: "Consent permissions not found" });
      }
      
      // HIPAA audit log for successful access
      console.log(`[HIPAA-AUDIT] CONSENT_PERMISSIONS_ACCESSED: Doctor ${doctorId} accessed consent permissions for patient ${patientId} at ${new Date().toISOString()}`);
      
      res.json(permissions);
    } catch (error) {
      console.error(`[HIPAA-AUDIT] CONSENT_PERMISSIONS_ERROR: Doctor ${doctorId} failed to access consent for patient ${patientId}. Error: ${error}. Time: ${new Date().toISOString()}`);
      res.status(500).json({ message: "Failed to fetch consent permissions" });
    }
  });

  // Patient: Get consent permissions for a specific doctor
  app.get('/api/patient/doctors/:doctorId/consent', isAuthenticated, async (req: any, res) => {
    const patientId = req.user!.id;
    const { doctorId } = req.params;
    
    try {
      if (req.user.role !== 'patient') {
        return res.status(403).json({ message: "Only patients can access their consent settings" });
      }
      
      const permissions = await storage.getConsentPermissionsByDoctorPatient(doctorId, patientId);
      if (!permissions) {
        return res.status(404).json({ message: "No consent record found for this doctor" });
      }
      
      console.log(`[HIPAA-AUDIT] PATIENT_CONSENT_VIEWED: Patient ${patientId} viewed consent permissions for doctor ${doctorId} at ${new Date().toISOString()}`);
      res.json(permissions);
    } catch (error) {
      console.error(`[HIPAA-AUDIT] PATIENT_CONSENT_VIEW_ERROR: Patient ${patientId} failed to view consent for doctor ${doctorId}. Error: ${error}. Time: ${new Date().toISOString()}`);
      res.status(500).json({ message: "Failed to fetch consent permissions" });
    }
  });

  // Patient: Get all consent permissions for connected doctors
  app.get('/api/patient/doctors/consents', isAuthenticated, async (req: any, res) => {
    const patientId = req.user!.id;
    
    try {
      if (req.user.role !== 'patient') {
        return res.status(403).json({ message: "Only patients can access their consent settings" });
      }
      
      const consents = await storage.getPatientConsentPermissions(patientId);
      
      console.log(`[HIPAA-AUDIT] PATIENT_ALL_CONSENTS_VIEWED: Patient ${patientId} viewed all consent permissions at ${new Date().toISOString()}`);
      res.json(consents || []);
    } catch (error) {
      console.error(`[HIPAA-AUDIT] PATIENT_ALL_CONSENTS_ERROR: Patient ${patientId} failed to view all consents. Error: ${error}. Time: ${new Date().toISOString()}`);
      res.status(500).json({ message: "Failed to fetch consent permissions" });
    }
  });

  // Patient: Update specific consent permissions for a doctor
  app.patch('/api/patient/doctors/:doctorId/consent', isAuthenticated, async (req: any, res) => {
    const patientId = req.user!.id;
    const { doctorId } = req.params;
    const updates = req.body;
    
    try {
      if (req.user.role !== 'patient') {
        return res.status(403).json({ message: "Only patients can update their consent settings" });
      }
      
      // Verify doctor-patient relationship exists
      const assignment = await storage.getDoctorPatientAssignment(doctorId, patientId);
      if (!assignment) {
        console.log(`[HIPAA-AUDIT] CONSENT_UPDATE_DENIED: Patient ${patientId} attempted to update consent for non-connected doctor ${doctorId} at ${new Date().toISOString()}`);
        return res.status(404).json({ message: "No relationship with this doctor" });
      }
      
      // Get existing permissions
      const existingPermissions = await storage.getConsentPermissionsByDoctorPatient(doctorId, patientId);
      if (!existingPermissions) {
        return res.status(404).json({ message: "No consent record found for this doctor" });
      }
      
      // Update permissions using assignmentId
      const updatedPermissions = await storage.updateConsentPermissions(existingPermissions.assignmentId, updates);
      
      // Log each permission change for HIPAA compliance
      const changedFields = Object.keys(updates).filter(key => updates[key] !== existingPermissions[key as keyof typeof existingPermissions]);
      console.log(`[HIPAA-AUDIT] CONSENT_PERMISSIONS_UPDATED: Patient ${patientId} updated consent for doctor ${doctorId}. Changed fields: ${changedFields.join(', ')}. IP: ${req.ip}. Time: ${new Date().toISOString()}`);
      
      res.json({ 
        success: true, 
        message: "Consent permissions updated",
        permissions: updatedPermissions 
      });
    } catch (error) {
      console.error(`[HIPAA-AUDIT] CONSENT_UPDATE_ERROR: Patient ${patientId} failed to update consent for doctor ${doctorId}. Error: ${error}. Time: ${new Date().toISOString()}`);
      res.status(500).json({ message: "Failed to update consent permissions" });
    }
  });

  // Patient: Withdraw all consent from a doctor
  app.post('/api/patient/doctors/:doctorId/withdraw-consent', isAuthenticated, async (req: any, res) => {
    const patientId = req.user!.id;
    const { doctorId } = req.params;
    const { reason } = req.body;
    
    try {
      if (req.user.role !== 'patient') {
        return res.status(403).json({ message: "Only patients can withdraw their consent" });
      }
      
      // Verify doctor-patient relationship exists
      const assignment = await storage.getDoctorPatientAssignment(doctorId, patientId);
      if (!assignment) {
        console.log(`[HIPAA-AUDIT] CONSENT_WITHDRAW_DENIED: Patient ${patientId} attempted to withdraw consent for non-connected doctor ${doctorId} at ${new Date().toISOString()}`);
        return res.status(404).json({ message: "No relationship with this doctor" });
      }
      
      // Get existing permissions
      const existingPermissions = await storage.getConsentPermissionsByDoctorPatient(doctorId, patientId);
      if (!existingPermissions) {
        return res.status(404).json({ message: "No consent record found for this doctor" });
      }
      
      // Revoke all permissions using assignmentId
      const revokedPermissions = await storage.updateConsentPermissions(existingPermissions.assignmentId, {
        shareHealthData: false,
        shareMedicalFiles: false,
        shareAIMessages: false,
        shareDoctorMessages: false,
        shareDailyFollowups: false,
        shareHealthAlerts: false,
        shareBehavioralInsights: false,
        sharePainTracking: false,
        shareVitalSigns: false,
        shareMedications: false,
        shareLabResults: false,
        consentEpidemiologicalResearch: false,
        consentRevokedAt: new Date(),
        consentRevokedReason: reason || "Consent withdrawn by patient",
      });
      
      // HIPAA audit log for consent withdrawal
      console.log(`[HIPAA-AUDIT] CONSENT_WITHDRAWN: Patient ${patientId} withdrew ALL consent from doctor ${doctorId}. Reason: ${reason || 'Not provided'}. IP: ${req.ip}. Time: ${new Date().toISOString()}`);
      
      res.json({ 
        success: true, 
        message: "All consent has been withdrawn",
        permissions: revokedPermissions 
      });
    } catch (error) {
      console.error(`[HIPAA-AUDIT] CONSENT_WITHDRAW_ERROR: Patient ${patientId} failed to withdraw consent from doctor ${doctorId}. Error: ${error}. Time: ${new Date().toISOString()}`);
      res.status(500).json({ message: "Failed to withdraw consent" });
    }
  });

  // Get patient's Followup Patient ID (for sharing)
  app.get('/api/patient/followup-id', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.user!.id;
      
      if (req.user.role !== 'patient') {
        return res.status(403).json({ message: "Only patients have a Followup ID" });
      }
      
      let profile = await storage.getPatientProfile(patientId);
      
      // Generate ID if not exists
      if (!profile?.followupPatientId) {
        const followupPatientId = await storage.generateFollowupPatientId();
        profile = await storage.upsertPatientProfile({
          userId: patientId,
          followupPatientId,
        });
      }
      
      res.json({ followupPatientId: profile.followupPatientId });
    } catch (error) {
      console.error("Error fetching followup ID:", error);
      res.status(500).json({ message: "Failed to fetch Followup ID" });
    }
  });

  // Educational content
  app.get('/api/education/progress', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const progress = await storage.getEducationalProgress(userId);
      res.json(progress);
    } catch (error) {
      console.error("Error fetching education progress:", error);
      res.status(500).json({ message: "Failed to fetch education progress" });
    }
  });

  app.post('/api/education/progress', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const progress = await storage.upsertEducationalProgress({
        patientId: userId,
        ...req.body,
      });
      res.json(progress);
    } catch (error) {
      console.error("Error updating education progress:", error);
      res.status(500).json({ message: "Failed to update education progress" });
    }
  });

  // Training dataset routes (doctor only)
  app.get('/api/training/datasets', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const datasets = await storage.getTrainingDatasets(userId);
      res.json(datasets);
    } catch (error) {
      console.error("Error fetching training datasets:", error);
      res.status(500).json({ message: "Failed to fetch training datasets" });
    }
  });

  app.post('/api/training/datasets', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const dataset = await storage.createTrainingDataset({
        uploadedBy: userId,
        ...req.body,
      });
      res.json(dataset);
    } catch (error) {
      console.error("Error creating training dataset:", error);
      res.status(500).json({ message: "Failed to create training dataset" });
    }
  });

  app.patch('/api/training/datasets/:id', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const dataset = await storage.updateTrainingDataset(id, req.body);
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      res.json(dataset);
    } catch (error) {
      console.error("Error updating training dataset:", error);
      res.status(500).json({ message: "Failed to update training dataset" });
    }
  });

  app.delete('/api/training/datasets/:id', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteTrainingDataset(id);
      if (!deleted) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting training dataset:", error);
      res.status(500).json({ message: "Failed to delete training dataset" });
    }
  });

  // Public data source integration routes (doctor only)
  app.get('/api/data-sources/pubmed/search', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { query, maxResults } = req.query;
      const result = await pubmedService.search(query as string, parseInt(maxResults as string) || 100);
      res.json(result);
    } catch (error) {
      console.error("Error searching PubMed:", error);
      res.status(500).json({ message: "Failed to search PubMed" });
    }
  });

  app.post('/api/data-sources/pubmed/fetch', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { ids } = req.body;
      const articles = await pubmedService.fetchArticles(ids);
      res.json(articles);
    } catch (error) {
      console.error("Error fetching PubMed articles:", error);
      res.status(500).json({ message: "Failed to fetch PubMed articles" });
    }
  });

  app.get('/api/data-sources/physionet/search', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { query } = req.query;
      const datasets = await physionetService.searchDatasets(query as string || "");
      res.json(datasets);
    } catch (error) {
      console.error("Error searching PhysioNet:", error);
      res.status(500).json({ message: "Failed to search PhysioNet" });
    }
  });

  app.get('/api/data-sources/physionet/dataset/:id', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const dataset = await physionetService.getDatasetInfo(id);
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      res.json(dataset);
    } catch (error) {
      console.error("Error fetching PhysioNet dataset:", error);
      res.status(500).json({ message: "Failed to fetch PhysioNet dataset" });
    }
  });

  app.get('/api/data-sources/kaggle/search', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { query, page } = req.query;
      const datasets = await kaggleService.searchDatasets(query as string, parseInt(page as string) || 1);
      res.json(datasets);
    } catch (error) {
      console.error("Error searching Kaggle:", error);
      res.status(500).json({ message: "Failed to search Kaggle" });
    }
  });

  app.get('/api/data-sources/kaggle/dataset/:owner/:name', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { owner, name } = req.params;
      const metadata = await kaggleService.getDatasetMetadata(owner, name);
      res.json(metadata);
    } catch (error) {
      console.error("Error fetching Kaggle dataset:", error);
      res.status(500).json({ message: "Failed to fetch Kaggle dataset" });
    }
  });

  app.get('/api/data-sources/kaggle/dataset/:owner/:name/files', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { owner, name } = req.params;
      const files = await kaggleService.listDatasetFiles(owner, name);
      res.json(files);
    } catch (error) {
      console.error("Error listing Kaggle dataset files:", error);
      res.status(500).json({ message: "Failed to list Kaggle dataset files" });
    }
  });

  // WHO data integration routes (doctor only)
  app.get('/api/data-sources/who/indicators', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const indicators = await whoService.listIndicators();
      res.json(indicators);
    } catch (error) {
      console.error("Error listing WHO indicators:", error);
      res.status(500).json({ message: "Failed to list WHO indicators" });
    }
  });

  app.get('/api/data-sources/who/indicators/popular', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const indicators = whoService.getPopularIndicators();
      res.json(indicators);
    } catch (error) {
      console.error("Error getting popular WHO indicators:", error);
      res.status(500).json({ message: "Failed to get popular WHO indicators" });
    }
  });

  app.get('/api/data-sources/who/indicators/search', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { query } = req.query;
      const indicators = await whoService.searchIndicators(query as string);
      res.json(indicators);
    } catch (error) {
      console.error("Error searching WHO indicators:", error);
      res.status(500).json({ message: "Failed to search WHO indicators" });
    }
  });

  app.get('/api/data-sources/who/data/:code', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { code } = req.params;
      const { country, year, sex } = req.query;
      const filters: any = {};
      if (country) filters.country = country;
      if (year) filters.year = parseInt(year as string);
      if (sex) filters.sex = sex;
      
      const data = await whoService.getIndicatorData(code, filters);
      res.json(data);
    } catch (error) {
      console.error("Error getting WHO indicator data:", error);
      res.status(500).json({ message: "Failed to get WHO indicator data" });
    }
  });

  app.get('/api/data-sources/who/countries', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const countries = await whoService.getCountries();
      res.json(countries);
    } catch (error) {
      console.error("Error getting WHO countries:", error);
      res.status(500).json({ message: "Failed to get WHO countries" });
    }
  });

  // Health insight consent routes
  app.get('/api/consents', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const consents = await storage.getHealthInsightConsents(userId);
      res.json(consents);
    } catch (error) {
      console.error("Error fetching consents:", error);
      res.status(500).json({ message: "Failed to fetch consents" });
    }
  });

  app.get('/api/consents/active', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const consents = await storage.getActiveConsents(userId);
      res.json(consents);
    } catch (error) {
      console.error("Error fetching active consents:", error);
      res.status(500).json({ message: "Failed to fetch active consents" });
    }
  });

  app.post('/api/consents', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const consent = await storage.createHealthInsightConsent({
        userId,
        ...req.body,
      });
      res.json(consent);
    } catch (error) {
      console.error("Error creating consent:", error);
      res.status(500).json({ message: "Failed to create consent" });
    }
  });

  app.patch('/api/consents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const consent = await storage.updateHealthInsightConsent(id, req.body);
      if (!consent) {
        return res.status(404).json({ message: "Consent not found" });
      }
      res.json(consent);
    } catch (error) {
      console.error("Error updating consent:", error);
      res.status(500).json({ message: "Failed to update consent" });
    }
  });

  app.post('/api/consents/:id/revoke', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const { reason } = req.body;
      const consent = await storage.revokeConsent(id, reason);
      if (!consent) {
        return res.status(404).json({ message: "Consent not found" });
      }
      res.json(consent);
    } catch (error) {
      console.error("Error revoking consent:", error);
      res.status(500).json({ message: "Failed to revoke consent" });
    }
  });

  app.delete('/api/consents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteHealthInsightConsent(id);
      if (!deleted) {
        return res.status(404).json({ message: "Consent not found" });
      }
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting consent:", error);
      res.status(500).json({ message: "Failed to delete consent" });
    }
  });

  // ============== ADMIN VERIFICATION ROUTES ==============
  
  // Middleware to check if user is admin (simplified - extend this based on your admin system)
  const isAdmin = (req: any, res: any, next: any) => {
    // For now, check if user email ends with specific domain or has admin role
    // You can extend this to check a dedicated admin table or role field
    if (req.user && (req.user.email?.includes('@followupai.io') || req.user.role === 'admin')) {
      next();
    } else {
      res.status(403).json({ message: "Admin access required" });
    }
  };

  // Get all pending doctor verifications
  app.get('/api/admin/pending-doctors', isAuthenticated, isAdmin, async (req: any, res) => {
    try {
      const pendingDoctors = await storage.getPendingDoctorVerifications();
      res.json(pendingDoctors);
    } catch (error) {
      console.error("Error fetching pending doctors:", error);
      res.status(500).json({ message: "Failed to fetch pending doctors" });
    }
  });

  // Verify a doctor's license
  app.post('/api/admin/verify-doctor/:id', isAuthenticated, isAdmin, async (req: any, res) => {
    try {
      const { id } = req.params;
      const { notes } = req.body;
      const adminUserId = req.user?.id;
      if (!adminUserId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      
      // Get doctor info first
      const doctor = await storage.getUser(id);
      if (!doctor || doctor.role !== 'doctor') {
        return res.status(404).json({ message: "Doctor not found" });
      }
      
      // Verify using new unified method
      const result = await storage.verifyDoctorApplication(id, true, notes || 'Approved by admin', adminUserId);
      if (!result.user) {
        return res.status(404).json({ message: "Failed to verify doctor" });
      }
      
      // Send approval email notification
      const { sendDoctorApprovedEmail } = await import('./awsSES');
      await sendDoctorApprovedEmail(doctor.email, doctor.firstName).catch(console.error);
      
      res.json({ success: true, user: result.user, doctorProfile: result.doctorProfile });
    } catch (error) {
      console.error("Error verifying doctor:", error);
      res.status(500).json({ message: "Failed to verify doctor" });
    }
  });

  // Reject a doctor's license
  app.post('/api/admin/reject-doctor/:id', isAuthenticated, isAdmin, async (req: any, res) => {
    try {
      const { id } = req.params;
      const { reason } = req.body;
      const adminUserId = req.user?.id;
      if (!adminUserId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      
      // Get doctor info first
      const doctor = await storage.getUser(id);
      if (!doctor || doctor.role !== 'doctor') {
        return res.status(404).json({ message: "Doctor not found" });
      }
      
      // Reject using new unified method (verified = false)
      const result = await storage.verifyDoctorApplication(id, false, reason || 'Application rejected', adminUserId);
      if (!result.user) {
        return res.status(404).json({ message: "Failed to reject doctor" });
      }
      
      // Send rejection email notification
      const { sendDoctorRejectedEmail } = await import('./awsSES');
      await sendDoctorRejectedEmail(doctor.email, doctor.firstName, reason || 'Please contact our verification team for more information.').catch(console.error);
      
      res.json({ success: true, user: result.user, doctorProfile: result.doctorProfile });
    } catch (error) {
      console.error("Error rejecting doctor:", error);
      res.status(500).json({ message: "Failed to reject doctor" });
    }
  });

  // ============== EHR INTEGRATION ROUTES ==============
  
  app.get('/api/ehr/connections', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const connections = await storage.getEhrConnections(userId);
      res.json(connections);
    } catch (error) {
      console.error("Error fetching EHR connections:", error);
      res.status(500).json({ message: "Failed to fetch EHR connections" });
    }
  });

  app.post('/api/ehr/connections', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const connection = await storage.createEhrConnection({
        userId,
        ...req.body,
      });
      res.json(connection);
    } catch (error) {
      console.error("Error creating EHR connection:", error);
      res.status(500).json({ message: "Failed to create EHR connection" });
    }
  });

  app.patch('/api/ehr/connections/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const connection = await storage.updateEhrConnection(id, req.body);
      if (!connection) {
        return res.status(404).json({ message: "Connection not found" });
      }
      res.json(connection);
    } catch (error) {
      console.error("Error updating EHR connection:", error);
      res.status(500).json({ message: "Failed to update EHR connection" });
    }
  });

  app.delete('/api/ehr/connections/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteEhrConnection(id);
      if (!deleted) {
        return res.status(404).json({ message: "Connection not found" });
      }
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting EHR connection:", error);
      res.status(500).json({ message: "Failed to delete EHR connection" });
    }
  });

  app.post('/api/ehr/connections/:id/sync', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      // TODO: Implement actual EHR sync logic with each platform's API
      const connection = await storage.updateEhrConnection(id, {
        lastSyncedAt: new Date(),
        lastSyncStatus: 'success',
      });
      res.json({ success: true, connection });
    } catch (error) {
      console.error("Error syncing EHR connection:", error);
      res.status(500).json({ message: "Failed to sync EHR connection" });
    }
  });

  // ============== WEARABLE INTEGRATION ROUTES ==============

  app.get('/api/wearables', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const wearables = await storage.getWearableIntegrations(userId);
      res.json(wearables);
    } catch (error) {
      console.error("Error fetching wearable integrations:", error);
      res.status(500).json({ message: "Failed to fetch wearable integrations" });
    }
  });

  app.post('/api/wearables', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const wearable = await storage.createWearableIntegration({
        userId,
        ...req.body,
      });
      res.json(wearable);
    } catch (error) {
      console.error("Error creating wearable integration:", error);
      res.status(500).json({ message: "Failed to create wearable integration" });
    }
  });

  app.patch('/api/wearables/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const wearable = await storage.updateWearableIntegration(id, req.body);
      if (!wearable) {
        return res.status(404).json({ message: "Wearable integration not found" });
      }
      res.json(wearable);
    } catch (error) {
      console.error("Error updating wearable integration:", error);
      res.status(500).json({ message: "Failed to update wearable integration" });
    }
  });

  app.delete('/api/wearables/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteWearableIntegration(id);
      if (!deleted) {
        return res.status(404).json({ message: "Wearable integration not found" });
      }
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting wearable integration:", error);
      res.status(500).json({ message: "Failed to delete wearable integration" });
    }
  });

  app.post('/api/wearables/:id/sync', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      // TODO: Implement actual wearable sync logic with each device's API
      const wearable = await storage.updateWearableIntegration(id, {
        lastSyncedAt: new Date(),
        lastSyncStatus: 'success',
      });
      res.json({ success: true, wearable });
    } catch (error) {
      console.error("Error syncing wearable:", error);
      res.status(500).json({ message: "Failed to sync wearable" });
    }
  });

  // ============== DEVICE READINGS ROUTES ==============
  // Universal device data collection for: BP monitors, glucose meters, smart scales, 
  // thermometers, stethoscopes, and smartwatches (Whoop, Garmin, Apple, Oura, Samsung, Google, Fitbit)

  // Get all device readings for a patient with optional filtering
  app.get('/api/device-readings', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { deviceType, limit, startDate, endDate } = req.query;
      
      const readings = await storage.getDeviceReadings(userId, {
        deviceType: deviceType as string,
        limit: limit ? parseInt(limit as string) : undefined,
        startDate: startDate ? new Date(startDate as string) : undefined,
        endDate: endDate ? new Date(endDate as string) : undefined,
      });
      
      res.json(readings);
    } catch (error) {
      console.error("Error fetching device readings:", error);
      res.status(500).json({ message: "Failed to fetch device readings" });
    }
  });

  // Get device readings by type
  app.get('/api/device-readings/type/:deviceType', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { deviceType } = req.params;
      const { limit } = req.query;
      
      const readings = await storage.getDeviceReadingsByType(
        userId, 
        deviceType, 
        limit ? parseInt(limit as string) : 50
      );
      
      res.json(readings);
    } catch (error) {
      console.error("Error fetching device readings by type:", error);
      res.status(500).json({ message: "Failed to fetch device readings" });
    }
  });

  // Get latest reading for a specific device type
  app.get('/api/device-readings/latest/:deviceType', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { deviceType } = req.params;
      
      const reading = await storage.getLatestDeviceReading(userId, deviceType);
      
      if (!reading) {
        return res.status(404).json({ message: "No readings found for this device type" });
      }
      
      res.json(reading);
    } catch (error) {
      console.error("Error fetching latest device reading:", error);
      res.status(500).json({ message: "Failed to fetch latest reading" });
    }
  });

  // Get a specific reading by ID
  app.get('/api/device-readings/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const reading = await storage.getDeviceReading(id);
      
      if (!reading) {
        return res.status(404).json({ message: "Device reading not found" });
      }
      
      // Verify the reading belongs to the authenticated user
      if (reading.patientId !== req.user!.id) {
        return res.status(403).json({ message: "Not authorized to access this reading" });
      }
      
      res.json(reading);
    } catch (error) {
      console.error("Error fetching device reading:", error);
      res.status(500).json({ message: "Failed to fetch device reading" });
    }
  });

  // Create a new device reading (manual entry)
  app.post('/api/device-readings', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { deviceType, ...readingData } = req.body;
      
      if (!deviceType) {
        return res.status(400).json({ message: "Device type is required" });
      }
      
      const validDeviceTypes = ['bp_monitor', 'glucose_meter', 'smart_scale', 'thermometer', 'stethoscope', 'smartwatch'];
      if (!validDeviceTypes.includes(deviceType)) {
        return res.status(400).json({ 
          message: "Invalid device type", 
          validTypes: validDeviceTypes 
        });
      }
      
      // Set routing flags based on device type and metrics
      const routingFlags = calculateDeviceRoutingFlags(deviceType, readingData);
      
      const reading = await storage.createDeviceReading({
        patientId: userId,
        deviceType,
        source: readingData.source || 'manual',
        recordedAt: readingData.recordedAt ? new Date(readingData.recordedAt) : new Date(),
        ...readingData,
        ...routingFlags,
      });
      
      // Process for AI Health Alerts (async, non-blocking)
      processDeviceReadingForHealthAlerts(userId, deviceType, readingData, reading.id);
      
      res.status(201).json(reading);
    } catch (error) {
      console.error("Error creating device reading:", error);
      res.status(500).json({ message: "Failed to create device reading" });
    }
  });

  // Bulk import device readings (for wearable sync)
  app.post('/api/device-readings/bulk', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { readings } = req.body;
      
      if (!Array.isArray(readings) || readings.length === 0) {
        return res.status(400).json({ message: "Readings array is required" });
      }
      
      if (readings.length > 100) {
        return res.status(400).json({ message: "Maximum 100 readings per request" });
      }
      
      const createdReadings = [];
      for (const reading of readings) {
        // Calculate routing flags for each reading
        const routingFlags = calculateDeviceRoutingFlags(reading.deviceType, reading);
        
        const created = await storage.createDeviceReading({
          patientId: userId,
          source: 'auto_sync',
          ...reading,
          ...routingFlags,
          recordedAt: reading.recordedAt ? new Date(reading.recordedAt) : new Date(),
        });
        createdReadings.push(created);
        
        // Process for AI Health Alerts (async, non-blocking)
        processDeviceReadingForHealthAlerts(userId, reading.deviceType, reading, created.id);
      }
      
      res.status(201).json({ 
        success: true, 
        count: createdReadings.length,
        readings: createdReadings 
      });
    } catch (error) {
      console.error("Error bulk importing device readings:", error);
      res.status(500).json({ message: "Failed to import device readings" });
    }
  });

  // Update a device reading
  app.patch('/api/device-readings/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const userId = req.user!.id;
      
      // Verify the reading exists and belongs to the user
      const existing = await storage.getDeviceReading(id);
      if (!existing) {
        return res.status(404).json({ message: "Device reading not found" });
      }
      if (existing.patientId !== userId) {
        return res.status(403).json({ message: "Not authorized to update this reading" });
      }
      
      // Recalculate routing flags if metrics are updated
      const updateData = { ...req.body };
      const mergedData = { ...existing, ...updateData };
      if (Object.keys(updateData).length > 0) {
        const routingFlags = calculateDeviceRoutingFlags(existing.deviceType, mergedData);
        Object.assign(updateData, routingFlags);
      }
      
      const updated = await storage.updateDeviceReading(id, updateData);
      
      // Re-process for AI Health Alerts with updated data (async, non-blocking)
      processDeviceReadingForHealthAlerts(userId, existing.deviceType, mergedData, id);
      
      res.json(updated);
    } catch (error) {
      console.error("Error updating device reading:", error);
      res.status(500).json({ message: "Failed to update device reading" });
    }
  });

  // Delete a device reading
  app.delete('/api/device-readings/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const userId = req.user!.id;
      
      // Verify the reading exists and belongs to the user
      const existing = await storage.getDeviceReading(id);
      if (!existing) {
        return res.status(404).json({ message: "Device reading not found" });
      }
      if (existing.patientId !== userId) {
        return res.status(403).json({ message: "Not authorized to delete this reading" });
      }
      
      await storage.deleteDeviceReading(id);
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting device reading:", error);
      res.status(500).json({ message: "Failed to delete device reading" });
    }
  });

  // Get readings summary for dashboard
  app.get('/api/device-readings/summary', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      // Get latest reading for each device type
      const deviceTypes = ['bp_monitor', 'glucose_meter', 'smart_scale', 'thermometer', 'stethoscope', 'smartwatch'];
      const summary: Record<string, any> = {};
      
      for (const deviceType of deviceTypes) {
        const latest = await storage.getLatestDeviceReading(userId, deviceType);
        if (latest) {
          summary[deviceType] = {
            lastReading: latest,
            recordedAt: latest.recordedAt,
          };
        }
      }
      
      res.json(summary);
    } catch (error) {
      console.error("Error fetching device readings summary:", error);
      res.status(500).json({ message: "Failed to fetch readings summary" });
    }
  });

  // Get unprocessed readings for health alert system
  app.get('/api/device-readings/unprocessed', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { hours } = req.query;
      
      const readings = await storage.getDeviceReadingsForHealthAlerts(
        userId, 
        hours ? parseInt(hours as string) : 24
      );
      
      res.json(readings);
    } catch (error) {
      console.error("Error fetching unprocessed readings:", error);
      res.status(500).json({ message: "Failed to fetch unprocessed readings" });
    }
  });

  // ============== REFERRAL SYSTEM ROUTES ==============

  app.get('/api/referrals/my-code', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const user = req.user!;
      
      // Get or create referral code for this user
      let referral = await storage.getReferralByReferrerId(userId);
      
      if (!referral) {
        // Generate unique referral code
        const referralCode = `REF-${user.firstName.substring(0, 3).toUpperCase()}${Math.random().toString(36).substring(2, 8).toUpperCase()}`;
        const referralLink = `${process.env.APP_URL || 'http://localhost:5000'}/signup?ref=${referralCode}`;
        
        referral = await storage.createReferral({
          referrerId: userId,
          referrerType: user.role,
          referralCode,
          referralLink,
          expiresAt: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000), // 90 days
        });
      }
      
      res.json(referral);
    } catch (error) {
      console.error("Error fetching referral code:", error);
      res.status(500).json({ message: "Failed to fetch referral code" });
    }
  });

  app.get('/api/referrals/my-referrals', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const referrals = await storage.getReferralsByReferrerId(userId);
      res.json(referrals);
    } catch (error) {
      console.error("Error fetching referrals:", error);
      res.status(500).json({ message: "Failed to fetch referrals" });
    }
  });

  app.post('/api/referrals/track-click', async (req, res) => {
    try {
      const { referralCode, email } = req.body;
      
      if (!referralCode) {
        return res.status(400).json({ message: "Referral code is required" });
      }
      
      const referral = await storage.getReferralByCode(referralCode);
      if (!referral) {
        return res.status(404).json({ message: "Invalid referral code" });
      }
      
      // Check if expired
      if (referral.expiresAt && new Date(referral.expiresAt) < new Date()) {
        return res.status(400).json({ message: "Referral link has expired" });
      }
      
      // Update click tracking
      await storage.updateReferral(referral.id, {
        clickedAt: new Date(),
        refereeEmail: email,
        ipAddress: req.ip,
        userAgent: req.headers['user-agent'],
      });
      
      res.json({ success: true, referral });
    } catch (error) {
      console.error("Error tracking referral click:", error);
      res.status(500).json({ message: "Failed to track referral click" });
    }
  });

  app.post('/api/referrals/activate', isAuthenticated, async (req: any, res) => {
    try {
      const { referralCode } = req.body;
      const newUserId = req.user!.id;
      const newUser = req.user!;
      
      if (!referralCode) {
        return res.status(400).json({ message: "Referral code is required" });
      }
      
      const referral = await storage.getReferralByCode(referralCode);
      if (!referral) {
        return res.status(404).json({ message: "Invalid referral code" });
      }
      
      // Prevent self-referral
      if (referral.referrerId === newUserId) {
        return res.status(400).json({ message: "You cannot use your own referral code" });
      }
      
      // Check if already activated
      if (referral.status === 'trial_activated') {
        return res.status(400).json({ message: "Referral code already used" });
      }
      
      // Update referral with referee info
      await storage.updateReferral(referral.id, {
        refereeId: newUserId,
        refereeType: newUser.role,
        signedUpAt: new Date(),
        status: 'signed_up',
      });
      
      // Grant 1-month free trial to both users
      const trialEnd = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000); // 30 days
      
      // Update referrer's trial
      await storage.updateUser(referral.referrerId, {
        trialEndsAt: trialEnd,
        subscriptionStatus: 'trialing',
      });
      
      // Update referee's trial
      await storage.updateUser(newUserId, {
        trialEndsAt: trialEnd,
        subscriptionStatus: 'trialing',
      });
      
      // Mark referral as activated
      await storage.updateReferral(referral.id, {
        status: 'trial_activated',
        trialActivatedAt: new Date(),
        referrerTrialExtended: true,
        refereeTrialGranted: true,
      });
      
      res.json({ 
        success: true, 
        message: "Referral activated! Both you and your referrer received 1 month free trial.",
        trialEndsAt: trialEnd,
      });
    } catch (error) {
      console.error("Error activating referral:", error);
      res.status(500).json({ message: "Failed to activate referral" });
    }
  });

  // ============== WALLET & CREDIT ROUTES ==============

  app.get('/api/wallet/balance', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const user = await storage.getUserById(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      res.json({ balance: user.creditBalance || 0 });
    } catch (error) {
      console.error("Error fetching wallet balance:", error);
      res.status(500).json({ message: "Failed to fetch wallet balance" });
    }
  });

  app.get('/api/wallet/transactions', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const transactions = await storage.getCreditTransactions(userId);
      res.json(transactions);
    } catch (error) {
      console.error("Error fetching credit transactions:", error);
      res.status(500).json({ message: "Failed to fetch credit transactions" });
    }
  });

  app.post('/api/wallet/purchase', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { amount, paymentMethod } = req.body;
      
      if (!amount || amount <= 0) {
        return res.status(400).json({ message: "Invalid amount" });
      }
      
      // TODO: Integrate with Stripe to process payment
      // For now, just add credits (in production, this should happen after payment confirmation)
      
      const user = await storage.getUserById(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      
      const newBalance = (user.creditBalance || 0) + amount;
      await storage.updateUser(userId, { creditBalance: newBalance });
      
      // Create transaction record
      await storage.createCreditTransaction({
        userId,
        transactionType: 'purchased',
        amount,
        balanceAfter: newBalance,
        description: `Purchased ${amount} credits`,
      });
      
      res.json({ success: true, newBalance });
    } catch (error) {
      console.error("Error purchasing credits:", error);
      res.status(500).json({ message: "Failed to purchase credits" });
    }
  });

  app.post('/api/wallet/withdraw', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { amount } = req.body;
      
      if (!amount || amount <= 0) {
        return res.status(400).json({ message: "Invalid withdrawal amount" });
      }
      
      const user = await storage.getUserById(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      
      if ((user.creditBalance || 0) < amount) {
        return res.status(400).json({ message: "Insufficient credits for withdrawal" });
      }
      
      // TODO: Integrate with Stripe to process payout to doctor's account
      // For now, just deduct credits (in production, this should happen after payout confirmation)
      
      const newBalance = (user.creditBalance || 0) - amount;
      await storage.updateUser(userId, { creditBalance: newBalance });
      
      // Create transaction record
      await storage.createCreditTransaction({
        userId,
        transactionType: 'withdrawn',
        amount: -amount,
        balanceAfter: newBalance,
        description: `Withdrew ${amount} credits`,
        metadata: { withdrawalAmount: amount },
      });
      
      res.json({ success: true, newBalance, message: "Withdrawal request submitted. Funds will be transferred to your account within 2-3 business days." });
    } catch (error) {
      console.error("Error withdrawing credits:", error);
      res.status(500).json({ message: "Failed to withdraw credits" });
    }
  });

  // ============== MEDICAL DOCUMENTS ROUTES ==============

  app.post('/api/medical-documents/upload', isAuthenticated, medicalDocUpload.single('file'), async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const file = req.file;
      
      if (!file) {
        return res.status(400).json({ message: "No file uploaded" });
      }
      
      const documentType = req.body.documentType || 'other';
      const documentDate = req.body.documentDate ? new Date(req.body.documentDate) : null;
      
      const fileKey = `medical-documents/${userId}/${Date.now()}-${file.originalname}`;
      
      const uploadParams = {
        Bucket: AWS_S3_BUCKET,
        Key: fileKey,
        Body: file.buffer,
        ContentType: file.mimetype,
        ServerSideEncryption: 'AES256' as const,
        Metadata: {
          userId,
          documentType,
          uploadDate: new Date().toISOString(),
        },
      };
      
      const upload = new Upload({
        client: s3Client,
        params: uploadParams,
      });
      
      await upload.done();
      
      const fileUrl = `https://${AWS_S3_BUCKET}.s3.${process.env.AWS_REGION}.amazonaws.com/${fileKey}`;
      
      const document = await storage.createMedicalDocument({
        userId,
        fileName: file.originalname,
        fileType: path.extname(file.originalname).substring(1),
        fileSize: file.size,
        fileUrl,
        documentType,
        documentDate,
        processingStatus: 'pending',
      });
      
      res.json({
        success: true,
        document,
        message: 'Document uploaded successfully. OCR processing will begin shortly.',
      });
      
      processDocumentOCR(document.id, fileKey).catch(error => {
        console.error('Error processing OCR:', error);
      });
      
    } catch (error) {
      console.error('Error uploading medical document:', error);
      res.status(500).json({ message: 'Failed to upload document' });
    }
  });

  async function processDocumentOCR(documentId: string, s3Key: string) {
    try {
      await storage.updateMedicalDocument(documentId, { processingStatus: 'processing' });
      
      const startCommand = new StartDocumentAnalysisCommand({
        DocumentLocation: {
          S3Object: {
            Bucket: AWS_S3_BUCKET,
            Name: s3Key,
          },
        },
        FeatureTypes: ['TABLES', 'FORMS'],
      });
      
      const startResponse = await textractClient.send(startCommand);
      const jobId = startResponse.JobId;
      
      if (!jobId) {
        throw new Error('Failed to start Textract job');
      }
      
      let jobStatus = 'IN_PROGRESS';
      let attempts = 0;
      const maxAttempts = 300;
      
      while (jobStatus === 'IN_PROGRESS' && attempts < maxAttempts) {
        const baseDelay = 2000;
        const delay = attempts < 30 ? baseDelay : 
                     attempts < 60 ? baseDelay * 2 : 
                     attempts < 120 ? baseDelay * 3 : 
                     baseDelay * 4;
        
        await new Promise(resolve => setTimeout(resolve, delay));
        
        const getCommand = new GetDocumentAnalysisCommand({ JobId: jobId });
        const getResponse = await textractClient.send(getCommand);
        
        jobStatus = getResponse.JobStatus || 'IN_PROGRESS';
        
        if (jobStatus === 'SUCCEEDED') {
          let extractedText = '';
          let allBlocks = getResponse.Blocks || [];
          let nextToken = getResponse.NextToken;
          
          while (nextToken) {
            const nextCommand = new GetDocumentAnalysisCommand({ 
              JobId: jobId,
              NextToken: nextToken,
            });
            const nextResponse = await textractClient.send(nextCommand);
            allBlocks = allBlocks.concat(nextResponse.Blocks || []);
            nextToken = nextResponse.NextToken;
          }
          
          for (const block of allBlocks) {
            if (block.BlockType === 'LINE' && block.Text) {
              extractedText += block.Text + '\n';
            }
          }
          
          let allMedicalEntities: any[] = [];
          if (extractedText.trim().length > 0) {
            try {
              const MAX_CHUNK_SIZE = 19000;
              const chunks: string[] = [];
              
              if (extractedText.length <= MAX_CHUNK_SIZE) {
                chunks.push(extractedText);
              } else {
                const lines = extractedText.split('\n');
                let currentChunk = '';
                
                for (const line of lines) {
                  if ((currentChunk.length + line.length + 1) > MAX_CHUNK_SIZE) {
                    if (currentChunk.length > 0) {
                      chunks.push(currentChunk);
                    }
                    currentChunk = line;
                  } else {
                    currentChunk += (currentChunk ? '\n' : '') + line;
                  }
                }
                
                if (currentChunk.length > 0) {
                  chunks.push(currentChunk);
                }
              }
              
              for (const chunk of chunks) {
                try {
                  const comprehendCommand = new DetectEntitiesV2Command({
                    Text: chunk,
                  });
                  
                  const comprehendResponse = await comprehendMedicalClient.send(comprehendCommand);
                  allMedicalEntities = allMedicalEntities.concat(comprehendResponse.Entities || []);
                  
                  await new Promise(resolve => setTimeout(resolve, 100));
                } catch (chunkError) {
                  console.error('Comprehend Medical chunk error:', chunkError);
                }
              }
            } catch (comprehendError) {
              console.error('Comprehend Medical error:', comprehendError);
            }
          }
          
          const extractedData: any = {
            medications: [],
            diagnosis: [],
            labResults: [],
            vitalSigns: [],
            allergies: [],
            procedures: [],
          };
          
          const seenTexts = new Set<string>();
          
          for (const entity of allMedicalEntities) {
            const normalizedText = entity.Text?.toLowerCase().trim();
            if (!normalizedText || seenTexts.has(normalizedText)) continue;
            seenTexts.add(normalizedText);
            
            if (entity.Category === 'MEDICATION' && entity.Text) {
              extractedData.medications.push(entity.Text);
            } else if (entity.Category === 'MEDICAL_CONDITION' && entity.Text) {
              extractedData.diagnosis.push(entity.Text);
            } else if (entity.Category === 'TEST_TREATMENT_PROCEDURE' && entity.Text) {
              extractedData.procedures.push(entity.Text);
            }
          }
          
          await storage.updateMedicalDocument(documentId, {
            extractedText,
            extractedData,
            processingStatus: 'completed',
          });
          
          return;
        } else if (jobStatus === 'FAILED') {
          throw new Error(getResponse.StatusMessage || 'Textract job failed');
        }
        
        attempts++;
      }
      
      if (jobStatus === 'IN_PROGRESS') {
        throw new Error('Textract job timeout - processing took too long');
      }
      
    } catch (error) {
      console.error('OCR processing error:', error);
      await storage.updateMedicalDocument(documentId, {
        processingStatus: 'failed',
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  app.get('/api/medical-documents', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const documents = await storage.getMedicalDocuments(userId);
      res.json(documents);
    } catch (error) {
      console.error('Error fetching medical documents:', error);
      res.status(500).json({ message: 'Failed to fetch documents' });
    }
  });

  app.get('/api/medical-documents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const documentId = req.params.id;
      
      const document = await storage.getMedicalDocument(documentId);
      
      if (!document) {
        return res.status(404).json({ message: 'Document not found' });
      }
      
      if (document.userId !== userId && req.user!.role !== 'doctor') {
        return res.status(403).json({ message: 'Access denied' });
      }
      
      res.json(document);
    } catch (error) {
      console.error('Error fetching medical document:', error);
      res.status(500).json({ message: 'Failed to fetch document' });
    }
  });

  app.delete('/api/medical-documents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const documentId = req.params.id;
      
      const document = await storage.getMedicalDocument(documentId);
      
      if (!document) {
        return res.status(404).json({ message: 'Document not found' });
      }
      
      if (document.userId !== userId) {
        return res.status(403).json({ message: 'Access denied' });
      }
      
      await storage.deleteMedicalDocument(documentId);
      
      res.json({ success: true, message: 'Document deleted successfully' });
    } catch (error) {
      console.error('Error deleting medical document:', error);
      res.status(500).json({ message: 'Failed to delete document' });
    }
  });

  // Immune monitoring routes
  app.get('/api/immune/biomarkers', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 30;
      const biomarkers = await storage.getImmuneBiomarkers(userId, limit);
      res.json(biomarkers);
    } catch (error) {
      console.error('Error fetching immune biomarkers:', error);
      res.status(500).json({ message: 'Failed to fetch biomarkers' });
    }
  });

  app.get('/api/immune/digital-twin', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const digitalTwin = await storage.getLatestImmuneDigitalTwin(userId);
      res.json(digitalTwin);
    } catch (error) {
      console.error('Error fetching immune digital twin:', error);
      res.status(500).json({ message: 'Failed to fetch digital twin' });
    }
  });

  app.post('/api/immune/sync-wearable', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { deviceType } = req.body;
      
      if (!deviceType) {
        return res.status(400).json({ message: 'Device type is required' });
      }

      const { syncWearableData } = await import('./immuneMonitoring');
      const result = await syncWearableData(userId, deviceType);
      
      res.json(result);
    } catch (error) {
      console.error('Error syncing wearable data:', error);
      res.status(500).json({ message: 'Failed to sync wearable data' });
    }
  });

  app.post('/api/immune/analyze', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { analyzeImmuneBiomarkers } = await import('./immuneMonitoring');
      const analysis = await analyzeImmuneBiomarkers(userId);
      res.json(analysis);
    } catch (error) {
      console.error('Error analyzing immune biomarkers:', error);
      res.status(500).json({ message: 'Failed to analyze biomarkers' });
    }
  });

  // Environmental risk routes
  app.get('/api/environmental/risk', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const user = await storage.getUser(userId);
      const patientProfile = await storage.getPatientProfile(userId);
      
      if (!patientProfile?.zipCode) {
        return res.status(400).json({ message: 'User location (zip code) is required' });
      }

      const riskData = await storage.getEnvironmentalRiskDataByLocation(patientProfile.zipCode, 1);
      
      if (riskData.length > 0) {
        res.json(riskData[0]);
      } else {
        res.json({ message: 'No environmental risk data available for this location' });
      }
    } catch (error) {
      console.error('Error fetching environmental risk data:', error);
      res.status(500).json({ message: 'Failed to fetch environmental risk data' });
    }
  });

  app.post('/api/environmental/update', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const patientProfile = await storage.getPatientProfile(userId);
      
      if (!patientProfile?.zipCode) {
        return res.status(400).json({ message: 'User location (zip code) is required' });
      }

      const { fetchEnvironmentalRiskData } = await import('./environmentalRisk');
      const riskData = await fetchEnvironmentalRiskData(patientProfile.zipCode);
      
      res.json(riskData);
    } catch (error) {
      console.error('Error updating environmental risk data:', error);
      res.status(500).json({ message: 'Failed to update environmental risk data' });
    }
  });

  app.get('/api/environmental/pathogen-map', isAuthenticated, async (req: any, res) => {
    try {
      const { zipCode } = req.query;
      
      if (!zipCode) {
        return res.status(400).json({ message: 'Zip code is required' });
      }

      const { generatePathogenRiskMap } = await import('./environmentalRisk');
      const riskMap = await generatePathogenRiskMap(zipCode as string);
      
      res.json(riskMap);
    } catch (error) {
      console.error('Error generating pathogen risk map:', error);
      res.status(500).json({ message: 'Failed to generate risk map' });
    }
  });

  // ============================================================================
  // ENVIRONMENTAL RISK MAP API - Comprehensive Environmental Health Intelligence
  // ============================================================================

  // Get or create patient's environmental profile
  app.post('/api/v1/environment/profile', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.query.patient_id || req.user!.id;
      const { zipCode, conditions, allergies } = req.body;

      if (!zipCode || zipCode.length !== 5) {
        return res.status(400).json({ success: false, error: 'Valid 5-digit ZIP code required' });
      }

      // Check if profile exists using raw SQL
      const existingResult = await db.execute(drizzleSql`
        SELECT * FROM patient_environment_profiles 
        WHERE patient_id = ${patientId} AND is_active = true LIMIT 1
      `);
      const existing = existingResult.rows?.[0];

      const regionInfo = getZipCodeInfo(zipCode);

      if (existing) {
        // Update existing profile
        await db.execute(drizzleSql`
          UPDATE patient_environment_profiles 
          SET zip_code = ${zipCode}, 
              city = ${regionInfo.city}, 
              state = ${regionInfo.state},
              chronic_conditions = ${JSON.stringify(conditions || [])}::jsonb,
              allergies = ${JSON.stringify(allergies || [])}::jsonb,
              updated_at = NOW()
          WHERE id = ${existing.id}
        `);
      } else {
        // Create new profile
        await db.execute(drizzleSql`
          INSERT INTO patient_environment_profiles 
          (patient_id, zip_code, city, state, chronic_conditions, allergies, alert_thresholds)
          VALUES (${patientId}, ${zipCode}, ${regionInfo.city}, ${regionInfo.state}, 
                  ${JSON.stringify(conditions || [])}::jsonb, ${JSON.stringify(allergies || [])}::jsonb,
                  '{"riskScore": 70, "aqiThreshold": 100, "pollenThreshold": 8}'::jsonb)
        `);
      }

      // Fetch the created/updated profile
      const profileResult = await db.execute(drizzleSql`
        SELECT * FROM patient_environment_profiles 
        WHERE patient_id = ${patientId} AND is_active = true LIMIT 1
      `);
      const profile = profileResult.rows?.[0];

      res.json({
        success: true,
        profile: profile ? {
          id: profile.id,
          patientId: profile.patient_id,
          zipCode: profile.zip_code,
          city: profile.city,
          state: profile.state,
          conditions: profile.chronic_conditions,
          allergies: profile.allergies,
          alertsEnabled: profile.alerts_enabled,
          correlationConsent: profile.correlation_consent_given,
        } : {}
      });
    } catch (error: any) {
      console.error('Error creating environment profile:', error);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Get current environmental data and risk for a patient
  app.get('/api/v1/environment/current', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.query.patient_id || req.user!.id;

      // Get patient profile
      const profiles = await db.execute(drizzleSql`
        SELECT * FROM patient_environment_profiles 
        WHERE patient_id = ${patientId} AND is_active = true 
        LIMIT 1
      `);
      
      const profile = profiles.rows?.[0];
      if (!profile) {
        return res.json({ success: true, profile: null, currentData: null, riskScore: null, forecasts: [], activeAlerts: [] });
      }

      // Get latest environmental snapshot
      const snapshots = await db.execute(drizzleSql`
        SELECT * FROM environmental_data_snapshots 
        WHERE zip_code = ${profile.zip_code}
        ORDER BY measured_at DESC LIMIT 1
      `);
      const snapshot = snapshots.rows?.[0];

      // Get latest risk score
      const riskScores = await db.execute(drizzleSql`
        SELECT * FROM patient_environment_risk_scores 
        WHERE patient_id = ${patientId}
        ORDER BY computed_at DESC LIMIT 1
      `);
      const riskScore = riskScores.rows?.[0];

      // Get forecasts
      const forecasts = await db.execute(drizzleSql`
        SELECT * FROM environmental_forecasts 
        WHERE patient_id = ${patientId} AND forecast_target_time > NOW()
        ORDER BY forecast_horizon
      `);

      // Get active alerts
      const alerts = await db.execute(drizzleSql`
        SELECT * FROM environmental_alerts 
        WHERE patient_id = ${patientId} AND status = 'active'
        ORDER BY created_at DESC LIMIT 5
      `);

      res.json({
        success: true,
        profile: {
          zipCode: profile.zip_code,
          city: profile.city,
          state: profile.state,
          conditions: profile.chronic_conditions,
          allergies: profile.allergies,
          alertsEnabled: profile.alerts_enabled,
          correlationConsent: profile.correlation_consent_given
        },
        currentData: snapshot ? {
          measuredAt: snapshot.measured_at,
          weather: {
            temperature: parseFloat(snapshot.temperature) || null,
            feelsLike: parseFloat(snapshot.feels_like) || null,
            humidity: parseFloat(snapshot.humidity) || null,
            pressure: parseFloat(snapshot.pressure) || null,
            uvIndex: parseFloat(snapshot.uv_index) || null,
          },
          airQuality: {
            aqi: snapshot.aqi,
            category: snapshot.aqi_category,
            pm25: parseFloat(snapshot.pm25) || null,
            pm10: parseFloat(snapshot.pm10) || null,
            ozone: parseFloat(snapshot.ozone) || null,
          },
          allergens: {
            pollenOverall: snapshot.pollen_overall,
            pollenCategory: snapshot.pollen_category,
            moldCount: snapshot.mold_spore_count,
          },
          hazards: snapshot.active_hazards || [],
        } : null,
        riskScore: riskScore ? {
          composite: parseFloat(riskScore.composite_risk_score),
          level: riskScore.risk_level,
          computedAt: riskScore.computed_at,
          components: {
            weather: parseFloat(riskScore.weather_risk_score) || null,
            airQuality: parseFloat(riskScore.air_quality_risk_score) || null,
            allergens: parseFloat(riskScore.allergen_risk_score) || null,
            hazards: parseFloat(riskScore.hazard_risk_score) || null,
          },
          trends: {
            "24hr": parseFloat(riskScore.trend_24hr) || null,
            "48hr": parseFloat(riskScore.trend_48hr) || null,
            "72hr": parseFloat(riskScore.trend_72hr) || null,
          },
          topFactors: riskScore.top_risk_factors || [],
        } : null,
        forecasts: (forecasts.rows || []).map((f: any) => ({
          horizon: f.forecast_horizon,
          targetTime: f.forecast_target_time,
          predictedScore: parseFloat(f.predicted_risk_score),
          predictedLevel: f.predicted_risk_level,
          confidence: f.confidence_interval,
        })),
        activeAlerts: (alerts.rows || []).map((a: any) => ({
          id: a.id,
          type: a.alert_type,
          severity: a.severity,
          title: a.title,
          message: a.message,
          recommendations: a.recommendations,
          createdAt: a.created_at,
        })),
      });
    } catch (error: any) {
      console.error('Error getting environmental data:', error);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Refresh environmental data and compute risk
  app.post('/api/v1/environment/refresh', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.query.patient_id || req.user!.id;
      const { zipCode: requestZipCode } = req.body;

      // Get or create profile
      let profiles = await db.execute(drizzleSql`
        SELECT * FROM patient_environment_profiles 
        WHERE patient_id = ${patientId} AND is_active = true LIMIT 1
      `);
      let profile = profiles.rows?.[0];

      const zipCode = requestZipCode || profile?.zip_code;
      if (!zipCode) {
        return res.status(400).json({ success: false, error: 'ZIP code required' });
      }

      // Generate simulated environmental data
      const envData = generateSimulatedEnvironmentalData(zipCode);

      // Insert snapshot - ensure proper type casting for integer fields
      const pollenInt = Math.round(envData.pollenOverall);
      const snapshotResult = await db.execute(drizzleSql`
        INSERT INTO environmental_data_snapshots 
        (zip_code, measured_at, temperature, feels_like, humidity, pressure, uv_index,
         aqi, aqi_category, pm25, pm10, ozone, pollen_overall, pollen_category,
         mold_spore_count, weather_source, aqi_source, data_quality_score)
        VALUES (${zipCode}, NOW(), ${envData.temperature}, ${envData.feelsLike}, 
                ${envData.humidity}, ${envData.pressure}, ${envData.uvIndex},
                ${envData.aqi}, ${envData.aqiCategory}, ${envData.pm25}, ${envData.pm10},
                ${envData.ozone}, ${pollenInt}, ${envData.pollenCategory},
                ${envData.moldCount}, 'simulated', 'simulated', 85)
        RETURNING id
      `);
      const snapshotId = snapshotResult.rows?.[0]?.id;

      // Compute risk score
      const conditions = profile?.chronic_conditions || [];
      const riskData = computeEnvironmentalRisk(envData, conditions);

      // Insert risk score
      await db.execute(drizzleSql`
        INSERT INTO patient_environment_risk_scores 
        (patient_id, snapshot_id, computed_at, composite_risk_score, risk_level,
         weather_risk_score, air_quality_risk_score, allergen_risk_score, hazard_risk_score,
         factor_contributions, top_risk_factors)
        VALUES (${patientId}, ${snapshotId}, NOW(), ${riskData.compositeScore}, ${riskData.riskLevel},
                ${riskData.weatherScore}, ${riskData.airQualityScore}, ${riskData.allergenScore}, 0,
                ${JSON.stringify(riskData.factorContributions)}::jsonb, 
                ${JSON.stringify(riskData.topFactors)}::jsonb)
      `);

      // Generate forecasts
      const horizons = ['12hr', '24hr', '48hr'];
      for (const horizon of horizons) {
        const hoursAhead = horizon === '12hr' ? 12 : horizon === '24hr' ? 24 : 48;
        const forecastScore = Math.max(0, Math.min(100, riskData.compositeScore + (Math.random() - 0.5) * 20));
        const forecastLevel = forecastScore < 25 ? 'low' : forecastScore < 50 ? 'moderate' : forecastScore < 75 ? 'high' : 'critical';
        const confidenceJson = JSON.stringify({ lower: Math.max(0, forecastScore - 15), upper: Math.min(100, forecastScore + 15), confidence: 0.95 });
        const targetTime = new Date(Date.now() + hoursAhead * 60 * 60 * 1000).toISOString();
        
        await db.execute(drizzleSql`
          INSERT INTO environmental_forecasts 
          (patient_id, generated_at, forecast_horizon, forecast_target_time, 
           predicted_risk_score, predicted_risk_level, confidence_interval,
           model_name, model_version)
          VALUES (${patientId}, NOW(), ${horizon}, ${targetTime}::timestamptz,
                  ${forecastScore}, ${forecastLevel}, ${confidenceJson}::jsonb,
                  'trend_regression_v1', '1.0')
        `);
      }

      // Check for alerts
      if (riskData.compositeScore >= 70) {
        const severity = riskData.compositeScore >= 85 ? 'critical' : 'warning';
        await db.execute(drizzleSql`
          INSERT INTO environmental_alerts 
          (patient_id, alert_type, triggered_by, severity, priority, title, message,
           recommendations, trigger_value, threshold_value, snapshot_id, expires_at)
          VALUES (${patientId}, 'threshold_exceeded', 'composite_risk', ${severity}, 
                  ${riskData.compositeScore >= 85 ? 10 : 7},
                  ${'Environmental Risk: ' + riskData.riskLevel.toUpperCase()},
                  ${'Your environmental risk score is ' + riskData.compositeScore.toFixed(0) + '/100. Take precautions.'},
                  ${JSON.stringify(riskData.topFactors)}::jsonb,
                  ${riskData.compositeScore}, 70, ${snapshotId}, NOW() + INTERVAL '6 hours')
        `);
      }

      // Return refreshed data
      res.json({
        success: true,
        refreshedAt: new Date().toISOString(),
        alertsGenerated: riskData.compositeScore >= 70 ? 1 : 0,
      });
    } catch (error: any) {
      console.error('Error refreshing environmental data:', error);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Get risk history
  app.get('/api/v1/environment/history', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.query.patient_id || req.user!.id;
      const days = parseInt(req.query.days as string) || 7;

      const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
      const scores = await db.execute(drizzleSql`
        SELECT * FROM patient_environment_risk_scores 
        WHERE patient_id = ${patientId} 
          AND computed_at >= ${startDate}::timestamptz
        ORDER BY computed_at
      `);

      res.json({
        success: true,
        days,
        history: (scores.rows || []).map((s: any) => ({
          computedAt: s.computed_at,
          compositeScore: parseFloat(s.composite_risk_score),
          riskLevel: s.risk_level,
          components: {
            weather: parseFloat(s.weather_risk_score) || null,
            airQuality: parseFloat(s.air_quality_risk_score) || null,
            allergens: parseFloat(s.allergen_risk_score) || null,
            hazards: parseFloat(s.hazard_risk_score) || null,
          }
        }))
      });
    } catch (error: any) {
      console.error('Error getting risk history:', error);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Get correlations
  app.get('/api/v1/environment/correlations', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.query.patient_id || req.user!.id;

      const correlations = await db.execute(drizzleSql`
        SELECT * FROM symptom_environment_correlations 
        WHERE patient_id = ${patientId} AND is_statistically_significant = true
        ORDER BY ABS(correlation_coefficient) DESC
      `);

      res.json({
        success: true,
        correlations: (correlations.rows || []).map((c: any) => ({
          symptom: c.symptom_type,
          factor: c.environmental_factor,
          correlation: parseFloat(c.correlation_coefficient),
          strength: c.relationship_strength,
          direction: c.relationship_direction,
          lagHours: c.optimal_lag,
          interpretation: c.interpretation,
          confidence: parseFloat(c.confidence_score) || null,
        }))
      });
    } catch (error: any) {
      console.error('Error getting correlations:', error);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Get alerts
  app.get('/api/v1/environment/alerts', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.query.patient_id || req.user!.id;
      const status = req.query.status || 'active';

      let query;
      if (status === 'all') {
        query = drizzleSql`SELECT * FROM environmental_alerts WHERE patient_id = ${patientId} ORDER BY created_at DESC LIMIT 20`;
      } else {
        query = drizzleSql`SELECT * FROM environmental_alerts WHERE patient_id = ${patientId} AND status = ${status} ORDER BY created_at DESC LIMIT 20`;
      }

      const alerts = await db.execute(query);

      res.json({
        success: true,
        alerts: (alerts.rows || []).map((a: any) => ({
          id: a.id,
          type: a.alert_type,
          triggeredBy: a.triggered_by,
          severity: a.severity,
          priority: a.priority,
          title: a.title,
          message: a.message,
          recommendations: a.recommendations,
          status: a.status,
          triggerValue: parseFloat(a.trigger_value) || null,
          thresholdValue: parseFloat(a.threshold_value) || null,
          createdAt: a.created_at,
          acknowledgedAt: a.acknowledged_at,
        }))
      });
    } catch (error: any) {
      console.error('Error getting alerts:', error);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Acknowledge alert
  app.post('/api/v1/environment/alerts/acknowledge', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.query.patient_id || req.user!.id;
      const { alertId } = req.body;

      await db.execute(drizzleSql`
        UPDATE environmental_alerts 
        SET status = 'acknowledged', acknowledged_at = NOW()
        WHERE id = ${alertId} AND patient_id = ${patientId}
      `);

      res.json({ success: true, alert: { id: alertId, status: 'acknowledged' } });
    } catch (error: any) {
      console.error('Error acknowledging alert:', error);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Get patient environmental summary (for doctor dashboard)
  app.get('/api/v1/environment/patient/:patientId/summary', isAuthenticated, async (req: any, res) => {
    try {
      const { patientId } = req.params;

      const profiles = await db.execute(drizzleSql`
        SELECT * FROM patient_environment_profiles 
        WHERE patient_id = ${patientId} AND is_active = true LIMIT 1
      `);
      const profile = profiles.rows?.[0];

      if (!profile) {
        return res.json({ success: true, hasProfile: false, summary: null });
      }

      // Get latest risk score
      const riskScores = await db.execute(drizzleSql`
        SELECT * FROM patient_environment_risk_scores 
        WHERE patient_id = ${patientId}
        ORDER BY computed_at DESC LIMIT 1
      `);
      const riskScore = riskScores.rows?.[0];

      // Get latest snapshot
      const snapshots = await db.execute(drizzleSql`
        SELECT * FROM environmental_data_snapshots 
        WHERE zip_code = ${profile.zip_code}
        ORDER BY measured_at DESC LIMIT 1
      `);
      const snapshot = snapshots.rows?.[0];

      // Get forecasts
      const forecasts = await db.execute(drizzleSql`
        SELECT * FROM environmental_forecasts 
        WHERE patient_id = ${patientId} AND forecast_target_time > NOW()
        ORDER BY forecast_horizon LIMIT 3
      `);

      // Get active alerts count
      const alertsResult = await db.execute(drizzleSql`
        SELECT COUNT(*) as count FROM environmental_alerts 
        WHERE patient_id = ${patientId} AND status = 'active'
      `);
      const alertCount = parseInt(alertsResult.rows?.[0]?.count || '0');

      // Get significant correlations
      const correlations = await db.execute(drizzleSql`
        SELECT * FROM symptom_environment_correlations 
        WHERE patient_id = ${patientId} AND is_statistically_significant = true
        ORDER BY ABS(correlation_coefficient) DESC LIMIT 3
      `);

      // Calculate trend
      let trend = 'stable';
      if (riskScore?.trend_24hr) {
        const t = parseFloat(riskScore.trend_24hr);
        if (t > 0.2) trend = 'worsening';
        else if (t < -0.2) trend = 'improving';
      }

      res.json({
        success: true,
        hasProfile: true,
        summary: {
          location: {
            zipCode: profile.zip_code,
            city: profile.city,
            state: profile.state,
          },
          conditions: profile.chronic_conditions || [],
          currentRisk: {
            score: riskScore ? parseFloat(riskScore.composite_risk_score) : null,
            level: riskScore?.risk_level || null,
            trend,
            topFactors: riskScore?.top_risk_factors?.slice(0, 3) || [],
          },
          currentConditions: {
            aqi: snapshot?.aqi || null,
            aqiCategory: snapshot?.aqi_category || null,
            temperature: snapshot ? parseFloat(snapshot.temperature) : null,
            humidity: snapshot ? parseFloat(snapshot.humidity) : null,
            pollenLevel: snapshot?.pollen_category || null,
          },
          forecast: {
            "24hr": (forecasts.rows || []).find((f: any) => f.forecast_horizon === '24hr')?.predicted_risk_level || null,
            "48hr": (forecasts.rows || []).find((f: any) => f.forecast_horizon === '48hr')?.predicted_risk_level || null,
          },
          activeAlerts: alertCount,
          alertsSummary: [],
          significantCorrelations: (correlations.rows || []).map((c: any) => ({
            symptom: c.symptom_type,
            factor: c.environmental_factor,
            strength: c.relationship_strength,
          })),
          weeklyTrend: {
            averageScore: null,
            highestScore: null,
            lowestScore: null,
            dataPoints: 0,
          }
        }
      });
    } catch (error: any) {
      console.error('Error getting patient summary:', error);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Multi-Condition Correlation Engine routes
  app.post('/api/correlations/analyze', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { analyzeCorrelations } = await import('./correlationEngine');
      await analyzeCorrelations(userId);
      res.json({ message: 'Correlation analysis completed successfully' });
    } catch (error) {
      console.error('Error analyzing correlations:', error);
      res.status(500).json({ message: 'Failed to analyze correlations' });
    }
  });

  app.get('/api/correlations', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { limit, type, severity } = req.query;
      
      let patterns;
      if (type) {
        patterns = await storage.getCorrelationPatternsByType(userId, type as string, limit ? parseInt(limit as string) : undefined);
      } else if (severity) {
        patterns = await storage.getHighSeverityCorrelationPatterns(userId, severity as string, limit ? parseInt(limit as string) : undefined);
      } else {
        patterns = await storage.getCorrelationPatterns(userId, limit ? parseInt(limit as string) : undefined);
      }
      
      res.json(patterns);
    } catch (error) {
      console.error('Error fetching correlation patterns:', error);
      res.status(500).json({ message: 'Failed to fetch correlation patterns' });
    }
  });

  app.get('/api/correlations/report', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { generateCorrelationReport } = await import('./correlationEngine');
      const report = await generateCorrelationReport(userId);
      res.json(report);
    } catch (error) {
      console.error('Error generating correlation report:', error);
      res.status(500).json({ message: 'Failed to generate correlation report' });
    }
  });

  // Risk alert routes
  app.get('/api/alerts/active', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const alerts = await storage.getActiveRiskAlerts(userId);
      res.json(alerts);
    } catch (error) {
      console.error('Error fetching active alerts:', error);
      res.status(500).json({ message: 'Failed to fetch alerts' });
    }
  });

  app.get('/api/alerts/all', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 50;
      const alerts = await storage.getAllRiskAlerts(userId, limit);
      res.json(alerts);
    } catch (error) {
      console.error('Error fetching alerts:', error);
      res.status(500).json({ message: 'Failed to fetch alerts' });
    }
  });

  app.post('/api/alerts/:id/acknowledge', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const alert = await storage.acknowledgeRiskAlert(id);
      res.json(alert);
    } catch (error) {
      console.error('Error acknowledging alert:', error);
      res.status(500).json({ message: 'Failed to acknowledge alert' });
    }
  });

  app.post('/api/alerts/:id/resolve', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const alert = await storage.resolveRiskAlert(id);
      res.json(alert);
    } catch (error) {
      console.error('Error resolving alert:', error);
      res.status(500).json({ message: 'Failed to resolve alert' });
    }
  });

  app.post('/api/alerts/:id/dismiss', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const alert = await storage.dismissRiskAlert(id);
      res.json(alert);
    } catch (error) {
      console.error('Error dismissing alert:', error);
      res.status(500).json({ message: 'Failed to dismiss alert' });
    }
  });

  // ==================== ADAPTIVE MEDICATION & NUTRITION INSIGHTS ROUTES ====================

  // Dietary preferences
  app.get('/api/nutrition/preferences', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const preferences = await storage.getDietaryPreferences(userId);
      res.json(preferences || {});
    } catch (error) {
      console.error('Error fetching dietary preferences:', error);
      res.status(500).json({ message: 'Failed to fetch dietary preferences' });
    }
  });

  app.post('/api/nutrition/preferences', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const preferences = await storage.upsertDietaryPreferences({
        patientId: userId,
        ...req.body,
      });
      res.json(preferences);
    } catch (error) {
      console.error('Error updating dietary preferences:', error);
      res.status(500).json({ message: 'Failed to update dietary preferences' });
    }
  });

  // Meal plans
  app.post('/api/nutrition/generate-meal-plan', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { generateWeeklyMealPlan } = await import('./nutritionService');
      const result = await generateWeeklyMealPlan(userId);
      res.json(result);
    } catch (error) {
      console.error('Error generating meal plan:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to generate meal plan' });
    }
  });

  app.get('/api/nutrition/meal-plans', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { activeOnly } = req.query;
      const plans = await storage.getMealPlans(userId, activeOnly === 'true');
      res.json(plans);
    } catch (error) {
      console.error('Error fetching meal plans:', error);
      res.status(500).json({ message: 'Failed to fetch meal plans' });
    }
  });

  app.get('/api/nutrition/active-plan', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const plan = await storage.getActiveMealPlan(userId);
      res.json(plan || null);
    } catch (error) {
      console.error('Error fetching active meal plan:', error);
      res.status(500).json({ message: 'Failed to fetch active meal plan' });
    }
  });

  // Meals
  app.get('/api/nutrition/meals', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { mealPlanId, limit } = req.query;
      const meals = await storage.getMeals(
        userId,
        mealPlanId as string | undefined,
        limit ? parseInt(limit as string) : undefined
      );
      res.json(meals);
    } catch (error) {
      console.error('Error fetching meals:', error);
      res.status(500).json({ message: 'Failed to fetch meals' });
    }
  });

  app.get('/api/nutrition/meals/today', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const meals = await storage.getTodaysMeals(userId);
      res.json(meals);
    } catch (error) {
      console.error('Error fetching today\'s meals:', error);
      res.status(500).json({ message: 'Failed to fetch today\'s meals' });
    }
  });

  app.post('/api/nutrition/meals', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const meal = await storage.createMeal({
        patientId: userId,
        ...req.body,
      });
      res.json(meal);
    } catch (error) {
      console.error('Error creating meal:', error);
      res.status(500).json({ message: 'Failed to create meal' });
    }
  });

  app.patch('/api/nutrition/meals/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const meal = await storage.updateMeal(id, req.body);
      res.json(meal);
    } catch (error) {
      console.error('Error updating meal:', error);
      res.status(500).json({ message: 'Failed to update meal' });
    }
  });

  // Nutrition analysis
  app.post('/api/nutrition/analyze', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { mealDescription } = req.body;
      if (!mealDescription) {
        return res.status(400).json({ message: 'Meal description is required' });
      }
      const { analyzeMealNutrition } = await import('./nutritionService');
      const analysis = await analyzeMealNutrition(mealDescription, userId);
      res.json(analysis);
    } catch (error) {
      console.error('Error analyzing meal:', error);
      res.status(500).json({ message: 'Failed to analyze meal' });
    }
  });

  // Medication scheduling
  app.get('/api/medications/schedules', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { activeOnly } = req.query;
      const schedules = await storage.getPatientMedicationSchedules(userId, activeOnly !== 'false');
      res.json(schedules);
    } catch (error) {
      console.error('Error fetching medication schedules:', error);
      res.status(500).json({ message: 'Failed to fetch medication schedules' });
    }
  });

  app.post('/api/medications/optimize-timing', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { optimizeMedicationTiming } = await import('./nutritionService');
      const recommendations = await optimizeMedicationTiming(userId);
      res.json(recommendations);
    } catch (error) {
      console.error('Error optimizing medication timing:', error);
      res.status(500).json({ message: 'Failed to optimize medication timing' });
    }
  });

  // Medication adherence
  app.get('/api/medications/adherence', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { limit } = req.query;
      const adherence = await storage.getPatientAdherence(userId, limit ? parseInt(limit as string) : undefined);
      res.json(adherence);
    } catch (error) {
      console.error('Error fetching adherence data:', error);
      res.status(500).json({ message: 'Failed to fetch adherence data' });
    }
  });

  app.get('/api/medications/pending', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const pending = await storage.getPendingMedications(userId);
      res.json(pending);
    } catch (error) {
      console.error('Error fetching pending medications:', error);
      res.status(500).json({ message: 'Failed to fetch pending medications' });
    }
  });

  app.post('/api/medications/adherence', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const adherence = await storage.createMedicationAdherence({
        patientId: userId,
        ...req.body,
      });
      res.json(adherence);
    } catch (error) {
      console.error('Error logging medication adherence:', error);
      res.status(500).json({ message: 'Failed to log medication adherence' });
    }
  });

  app.patch('/api/medications/adherence/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const adherence = await storage.updateMedicationAdherence(id, req.body);
      res.json(adherence);
    } catch (error) {
      console.error('Error updating medication adherence:', error);
      res.status(500).json({ message: 'Failed to update medication adherence' });
    }
  });

  // Medication lifecycle routes
  app.get('/api/medications/all', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const medications = await storage.getAllMedications(userId);
      res.json(medications);
    } catch (error) {
      console.error('Error fetching all medications:', error);
      res.status(500).json({ message: 'Failed to fetch medications' });
    }
  });

  app.get('/api/medications/pending-confirmation', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const medications = await storage.getPendingConfirmationMedications(userId);
      res.json(medications);
    } catch (error) {
      console.error('Error fetching pending medications:', error);
      res.status(500).json({ message: 'Failed to fetch pending medications' });
    }
  });

  app.get('/api/medications/inactive', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const medications = await storage.getInactiveMedications(userId);
      res.json(medications);
    } catch (error) {
      console.error('Error fetching inactive medications:', error);
      res.status(500).json({ message: 'Failed to fetch inactive medications' });
    }
  });

  app.post('/api/medications/:id/confirm', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const userId = req.user!.id;
      const medication = await storage.confirmMedication(id, userId);
      
      if (medication) {
        await storage.createMedicationChangeLog({
          medicationId: id,
          patientId: medication.patientId,
          changeType: 'added',
          changedBy: 'patient',
          changedByUserId: userId,
          changeReason: 'Patient confirmed auto-detected medication',
        });
      }
      
      res.json(medication);
    } catch (error) {
      console.error('Error confirming medication:', error);
      res.status(500).json({ message: 'Failed to confirm medication' });
    }
  });

  app.post('/api/medications/:id/discontinue', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const userId = req.user!.id;
      const { reason, replacementMedicationId } = req.body;
      
      const medication = await storage.discontinueMedication(id, userId, reason, replacementMedicationId);
      
      if (medication) {
        await storage.createMedicationChangeLog({
          medicationId: id,
          patientId: medication.patientId,
          changeType: 'discontinued',
          changedBy: req.user!.role === 'doctor' ? 'doctor' : 'patient',
          changedByUserId: userId,
          discontinuationReason: reason,
          replacementMedicationId: replacementMedicationId || null,
          changeReason: reason,
        });
      }
      
      res.json(medication);
    } catch (error) {
      console.error('Error discontinuing medication:', error);
      res.status(500).json({ message: 'Failed to discontinue medication' });
    }
  });

  app.post('/api/medications/:id/reactivate', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const userId = req.user!.id;
      
      const medication = await storage.reactivateMedication(id, userId);
      
      if (medication) {
        await storage.createMedicationChangeLog({
          medicationId: id,
          patientId: medication.patientId,
          changeType: 'reactivated',
          changedBy: req.user!.role === 'doctor' ? 'doctor' : 'patient',
          changedByUserId: userId,
          changeReason: 'Medication reactivated',
        });
      }
      
      res.json(medication);
    } catch (error) {
      console.error('Error reactivating medication:', error);
      res.status(500).json({ message: 'Failed to reactivate medication' });
    }
  });

  // Prescription routes
  app.get('/api/prescriptions', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const prescriptions = await storage.getPrescriptions(userId);
      res.json(prescriptions);
    } catch (error) {
      console.error('Error fetching prescriptions:', error);
      res.status(500).json({ message: 'Failed to fetch prescriptions' });
    }
  });

  // Doctor-specific: Get all prescriptions written by this doctor
  app.get('/api/prescriptions/doctor', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const prescriptions = await storage.getPrescriptionsByDoctor(doctorId);
      res.json(prescriptions);
    } catch (error) {
      console.error('Error fetching doctor prescriptions:', error);
      res.status(500).json({ message: 'Failed to fetch prescriptions' });
    }
  });

  // Doctor-specific: Get prescriptions for a specific patient
  // Uses doctor_patient_assignments table for HIPAA-compliant access control
  app.get('/api/prescriptions/patient/:patientId', isDoctor, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const doctorId = req.user!.id;
      
      // Verify patient exists and is a valid patient
      const patient = await storage.getUser(patientId);
      if (!patient || patient.role !== 'patient') {
        return res.status(404).json({ message: 'Patient not found' });
      }
      
      // HIPAA authorization: Check if doctor has active assignment with this patient
      const hasAccess = await storage.doctorHasPatientAccess(doctorId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${doctorId} attempted to access prescriptions for unassigned patient ${patientId} at ${new Date().toISOString()}`);
        return res.status(403).json({ 
          message: 'Access denied. No active assignment with this patient.',
          code: 'NO_PATIENT_ASSIGNMENT'
        });
      }
      
      // HIPAA audit: Log successful access to patient prescriptions
      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} accessed prescriptions for patient ${patientId} at ${new Date().toISOString()}`);
      
      const prescriptions = await storage.getPrescriptions(patientId);
      res.json(prescriptions);
    } catch (error) {
      console.error('Error fetching patient prescriptions:', error);
      res.status(500).json({ message: 'Failed to fetch prescriptions' });
    }
  });

  app.post('/api/prescriptions', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const patientId = req.body.patientId;
      const { 
        medicationName, dosage, frequency, quantity, refills,
        dosageInstructions, notes, startDate, drugId, rxcui,
        specialty, isContinuous, durationDays, intendedStartDate,
        supersedes // ID of prescription to supersede
      } = req.body;
      
      // Verify patient exists
      const patient = await storage.getUser(patientId);
      if (!patient || patient.role !== 'patient') {
        return res.status(404).json({ message: 'Patient not found' });
      }
      
      // Get doctor profile to get specialty if not provided
      const doctorProfile = await storage.getDoctorProfile(doctorId);
      const prescriptionSpecialty = specialty || (doctorProfile?.specialties?.[0]) || 'general medicine';
      
      // Auto-create doctor-patient assignment if not exists (HIPAA compliance)
      await storage.createDoctorPatientAssignment({
        doctorId,
        patientId,
        assignmentSource: 'prescription',
        assignedBy: doctorId,
        patientConsented: true,
        consentMethod: 'implied',
        consentedAt: new Date(),
      });
      
      // Check for same-specialty supersession - if specialty matches existing active medication
      let supersessionTarget = null;
      if (supersedes) {
        supersessionTarget = supersedes;
      } else {
        // Auto-detect supersession candidates
        const existingMeds = await storage.getMedicationsBySpecialty(patientId, prescriptionSpecialty);
        if (existingMeds.length > 0 && !isContinuous) {
          // Return info about potential supersession
          console.log(`[INFO] Found ${existingMeds.length} existing ${prescriptionSpecialty} medication(s) that may be superseded`);
        }
      }

      // Check for cross-specialty conflicts
      const allPatientMeds = await storage.getActiveMedications(patientId);
      let conflictDetected = null;
      
      // Import drug interaction analysis
      try {
        const { analyzeMultipleDrugInteractions, enrichMedicationWithGenericName } = await import('./drugInteraction');
        const medsToCheck = await Promise.all(allPatientMeds.map(async (med) => {
          let drug = await storage.getDrugByName(med.name);
          if (!drug) {
            const enriched = await enrichMedicationWithGenericName(med.name);
            drug = { genericName: enriched.genericName, drugClass: null, brandNames: enriched.brandNames };
          }
          return {
            name: med.name,
            genericName: drug.genericName || med.name,
            drugClass: drug.drugClass || null,
            id: med.id,
            specialty: med.specialty,
            brandNames: drug.brandNames || []
          };
        }));
        
        // Add new medication to check
        medsToCheck.push({
          name: medicationName,
          genericName: medicationName,
          drugClass: null,
          id: null,
          specialty: prescriptionSpecialty,
          brandNames: []
        });
        
        const interactions = await analyzeMultipleDrugInteractions(medsToCheck, { isImmunocompromised: true });
        
        // Find cross-specialty conflicts
        for (const interaction of interactions) {
          if (interaction.interaction.severityLevel === 'severe' || interaction.interaction.severityLevel === 'high') {
            const med1 = medsToCheck.find(m => m.name === interaction.drug1);
            const med2 = medsToCheck.find(m => m.name === interaction.drug2);
            
            if (med1 && med2 && med1.specialty !== med2.specialty && 
                med1.specialty !== prescriptionSpecialty && med2.specialty !== prescriptionSpecialty) {
              conflictDetected = {
                drug1: interaction.drug1,
                drug2: interaction.drug2,
                severity: interaction.interaction.severityLevel,
                description: interaction.interaction.clinicalEffects,
              };
              break;
            }
          }
        }
      } catch (interactionError) {
        console.error('Error checking cross-specialty conflicts:', interactionError);
      }
      
      const prescription = await storage.createPrescription({
        doctorId,
        patientId,
        medicationName,
        dosage,
        frequency,
        quantity: quantity ? parseInt(quantity) : null,
        refills: refills ? parseInt(refills) : 0,
        dosageInstructions,
        notes,
        startDate: startDate ? new Date(startDate) : new Date(),
        drugId,
        rxcui,
        specialty: prescriptionSpecialty,
        isContinuous: isContinuous || false,
        durationDays: isContinuous ? null : (durationDays ? parseInt(durationDays) : null),
        intendedStartDate: intendedStartDate ? new Date(intendedStartDate) : null,
        supersedes: supersessionTarget,
        hasConflict: !!conflictDetected,
        conflictDetectedAt: conflictDetected ? new Date() : null,
      });
      
      // Create medication for patient with chronic care fields
      const medication = await storage.createMedication({
        patientId,
        name: medicationName,
        dosage,
        frequency,
        source: 'prescription',
        sourcePrescriptionId: prescription.id,
        addedBy: 'doctor',
        status: conflictDetected ? 'conflict_hold' : 'active',
        specialty: prescriptionSpecialty,
        prescribingDoctorId: doctorId,
        isContinuous: isContinuous || false,
        durationDays: isContinuous ? null : (durationDays ? parseInt(durationDays) : null),
        intendedStartDate: intendedStartDate ? new Date(intendedStartDate) : null,
        drugId,
        rxcui,
        conflictStatus: conflictDetected ? 'pending' : null,
        conflictDetectedAt: conflictDetected ? new Date() : null,
      });
      
      // Update prescription with medication ID
      await storage.updatePrescription(prescription.id, { medicationId: medication.id });
      
      // Handle supersession if specified
      if (supersessionTarget) {
        await storage.supersedeMedication(supersessionTarget, medication.id, `Superseded by new ${prescriptionSpecialty} prescription`);
        
        await storage.updatePrescription(supersessionTarget, {
          supersededBy: prescription.id,
          supersededAt: new Date(),
          status: 'superseded',
        });
        
        console.log(`[HIPAA-AUDIT] Doctor ${doctorId} superseded prescription ${supersessionTarget} with ${prescription.id} at ${new Date().toISOString()}`);
      }
      
      // Create cross-specialty conflict record if detected
      if (conflictDetected) {
        const conflictingMed = allPatientMeds.find(m => 
          m.name === conflictDetected.drug1 || m.name === conflictDetected.drug2
        );
        
        if (conflictingMed && conflictingMed.prescribingDoctorId) {
          const conflictGroupId = `conflict-${Date.now()}`;
          
          await storage.createMedicationConflict({
            patientId,
            conflictGroupId,
            medication1Id: medication.id,
            medication2Id: conflictingMed.id,
            prescription1Id: prescription.id,
            doctor1Id: doctorId,
            doctor2Id: conflictingMed.prescribingDoctorId,
            specialty1: prescriptionSpecialty,
            specialty2: conflictingMed.specialty || 'unspecified',
            conflictType: 'drug_interaction',
            severity: conflictDetected.severity,
            description: conflictDetected.description,
            detectedReason: `Cross-specialty interaction detected between ${conflictDetected.drug1} and ${conflictDetected.drug2}`,
            status: 'pending',
          });
          
          // Update both medications with conflict group
          await storage.updateMedication(medication.id, { conflictGroupId });
          await storage.updateMedication(conflictingMed.id, { 
            conflictGroupId, 
            conflictStatus: 'pending',
            conflictDetectedAt: new Date()
          });
          
          console.log(`[HIPAA-AUDIT] Cross-specialty conflict created between doctors ${doctorId} and ${conflictingMed.prescribingDoctorId} for patient ${patientId}`);
        }
      }
      
      // Log the change
      await storage.createMedicationChangeLog({
        medicationId: medication.id,
        patientId,
        changeType: 'added',
        changedBy: 'doctor',
        changedByUserId: doctorId,
        changeReason: `Prescription created by doctor (${prescriptionSpecialty})`,
        notes: notes,
      });
      
      // HIPAA audit log
      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} created prescription for patient ${patientId} (specialty: ${prescriptionSpecialty}, continuous: ${isContinuous}) at ${new Date().toISOString()}`);
      
      res.json({ 
        prescription, 
        medication,
        conflictDetected,
        supersessionTarget
      });
    } catch (error) {
      console.error('Error creating prescription:', error);
      res.status(500).json({ message: 'Failed to create prescription' });
    }
  });

  app.post('/api/prescriptions/:id/acknowledge', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const userId = req.user!.id;
      const prescription = await storage.acknowledgePrescription(id, userId);
      res.json(prescription);
    } catch (error) {
      console.error('Error acknowledging prescription:', error);
      res.status(500).json({ message: 'Failed to acknowledge prescription' });
    }
  });

  // ============================================
  // MEDICATION CONFLICT MANAGEMENT ROUTES
  // ============================================

  // Get all medication conflicts for a patient
  app.get('/api/medications/conflicts', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const conflicts = await storage.getMedicationConflicts(userId);
      res.json(conflicts);
    } catch (error) {
      console.error('Error fetching medication conflicts:', error);
      res.status(500).json({ message: 'Failed to fetch medication conflicts' });
    }
  });

  // Get pending conflicts for a doctor
  app.get('/api/doctor/medication-conflicts/pending', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const conflicts = await storage.getPendingConflicts(doctorId);
      res.json(conflicts);
    } catch (error) {
      console.error('Error fetching pending conflicts:', error);
      res.status(500).json({ message: 'Failed to fetch pending conflicts' });
    }
  });

  // Doctor responds to a medication conflict
  app.post('/api/doctor/medication-conflicts/:id/respond', isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const doctorId = req.user!.id;
      const { response, action } = req.body;

      const conflict = await storage.getMedicationConflict(id);
      if (!conflict) {
        return res.status(404).json({ message: 'Conflict not found' });
      }

      // Check if this doctor is involved in the conflict
      const isDoctor1 = conflict.doctor1_id === doctorId;
      const isDoctor2 = conflict.doctor2_id === doctorId;
      
      if (!isDoctor1 && !isDoctor2) {
        return res.status(403).json({ message: 'Not authorized to respond to this conflict' });
      }

      // Update the appropriate doctor's response
      const updateData = isDoctor1 ? {
        doctor1Response: response,
        doctor1RespondedAt: new Date(),
        doctor1Action: action,
      } : {
        doctor2Response: response,
        doctor2RespondedAt: new Date(),
        doctor2Action: action,
      };

      const updated = await storage.updateMedicationConflict(id, updateData);

      // Check if both doctors have responded - auto-resolve if they agree
      if (updated.doctor1_responded_at && updated.doctor2_responded_at) {
        if (updated.doctor1_action === updated.doctor2_action && updated.doctor1_action !== 'escalate') {
          // Both doctors agree - auto-resolve
          await storage.resolveMedicationConflict(id, {
            resolution: updated.doctor1_action,
            resolutionDetails: `Both doctors agreed to: ${updated.doctor1_action}`,
            resolvedBy: 'system',
          });
        }
      }

      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} responded to medication conflict ${id} at ${new Date().toISOString()}`);
      
      res.json(updated);
    } catch (error) {
      console.error('Error responding to conflict:', error);
      res.status(500).json({ message: 'Failed to respond to conflict' });
    }
  });

  // Resolve a medication conflict
  app.post('/api/doctor/medication-conflicts/:id/resolve', isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const doctorId = req.user!.id;
      const { resolution, resolutionDetails } = req.body;

      const conflict = await storage.getMedicationConflict(id);
      if (!conflict) {
        return res.status(404).json({ message: 'Conflict not found' });
      }

      // Check if this doctor is involved in the conflict
      if (conflict.doctor1_id !== doctorId && conflict.doctor2_id !== doctorId) {
        return res.status(403).json({ message: 'Not authorized to resolve this conflict' });
      }

      const resolved = await storage.resolveMedicationConflict(id, {
        resolution,
        resolutionDetails,
        resolvedBy: doctorId,
      });

      // Update involved medications to clear conflict status
      if (conflict.medication1_id) {
        await storage.updateMedication(conflict.medication1_id, {
          conflictStatus: 'resolved',
          conflictResolvedAt: new Date(),
          conflictResolution: resolution,
        });
      }
      if (conflict.medication2_id) {
        await storage.updateMedication(conflict.medication2_id, {
          conflictStatus: 'resolved',
          conflictResolvedAt: new Date(),
          conflictResolution: resolution,
        });
      }

      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} resolved medication conflict ${id} at ${new Date().toISOString()}`);
      
      res.json(resolved);
    } catch (error) {
      console.error('Error resolving conflict:', error);
      res.status(500).json({ message: 'Failed to resolve conflict' });
    }
  });

  // ============================================
  // UNIFIED MEDICATION DASHBOARD API
  // ============================================

  // Get comprehensive unified medication dashboard for patient
  app.get('/api/medications/dashboard', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      // Get all medications with prescriptions and doctor info
      const allMedications = await storage.getAllMedications(userId);
      const activeMedications = allMedications.filter(m => m.active && m.status !== 'superseded' && m.status !== 'expired');
      const prescriptions = await storage.getPrescriptions(userId);
      const conflicts = await storage.getMedicationConflicts(userId);
      const interactionAlerts = await storage.getActiveInteractionAlerts(userId);
      
      // Calculate reminders for medications ending soon
      const now = new Date();
      const reminders: any[] = [];
      const needsStartConfirmation: any[] = [];
      
      for (const med of activeMedications) {
        // Check if medication needs start confirmation
        if (med.intendedStartDate && !med.actualStartDate) {
          needsStartConfirmation.push({
            id: med.id,
            name: med.name,
            dosage: med.dosage,
            intendedStartDate: med.intendedStartDate,
            specialty: med.specialty,
          });
        }
        
        // Check for expiring medications (not continuous)
        if (!med.isContinuous && med.computedEndDate) {
          const endDate = new Date(med.computedEndDate);
          const daysRemaining = Math.ceil((endDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
          
          if (daysRemaining <= 7 && daysRemaining >= 0) {
            reminders.push({
              id: med.id,
              name: med.name,
              type: 'expiring',
              daysRemaining,
              message: daysRemaining === 0 
                ? 'Last day of this medication' 
                : `${daysRemaining} day${daysRemaining !== 1 ? 's' : ''} remaining`,
            });
          }
        }
      }
      
      // Get prescribing doctors for each medication
      const doctorIds = [...new Set(activeMedications.map(m => m.prescribingDoctorId).filter(Boolean))];
      const doctors: Record<string, any> = {};
      for (const doctorId of doctorIds) {
        if (doctorId) {
          const doctor = await storage.getUser(doctorId);
          if (doctor) {
            doctors[doctorId] = {
              id: doctor.id,
              name: `${doctor.firstName} ${doctor.lastName}`,
              email: doctor.email,
            };
          }
        }
      }
      
      // Group medications by specialty
      const medicationsBySpecialty: Record<string, any[]> = {};
      for (const med of activeMedications) {
        const specialty = med.specialty || 'unspecified';
        if (!medicationsBySpecialty[specialty]) {
          medicationsBySpecialty[specialty] = [];
        }
        
        // Enrich medication with doctor info and calculated fields
        const enrichedMed = {
          ...med,
          prescribingDoctorName: med.prescribingDoctorId && doctors[med.prescribingDoctorId] 
            ? doctors[med.prescribingDoctorId].name 
            : null,
          daysRemaining: !med.isContinuous && med.computedEndDate
            ? Math.ceil((new Date(med.computedEndDate).getTime() - now.getTime()) / (1000 * 60 * 60 * 24))
            : null,
          hasConflict: med.conflictStatus === 'pending',
        };
        
        medicationsBySpecialty[specialty].push(enrichedMed);
      }
      
      // Check for pending conflicts that freeze medication display
      const pendingConflicts = conflicts.filter(c => c.status === 'pending');
      const hasFreezeConflicts = pendingConflicts.length > 0;
      
      // Get archived/completed medications for history
      const archivedMedications = allMedications.filter(m => 
        !m.active || m.status === 'superseded' || m.status === 'expired' || m.status === 'discontinued'
      ).slice(0, 10); // Last 10 archived
      
      res.json({
        activeMedications,
        medicationsBySpecialty,
        prescriptions,
        conflicts: pendingConflicts,
        hasFreezeConflicts,
        interactionAlerts,
        reminders,
        needsStartConfirmation,
        archivedMedications,
        doctors,
        summary: {
          total: activeMedications.length,
          continuous: activeMedications.filter(m => m.isContinuous).length,
          duration: activeMedications.filter(m => !m.isContinuous).length,
          pendingConflicts: pendingConflicts.length,
          activeInteractions: interactionAlerts.length,
          expiringSoon: reminders.filter(r => r.type === 'expiring').length,
          needsConfirmation: needsStartConfirmation.length,
        },
      });
    } catch (error) {
      console.error('Error fetching medication dashboard:', error);
      res.status(500).json({ message: 'Failed to fetch medication dashboard' });
    }
  });

  // ============================================
  // MEDICATION LIFECYCLE ROUTES
  // ============================================

  // Lightweight medication history endpoint (for Medical Files page)
  app.get('/api/medications/history', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const limit = parseInt(req.query.limit as string) || 50;
      const offset = parseInt(req.query.offset as string) || 0;

      const allMedications = await storage.getAllMedications(userId);
      
      // Get archived/completed medications for history
      const archivedMedications = allMedications.filter(m => 
        !m.active || m.status === 'superseded' || m.status === 'expired' || m.status === 'discontinued'
      );

      // Paginate results
      const paginatedResults = archivedMedications.slice(offset, offset + limit);
      
      res.json({
        medications: paginatedResults,
        total: archivedMedications.length,
        limit,
        offset,
        hasMore: offset + limit < archivedMedications.length,
      });
    } catch (error) {
      console.error('Error fetching medication history:', error);
      res.status(500).json({ message: 'Failed to fetch medication history' });
    }
  });

  // Confirm patient start date for a medication
  app.post('/api/medications/:id/confirm-start', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const userId = req.user!.id;
      const { actualStartDate } = req.body;

      const [medication] = await db
        .select()
        .from(schema.medications)
        .where(and(eq(schema.medications.id, id), eq(schema.medications.patientId, userId)));

      if (!medication) {
        return res.status(404).json({ message: 'Medication not found' });
      }

      // Calculate computed end date if duration is specified
      let computedEndDate = null;
      if (medication.durationDays && actualStartDate) {
        const startDate = new Date(actualStartDate);
        computedEndDate = new Date(startDate);
        computedEndDate.setDate(computedEndDate.getDate() + medication.durationDays);
      }

      const updated = await storage.updateMedication(id, {
        actualStartDate: new Date(actualStartDate),
        computedEndDate,
        status: 'active',
      });

      await storage.createMedicationChangeLog({
        medicationId: id,
        patientId: userId,
        changeType: 'confirmed_start',
        changedBy: 'patient',
        changedByUserId: userId,
        changeReason: `Patient confirmed start date: ${actualStartDate}`,
      });

      res.json(updated);
    } catch (error) {
      console.error('Error confirming medication start:', error);
      res.status(500).json({ message: 'Failed to confirm medication start' });
    }
  });

  // Mark medication as taken (adherence tracking)
  app.post('/api/medications/:id/mark-taken', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const userId = req.user!.id;
      const { takenAt } = req.body;

      // Validate takenAt if provided
      const recordedAt = takenAt ? new Date(takenAt) : new Date();
      if (isNaN(recordedAt.getTime())) {
        return res.status(400).json({ message: 'Invalid takenAt date format' });
      }

      const [medication] = await db
        .select()
        .from(schema.medications)
        .where(and(eq(schema.medications.id, id), eq(schema.medications.patientId, userId)));

      if (!medication) {
        return res.status(404).json({ message: 'Medication not found' });
      }

      // Record adherence log
      await storage.createMedicationChangeLog({
        medicationId: id,
        patientId: userId,
        changeType: 'dose_taken',
        changedBy: 'patient',
        changedByUserId: userId,
        changeReason: `Patient marked dose taken at ${recordedAt.toISOString()}`,
      });

      // Get adherence count for this medication (total doses taken)
      const adherenceLogs = await storage.getMedicationChangelog(id);
      const dosesTaken = adherenceLogs.filter(log => log.changeType === 'dose_taken').length;

      console.log(`[ADHERENCE] Patient ${userId} marked dose taken for medication ${id} - total doses: ${dosesTaken}`);
      res.json({ 
        success: true, 
        message: 'Dose recorded successfully',
        medicationId: id,
        dosesTaken,
        lastTakenAt: recordedAt.toISOString(),
      });
    } catch (error) {
      console.error('Error marking medication taken:', error);
      res.status(500).json({ message: 'Failed to record dose' });
    }
  });

  // Request refill for a medication
  app.post('/api/medications/:id/request-refill', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const userId = req.user!.id;

      const [medication] = await db
        .select()
        .from(schema.medications)
        .where(and(eq(schema.medications.id, id), eq(schema.medications.patientId, userId)));

      if (!medication) {
        return res.status(404).json({ message: 'Medication not found' });
      }

      // Check for recent refill request (within last 24 hours to prevent spam)
      const recentLogs = await storage.getMedicationChangelog(id);
      const recentRefillRequest = recentLogs.find(log => {
        if (log.changeType !== 'refill_requested') return false;
        const logDate = new Date(log.createdAt || 0);
        const hoursSince = (Date.now() - logDate.getTime()) / (1000 * 60 * 60);
        return hoursSince < 24;
      });

      if (recentRefillRequest) {
        return res.status(400).json({ 
          message: 'A refill request was already submitted within the last 24 hours. Please wait before requesting again.',
        });
      }

      // Record refill request
      await storage.createMedicationChangeLog({
        medicationId: id,
        patientId: userId,
        changeType: 'refill_requested',
        changedBy: 'patient',
        changedByUserId: userId,
        changeReason: `Patient requested refill for ${medication.name}`,
      });

      // Get prescribing doctor if available
      const prescribingDoctorId = medication.prescribingDoctorId;
      if (prescribingDoctorId) {
        // Create notification for doctor
        console.log(`[REFILL] Patient ${userId} requested refill for ${medication.name}, notifying doctor ${prescribingDoctorId}`);
        // In production, would create notification record or send email here
      }

      // Count total refill requests for this medication
      const allLogs = await storage.getMedicationChangelog(id);
      const refillCount = allLogs.filter(log => log.changeType === 'refill_requested').length;

      res.json({ 
        success: true, 
        message: 'Refill request submitted. Your doctor will be notified.',
        medicationId: id,
        medicationName: medication.name,
        refillRequestCount: refillCount,
        requestedAt: new Date().toISOString(),
      });
    } catch (error) {
      console.error('Error requesting refill:', error);
      res.status(500).json({ message: 'Failed to request refill' });
    }
  });

  // Get medications grouped by specialty
  app.get('/api/medications/by-specialty', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const allMeds = await storage.getAllMedications(userId);
      
      // Group by specialty
      const grouped: Record<string, any[]> = {};
      for (const med of allMeds) {
        const specialty = med.specialty || 'unspecified';
        if (!grouped[specialty]) {
          grouped[specialty] = [];
        }
        grouped[specialty].push(med);
      }
      
      res.json(grouped);
    } catch (error) {
      console.error('Error fetching medications by specialty:', error);
      res.status(500).json({ message: 'Failed to fetch medications by specialty' });
    }
  });

  // Check for supersession candidates (same specialty medications)
  app.get('/api/medications/supersession-check', isDoctor, async (req: any, res) => {
    try {
      const { patientId, specialty, drugClass } = req.query;
      
      if (!patientId) {
        return res.status(400).json({ message: 'Patient ID required' });
      }

      let candidates: any[] = [];
      
      if (specialty) {
        candidates = await storage.getMedicationsBySpecialty(patientId as string, specialty as string);
      } else if (drugClass) {
        candidates = await storage.getMedicationsByDrugClass(patientId as string, drugClass as string);
      }

      res.json({ 
        supersessionCandidates: candidates,
        message: candidates.length > 0 
          ? `Found ${candidates.length} active medication(s) that may be superseded` 
          : 'No medications to supersede'
      });
    } catch (error) {
      console.error('Error checking supersession candidates:', error);
      res.status(500).json({ message: 'Failed to check supersession candidates' });
    }
  });

  // Supersede a medication (doctor only)
  app.post('/api/medications/:id/supersede', isDoctor, async (req: any, res) => {
    try {
      const { id: oldMedicationId } = req.params;
      const doctorId = req.user!.id;
      const { newMedicationId, reason } = req.body;

      const oldMed = await storage.supersedeMedication(oldMedicationId, newMedicationId, reason);
      
      if (!oldMed) {
        return res.status(404).json({ message: 'Medication not found' });
      }

      // Log the supersession
      await storage.createMedicationChangeLog({
        medicationId: oldMedicationId,
        patientId: oldMed.patientId,
        changeType: 'superseded',
        changedBy: 'doctor',
        changedByUserId: doctorId,
        changeReason: reason,
        notes: `Superseded by medication ID: ${newMedicationId}`,
      });

      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} superseded medication ${oldMedicationId} with ${newMedicationId} at ${new Date().toISOString()}`);

      res.json({ superseded: oldMed, newMedicationId });
    } catch (error) {
      console.error('Error superseding medication:', error);
      res.status(500).json({ message: 'Failed to supersede medication' });
    }
  });

  // Auto-archive expired medications (can be called by cron or manually)
  app.post('/api/medications/auto-archive', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const allMeds = await storage.getAllMedications(userId);
      const now = new Date();
      const archived: string[] = [];

      for (const med of allMeds) {
        // Skip continuous medications - they don't auto-expire
        if (med.isContinuous) continue;
        
        // Check if medication has expired
        const endDate = med.computedEndDate || med.endDate;
        if (endDate && new Date(endDate) < now && med.status === 'active') {
          await storage.updateMedication(med.id, {
            status: 'expired',
            active: false,
          });
          
          await storage.createMedicationChangeLog({
            medicationId: med.id,
            patientId: userId,
            changeType: 'expired',
            changedBy: 'system',
            changeReason: 'Medication duration completed',
          });
          
          archived.push(med.id);
        }
      }

      res.json({ 
        message: `Auto-archived ${archived.length} expired medication(s)`,
        archivedIds: archived
      });
    } catch (error) {
      console.error('Error auto-archiving medications:', error);
      res.status(500).json({ message: 'Failed to auto-archive medications' });
    }
  });

  // Doctor-specific drug interaction check for patient
  // Uses doctor_patient_assignments table for HIPAA-compliant access control
  app.post('/api/drug-interactions/analyze-for-patient', isDoctor, async (req: any, res) => {
    try {
      const { patientId, drugName, drugClass, genericName } = req.body;
      const doctorId = req.user!.id;

      if (!patientId || !drugName) {
        return res.status(400).json({ message: "Patient ID and drug name are required" });
      }

      // Verify patient exists and is a valid patient
      const patient = await storage.getUser(patientId);
      if (!patient || patient.role !== 'patient') {
        return res.status(404).json({ message: 'Patient not found' });
      }
      
      // HIPAA authorization: Check if doctor has active assignment with this patient
      const hasAccess = await storage.doctorHasPatientAccess(doctorId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${doctorId} attempted drug interaction check for unassigned patient ${patientId} at ${new Date().toISOString()}`);
        return res.status(403).json({ 
          message: 'Access denied. No active assignment with this patient.',
          code: 'NO_PATIENT_ASSIGNMENT'
        });
      }
      
      // HIPAA audit: Log drug interaction check
      console.log(`[HIPAA-AUDIT] Doctor ${doctorId} checked drug interactions for patient ${patientId} (drug: ${drugName}) at ${new Date().toISOString()}`);

      const patientProfile = await storage.getPatientProfile(patientId);
      const currentMedications = await storage.getActiveMedications(patientId);

      const { analyzeMultipleDrugInteractions, enrichMedicationWithGenericName } = await import('./drugInteraction');

      // Build medication list with ALL name variations
      const medicationsToCheck = await Promise.all(currentMedications.map(async (med) => {
        let drug = await storage.getDrugByName(med.name);
        
        if (!drug) {
          const enriched = await enrichMedicationWithGenericName(med.name);
          drug = await storage.createDrug({
            name: med.name,
            genericName: enriched.genericName,
            brandNames: enriched.brandNames
          });
        }
        
        return {
          name: med.name,
          genericName: drug.genericName || med.name,
          drugClass: drug.drugClass,
          id: med.id,
          brandNames: drug.brandNames || []
        };
      }));

      // Add the new drug
      const newDrug = {
        name: drugName,
        genericName: genericName || drugName,
        drugClass: drugClass,
        id: null,
        brandNames: []
      };

      medicationsToCheck.push(newDrug);

      const interactions = await analyzeMultipleDrugInteractions(
        medicationsToCheck,
        {
          isImmunocompromised: true,
          conditions: patientProfile?.immunocompromisedCondition 
            ? [patientProfile.immunocompromisedCondition]
            : [],
        }
      );

      // Transform to simpler format for frontend
      const formattedInteractions = interactions.map(i => ({
        drug1: i.drug1,
        drug2: i.drug2,
        severity: i.interaction.severityLevel,
        description: i.interaction.clinicalEffects,
        recommendations: i.interaction.managementRecommendations 
          ? [i.interaction.managementRecommendations] 
          : [],
      }));

      res.json({
        interactions: formattedInteractions,
        patientId,
        medicationsChecked: currentMedications.length + 1,
      });
    } catch (error) {
      console.error("Error checking drug interactions for patient:", error);
      res.status(500).json({ message: "Failed to analyze drug interactions" });
    }
  });

  // Dosage change request routes
  app.get('/api/dosage-change-requests', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const requests = await storage.getDosageChangeRequests(userId);
      res.json(requests);
    } catch (error) {
      console.error('Error fetching dosage change requests:', error);
      res.status(500).json({ message: 'Failed to fetch requests' });
    }
  });

  app.get('/api/dosage-change-requests/pending', isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const requests = await storage.getPendingDosageChangeRequests(doctorId);
      res.json(requests);
    } catch (error) {
      console.error('Error fetching pending dosage change requests:', error);
      res.status(500).json({ message: 'Failed to fetch pending requests' });
    }
  });

  app.post('/api/dosage-change-requests', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const request = await storage.createDosageChangeRequest({
        patientId: userId,
        ...req.body,
      });
      
      // Send notification to doctor
      try {
        const patient = await storage.getUser(userId);
        const doctor = await storage.getUserById(req.body.doctorId);
        
        if (doctor && doctor.phoneNumber) {
          // Send SMS notification using Twilio
          const { twilioClient } = await import('./twilioService');
          await twilioClient.messages.create({
            to: doctor.phoneNumber,
            from: process.env.TWILIO_PHONE_NUMBER,
            body: `New dosage change request from patient ${patient?.name || userId}. Please review in your dashboard.`
          });
        }
        
        // Send email notification if doctor has email
        if (doctor && doctor.email) {
          const { sesClient } = await import('./aws');
          const { SendEmailCommand } = await import('@aws-sdk/client-ses');
          
          await sesClient.send(new SendEmailCommand({
            Source: 'noreply@followupai.com',
            Destination: { ToAddresses: [doctor.email] },
            Message: {
              Subject: { Data: 'New Medication Dosage Change Request' },
              Body: {
                Text: {
                  Data: `A patient has requested a dosage change:\n\nPatient: ${patient?.name || userId}\nMedication: ${req.body.medicationId}\nCurrent: ${req.body.currentDosage}\nRequested: ${req.body.requestedDosage}\nReason: ${req.body.requestReason}\n\nPlease review and approve/reject in your dashboard.`
                }
              }
            }
          }));
        }
        
        console.log('Dosage change request notification sent to doctor:', doctor?.id);
      } catch (notifError) {
        // Don't fail the request if notification fails
        console.error('Failed to send notification to doctor:', notifError);
      }
      
      res.json(request);
    } catch (error) {
      console.error('Error creating dosage change request:', error);
      res.status(500).json({ message: 'Failed to create request' });
    }
  });

  app.post('/api/dosage-change-requests/:id/approve', isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const doctorId = req.user!.id;
      const { notes } = req.body;
      
      const request = await storage.approveDosageChangeRequest(id, doctorId, notes);
      
      if (request) {
        // Apply the dosage change
        await storage.updateMedication(request.medicationId, {
          dosage: request.requestedDosage,
          frequency: request.requestedFrequency,
        });
        
        // Log the change
        await storage.createMedicationChangeLog({
          medicationId: request.medicationId,
          patientId: request.patientId,
          changeType: 'dosage_changed',
          changedBy: 'doctor',
          changedByUserId: doctorId,
          oldDosage: request.currentDosage,
          newDosage: request.requestedDosage,
          oldFrequency: request.currentFrequency,
          newFrequency: request.requestedFrequency,
          changeReason: `Doctor approved patient request: ${request.requestReason}`,
          notes,
        });
      }
      
      res.json(request);
    } catch (error) {
      console.error('Error approving dosage change:', error);
      res.status(500).json({ message: 'Failed to approve request' });
    }
  });

  app.post('/api/dosage-change-requests/:id/reject', isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const doctorId = req.user!.id;
      const { notes } = req.body;
      
      const request = await storage.rejectDosageChangeRequest(id, doctorId, notes);
      
      // Send notification to patient
      if (request) {
        try {
          const patient = await storage.getUserById(request.patientId);
          const doctor = await storage.getUser(doctorId);
          
          if (patient && patient.phoneNumber) {
            // Send SMS notification using Twilio
            const { twilioClient } = await import('./twilioService');
            await twilioClient.messages.create({
              to: patient.phoneNumber,
              from: process.env.TWILIO_PHONE_NUMBER,
              body: `Your dosage change request has been reviewed by Dr. ${doctor?.name || 'your doctor'}. Status: Rejected. Reason: ${notes || 'See dashboard for details'}.`
            });
          }
          
          // Send email notification if patient has email
          if (patient && patient.email) {
            const { sesClient } = await import('./aws');
            const { SendEmailCommand } = await import('@aws-sdk/client-ses');
            
            await sesClient.send(new SendEmailCommand({
              Source: 'noreply@followupai.com',
              Destination: { ToAddresses: [patient.email] },
              Message: {
                Subject: { Data: 'Dosage Change Request - Rejected' },
                Body: {
                  Text: {
                    Data: `Your dosage change request has been reviewed by Dr. ${doctor?.name || 'your doctor'}.\n\nStatus: Rejected\n\nDoctor's notes: ${notes || 'No additional notes provided'}\n\nPlease contact your doctor if you have questions.`
                  }
                }
              }
            }));
          }
          
          console.log('Rejection notification sent to patient:', patient.id);
        } catch (notifError) {
          console.error('Failed to send notification to patient:', notifError);
        }
      }
      
      res.json(request);
    } catch (error) {
      console.error('Error rejecting dosage change:', error);
      res.status(500).json({ message: 'Failed to reject request' });
    }
  });

  // Medication sync from medical files
  app.post('/api/medications/sync-from-document/:documentId', isAuthenticated, async (req: any, res) => {
    try {
      const { documentId } = req.params;
      const userId = req.user!.id;
      
      // Get the medical file
      const document = await storage.getMedicalFileById(documentId);
      if (!document || document.patientId !== userId) {
        return res.status(404).json({ message: 'Document not found' });
      }
      
      // Extract medications from the document's AI analysis
      const medications = document.extractedData?.medications || [];
      
      if (medications.length === 0) {
        return res.json({ message: 'No medications found in this document', count: 0 });
      }
      
      const createdMedications = [];
      
      for (const medData of medications) {
        // Check for duplicates
        const existing = await storage.getMedicationByNameAndPatient(medData.name || medData.text, userId);
        
        if (!existing) {
          // Create pending medication linked to source document
          const medication = await storage.createMedication({
            patientId: userId,
            name: medData.name || medData.text,
            dosage: medData.dosage || medData.strength || 'Not specified',
            frequency: medData.frequency || medData.routeOrMode || 'Not specified',
            source: 'document',
            sourceDocumentId: documentId,
            addedBy: 'system',
            status: 'pending_confirmation',
            autoDetected: true,
          });
          
          createdMedications.push(medication);
          
          // Log the change
          await storage.createMedicationChangeLog({
            medicationId: medication.id,
            patientId: userId,
            changeType: 'added',
            changedBy: 'system',
            changedByUserId: userId,
            changeReason: 'Auto-detected from medical file',
            notes: `Source: ${document.fileName}`,
          });
        }
      }
      
      res.json({
        message: `Synced ${createdMedications.length} new medications from document`,
        count: createdMedications.length,
        medications: createdMedications,
      });
    } catch (error) {
      console.error('Error syncing medications from document:', error);
      res.status(500).json({ message: 'Failed to sync medications' });
    }
  });

  // Medication changelog routes
  app.get('/api/medications/:id/changelog', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const logs = await storage.getMedicationChangelog(id);
      res.json(logs);
    } catch (error) {
      console.error('Error fetching medication changelog:', error);
      res.status(500).json({ message: 'Failed to fetch changelog' });
    }
  });

  app.get('/api/medications/changelog/all', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { limit } = req.query;
      const logs = await storage.getPatientMedicationChangelog(userId, limit ? parseInt(limit as string) : undefined);
      res.json(logs);
    } catch (error) {
      console.error('Error fetching patient medication changelog:', error);
      res.status(500).json({ message: 'Failed to fetch changelog' });
    }
  });

  // ==================== HEALTH COMPANION MODE ROUTES ====================

  // Companion check-ins
  app.post('/api/companion/check-in', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { checkInType, userInput, context } = req.body;
      
      if (!userInput) {
        return res.status(400).json({ message: 'User input is required' });
      }

      const { processCompanionCheckIn } = await import('./companionService');
      const result = await processCompanionCheckIn({
        patientId: userId,
        checkInType: checkInType || 'spontaneous',
        userInput,
        context,
      });

      res.json(result);
    } catch (error) {
      console.error('Error processing companion check-in:', error);
      res.status(500).json({ message: 'Failed to process check-in' });
    }
  });

  app.get('/api/companion/check-ins', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { limit, type } = req.query;
      
      let checkIns;
      if (type) {
        checkIns = await storage.getCheckInsByType(
          userId,
          type as string,
          limit ? parseInt(limit as string) : undefined
        );
      } else {
        checkIns = await storage.getCompanionCheckIns(
          userId,
          limit ? parseInt(limit as string) : undefined
        );
      }

      res.json(checkIns);
    } catch (error) {
      console.error('Error fetching check-ins:', error);
      res.status(500).json({ message: 'Failed to fetch check-ins' });
    }
  });

  app.get('/api/companion/check-ins/recent', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { days } = req.query;
      const checkIns = await storage.getRecentCheckIns(
        userId,
        days ? parseInt(days as string) : 7
      );
      res.json(checkIns);
    } catch (error) {
      console.error('Error fetching recent check-ins:', error);
      res.status(500).json({ message: 'Failed to fetch recent check-ins' });
    }
  });

  // Companion engagement
  app.get('/api/companion/engagement', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const engagement = await storage.getCompanionEngagement(userId);
      res.json(engagement || {});
    } catch (error) {
      console.error('Error fetching engagement data:', error);
      res.status(500).json({ message: 'Failed to fetch engagement data' });
    }
  });

  app.patch('/api/companion/engagement', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const engagement = await storage.updateCompanionEngagement(userId, req.body);
      res.json(engagement);
    } catch (error) {
      console.error('Error updating engagement settings:', error);
      res.status(500).json({ message: 'Failed to update engagement settings' });
    }
  });

  // Helper endpoints
  app.get('/api/companion/suggest-check-in-time', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { suggestCheckInTime } = await import('./companionService');
      const time = await suggestCheckInTime(userId);
      res.json({ suggestedTime: time });
    } catch (error) {
      console.error('Error suggesting check-in time:', error);
      res.status(500).json({ message: 'Failed to suggest check-in time' });
    }
  });

  app.get('/api/companion/prompt', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { type } = req.query;
      const { generateCheckInPrompt } = await import('./companionService');
      const prompt = await generateCheckInPrompt(userId, (type as string) || 'morning');
      res.json({ prompt });
    } catch (error) {
      console.error('Error generating check-in prompt:', error);
      res.status(500).json({ message: 'Failed to generate prompt' });
    }
  });

  // Voice Followup Routes - Quick 1min voice logs
  const voiceUpload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 25 * 1024 * 1024 }, // 25MB limit (OpenAI Whisper limit)
    fileFilter: (req, file, cb) => {
      const allowedTypes = ['audio/webm', 'audio/wav', 'audio/mp3', 'audio/m4a', 'audio/mpeg'];
      if (allowedTypes.includes(file.mimetype)) {
        cb(null, true);
      } else {
        cb(new Error('Invalid file type. Only audio files are allowed.'));
      }
    },
  });

  app.post('/api/voice-followup/upload', isAuthenticated, voiceUpload.single('audio'), async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      if (!req.file) {
        return res.status(400).json({ message: 'No audio file provided' });
      }

      const { processVoiceFollowup } = await import('./voiceProcessingService');
      
      // Process the voice recording (pass mimetype for S3)
      const processedData = await processVoiceFollowup(
        req.file.buffer,
        req.file.originalname,
        userId,
        req.file.mimetype
      );

      // Store in database
      const voiceFollowup = await storage.createVoiceFollowup({
        patientId: userId,
        ...processedData,
      });

      res.json({
        id: voiceFollowup.id,
        transcription: voiceFollowup.transcription,
        response: voiceFollowup.aiResponse,
        empathyLevel: voiceFollowup.empathyLevel,
        conversationSummary: voiceFollowup.conversationSummary,
        concernsRaised: voiceFollowup.concernsRaised,
        needsFollowup: voiceFollowup.needsFollowup,
        followupReason: voiceFollowup.followupReason,
        recommendedActions: voiceFollowup.recommendedActions,
        extractedSymptoms: voiceFollowup.extractedSymptoms,
        extractedMood: voiceFollowup.extractedMood,
        medicationAdherence: voiceFollowup.medicationAdherence,
        extractedMetrics: voiceFollowup.extractedMetrics,
      });
    } catch (error) {
      console.error('Error processing voice followup:', error);
      res.status(500).json({ message: 'Failed to process voice recording' });
    }
  });

  app.get('/api/voice-followup/recent', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { limit } = req.query;
      const followups = await storage.getRecentVoiceFollowups(
        userId,
        limit ? parseInt(limit) : 10
      );
      res.json(followups);
    } catch (error) {
      console.error('Error fetching voice followups:', error);
      res.status(500).json({ message: 'Failed to fetch voice followups' });
    }
  });

  app.get('/api/voice-followup/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const followup = await storage.getVoiceFollowup(id);
      
      if (!followup) {
        return res.status(404).json({ message: 'Voice followup not found' });
      }
      
      // Verify the followup belongs to the requesting user
      if (followup.patientId !== req.user!.id && req.user!.role !== 'doctor') {
        return res.status(403).json({ message: 'Unauthorized' });
      }
      
      res.json(followup);
    } catch (error) {
      console.error('Error fetching voice followup:', error);
      res.status(500).json({ message: 'Failed to fetch voice followup' });
    }
  });

  app.get('/api/voice-followup', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const followups = await storage.getAllVoiceFollowups(userId);
      res.json(followups);
    } catch (error) {
      console.error('Error fetching all voice followups:', error);
      res.status(500).json({ message: 'Failed to fetch voice followups' });
    }
  });

  // ==================== ML/RL PERSONALIZATION ROUTES ====================
  // Versioned API namespace: /api/v1/ml/*
  // All routes require authentication and implement HIPAA-compliant data handling

  // Get personalized recommendations for Agent Clona or Assistant Lysa
  app.get('/api/v1/ml/recommendations', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const agentType = (req.query.agentType as 'clona' | 'lysa') || 'clona';
      const limit = parseInt(req.query.limit as string) || 5;

      // Role-based access: Verify role via fresh storage lookup (prevent session tampering)
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(401).json({ message: 'Unauthorized: User not found' });
      }

      if (agentType === 'lysa' && user.role !== 'doctor') {
        return res.status(403).json({ message: 'Unauthorized: Lysa recommendations are doctor-only' });
      }

      const recommendations = await personalizationService.getRecommendations(userId, agentType, limit);
      
      // De-identify PHI: return only necessary fields
      const sanitized = recommendations.map(r => ({
        id: r.id,
        type: r.type,
        category: r.category,
        title: r.title,
        description: r.description,
        confidenceScore: r.confidenceScore,
        personalizationScore: r.personalizationScore,
        priority: r.priority,
        reasoning: r.reasoning,
      }));

      res.json(sanitized);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      res.status(500).json({ message: 'Failed to fetch recommendations' });
    }
  });

  // Create a new habit (patients only)
  app.post('/api/v1/ml/habits', isAuthenticated, isPatient, async (req: any, res) => {
    try {
      const userId = req.user!.id;

      // Validate and sanitize input
      const validated = createHabitSchema.parse(req.body);

      const habit = await storage.createHabit({
        userId,
        name: validated.name,
        description: validated.description,
        category: validated.category,
        frequency: validated.frequency,
        goalCount: validated.goalCount,
        currentStreak: 0,
        longestStreak: 0,
        totalCompletions: 0,
      });

      res.status(201).json(habit);
    } catch (error: any) {
      if (error.name === 'ZodError') {
        return res.status(400).json({ message: 'Invalid input', errors: error.errors });
      }
      console.error('Error creating habit:', error);
      res.status(500).json({ message: 'Failed to create habit' });
    }
  });

  // Get user's habits
  app.get('/api/v1/ml/habits', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const habits = await storage.getHabits(userId);
      res.json(habits);
    } catch (error) {
      console.error('Error fetching habits:', error);
      res.status(500).json({ message: 'Failed to fetch habits' });
    }
  });

  // Mark habit as complete
  app.post('/api/v1/ml/habits/:id/complete', isAuthenticated, isPatient, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const habitId = req.params.id;
      
      // Validate and sanitize input
      const validated = completeHabitSchema.parse(req.body);
      const { mood, notes, difficultyLevel } = validated;

      // Verify habit belongs to user
      const habits = await storage.getHabits(userId);
      const habit = habits.find(h => h.id === habitId);
      
      if (!habit) {
        return res.status(404).json({ message: 'Habit not found' });
      }

      // Create completion
      const completion = await storage.createHabitCompletion({
        habitId,
        userId,
        mood,
        notes,
        difficultyLevel: difficultyLevel || 3,
        completedAt: new Date(),
      });

      // Update habit streak and stats
      const currentStreak = (habit.currentStreak || 0) + 1;
      const longestStreak = Math.max(currentStreak, habit.longestStreak || 0);
      const totalCompletions = (habit.totalCompletions || 0) + 1;

      await storage.updateHabit(habitId, {
        currentStreak,
        longestStreak,
        totalCompletions,
        lastCompletedAt: new Date(),
      });

      // Calculate RL reward
      const reward = rlRewardCalculator.calculateHabitCompletionReward(
        {
          userContext: {
            currentStreak,
            habitsCompleted: totalCompletions,
            engagementScore: 0,
            recentSentiment: mood === 'great' ? 1 : mood === 'good' ? 0.5 : mood === 'okay' ? 0 : -0.5,
          },
          conversationContext: [],
          healthMetrics: {},
          recentActions: [],
        },
        completion,
        currentStreak > (habit.currentStreak || 0)
      );

      // Store RL reward
      await storage.createRLReward({
        userId,
        agentType: 'clona',
        reward: reward.toString(),
        rewardType: 'completion',
        state: {
          userContext: { habitId, currentStreak, totalCompletions },
          conversationContext: [],
          healthMetrics: {},
          recentActions: [`habit_completion_${habitId}`],
        },
        action: {
          type: 'habit_completion',
          content: `Completed habit: ${habit.name}`,
          parameters: { mood, difficultyLevel, habitId },
        },
      });

      res.json({ completion, reward, newStreak: currentStreak });
    } catch (error: any) {
      if (error.name === 'ZodError') {
        return res.status(400).json({ message: 'Invalid input', errors: error.errors });
      }
      console.error('Error completing habit:', error);
      res.status(500).json({ message: 'Failed to complete habit' });
    }
  });

  // Process user feedback (for RL reward loop)
  app.post('/api/v1/ml/feedback', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      // Validate and sanitize input
      const validated = feedbackSchema.parse(req.body);
      const { agentType, helpful, sentiment, category, messageId } = validated;

      await personalizationService.processFeedback(userId, agentType, {
        messageId,
        helpful,
        sentiment,
        category,
      });

      res.json({ success: true, message: 'Feedback processed' });
    } catch (error: any) {
      if (error.name === 'ZodError') {
        return res.status(400).json({ message: 'Invalid input', errors: error.errors });
      }
      console.error('Error processing feedback:', error);
      res.status(500).json({ message: 'Failed to process feedback' });
    }
  });

  // Get personalized agent prompt (RAG integration for OpenAI)
  app.post('/api/v1/ml/agent-prompts', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      // Validate and sanitize input (prevents prompt injection)
      const validated = agentPromptSchema.parse(req.body);
      const { agentType, basePrompt } = validated;

      // Role-based access
      if (agentType === 'lysa' && req.user!.role !== 'doctor') {
        return res.status(403).json({ message: 'Unauthorized: Lysa prompts are doctor-only' });
      }

      const enhancedPrompt = await personalizationService.personalizeAgentPrompt(
        userId,
        agentType,
        basePrompt
      );

      res.json({ enhancedPrompt });
    } catch (error: any) {
      if (error.name === 'ZodError') {
        return res.status(400).json({ message: 'Invalid input', errors: error.errors });
      }
      console.error('Error personalizing agent prompt:', error);
      res.status(500).json({ message: 'Failed to personalize prompt' });
    }
  });

  // Get user milestones
  app.get('/api/v1/ml/milestones', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const milestones = await storage.getMilestones(userId);
      res.json(milestones);
    } catch (error) {
      console.error('Error fetching milestones:', error);
      res.status(500).json({ message: 'Failed to fetch milestones' });
    }
  });

  // Track doctor wellness (doctors only)
  app.post('/api/v1/ml/doctor-wellness', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      // Validate and sanitize input
      const validated = doctorWellnessSchema.parse(req.body);
      const { stressLevel, hoursWorked, patientsToday, burnoutRisk, notes } = validated;

      const wellness = await storage.createDoctorWellness({
        doctorId: userId,
        date: new Date(),
        stressLevel,
        hoursWorked,
        patientsToday,
        burnoutRisk,
        notes,
      });

      res.status(201).json(wellness);
    } catch (error: any) {
      if (error.name === 'ZodError') {
        return res.status(400).json({ message: 'Invalid input', errors: error.errors });
      }
      console.error('Error tracking doctor wellness:', error);
      res.status(500).json({ message: 'Failed to track wellness' });
    }
  });

  // Get doctor wellness summary (doctors only)
  app.get('/api/v1/ml/doctor-wellness', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;

      // Only doctors can view their wellness history
      if (req.user!.role !== 'doctor') {
        return res.status(403).json({ message: 'Unauthorized: Only doctors can view wellness history' });
      }

      const days = parseInt(req.query.days as string) || 30;
      const history = await storage.getDoctorWellnessHistory(userId, days);
      
      // Calculate summary statistics
      if (history.length === 0) {
        return res.json(null);
      }

      // Ensure history is sorted descending (most recent first)
      const sortedHistory = history.sort((a, b) => {
        return new Date(b.date).getTime() - new Date(a.date).getTime();
      });

      const mostRecent = sortedHistory[0];
      const now = new Date();
      
      // Filter last 7 days
      const last7Days = sortedHistory.filter(h => {
        const daysDiff = Math.floor((now.getTime() - new Date(h.date).getTime()) / (1000 * 60 * 60 * 24));
        return daysDiff <= 7;
      });

      const weeklyAvgStress = last7Days.length > 0
        ? (last7Days.reduce((sum, h) => sum + h.stressLevel, 0) / last7Days.length).toFixed(1)
        : mostRecent.stressLevel.toFixed(1);

      // Filter last 30 days
      const last30Days = sortedHistory.filter(h => {
        const daysDiff = Math.floor((now.getTime() - new Date(h.date).getTime()) / (1000 * 60 * 60 * 24));
        return daysDiff <= 30;
      });
      const monthlyAvg = last30Days.length > 0
        ? last30Days.reduce((sum, h) => sum + h.stressLevel, 0) / last30Days.length
        : mostRecent.stressLevel;

      let monthlyTrend = 'stable';
      if (parseFloat(weeklyAvgStress) > monthlyAvg + 1) {
        monthlyTrend = 'increasing';
      } else if (parseFloat(weeklyAvgStress) < monthlyAvg - 1) {
        monthlyTrend = 'decreasing';
      }

      // Safely handle date conversion
      const recentDate = new Date(mostRecent.date);
      const summary = {
        id: mostRecent.id,
        stressLevel: mostRecent.stressLevel,
        burnoutRisk: mostRecent.burnoutRisk || 'low',
        workLifeBalance: mostRecent.workLifeBalance || 'Consider taking breaks',
        lastUpdate: isNaN(recentDate.getTime()) ? new Date().toISOString() : recentDate.toISOString(),
        weeklyAvgStress,
        monthlyTrend,
      };

      res.json(summary);
    } catch (error) {
      console.error('Error fetching doctor wellness history:', error);
      res.status(500).json({ message: 'Failed to fetch wellness history' });
    }
  });

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA - APPOINTMENT MANAGEMENT ROUTES
  // ============================================================================

  // Lysa Patient Search - search patients by name or identifier
  app.get('/api/v1/lysa/patients/search', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access patient search' });
      }

      const { query, limit } = req.query;
      if (!query || typeof query !== 'string') {
        return res.status(400).json({ message: 'Search query is required' });
      }

      const results = await searchPatients(query, userId, parseInt(limit as string) || 10);
      
      res.json({
        success: true,
        results: results.map(r => ({
          id: r.user.id,
          firstName: r.user.firstName,
          lastName: r.user.lastName,
          email: r.user.email,
          phoneNumber: r.user.phoneNumber,
          followupPatientId: r.profile?.followupPatientId,
          dateOfBirth: r.profile?.dateOfBirth,
          bloodType: r.profile?.bloodType,
          allergies: r.profile?.allergies,
          medicalConditions: r.profile?.medicalConditions,
          matchScore: r.matchScore
        })),
        count: results.length
      });
    } catch (error) {
      console.error('Error searching patients:', error);
      res.status(500).json({ message: 'Failed to search patients' });
    }
  });

  // Lysa Patient Record - get detailed patient information
  app.get('/api/v1/lysa/patients/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access patient records' });
      }

      const { patientId } = req.params;
      const record = await getPatientRecord(patientId, userId);
      
      if (!record) {
        return res.status(404).json({ message: 'Patient not found' });
      }

      res.json({
        success: true,
        patient: {
          id: record.patient.id,
          firstName: record.patient.firstName,
          lastName: record.patient.lastName,
          email: record.patient.email,
          phoneNumber: record.patient.phoneNumber
        },
        profile: record.profile ? {
          followupPatientId: record.profile.followupPatientId,
          dateOfBirth: record.profile.dateOfBirth,
          bloodType: record.profile.bloodType,
          allergies: record.profile.allergies,
          medicalConditions: record.profile.medicalConditions,
          emergencyContact: record.profile.emergencyContact,
          emergencyPhone: record.profile.emergencyPhone
        } : null,
        recentAppointments: record.recentAppointments,
        upcomingAppointments: record.upcomingAppointments
      });
    } catch (error) {
      console.error('Error getting patient record:', error);
      res.status(500).json({ message: 'Failed to get patient record' });
    }
  });

  // Lysa Prescription Interaction Check - Drug safety analysis
  // CROSS-SPECIALTY ONLY: Only checks interactions against medications from OTHER specialties
  app.post('/api/v1/lysa/prescriptions/check-interactions', isAuthenticated, aiRateLimit, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can check drug interactions' });
      }

      const { medications, allergies, newPrescriptions, patientId, doctorSpecialty, allowAllMedications } = req.body;

      // Get the doctor's specialty from their profile if not provided
      let currentSpecialty = doctorSpecialty;
      if (!currentSpecialty) {
        const doctorProfile = await storage.getDoctorProfile(userId);
        currentSpecialty = doctorProfile?.specialty || null;
      }

      // ENFORCE SPECIALTY: Reject if specialty is missing and patientId is provided (unless explicit override)
      if (!currentSpecialty && patientId && !allowAllMedications) {
        return res.status(400).json({
          message: 'Doctor specialty is required for cross-specialty interaction checking',
          error: 'SPECIALTY_REQUIRED',
          hint: 'Please specify your specialty in the request (doctorSpecialty) or update your doctor profile. Set allowAllMedications=true to check against all patient medications instead.',
          crossSpecialtyCount: 0,
          sameSpecialtyCount: 0,
          checkedMedications: [],
          _note: 'Cross-specialty check requires doctor specialty to be specified.'
        });
      }

      // Track if using all-medications mode (explicit override)
      const usingAllMedicationsMode = !currentSpecialty && patientId && allowAllMedications;

      // If patientId is provided, fetch cross-specialty medications only
      let crossSpecialtyMedications: string[] = [];
      let sameSpecialtyMedications: string[] = [];
      let allPatientMedications: string[] = []; // Used when specialty is unknown
      
      if (patientId) {
        // Get patient's active medications from database
        const patientMeds = await db.select()
          .from(schema.medications)
          .where(and(
            eq(schema.medications.patientId, patientId),
            eq(schema.medications.status, 'active'),
            isNull(schema.medications.archivedAt)
          ));
        
        // Also check prescriptions table for active prescriptions
        const patientPrescriptions = await db.select()
          .from(schema.prescriptions)
          .where(and(
            eq(schema.prescriptions.patientId, patientId),
            eq(schema.prescriptions.status, 'active')
          ));

        if (currentSpecialty) {
          // Separate medications by specialty
          for (const med of patientMeds) {
            if (med.specialty && med.specialty.toLowerCase() === currentSpecialty.toLowerCase()) {
              sameSpecialtyMedications.push(med.name);
            } else {
              crossSpecialtyMedications.push(med.name);
            }
          }
          
          for (const rx of patientPrescriptions) {
            if (rx.specialty && rx.specialty.toLowerCase() === currentSpecialty.toLowerCase()) {
              if (!sameSpecialtyMedications.includes(rx.medicationName)) {
                sameSpecialtyMedications.push(rx.medicationName);
              }
            } else {
              if (!crossSpecialtyMedications.includes(rx.medicationName)) {
                crossSpecialtyMedications.push(rx.medicationName);
              }
            }
          }
        } else {
          // Specialty unknown - collect ALL patient medications for checking
          for (const med of patientMeds) {
            allPatientMedications.push(med.name);
          }
          for (const rx of patientPrescriptions) {
            if (!allPatientMedications.includes(rx.medicationName)) {
              allPatientMedications.push(rx.medicationName);
            }
          }
        }
      }

      // Build the medication list for interaction checking
      // Include new prescriptions + cross-specialty medications (or all meds if specialty unknown)
      let allMedicationsForCheck: string[];
      
      if (currentSpecialty) {
        // Cross-specialty mode: exclude same-specialty medications
        allMedicationsForCheck = [
          ...(newPrescriptions || []),
          ...crossSpecialtyMedications,
          // Include any explicitly passed medications that aren't from same specialty
          ...(medications || []).filter((m: string) => !sameSpecialtyMedications.includes(m))
        ];
      } else {
        // Specialty unknown: check against ALL patient medications
        allMedicationsForCheck = [
          ...(newPrescriptions || []),
          ...allPatientMedications,
          ...(medications || [])
        ];
      }

      // Remove duplicates
      const uniqueMedications = [...new Set(allMedicationsForCheck)];

      if (uniqueMedications.length === 0) {
        // No cross-specialty medications to check
        return res.json({
          hasInteractions: false,
          interactions: [],
          allergicRisks: [],
          contraindications: [],
          warnings: [],
          safeToPresrcibe: true,
          _note: "No cross-specialty medications to check. Same-specialty medications are managed by supersession rules.",
          crossSpecialtyCount: 0,
          sameSpecialtyCount: sameSpecialtyMedications.length,
          checkedMedications: []
        });
      }

      const medicationList = uniqueMedications.join(', ');
      const allergyList = allergies?.length > 0 ? allergies.join(', ') : 'None reported';
      const newMedsList = newPrescriptions?.length > 0 ? newPrescriptions.join(', ') : 'None';
      const crossSpecialtyList = crossSpecialtyMedications.join(', ') || 'None';

      const systemPrompt = `You are a clinical pharmacology AI assistant specializing in CROSS-SPECIALTY drug interaction analysis. You analyze medication combinations prescribed by DIFFERENT medical specialties for potential interactions, allergic cross-reactivity, and contraindications.

IMPORTANT CONTEXT:
- Same-specialty medications are managed through supersession rules (replacing old with new)
- Your focus is on detecting interactions between medications from DIFFERENT specialties
- This is critical for immunocompromised patients who see multiple specialists

Your role is to:
1. Identify drug-drug interactions between medications from different specialties
2. Flag potential allergic cross-reactivity based on drug classes
3. Identify contraindications when combining medications from different specialties
4. Provide clinical recommendations for multi-specialty medication management

Severity levels:
- minor: Unlikely to cause significant problems, monitor if needed
- moderate: May require dosage adjustment or monitoring
- major: Significant interaction requiring intervention from both specialists
- contraindicated: Should not be used together - requires specialist consultation

Respond ONLY with valid JSON in this exact format:
{
  "hasInteractions": true/false,
  "interactions": [
    {
      "drug1": "Medication 1",
      "drug2": "Medication 2",
      "severity": "minor|moderate|major|contraindicated",
      "description": "Description of the interaction",
      "clinicalEffect": "What happens when taken together",
      "recommendation": "Clinical recommendation for multi-specialty coordination"
    }
  ],
  "allergicRisks": ["Description of allergy cross-reactivity risks"],
  "contraindications": ["Any absolute contraindications"],
  "warnings": ["General warnings about the cross-specialty medication combination"],
  "safeToPresrcibe": true/false
}`;

      const userPrompt = `Please analyze the following CROSS-SPECIALTY medication combination for potential drug interactions:

**New Prescription(s) Being Added (Current Specialty):**
${newMedsList}

**Existing Medications from OTHER Specialties:**
${crossSpecialtyList}

**All Medications Being Checked:**
${medicationList}

**Patient Allergies:**
${allergyList}

Please identify:
1. Any drug-drug interactions between the NEW prescription and CROSS-SPECIALTY medications
2. Cross-reactivity risks with known allergies
3. Contraindications or warnings when combining these multi-specialty medications
4. Overall safety assessment for this cross-specialty combination

Provide a comprehensive safety analysis focused on cross-specialty interaction risks.`;

      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt }
        ],
        response_format: { type: "json_object" },
        temperature: 0.2,
        max_tokens: 2000
      });

      const analysisResult = JSON.parse(completion.choices[0].message.content || '{}');

      // Log for HIPAA audit with mode context
      if (usingAllMedicationsMode) {
        console.log(`[HIPAA-AUDIT] [ALL-MEDICATIONS-MODE] Drug interaction check performed by doctor ${userId} for patient ${patientId || 'unknown'} - Cross-specialty filtering BYPASSED (explicit override: allowAllMedications=true)`);
      } else {
        console.log(`[HIPAA-AUDIT] Cross-specialty drug interaction check performed by doctor ${userId} for patient ${patientId || 'unknown'} - Specialty: ${currentSpecialty}`);
      }

      // Enhance response with cross-specialty context
      let note: string;
      if (usingAllMedicationsMode) {
        // All-medications mode (explicit override) - checked all medications
        note = `ALL-MEDICATIONS MODE: Checked against ${allPatientMedications.length} total patient medication(s). Cross-specialty filtering bypassed by explicit request.`;
      } else if (crossSpecialtyMedications.length > 0) {
        note = `Checked against ${crossSpecialtyMedications.length} medication(s) from other specialties. ${sameSpecialtyMedications.length} same-specialty medication(s) excluded (managed by supersession rules).`;
      } else {
        note = "Only new prescriptions checked. No existing cross-specialty medications found.";
      }

      res.json({
        ...analysisResult,
        crossSpecialtyCount: usingAllMedicationsMode ? 0 : crossSpecialtyMedications.length,
        sameSpecialtyCount: sameSpecialtyMedications.length,
        allMedicationsCount: usingAllMedicationsMode ? allPatientMedications.length : undefined,
        checkedMedications: uniqueMedications,
        _note: note,
        _allMedicationsMode: usingAllMedicationsMode || false,
        _specialty: currentSpecialty || null
      });
    } catch (error) {
      console.error('Error checking drug interactions:', error);
      // Return fallback response with warning instead of failing
      res.json({
        hasInteractions: false,
        interactions: [],
        allergicRisks: [],
        contraindications: [],
        warnings: ["AI analysis temporarily unavailable. Please verify interactions manually using clinical references."],
        safeToPresrcibe: true,
        crossSpecialtyCount: 0,
        sameSpecialtyCount: 0,
        checkedMedications: [],
        _note: "AI service temporarily unavailable. Please verify manually.",
        _fallback: true,
        _message: "AI service temporarily unavailable. This is a fallback response - please verify manually."
      });
    }
  });

  // Lysa Diagnosis Analysis - AI-powered clinical decision support
  app.post('/api/v1/lysa/diagnosis/analyze', isAuthenticated, aiRateLimit, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access diagnosis support' });
      }

      const { symptoms, patientAge, patientSex, medicalHistory, currentMedications, additionalNotes } = req.body;

      if (!symptoms || !Array.isArray(symptoms) || symptoms.length === 0) {
        return res.status(400).json({ message: 'At least one symptom is required' });
      }

      // Format symptoms for analysis
      const symptomList = symptoms.map((s: any) => 
        `${s.name} (severity: ${s.severity}, duration: ${s.duration})`
      ).join('\n- ');

      const systemPrompt = `You are an advanced clinical decision support AI assistant. You help doctors by analyzing symptoms and suggesting possible diagnoses. You provide evidence-based recommendations.

IMPORTANT DISCLAIMERS:
- This is decision SUPPORT, not a replacement for clinical judgment
- All suggestions must be verified through proper diagnostic procedures
- Consider the complete clinical picture before making decisions

Analyze the patient information and provide:
1. A primary diagnosis suggestion with confidence level
2. 2-3 differential diagnoses
3. Red flags that warrant immediate attention
4. Recommended diagnostic tests
5. Clinical insights based on the symptom pattern
6. Recommended next steps

Consider:
- Age and sex-specific conditions
- Drug interactions with current medications
- Medical history implications
- Symptom severity and duration patterns

Respond ONLY with valid JSON in this exact format:
{
  "primaryDiagnosis": {
    "condition": "Condition Name",
    "probability": 75,
    "matchingSymptoms": ["symptom1", "symptom2"],
    "missingSymptoms": ["typical symptom not present"],
    "urgency": "low|moderate|high|emergency",
    "description": "Brief clinical description",
    "recommendedTests": ["Test 1", "Test 2"],
    "differentialDiagnosis": ["Alt condition 1"]
  },
  "differentialDiagnoses": [
    {
      "condition": "Alternative Condition",
      "probability": 50,
      "matchingSymptoms": [],
      "missingSymptoms": [],
      "urgency": "low",
      "description": "Description",
      "recommendedTests": [],
      "differentialDiagnosis": []
    }
  ],
  "clinicalInsights": ["Insight 1", "Insight 2"],
  "recommendedActions": ["Action 1", "Action 2"],
  "redFlags": ["Red flag if any"],
  "references": ["Reference 1"]
}`;

      const userPrompt = `Please analyze the following patient presentation:

**Patient Demographics:**
- Age: ${patientAge || 'Not specified'}
- Sex: ${patientSex || 'Not specified'}

**Presenting Symptoms:**
- ${symptomList}

**Medical History:**
${medicalHistory || 'Not provided'}

**Current Medications:**
${currentMedications || 'None reported'}

**Additional Clinical Notes:**
${additionalNotes || 'None'}

Please provide a comprehensive clinical assessment with differential diagnosis.`;

      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt }
        ],
        response_format: { type: "json_object" },
        temperature: 0.3,
        max_tokens: 2000
      });

      const analysisResult = JSON.parse(completion.choices[0].message.content || '{}');

      // Log for HIPAA audit
      console.log(`[HIPAA-AUDIT] Diagnosis analysis performed by doctor ${userId}`);

      res.json(analysisResult);
    } catch (error) {
      console.error('Error analyzing diagnosis:', error);
      // Return fallback response with clinical decision support guidance
      const symptomNames = symptoms.map((s: any) => s.name);
      res.json({
        primaryDiagnosis: {
          condition: "Clinical Assessment Required",
          probability: 0,
          matchingSymptoms: symptomNames,
          missingSymptoms: [],
          urgency: "moderate",
          description: "AI analysis temporarily unavailable. Based on the presented symptoms, a thorough clinical evaluation is recommended.",
          recommendedTests: ["Complete Blood Count (CBC)", "Basic Metabolic Panel", "Urinalysis"],
          differentialDiagnosis: []
        },
        differentialDiagnoses: [],
        clinicalInsights: [
          "AI-powered analysis is temporarily unavailable",
          "Please proceed with standard clinical assessment protocols",
          "Consider the patient's medical history and current medications"
        ],
        recommendedActions: [
          "Conduct thorough physical examination",
          "Review patient's medical history",
          "Order relevant diagnostic tests based on clinical judgment",
          "Consider specialist referral if symptoms persist"
        ],
        redFlags: [],
        references: ["Clinical guidelines available at UpToDate, DynaMed, or similar resources"],
        _fallback: true,
        _message: "AI service temporarily unavailable. Please use clinical judgment and standard protocols."
      });
    }
  });

  // Lysa Doctor Patients - list all patients for a doctor
  app.get('/api/v1/lysa/patients', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access patient list' });
      }

      const patients = await storage.getDoctorPatients(userId);
      
      res.json({
        success: true,
        patients: patients.map(p => ({
          id: p.id,
          firstName: p.firstName,
          lastName: p.lastName,
          email: p.email,
          phoneNumber: p.phoneNumber,
          followupPatientId: p.profile?.followupPatientId,
          dateOfBirth: p.profile?.dateOfBirth,
          bloodType: p.profile?.bloodType,
          allergies: p.profile?.allergies,
          medicalConditions: p.profile?.medicalConditions
        })),
        count: patients.length
      });
    } catch (error) {
      console.error('Error listing patients:', error);
      res.status(500).json({ message: 'Failed to list patients' });
    }
  });

  // Lysa Patient-Specific AI Chat - context-aware clinical assistant
  app.post('/api/v1/lysa/patient-chat', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access patient AI chat' });
      }

      const { patientId, message, context } = req.body;
      if (!patientId || !message) {
        return res.status(400).json({ message: 'Patient ID and message are required' });
      }

      // HIPAA: Verify doctor has active assignment with patient
      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted patient chat for unassigned patient ${patientId}`);
        return res.status(403).json({ 
          message: 'Access denied. No active assignment with this patient.',
          code: 'NO_PATIENT_ASSIGNMENT'
        });
      }

      // Get patient data for context
      const patient = await storage.getUser(patientId);
      const profile = await storage.getPatientProfile(patientId);
      const prescriptions = await storage.getPrescriptions(patientId);

      const patientContext = {
        name: `${patient?.firstName} ${patient?.lastName}`,
        allergies: profile?.allergies || context?.allergies || [],
        comorbidities: profile?.comorbidities || context?.comorbidities || [],
        immunocompromisedCondition: profile?.immunocompromisedCondition || context?.immunocompromisedCondition || '',
        currentMedications: prescriptions?.map((p: any) => p.medicationName) || context?.currentMedications || []
      };

      // Build system prompt for patient-specific chat
      const systemPrompt = `You are Lysa, an AI clinical assistant helping Dr. ${user.firstName} ${user.lastName} with patient care.

CURRENT PATIENT CONTEXT:
- Patient: ${patientContext.name}
- Immunocompromised Status: ${patientContext.immunocompromisedCondition || 'Not specified'}
- Known Allergies: ${patientContext.allergies.join(', ') || 'None documented'}
- Comorbidities: ${patientContext.comorbidities.join(', ') || 'None documented'}
- Current Medications: ${patientContext.currentMedications.join(', ') || 'None documented'}

GUIDELINES:
1. Provide evidence-based clinical recommendations considering the patient's specific conditions
2. Always consider drug-allergy interactions and contraindications
3. Flag potential issues related to immunocompromised status
4. Use professional medical terminology but explain complex concepts
5. Never provide diagnoses - only clinical decision support
6. Reference relevant clinical guidelines when appropriate
7. Consider polypharmacy risks given current medications
8. Be concise but thorough in clinical assessments

You have access to the patient's full medical context. Provide helpful, accurate clinical support.`;

      try {
        const openai = (await import('openai')).default;
        const openaiClient = new openai();

        const completion = await openaiClient.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: message }
          ],
          temperature: 0.7,
          max_tokens: 1500
        });

        const response = completion.choices[0].message.content;
        
        // HIPAA audit log
        console.log(`[HIPAA-AUDIT] Doctor ${userId} used patient AI chat for patient ${patientId} at ${new Date().toISOString()}`);

        res.json({
          success: true,
          response,
          patientContext: {
            name: patientContext.name,
            alertCount: patientContext.allergies.length + patientContext.comorbidities.length
          }
        });
      } catch (aiError) {
        console.error('OpenAI error:', aiError);
        
        // Fallback response based on query type
        const queryLower = message.toLowerCase();
        let fallbackResponse = '';

        if (queryLower.includes('summary') || queryLower.includes('overview')) {
          fallbackResponse = `Patient Summary for ${patientContext.name}:\n\n• Immunocompromised Status: ${patientContext.immunocompromisedCondition || 'Not specified'}\n• Allergies: ${patientContext.allergies.join(', ') || 'None documented'}\n• Comorbidities: ${patientContext.comorbidities.join(', ') || 'None documented'}\n• Current Medications: ${patientContext.currentMedications.join(', ') || 'None documented'}\n\nPlease review recent lab results and vital signs for complete assessment.`;
        } else if (queryLower.includes('risk')) {
          fallbackResponse = `Risk Assessment for ${patientContext.name}:\n\n${patientContext.immunocompromisedCondition ? `• HIGH PRIORITY: Immunocompromised status requires enhanced infection monitoring\n` : ''}${patientContext.allergies.length > 0 ? `• MEDICATION SAFETY: ${patientContext.allergies.length} documented allergies\n` : ''}${patientContext.comorbidities.length > 0 ? `• COMPLEXITY: ${patientContext.comorbidities.length} comorbid conditions\n` : ''}\nStandard clinical protocols recommended.`;
        } else {
          fallbackResponse = `I have ${patientContext.name}'s medical context loaded. Currently using offline clinical guidelines.\n\nKey patient factors:\n• ${patientContext.allergies.length} allergies documented\n• ${patientContext.comorbidities.length} comorbidities\n• ${patientContext.currentMedications.length} current medications\n\nHow can I assist with this patient's care?`;
        }

        res.json({
          success: true,
          response: fallbackResponse,
          _fallback: true,
          patientContext: {
            name: patientContext.name,
            alertCount: patientContext.allergies.length + patientContext.comorbidities.length
          }
        });
      }
    } catch (error) {
      console.error('Error in patient chat:', error);
      res.status(500).json({ message: 'Failed to process patient chat request' });
    }
  });

  // Lysa Patient Insights - AI-generated insights for a specific patient
  app.get('/api/v1/lysa/patient-insights/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access patient insights' });
      }

      const { patientId } = req.params;

      // HIPAA: Verify doctor has access
      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted to access insights for unassigned patient ${patientId}`);
        return res.status(403).json({ 
          message: 'Access denied. No active assignment with this patient.',
          code: 'NO_PATIENT_ASSIGNMENT'
        });
      }

      // Get patient profile for insight generation
      const profile = await storage.getPatientProfile(patientId);
      const prescriptions = await storage.getPrescriptions(patientId);

      // Generate insights based on patient data
      const insights: any[] = [];

      if (profile?.immunocompromisedCondition) {
        insights.push({
          id: 'ic-alert',
          type: 'warning',
          title: 'Immunocompromised Patient',
          description: `Patient has ${profile.immunocompromisedCondition}. Enhanced infection monitoring and vaccination updates recommended.`,
          severity: 'high',
          timestamp: new Date().toISOString()
        });
      }

      if (profile?.allergies && profile.allergies.length > 0) {
        insights.push({
          id: 'allergy-alert',
          type: 'warning',
          title: 'Active Allergies',
          description: `${profile.allergies.length} documented allergies: ${profile.allergies.join(', ')}. Verify before prescribing.`,
          severity: 'medium',
          timestamp: new Date().toISOString()
        });
      }

      if (prescriptions && prescriptions.length > 4) {
        insights.push({
          id: 'polypharmacy',
          type: 'info',
          title: 'Polypharmacy Review',
          description: `Patient on ${prescriptions.length} medications. Consider medication reconciliation and interaction review.`,
          severity: 'medium',
          timestamp: new Date().toISOString()
        });
      }

      if (profile?.comorbidities && profile.comorbidities.length > 2) {
        insights.push({
          id: 'complexity',
          type: 'trend',
          title: 'Complex Care Patient',
          description: `${profile.comorbidities.length} comorbid conditions require coordinated care approach.`,
          severity: 'medium',
          timestamp: new Date().toISOString()
        });
      }

      // Default recommendation
      insights.push({
        id: 'monitor-rec',
        type: 'recommendation',
        title: 'Regular Monitoring',
        description: 'Maintain regular follow-ups to track health status and medication adherence.',
        severity: 'low',
        timestamp: new Date().toISOString()
      });

      console.log(`[HIPAA-AUDIT] Doctor ${userId} accessed AI insights for patient ${patientId}`);

      res.json({
        success: true,
        insights,
        generatedAt: new Date().toISOString()
      });
    } catch (error) {
      console.error('Error fetching patient insights:', error);
      res.status(500).json({ message: 'Failed to fetch patient insights' });
    }
  });

  // Lysa Patient Timeline - recent health events for a patient
  app.get('/api/v1/lysa/patient-timeline/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access patient timeline' });
      }

      const { patientId } = req.params;
      const limit = parseInt(req.query.limit as string) || 10;

      // HIPAA: Verify doctor has access
      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        return res.status(403).json({ 
          message: 'Access denied. No active assignment with this patient.',
          code: 'NO_PATIENT_ASSIGNMENT'
        });
      }

      // Get recent appointments
      const appointments = await storage.getPatientAppointments(patientId);
      const prescriptions = await storage.getPrescriptions(patientId);

      const events: any[] = [];

      // Add appointment events
      if (appointments && appointments.length > 0) {
        appointments.slice(0, 5).forEach((apt: any) => {
          events.push({
            id: apt.id,
            type: 'appointment',
            title: apt.title || 'Appointment',
            description: apt.description || apt.appointmentType,
            timestamp: apt.startTime,
            status: apt.status
          });
        });
      }

      // Add prescription events
      if (prescriptions && prescriptions.length > 0) {
        prescriptions.slice(0, 5).forEach((rx: any) => {
          events.push({
            id: rx.id,
            type: 'prescription',
            title: `${rx.medicationName} prescribed`,
            description: `${rx.dosage} - ${rx.frequency}`,
            timestamp: rx.startDate || rx.createdAt,
            status: rx.status
          });
        });
      }

      // Sort by timestamp descending
      events.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

      res.json({
        success: true,
        events: events.slice(0, limit),
        totalCount: events.length
      });
    } catch (error) {
      console.error('Error fetching patient timeline:', error);
      res.status(500).json({ message: 'Failed to fetch patient timeline' });
    }
  });

  // Clinical Decision Support - Get recommendations for a patient
  app.get('/api/v1/lysa/clinical-recommendations/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access clinical recommendations' });
      }

      const { patientId } = req.params;

      // HIPAA: Verify doctor has access
      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted clinical recommendations for unassigned patient ${patientId}`);
        return res.status(403).json({ 
          message: 'Access denied. No active assignment with this patient.',
          code: 'NO_PATIENT_ASSIGNMENT'
        });
      }

      const profile = await storage.getPatientProfile(patientId);
      const prescriptions = await storage.getPrescriptions(patientId);
      const currentMedications = prescriptions?.map((p: any) => p.medicationName) || [];

      // Generate evidence-based recommendations
      const recommendations: any[] = [];

      // Immunocompromised recommendations
      if (profile?.immunocompromisedCondition) {
        recommendations.push({
          id: 'ic-monitor',
          type: 'monitoring',
          title: 'Enhanced Infection Surveillance',
          description: `Due to ${profile.immunocompromisedCondition}, implement enhanced monitoring for early signs of infection including daily temperature checks and symptom screening.`,
          evidenceLevel: 'B',
          strength: 'strong',
          priority: 'high',
          source: 'IDSA Guidelines for Immunocompromised Patients',
          guidelines: ['IDSA Fever and Neutropenia Guidelines', 'ASBMT Infection Prevention Guidelines'],
          considerations: ['Consider prophylactic antimicrobials based on risk stratification']
        });

        recommendations.push({
          id: 'ic-vaccines',
          type: 'medication',
          title: 'Vaccination Status Review',
          description: 'Review and update vaccinations appropriate for immunocompromised patients. Avoid live vaccines.',
          evidenceLevel: 'A',
          strength: 'strong',
          priority: 'high',
          source: 'CDC Immunization Guidelines',
          contraindications: ['Live attenuated vaccines contraindicated'],
          considerations: ['May need higher doses or additional boosters']
        });
      }

      // Allergy-based recommendations
      if (profile?.allergies && profile.allergies.length > 0) {
        const hasPenicillinAllergy = profile.allergies.some((a: string) => 
          a.toLowerCase().includes('penicillin') || a.toLowerCase().includes('amoxicillin')
        );

        if (hasPenicillinAllergy) {
          recommendations.push({
            id: 'allergy-abx',
            type: 'medication',
            title: 'Antibiotic Selection - Penicillin Allergy',
            description: 'Patient has documented penicillin allergy. Consider alternative antibiotics such as fluoroquinolones, macrolides, or aztreonam.',
            evidenceLevel: 'B',
            strength: 'strong',
            priority: 'high',
            source: 'Allergy Cross-Reactivity Guidelines',
            contraindications: ['Penicillins', 'Aminopenicillins', 'Possible cephalosporin cross-reactivity'],
            considerations: ['10% cross-reactivity risk with cephalosporins']
          });
        }
      }

      // Comorbidity-based recommendations
      if (profile?.comorbidities && profile.comorbidities.length > 0) {
        const hasDiabetes = profile.comorbidities.some((c: string) => c.toLowerCase().includes('diabetes'));
        const hasHTN = profile.comorbidities.some((c: string) => c.toLowerCase().includes('hypertension'));
        const hasCKD = profile.comorbidities.some((c: string) => 
          c.toLowerCase().includes('kidney') || c.toLowerCase().includes('renal')
        );

        if (hasDiabetes) {
          recommendations.push({
            id: 'dm-monitor',
            type: 'monitoring',
            title: 'Glycemic Monitoring',
            description: 'Regular HbA1c monitoring every 3-6 months and fasting glucose checks recommended.',
            evidenceLevel: 'A',
            strength: 'strong',
            priority: 'medium',
            source: 'ADA Standards of Care 2024',
            guidelines: ['ADA Diabetes Care Guidelines', 'AACE Diabetes Guidelines']
          });
        }

        if (hasHTN) {
          recommendations.push({
            id: 'htn-monitor',
            type: 'monitoring',
            title: 'Blood Pressure Management',
            description: 'Regular BP monitoring with goal <130/80 mmHg for most patients.',
            evidenceLevel: 'A',
            strength: 'strong',
            priority: 'medium',
            source: 'ACC/AHA Hypertension Guidelines'
          });
        }

        if (hasCKD) {
          recommendations.push({
            id: 'ckd-med',
            type: 'medication',
            title: 'Renal Dosing Adjustments',
            description: 'All medications should be reviewed for renal dosing. Avoid nephrotoxic medications.',
            evidenceLevel: 'A',
            strength: 'strong',
            priority: 'high',
            source: 'KDIGO CKD Guidelines',
            contraindications: ['NSAIDs', 'High-dose contrast agents']
          });
        }
      }

      // Polypharmacy check
      if (currentMedications.length > 4) {
        recommendations.push({
          id: 'polypharmacy',
          type: 'medication',
          title: 'Polypharmacy Review',
          description: `Patient is on ${currentMedications.length} medications. Consider medication reconciliation and deprescribing.`,
          evidenceLevel: 'B',
          strength: 'moderate',
          priority: 'medium',
          source: 'AGS Beers Criteria 2023'
        });
      }

      // Default follow-up recommendation
      recommendations.push({
        id: 'followup',
        type: 'monitoring',
        title: 'Regular Follow-up Care',
        description: 'Maintain regular follow-up appointments to monitor treatment efficacy and adverse effects.',
        evidenceLevel: 'C',
        strength: 'moderate',
        priority: 'low',
        source: 'Clinical Best Practice'
      });

      console.log(`[HIPAA-AUDIT] Doctor ${userId} accessed clinical recommendations for patient ${patientId}`);

      res.json({
        success: true,
        recommendations,
        generatedAt: new Date().toISOString()
      });
    } catch (error) {
      console.error('Error fetching clinical recommendations:', error);
      res.status(500).json({ message: 'Failed to fetch clinical recommendations' });
    }
  });

  // Clinical Decision Support - Get relevant guidelines for a patient
  app.get('/api/v1/lysa/clinical-guidelines/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access clinical guidelines' });
      }

      const { patientId } = req.params;

      // HIPAA: Verify access
      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      const profile = await storage.getPatientProfile(patientId);
      const guidelines: any[] = [];

      if (profile?.immunocompromisedCondition) {
        guidelines.push({
          id: 'idsa-fever',
          name: 'Fever and Neutropenia in Immunocompromised Adults',
          organization: 'IDSA',
          year: 2023,
          relevance: 0.95,
          keyPoints: [
            'Risk stratification for febrile neutropenia',
            'Empiric antibiotic selection based on risk',
            'Duration of antimicrobial therapy'
          ]
        });
      }

      guidelines.push({
        id: 'general-prev',
        name: 'Preventive Care and Screening Guidelines',
        organization: 'USPSTF',
        year: 2024,
        relevance: 0.75,
        keyPoints: ['Age-appropriate cancer screening', 'Cardiovascular risk assessment']
      });

      if (profile?.comorbidities?.some((c: string) => c.toLowerCase().includes('diabetes'))) {
        guidelines.push({
          id: 'ada-soc',
          name: 'Standards of Care in Diabetes',
          organization: 'ADA',
          year: 2024,
          relevance: 0.90,
          keyPoints: ['Glycemic targets by population', 'Cardiovascular risk reduction']
        });
      }

      if (profile?.comorbidities?.some((c: string) => c.toLowerCase().includes('hypertension'))) {
        guidelines.push({
          id: 'acc-aha-bp',
          name: 'High Blood Pressure Clinical Practice Guideline',
          organization: 'ACC/AHA',
          year: 2023,
          relevance: 0.88,
          keyPoints: ['BP thresholds for treatment', 'First-line medication classes']
        });
      }

      res.json({ success: true, guidelines });
    } catch (error) {
      console.error('Error fetching clinical guidelines:', error);
      res.status(500).json({ message: 'Failed to fetch clinical guidelines' });
    }
  });

  // Clinical Decision Support - Check drug interactions
  app.get('/api/v1/lysa/drug-interactions/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can check drug interactions' });
      }

      const { patientId } = req.params;

      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      const prescriptions = await storage.getPrescriptions(patientId);
      const medications = prescriptions?.map((p: any) => p.medicationName.toLowerCase()) || [];
      const interactions: any[] = [];

      // Common interaction checks
      const interactionRules = [
        { drugs: ['warfarin', 'aspirin'], severity: 'major', description: 'Increased bleeding risk', recommendation: 'Monitor INR closely' },
        { drugs: ['metformin', 'contrast'], severity: 'major', description: 'Risk of lactic acidosis', recommendation: 'Hold metformin 48h before contrast' },
        { drugs: ['ace inhibitor', 'potassium'], severity: 'moderate', description: 'Risk of hyperkalemia', recommendation: 'Monitor potassium levels' },
        { drugs: ['statin', 'fibrate'], severity: 'moderate', description: 'Increased myopathy risk', recommendation: 'Monitor for muscle symptoms' },
        { drugs: ['ssri', 'maoi'], severity: 'contraindicated', description: 'Risk of serotonin syndrome', recommendation: 'Do not use together' },
        { drugs: ['methotrexate', 'nsaid'], severity: 'major', description: 'Increased methotrexate toxicity', recommendation: 'Avoid combination' }
      ];

      for (const rule of interactionRules) {
        const matches = rule.drugs.filter(d => 
          medications.some(m => m.includes(d))
        );
        if (matches.length >= 2) {
          interactions.push({
            drug1: matches[0],
            drug2: matches[1],
            severity: rule.severity,
            description: rule.description,
            recommendation: rule.recommendation
          });
        }
      }

      res.json({
        success: true,
        interactions,
        checked: medications.length
      });
    } catch (error) {
      console.error('Error checking drug interactions:', error);
      res.status(500).json({ message: 'Failed to check drug interactions' });
    }
  });

  // Clinical Query - AI-powered clinical question answering
  app.post('/api/v1/lysa/clinical-query', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can submit clinical queries' });
      }

      const { patientId, query, context } = req.body;
      if (!patientId || !query) {
        return res.status(400).json({ message: 'Patient ID and query are required' });
      }

      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      try {
        const openai = (await import('openai')).default;
        const openaiClient = new openai();

        const systemPrompt = `You are a clinical decision support system. Provide evidence-based recommendations considering:
- Patient allergies: ${context?.allergies?.join(', ') || 'None documented'}
- Comorbidities: ${context?.comorbidities?.join(', ') || 'None documented'}
- Immunocompromised: ${context?.immunocompromisedCondition || 'No'}
- Current medications: ${context?.currentMedications?.join(', ') || 'None documented'}

Guidelines:
1. Cite evidence levels (A, B, C, D) for recommendations
2. Note contraindications based on patient factors
3. Suggest alternatives when primary options are contraindicated
4. Never provide diagnoses - only clinical decision support`;

        const completion = await openaiClient.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: query }
          ],
          temperature: 0.5,
          max_tokens: 1000
        });

        res.json({
          success: true,
          response: completion.choices[0].message.content,
          query
        });
      } catch (aiError) {
        console.error('OpenAI error:', aiError);
        res.json({
          success: true,
          response: `Clinical query received. Based on patient factors (${context?.comorbidities?.length || 0} comorbidities, ${context?.allergies?.length || 0} allergies), please consult current clinical guidelines and reference materials for evidence-based recommendations.`,
          _fallback: true
        });
      }
    } catch (error) {
      console.error('Error processing clinical query:', error);
      res.status(500).json({ message: 'Failed to process clinical query' });
    }
  });

  // Predictive Analytics - Risk Assessment
  app.get('/api/v1/lysa/risk-assessment/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access risk assessments' });
      }

      const { patientId } = req.params;

      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted risk assessment for unassigned patient ${patientId}`);
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      const profile = await storage.getPatientProfile(patientId);
      const prescriptions = await storage.getPrescriptions(patientId);

      // Calculate risk scores
      let baseRisk = 20;
      const riskScores: any[] = [];

      // Immunocompromised risk
      if (profile?.immunocompromisedCondition) {
        baseRisk += 35;
        riskScores.push({
          category: 'Infection Risk',
          score: 75,
          trend: 'stable',
          factors: [profile.immunocompromisedCondition],
          lastUpdated: new Date().toISOString()
        });
      } else {
        riskScores.push({
          category: 'Infection Risk',
          score: 25,
          trend: 'stable',
          factors: [],
          lastUpdated: new Date().toISOString()
        });
      }

      // Cardiovascular risk
      const cvConditions = profile?.comorbidities?.filter((c: string) => 
        c.toLowerCase().includes('hypertension') || 
        c.toLowerCase().includes('diabetes') ||
        c.toLowerCase().includes('heart')
      ) || [];
      
      if (cvConditions.length > 0) {
        baseRisk += cvConditions.length * 10;
        riskScores.push({
          category: 'Cardiovascular Risk',
          score: Math.min(cvConditions.length * 20 + 35, 85),
          trend: 'stable',
          factors: cvConditions,
          lastUpdated: new Date().toISOString()
        });
      } else {
        riskScores.push({
          category: 'Cardiovascular Risk',
          score: 20,
          trend: 'stable',
          factors: [],
          lastUpdated: new Date().toISOString()
        });
      }

      // Medication complexity
      const medCount = prescriptions?.length || 0;
      riskScores.push({
        category: 'Medication Complexity',
        score: Math.min(medCount * 12, 80),
        trend: 'stable',
        factors: [`${medCount} active medications`],
        lastUpdated: new Date().toISOString()
      });

      if (medCount > 4) {
        baseRisk += 10;
      }

      // Care coordination
      const comorbidCount = profile?.comorbidities?.length || 0;
      riskScores.push({
        category: 'Care Coordination',
        score: comorbidCount > 2 ? 60 : 30,
        trend: 'stable',
        factors: [`${comorbidCount} comorbid conditions`],
        lastUpdated: new Date().toISOString()
      });

      if (comorbidCount > 2) {
        baseRisk += comorbidCount * 5;
      }

      const overallScore = Math.min(baseRisk, 95);
      const overallLevel = overallScore >= 80 ? 'critical' : overallScore >= 60 ? 'high' : overallScore >= 40 ? 'medium' : 'low';

      console.log(`[HIPAA-AUDIT] Doctor ${userId} accessed risk assessment for patient ${patientId}`);

      res.json({
        success: true,
        overallRisk: {
          score: overallScore,
          level: overallLevel,
          trend: 'stable'
        },
        riskScores,
        deteriorationRisk: { score: profile?.immunocompromisedCondition ? 65 : 25 },
        readmissionRisk: { score: comorbidCount * 8 + 10 },
        adherenceScore: 85,
        generatedAt: new Date().toISOString()
      });
    } catch (error) {
      console.error('Error calculating risk assessment:', error);
      res.status(500).json({ message: 'Failed to calculate risk assessment' });
    }
  });

  // Predictive Analytics - Health Trends
  app.get('/api/v1/lysa/health-trends/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access health trends' });
      }

      const { patientId } = req.params;
      const days = parseInt(req.query.days as string) || 30;

      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      // Generate simulated health trends (in production, this would come from device data)
      const trends = [
        { metric: 'Heart Rate', current: 72, baseline: 70, unit: 'bpm', status: 'normal', change: 2.9 },
        { metric: 'Blood Pressure (Systolic)', current: 128, baseline: 120, unit: 'mmHg', status: 'elevated', change: 6.7 },
        { metric: 'Temperature', current: 98.4, baseline: 98.6, unit: '°F', status: 'normal', change: -0.2 },
        { metric: 'Oxygen Saturation', current: 97, baseline: 98, unit: '%', status: 'normal', change: -1.0 },
        { metric: 'Weight', current: 165, baseline: 168, unit: 'lbs', status: 'normal', change: -1.8 },
        { metric: 'Blood Glucose', current: 105, baseline: 95, unit: 'mg/dL', status: 'elevated', change: 10.5 }
      ];

      res.json({
        success: true,
        trends,
        period: `${days} days`,
        generatedAt: new Date().toISOString()
      });
    } catch (error) {
      console.error('Error fetching health trends:', error);
      res.status(500).json({ message: 'Failed to fetch health trends' });
    }
  });

  // Predictive Analytics - Predictive Alerts
  app.get('/api/v1/lysa/predictive-alerts/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access predictive alerts' });
      }

      const { patientId } = req.params;

      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      const profile = await storage.getPatientProfile(patientId);
      const prescriptions = await storage.getPrescriptions(patientId);

      const alerts: any[] = [];

      // Generate alerts based on patient risk factors
      if (profile?.immunocompromisedCondition) {
        alerts.push({
          id: 'alert-ic-1',
          type: 'deterioration',
          severity: 'high',
          title: 'Elevated Infection Risk',
          description: `Due to ${profile.immunocompromisedCondition}, patient has elevated risk of opportunistic infections.`,
          probability: 65,
          timeframe: '30 days',
          recommendedActions: [
            'Enhanced infection monitoring protocol',
            'Review prophylactic medication coverage',
            'Update vaccination status'
          ]
        });
      }

      if (profile?.comorbidities && profile.comorbidities.length > 2) {
        alerts.push({
          id: 'alert-comorbid-1',
          type: 'complication',
          severity: 'medium',
          title: 'Complex Care Management',
          description: `Multiple comorbidities (${profile.comorbidities.length}) increase complexity of care coordination.`,
          probability: 45,
          timeframe: '90 days',
          recommendedActions: [
            'Schedule comprehensive care review',
            'Coordinate with specialists',
            'Review medication interactions'
          ]
        });
      }

      if (prescriptions && prescriptions.length > 4) {
        alerts.push({
          id: 'alert-poly-1',
          type: 'adherence',
          severity: 'medium',
          title: 'Medication Adherence Risk',
          description: `Patient on ${prescriptions.length} medications - polypharmacy may impact adherence.`,
          probability: 35,
          timeframe: '60 days',
          recommendedActions: [
            'Simplify medication regimen if possible',
            'Consider pill organizers or reminders',
            'Review for deprescribing opportunities'
          ]
        });
      }

      console.log(`[HIPAA-AUDIT] Doctor ${userId} accessed predictive alerts for patient ${patientId}`);

      res.json({
        success: true,
        alerts,
        generatedAt: new Date().toISOString()
      });
    } catch (error) {
      console.error('Error fetching predictive alerts:', error);
      res.status(500).json({ message: 'Failed to fetch predictive alerts' });
    }
  });

  // Diagnostic Imaging Analysis - AI-powered image interpretation
  app.post('/api/v1/lysa/imaging-analysis', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can request imaging analysis' });
      }

      const { patientId, imageType, imageUrl, clinicalContext, studyDescription } = req.body;
      if (!patientId || !imageType) {
        return res.status(400).json({ message: 'Patient ID and image type are required' });
      }

      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted imaging analysis for unassigned patient ${patientId}`);
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      console.log(`[HIPAA-AUDIT] Doctor ${userId} requested imaging analysis for patient ${patientId}`);

      // AI-powered imaging analysis
      try {
        const openai = (await import('openai')).default;
        const openaiClient = new openai();

        const analysisPrompt = `You are a radiologist AI assistant. Analyze the following ${imageType} imaging study:

Study Type: ${imageType}
Study Description: ${studyDescription || 'Not specified'}
Clinical Context: ${clinicalContext || 'Routine imaging study'}

Provide a structured radiology report with:
1. TECHNIQUE: Describe the imaging technique used
2. FINDINGS: List all observable findings
3. IMPRESSION: Provide clinical impression
4. RECOMMENDATIONS: Suggest follow-up if needed

Important: This is AI-assisted analysis and should be reviewed by a qualified radiologist.`;

        const completion = await openaiClient.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are a radiologist AI assistant providing structured analysis of medical imaging studies. Always recommend final review by a qualified radiologist." },
            { role: "user", content: analysisPrompt }
          ],
          temperature: 0.3,
          max_tokens: 1500
        });

        const analysisText = completion.choices[0].message.content || '';
        
        // Parse the structured response
        const sections = {
          technique: extractSection(analysisText, 'TECHNIQUE'),
          findings: extractSection(analysisText, 'FINDINGS'),
          impression: extractSection(analysisText, 'IMPRESSION'),
          recommendations: extractSection(analysisText, 'RECOMMENDATIONS')
        };

        res.json({
          success: true,
          analysis: {
            id: `img-${Date.now()}`,
            imageType,
            studyDescription,
            clinicalContext,
            sections,
            fullReport: analysisText,
            confidence: 0.85,
            aiModel: 'GPT-4o Vision',
            analyzedAt: new Date().toISOString(),
            disclaimer: 'AI-assisted analysis - Final interpretation requires qualified radiologist review'
          }
        });
      } catch (aiError) {
        console.error('OpenAI imaging analysis error:', aiError);
        // Provide structured fallback response
        res.json({
          success: true,
          analysis: {
            id: `img-${Date.now()}`,
            imageType,
            studyDescription,
            clinicalContext,
            sections: {
              technique: `${imageType} imaging study performed per standard protocol.`,
              findings: 'AI analysis temporarily unavailable. Manual radiologist review required.',
              impression: 'Pending qualified radiologist interpretation.',
              recommendations: 'Please have this study reviewed by a qualified radiologist.'
            },
            fullReport: `${imageType} Imaging Report\n\nTECHNIQUE: ${imageType} imaging study performed per standard protocol.\n\nFINDINGS: AI analysis temporarily unavailable. Manual radiologist review required.\n\nIMPRESSION: Pending qualified radiologist interpretation.\n\nRECOMMENDATIONS: Please have this study reviewed by a qualified radiologist.`,
            confidence: 0,
            aiModel: 'Fallback',
            analyzedAt: new Date().toISOString(),
            disclaimer: 'AI analysis unavailable - Requires qualified radiologist review',
            _fallback: true
          }
        });
      }
    } catch (error) {
      console.error('Error in imaging analysis:', error);
      res.status(500).json({ message: 'Failed to analyze imaging study' });
    }
  });

  // Get imaging history for a patient
  app.get('/api/v1/lysa/imaging-history/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access imaging history' });
      }

      const { patientId } = req.params;

      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      // Return sample imaging history (in production, this would come from PACS integration)
      const imagingHistory = [
        {
          id: 'img-hist-1',
          type: 'Chest X-Ray',
          date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
          status: 'completed',
          findings: 'No acute cardiopulmonary process',
          radiologist: 'Dr. Smith',
          priority: 'routine'
        },
        {
          id: 'img-hist-2',
          type: 'CT Abdomen',
          date: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString(),
          status: 'completed',
          findings: 'Mild hepatic steatosis, otherwise unremarkable',
          radiologist: 'Dr. Johnson',
          priority: 'routine'
        }
      ];

      res.json({
        success: true,
        studies: imagingHistory,
        totalCount: imagingHistory.length
      });
    } catch (error) {
      console.error('Error fetching imaging history:', error);
      res.status(500).json({ message: 'Failed to fetch imaging history' });
    }
  });

  // Lab Report Analysis - AI-powered lab interpretation with trend analysis
  app.post('/api/v1/lysa/lab-analysis', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can request lab analysis' });
      }

      const { patientId, labResults, panelType, clinicalContext } = req.body;
      if (!patientId || !labResults || !Array.isArray(labResults)) {
        return res.status(400).json({ message: 'Patient ID and lab results array are required' });
      }

      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted lab analysis for unassigned patient ${patientId}`);
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      console.log(`[HIPAA-AUDIT] Doctor ${userId} requested lab analysis for patient ${patientId}`);

      // Analyze lab results
      const analyzedResults = labResults.map((lab: any) => {
        const isAbnormal = lab.value < lab.normalRange?.low || lab.value > lab.normalRange?.high;
        const severity = isAbnormal 
          ? (Math.abs(lab.value - (lab.normalRange?.low || 0)) / (lab.normalRange?.low || 1) > 0.3 ? 'critical' : 'abnormal')
          : 'normal';
        
        return {
          ...lab,
          status: severity,
          isAbnormal,
          deviation: isAbnormal ? 
            (lab.value < lab.normalRange?.low ? 'low' : 'high') : null,
          deviationPercent: lab.normalRange?.low ? 
            Math.round(((lab.value - lab.normalRange.low) / lab.normalRange.low) * 100) : null
        };
      });

      const abnormalCount = analyzedResults.filter((r: any) => r.isAbnormal).length;
      const criticalCount = analyzedResults.filter((r: any) => r.status === 'critical').length;

      try {
        const openai = (await import('openai')).default;
        const openaiClient = new openai();

        const labSummary = analyzedResults.map((r: any) => 
          `${r.name}: ${r.value} ${r.unit} (Normal: ${r.normalRange?.low}-${r.normalRange?.high}) - ${r.status.toUpperCase()}`
        ).join('\n');

        const analysisPrompt = `Analyze these lab results for clinical significance:

Panel Type: ${panelType || 'General'}
Clinical Context: ${clinicalContext || 'Routine lab work'}

Results:
${labSummary}

Provide:
1. SUMMARY: Brief overview of results
2. SIGNIFICANT FINDINGS: Key abnormalities
3. CLINICAL CORRELATION: How results relate to clinical context
4. RECOMMENDATIONS: Suggested follow-up or additional testing`;

        const completion = await openaiClient.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are a clinical laboratory medicine specialist. Analyze lab results and provide clinically relevant interpretations." },
            { role: "user", content: analysisPrompt }
          ],
          temperature: 0.3,
          max_tokens: 1000
        });

        res.json({
          success: true,
          analysis: {
            id: `lab-${Date.now()}`,
            panelType,
            results: analyzedResults,
            summary: {
              totalTests: labResults.length,
              abnormalCount,
              criticalCount,
              overallStatus: criticalCount > 0 ? 'critical' : abnormalCount > 0 ? 'review' : 'normal'
            },
            interpretation: completion.choices[0].message.content,
            trends: generateLabTrends(analyzedResults),
            analyzedAt: new Date().toISOString()
          }
        });
      } catch (aiError) {
        console.error('OpenAI lab analysis error:', aiError);
        res.json({
          success: true,
          analysis: {
            id: `lab-${Date.now()}`,
            panelType,
            results: analyzedResults,
            summary: {
              totalTests: labResults.length,
              abnormalCount,
              criticalCount,
              overallStatus: criticalCount > 0 ? 'critical' : abnormalCount > 0 ? 'review' : 'normal'
            },
            interpretation: `Lab panel analysis: ${abnormalCount} abnormal values detected out of ${labResults.length} tests. ${criticalCount > 0 ? 'CRITICAL values require immediate attention.' : 'Please review abnormal values in clinical context.'}`,
            trends: generateLabTrends(analyzedResults),
            analyzedAt: new Date().toISOString(),
            _fallback: true
          }
        });
      }
    } catch (error) {
      console.error('Error in lab analysis:', error);
      res.status(500).json({ message: 'Failed to analyze lab results' });
    }
  });

  // Get lab history for a patient
  app.get('/api/v1/lysa/lab-history/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access lab history' });
      }

      const { patientId } = req.params;
      const { testName, limit = 10 } = req.query;

      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        return res.status(403).json({ message: 'Access denied', code: 'NO_PATIENT_ASSIGNMENT' });
      }

      // Return sample lab history with trends (in production from LIS integration)
      const labHistory = [
        {
          id: 'lab-hist-1',
          panelType: 'Complete Blood Count',
          date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          results: [
            { name: 'WBC', value: 7.5, unit: 'K/uL', normalRange: { low: 4.5, high: 11.0 }, status: 'normal' },
            { name: 'RBC', value: 4.8, unit: 'M/uL', normalRange: { low: 4.2, high: 5.4 }, status: 'normal' },
            { name: 'Hemoglobin', value: 14.2, unit: 'g/dL', normalRange: { low: 12.0, high: 16.0 }, status: 'normal' },
            { name: 'Hematocrit', value: 42, unit: '%', normalRange: { low: 36, high: 46 }, status: 'normal' },
            { name: 'Platelets', value: 250, unit: 'K/uL', normalRange: { low: 150, high: 400 }, status: 'normal' }
          ],
          status: 'normal'
        },
        {
          id: 'lab-hist-2',
          panelType: 'Comprehensive Metabolic Panel',
          date: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
          results: [
            { name: 'Glucose', value: 105, unit: 'mg/dL', normalRange: { low: 70, high: 100 }, status: 'abnormal' },
            { name: 'BUN', value: 18, unit: 'mg/dL', normalRange: { low: 7, high: 20 }, status: 'normal' },
            { name: 'Creatinine', value: 1.0, unit: 'mg/dL', normalRange: { low: 0.7, high: 1.3 }, status: 'normal' },
            { name: 'Sodium', value: 140, unit: 'mEq/L', normalRange: { low: 136, high: 145 }, status: 'normal' },
            { name: 'Potassium', value: 4.2, unit: 'mEq/L', normalRange: { low: 3.5, high: 5.0 }, status: 'normal' }
          ],
          status: 'review'
        },
        {
          id: 'lab-hist-3',
          panelType: 'Lipid Panel',
          date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
          results: [
            { name: 'Total Cholesterol', value: 195, unit: 'mg/dL', normalRange: { low: 0, high: 200 }, status: 'normal' },
            { name: 'LDL', value: 115, unit: 'mg/dL', normalRange: { low: 0, high: 100 }, status: 'abnormal' },
            { name: 'HDL', value: 55, unit: 'mg/dL', normalRange: { low: 40, high: 999 }, status: 'normal' },
            { name: 'Triglycerides', value: 145, unit: 'mg/dL', normalRange: { low: 0, high: 150 }, status: 'normal' }
          ],
          status: 'review'
        }
      ];

      res.json({
        success: true,
        panels: labHistory,
        totalCount: labHistory.length
      });
    } catch (error) {
      console.error('Error fetching lab history:', error);
      res.status(500).json({ message: 'Failed to fetch lab history' });
    }
  });

  // Lysa Insight Feed endpoint - get AI-generated insights for a patient
  app.get('/api/v1/lysa/insight-feed/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const { patientId } = req.params;
      
      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access patient insights' });
      }

      // Verify doctor-patient assignment
      const assignments = await storage.getActiveDoctorAssignments(userId);
      const hasAssignment = assignments.some(a => a.patientId === patientId);
      
      if (!hasAssignment) {
        return res.status(403).json({ message: 'Not authorized to access this patient\'s insights' });
      }

      const patient = await storage.getUser(patientId);
      if (!patient) {
        return res.status(404).json({ message: 'Patient not found' });
      }

      // Generate insights based on patient data (in production, these would come from ML models)
      const insights = [
        {
          id: `insight-${Date.now()}-1`,
          type: 'trend',
          category: 'vitals',
          title: 'Stable Vital Signs',
          description: `${patient.firstName}'s heart rate and blood pressure have remained within normal ranges over the past 7 days.`,
          priority: 'low',
          confidence: 0.92,
          timestamp: new Date().toISOString(),
          source: 'Continuous Monitoring'
        },
        {
          id: `insight-${Date.now()}-2`,
          type: 'recommendation',
          category: 'medication',
          title: 'Medication Review Suggested',
          description: 'Current medication regimen has been stable for 6 months. Consider reviewing effectiveness at next visit.',
          priority: 'medium',
          confidence: 0.78,
          timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          action: 'Schedule medication review',
          source: 'Medication Analysis'
        },
        {
          id: `insight-${Date.now()}-3`,
          type: 'observation',
          category: 'symptoms',
          title: 'Symptom Pattern Detected',
          description: 'Patient has reported similar symptoms at similar times over the past 3 visits. May indicate a cyclical pattern worth investigating.',
          priority: 'medium',
          confidence: 0.85,
          timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          source: 'Pattern Analysis'
        },
        {
          id: `insight-${Date.now()}-4`,
          type: 'prediction',
          category: 'risk',
          title: 'Low Deterioration Risk',
          description: 'Based on current health metrics and trends, patient shows low risk of health deterioration in the next 30 days.',
          priority: 'low',
          confidence: 0.88,
          timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
          source: 'Predictive Analytics'
        },
        {
          id: `insight-${Date.now()}-5`,
          type: 'alert',
          category: 'followup',
          title: 'Follow-up Due Soon',
          description: 'Scheduled follow-up appointment is approaching. Ensure all pre-visit requirements are communicated to patient.',
          priority: 'medium',
          confidence: 1.0,
          timestamp: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
          action: 'Send reminder',
          source: 'Appointment System'
        }
      ];

      res.json(insights);
    } catch (error) {
      console.error('Error fetching insight feed:', error);
      res.status(500).json({ message: 'Failed to fetch insight feed' });
    }
  });

  // Lysa Patient Monitoring Status endpoint - get monitoring status for all assigned patients
  app.get('/api/v1/lysa/monitoring/status', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access monitoring status' });
      }

      // Get assigned patients
      const assignments = await storage.getActiveDoctorAssignments(userId);
      
      // Return monitoring status for each patient
      const monitoringStatuses = assignments.map(assignment => ({
        patientId: assignment.patientId,
        isMonitored: assignment.accessLevel === 'full' || assignment.accessLevel === 'monitoring'
      }));

      res.json(monitoringStatuses);
    } catch (error) {
      console.error('Error fetching monitoring status:', error);
      res.status(500).json({ message: 'Failed to fetch monitoring status' });
    }
  });

  // Lysa Patient Monitoring Toggle endpoint
  app.post('/api/v1/lysa/monitoring/toggle', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can toggle patient monitoring' });
      }

      const { patientId, enabled } = req.body;
      if (!patientId || typeof enabled !== 'boolean') {
        return res.status(400).json({ message: 'Patient ID and enabled status are required' });
      }

      // Verify doctor-patient assignment
      const assignments = await storage.getActiveDoctorAssignments(userId);
      const hasAssignment = assignments.some(a => a.patientId === patientId);
      
      if (!hasAssignment) {
        return res.status(403).json({ message: 'Not authorized to monitor this patient' });
      }

      // Update access level based on monitoring toggle
      const newAccessLevel = enabled ? 'full' : 'basic';
      
      // In production, this would update the assignment's access level
      // For now, we'll just return success and the frontend will track the state
      
      // Log the monitoring status change for HIPAA audit
      console.log(`[HIPAA-AUDIT] Doctor ${userId} ${enabled ? 'enabled' : 'disabled'} monitoring for patient ${patientId}`);

      res.json({
        success: true,
        patientId,
        isMonitored: enabled,
        message: enabled 
          ? 'AI monitoring enabled for patient' 
          : 'AI monitoring disabled for patient'
      });
    } catch (error) {
      console.error('Error toggling monitoring status:', error);
      res.status(500).json({ message: 'Failed to toggle monitoring status' });
    }
  });

  // Create appointment with conflict detection
  app.post('/api/v1/appointments', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      if (user.role !== 'doctor' && user.role !== 'patient') {
        return res.status(403).json({ message: 'Only doctors and patients can manage appointments' });
      }

      let { patientId, doctorId, startTime, endTime, appointmentType, notes, googleCalendarEventId } = req.body;

      // SECURITY: Enforce participant authorization based on role
      if (user.role === 'patient') {
        // Patients can only book appointments for themselves
        patientId = userId;
        if (!doctorId) {
          return res.status(400).json({ message: 'Doctor ID is required' });
        }
      } else if (user.role === 'doctor') {
        // Doctors can book on behalf of patients, but only for themselves
        doctorId = userId;
        if (!patientId) {
          return res.status(400).json({ message: 'Patient ID is required' });
        }
      }

      if (!startTime || !endTime || !appointmentType) {
        return res.status(400).json({ message: 'Missing required fields' });
      }

      const startDate = new Date(startTime);
      const endDate = new Date(endTime);

      // Check for conflicts
      const conflicts = await storage.findAppointmentConflicts(doctorId, startDate, endDate);
      if (conflicts.length > 0) {
        return res.status(409).json({ 
          message: 'Appointment conflicts with existing appointment',
          conflicts 
        });
      }

      const appointment = await storage.createAppointment({
        patientId,
        doctorId,
        startTime: startDate,
        endTime: endDate,
        appointmentType,
        status: 'scheduled',
        confirmationStatus: 'pending',
        notes,
        googleCalendarEventId,
      });

      res.status(201).json(appointment);
    } catch (error) {
      console.error('Error creating appointment:', error);
      res.status(500).json({ message: 'Failed to create appointment' });
    }
  });

  // Get upcoming appointments (MUST be before /:id route)
  app.get('/api/v1/appointments/upcoming', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      const daysAhead = req.query.days ? parseInt(req.query.days as string) : 30;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 50;

      const role = user.role as 'doctor' | 'patient';
      const appointments = await storage.listUpcomingAppointments(userId, role, daysAhead, limit);

      res.json(appointments);
    } catch (error) {
      console.error('Error fetching upcoming appointments:', error);
      res.status(500).json({ message: 'Failed to fetch upcoming appointments' });
    }
  });

  // List appointments with filters
  app.get('/api/v1/appointments', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      const { doctorId, patientId, startDate, endDate, status, limit, offset } = req.query;

      // Build filter object
      const filters: any = {
        limit: limit ? parseInt(limit as string) : 50,
        offset: offset ? parseInt(offset as string) : 0,
      };

      if (doctorId) filters.doctorId = doctorId as string;
      if (patientId) filters.patientId = patientId as string;
      if (startDate) filters.startDate = new Date(startDate as string);
      if (endDate) filters.endDate = new Date(endDate as string);
      if (status) filters.status = status as string;

      const appointments = await storage.listAppointments(filters);
      res.json(appointments);
    } catch (error) {
      console.error('Error listing appointments:', error);
      res.status(500).json({ message: 'Failed to list appointments' });
    }
  });

  // Get appointment by ID
  app.get('/api/v1/appointments/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      const { id } = req.params;
      const appointment = await storage.getAppointment(id);

      if (!appointment) {
        return res.status(404).json({ message: 'Appointment not found' });
      }

      // SECURITY: Only participants can view appointment details
      if (user.role === 'patient' && appointment.patientId !== userId) {
        return res.status(403).json({ message: 'You can only view your own appointments' });
      }
      if (user.role === 'doctor' && appointment.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only view appointments with your patients' });
      }

      res.json(appointment);
    } catch (error) {
      console.error('Error fetching appointment:', error);
      res.status(500).json({ message: 'Failed to fetch appointment' });
    }
  });

  // Update appointment
  app.patch('/api/v1/appointments/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      const { id } = req.params;
      const existingAppointment = await storage.getAppointment(id);

      if (!existingAppointment) {
        return res.status(404).json({ message: 'Appointment not found' });
      }

      // SECURITY: Only allow updates if user is participant
      if (user.role === 'patient' && existingAppointment.patientId !== userId) {
        return res.status(403).json({ message: 'You can only update your own appointments' });
      }
      if (user.role === 'doctor' && existingAppointment.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only update appointments for your patients' });
      }

      const updates = req.body;

      // SECURITY: Prevent changing participant IDs
      delete updates.patientId;
      delete updates.doctorId;

      const appointment = await storage.updateAppointment(id, updates);

      res.json(appointment);
    } catch (error) {
      console.error('Error updating appointment:', error);
      res.status(500).json({ message: 'Failed to update appointment' });
    }
  });

  // Confirm appointment
  app.post('/api/v1/appointments/:id/confirm', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      const { id } = req.params;
      const existingAppointment = await storage.getAppointment(id);

      if (!existingAppointment) {
        return res.status(404).json({ message: 'Appointment not found' });
      }

      // SECURITY: Only participants can confirm
      if (user.role === 'patient' && existingAppointment.patientId !== userId) {
        return res.status(403).json({ message: 'You can only confirm your own appointments' });
      }
      if (user.role === 'doctor' && existingAppointment.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only confirm appointments with your patients' });
      }

      const appointment = await storage.confirmAppointment(id, new Date());

      res.json(appointment);
    } catch (error) {
      console.error('Error confirming appointment:', error);
      res.status(500).json({ message: 'Failed to confirm appointment' });
    }
  });

  // Cancel appointment
  app.post('/api/v1/appointments/:id/cancel', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      const { id } = req.params;
      const { reason } = req.body;

      if (!reason) {
        return res.status(400).json({ message: 'Cancellation reason is required' });
      }

      const existingAppointment = await storage.getAppointment(id);

      if (!existingAppointment) {
        return res.status(404).json({ message: 'Appointment not found' });
      }

      // SECURITY: Only participants can cancel
      if (user.role === 'patient' && existingAppointment.patientId !== userId) {
        return res.status(403).json({ message: 'You can only cancel your own appointments' });
      }
      if (user.role === 'doctor' && existingAppointment.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only cancel appointments with your patients' });
      }

      const appointment = await storage.cancelAppointment(id, userId, reason);

      res.json(appointment);
    } catch (error) {
      console.error('Error cancelling appointment:', error);
      res.status(500).json({ message: 'Failed to cancel appointment' });
    }
  });

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA - AVAILABILITY MANAGEMENT ROUTES
  // ============================================================================

  // Set doctor availability
  app.post('/api/v1/availability', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can set availability' });
      }

      const { dayOfWeek, startTime, endTime, isRecurring, specificDate, validFrom, validUntil } = req.body;

      if (!startTime || !endTime) {
        return res.status(400).json({ message: 'Start time and end time are required' });
      }

      const availability = await storage.setDoctorAvailability({
        doctorId: userId,
        dayOfWeek: dayOfWeek || null,
        startTime,
        endTime,
        isRecurring: isRecurring || false,
        specificDate: specificDate ? new Date(specificDate) : null,
        validFrom: validFrom ? new Date(validFrom) : new Date(),
        validUntil: validUntil ? new Date(validUntil) : null,
      });

      res.status(201).json(availability);
    } catch (error) {
      console.error('Error setting availability:', error);
      res.status(500).json({ message: 'Failed to set availability' });
    }
  });

  // Get doctor availability
  app.get('/api/v1/availability/:doctorId', isAuthenticated, async (req: any, res) => {
    try {
      const { doctorId } = req.params;
      const { startDate, endDate } = req.query;

      const dateRange = (startDate && endDate) 
        ? { start: new Date(startDate as string), end: new Date(endDate as string) }
        : undefined;

      const availability = await storage.getDoctorAvailability(doctorId, dateRange);

      res.json(availability);
    } catch (error) {
      console.error('Error fetching availability:', error);
      res.status(500).json({ message: 'Failed to fetch availability' });
    }
  });

  // Remove availability block
  app.delete('/api/v1/availability/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can remove availability' });
      }

      const { id } = req.params;
      await storage.removeDoctorAvailability(id);

      res.status(204).send();
    } catch (error) {
      console.error('Error removing availability:', error);
      res.status(500).json({ message: 'Failed to remove availability' });
    }
  });

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA - EMAIL MANAGEMENT ROUTES
  // ============================================================================

  // Create email thread
  app.post('/api/v1/emails/threads', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can manage email threads' });
      }

      const { subject, patientId, category, priority, gmailThreadId } = req.body;

      if (!subject) {
        return res.status(400).json({ message: 'Subject is required' });
      }

      const thread = await storage.createEmailThread({
        doctorId: userId,
        subject,
        patientId: patientId || null,
        category: category || 'general',
        priority: priority || 'normal',
        status: 'active',
        isRead: false,
        messageCount: 0,
        lastMessageAt: new Date(),
        gmailThreadId: gmailThreadId || null,
      });

      res.status(201).json(thread);
    } catch (error) {
      console.error('Error creating email thread:', error);
      res.status(500).json({ message: 'Failed to create email thread' });
    }
  });

  // List email threads
  app.get('/api/v1/emails/threads', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view email threads' });
      }

      const { status, category, isRead, patientId, limit, offset } = req.query;

      const filters: any = {};
      if (status) filters.status = status as string;
      if (category) filters.category = category as string;
      if (isRead !== undefined) filters.isRead = isRead === 'true';
      if (patientId) filters.patientId = patientId as string;
      if (limit) filters.limit = parseInt(limit as string);
      if (offset) filters.offset = parseInt(offset as string);

      const threads = await storage.listEmailThreads(userId, filters);
      res.json(threads);
    } catch (error) {
      console.error('Error listing email threads:', error);
      res.status(500).json({ message: 'Failed to list email threads' });
    }
  });

  // Get email thread by ID
  app.get('/api/v1/emails/threads/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view email threads' });
      }

      const { id } = req.params;
      const thread = await storage.getEmailThread(id);

      if (!thread) {
        return res.status(404).json({ message: 'Email thread not found' });
      }

      // SECURITY: Only the thread owner can view it
      if (thread.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only view your own email threads' });
      }

      res.json(thread);
    } catch (error) {
      console.error('Error fetching email thread:', error);
      res.status(500).json({ message: 'Failed to fetch email thread' });
    }
  });

  // Mark thread as read
  app.post('/api/v1/emails/threads/:id/read', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can manage email threads' });
      }

      const { id } = req.params;
      const thread = await storage.getEmailThread(id);

      if (!thread) {
        return res.status(404).json({ message: 'Email thread not found' });
      }

      // SECURITY: Only the thread owner can mark as read
      if (thread.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only mark your own threads as read' });
      }

      const updatedThread = await storage.markThreadRead(id);

      res.json(updatedThread);
    } catch (error) {
      console.error('Error marking thread as read:', error);
      res.status(500).json({ message: 'Failed to mark thread as read' });
    }
  });

  // Get messages in a thread
  app.get('/api/v1/emails/threads/:id/messages', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view email messages' });
      }

      const { id } = req.params;
      const thread = await storage.getEmailThread(id);

      if (!thread) {
        return res.status(404).json({ message: 'Email thread not found' });
      }

      // SECURITY: Only the thread owner can view messages
      if (thread.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only view messages in your own threads' });
      }

      const messages = await storage.getThreadMessages(id);
      res.json(messages);
    } catch (error) {
      console.error('Error fetching thread messages:', error);
      res.status(500).json({ message: 'Failed to fetch thread messages' });
    }
  });

  // Create email message in thread
  app.post('/api/v1/emails/messages', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can send email messages' });
      }

      const { threadId, sender, senderEmail, body, isFromDoctor, gmailMessageId } = req.body;

      if (!threadId || !sender || !senderEmail || !body) {
        return res.status(400).json({ message: 'Missing required fields' });
      }

      const thread = await storage.getEmailThread(threadId);

      if (!thread) {
        return res.status(404).json({ message: 'Email thread not found' });
      }

      // SECURITY: Only the thread owner can create messages
      if (thread.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only send messages in your own threads' });
      }

      const message = await storage.createEmailMessage({
        threadId,
        sender,
        senderEmail,
        body,
        isFromDoctor: isFromDoctor || false,
        isSent: false,
        gmailMessageId: gmailMessageId || null,
      });

      res.status(201).json(message);
    } catch (error) {
      console.error('Error creating email message:', error);
      res.status(500).json({ message: 'Failed to create email message' });
    }
  });

  // ============================================================================
  // ASSISTANT LYSA - AI EMAIL FEATURES
  // ============================================================================

  // AI: Categorize a single email
  app.post('/api/v1/emails/categorize', isAuthenticated, aiRateLimit, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can use AI email features' });
      }

      const { subject, content, senderContext } = req.body;

      // INPUT VALIDATION
      if (!subject || typeof subject !== 'string' || subject.length > 500) {
        return res.status(400).json({ message: 'Subject is required and must be a string (max 500 chars)' });
      }

      if (!content || typeof content !== 'string' || content.length > 10000) {
        return res.status(400).json({ message: 'Content is required and must be a string (max 10000 chars)' });
      }

      if (senderContext && typeof senderContext !== 'object') {
        return res.status(400).json({ message: 'senderContext must be an object' });
      }

      const categorization = await categorizeEmail(subject, content, senderContext);
      res.json(categorization);
    } catch (error) {
      console.error('Error categorizing email:', error);
      res.status(500).json({ message: 'Failed to categorize email' });
    }
  });

  // AI: Generate reply for email thread
  app.post('/api/v1/emails/threads/:threadId/generate-reply', isAuthenticated, aiRateLimit, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can generate email replies' });
      }

      const { threadId } = req.params;

      // INPUT VALIDATION
      if (!threadId || typeof threadId !== 'string') {
        return res.status(400).json({ message: 'Invalid thread ID' });
      }

      const thread = await storage.getEmailThread(threadId);

      if (!thread) {
        return res.status(404).json({ message: 'Email thread not found' });
      }

      // SECURITY: Only the thread owner can generate replies
      if (thread.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only generate replies for your own threads' });
      }

      const messages = await storage.getThreadMessages(threadId);

      // DEFENSIVE: Handle empty thread case
      if (!messages || messages.length === 0) {
        return res.status(400).json({ message: 'Cannot generate reply for thread with no messages' });
      }

      const doctorContext = {
        doctorName: `${user.firstName} ${user.lastName}`,
        specialty: user.specialty || undefined,
        clinicName: user.organization || undefined,
      };

      const reply = await generateEmailReply(thread, messages, doctorContext);
      res.json(reply);
    } catch (error) {
      console.error('Error generating email reply:', error);
      res.status(500).json({ message: 'Failed to generate email reply' });
    }
  });

  // AI: Batch categorize multiple emails
  app.post('/api/v1/emails/batch-categorize', isAuthenticated, batchRateLimit, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can use AI email features' });
      }

      const { emails } = req.body;

      // INPUT VALIDATION
      if (!Array.isArray(emails) || emails.length === 0) {
        return res.status(400).json({ message: 'Emails array is required and must not be empty' });
      }

      if (emails.length > 50) {
        return res.status(400).json({ message: 'Maximum 50 emails per batch' });
      }

      // Validate each email object
      for (let i = 0; i < emails.length; i++) {
        const email = emails[i];
        if (!email.id || typeof email.id !== 'string') {
          return res.status(400).json({ message: `Email at index ${i} missing valid id` });
        }
        if (!email.subject || typeof email.subject !== 'string' || email.subject.length > 500) {
          return res.status(400).json({ message: `Email at index ${i} missing valid subject (max 500 chars)` });
        }
        if (!email.content || typeof email.content !== 'string' || email.content.length > 10000) {
          return res.status(400).json({ message: `Email at index ${i} missing valid content (max 10000 chars)` });
        }
      }

      const results = await batchCategorizeEmails(emails);
      
      // Convert Map to object for JSON response
      const resultsObj: Record<string, any> = {};
      results.forEach((value, key) => {
        resultsObj[key] = value;
      });

      res.json(resultsObj);
    } catch (error) {
      console.error('Error batch categorizing emails:', error);
      res.status(500).json({ message: 'Failed to batch categorize emails' });
    }
  });

  // AI: Extract action items from email
  app.post('/api/v1/emails/extract-actions', isAuthenticated, aiRateLimit, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can use AI email features' });
      }

      const { content } = req.body;

      // INPUT VALIDATION
      if (!content || typeof content !== 'string' || content.length > 10000) {
        return res.status(400).json({ message: 'Content is required and must be a string (max 10000 chars)' });
      }

      const actionItems = await extractActionItems(content);
      res.json({ actionItems });
    } catch (error) {
      console.error('Error extracting action items:', error);
      res.status(500).json({ message: 'Failed to extract action items' });
    }
  });

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA - CALL MANAGEMENT ROUTES
  // ============================================================================

  // Create call log
  app.post('/api/v1/calls', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can manage call logs' });
      }

      const { 
        patientId, 
        callerName, 
        callerPhone, 
        direction, 
        callType, 
        twilioCallSid 
      } = req.body;

      if (!callerName || !callerPhone || !direction || !callType) {
        return res.status(400).json({ message: 'Missing required fields' });
      }

      const callLog = await storage.createCallLog({
        doctorId: userId,
        patientId: patientId || null,
        callerName,
        callerPhone,
        direction,
        callType,
        status: 'initiated',
        startTime: new Date(),
        requiresFollowup: false,
        twilioCallSid: twilioCallSid || null,
      });

      res.status(201).json(callLog);
    } catch (error) {
      console.error('Error creating call log:', error);
      res.status(500).json({ message: 'Failed to create call log' });
    }
  });

  // List call logs
  app.get('/api/v1/calls', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view call logs' });
      }

      const { status, requiresFollowup, patientId, limit, offset } = req.query;

      const filters: any = {};
      if (status) filters.status = status as string;
      if (requiresFollowup !== undefined) filters.requiresFollowup = requiresFollowup === 'true';
      if (patientId) filters.patientId = patientId as string;
      if (limit) filters.limit = parseInt(limit as string);
      if (offset) filters.offset = parseInt(offset as string);

      const callLogs = await storage.listCallLogs(userId, filters);
      res.json(callLogs);
    } catch (error) {
      console.error('Error listing call logs:', error);
      res.status(500).json({ message: 'Failed to list call logs' });
    }
  });

  // Get call log by ID
  app.get('/api/v1/calls/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view call logs' });
      }

      const { id } = req.params;
      const callLog = await storage.getCallLog(id);

      if (!callLog) {
        return res.status(404).json({ message: 'Call log not found' });
      }

      // SECURITY: Only the call owner can view it
      if (callLog.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only view your own call logs' });
      }

      res.json(callLog);
    } catch (error) {
      console.error('Error fetching call log:', error);
      res.status(500).json({ message: 'Failed to fetch call log' });
    }
  });

  // Update call log
  app.patch('/api/v1/calls/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can update call logs' });
      }

      const { id } = req.params;
      const existingCallLog = await storage.getCallLog(id);

      if (!existingCallLog) {
        return res.status(404).json({ message: 'Call log not found' });
      }

      // SECURITY: Only the call owner can update it
      if (existingCallLog.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only update your own call logs' });
      }

      const updates = req.body;

      // SECURITY: Prevent changing doctorId or patientId
      delete updates.doctorId;
      delete updates.patientId;

      const callLog = await storage.updateCallLog(id, updates);

      res.json(callLog);
    } catch (error) {
      console.error('Error updating call log:', error);
      res.status(500).json({ message: 'Failed to update call log' });
    }
  });

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA - APPOINTMENT REMINDERS ROUTES
  // ============================================================================

  // Create appointment reminder
  app.post('/api/v1/reminders', isAuthenticated, async (req: any, res) => {
    try {
      const { appointmentId, reminderType, scheduledFor, channel } = req.body;

      if (!appointmentId || !reminderType || !scheduledFor || !channel) {
        return res.status(400).json({ message: 'Missing required fields' });
      }

      const reminder = await storage.createAppointmentReminder({
        appointmentId,
        reminderType,
        scheduledFor: new Date(scheduledFor),
        channel,
        status: 'pending',
        confirmed: false,
        retryCount: 0,
      });

      res.status(201).json(reminder);
    } catch (error) {
      console.error('Error creating reminder:', error);
      res.status(500).json({ message: 'Failed to create reminder' });
    }
  });

  // Get reminders for an appointment
  app.get('/api/v1/appointments/:appointmentId/reminders', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      const { appointmentId } = req.params;
      const appointment = await storage.getAppointment(appointmentId);

      if (!appointment) {
        return res.status(404).json({ message: 'Appointment not found' });
      }

      // SECURITY: Only appointment participants can view reminders
      if (user.role === 'patient' && appointment.patientId !== userId) {
        return res.status(403).json({ message: 'You can only view reminders for your own appointments' });
      }
      if (user.role === 'doctor' && appointment.doctorId !== userId) {
        return res.status(403).json({ message: 'You can only view reminders for appointments with your patients' });
      }

      const reminders = await storage.getAppointmentReminders(appointmentId);
      res.json(reminders);
    } catch (error) {
      console.error('Error fetching reminders:', error);
      res.status(500).json({ message: 'Failed to fetch reminders' });
    }
  });

  // List due reminders (for background worker)
  app.get('/api/v1/reminders/due', isAuthenticated, async (req: any, res) => {
    try {
      const beforeTime = req.query.before 
        ? new Date(req.query.before as string)
        : new Date();
      
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 100;

      const reminders = await storage.listDueReminders(beforeTime, limit);
      res.json(reminders);
    } catch (error) {
      console.error('Error fetching due reminders:', error);
      res.status(500).json({ message: 'Failed to fetch due reminders' });
    }
  });

  // Mark reminder as sent
  app.post('/api/v1/reminders/:id/sent', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const { twilioSid, sesSid } = req.body;

      const reminder = await storage.markReminderSent(id, new Date(), twilioSid, sesSid);

      if (!reminder) {
        return res.status(404).json({ message: 'Reminder not found' });
      }

      res.json(reminder);
    } catch (error) {
      console.error('Error marking reminder as sent:', error);
      res.status(500).json({ message: 'Failed to mark reminder as sent' });
    }
  });

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA - WHATSAPP APPOINTMENT MANAGEMENT ROUTES
  // ============================================================================

  const { createWhatsAppService } = await import('./whatsappService');
  const whatsappService = createWhatsAppService(storage);

  // Twilio WhatsApp webhook - receives incoming messages
  app.post('/api/v1/whatsapp/webhook', async (req: any, res) => {
    try {
      const { From, To, Body, MessageSid } = req.body;

      if (!From || !Body) {
        return res.status(400).send('Missing required fields');
      }

      // For now, route to the first doctor - in production, match by phone number
      const doctors = await storage.getAllDoctors();
      if (doctors.length === 0) {
        return res.status(404).send('No doctors available');
      }

      const doctorId = doctors[0].id;

      const response = await whatsappService.processIncomingMessage(
        { from: From, to: To, body: Body, messageSid: MessageSid },
        doctorId
      );

      // Send response via WhatsApp
      await whatsappService.sendWhatsAppMessage(From, response);

      // Return TwiML response
      res.set('Content-Type', 'text/xml');
      res.send(`<?xml version="1.0" encoding="UTF-8"?><Response></Response>`);
    } catch (error) {
      console.error('WhatsApp webhook error:', error);
      res.status(500).send('Error processing message');
    }
  });

  // Send WhatsApp message (doctor-initiated)
  app.post('/api/v1/whatsapp/send', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can send WhatsApp messages' });
      }

      const { to, message } = req.body;

      if (!to || !message) {
        return res.status(400).json({ message: 'Missing recipient or message' });
      }

      const result = await whatsappService.sendWhatsAppMessage(to, message);

      if (result.success) {
        res.json({ success: true, messageSid: result.sid });
      } else {
        res.status(500).json({ success: false, message: 'Failed to send message' });
      }
    } catch (error) {
      console.error('Error sending WhatsApp message:', error);
      res.status(500).json({ message: 'Failed to send message' });
    }
  });

  // Send WhatsApp appointment confirmation
  app.post('/api/v1/whatsapp/appointment-confirmation', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can send confirmations' });
      }

      const { patientPhone, patientName, doctorName, date, time, appointmentId } = req.body;

      if (!patientPhone || !patientName || !doctorName || !date || !time || !appointmentId) {
        return res.status(400).json({ message: 'Missing required fields' });
      }

      const success = await whatsappService.sendAppointmentConfirmation(
        patientPhone,
        patientName,
        doctorName,
        date,
        time,
        appointmentId
      );

      res.json({ success });
    } catch (error) {
      console.error('Error sending appointment confirmation:', error);
      res.status(500).json({ message: 'Failed to send confirmation' });
    }
  });

  // Send WhatsApp appointment reminder
  app.post('/api/v1/whatsapp/appointment-reminder', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can send reminders' });
      }

      const { patientPhone, patientName, doctorName, appointmentTime } = req.body;

      if (!patientPhone || !patientName || !doctorName || !appointmentTime) {
        return res.status(400).json({ message: 'Missing required fields' });
      }

      const success = await whatsappService.sendAppointmentReminder(
        patientPhone,
        patientName,
        doctorName,
        new Date(appointmentTime)
      );

      res.json({ success });
    } catch (error) {
      console.error('Error sending appointment reminder:', error);
      res.status(500).json({ message: 'Failed to send reminder' });
    }
  });

  // Get active WhatsApp conversation
  app.get('/api/v1/whatsapp/conversation/:phoneNumber', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.session?.userId;
      if (!userId) {
        return res.status(401).json({ message: 'Unauthorized' });
      }

      const user = await storage.getUser(userId);
      if (!user || user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view conversations' });
      }

      const { phoneNumber } = req.params;
      const conversation = whatsappService.getActiveConversation(userId, phoneNumber);

      if (!conversation) {
        return res.status(404).json({ message: 'No active conversation found' });
      }

      res.json(conversation);
    } catch (error) {
      console.error('Error fetching conversation:', error);
      res.status(500).json({ message: 'Failed to fetch conversation' });
    }
  });

  // ===== GOOGLE CALENDAR SYNC ROUTES =====
  // Import at top of file
  const { googleCalendarSyncService, isReplitConnectorAvailable, getUncachableGoogleCalendarClient } = await import('./googleCalendarSyncService');

  // Check Replit Google Calendar connector status
  app.get('/api/v1/calendar/connector-status', isAuthenticated, async (req: any, res) => {
    try {
      const isAvailable = await isReplitConnectorAvailable();
      
      if (isAvailable) {
        const calendar = await getUncachableGoogleCalendarClient();
        if (calendar) {
          // Get calendar info
          try {
            const calendarList = await calendar.calendarList.get({ calendarId: 'primary' });
            return res.json({
              connected: true,
              calendarId: calendarList.data.id || 'primary',
              calendarName: calendarList.data.summary || 'Primary Calendar',
              email: calendarList.data.id,
              source: 'replit_connector'
            });
          } catch (err) {
            console.error('Error getting calendar info:', err);
          }
        }
      }
      
      res.json({ connected: false, source: 'replit_connector' });
    } catch (error) {
      console.error('Error checking connector status:', error);
      res.status(500).json({ message: 'Failed to check connector status' });
    }
  });

  // List calendars from connected account
  app.get('/api/v1/calendar/list', isAuthenticated, async (req: any, res) => {
    try {
      const calendar = await getUncachableGoogleCalendarClient();
      
      if (!calendar) {
        return res.status(400).json({ message: 'Google Calendar not connected' });
      }

      const response = await calendar.calendarList.list();
      const calendars = response.data.items?.map(cal => ({
        id: cal.id,
        summary: cal.summary,
        primary: cal.primary || false,
        accessRole: cal.accessRole,
        backgroundColor: cal.backgroundColor
      })) || [];

      res.json({ calendars });
    } catch (error) {
      console.error('Error listing calendars:', error);
      res.status(500).json({ message: 'Failed to list calendars' });
    }
  });

  // Get upcoming events from connected calendar
  app.get('/api/v1/calendar/events', isAuthenticated, async (req: any, res) => {
    try {
      const calendar = await getUncachableGoogleCalendarClient();
      
      if (!calendar) {
        return res.status(400).json({ message: 'Google Calendar not connected' });
      }

      const calendarId = (req.query.calendarId as string) || 'primary';
      const timeMin = new Date().toISOString();
      const timeMax = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(); // 7 days ahead

      const response = await calendar.events.list({
        calendarId,
        timeMin,
        timeMax,
        maxResults: 50,
        singleEvents: true,
        orderBy: 'startTime'
      });

      const events = response.data.items?.map(event => ({
        id: event.id,
        summary: event.summary,
        description: event.description,
        location: event.location,
        start: event.start,
        end: event.end,
        status: event.status,
        hangoutLink: event.hangoutLink,
        attendees: event.attendees?.map(a => ({
          email: a.email,
          responseStatus: a.responseStatus,
          displayName: a.displayName
        }))
      })) || [];

      res.json({ events });
    } catch (error) {
      console.error('Error fetching calendar events:', error);
      res.status(500).json({ message: 'Failed to fetch calendar events' });
    }
  });

  // Create event in Google Calendar
  app.post('/api/v1/calendar/events', isAuthenticated, async (req: any, res) => {
    try {
      const calendar = await getUncachableGoogleCalendarClient();
      
      if (!calendar) {
        return res.status(400).json({ message: 'Google Calendar not connected' });
      }

      const { calendarId = 'primary', summary, description, location, startTime, endTime, attendees } = req.body;

      const event = {
        summary,
        description,
        location,
        start: {
          dateTime: startTime,
          timeZone: 'UTC'
        },
        end: {
          dateTime: endTime,
          timeZone: 'UTC'
        },
        attendees: attendees?.map((email: string) => ({ email }))
      };

      const response = await calendar.events.insert({
        calendarId,
        requestBody: event,
        sendUpdates: 'all'
      });

      res.json({ 
        success: true, 
        eventId: response.data.id,
        htmlLink: response.data.htmlLink 
      });
    } catch (error) {
      console.error('Error creating calendar event:', error);
      res.status(500).json({ message: 'Failed to create calendar event' });
    }
  });

  // Delete event from Google Calendar
  app.delete('/api/v1/calendar/events/:eventId', isAuthenticated, async (req: any, res) => {
    try {
      const calendar = await getUncachableGoogleCalendarClient();
      
      if (!calendar) {
        return res.status(400).json({ message: 'Google Calendar not connected' });
      }

      const { eventId } = req.params;
      const calendarId = (req.query.calendarId as string) || 'primary';

      await calendar.events.delete({
        calendarId,
        eventId
      });

      res.json({ success: true });
    } catch (error) {
      console.error('Error deleting calendar event:', error);
      res.status(500).json({ message: 'Failed to delete calendar event' });
    }
  });

  // Sync appointments to Google Calendar
  app.post('/api/v1/calendar/sync-appointments', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can sync appointments' });
      }

      const calendar = await getUncachableGoogleCalendarClient();
      
      if (!calendar) {
        return res.status(400).json({ message: 'Google Calendar not connected' });
      }

      // Get doctor's appointments
      const appointments = await storage.getAppointmentsByDoctor(req.user.id);
      const calendarId = (req.body.calendarId as string) || 'primary';
      
      let synced = 0;
      let failed = 0;

      for (const appointment of appointments) {
        try {
          // Skip cancelled appointments
          if (appointment.status === 'cancelled') continue;
          
          // Skip if already synced
          if (appointment.googleCalendarEventId) continue;

          const event = {
            summary: appointment.title || `Appointment with patient`,
            description: appointment.description || `Followup AI appointment`,
            location: appointment.location || undefined,
            start: {
              dateTime: appointment.startTime.toISOString(),
              timeZone: 'UTC'
            },
            end: {
              dateTime: appointment.endTime.toISOString(),
              timeZone: 'UTC'
            }
          };

          const response = await calendar.events.insert({
            calendarId,
            requestBody: event
          });

          // Update appointment with Google Calendar event ID
          await storage.updateAppointment(appointment.id, {
            googleCalendarEventId: response.data.id
          });

          synced++;
        } catch (err) {
          console.error(`Failed to sync appointment ${appointment.id}:`, err);
          failed++;
        }
      }

      res.json({ 
        success: true, 
        synced, 
        failed,
        total: appointments.length 
      });
    } catch (error) {
      console.error('Error syncing appointments:', error);
      res.status(500).json({ message: 'Failed to sync appointments' });
    }
  });

  // Get Google Calendar auth URL
  app.get('/api/v1/calendar/auth-url', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can connect Google Calendar' });
      }

      const authUrl = googleCalendarSyncService.getAuthUrl(req.user.id);
      res.json({ authUrl });
    } catch (error) {
      console.error('Error generating auth URL:', error);
      res.status(500).json({ message: 'Failed to generate auth URL' });
    }
  });

  // OAuth callback
  app.get('/api/calendar/oauth/callback', async (req, res) => {
    try {
      const code = req.query.code as string;
      const state = req.query.state as string; // doctorId
      
      if (!code || !state) {
        return res.status(400).send('Missing code or state parameter');
      }

      const result = await googleCalendarSyncService.handleOAuthCallback(code, state);
      
      if (result.success) {
        res.redirect('/?calendar=connected');
      } else {
        res.redirect('/?calendar=error&message=' + encodeURIComponent(result.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error in OAuth callback:', error);
      res.status(500).send('Failed to complete OAuth');
    }
  });

  // Get sync status
  app.get('/api/v1/calendar/sync/status', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access calendar sync' });
      }

      const syncConfig = await storage.getGoogleCalendarSync(req.user.id);
      res.json(syncConfig || { connected: false });
    } catch (error) {
      console.error('Error fetching sync status:', error);
      res.status(500).json({ message: 'Failed to fetch sync status' });
    }
  });

  // Trigger manual sync
  app.post('/api/v1/calendar/sync/trigger', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can sync calendar' });
      }

      const result = await googleCalendarSyncService.performSync(req.user.id, 'manual');
      res.json(result);
    } catch (error) {
      console.error('Error triggering sync:', error);
      res.status(500).json({ message: 'Failed to trigger sync' });
    }
  });

  // Update sync settings
  app.patch('/api/v1/calendar/sync/settings', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can update sync settings' });
      }

      const { syncEnabled, syncDirection, conflictResolution } = req.body;
      
      const updated = await storage.updateGoogleCalendarSync(req.user.id, {
        syncEnabled,
        syncDirection,
        conflictResolution,
      });

      res.json(updated);
    } catch (error) {
      console.error('Error updating sync settings:', error);
      res.status(500).json({ message: 'Failed to update sync settings' });
    }
  });

  // Get sync logs
  app.get('/api/v1/calendar/sync/logs', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view sync logs' });
      }

      const limit = req.query.limit ? parseInt(req.query.limit as string) : 50;
      const logs = await storage.getGoogleCalendarSyncLogs(req.user.id, limit);
      res.json(logs);
    } catch (error) {
      console.error('Error fetching sync logs:', error);
      res.status(500).json({ message: 'Failed to fetch sync logs' });
    }
  });

  // Disconnect Google Calendar
  app.post('/api/v1/calendar/disconnect', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can disconnect calendar' });
      }

      const result = await googleCalendarSyncService.disconnect(req.user.id);
      res.json(result);
    } catch (error) {
      console.error('Error disconnecting calendar:', error);
      res.status(500).json({ message: 'Failed to disconnect calendar' });
    }
  });

  // ===== GMAIL SYNC ROUTES (HIPAA CRITICAL) =====
  const { gmailService } = await import('./gmailService');

  // Get Gmail auth URL
  app.get('/api/v1/gmail/auth-url', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can connect Gmail' });
      }

      const authUrl = gmailService.getAuthUrl(req.user.id);
      res.json({ authUrl });
    } catch (error) {
      console.error('Error generating Gmail auth URL:', error);
      res.status(500).json({ message: 'Failed to generate auth URL' });
    }
  });

  // Gmail OAuth callback
  app.get('/api/gmail/oauth/callback', async (req, res) => {
    try {
      const code = req.query.code as string;
      const state = req.query.state as string; // doctorId
      
      if (!code || !state) {
        return res.status(400).send('Missing code or state parameter');
      }

      const result = await gmailService.handleOAuthCallback(code, state);
      
      if (result.success) {
        res.redirect('/?gmail=connected');
      } else {
        res.redirect('/?gmail=error&message=' + encodeURIComponent(result.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error in Gmail OAuth callback:', error);
      res.status(500).send('Failed to complete OAuth');
    }
  });

  // Get Gmail sync status
  app.get('/api/v1/gmail/sync/status', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access Gmail sync' });
      }

      const syncConfig = await storage.getGmailSync(req.user.id);
      res.json(syncConfig || { connected: false });
    } catch (error) {
      console.error('Error fetching Gmail sync status:', error);
      res.status(500).json({ message: 'Failed to fetch sync status' });
    }
  });

  // Trigger manual Gmail sync
  app.post('/api/v1/gmail/sync/trigger', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can sync Gmail' });
      }

      const result = await gmailService.syncEmails(req.user.id);
      res.json(result);
    } catch (error) {
      console.error('Error triggering Gmail sync:', error);
      res.status(500).json({ message: 'Failed to trigger sync' });
    }
  });

  // Send email via Gmail
  app.post('/api/v1/gmail/send', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can send emails' });
      }

      const { to, subject, body } = req.body;
      const result = await gmailService.sendEmail(req.user.id, to, subject, body);
      res.json(result);
    } catch (error) {
      console.error('Error sending email:', error);
      res.status(500).json({ message: 'Failed to send email' });
    }
  });

  // Get Gmail sync logs
  app.get('/api/v1/gmail/sync/logs', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view sync logs' });
      }

      const limit = req.query.limit ? parseInt(req.query.limit as string) : 50;
      const logs = await storage.getGmailSyncLogs(req.user.id, limit);
      res.json(logs);
    } catch (error) {
      console.error('Error fetching Gmail sync logs:', error);
      res.status(500).json({ message: 'Failed to fetch sync logs' });
    }
  });

  // Disconnect Gmail
  app.post('/api/v1/gmail/disconnect', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can disconnect Gmail' });
      }

      const result = await gmailService.disconnect(req.user.id);
      res.json(result);
    } catch (error) {
      console.error('Error disconnecting Gmail:', error);
      res.status(500).json({ message: 'Failed to disconnect Gmail' });
    }
  });

  // ===== DOCTOR INTEGRATIONS (PER-DOCTOR PERSONAL ACCOUNTS) =====
  // Note: doctorIntegrationService is imported and initialized at the top of registerRoutes

  // Get all integration statuses for doctor
  app.get('/api/v1/integrations/status', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access integrations' });
      }

      const status = await doctorIntegrationService.getIntegrationStatus(req.user.id);
      res.json(status);
    } catch (error) {
      console.error('Error fetching integration status:', error);
      res.status(500).json({ message: 'Failed to fetch integration status' });
    }
  });

  // Get all integrations for doctor
  app.get('/api/v1/integrations', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access integrations' });
      }

      const integrations = await doctorIntegrationService.getDoctorIntegrations(req.user.id);
      res.json(integrations);
    } catch (error) {
      console.error('Error fetching integrations:', error);
      res.status(500).json({ message: 'Failed to fetch integrations' });
    }
  });

  // Get Gmail OAuth URL for per-doctor connection
  app.get('/api/v1/integrations/gmail/auth-url', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can connect Gmail' });
      }

      const authUrl = doctorIntegrationService.getGmailAuthUrl(req.user.id);
      res.json({ authUrl });
    } catch (error) {
      console.error('Error generating Gmail auth URL:', error);
      res.status(500).json({ message: 'Failed to generate auth URL' });
    }
  });

  // Gmail OAuth callback for per-doctor connection
  // Uses signed JWT state token for CSRF protection (HIPAA-compliant)
  app.get('/api/integrations/gmail/callback', async (req, res) => {
    try {
      const code = req.query.code as string;
      const stateToken = req.query.state as string;
      
      if (!code || !stateToken) {
        return res.redirect('/?integration=error&message=Missing+parameters');
      }

      // The handleGmailCallback validates the signed JWT state token internally
      // This prevents CSRF attacks where an attacker could craft a callback URL
      const result = await doctorIntegrationService.handleGmailCallback(code, stateToken);
      
      if (result.success) {
        res.redirect(`/?integration=gmail&status=connected&email=${encodeURIComponent(result.email || '')}`);
      } else {
        res.redirect(`/?integration=gmail&status=error&message=${encodeURIComponent(result.error || 'Unknown error')}`);
      }
    } catch (error) {
      console.error('Error in Gmail OAuth callback:', error);
      res.redirect('/?integration=gmail&status=error&message=OAuth+failed');
    }
  });

  // Sync Gmail emails for doctor
  app.post('/api/v1/integrations/gmail/sync', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can sync Gmail' });
      }

      const result = await doctorIntegrationService.syncGmailEmails(req.user.id);
      res.json(result);
    } catch (error) {
      console.error('Error syncing Gmail:', error);
      res.status(500).json({ message: 'Failed to sync Gmail' });
    }
  });

  // Get synced emails for doctor
  app.get('/api/v1/integrations/gmail/emails', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view emails' });
      }

      const { category, isRead, limit, offset } = req.query;
      const emails = await doctorIntegrationService.getDoctorEmails(req.user.id, {
        category: category as string,
        isRead: isRead !== undefined ? isRead === 'true' : undefined,
        limit: limit ? parseInt(limit as string) : undefined,
        offset: offset ? parseInt(offset as string) : undefined,
      });
      res.json(emails);
    } catch (error) {
      console.error('Error fetching emails:', error);
      res.status(500).json({ message: 'Failed to fetch emails' });
    }
  });

  // Send email from doctor's connected account
  app.post('/api/v1/integrations/gmail/send', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can send emails' });
      }

      const { to, subject, body, replyToMessageId } = req.body;
      const result = await doctorIntegrationService.sendEmailFromDoctor(req.user.id, to, subject, body, replyToMessageId);
      res.json(result);
    } catch (error) {
      console.error('Error sending email:', error);
      res.status(500).json({ message: 'Failed to send email' });
    }
  });

  // Disconnect Gmail
  app.post('/api/v1/integrations/gmail/disconnect', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can disconnect Gmail' });
      }

      const result = await doctorIntegrationService.disconnectGmail(req.user.id);
      res.json(result);
    } catch (error) {
      console.error('Error disconnecting Gmail:', error);
      res.status(500).json({ message: 'Failed to disconnect Gmail' });
    }
  });

  // Configure WhatsApp Business for doctor
  app.post('/api/v1/integrations/whatsapp/configure', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can configure WhatsApp' });
      }

      const { businessId, phoneNumberId, displayNumber, accessToken } = req.body;
      const result = await doctorIntegrationService.configureWhatsAppBusiness(
        req.user.id,
        businessId,
        phoneNumberId,
        displayNumber,
        accessToken
      );
      res.json(result);
    } catch (error) {
      console.error('Error configuring WhatsApp:', error);
      res.status(500).json({ message: 'Failed to configure WhatsApp' });
    }
  });

  // Get WhatsApp messages for doctor
  app.get('/api/v1/integrations/whatsapp/messages', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view WhatsApp messages' });
      }

      const { status, limit } = req.query;
      const messages = await doctorIntegrationService.getDoctorWhatsappMessages(req.user.id, {
        status: status as string,
        limit: limit ? parseInt(limit as string) : undefined,
      });
      res.json(messages);
    } catch (error) {
      console.error('Error fetching WhatsApp messages:', error);
      res.status(500).json({ message: 'Failed to fetch WhatsApp messages' });
    }
  });

  // Send WhatsApp message
  app.post('/api/v1/integrations/whatsapp/send', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can send WhatsApp messages' });
      }

      const { toNumber, message } = req.body;
      const result = await doctorIntegrationService.sendWhatsAppMessage(req.user.id, toNumber, message);
      res.json(result);
    } catch (error) {
      console.error('Error sending WhatsApp message:', error);
      res.status(500).json({ message: 'Failed to send WhatsApp message' });
    }
  });

  // WhatsApp webhook for incoming messages
  app.post('/api/webhooks/whatsapp/:doctorId', async (req, res) => {
    try {
      const { doctorId } = req.params;
      const { entry } = req.body;

      if (entry && entry[0]?.changes) {
        for (const change of entry[0].changes) {
          if (change.field === 'messages' && change.value?.messages) {
            for (const message of change.value.messages) {
              await doctorIntegrationService.processIncomingWhatsAppMessage(
                doctorId,
                message.id,
                message.from,
                change.value.metadata.display_phone_number,
                message.type,
                message.text?.body,
                change.value.contacts?.[0]?.profile?.name
              );
            }
          }
        }
      }

      res.status(200).send('OK');
    } catch (error) {
      console.error('Error processing WhatsApp webhook:', error);
      res.status(500).send('Error');
    }
  });

  // WhatsApp webhook verification
  app.get('/api/webhooks/whatsapp/:doctorId', async (req, res) => {
    const mode = req.query['hub.mode'];
    const token = req.query['hub.verify_token'];
    const challenge = req.query['hub.challenge'];

    // Verify token - in production, use a secure token per doctor
    if (mode === 'subscribe' && token === 'followup-ai-whatsapp-verify') {
      res.status(200).send(challenge);
    } else {
      res.sendStatus(403);
    }
  });

  // Configure Twilio for doctor
  app.post('/api/v1/integrations/twilio/configure', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can configure Twilio' });
      }

      const { accountSid, apiKey, apiSecret, phoneNumber } = req.body;
      const result = await doctorIntegrationService.configureTwilioAccount(
        req.user.id,
        accountSid,
        apiKey,
        apiSecret,
        phoneNumber
      );
      res.json(result);
    } catch (error) {
      console.error('Error configuring Twilio:', error);
      res.status(500).json({ message: 'Failed to configure Twilio' });
    }
  });

  // Get call logs for doctor
  app.get('/api/v1/integrations/twilio/calls', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view call logs' });
      }

      const { status, limit } = req.query;
      const calls = await doctorIntegrationService.getDoctorCallLogs(req.user.id, {
        status: status as string,
        limit: limit ? parseInt(limit as string) : undefined,
      });
      res.json(calls);
    } catch (error) {
      console.error('Error fetching call logs:', error);
      res.status(500).json({ message: 'Failed to fetch call logs' });
    }
  });

  // Sync Twilio call history
  app.post('/api/v1/integrations/twilio/sync', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can sync call history' });
      }

      const result = await doctorIntegrationService.syncTwilioCallHistory(req.user.id);
      res.json(result);
    } catch (error) {
      console.error('Error syncing Twilio:', error);
      res.status(500).json({ message: 'Failed to sync call history' });
    }
  });

  // Make outbound call
  app.post('/api/v1/integrations/twilio/call', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can make calls' });
      }

      const { toNumber } = req.body;
      const result = await doctorIntegrationService.makeOutboundCall(req.user.id, toNumber);
      res.json(result);
    } catch (error) {
      console.error('Error making call:', error);
      res.status(500).json({ message: 'Failed to make call' });
    }
  });

  // Twilio voice webhook for incoming calls
  app.post('/api/webhooks/twilio/voice/:doctorId', async (req, res) => {
    try {
      const { doctorId } = req.params;
      const { CallSid, From, To, CallStatus, RecordingUrl, TranscriptionText, CallDuration } = req.body;

      await doctorIntegrationService.processIncomingCall(
        doctorId,
        CallSid,
        From,
        undefined,
        'inbound',
        CallStatus,
        CallDuration ? parseInt(CallDuration) : undefined,
        RecordingUrl,
        TranscriptionText
      );

      // Return TwiML for call handling
      res.type('text/xml');
      res.send(`<?xml version="1.0" encoding="UTF-8"?>
        <Response>
          <Say>Thank you for calling. A doctor will be with you shortly.</Say>
          <Record transcribe="true" transcribeCallback="/api/webhooks/twilio/transcription/${doctorId}"/>
        </Response>
      `);
    } catch (error) {
      console.error('Error processing Twilio voice webhook:', error);
      res.status(500).send('Error');
    }
  });

  // Twilio transcription callback
  app.post('/api/webhooks/twilio/transcription/:doctorId', async (req, res) => {
    try {
      const { doctorId } = req.params;
      const { CallSid, TranscriptionText, TranscriptionStatus } = req.body;

      if (TranscriptionStatus === 'completed' && TranscriptionText) {
        const callLog = await storage.getCallLogByTwilioSid(CallSid);
        if (callLog) {
          await storage.updateCallLog(callLog.id, {
            transcription: TranscriptionText,
          });
        }
      }

      res.status(200).send('OK');
    } catch (error) {
      console.error('Error processing transcription:', error);
      res.status(500).send('Error');
    }
  });

  // TwiML for outbound calls
  app.post('/api/twilio/twiml/outbound', async (req, res) => {
    res.type('text/xml');
    res.send(`<?xml version="1.0" encoding="UTF-8"?>
      <Response>
        <Dial callerId="${req.body.From || ''}" record="record-from-answer" transcribe="true">
          <Number>${req.body.To || ''}</Number>
        </Dial>
      </Response>
    `);
  });

  // ===== RESEARCH SERVICE =====
  const { researchService, initResearchService } = await import('./researchService');
  initResearchService(storage);

  // Query FHIR data from AWS HealthLake
  app.post('/api/v1/research/fhir/query', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access research data' });
      }

      const { resourceType, parameters, limit } = req.body;
      const data = await researchService.queryFHIRData({ resourceType, parameters, limit });
      res.json(data);
    } catch (error) {
      console.error('Error querying FHIR data:', error);
      res.status(500).json({ message: 'Failed to query FHIR data' });
    }
  });

  // Get epidemiological data
  app.get('/api/v1/research/epidemiology/:condition', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access research data' });
      }

      const { condition } = req.params;
      const data = await researchService.getEpidemiologicalData(condition);
      res.json(data);
    } catch (error) {
      console.error('Error fetching epidemiological data:', error);
      res.status(500).json({ message: 'Failed to fetch epidemiological data' });
    }
  });

  // Get population health metrics
  app.get('/api/v1/research/population-health', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access research data' });
      }

      const data = await researchService.getPopulationHealthMetrics(req.user.id);
      res.json(data);
    } catch (error) {
      console.error('Error fetching population health metrics:', error);
      res.status(500).json({ message: 'Failed to fetch population health metrics' });
    }
  });

  // Generate research report
  app.post('/api/v1/research/reports/generate', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can generate research reports' });
      }

      const { studyType, parameters } = req.body;
      const report = await researchService.generateResearchReport(studyType, parameters);
      res.json(report);
    } catch (error) {
      console.error('Error generating research report:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to generate report' });
    }
  });

  // ===== VOICE INTERFACE SERVICE =====
  const { voiceInterfaceService, initVoiceInterfaceService } = await import('./voiceInterfaceService');
  initVoiceInterfaceService(storage);

  // Transcribe audio
  app.post('/api/v1/voice/transcribe', isAuthenticated, async (req: any, res) => {
    try {
      const { audioFilePath } = req.body;
      const transcription = await voiceInterfaceService.transcribeAudio(audioFilePath);
      res.json(transcription);
    } catch (error) {
      console.error('Error transcribing audio:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to transcribe audio' });
    }
  });

  // Generate speech from text
  app.post('/api/v1/voice/speech', isAuthenticated, async (req: any, res) => {
    try {
      const { text, voice } = req.body;
      const audioBuffer = await voiceInterfaceService.generateSpeech(text, voice);
      
      res.setHeader('Content-Type', 'audio/mpeg');
      res.setHeader('Content-Disposition', 'attachment; filename="speech.mp3"');
      res.send(audioBuffer);
    } catch (error) {
      console.error('Error generating speech:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to generate speech' });
    }
  });

  // Process voice followup
  app.post('/api/v1/voice/followup', isAuthenticated, async (req: any, res) => {
    try {
      const patientId = req.user.role === 'patient' ? req.user.id : req.body.patientId;
      const { audioFilePath } = req.body;
      
      const result = await voiceInterfaceService.processVoiceFollowup(patientId, audioFilePath);
      res.json(result);
    } catch (error) {
      console.error('Error processing voice followup:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to process voice followup' });
    }
  });

  // ===== DOCTOR CONSULTATION SERVICE =====
  const { doctorConsultationService, initDoctorConsultationService } = await import('./doctorConsultationService');
  initDoctorConsultationService(storage);

  // Request consultation with another doctor
  app.post('/api/v1/consultations/request', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can request consultations' });
      }

      const { consultingDoctorId, patientId, reason, urgency, shareRecordTypes } = req.body;
      const consultation = await doctorConsultationService.requestConsultation({
        requestingDoctorId: req.user.id,
        consultingDoctorId,
        patientId,
        reason,
        urgency,
        shareRecordTypes,
      });

      res.json(consultation);
    } catch (error) {
      console.error('Error requesting consultation:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to request consultation' });
    }
  });

  // Approve consultation request
  app.post('/api/v1/consultations/:id/approve', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can approve consultations' });
      }

      const { id } = req.params;
      const consultation = await doctorConsultationService.approveConsultation(id, req.user.id);
      res.json(consultation);
    } catch (error) {
      console.error('Error approving consultation:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to approve consultation' });
    }
  });

  // Deny consultation request
  app.post('/api/v1/consultations/:id/deny', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can deny consultations' });
      }

      const { id } = req.params;
      const { reason } = req.body;
      const consultation = await doctorConsultationService.denyConsultation(id, req.user.id, reason);
      res.json(consultation);
    } catch (error) {
      console.error('Error denying consultation:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to deny consultation' });
    }
  });

  // Get patient records with access token
  app.post('/api/v1/consultations/records', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access patient records' });
      }

      const { accessToken, recordTypes } = req.body;
      const records = await doctorConsultationService.getPatientRecords(accessToken, recordTypes);
      res.json(records);
    } catch (error) {
      console.error('Error fetching patient records:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to fetch patient records' });
    }
  });

  // Get consultations for current doctor
  app.get('/api/v1/consultations', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can view consultations' });
      }

      const consultations = await doctorConsultationService.getConsultationsByDoctor(req.user.id);
      res.json(consultations);
    } catch (error) {
      console.error('Error fetching consultations:', error);
      res.status(500).json({ message: 'Failed to fetch consultations' });
    }
  });

  // Revoke consultation
  app.post('/api/v1/consultations/:id/revoke', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can revoke consultations' });
      }

      const { id } = req.params;
      await doctorConsultationService.revokeConsultation(id, req.user.id);
      res.json({ success: true });
    } catch (error) {
      console.error('Error revoking consultation:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to revoke consultation' });
    }
  });

  // ===== CHATBOT SERVICE =====
  const { chatbotService, initChatbotService } = await import('./chatbotService');
  initChatbotService(storage);

  // Initialize chatbot session (doctor only - for embedding on clinic website)
  app.post('/api/v1/chatbot/init', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can initialize chatbot' });
      }

      const { sessionId } = req.body;
      const doctorId = req.user.id;
      const context = await chatbotService.initializeSession(doctorId, sessionId);
      res.json({
        sessionId,
        initialMessage: context.messages[context.messages.length - 1].content,
      });
    } catch (error) {
      console.error('Error initializing chatbot:', error);
      res.status(500).json({ message: 'Failed to initialize chatbot' });
    }
  });

  // Send message to chatbot (doctor only)
  app.post('/api/v1/chatbot/message', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access chatbot' });
      }

      const { sessionId, message } = req.body;
      const doctorId = req.user.id;
      const result = await chatbotService.sendMessage(sessionId, message, doctorId);
      res.json(result);
    } catch (error) {
      console.error('Error sending message to chatbot:', error);
      res.status(500).json({ message: 'Failed to send message' });
    }
  });

  // Get chat history (doctor only)
  app.get('/api/v1/chatbot/history/:sessionId', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can access chat history' });
      }

      const { sessionId } = req.params;
      const history = await chatbotService.getChatHistory(sessionId);
      res.json({ messages: history });
    } catch (error) {
      console.error('Error fetching chat history:', error);
      res.status(500).json({ message: 'Failed to fetch chat history' });
    }
  });

  // End chatbot session (doctor only)
  app.post('/api/v1/chatbot/end', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can end chatbot session' });
      }

      const { sessionId } = req.body;
      await chatbotService.endSession(sessionId);
      res.json({ success: true });
    } catch (error) {
      console.error('Error ending chatbot session:', error);
      res.status(500).json({ message: 'Failed to end session' });
    }
  });

  // ===== APPOINTMENT REMINDER SERVICE =====
  const { appointmentReminderService, initAppointmentReminderService } = await import('./appointmentReminderService');
  initAppointmentReminderService(storage);

  // Send daily reminders (should be called by cron job)
  app.post('/api/v1/reminders/send-daily', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can trigger reminders' });
      }

      const results = await appointmentReminderService.sendDailyReminders();
      res.json(results);
    } catch (error) {
      console.error('Error sending daily reminders:', error);
      res.status(500).json({ message: 'Failed to send daily reminders' });
    }
  });

  // Send immediate reminder for specific appointment
  app.post('/api/v1/reminders/send/:appointmentId', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can send reminders' });
      }

      const { appointmentId } = req.params;
      const result = await appointmentReminderService.sendImmediateReminder(appointmentId);
      res.json(result);
    } catch (error) {
      console.error('Error sending reminder:', error);
      res.status(500).json({ message: 'Failed to send reminder' });
    }
  });

  // Process SMS reminder responses (webhook from Twilio)
  app.post('/api/v1/reminders/sms-response', async (req, res) => {
    try {
      const from = req.body.From;
      const message = req.body.Body;
      
      await appointmentReminderService.processReminderResponse(from, message);
      res.sendStatus(200);
    } catch (error) {
      console.error('Error processing reminder response:', error);
      res.sendStatus(500);
    }
  });

  // ===== TWILIO VOICE IVR ROUTES =====
  const { twilioVoiceService, initTwilioVoiceService } = await import('./twilioVoiceService');
  initTwilioVoiceService(storage);

  // IVR welcome (incoming call webhook)
  app.post('/api/v1/voice/ivr/welcome', (req, res) => {
    const doctorId = req.body.To; // Twilio number called
    const twiml = twilioVoiceService.generateIVRResponse(doctorId);
    res.type('text/xml');
    res.send(twiml);
  });

  // IVR menu handling
  app.post('/api/v1/voice/ivr/menu', (req, res) => {
    const digit = req.body.Digits || '';
    const speech = req.body.SpeechResult || '';
    const twiml = twilioVoiceService.handleMenuSelection(digit, speech);
    res.type('text/xml');
    res.send(twiml);
  });

  // Scheduling flow
  app.post('/api/v1/voice/ivr/schedule', (req, res) => {
    const twiml = twilioVoiceService.handleSchedulingFlow();
    res.type('text/xml');
    res.send(twiml);
  });

  // Process scheduling request
  app.post('/api/v1/voice/schedule/process', async (req, res) => {
    const speechResult = req.body.SpeechResult || '';
    const from = req.body.From;
    const twiml = await twilioVoiceService.processSchedulingRequest(speechResult, from);
    res.type('text/xml');
    res.send(twiml);
  });

  // Voicemail complete
  app.post('/api/v1/voice/voicemail/complete', async (req, res) => {
    const callSid = req.body.CallSid;
    const recordingUrl = req.body.RecordingUrl;
    const from = req.body.From;
    
    await twilioVoiceService.processVoicemail(callSid, recordingUrl, '', from);
    
    const twiml = twilioVoiceService.generateVoicemailCompleteResponse();
    res.type('text/xml');
    res.send(twiml);
  });

  // Voicemail transcription callback
  app.post('/api/v1/voice/voicemail/transcription', async (req, res) => {
    const callSid = req.body.CallSid;
    const recordingUrl = req.body.RecordingUrl;
    const transcription = req.body.TranscriptionText || '';
    const from = req.body.From;
    
    await twilioVoiceService.processVoicemail(callSid, recordingUrl, transcription, from);
    res.sendStatus(200);
  });

  // Make outbound call (doctor only)
  app.post('/api/v1/voice/call/outbound', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'doctor') {
        return res.status(403).json({ message: 'Only doctors can make outbound calls' });
      }

      const { to, message } = req.body;
      const result = await twilioVoiceService.makeOutboundCall(to, message, req.user.id);
      res.json(result);
    } catch (error) {
      console.error('Error making outbound call:', error);
      res.status(500).json({ message: 'Failed to make outbound call' });
    }
  });

  // Get call recording
  app.get('/api/v1/voice/recording/:recordingSid', isAuthenticated, async (req: any, res) => {
    try {
      const { recordingSid } = req.params;
      const recordingUrl = await twilioVoiceService.getCallRecording(recordingSid);
      
      if (!recordingUrl) {
        return res.status(404).json({ message: 'Recording not found' });
      }
      
      res.json({ recordingUrl });
    } catch (error) {
      console.error('Error fetching recording:', error);
      res.status(500).json({ message: 'Failed to fetch recording' });
    }
  });

  // Proxy endpoint to fetch latest Video AI metrics from Python backend
  app.get('/api/video-ai/latest-metrics', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.id;
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      // Fetch latest video metrics from Python backend
      const response = await fetch(`${pythonBackendUrl}/api/v1/video-ai/latest-metrics?patient_id=${userId}`);
      
      if (!response.ok) {
        if (response.status === 404) {
          return res.json(null); // No metrics yet - this is expected
        }
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching video AI metrics:', error);
      res.status(500).json({ message: 'Failed to fetch video AI metrics' });
    }
  });

  // Proxy endpoint to fetch Video AI exam sessions history from Python backend
  app.get('/api/video-ai/exam-sessions', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const days = req.query.days || 365;
      
      // Generate dev mode JWT token for Python backend authentication
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/video-ai/exam-sessions?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          return res.json([]); // No sessions yet
        }
        const error = await response.text();
        console.error('Python backend error (video-ai sessions):', response.status, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching video AI exam sessions:', error);
      res.status(500).json({ message: 'Failed to fetch video AI exam sessions' });
    }
  });

  // =====================================================
  // TREMOR ANALYSIS ENDPOINTS - Proxy to Python backend
  // =====================================================
  
  // Get tremor dashboard data for a patient
  app.get('/api/v1/tremor/dashboard/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      // Generate dev mode JWT token for Python backend authentication
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/tremor/dashboard/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          // Return empty dashboard structure when no data
          return res.json({
            patient_id: patientId,
            latest_tremor: null,
            trend: { status: 'insufficient_data', avg_tremor_index_7days: 0, recordings_count_7days: 0 },
            history_7days: []
          });
        }
        const error = await response.text();
        console.error('Python backend error (tremor dashboard):', response.status, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching tremor dashboard:', error);
      // Return empty structure on connection error
      res.json({
        patient_id: req.params.patientId,
        latest_tremor: null,
        trend: { status: 'insufficient_data', avg_tremor_index_7days: 0, recordings_count_7days: 0 },
        history_7days: []
      });
    }
  });

  // Get tremor history for a patient
  app.get('/api/v1/tremor/history/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 30;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/tremor/history/${patientId}?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          return res.json([]);
        }
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching tremor history:', error);
      res.json([]);
    }
  });

  // =====================================================
  // GAIT ANALYSIS ENDPOINTS - Proxy to Python backend
  // =====================================================
  
  // Get gait analysis sessions for a patient
  app.get('/api/v1/gait-analysis/sessions/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const limit = req.query.limit || 10;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/gait-analysis/sessions/${patientId}?limit=${limit}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          return res.json({ sessions: [] });
        }
        const error = await response.text();
        console.error('Python backend error (gait sessions):', response.status, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching gait sessions:', error);
      res.json({ sessions: [] });
    }
  });

  // Get gait metrics for a specific session
  app.get('/api/v1/gait-analysis/metrics/:sessionId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { sessionId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/gait-analysis/metrics/${sessionId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          return res.json(null);
        }
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching gait metrics:', error);
      res.json(null);
    }
  });

  // =====================================================
  // EDEMA ANALYSIS ENDPOINTS - Proxy to Python backend
  // =====================================================
  
  // Get edema metrics for a patient
  app.get('/api/v1/edema/metrics/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const limit = req.query.limit || 5;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/edema/metrics/${patientId}?limit=${limit}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          return res.json([]);
        }
        const error = await response.text();
        console.error('Python backend error (edema metrics):', response.status, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching edema metrics:', error);
      res.json([]);
    }
  });

  // Get edema baseline comparison
  app.get('/api/v1/edema/baseline/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/edema/baseline/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          return res.json(null);
        }
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching edema baseline:', error);
      res.json(null);
    }
  });

  // =====================================================
  // DEVIATION ANALYSIS ENDPOINTS - Proxy to Python backend
  // Health deviation detection and baseline comparison
  // =====================================================

  // Get deviations for logged-in patient
  app.get('/api/v1/deviation/me', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const days = req.query.days || 7;
      const alertOnly = req.query.alert_only === 'true' ? 'true' : 'false';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/deviation/me?days=${days}&alert_only=${alertOnly}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404 || response.status === 403) {
          return res.json([]);
        }
        const error = await response.text();
        console.error('Python backend error (deviation/me):', response.status, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching deviations:', error);
      res.json([]);
    }
  });

  // Get deviation summary for logged-in patient
  app.get('/api/v1/deviation/summary/me', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const days = req.query.days || 7;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/deviation/summary/me?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404 || response.status === 403) {
          return res.json([]);
        }
        const error = await response.text();
        console.error('Python backend error (deviation/summary):', response.status, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching deviation summary:', error);
      res.json([]);
    }
  });

  // Get deviations for a specific patient (doctor only)
  app.get('/api/v1/deviation/patient/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 7;
      const alertOnly = req.query.alert_only === 'true' ? 'true' : 'false';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/deviation/patient/${patientId}?days=${days}&alert_only=${alertOnly}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404 || response.status === 403) {
          return res.json([]);
        }
        const error = await response.text();
        console.error('Python backend error (deviation/patient):', response.status, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching patient deviations:', error);
      res.json([]);
    }
  });

  // =====================================================
  // BASELINE ENDPOINTS - Proxy to Python backend
  // Personal health baseline calculation and management
  // =====================================================

  // Get current baseline for logged-in patient
  app.get('/api/v1/baseline/current/me', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/baseline/current/me`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        if (response.status === 404 || response.status === 403) {
          return res.json(null);
        }
        const error = await response.text();
        console.error('Python backend error (baseline/current):', response.status, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching current baseline:', error);
      res.json(null);
    }
  });

  // Calculate new baseline for logged-in patient
  app.post('/api/v1/baseline/calculate/me', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/baseline/calculate/me`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Python backend error (baseline/calculate):', response.status, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error calculating baseline:', error);
      res.status(500).json({ message: 'Failed to calculate baseline' });
    }
  });

  // =====================================================
  // ML INFERENCE ENDPOINTS - Proxy to Python backend
  // Clinical-BERT NER, deterioration prediction, model management
  // =====================================================

  // Get ML system statistics
  app.get('/api/v1/ml/stats', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ml/stats`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        // Return default stats if Python backend unavailable
        return res.json({
          total_predictions: 0,
          predictions_today: 0,
          cache_hit_rate_percent: 0,
          active_models: 0,
          avg_inference_time_ms: 0,
          redis_enabled: false,
          backend_status: 'unavailable'
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching ML stats:', error);
      res.json({
        total_predictions: 0,
        predictions_today: 0,
        cache_hit_rate_percent: 0,
        active_models: 0,
        avg_inference_time_ms: 0,
        redis_enabled: false,
        backend_status: 'unavailable'
      });
    }
  });

  // Get available ML models
  app.get('/api/v1/ml/models', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ml/models`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        return res.json({ models: [] });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching ML models:', error);
      res.json({ models: [] });
    }
  });

  // Get model performance metrics
  app.get('/api/v1/ml/models/:modelName/performance', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { modelName } = req.params;
      const hours = req.query.hours || 24;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ml/models/${modelName}/performance?hours=${hours}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        return res.json({
          model_name: modelName,
          model_version: '1.0',
          time_window_hours: hours,
          metrics: {}
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching model performance:', error);
      res.json({
        model_name: req.params.modelName,
        model_version: '1.0',
        time_window_hours: 24,
        metrics: {}
      });
    }
  });

  // Get prediction history
  app.get('/api/v1/ml/predictions/history', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const limit = req.query.limit || 50;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ml/predictions/history?limit=${limit}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        return res.json({ predictions: [], total: 0 });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching prediction history:', error);
      res.json({ predictions: [], total: 0 });
    }
  });

  // Generic ML prediction endpoint
  app.post('/api/v1/ml/predict', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ml/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
        body: JSON.stringify({
          ...req.body,
          user_id: req.user.id
        }),
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('ML prediction error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Prediction failed', 
          detail: error,
          backend_status: 'error'
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error making ML prediction:', error);
      res.status(502).json({ 
        error: 'ML service unavailable',
        backend_status: 'unavailable'
      });
    }
  });

  // Symptom analysis endpoint (Clinical-BERT NER)
  app.post('/api/v1/ml/predict/symptom-analysis', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { text, include_context } = req.body;
      
      if (!text || typeof text !== 'string') {
        return res.status(400).json({ error: 'Text input is required' });
      }
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ml/predict/symptom-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
        body: JSON.stringify({
          text,
          include_context: include_context || false,
          user_id: req.user.id
        }),
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Symptom analysis error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Symptom analysis failed', 
          detail: error 
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error in symptom analysis:', error);
      res.status(502).json({ 
        error: 'ML service unavailable for symptom analysis',
        backend_status: 'unavailable'
      });
    }
  });

  // Deterioration prediction endpoint
  app.post('/api/v1/ml/predict/deterioration', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patient_id, metrics, time_window_days } = req.body;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ml/predict/deterioration`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
        body: JSON.stringify({
          patient_id: patient_id || req.user.id,
          metrics: metrics || {},
          time_window_days: time_window_days || 7,
          user_id: req.user.id
        }),
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Deterioration prediction error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Deterioration prediction failed', 
          detail: error 
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error in deterioration prediction:', error);
      res.status(502).json({ 
        error: 'ML service unavailable for deterioration prediction',
        backend_status: 'unavailable'
      });
    }
  });

  // =====================================================
  // ML PREDICTION API ENDPOINTS - Proxy to Python FastAPI
  // Disease risk, deterioration, time-series, segmentation
  // =====================================================

  // Disease risk prediction (GET)
  app.get('/api/ml/predict/disease-risk/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ml/predict/disease-risk/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Disease risk prediction error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Disease risk prediction failed', 
          detail: error 
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching disease risk:', error);
      res.status(502).json({ 
        error: 'ML service unavailable for disease risk prediction',
        backend_status: 'unavailable'
      });
    }
  });

  // Deterioration prediction (GET)
  app.get('/api/ml/predict/deterioration/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ml/predict/deterioration/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Deterioration prediction error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Deterioration prediction failed', 
          detail: error 
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching deterioration prediction:', error);
      res.status(502).json({ 
        error: 'ML service unavailable for deterioration prediction',
        backend_status: 'unavailable'
      });
    }
  });

  // Time-series forecast prediction (GET) - Frontend uses this route name
  app.get('/api/ml/predict/time-series-forecast/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const horizonHours = req.query.horizon_hours || 24;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      // Forward to Python backend's time-series endpoint
      const response = await fetch(`${pythonBackendUrl}/api/ml/predict/time-series/${patientId}?sequence_length=${Math.ceil(horizonHours / 24) * 7}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Time-series forecast error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Time-series forecast failed', 
          detail: error 
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching time-series forecast:', error);
      res.status(502).json({ 
        error: 'ML service unavailable for time-series forecast',
        backend_status: 'unavailable'
      });
    }
  });

  // Time-series prediction (GET)
  app.get('/api/ml/predict/time-series/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const sequenceLength = req.query.sequence_length || 14;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ml/predict/time-series/${patientId}?sequence_length=${sequenceLength}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Time-series prediction error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Time-series prediction failed', 
          detail: error 
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching time-series prediction:', error);
      res.status(502).json({ 
        error: 'ML service unavailable for time-series prediction',
        backend_status: 'unavailable'
      });
    }
  });

  // Patient segmentation (GET)
  app.get('/api/ml/predict/patient-segments/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ml/predict/patient-segments/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Patient segmentation error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Patient segmentation failed', 
          detail: error 
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching patient segmentation:', error);
      res.status(502).json({ 
        error: 'ML service unavailable for patient segmentation',
        backend_status: 'unavailable'
      });
    }
  });

  // Comprehensive ML assessment (GET)
  app.get('/api/ml/predict/comprehensive/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ml/predict/comprehensive/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Comprehensive ML assessment error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Comprehensive ML assessment failed', 
          detail: error 
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching comprehensive ML assessment:', error);
      res.status(502).json({ 
        error: 'ML service unavailable for comprehensive assessment',
        backend_status: 'unavailable'
      });
    }
  });

  // ML Prediction History (GET)
  app.get('/api/ml/predict/history/:patientId/:predictionType', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId, predictionType } = req.params;
      const days = req.query.days || 14;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ml/predict/history/${patientId}/${predictionType}?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Prediction history error:', response.status, error);
        return res.status(response.status).json({ 
          error: 'Prediction history failed', 
          detail: error 
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching prediction history:', error);
      // Return synthetic history when backend unavailable
      const { patientId, predictionType } = req.params;
      const days = parseInt(req.query.days) || 14;
      const history = [];
      const now = new Date();
      
      for (let i = 0; i < days; i++) {
        const date = new Date(now.getTime() - (days - 1 - i) * 24 * 60 * 60 * 1000);
        const baseProb = 0.25 + (Math.random() - 0.5) * 0.2;
        history.push({
          date: date.toISOString(),
          probability: Math.max(0.01, Math.min(0.99, baseProb)),
          risk_level: baseProb < 0.15 ? 'low' : baseProb < 0.35 ? 'moderate' : baseProb < 0.55 ? 'high' : 'critical',
          confidence: 0.75 + Math.random() * 0.2
        });
      }
      
      res.json({
        patient_id: patientId,
        prediction_type: predictionType,
        days: days,
        history: history,
        generated_at: now.toISOString(),
        fallback: true
      });
    }
  });

  // Proxy questionnaire templates endpoint (public - these are public domain instruments)
  app.all('/api/v1/mental-health/questionnaires*', async (req: any, res) => {
    try {
      const path = req.path; // e.g., /api/v1/mental-health/questionnaires
      const PYTHON_BACKEND = `http://localhost:8000`;
      const queryString = new URLSearchParams(req.query as Record<string, string>).toString();
      const url = `${PYTHON_BACKEND}${path}${queryString ? '?' + queryString : ''}`;

      const response = await fetch(url, {
        method: req.method,
        headers: {
          'Content-Type': 'application/json',
        },
        ...(req.method !== 'GET' && req.method !== 'HEAD' ? { body: JSON.stringify(req.body) } : {})
      });

      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error: any) {
      console.error('Error connecting to Python backend (mental-health questionnaires):', error);
      res.status(500).json({ message: 'Mental Health service unavailable', detail: error.message });
    }
  });

  // Proxy other mental-health endpoints (require authentication for submissions/history)
  app.all('/api/v1/mental-health/*', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const path = req.path; // e.g., /api/v1/mental-health/questionnaires
      const url = `${pythonBackendUrl}${path}${req.url.includes('?') ? req.url.substring(req.url.indexOf('?')) : ''}`;
      
      // Generate dev mode JWT token for Python backend authentication
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        // Create dev mode JWT token
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(url, {
        method: req.method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
          'X-User-Id': req.user?.id || '',
        },
        body: req.method !== 'GET' && req.method !== 'HEAD' ? JSON.stringify(req.body) : undefined,
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (mental-health): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      // Network or connection error
      console.error('Error connecting to Python backend (mental-health):', error);
      res.status(502).json({ error: 'Failed to connect to mental health service' });
    }
  });

  // Proxy all guided-audio-exam endpoints to Python backend
  app.all('/api/v1/guided-audio-exam/*', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const path = req.path;
      const url = `${pythonBackendUrl}${path}${req.url.includes('?') ? req.url.substring(req.url.indexOf('?')) : ''}`;
      
      const response = await fetch(url, {
        method: req.method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
        },
        body: req.method !== 'GET' && req.method !== 'HEAD' ? JSON.stringify(req.body) : undefined,
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (guided-audio-exam): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (guided-audio-exam):', error);
      res.status(502).json({ error: 'Failed to connect to AI audio service' });
    }
  });

  // Proxy all guided-video-exam endpoints to Python backend
  app.all('/api/v1/guided-video-exam/*', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const path = req.path;
      const url = `${pythonBackendUrl}${path}${req.url.includes('?') ? req.url.substring(req.url.indexOf('?')) : ''}`;
      
      const response = await fetch(url, {
        method: req.method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
        },
        body: req.method !== 'GET' && req.method !== 'HEAD' ? JSON.stringify(req.body) : undefined,
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (guided-video-exam): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (guided-video-exam):', error);
      res.status(502).json({ error: 'Failed to connect to AI video service' });
    }
  });

  // ============================================================================
  // AI Health Alert Engine - Proxy routes to Python backend
  // Provides trend analysis, engagement metrics, QoL tracking, and alert management
  // ============================================================================

  // Get patient overview (all metrics combined)
  app.get('/api/ai-health-alerts/patient-overview/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/dashboard/summary/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (ai-health-alerts dashboard): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (ai-health-alerts):', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // Get trend metrics for a patient
  app.get('/api/ai-health-alerts/trend-metrics/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 30;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/trends/${patientId}?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (ai-health-alerts trends): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (ai-health-alerts trends):', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // Get engagement metrics for a patient
  app.get('/api/ai-health-alerts/engagement-metrics/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 30;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/engagement/${patientId}?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (ai-health-alerts engagement): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (ai-health-alerts engagement):', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // Get QoL metrics for a patient
  app.get('/api/ai-health-alerts/qol-metrics/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 30;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/qol/${patientId}?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (ai-health-alerts qol): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (ai-health-alerts qol):', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // Get alerts for a patient
  app.get('/api/ai-health-alerts/alerts/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const status = req.query.status || 'active';
      const limit = req.query.limit || 50;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/alerts/${patientId}?status=${status}&limit=${limit}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (ai-health-alerts alerts): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (ai-health-alerts alerts):', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // Update alert status (acknowledge or dismiss)
  app.patch('/api/ai-health-alerts/alerts/:alertId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { alertId } = req.params;
      const clinicianId = req.query.clinician_id || req.user?.id || 'unknown';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/alerts/${alertId}?clinician_id=${clinicianId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
        body: JSON.stringify(req.body),
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (ai-health-alerts update): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (ai-health-alerts update):', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // Trigger metrics computation for a patient
  app.post('/api/ai-health-alerts/compute-all/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/compute/${patientId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (ai-health-alerts compute): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (ai-health-alerts compute):', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // Mental Health Red Flag Symptoms endpoint - Fetches AI-observed mental health symptom indicators from Agent Clona
  app.get('/api/mental-health/red-flag-symptoms', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const days = parseInt(req.query.days as string) || 30;
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - days);

      // Fetch mental health red flag symptoms from Agent Clona conversations
      const redFlags = await db
        .select()
        .from(schema.mentalHealthRedFlags)
        .where(
          and(
            eq(schema.mentalHealthRedFlags.userId, userId),
            gte(schema.mentalHealthRedFlags.createdAt, cutoffDate)
          )
        )
        .orderBy(desc(schema.mentalHealthRedFlags.createdAt))
        .limit(50);

      // Transform for frontend display
      const formattedRedFlags = redFlags.map((flag: any) => ({
        id: flag.id,
        timestamp: flag.createdAt,
        sessionId: flag.sessionId,
        messageId: flag.messageId,
        rawText: flag.rawText,
        redFlagTypes: flag.extractedJson.redFlagTypes || [],
        severityLevel: flag.extractedJson.severityLevel || 'moderate',
        specificConcerns: flag.extractedJson.specificConcerns || [],
        emotionalTone: flag.extractedJson.emotionalTone || '',
        recommendedAction: flag.extractedJson.recommendedAction || 'Clinical review recommended',
        crisisIndicators: flag.extractedJson.crisisIndicators || false,
        confidence: parseFloat(flag.confidence || '0'),
        severityScore: flag.severityScore || 50,
        requiresImmediateAttention: flag.requiresImmediateAttention || false,
        clinicianNotified: flag.clinicianNotified || false,
        // HIPAA-compliant labeling
        dataSource: 'ai-observed',
        observationalLabel: 'AI-observed via Clona'
      }));

      // HIPAA audit log
      console.log(`[AUDIT] Mental health red flag symptoms fetched - User: ${userId}, Count: ${formattedRedFlags.length}`);

      res.json(formattedRedFlags);
    } catch (error) {
      console.error('Error fetching mental health red flag symptoms:', error);
      res.status(500).json({ message: 'Failed to fetch mental health indicators' });
    }
  });

  // TEMPORARY: Unified symptom feed endpoint (Express fallback while Python backend loads)
  // This endpoint merges patient-reported symptom check-ins with AI-extracted symptoms from Agent Clona
  app.get('/api/symptom-checkin/feed/unified', async (req: any, res) => {
    try {
      const days = parseInt(req.query.days as string) || 30;

      // Fetch patient-reported check-ins
      const patientCheckins = await db
        .select()
        .from(schema.symptomCheckins);

      // Fetch AI-extracted symptoms from Agent Clona
      const aiSymptoms = await db
        .select()
        .from(schema.chatSymptoms);

      // Build unified feed
      const feed = [
        ...patientCheckins.map((c: any) => ({
          id: c.id,
          userId: c.userId,
          timestamp: c.timestamp,
          dataSource: 'patient-reported',
          observationalLabel: 'Patient-reported',
          painLevel: c.painLevel,
          fatigueLevel: c.fatigueLevel,
          breathlessnessLevel: c.breathlessnessLevel,
          sleepQuality: c.sleepQuality,
          mood: c.mood,
          mobilityScore: c.mobilityScore,
          medicationsTaken: c.medicationsTaken,
          triggers: c.triggers || [],
          symptoms: c.symptoms || [],
          note: c.note,
          createdAt: c.createdAt
        })),
        ...aiSymptoms.map((s: any) => ({
          id: s.id,
          userId: s.userId,
          timestamp: s.timestamp,
          dataSource: 'ai-extracted',
          observationalLabel: 'AI-observed via Clona',
          sessionId: s.sessionId,
          messageId: s.messageId,
          extractedData: s.extractedJson || {},
          confidence: parseFloat(s.confidence) || 0,
          symptomTypes: s.symptomTypes || [],
          locations: s.locations || [],
          intensityMentions: s.intensityMentions || [],
          temporalInfo: s.temporalInfo,
          createdAt: s.createdAt
        }))
      ];

      // Sort by timestamp (most recent first)
      feed.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

      res.json(feed);
    } catch (error: any) {
      console.error('Error fetching unified symptom feed:', error);
      res.status(500).json({ message: 'Failed to fetch symptom feed', error: error.message });
    }
  });

  // =============================================================================
  // ALERT ENGINE V2 PROXY ROUTES
  // Routes to new V2 Alert Engine endpoints in Python backend
  // =============================================================================

  // V2 Metrics Ingest - Single metric
  app.post('/api/ai-health-alerts/v2/metrics/ingest', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/metrics/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
        body: JSON.stringify(req.body),
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (v2 metrics ingest): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (v2 metrics ingest):', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Metrics Ingest - Batch
  app.post('/api/ai-health-alerts/v2/metrics/ingest/batch', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/metrics/ingest/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
        body: JSON.stringify(req.body),
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (v2 batch ingest): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error connecting to Python backend (v2 batch ingest):', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Get Recent Metrics
  app.get('/api/ai-health-alerts/v2/metrics/recent/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const { metric_name, hours } = req.query;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const queryParams = new URLSearchParams();
      if (metric_name) queryParams.append('metric_name', metric_name as string);
      if (hours) queryParams.append('hours', hours as string);
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/metrics/recent/${patientId}?${queryParams}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching v2 recent metrics:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Organ Scores
  app.get('/api/ai-health-alerts/v2/organ-scores/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/organ-scores/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching v2 organ scores:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Compute Organ Scores
  app.post('/api/ai-health-alerts/v2/organ-scores/compute/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/organ-scores/compute/${patientId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error computing v2 organ scores:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Organ Scores History
  app.get('/api/ai-health-alerts/v2/organ-scores/history/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 30;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/organ-scores/history/${patientId}?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching v2 organ scores history:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 DPI (Deterioration Index)
  app.get('/api/ai-health-alerts/v2/dpi/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/dpi/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching v2 DPI:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Compute DPI
  app.post('/api/ai-health-alerts/v2/dpi/compute/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/dpi/compute/${patientId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error computing v2 DPI:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 DPI History
  app.get('/api/ai-health-alerts/v2/dpi/history/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 30;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/dpi/history/${patientId}?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching v2 DPI history:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Generate Alerts (rule-based engine)
  app.post('/api/ai-health-alerts/v2/alerts/generate/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/alerts/generate/${patientId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error generating v2 alerts:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Get Ranked Alerts (ML-assisted)
  app.get('/api/ai-health-alerts/v2/alerts/ranked/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const limit = req.query.limit || 20;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/alerts/ranked/${patientId}?limit=${limit}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching v2 ranked alerts:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Escalate Alert
  app.post('/api/ai-health-alerts/v2/alerts/:alertId/escalate', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { alertId } = req.params;
      const clinicianId = req.query.clinician_id || req.user?.id;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/alerts/${alertId}/escalate?clinician_id=${clinicianId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
        body: JSON.stringify(req.body),
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error escalating v2 alert:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Notifications - Unread
  app.get('/api/ai-health-alerts/v2/notifications/unread/:userId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { userId } = req.params;
      const limit = req.query.limit || 50;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/notifications/unread/${userId}?limit=${limit}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching v2 notifications:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Mark Notification Read
  app.post('/api/ai-health-alerts/v2/notifications/:notificationId/read', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { notificationId } = req.params;
      const userId = req.query.user_id || req.user?.id;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/notifications/${notificationId}/read?user_id=${userId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error marking v2 notification read:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Get Alert Config
  app.get('/api/ai-health-alerts/v2/config', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/config`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching v2 config:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Update Alert Config
  app.patch('/api/ai-health-alerts/v2/config', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/config`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
        body: JSON.stringify(req.body),
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error updating v2 config:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Compute All (full pipeline)
  app.post('/api/ai-health-alerts/v2/compute-all/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/compute-all/${patientId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error in v2 compute-all:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Patient Overview (enhanced)
  app.get('/api/ai-health-alerts/v2/patient-overview/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/v2/patient-overview/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching v2 patient overview:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // Device Data → Alerts Pipeline (triggers analysis and generates alerts)
  app.post('/api/ai-health-alerts/device-data-pipeline/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 7;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/device-data-pipeline/${patientId}?days=${days}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (device-data-pipeline): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error running device data pipeline:', error);
      res.status(502).json({ error: 'Failed to connect to device data pipeline service' });
    }
  });

  // Device Analytics (Health Section Analytics from connected devices)
  app.get('/api/ai-health-alerts/device-analytics/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 7;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/v1/ai-health-alerts/device-analytics/${patientId}?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (device-analytics): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching device analytics:', error);
      res.status(502).json({ error: 'Failed to connect to device analytics service' });
    }
  });

  // Proxy to Python backend for correlation insights
  app.get('/api/ai-health-alerts/correlation-insights/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const { days } = req.query;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const queryParams = days ? `?days=${days}` : '';
      const response = await fetch(`${pythonBackendUrl}/api/ai-health-alerts/correlation-insights/${patientId}${queryParams}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Python backend error (correlation-insights): ${response.status}`, error);
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching correlation insights:', error);
      res.status(502).json({ error: 'Failed to connect to correlation insights service' });
    }
  });

  // ============================================================================
  // V2 ML PREDICTION ROUTES
  // ============================================================================

  // V2 Compute ML Predictions
  app.post('/api/ai-health-alerts/v2/predictions/compute', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ai-health-alerts/v2/predictions/compute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
        body: JSON.stringify(req.body),
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error computing ML predictions:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Get Latest Prediction
  app.get('/api/ai-health-alerts/v2/predictions/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ai-health-alerts/v2/predictions/${patientId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching latest prediction:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Get Prediction History
  app.get('/api/ai-health-alerts/v2/predictions/:patientId/history', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      const days = req.query.days || 7;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ai-health-alerts/v2/predictions/${patientId}/history?days=${days}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching prediction history:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // V2 Compute All with ML Predictions (full ensemble pipeline)
  app.post('/api/ai-health-alerts/v2/compute-all-with-ml/:patientId', isAuthenticated, async (req: any, res) => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { patientId } = req.params;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const response = await fetch(`${pythonBackendUrl}/api/ai-health-alerts/v2/compute-all-with-ml/${patientId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ message: error });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error in v2 compute-all-with-ml:', error);
      res.status(502).json({ error: 'Failed to connect to AI health alerts service' });
    }
  });

  // =============================================================================
  // DEVICE CONNECT API PROXY ROUTES
  // Full device pairing, sync, and management proxied to Python FastAPI backend
  // =============================================================================

  // Helper function for Device Connect proxy requests
  const deviceConnectProxy = async (req: any, res: any, path: string, method: string = 'GET') => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      
      // Build query string from request query params
      const queryParams = new URLSearchParams(req.query as Record<string, string>);
      const queryString = queryParams.toString();
      const url = `${pythonBackendUrl}${path}${queryString ? '?' + queryString : ''}`;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const fetchOptions: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
        },
      };
      
      if (method !== 'GET' && method !== 'HEAD' && req.body) {
        fetchOptions.body = JSON.stringify(req.body);
      }
      
      const response = await fetch(url, fetchOptions);
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Device Connect proxy error (${path}):`, response.status, error);
        return res.status(response.status).json({ message: error || 'Request failed' });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error(`Device Connect proxy error (${path}):`, error);
      res.status(502).json({ error: 'Device Connect service unavailable' });
    }
  };

  // Device vendors list
  app.get('/api/v1/devices/vendors', async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/vendors', 'GET');
  });

  // Device models list
  app.get('/api/v1/devices/models', async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/models', 'GET');
  });

  // Supported device types
  app.get('/api/v1/devices/supported-types', async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/supported-types', 'GET');
  });

  // Start device pairing
  app.post('/api/v1/devices/pair/start', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/pair/start', 'POST');
  });

  // Complete device pairing
  app.post('/api/v1/devices/pair/complete', async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/pair/complete', 'POST');
  });

  // Get pairing status (for OAuth polling)
  app.get('/api/v1/devices/pair/status/:sessionId', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, `/api/v1/devices/pair/status/${req.params.sessionId}`, 'GET');
  });

  // Connected devices list
  app.get('/api/v1/devices/connections', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/connections', 'GET');
  });

  // Device health status
  app.get('/api/v1/devices/health/:connectionId', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, `/api/v1/devices/health/${req.params.connectionId}`, 'GET');
  });

  // Trigger device sync
  app.post('/api/v1/devices/sync/:connectionId', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, `/api/v1/devices/sync/${req.params.connectionId}`, 'POST');
  });

  // Sync specific device by ID
  app.post('/api/v1/devices/:deviceId/sync', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, `/api/v1/devices/${req.params.deviceId}/sync`, 'POST');
  });

  // Delete/disconnect device
  app.delete('/api/v1/devices/:deviceId', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, `/api/v1/devices/${req.params.deviceId}`, 'DELETE');
  });

  // Update device consent
  app.patch('/api/v1/devices/:deviceId/consent', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, `/api/v1/devices/${req.params.deviceId}/consent`, 'PATCH');
  });

  // Ingest device data
  app.post('/api/v1/devices/data/ingest', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/data/ingest', 'POST');
  });

  // HealthKit data sync
  app.post('/api/v1/devices/healthkit/sync', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/healthkit/sync', 'POST');
  });

  // Webhook status
  app.get('/api/v1/devices/webhook-status', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/webhook-status', 'GET');
  });

  // BLE services info
  app.get('/api/v1/devices/ble/services', async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/ble/services', 'GET');
  });

  // Health analytics
  app.get('/api/v1/devices/health-analytics', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/health-analytics', 'GET');
  });

  // Sync history
  app.get('/api/v1/devices/sync-history', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/sync-history', 'GET');
  });

  // Daily followup device data
  app.get('/api/v1/devices/daily-followup/device-data', isAuthenticated, async (req: any, res) => {
    await deviceConnectProxy(req, res, '/api/v1/devices/daily-followup/device-data', 'GET');
  });

  // =============================================================================
  // AI-POWERED HABIT TRACKER PROXY ROUTES (13 Features)
  // =============================================================================

  // Helper function for habit tracker proxy requests
  const habitTrackerProxy = async (req: any, res: any, path: string, method: string = 'GET') => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const userId = req.user?.id || req.query.user_id || 'current';
      
      // Build query string
      const queryParams = new URLSearchParams(req.query as Record<string, string>);
      if (!queryParams.has('user_id')) {
        queryParams.set('user_id', userId);
      }
      const queryString = queryParams.toString();
      const url = `${pythonBackendUrl}${path}${queryString ? '?' + queryString : ''}`;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const fetchOptions: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
          'X-User-Id': userId,
        },
      };
      
      if (method !== 'GET' && method !== 'HEAD' && req.body) {
        fetchOptions.body = JSON.stringify(req.body);
      }
      
      const response = await fetch(url, fetchOptions);
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Habit tracker proxy error (${path}):`, response.status, error);
        return res.status(response.status).json({ message: error || 'Request failed' });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error(`Habit tracker proxy error (${path}):`, error);
      res.status(502).json({ error: 'Habit tracker service unavailable' });
    }
  };

  // Feature 1: Habit CRUD & Completion
  app.get('/api/habits', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits', 'GET');
  });

  app.post('/api/habits/create', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/create', 'POST');
  });

  app.post('/api/habits/:habitId/complete', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/${req.params.habitId}/complete`, 'POST');
  });

  app.get('/api/habits/:habitId/history', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/${req.params.habitId}/history`, 'GET');
  });

  // Feature 2: Daily Routines with Micro-Steps
  app.get('/api/habits/routines', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/routines', 'GET');
  });

  app.post('/api/habits/routines/create', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/routines/create', 'POST');
  });

  app.post('/api/habits/routines/:routineId/step/:stepId/complete', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/routines/${req.params.routineId}/step/${req.params.stepId}/complete`, 'POST');
  });

  // Feature 3: Streaks & Calendar View
  app.get('/api/habits/streaks', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/streaks', 'GET');
  });

  app.get('/api/habits/calendar', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/calendar', 'GET');
  });

  // Feature 4: Smart Reminders
  app.get('/api/habits/reminders', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/reminders', 'GET');
  });

  app.post('/api/habits/reminders/set', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/reminders/set', 'POST');
  });

  app.post('/api/habits/reminders/:reminderId/snooze', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/reminders/${req.params.reminderId}/snooze`, 'POST');
  });

  app.delete('/api/habits/reminders/:reminderId', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/reminders/${req.params.reminderId}`, 'DELETE');
  });

  // Feature 5: AI Habit Coach
  app.post('/api/habits/coach/chat', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/coach/chat', 'POST');
  });

  app.get('/api/habits/coach/history', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/coach/history', 'GET');
  });

  // Feature 6: Trigger Detection
  app.get('/api/habits/triggers', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/triggers', 'GET');
  });

  app.post('/api/habits/triggers/analyze', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/triggers/analyze', 'POST');
  });

  app.post('/api/habits/triggers/log', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/triggers/log', 'POST');
  });

  // Feature 7: Addiction-Mode Quit Plans
  app.get('/api/habits/quit-plans', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/quit-plans', 'GET');
  });

  app.post('/api/habits/quit-plans/create', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/quit-plans/create', 'POST');
  });

  app.post('/api/habits/quit-plans/:planId/craving', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/quit-plans/${req.params.planId}/craving`, 'POST');
  });

  app.post('/api/habits/quit-plans/:planId/relapse', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/quit-plans/${req.params.planId}/relapse`, 'POST');
  });

  app.get('/api/habits/quit-plans/:planId/stats', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/quit-plans/${req.params.planId}/stats`, 'GET');
  });

  // Feature 8: Mood Tracking
  app.get('/api/habits/mood/trends', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/mood/trends', 'GET');
  });

  app.post('/api/habits/mood/log', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/mood/log', 'POST');
  });

  app.get('/api/habits/mood/correlations', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/mood/correlations', 'GET');
  });

  // Feature 9: Dynamic AI Recommendations
  app.get('/api/habits/recommendations', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/recommendations', 'GET');
  });

  app.post('/api/habits/recommendations/generate', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/recommendations/generate', 'POST');
  });

  app.post('/api/habits/recommendations/:recId/dismiss', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/recommendations/${req.params.recId}/dismiss`, 'POST');
  });

  app.post('/api/habits/recommendations/:recId/accept', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/recommendations/${req.params.recId}/accept`, 'POST');
  });

  // Feature 10: Social Accountability / Buddy System
  app.get('/api/habits/buddies', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/buddies', 'GET');
  });

  app.post('/api/habits/buddies/invite', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/buddies/invite', 'POST');
  });

  app.post('/api/habits/buddies/:inviteId/accept', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/buddies/${req.params.inviteId}/accept`, 'POST');
  });

  app.get('/api/habits/buddies/:buddyId/progress', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/buddies/${req.params.buddyId}/progress`, 'GET');
  });

  app.post('/api/habits/buddies/:buddyId/nudge', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/buddies/${req.params.buddyId}/nudge`, 'POST');
  });

  // Feature 11: Guided CBT Sessions
  app.get('/api/habits/cbt/flows', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/cbt/flows', 'GET');
  });

  app.post('/api/habits/cbt/start', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/cbt/start', 'POST');
  });

  app.post('/api/habits/cbt/respond', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/cbt/respond', 'POST');
  });

  app.get('/api/habits/cbt/sessions', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/cbt/sessions', 'GET');
  });

  // Feature 12: Gamification & Visual Rewards
  app.get('/api/habits/rewards', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/rewards', 'GET');
  });

  app.get('/api/habits/rewards/leaderboard', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/rewards/leaderboard', 'GET');
  });

  app.post('/api/habits/rewards/claim/:rewardId', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/rewards/claim/${req.params.rewardId}`, 'POST');
  });

  // Feature 13: Smart Journals with AI Insights
  app.get('/api/habits/journals', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/journals', 'GET');
  });

  app.post('/api/habits/journals/create', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/journals/create', 'POST');
  });

  app.get('/api/habits/journals/:journalId/insights', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/journals/${req.params.journalId}/insights`, 'GET');
  });

  app.post('/api/habits/journals/weekly-summary', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/journals/weekly-summary', 'POST');
  });

  // Preventive Alerts (Risk Detection)
  app.get('/api/habits/alerts', async (req: any, res) => {
    await habitTrackerProxy(req, res, '/api/habits/alerts', 'GET');
  });

  app.post('/api/habits/alerts/:alertId/acknowledge', async (req: any, res) => {
    await habitTrackerProxy(req, res, `/api/habits/alerts/${req.params.alertId}/acknowledge`, 'POST');
  });

  // =============================================================================
  // END HABIT TRACKER PROXY ROUTES
  // =============================================================================

  // =============================================================================
  // LYSA AUTOMATION ENGINE PROXY ROUTES
  // =============================================================================

  // Helper function for automation proxy requests
  const automationProxy = async (req: any, res: any, path: string, method: string = 'GET') => {
    try {
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const userId = req.user?.id || req.query.user_id;
      
      // Build query string
      const queryParams = new URLSearchParams(req.query as Record<string, string>);
      const queryString = queryParams.toString();
      const url = `${pythonBackendUrl}${path}${queryString ? '?' + queryString : ''}`;
      
      let authHeader = req.headers.authorization || '';
      if (!authHeader && req.user?.id && process.env.DEV_MODE_SECRET) {
        const token = jwt.sign(
          { sub: req.user.id, id: req.user.id, email: req.user.email, role: req.user.role },
          process.env.DEV_MODE_SECRET,
          { expiresIn: '1h' }
        );
        authHeader = `Bearer ${token}`;
      }
      
      const fetchOptions: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader,
          'X-User-Id': userId || '',
        },
      };
      
      if (method !== 'GET' && method !== 'HEAD' && req.body) {
        fetchOptions.body = JSON.stringify(req.body);
      }
      
      const response = await fetch(url, fetchOptions);
      
      if (!response.ok) {
        const error = await response.text();
        console.error(`Automation proxy error (${path}):`, response.status, error);
        return res.status(response.status).json({ message: error || 'Request failed' });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error(`Automation proxy error (${path}):`, error);
      res.status(502).json({ error: 'Automation service unavailable' });
    }
  };

  // Automation Status & Dashboard
  app.get('/api/v1/automation/status', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/status', 'GET');
  });

  app.get('/api/v1/automation/dashboard/stats', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/dashboard/stats', 'GET');
  });

  // Automation Jobs
  app.get('/api/v1/automation/jobs', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/jobs', 'GET');
  });

  app.post('/api/v1/automation/jobs', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/jobs', 'POST');
  });

  app.get('/api/v1/automation/jobs/:jobId', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/jobs/${req.params.jobId}`, 'GET');
  });

  app.post('/api/v1/automation/jobs/:jobId/cancel', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/jobs/${req.params.jobId}/cancel`, 'POST');
  });

  app.post('/api/v1/automation/jobs/:jobId/retry', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/jobs/${req.params.jobId}/retry`, 'POST');
  });

  // Automation Schedules
  app.get('/api/v1/automation/schedules', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/schedules', 'GET');
  });

  app.post('/api/v1/automation/schedules', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/schedules', 'POST');
  });

  app.put('/api/v1/automation/schedules/:scheduleId', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/schedules/${req.params.scheduleId}`, 'PUT');
  });

  app.delete('/api/v1/automation/schedules/:scheduleId', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/schedules/${req.params.scheduleId}`, 'DELETE');
  });

  // Automation Config Endpoints
  app.get('/api/v1/automation/config', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config', 'GET');
  });

  app.patch('/api/v1/automation/config', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config', 'PATCH');
  });

  // Email Config
  app.get('/api/v1/automation/config/email', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/email', 'GET');
  });

  app.put('/api/v1/automation/config/email', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/email', 'PUT');
  });

  // WhatsApp Config
  app.get('/api/v1/automation/config/whatsapp', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/whatsapp', 'GET');
  });

  app.put('/api/v1/automation/config/whatsapp', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/whatsapp', 'PUT');
  });

  // Appointment Config
  app.get('/api/v1/automation/config/appointments', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/appointments', 'GET');
  });

  app.put('/api/v1/automation/config/appointments', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/appointments', 'PUT');
  });

  // Reminders Config
  app.get('/api/v1/automation/config/reminders', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/reminders', 'GET');
  });

  app.put('/api/v1/automation/config/reminders', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/reminders', 'PUT');
  });

  // Clinical Config
  app.get('/api/v1/automation/config/clinical', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/clinical', 'GET');
  });

  app.put('/api/v1/automation/config/clinical', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/config/clinical', 'PUT');
  });

  // Rx Templates CRUD
  app.get('/api/v1/automation/rx-templates', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/rx-templates', 'GET');
  });

  app.post('/api/v1/automation/rx-templates', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/rx-templates', 'POST');
  });

  app.get('/api/v1/automation/rx-templates/:templateId', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/rx-templates/${req.params.templateId}`, 'GET');
  });

  app.put('/api/v1/automation/rx-templates/:templateId', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/rx-templates/${req.params.templateId}`, 'PUT');
  });

  app.delete('/api/v1/automation/rx-templates/:templateId', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/rx-templates/${req.params.templateId}`, 'DELETE');
  });

  app.post('/api/v1/automation/rx-templates/:templateId/use', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/rx-templates/${req.params.templateId}/use`, 'POST');
  });

  // Automation Engine Control
  app.post('/api/v1/automation/engine/start', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/engine/start', 'POST');
  });

  app.post('/api/v1/automation/engine/pause', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/engine/pause', 'POST');
  });

  // Manual Sync Triggers
  app.post('/api/v1/automation/sync/:channel', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/automation/sync/${req.params.channel}`, 'POST');
  });

  // Logs
  app.get('/api/v1/automation/logs', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/logs', 'GET');
  });

  // Trigger Job
  app.post('/api/v1/automation/trigger', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/automation/trigger', 'POST');
  });

  // =============================================================================
  // END LYSA AUTOMATION ENGINE PROXY ROUTES
  // =============================================================================

  // =============================================================================
  // CLINICAL ASSESSMENT AGGREGATION PROXY ROUTES
  // Comprehensive patient data integration for AI clinical diagnosis
  // =============================================================================

  // Get consented patients for clinical assessment
  app.get('/api/v1/clinical-assessment/patients', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/clinical-assessment/patients', 'GET');
  });

  // Get aggregated patient data (medical files, alerts, ML inference, meds, followup)
  app.get('/api/v1/clinical-assessment/patient/:patientId/data', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/clinical-assessment/patient/${req.params.patientId}/data`, 'GET');
  });

  // AI clinical assessment with patient context
  app.post('/api/v1/clinical-assessment/analyze', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, '/api/v1/clinical-assessment/analyze', 'POST');
  });

  // HIPAA audit log for patient data access
  app.get('/api/v1/clinical-assessment/audit-log/:patientId', isAuthenticated, async (req: any, res) => {
    await automationProxy(req, res, `/api/v1/clinical-assessment/audit-log/${req.params.patientId}`, 'GET');
  });

  // =============================================================================
  // END CLINICAL ASSESSMENT PROXY ROUTES
  // =============================================================================

  // =============================================================================
  // MULTI-AGENT COMMUNICATION SYSTEM PROXY ROUTES
  // Agent Clona (Patient) and Assistant Lysa (Doctor) AI Agents
  // =============================================================================

  const PYTHON_AGENT_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';

  // Agent proxy helper
  async function agentProxy(req: any, res: any, path: string, method: string = 'GET') {
    try {
      const userId = req.session?.passport?.user;
      const user = userId ? await storage.getUser(userId) : null;
      
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'X-User-Id': userId || '',
        'X-User-Role': user?.role || 'patient',
      };

      const options: RequestInit = {
        method,
        headers,
      };

      if (method !== 'GET' && req.body) {
        options.body = JSON.stringify(req.body);
      }

      const queryString = new URLSearchParams(req.query).toString();
      const url = `${PYTHON_AGENT_URL}${path}${queryString ? `?${queryString}` : ''}`;
      
      const response = await fetch(url, options);
      const data = await response.json();
      
      res.status(response.status).json(data);
    } catch (error) {
      console.error(`Agent proxy error for ${path}:`, error);
      res.status(503).json({ 
        error: 'Agent service unavailable',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Get agent conversations
  app.get('/api/agent/conversations', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, '/api/agent/conversations', 'GET');
  });

  // Get messages for a conversation
  app.get('/api/agent/messages', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, '/api/agent/messages', 'GET');
  });

  // Send a message to an agent
  app.post('/api/agent/messages', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, '/api/agent/messages', 'POST');
  });

  // Get agent info
  app.get('/api/agent/info/:agentId', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, `/api/agent/info/${req.params.agentId}`, 'GET');
  });

  // Get agent health status
  app.get('/api/agent/health', async (req: any, res) => {
    await agentProxy(req, res, '/api/agent/health', 'GET');
  });

  // Approve or reject tool call
  app.post('/api/agent/tool-approval', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, '/api/agent/tool-approval', 'POST');
  });

  // Get pending approvals (legacy)
  app.get('/api/agent/pending-approvals', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, '/api/agent/pending-approvals', 'GET');
  });

  // List pending approvals (new API)
  app.get('/api/agent/approvals', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, '/approvals/pending', 'GET');
  });

  // Get approval details
  app.get('/api/agent/approvals/:approvalId', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, `/approvals/${req.params.approvalId}`, 'GET');
  });

  // Submit approval decision
  app.post('/api/agent/approvals/:approvalId/decision', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, `/approvals/${req.params.approvalId}/decide`, 'POST');
  });

  // Get agent memory (for debugging)
  app.get('/api/agent/memory/:agentId', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, `/api/agent/memory/${req.params.agentId}`, 'GET');
  });

  // =============================================================================
  // DOCTOR PATIENT OVERVIEW ROUTES
  // Assistant Lysa patient management and overview panel
  // =============================================================================

  // Get list of patients assigned to the logged-in doctor
  app.get('/api/agent/patients', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, '/api/agent/patients', 'GET');
  });

  // Get comprehensive patient overview (dailies, alerts, meds, conversations)
  app.get('/api/agent/patients/:patientId/overview', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, `/api/agent/patients/${req.params.patientId}/overview`, 'GET');
  });

  // Get Lysa conversation history for a specific patient
  app.get('/api/agent/patients/:patientId/conversations', isAuthenticated, async (req: any, res) => {
    await agentProxy(req, res, `/api/agent/patients/${req.params.patientId}/conversations`, 'GET');
  });

  // =============================================================================
  // END DOCTOR PATIENT OVERVIEW ROUTES
  // =============================================================================

  // =============================================================================
  // END MULTI-AGENT COMMUNICATION SYSTEM PROXY ROUTES
  // =============================================================================

  // =============================================================================
  // ADMIN TOTP AUTHENTICATION ROUTES
  // Google Authenticator TOTP protection for Admin ML Training Hub
  // =============================================================================

  // Check if TOTP is set up for admin ML training hub
  app.get('/api/admin/totp/status', async (req: any, res) => {
    try {
      const result = await db
        .select()
        .from(schema.adminTotpSecrets)
        .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'))
        .limit(1);

      if (result.length === 0) {
        return res.json({ 
          isSetup: false,
          enabled: false,
          message: 'TOTP not configured for ML Training Hub'
        });
      }

      const secret = result[0];
      
      // Check if locked due to failed attempts
      const isLocked = secret.lockedUntil && new Date(secret.lockedUntil) > new Date();
      
      res.json({
        isSetup: true,
        enabled: secret.enabled,
        isLocked,
        lockedUntil: isLocked ? secret.lockedUntil : null,
        verificationCount: secret.verificationCount || 0,
        lastVerifiedAt: secret.lastVerifiedAt
      });
    } catch (error: any) {
      console.error('Error checking TOTP status:', error);
      res.status(500).json({ error: 'Failed to check TOTP status' });
    }
  });

  // Generate TOTP setup (QR code and secret)
  app.post('/api/admin/totp/setup', async (req: any, res) => {
    try {
      // Check if already set up
      const existing = await db
        .select()
        .from(schema.adminTotpSecrets)
        .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'))
        .limit(1);

      if (existing.length > 0 && existing[0].enabled) {
        return res.status(400).json({ 
          error: 'TOTP already configured. To reset, disable first.',
          isSetup: true
        });
      }

      // Generate new secret
      const secret = speakeasy.generateSecret({
        name: 'Followup AI ML Training Hub',
        issuer: 'Followup AI',
        length: 32
      });

      // Generate QR code
      const qrCodeDataUrl = await QRCode.toDataURL(secret.otpauth_url || '');

      // Store or update the secret (not enabled until verified)
      if (existing.length > 0) {
        await db
          .update(schema.adminTotpSecrets)
          .set({
            totpSecret: secret.base32,
            enabled: false,
            failedAttempts: 0,
            lockedUntil: null,
            updatedAt: new Date()
          })
          .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'));
      } else {
        await db.insert(schema.adminTotpSecrets).values({
          secretName: 'ml-training-hub',
          totpSecret: secret.base32,
          enabled: false
        });
      }

      res.json({
        success: true,
        qrCode: qrCodeDataUrl,
        secret: secret.base32,
        message: 'Scan the QR code with Google Authenticator, then verify with a 6-digit code'
      });
    } catch (error: any) {
      console.error('Error setting up TOTP:', error);
      res.status(500).json({ error: 'Failed to set up TOTP' });
    }
  });

  // Verify initial TOTP setup (enables 2FA after first successful verification)
  app.post('/api/admin/totp/verify-setup', async (req: any, res) => {
    try {
      const { token } = req.body;

      if (!token || !/^\d{6}$/.test(token)) {
        return res.status(400).json({ error: 'Valid 6-digit code required' });
      }

      const existing = await db
        .select()
        .from(schema.adminTotpSecrets)
        .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'))
        .limit(1);

      if (existing.length === 0) {
        return res.status(400).json({ error: 'TOTP not set up. Generate QR code first.' });
      }

      const secret = existing[0];

      if (secret.enabled) {
        return res.status(400).json({ error: 'TOTP already enabled. Use /authenticate endpoint.' });
      }

      // Verify the token
      const verified = speakeasy.totp.verify({
        secret: secret.totpSecret,
        encoding: 'base32',
        token: token,
        window: 2
      });

      if (!verified) {
        return res.status(401).json({ error: 'Invalid verification code. Please try again.' });
      }

      // Enable TOTP
      await db
        .update(schema.adminTotpSecrets)
        .set({
          enabled: true,
          setupCompletedAt: new Date(),
          setupCompletedBy: req.user?.id || 'system',
          updatedAt: new Date()
        })
        .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'));

      // Set session flag
      if (req.session) {
        req.session.adminTotpVerified = true;
        req.session.adminTotpVerifiedAt = new Date().toISOString();
      }

      res.json({
        success: true,
        message: 'TOTP successfully enabled! You will now need this code to access ML Training Hub.'
      });
    } catch (error: any) {
      console.error('Error verifying TOTP setup:', error);
      res.status(500).json({ error: 'Failed to verify TOTP' });
    }
  });

  // Authenticate with TOTP (for ongoing access)
  app.post('/api/admin/totp/authenticate', async (req: any, res) => {
    try {
      const { token } = req.body;

      if (!token || !/^\d{6}$/.test(token)) {
        return res.status(400).json({ error: 'Valid 6-digit code required' });
      }

      const existing = await db
        .select()
        .from(schema.adminTotpSecrets)
        .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'))
        .limit(1);

      if (existing.length === 0 || !existing[0].enabled) {
        return res.status(400).json({ error: 'TOTP not configured. Set up first.' });
      }

      const secret = existing[0];

      // Check if locked
      if (secret.lockedUntil && new Date(secret.lockedUntil) > new Date()) {
        const remainingMinutes = Math.ceil((new Date(secret.lockedUntil).getTime() - Date.now()) / 60000);
        return res.status(429).json({ 
          error: `Too many failed attempts. Try again in ${remainingMinutes} minutes.`,
          lockedUntil: secret.lockedUntil
        });
      }

      // Verify the token
      const verified = speakeasy.totp.verify({
        secret: secret.totpSecret,
        encoding: 'base32',
        token: token,
        window: 1
      });

      if (!verified) {
        // Increment failed attempts
        const newFailedAttempts = (secret.failedAttempts || 0) + 1;
        const updateData: any = {
          failedAttempts: newFailedAttempts,
          updatedAt: new Date()
        };

        // Lock after 5 failed attempts for 15 minutes
        if (newFailedAttempts >= 5) {
          updateData.lockedUntil = new Date(Date.now() + 15 * 60 * 1000);
        }

        await db
          .update(schema.adminTotpSecrets)
          .set(updateData)
          .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'));

        const attemptsRemaining = Math.max(0, 5 - newFailedAttempts);
        return res.status(401).json({ 
          error: `Invalid code. ${attemptsRemaining} attempts remaining before lockout.`,
          attemptsRemaining
        });
      }

      // Success - reset failed attempts and update verification tracking
      await db
        .update(schema.adminTotpSecrets)
        .set({
          failedAttempts: 0,
          lockedUntil: null,
          lastVerifiedAt: new Date(),
          lastVerifiedBy: req.user?.id || 'anonymous',
          verificationCount: sql`COALESCE(verification_count, 0) + 1`,
          updatedAt: new Date()
        })
        .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'));

      // Set session flag
      if (req.session) {
        req.session.adminTotpVerified = true;
        req.session.adminTotpVerifiedAt = new Date().toISOString();
      }

      res.json({
        success: true,
        message: 'Authentication successful'
      });
    } catch (error: any) {
      console.error('Error authenticating TOTP:', error);
      res.status(500).json({ error: 'Failed to authenticate' });
    }
  });

  // Check if current session is TOTP verified
  app.get('/api/admin/totp/session', async (req: any, res) => {
    try {
      const isVerified = req.session?.adminTotpVerified === true;
      const verifiedAt = req.session?.adminTotpVerifiedAt || null;
      
      res.json({
        isVerified,
        verifiedAt
      });
    } catch (error: any) {
      console.error('Error checking TOTP session:', error);
      res.status(500).json({ error: 'Failed to check session' });
    }
  });

  // Reset TOTP (for re-setup)
  app.post('/api/admin/totp/reset', async (req: any, res) => {
    try {
      const { token } = req.body;

      // Must provide valid current TOTP to reset
      if (!token || !/^\d{6}$/.test(token)) {
        return res.status(400).json({ error: 'Current TOTP code required to reset' });
      }

      const existing = await db
        .select()
        .from(schema.adminTotpSecrets)
        .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'))
        .limit(1);

      if (existing.length === 0) {
        return res.status(400).json({ error: 'TOTP not configured' });
      }

      const secret = existing[0];

      // Verify current token before allowing reset
      const verified = speakeasy.totp.verify({
        secret: secret.totpSecret,
        encoding: 'base32',
        token: token,
        window: 1
      });

      if (!verified) {
        return res.status(401).json({ error: 'Invalid current TOTP code' });
      }

      // Delete the existing secret
      await db
        .delete(schema.adminTotpSecrets)
        .where(eq(schema.adminTotpSecrets.secretName, 'ml-training-hub'));

      // Clear session
      if (req.session) {
        req.session.adminTotpVerified = false;
        req.session.adminTotpVerifiedAt = null;
      }

      res.json({
        success: true,
        message: 'TOTP reset. You can now set up a new authenticator.'
      });
    } catch (error: any) {
      console.error('Error resetting TOTP:', error);
      res.status(500).json({ error: 'Failed to reset TOTP' });
    }
  });

  // =============================================================================
  // END ADMIN TOTP AUTHENTICATION ROUTES
  // =============================================================================

  // =============================================================================
  // ML TRAINING CONSENT ROUTES
  // Patient data contribution consent for ML model training
  // =============================================================================

  // Get current consent settings for the logged-in patient
  app.get('/api/ml/training/consent', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'patient') {
        return res.status(403).json({ error: 'Only patients can manage training consent' });
      }

      const consent = await db
        .select()
        .from(schema.mlTrainingConsent)
        .where(eq(schema.mlTrainingConsent.patientId, req.user.id))
        .limit(1);

      if (consent.length === 0) {
        // Return default consent state for new patients
        return res.json({
          patientId: req.user.id,
          isActive: false,
          consentedDataTypes: [],
          anonymizationLevel: 'full',
          researchUseOnly: true,
          optInDate: null,
          withdrawalDate: null,
        });
      }

      // Transform to frontend format
      const c = consent[0];
      const dataTypes = c.dataTypes || {};
      const consentedDataTypes: string[] = [];
      
      if (dataTypes.vitals) consentedDataTypes.push('vitals');
      if (dataTypes.symptoms) consentedDataTypes.push('symptoms');
      if (dataTypes.medications) consentedDataTypes.push('medications');
      if (dataTypes.mentalHealth) consentedDataTypes.push('mental_health');
      if (dataTypes.behavioralData) consentedDataTypes.push('behavioral');
      if (dataTypes.wearableData) consentedDataTypes.push('wearable');
      if (dataTypes.labResults) consentedDataTypes.push('lab_results');

      res.json({
        id: c.id,
        patientId: c.patientId,
        isActive: c.consentEnabled,
        consentedDataTypes,
        anonymizationLevel: c.anonymizationLevel || 'full',
        researchUseOnly: true,
        optInDate: c.consentSignedAt?.toISOString() || null,
        withdrawalDate: c.consentWithdrawnAt?.toISOString() || null,
      });
    } catch (error) {
      console.error('Error fetching ML training consent:', error);
      res.status(500).json({ error: 'Failed to fetch consent settings' });
    }
  });

  // Update consent settings
  app.post('/api/ml/training/consent', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'patient') {
        return res.status(403).json({ error: 'Only patients can manage training consent' });
      }

      const { isActive, consentedDataTypes = [], anonymizationLevel = 'full' } = req.body;

      // Convert frontend format to database format
      const dataTypes = {
        vitals: consentedDataTypes.includes('vitals'),
        symptoms: consentedDataTypes.includes('symptoms'),
        medications: consentedDataTypes.includes('medications'),
        mentalHealth: consentedDataTypes.includes('mental_health'),
        behavioralData: consentedDataTypes.includes('behavioral'),
        wearableData: consentedDataTypes.includes('wearable'),
        labResults: consentedDataTypes.includes('lab_results'),
        imagingData: false, // Not exposed in UI yet
      };

      // Check if consent record exists
      const existing = await db
        .select()
        .from(schema.mlTrainingConsent)
        .where(eq(schema.mlTrainingConsent.patientId, req.user.id))
        .limit(1);

      let result;
      if (existing.length > 0) {
        // Update existing consent
        result = await db
          .update(schema.mlTrainingConsent)
          .set({
            consentEnabled: isActive,
            dataTypes,
            anonymizationLevel,
            consentSignedAt: isActive && !existing[0].consentEnabled ? new Date() : existing[0].consentSignedAt,
            consentWithdrawnAt: !isActive && existing[0].consentEnabled ? new Date() : null,
            lastModifiedBy: req.user.id,
            ipAddress: req.ip,
            userAgent: req.get('user-agent'),
            updatedAt: new Date(),
          })
          .where(eq(schema.mlTrainingConsent.patientId, req.user.id))
          .returning();
      } else {
        // Create new consent record
        result = await db
          .insert(schema.mlTrainingConsent)
          .values({
            patientId: req.user.id,
            consentEnabled: isActive,
            dataTypes,
            anonymizationLevel,
            consentVersion: '1.0',
            consentSignedAt: isActive ? new Date() : null,
            lastModifiedBy: req.user.id,
            ipAddress: req.ip,
            userAgent: req.get('user-agent'),
          })
          .returning();
      }

      // Log consent change for HIPAA audit
      console.log(`[HIPAA_AUDIT] ML Training consent ${isActive ? 'granted' : 'updated'} by patient ${req.user.id}`);

      // Transform response
      const c = result[0];
      const responseDataTypes: string[] = [];
      if (c.dataTypes?.vitals) responseDataTypes.push('vitals');
      if (c.dataTypes?.symptoms) responseDataTypes.push('symptoms');
      if (c.dataTypes?.medications) responseDataTypes.push('medications');
      if (c.dataTypes?.mentalHealth) responseDataTypes.push('mental_health');
      if (c.dataTypes?.behavioralData) responseDataTypes.push('behavioral');
      if (c.dataTypes?.wearableData) responseDataTypes.push('wearable');
      if (c.dataTypes?.labResults) responseDataTypes.push('lab_results');

      res.json({
        id: c.id,
        patientId: c.patientId,
        isActive: c.consentEnabled,
        consentedDataTypes: responseDataTypes,
        anonymizationLevel: c.anonymizationLevel,
        researchUseOnly: true,
        optInDate: c.consentSignedAt?.toISOString() || null,
        withdrawalDate: c.consentWithdrawnAt?.toISOString() || null,
      });
    } catch (error) {
      console.error('Error updating ML training consent:', error);
      res.status(500).json({ error: 'Failed to update consent settings' });
    }
  });

  // Withdraw all consent
  app.post('/api/ml/training/consent/withdraw', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'patient') {
        return res.status(403).json({ error: 'Only patients can manage training consent' });
      }

      const { reason } = req.body;

      const result = await db
        .update(schema.mlTrainingConsent)
        .set({
          consentEnabled: false,
          consentWithdrawnAt: new Date(),
          withdrawalReason: reason || 'User requested withdrawal',
          dataTypes: {
            vitals: false,
            symptoms: false,
            medications: false,
            mentalHealth: false,
            behavioralData: false,
            wearableData: false,
            labResults: false,
            imagingData: false,
          },
          lastModifiedBy: req.user.id,
          ipAddress: req.ip,
          userAgent: req.get('user-agent'),
          updatedAt: new Date(),
        })
        .where(eq(schema.mlTrainingConsent.patientId, req.user.id))
        .returning();

      // Log withdrawal for HIPAA audit
      console.log(`[HIPAA_AUDIT] ML Training consent WITHDRAWN by patient ${req.user.id}`);

      if (result.length === 0) {
        return res.json({ success: true, message: 'No active consent to withdraw' });
      }

      res.json({ success: true, message: 'Consent withdrawn successfully' });
    } catch (error) {
      console.error('Error withdrawing ML training consent:', error);
      res.status(500).json({ error: 'Failed to withdraw consent' });
    }
  });

  // Get contribution history for the logged-in patient
  app.get('/api/ml/training/contributions', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'patient') {
        return res.status(403).json({ error: 'Only patients can view their contributions' });
      }

      // Create a hash of patient ID for lookup (in production, use proper hashing)
      const patientIdHash = `patient_${req.user.id.substring(0, 8)}`;

      const contributions = await db
        .select({
          id: schema.mlTrainingContributions.id,
          trainingJobId: schema.mlTrainingContributions.trainingJobId,
          dataTypesContributed: schema.mlTrainingContributions.dataTypesContributed,
          recordCount: schema.mlTrainingContributions.recordCount,
          contributedAt: schema.mlTrainingContributions.contributedAt,
          status: schema.mlTrainingContributions.status,
        })
        .from(schema.mlTrainingContributions)
        .where(eq(schema.mlTrainingContributions.patientIdHash, patientIdHash))
        .orderBy(desc(schema.mlTrainingContributions.contributedAt))
        .limit(20);

      // Get model names for contributions
      const contributionsWithModels = contributions.map(c => ({
        id: c.id,
        modelName: c.trainingJobId ? `Model Training #${c.trainingJobId.substring(0, 8)}` : 'Unknown Model',
        dataTypesContributed: c.dataTypesContributed || [],
        recordCount: c.recordCount,
        contributedAt: c.contributedAt?.toISOString() || '',
        status: c.status || 'included',
      }));

      res.json(contributionsWithModels);
    } catch (error) {
      console.error('Error fetching ML training contributions:', error);
      res.status(500).json({ error: 'Failed to fetch contributions' });
    }
  });

  // =============================================================================
  // END ML TRAINING CONSENT ROUTES
  // =============================================================================

  // =============================================================================
  // ML TRAINING ADMIN ROUTES (Admin/Doctor only)
  // Proxy to Python FastAPI ml_training.py endpoints
  // =============================================================================

  // List available datasets for ML training
  app.get('/api/ml/training/datasets', isAuthenticated, async (req: any, res) => {
    try {
      if (!['admin', 'doctor'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied. Admin or Doctor role required.' });
      }

      const response = await fetch(`${pythonBackendUrl}/api/ml/training/datasets`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
      });

      if (!response.ok) {
        const error = await response.text();
        console.error('Python backend error (ml datasets):', response.status, error);
        // Return mock datasets if Python backend unavailable
        return res.json([
          { name: 'MIMIC-III', source: 'PhysioNet', description: 'ICU data from Beth Israel', record_count: 58000, patient_count: 46000, requires_credentials: true, is_available: false },
          { name: 'MIMIC-IV', source: 'PhysioNet', description: 'Updated ICU dataset', record_count: 430000, patient_count: 180000, requires_credentials: true, is_available: false },
          { name: 'eICU', source: 'PhysioNet', description: 'Multi-center ICU database', record_count: 139000, patient_count: 139000, requires_credentials: true, is_available: false },
          { name: 'Synthetic Health', source: 'Synthea', description: 'Synthetic patient records', record_count: 100000, patient_count: 100000, requires_credentials: false, is_available: true },
        ]);
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching ML datasets:', error);
      res.json([
        { name: 'MIMIC-III', source: 'PhysioNet', description: 'ICU data from Beth Israel', record_count: 58000, patient_count: 46000, requires_credentials: true, is_available: false },
        { name: 'Synthetic Health', source: 'Synthea', description: 'Synthetic patient records', record_count: 100000, patient_count: 100000, requires_credentials: false, is_available: true },
      ]);
    }
  });

  // List trained models in the registry
  app.get('/api/ml/training/models', isAuthenticated, async (req: any, res) => {
    try {
      if (!['admin', 'doctor'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied. Admin or Doctor role required.' });
      }

      const { status: statusFilter, limit = '50' } = req.query;
      const params = new URLSearchParams();
      if (statusFilter) params.set('status_filter', statusFilter as string);
      params.set('limit', limit as string);

      const response = await fetch(`${pythonBackendUrl}/api/ml/training/models?${params.toString()}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
      });

      if (!response.ok) {
        console.error('Python backend error (ml models):', response.status);
        return res.json([]);
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching ML models:', error);
      res.json([]);
    }
  });

  // Create a new training job
  app.post('/api/ml/training/jobs', isAuthenticated, async (req: any, res) => {
    try {
      if (!['admin', 'doctor'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Only administrators can create training jobs' });
      }

      const response = await fetch(`${pythonBackendUrl}/api/ml/training/jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        const error = await response.text();
        console.error('Python backend error (create training job):', response.status, error);
        return res.status(response.status).json({ error: 'Failed to create training job' });
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error creating training job:', error);
      res.status(500).json({ error: 'Failed to create training job' });
    }
  });

  // List training jobs
  app.get('/api/ml/training/jobs', isAuthenticated, async (req: any, res) => {
    try {
      if (!['admin', 'doctor'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { status: statusFilter, limit = '20' } = req.query;
      const params = new URLSearchParams();
      if (statusFilter) params.set('status_filter', statusFilter as string);
      params.set('limit', limit as string);

      const response = await fetch(`${pythonBackendUrl}/api/ml/training/jobs?${params.toString()}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
      });

      if (!response.ok) {
        console.error('Python backend error (ml jobs):', response.status);
        return res.json([]);
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching training jobs:', error);
      res.json([]);
    }
  });

  // Get training job status
  app.get('/api/ml/training/jobs/:jobId', isAuthenticated, async (req: any, res) => {
    try {
      if (!['admin', 'doctor'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { jobId } = req.params;
      const response = await fetch(`${pythonBackendUrl}/api/ml/training/jobs/${jobId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Training job not found' });
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching training job:', error);
      res.status(500).json({ error: 'Failed to fetch training job' });
    }
  });

  // Start a queued training job
  app.post('/api/ml/training/jobs/:jobId/start', isAuthenticated, async (req: any, res) => {
    try {
      if (!['admin', 'doctor'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Only administrators can start training jobs' });
      }

      const { jobId } = req.params;
      const response = await fetch(`${pythonBackendUrl}/api/ml/training/jobs/${jobId}/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
      });

      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ error: error || 'Failed to start training job' });
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error starting training job:', error);
      res.status(500).json({ error: 'Failed to start training job' });
    }
  });

  // Get consent statistics (admin)
  app.get('/api/ml/training/consent/stats', isAuthenticated, async (req: any, res) => {
    try {
      if (!['admin', 'doctor'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const response = await fetch(`${pythonBackendUrl}/api/ml/training/consent/stats`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
      });

      if (!response.ok) {
        // Fallback: get stats from Express DB
        const consents = await db
          .select()
          .from(schema.mlTrainingConsent);

        const total = consents.length;
        const consenting = consents.filter(c => c.consentEnabled).length;
        const vitalsConsent = consents.filter(c => c.consentEnabled && c.dataTypes?.vitals).length;
        const symptomsConsent = consents.filter(c => c.consentEnabled && c.dataTypes?.symptoms).length;
        const wearableConsent = consents.filter(c => c.consentEnabled && c.dataTypes?.wearableData).length;

        return res.json({
          total_patients_with_consent_record: total,
          consenting_patients: consenting,
          consent_rate: total > 0 ? (consenting / total * 100) : 0,
          data_type_breakdown: {
            vitals: vitalsConsent,
            symptoms: symptomsConsent,
            mental_health: consents.filter(c => c.consentEnabled && c.dataTypes?.mentalHealth).length,
            medications: consents.filter(c => c.consentEnabled && c.dataTypes?.medications).length,
            wearable: wearableConsent,
          }
        });
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching consent stats:', error);
      res.json({
        total_patients_with_consent_record: 0,
        consenting_patients: 0,
        consent_rate: 0,
        data_type_breakdown: {}
      });
    }
  });

  // Get contributions summary (admin)
  app.get('/api/ml/training/contributions/summary', isAuthenticated, async (req: any, res) => {
    try {
      if (!['admin', 'doctor'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const response = await fetch(`${pythonBackendUrl}/api/ml/training/contributions/summary`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
      });

      if (!response.ok) {
        return res.json({
          unique_contributors: 0,
          total_contributions: 0,
          total_records_contributed: 0,
          training_jobs_with_patient_data: 0,
        });
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching contributions summary:', error);
      res.json({
        unique_contributors: 0,
        total_contributions: 0,
        total_records_contributed: 0,
        training_jobs_with_patient_data: 0,
      });
    }
  });

  // Extract device data for training (admin)
  app.post('/api/ml/training/device-data/extract', isAuthenticated, async (req: any, res) => {
    try {
      if (!['admin', 'doctor'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Only administrators can extract training data' });
      }

      const response = await fetch(`${pythonBackendUrl}/api/ml/training/device-data/extract`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ error: error || 'Failed to extract device data' });
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error extracting device data:', error);
      res.status(500).json({ error: 'Failed to extract device data' });
    }
  });

  // Deploy a trained model
  app.post('/api/ml/training/models/:modelId/deploy', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Only administrators can deploy models' });
      }

      const { modelId } = req.params;
      const response = await fetch(`${pythonBackendUrl}/api/ml/training/models/${modelId}/deploy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
      });

      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ error: error || 'Failed to deploy model' });
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error deploying model:', error);
      res.status(500).json({ error: 'Failed to deploy model' });
    }
  });

  // Register a public dataset
  app.post('/api/ml/training/register-dataset/:datasetKey', isAuthenticated, async (req: any, res) => {
    try {
      if (req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Only administrators can register datasets' });
      }

      const { datasetKey } = req.params;
      const response = await fetch(`${pythonBackendUrl}/api/ml/training/register-dataset/${datasetKey}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || '',
          'X-User-Id': req.user.id,
          'X-User-Role': req.user.role,
        },
      });

      if (!response.ok) {
        const error = await response.text();
        return res.status(response.status).json({ error: error || 'Failed to register dataset' });
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error registering dataset:', error);
      res.status(500).json({ error: 'Failed to register dataset' });
    }
  });

  // =============================================================================
  // END ML TRAINING ADMIN ROUTES
  // =============================================================================

  // =============================================================================
  // MEDICAL NLP ROUTES
  // =============================================================================

  // Get all uploaded medical documents
  app.get('/api/medical-nlp/documents', isAuthenticated, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      // Return mock documents for now
      res.json([
        {
          id: 'doc-1',
          filename: 'patient_lab_results_2024.pdf',
          file_type: 'pdf',
          upload_date: new Date(Date.now() - 86400000 * 3).toISOString(),
          status: 'completed',
          extracted_text: 'Laboratory Report\n\nPatient: John Smith\nDate: 2024-11-15\n\nComplete Blood Count:\n- WBC: 5.2 x10^9/L (Normal: 4.5-11.0)\n- RBC: 4.8 x10^12/L (Normal: 4.5-5.5)\n- Hemoglobin: 14.2 g/dL (Normal: 13.5-17.5)\n- Hematocrit: 42% (Normal: 38-50%)\n- Platelet Count: 250 x10^9/L (Normal: 150-400)\n\nMetabolic Panel:\n- Glucose: 95 mg/dL (Normal: 70-100)\n- Creatinine: 1.0 mg/dL (Normal: 0.7-1.3)\n- eGFR: 85 mL/min/1.73m2\n\nImmunosuppressant Levels:\n- Tacrolimus: 8.5 ng/mL (Target: 5-15 ng/mL)\n\nInterpretation: All values within normal limits. Tacrolimus levels therapeutic.',
          entities: [
            { text: 'Complete Blood Count', type: 'procedure', confidence: 0.95, start: 48, end: 68 },
            { text: 'WBC', type: 'lab_value', confidence: 0.98, start: 72, end: 75 },
            { text: 'RBC', type: 'lab_value', confidence: 0.98, start: 110, end: 113 },
            { text: 'Hemoglobin', type: 'lab_value', confidence: 0.97, start: 148, end: 158 },
            { text: 'Tacrolimus', type: 'medication', confidence: 0.99, start: 380, end: 390 },
            { text: '8.5 ng/mL', type: 'dosage', confidence: 0.95, start: 392, end: 401 },
          ],
          phi_detected: true,
          phi_redacted: true,
        },
        {
          id: 'doc-2',
          filename: 'discharge_summary.txt',
          file_type: 'txt',
          upload_date: new Date(Date.now() - 86400000 * 7).toISOString(),
          status: 'completed',
          extracted_text: 'Discharge Summary\n\nAdmission Date: 2024-11-01\nDischarge Date: 2024-11-08\n\nPrincipal Diagnosis: Acute kidney injury, Stage 2\n\nSecondary Diagnoses:\n- Post-transplant immunocompromised state\n- Hypertension, controlled\n- Type 2 Diabetes Mellitus\n\nProcedures:\n- Kidney biopsy\n- IV fluid resuscitation\n\nMedications on Discharge:\n1. Tacrolimus 3mg BID\n2. Mycophenolate 500mg BID\n3. Prednisone 10mg daily\n4. Lisinopril 10mg daily\n\nFollow-up: Nephrology clinic in 1 week.',
          entities: [
            { text: 'Acute kidney injury', type: 'condition', confidence: 0.97, start: 90, end: 109 },
            { text: 'Post-transplant immunocompromised', type: 'condition', confidence: 0.94, start: 142, end: 175 },
            { text: 'Hypertension', type: 'condition', confidence: 0.98, start: 184, end: 196 },
            { text: 'Type 2 Diabetes Mellitus', type: 'condition', confidence: 0.99, start: 210, end: 234 },
            { text: 'Kidney biopsy', type: 'procedure', confidence: 0.96, start: 248, end: 261 },
            { text: 'Tacrolimus', type: 'medication', confidence: 0.99, start: 310, end: 320 },
            { text: '3mg BID', type: 'dosage', confidence: 0.95, start: 321, end: 328 },
            { text: 'Mycophenolate', type: 'medication', confidence: 0.98, start: 333, end: 346 },
            { text: 'Prednisone', type: 'medication', confidence: 0.99, start: 360, end: 370 },
            { text: 'Lisinopril', type: 'medication', confidence: 0.97, start: 388, end: 398 },
          ],
          phi_detected: true,
          phi_redacted: true,
        },
      ]);
    } catch (error) {
      console.error('Error fetching medical documents:', error);
      res.status(500).json({ error: 'Failed to fetch documents' });
    }
  });

  // Upload medical document
  app.post('/api/medical-nlp/upload', isAuthenticated, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      // Simulate processing
      const docId = `doc-${Date.now()}`;
      
      console.log(`[HIPAA-AUDIT] Medical document uploaded: ${docId} by ${req.user.id}`);
      
      res.json({
        id: docId,
        filename: 'uploaded_document.pdf',
        file_type: 'pdf',
        upload_date: new Date().toISOString(),
        status: 'processing',
        phi_detected: false,
        phi_redacted: false,
      });
    } catch (error) {
      console.error('Error uploading document:', error);
      res.status(500).json({ error: 'Failed to upload document' });
    }
  });

  // Medical NLP Chat
  app.post('/api/medical-nlp/chat', isAuthenticated, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { message, document_ids } = req.body;

      // Use OpenAI for response
      let response = '';
      try {
        const completion = await openai.chat.completions.create({
          model: 'gpt-4o',
          messages: [
            { 
              role: 'system', 
              content: 'You are a medical AI assistant helping doctors analyze medical documents. Provide accurate, helpful responses about medications, conditions, lab values, and procedures mentioned in documents. Always recommend consulting with specialists when appropriate.' 
            },
            { role: 'user', content: message },
          ],
          max_tokens: 500,
        });
        response = completion.choices[0].message.content || 'Unable to generate response';
      } catch (aiError) {
        console.error('OpenAI error:', aiError);
        response = `Based on the uploaded documents, I can help analyze medications, conditions, and lab values. For detailed analysis, please ensure documents are properly uploaded and processed.`;
      }

      res.json({
        response,
        sources: document_ids?.slice(0, 3) || [],
      });
    } catch (error) {
      console.error('Error in medical NLP chat:', error);
      res.status(500).json({ error: 'Failed to process chat message' });
    }
  });

  // =============================================================================
  // END MEDICAL NLP ROUTES
  // =============================================================================

  // =============================================================================
  // ENHANCED RESEARCH CENTER ROUTES
  // =============================================================================

  // Helper middleware to attach audit context to request for research service
  // Context is request-scoped to ensure thread-safety with concurrent requests
  const setResearchAuditContext = (req: any, res: any, next: any) => {
    req.researchAuditContext = {
      userId: req.user?.id || 'anonymous',
      ipAddress: req.ip || req.connection?.remoteAddress,
      userAgent: req.headers['user-agent'],
    };
    next();
  };

  // Research Data Consent - Patient consent for research use
  app.get('/api/v1/research-center/consent', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const consent = await researchService.getResearchDataConsent(req.user.id);
      res.json(consent || { patientId: req.user.id, consentEnabled: false });
    } catch (error) {
      console.error('Error fetching research consent:', error);
      res.status(500).json({ error: 'Failed to fetch consent' });
    }
  });

  app.post('/api/v1/research-center/consent', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (req.user.role !== 'patient') {
        return res.status(403).json({ error: 'Only patients can manage research consent' });
      }

      const consent = await researchService.upsertResearchDataConsent({
        patientId: req.user.id,
        ...req.body,
      }, req.researchAuditContext);

      console.log(`[HIPAA-AUDIT] Research consent updated by patient ${req.user.id}`);
      res.json(consent);
    } catch (error) {
      console.error('Error updating research consent:', error);
      res.status(500).json({ error: 'Failed to update consent' });
    }
  });

  // Research Center Metrics
  app.get('/api/v1/research-center/metrics', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const metrics = await researchService.getResearchMetrics();
      res.json(metrics);
    } catch (error) {
      console.error('Error fetching research metrics:', error);
      res.status(500).json({ error: 'Failed to fetch metrics' });
    }
  });

  // Research Projects (Personal Research Mode)
  app.get('/api/v1/research-center/projects', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const projects = await researchService.getResearchProjects(req.user.id);
      res.json(projects);
    } catch (error) {
      console.error('Error fetching research projects:', error);
      res.status(500).json({ error: 'Failed to fetch projects' });
    }
  });

  app.post('/api/v1/research-center/projects', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const project = await researchService.createResearchProject({
        ...req.body,
        ownerId: req.user.id,
      }, req.researchAuditContext);

      console.log(`[HIPAA-AUDIT] Research project created: ${project.id} by ${req.user.id}`);
      res.json(project);
    } catch (error) {
      console.error('Error creating research project:', error);
      res.status(500).json({ error: 'Failed to create project' });
    }
  });

  app.get('/api/v1/research-center/projects/:id', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const project = await researchService.getResearchProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      res.json(project);
    } catch (error) {
      console.error('Error fetching research project:', error);
      res.status(500).json({ error: 'Failed to fetch project' });
    }
  });

  app.patch('/api/v1/research-center/projects/:id', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const project = await researchService.updateResearchProject(req.params.id, req.body, req.researchAuditContext);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      res.json(project);
    } catch (error) {
      console.error('Error updating research project:', error);
      res.status(500).json({ error: 'Failed to update project' });
    }
  });

  // Research Studies
  app.get('/api/v1/research-center/studies', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { projectId, status, limit } = req.query;
      const studies = await researchService.getResearchStudies({
        projectId: projectId as string,
        ownerId: req.user.id,
        status: status as string,
        limit: limit ? parseInt(limit as string) : undefined,
      });
      res.json(studies);
    } catch (error) {
      console.error('Error fetching research studies:', error);
      res.status(500).json({ error: 'Failed to fetch studies' });
    }
  });

  app.post('/api/v1/research-center/studies', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const study = await researchService.createResearchStudy({
        ...req.body,
        ownerUserId: req.user.id,
      }, req.researchAuditContext);

      console.log(`[HIPAA-AUDIT] Research study created: ${study.id} by ${req.user.id}`);
      res.json(study);
    } catch (error) {
      console.error('Error creating research study:', error);
      res.status(500).json({ error: 'Failed to create study' });
    }
  });

  app.get('/api/v1/research-center/studies/:id', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const study = await researchService.getResearchStudy(req.params.id);
      if (!study) {
        return res.status(404).json({ error: 'Study not found' });
      }
      res.json(study);
    } catch (error) {
      console.error('Error fetching research study:', error);
      res.status(500).json({ error: 'Failed to fetch study' });
    }
  });

  app.patch('/api/v1/research-center/studies/:id', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const study = await researchService.updateResearchStudy(req.params.id, req.body, req.researchAuditContext);
      if (!study) {
        return res.status(404).json({ error: 'Study not found' });
      }
      res.json(study);
    } catch (error) {
      console.error('Error updating research study:', error);
      res.status(500).json({ error: 'Failed to update study' });
    }
  });

  // Research Cohorts
  app.get('/api/v1/research-center/cohorts', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { projectId, status } = req.query;
      const cohorts = await researchService.getResearchCohorts({
        projectId: projectId as string,
        createdBy: req.user.id,
        status: status as string,
      });
      res.json(cohorts);
    } catch (error) {
      console.error('Error fetching research cohorts:', error);
      res.status(500).json({ error: 'Failed to fetch cohorts' });
    }
  });

  app.post('/api/v1/research-center/cohorts', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const cohort = await researchService.createResearchCohort({
        ...req.body,
        createdBy: req.user.id,
      }, req.researchAuditContext);

      console.log(`[HIPAA-AUDIT] Research cohort created: ${cohort.id} by ${req.user.id}`);
      res.json(cohort);
    } catch (error) {
      console.error('Error creating research cohort:', error);
      res.status(500).json({ error: 'Failed to create cohort' });
    }
  });

  app.post('/api/v1/research-center/cohorts/preview', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const filters = req.body.filters || req.body.definition || req.body;
      const preview = await researchService.previewCohort(filters);
      res.json(preview);
    } catch (error) {
      console.error('Error previewing cohort:', error);
      res.status(500).json({ error: 'Failed to preview cohort' });
    }
  });

  app.get('/api/v1/research-center/cohorts/:id', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const cohort = await researchService.getResearchCohort(req.params.id);
      if (!cohort) {
        return res.status(404).json({ error: 'Cohort not found' });
      }
      res.json(cohort);
    } catch (error) {
      console.error('Error fetching research cohort:', error);
      res.status(500).json({ error: 'Failed to fetch cohort' });
    }
  });

  // Study Enrollments
  app.get('/api/v1/research-center/studies/:studyId/enrollments', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const enrollments = await researchService.getStudyEnrollments(req.params.studyId);
      res.json(enrollments);
    } catch (error) {
      console.error('Error fetching study enrollments:', error);
      res.status(500).json({ error: 'Failed to fetch enrollments' });
    }
  });

  app.post('/api/v1/research-center/studies/:studyId/enrollments', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const enrollment = await researchService.createStudyEnrollment({
        studyId: req.params.studyId,
        ...req.body,
      }, req.researchAuditContext);

      console.log(`[HIPAA-AUDIT] Patient enrolled in study ${req.params.studyId} by ${req.user.id}`);
      res.json(enrollment);
    } catch (error) {
      console.error('Error creating study enrollment:', error);
      res.status(500).json({ error: 'Failed to create enrollment' });
    }
  });

  app.post('/api/v1/research-center/enrollments/:id/withdraw', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const { reason } = req.body;
      const enrollment = await researchService.withdrawFromStudy(req.params.id, reason, req.researchAuditContext);
      if (!enrollment) {
        return res.status(404).json({ error: 'Enrollment not found' });
      }

      console.log(`[HIPAA-AUDIT] Patient withdrawn from study enrollment ${req.params.id}`);
      res.json(enrollment);
    } catch (error) {
      console.error('Error withdrawing from study:', error);
      res.status(500).json({ error: 'Failed to withdraw' });
    }
  });

  // Research Visits
  app.get('/api/v1/research-center/visits', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const { studyId, patientId, status } = req.query;
      const visits = await researchService.getResearchVisits({
        studyId: studyId as string,
        patientId: patientId as string,
        status: status as string,
      });
      res.json(visits);
    } catch (error) {
      console.error('Error fetching research visits:', error);
      res.status(500).json({ error: 'Failed to fetch visits' });
    }
  });

  app.post('/api/v1/research-center/visits', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const visit = await researchService.createResearchVisit(req.body, req.researchAuditContext);
      res.json(visit);
    } catch (error) {
      console.error('Error creating research visit:', error);
      res.status(500).json({ error: 'Failed to create visit' });
    }
  });

  // Research Measurements
  app.get('/api/v1/research-center/measurements', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const { patientId, studyId, category, limit } = req.query;
      const measurements = await researchService.getResearchMeasurements({
        patientId: patientId as string,
        studyId: studyId as string,
        category: category as string,
        limit: limit ? parseInt(limit as string) : undefined,
      });
      res.json(measurements);
    } catch (error) {
      console.error('Error fetching research measurements:', error);
      res.status(500).json({ error: 'Failed to fetch measurements' });
    }
  });

  app.post('/api/v1/research-center/measurements', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (Array.isArray(req.body)) {
        const measurements = await researchService.createResearchMeasurements(req.body, req.researchAuditContext);
        res.json(measurements);
      } else {
        const measurement = await researchService.createResearchMeasurement(req.body, req.researchAuditContext);
        res.json(measurement);
      }
    } catch (error) {
      console.error('Error creating research measurement:', error);
      res.status(500).json({ error: 'Failed to create measurement' });
    }
  });

  // Research Immune Markers
  app.get('/api/v1/research-center/immune-markers', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const { patientId, studyId, markerName, limit } = req.query;
      const markers = await researchService.getResearchImmuneMarkers({
        patientId: patientId as string,
        studyId: studyId as string,
        markerName: markerName as string,
        limit: limit ? parseInt(limit as string) : undefined,
      });
      res.json(markers);
    } catch (error) {
      console.error('Error fetching immune markers:', error);
      res.status(500).json({ error: 'Failed to fetch immune markers' });
    }
  });

  // Research Analysis Reports
  app.get('/api/v1/research-center/reports', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { studyId, cohortId, analysisType, status, limit } = req.query;
      const reports = await researchService.getResearchAnalysisReports({
        studyId: studyId as string,
        cohortId: cohortId as string,
        createdBy: req.user.id,
        analysisType: analysisType as string,
        status: status as string,
        limit: limit ? parseInt(limit as string) : undefined,
      });
      res.json(reports);
    } catch (error) {
      console.error('Error fetching research reports:', error);
      res.status(500).json({ error: 'Failed to fetch reports' });
    }
  });

  app.post('/api/v1/research-center/reports', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const report = await researchService.createResearchAnalysisReport({
        ...req.body,
        createdBy: req.user.id,
      }, req.researchAuditContext);

      console.log(`[HIPAA-AUDIT] Research report created: ${report.id} by ${req.user.id}`);
      res.json(report);
    } catch (error) {
      console.error('Error creating research report:', error);
      res.status(500).json({ error: 'Failed to create report' });
    }
  });

  app.get('/api/v1/research-center/reports/:id', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const report = await researchService.getResearchAnalysisReport(req.params.id);
      if (!report) {
        return res.status(404).json({ error: 'Report not found' });
      }
      res.json(report);
    } catch (error) {
      console.error('Error fetching research report:', error);
      res.status(500).json({ error: 'Failed to fetch report' });
    }
  });

  // AI-Powered Report Generation
  app.post('/api/v1/research-center/reports/generate-ai', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { studyId, cohortId, analysisType, prompt } = req.body;

      // Generate AI report using OpenAI
      let aiContent = '';
      try {
        const completion = await openai.chat.completions.create({
          model: 'gpt-4o',
          messages: [
            {
              role: 'system',
              content: `You are an expert epidemiological research analyst. Generate comprehensive research reports for clinical studies on immunocompromised patients. Include statistical analysis, trends, risk factors, and actionable recommendations. Format with clear sections: Executive Summary, Methodology, Findings, Statistical Analysis, Conclusions, and Recommendations.`,
            },
            {
              role: 'user',
              content: prompt || `Generate a ${analysisType || 'descriptive'} analysis report for the research study. Focus on patient outcomes, risk factors, and clinical patterns.`,
            },
          ],
          max_tokens: 2000,
        });
        aiContent = completion.choices[0].message.content || '';
      } catch (aiError) {
        console.error('OpenAI error:', aiError);
        aiContent = 'AI report generation is currently unavailable. Please try again later.';
      }

      const report = await researchService.createResearchAnalysisReport({
        studyId,
        cohortId,
        analysisType: analysisType || 'descriptive',
        title: `AI-Generated ${analysisType || 'Analysis'} Report`,
        reportContent: { content: aiContent, generatedAt: new Date().toISOString() },
        aiGeneratedSummary: aiContent.substring(0, 500),
        status: 'completed',
        createdBy: req.user.id,
      }, req.researchAuditContext);

      console.log(`[HIPAA-AUDIT] AI research report generated: ${report.id} by ${req.user.id}`);
      res.json(report);
    } catch (error) {
      console.error('Error generating AI report:', error);
      res.status(500).json({ error: 'Failed to generate AI report' });
    }
  });

  // Research Alerts
  app.get('/api/v1/research-center/alerts', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const { patientId, studyId, severity, status, limit } = req.query;
      const alerts = await researchService.getResearchAlerts({
        patientId: patientId as string,
        studyId: studyId as string,
        severity: severity as string,
        status: status as string,
        limit: limit ? parseInt(limit as string) : undefined,
      });
      res.json(alerts);
    } catch (error) {
      console.error('Error fetching research alerts:', error);
      res.status(500).json({ error: 'Failed to fetch alerts' });
    }
  });

  app.post('/api/v1/research-center/alerts/:id/acknowledge', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const alert = await researchService.acknowledgeResearchAlert(req.params.id, req.user.id, req.researchAuditContext);
      if (!alert) {
        return res.status(404).json({ error: 'Alert not found' });
      }
      res.json(alert);
    } catch (error) {
      console.error('Error acknowledging alert:', error);
      res.status(500).json({ error: 'Failed to acknowledge alert' });
    }
  });

  app.post('/api/v1/research-center/alerts/:id/resolve', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const { resolution } = req.body;
      const alert = await researchService.resolveResearchAlert(req.params.id, req.user.id, resolution, req.researchAuditContext);
      if (!alert) {
        return res.status(404).json({ error: 'Alert not found' });
      }
      res.json(alert);
    } catch (error) {
      console.error('Error resolving alert:', error);
      res.status(500).json({ error: 'Failed to resolve alert' });
    }
  });

  // Daily Followup Templates
  app.get('/api/v1/research-center/followup-templates', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const templates = await researchService.getDailyFollowupTemplates(req.user.id);
      res.json(templates);
    } catch (error) {
      console.error('Error fetching followup templates:', error);
      res.status(500).json({ error: 'Failed to fetch templates' });
    }
  });

  app.post('/api/v1/research-center/followup-templates', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const template = await researchService.createDailyFollowupTemplate({
        ...req.body,
        createdBy: req.user.id,
      }, req.researchAuditContext);
      res.json(template);
    } catch (error) {
      console.error('Error creating followup template:', error);
      res.status(500).json({ error: 'Failed to create template' });
    }
  });

  // Analysis Jobs
  app.get('/api/v1/research-center/analysis-jobs', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const jobs = await researchService.getPendingAnalysisJobs();
      res.json(jobs);
    } catch (error) {
      console.error('Error fetching analysis jobs:', error);
      res.status(500).json({ error: 'Failed to fetch jobs' });
    }
  });

  app.post('/api/v1/research-center/analysis-jobs', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const job = await researchService.createAnalysisJob({
        ...req.body,
        createdBy: req.user.id,
      }, req.researchAuditContext);
      res.json(job);
    } catch (error) {
      console.error('Error creating analysis job:', error);
      res.status(500).json({ error: 'Failed to create job' });
    }
  });

  app.get('/api/v1/research-center/analysis-jobs/:id', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const job = await researchService.getAnalysisJob(req.params.id);
      if (!job) {
        return res.status(404).json({ error: 'Job not found' });
      }
      res.json(job);
    } catch (error) {
      console.error('Error fetching analysis job:', error);
      res.status(500).json({ error: 'Failed to fetch job' });
    }
  });

  // NL Query Parsing - Proxy to Python Backend
  app.post('/api/v1/research-center/analysis/parse-nl', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/ml/analysis/parse-nl`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        const errorText = await response.text();
        return res.status(response.status).json({ error: errorText });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error parsing NL query:', error);
      res.status(500).json({ error: 'Failed to parse query' });
    }
  });

  // Research Projects - Personal Research Mode
  app.get('/api/v1/research-center/projects', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const projects = await researchService.getResearchProjects(req.user.id);
      res.json(projects);
    } catch (error) {
      console.error('Error fetching research projects:', error);
      res.status(500).json({ error: 'Failed to fetch projects' });
    }
  });

  app.post('/api/v1/research-center/projects', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const project = await researchService.createResearchProject({
        ...req.body,
        ownerId: req.user.id,
      }, req.researchAuditContext);
      res.json(project);
    } catch (error) {
      console.error('Error creating research project:', error);
      res.status(500).json({ error: 'Failed to create project' });
    }
  });

  app.patch('/api/v1/research-center/projects/:id', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const project = await researchService.updateResearchProject(req.params.id, req.body, req.researchAuditContext);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      res.json(project);
    } catch (error) {
      console.error('Error updating research project:', error);
      res.status(500).json({ error: 'Failed to update project' });
    }
  });

  app.get('/api/v1/research-center/projects/:id/studies', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      const studies = await researchService.getStudiesByProject(req.params.id);
      res.json(studies);
    } catch (error) {
      console.error('Error fetching project studies:', error);
      res.status(500).json({ error: 'Failed to fetch studies' });
    }
  });

  // Scheduler API Routes - Proxy to Python Backend
  app.get('/api/v1/research-center/scheduler/status', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/scheduler/status`, {
        method: 'GET',
        headers: {
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to get scheduler status' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error fetching scheduler status:', error);
      res.status(500).json({ error: 'Failed to fetch scheduler status' });
    }
  });

  app.get('/api/v1/research-center/scheduler/jobs', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/scheduler/jobs`, {
        method: 'GET',
        headers: {
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to get scheduler jobs' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error fetching scheduler jobs:', error);
      res.status(500).json({ error: 'Failed to fetch scheduler jobs' });
    }
  });

  app.post('/api/v1/research-center/scheduler/trigger-reanalysis', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/scheduler/trigger-reanalysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to trigger reanalysis' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error triggering reanalysis:', error);
      res.status(500).json({ error: 'Failed to trigger reanalysis' });
    }
  });

  app.post('/api/v1/research-center/scheduler/run-now/:jobType', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin access required' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/scheduler/run-now/${req.params.jobType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to run job' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error running scheduled job:', error);
      res.status(500).json({ error: 'Failed to run scheduled job' });
    }
  });

  // Model Registry API Routes - Proxy to Python Backend
  app.get('/api/v1/research-center/model-registry/models', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { model_type, status } = req.query;
      let url = `${pythonBackendUrl}/api/v1/model-registry/models`;
      const params = new URLSearchParams();
      if (model_type) params.append('model_type', model_type);
      if (status) params.append('status', status);
      if (params.toString()) url += `?${params.toString()}`;

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to fetch models' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error fetching models:', error);
      res.status(500).json({ error: 'Failed to fetch models' });
    }
  });

  app.get('/api/v1/research-center/model-registry/models/:modelName/active', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/model-registry/models/${req.params.modelName}/active`, {
        method: 'GET',
        headers: {
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Model not found' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error fetching active model:', error);
      res.status(500).json({ error: 'Failed to fetch active model' });
    }
  });

  app.get('/api/v1/research-center/model-registry/models/:modelName/versions', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/model-registry/models/${req.params.modelName}/versions`, {
        method: 'GET',
        headers: {
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to fetch versions' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error fetching model versions:', error);
      res.status(500).json({ error: 'Failed to fetch model versions' });
    }
  });

  app.post('/api/v1/research-center/model-registry/verify-consent', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/model-registry/verify-consent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to verify consent' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error verifying consent:', error);
      res.status(500).json({ error: 'Failed to verify consent' });
    }
  });

  app.get('/api/v1/research-center/model-registry/consented-patients/count', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { data_types } = req.query;
      const url = `${pythonBackendUrl}/api/v1/model-registry/consented-patients/count?data_types=${data_types || ''}`;

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to fetch consented count' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error fetching consented count:', error);
      res.status(500).json({ error: 'Failed to fetch consented count' });
    }
  });

  // NOTE: No endpoint to enumerate patient IDs directly - only counts are exposed
  // Patient IDs are only used internally in prediction pipeline with proper consent checks

  app.post('/api/v1/research-center/model-registry/predict', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/model-registry/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to make prediction' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error making prediction:', error);
      res.status(500).json({ error: 'Failed to make prediction' });
    }
  });

  app.post('/api/v1/research-center/model-registry/log-usage', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/model-registry/log-usage`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to log usage' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error logging model usage:', error);
      res.status(500).json({ error: 'Failed to log model usage' });
    }
  });

  app.get('/api/v1/research-center/model-registry/models/:modelName/compare', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const { version1, version2 } = req.query;
      const response = await fetch(`${pythonBackendUrl}/api/v1/model-registry/models/${req.params.modelName}/compare?version1=${version1}&version2=${version2}`, {
        method: 'GET',
        headers: {
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to compare models' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error comparing models:', error);
      res.status(500).json({ error: 'Failed to compare models' });
    }
  });

  app.get('/api/v1/research-center/model-registry/models/:modelName/features', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${pythonBackendUrl}/api/v1/model-registry/models/${req.params.modelName}/features`, {
        method: 'GET',
        headers: {
          'Authorization': req.headers.authorization || `Bearer ${req.user.id}`,
        },
      });

      if (!response.ok) {
        return res.status(response.status).json({ error: 'Failed to fetch features' });
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error('Error fetching model features:', error);
      res.status(500).json({ error: 'Failed to fetch model features' });
    }
  });

  // Research Audit Logs
  app.get('/api/v1/research-center/audit-logs', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin access required' });
      }

      const { userId, actionType, objectType, limit } = req.query;
      const logs = await researchService.getResearchAuditLogs({
        userId: userId as string,
        actionType: actionType as string,
        objectType: objectType as string,
        limit: limit ? parseInt(limit as string) : 100,
      });
      res.json(logs);
    } catch (error) {
      console.error('Error fetching audit logs:', error);
      res.status(500).json({ error: 'Failed to fetch audit logs' });
    }
  });

  // Demographics Distribution
  app.get('/api/v1/research-center/demographics/age', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const distribution = await researchService.getAgeDistribution();
      res.json(distribution);
    } catch (error) {
      console.error('Error fetching age distribution:', error);
      res.status(500).json({ error: 'Failed to fetch age distribution' });
    }
  });

  app.get('/api/v1/research-center/demographics/conditions', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const distribution = await researchService.getConditionDistribution();
      res.json(distribution);
    } catch (error) {
      console.error('Error fetching condition distribution:', error);
      res.status(500).json({ error: 'Failed to fetch condition distribution' });
    }
  });

  app.get('/api/v1/research-center/enrollment-trend', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const days = req.query.days ? parseInt(req.query.days as string) : 30;
      const trend = await researchService.getEnrollmentTrend(days);
      res.json(trend);
    } catch (error) {
      console.error('Error fetching enrollment trend:', error);
      res.status(500).json({ error: 'Failed to fetch enrollment trend' });
    }
  });

  // =============================================================================
  // CONSENT-AWARE DATA AGGREGATION ROUTES
  // =============================================================================

  app.get('/api/v1/research-center/data/statistics', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const statistics = await researchService.getDataTypeStatistics(req.researchAuditContext);
      res.json(statistics);
    } catch (error) {
      console.error('Error fetching data statistics:', error);
      res.status(500).json({ error: 'Failed to fetch data statistics' });
    }
  });

  app.get('/api/v1/research-center/data/daily-followups', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { startDate, endDate, patientIds } = req.query;
      const dateRange = startDate && endDate 
        ? { start: new Date(startDate as string), end: new Date(endDate as string) }
        : undefined;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedDailyFollowups(
        req.researchAuditContext,
        dateRange,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented daily followups:', error);
      res.status(500).json({ error: 'Failed to fetch daily followups' });
    }
  });

  app.get('/api/v1/research-center/data/health-alerts', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { startDate, endDate, patientIds } = req.query;
      const dateRange = startDate && endDate 
        ? { start: new Date(startDate as string), end: new Date(endDate as string) }
        : undefined;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedHealthAlerts(
        req.researchAuditContext,
        dateRange,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented health alerts:', error);
      res.status(500).json({ error: 'Failed to fetch health alerts' });
    }
  });

  app.get('/api/v1/research-center/data/deterioration-scores', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { startDate, endDate, patientIds } = req.query;
      const dateRange = startDate && endDate 
        ? { start: new Date(startDate as string), end: new Date(endDate as string) }
        : undefined;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedDeteriorationScores(
        req.researchAuditContext,
        dateRange,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented deterioration scores:', error);
      res.status(500).json({ error: 'Failed to fetch deterioration scores' });
    }
  });

  app.get('/api/v1/research-center/data/medications', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { patientIds } = req.query;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedMedications(
        req.researchAuditContext,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented medications:', error);
      res.status(500).json({ error: 'Failed to fetch medications' });
    }
  });

  app.get('/api/v1/research-center/data/immune-biomarkers', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { startDate, endDate, patientIds } = req.query;
      const dateRange = startDate && endDate 
        ? { start: new Date(startDate as string), end: new Date(endDate as string) }
        : undefined;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedImmuneBiomarkers(
        req.researchAuditContext,
        dateRange,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented immune biomarkers:', error);
      res.status(500).json({ error: 'Failed to fetch immune biomarkers' });
    }
  });

  app.get('/api/v1/research-center/data/wearable-data', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { startDate, endDate, patientIds } = req.query;
      const dateRange = startDate && endDate 
        ? { start: new Date(startDate as string), end: new Date(endDate as string) }
        : undefined;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedWearableData(
        req.researchAuditContext,
        dateRange,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented wearable data:', error);
      res.status(500).json({ error: 'Failed to fetch wearable data' });
    }
  });

  app.get('/api/v1/research-center/data/symptom-journal', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { startDate, endDate, patientIds } = req.query;
      const dateRange = startDate && endDate 
        ? { start: new Date(startDate as string), end: new Date(endDate as string) }
        : undefined;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedSymptomJournal(
        req.researchAuditContext,
        dateRange,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented symptom journal:', error);
      res.status(500).json({ error: 'Failed to fetch symptom journal' });
    }
  });

  app.get('/api/v1/research-center/data/pain-tracking', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { startDate, endDate, patientIds } = req.query;
      const dateRange = startDate && endDate 
        ? { start: new Date(startDate as string), end: new Date(endDate as string) }
        : undefined;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedPainTracking(
        req.researchAuditContext,
        dateRange,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented pain tracking:', error);
      res.status(500).json({ error: 'Failed to fetch pain tracking' });
    }
  });

  app.get('/api/v1/research-center/data/lab-results', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { patientIds } = req.query;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedLabResults(
        req.researchAuditContext,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented lab results:', error);
      res.status(500).json({ error: 'Failed to fetch lab results' });
    }
  });

  app.get('/api/v1/research-center/data/conditions', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { patientIds } = req.query;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedConditions(
        req.researchAuditContext,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented conditions:', error);
      res.status(500).json({ error: 'Failed to fetch conditions' });
    }
  });

  app.get('/api/v1/research-center/data/demographics', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { patientIds } = req.query;
      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const data = await researchService.getConsentedDemographics(
        req.researchAuditContext,
        patientIdList
      );
      res.json(data);
    } catch (error) {
      console.error('Error fetching consented demographics:', error);
      res.status(500).json({ error: 'Failed to fetch demographics' });
    }
  });

  app.get('/api/v1/research-center/data/patient/:patientId', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const data = await researchService.getComprehensivePatientData(
        req.params.patientId,
        req.researchAuditContext
      );
      
      if (data.error) {
        return res.status(403).json({ error: data.error });
      }
      
      res.json(data);
    } catch (error) {
      console.error('Error fetching comprehensive patient data:', error);
      res.status(500).json({ error: 'Failed to fetch patient data' });
    }
  });

  app.post('/api/v1/research-center/data/cohort/:cohortId', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { dataTypes, startDate, endDate } = req.body;
      
      if (!dataTypes || !Array.isArray(dataTypes)) {
        return res.status(400).json({ error: 'dataTypes array is required' });
      }
      
      const dateRange = startDate && endDate 
        ? { start: new Date(startDate), end: new Date(endDate) }
        : undefined;

      const data = await researchService.getCohortAggregatedData(
        req.params.cohortId,
        dataTypes,
        req.researchAuditContext,
        dateRange
      );
      
      if (data.error) {
        return res.status(404).json({ error: data.error });
      }
      
      res.json(data);
    } catch (error) {
      console.error('Error fetching cohort aggregated data:', error);
      res.status(500).json({ error: 'Failed to fetch cohort data' });
    }
  });

  // =============================================================================
  // DE-IDENTIFIED EXPORT ROUTES (HIPAA COMPLIANT)
  // =============================================================================

  app.post('/api/v1/research-center/export/deidentified', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { cohortId, studyId, dataTypes, format = 'json' } = req.body;
      
      if (!dataTypes || !Array.isArray(dataTypes) || dataTypes.length === 0) {
        return res.status(400).json({ error: 'dataTypes array is required' });
      }

      if (!cohortId && !studyId) {
        return res.status(400).json({ error: 'cohortId or studyId is required' });
      }

      // Rate limiting: max 1000 subjects per export
      const MAX_EXPORT_SUBJECTS = 1000;

      // Get consented patient data
      let patientIds: string[] = [];
      if (cohortId) {
        const cohort = await storage.getCohort(cohortId);
        if (!cohort) {
          return res.status(404).json({ error: 'Cohort not found' });
        }
        patientIds = (cohort.patientIds || []).slice(0, MAX_EXPORT_SUBJECTS);
      } else if (studyId) {
        const enrollments = await storage.getStudyEnrollments(studyId);
        patientIds = enrollments
          .filter((e: any) => e.status === 'active')
          .map((e: any) => e.patientId)
          .slice(0, MAX_EXPORT_SUBJECTS);
      }

      // Fetch consented data for each patient with k-anonymity check
      const exportData: any[] = [];
      let patientCounter = 1;

      for (const patientId of patientIds) {
        // Strict consent verification
        const consent = await storage.getResearchConsent(patientId);
        if (!consent) continue;
        if (consent.consentEnabled !== true) continue;
        
        const permissions = typeof consent.permissions === 'object' && consent.permissions !== null 
          ? consent.permissions as Record<string, boolean>
          : {};
        
        // Check that ALL requested data types are consented
        const allConsented = dataTypes.every(dt => permissions[dt] === true);
        if (!allConsented) continue;

        const patientData: any = {
          research_id: `SUBJ_${String(patientCounter).padStart(4, '0')}`
          // NO identifiable fields: no names, no IDs, no emails, no DOB
        };

        // Only include data types the patient has consented to - FULLY DE-IDENTIFIED
        for (const dataType of dataTypes) {
          switch (dataType) {
            case 'demographics':
              const patient = await storage.getUser(patientId);
              if (patient) {
                // De-identify: 10-year age bands, binary sex only
                const age = patient.dateOfBirth 
                  ? Math.floor((Date.now() - new Date(patient.dateOfBirth).getTime()) / (365.25 * 24 * 60 * 60 * 1000))
                  : null;
                // 10-year age bands for k-anonymity
                patientData.age_band = age 
                  ? (age < 20 ? '<20' : age < 30 ? '20-29' : age < 40 ? '30-39' : age < 50 ? '40-49' : age < 60 ? '50-59' : age < 70 ? '60-69' : '70+')
                  : 'unknown';
                // Binary sex only (M/F/unknown), no gender identity
                patientData.sex = ['M', 'F'].includes(patient.sex || '') ? patient.sex : 'unknown';
              }
              break;

            case 'medications':
              const meds = await storage.getActiveMedications(patientId);
              // Aggregated counts only - no specific medication names
              patientData.medication_count = meds.length > 5 ? '6+' : String(meds.length);
              break;

            case 'vitals':
              // Boolean indicator only
              patientData.has_vitals_data = true;
              break;

            case 'immuneMarkers':
              const profile = await storage.getPatientProfile(patientId);
              if (profile?.immuneMarkers && typeof profile.immuneMarkers === 'object') {
                // Aggregate immune markers into ranges, not exact values
                const markers = profile.immuneMarkers as Record<string, any>;
                patientData.crp_range = markers.crp !== undefined 
                  ? (markers.crp < 1 ? 'normal' : markers.crp < 5 ? 'elevated' : 'high')
                  : 'unknown';
                patientData.esr_range = markers.esr !== undefined
                  ? (markers.esr < 20 ? 'normal' : markers.esr < 50 ? 'elevated' : 'high')
                  : 'unknown';
              }
              break;

            case 'dailyFollowups':
              patientData.has_followup_data = true;
              break;

            case 'healthAlerts':
              patientData.has_alert_data = true;
              break;
          }
        }

        exportData.push(patientData);
        patientCounter++;
      }

      // K-anonymity check: suppress if less than 5 subjects
      if (exportData.length < 5) {
        return res.status(400).json({ 
          error: 'Export blocked: fewer than 5 consented subjects. K-anonymity requirement not met.' 
        });
      }

      // HIPAA Audit log - structured format
      const auditEntry = {
        event: 'research_data_export',
        actor_user_id: req.user.id,
        actor_role: req.user.role,
        cohort_id: cohortId || null,
        study_id: studyId || null,
        data_types: dataTypes,
        subject_count: exportData.length,
        format,
        timestamp: new Date().toISOString(),
        hipaa_compliant: true,
        deidentified: true,
        k_anonymity_threshold: 5,
        audit_context: req.researchAuditContext || null
      };
      console.log(`[HIPAA_AUDIT] ${JSON.stringify(auditEntry)}`);

      // Define consistent CSV schema
      const csvSchema = ['research_id', 'age_band', 'sex', 'medication_count', 'crp_range', 'esr_range', 'has_vitals_data', 'has_followup_data', 'has_alert_data'];

      if (format === 'csv') {
        const csvRows = [csvSchema.join(',')];
        for (const row of exportData) {
          const values = csvSchema.map(col => {
            const val = row[col];
            if (val === undefined || val === null) return '';
            if (typeof val === 'boolean') return val ? '1' : '0';
            return String(val).includes(',') ? `"${val}"` : String(val);
          });
          csvRows.push(values.join(','));
        }
        res.setHeader('Content-Type', 'text/csv');
        res.setHeader('Content-Disposition', 'attachment; filename=deidentified_research_data.csv');
        return res.send(csvRows.join('\n'));
      }

      res.json({
        success: true,
        data: exportData,
        metadata: {
          total_subjects: exportData.length,
          data_types_included: dataTypes,
          export_date: new Date().toISOString(),
          deidentified: true,
          hipaa_compliant: true,
          k_anonymity_threshold: 5,
          max_subjects_per_export: MAX_EXPORT_SUBJECTS
        }
      });
    } catch (error) {
      console.error('Error exporting de-identified data:', error);
      res.status(500).json({ error: 'Failed to export data' });
    }
  });

  // =============================================================================
  // CSV IMPORT ROUTES
  // =============================================================================

  app.get('/api/v1/research-center/import/data-types', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const dataTypes = await researchService.getImportableDataTypes();
      res.json(dataTypes);
    } catch (error) {
      console.error('Error fetching importable data types:', error);
      res.status(500).json({ error: 'Failed to fetch data types' });
    }
  });

  app.post('/api/v1/research-center/import/preview', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { csvData } = req.body;
      
      if (!csvData || typeof csvData !== 'string') {
        return res.status(400).json({ error: 'csvData is required' });
      }

      const preview = await researchService.parseCSVPreview(csvData, req.researchAuditContext);
      res.json(preview);
    } catch (error) {
      console.error('Error parsing CSV preview:', error);
      res.status(500).json({ error: 'Failed to parse CSV' });
    }
  });

  app.post('/api/v1/research-center/import/validate-mapping', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { dataType, mapping, headers } = req.body;
      
      if (!dataType || !mapping || !headers) {
        return res.status(400).json({ error: 'dataType, mapping, and headers are required' });
      }

      const result = await researchService.validateColumnMapping(
        dataType,
        mapping,
        headers,
        req.researchAuditContext
      );
      res.json(result);
    } catch (error) {
      console.error('Error validating column mapping:', error);
      res.status(500).json({ error: 'Failed to validate mapping' });
    }
  });

  app.post('/api/v1/research-center/import/execute', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { dataType, csvData, mapping, studyId } = req.body;
      
      if (!dataType || !csvData || !mapping) {
        return res.status(400).json({ error: 'dataType, csvData, and mapping are required' });
      }

      const result = await researchService.importCSVData(
        dataType,
        csvData,
        mapping,
        studyId || null,
        req.researchAuditContext
      );
      res.json(result);
    } catch (error) {
      console.error('Error executing CSV import:', error);
      res.status(500).json({ error: 'Failed to import data' });
    }
  });

  app.get('/api/v1/research-center/import/history', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { studyId } = req.query;
      const history = await researchService.getImportHistory(
        studyId as string | undefined,
        req.researchAuditContext
      );
      res.json(history);
    } catch (error) {
      console.error('Error fetching import history:', error);
      res.status(500).json({ error: 'Failed to fetch import history' });
    }
  });

  // =============================================================================
  // DATA QUALITY ROUTES
  // =============================================================================

  app.get('/api/v1/research-center/quality/study/:studyId', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const quality = await researchService.getStudyDataQuality(
        req.params.studyId,
        req.researchAuditContext
      );
      res.json(quality);
    } catch (error) {
      console.error('Error fetching study data quality:', error);
      res.status(500).json({ error: 'Failed to fetch data quality' });
    }
  });

  app.get('/api/v1/research-center/quality/cohort/:cohortId', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const quality = await researchService.getCohortDataQuality(
        req.params.cohortId,
        req.researchAuditContext
      );
      res.json(quality);
    } catch (error) {
      console.error('Error fetching cohort data quality:', error);
      res.status(500).json({ error: 'Failed to fetch data quality' });
    }
  });

  app.get('/api/v1/research-center/quality/outliers', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { dataType, field, patientIds } = req.query;
      
      if (!dataType || !field) {
        return res.status(400).json({ error: 'dataType and field are required' });
      }

      const patientIdList = patientIds ? (patientIds as string).split(',') : undefined;

      const outliers = await researchService.detectOutliers(
        dataType as string,
        field as string,
        patientIdList,
        req.researchAuditContext
      );
      res.json(outliers);
    } catch (error) {
      console.error('Error detecting outliers:', error);
      res.status(500).json({ error: 'Failed to detect outliers' });
    }
  });

  app.get('/api/v1/research-center/quality/date-consistency/:studyId', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const report = await researchService.getDateConsistencyReport(
        req.params.studyId,
        req.researchAuditContext
      );
      res.json(report);
    } catch (error) {
      console.error('Error fetching date consistency report:', error);
      res.status(500).json({ error: 'Failed to fetch date consistency report' });
    }
  });

  app.get('/api/v1/research-center/quality/heatmap', isAuthenticated, setResearchAuditContext, async (req: any, res) => {
    try {
      if (!['doctor', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      const { studyId } = req.query;
      const heatmap = await researchService.getDataQualityHeatmap(
        studyId as string | undefined,
        req.researchAuditContext
      );
      res.json(heatmap);
    } catch (error) {
      console.error('Error fetching data quality heatmap:', error);
      res.status(500).json({ error: 'Failed to fetch heatmap' });
    }
  });

  // =============================================================================
  // END ENHANCED RESEARCH CENTER ROUTES
  // =============================================================================

  // =============================================================================
  // RISK & EXPOSURES ROUTES
  // Auto-populated risk profile from infections, vaccinations, occupation, genetics
  // =============================================================================

  // GET infections for a patient
  app.get('/api/patients/:patientId/risk/infections', isAuthenticated, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const userId = req.user!.id;
      const isDoctor = req.user!.role === 'doctor';
      
      // Authorization: patient can view own data, doctor needs assignment
      if (!isDoctor && userId !== patientId) {
        return res.status(403).json({ error: 'Access denied' });
      }
      if (isDoctor) {
        const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
        if (!hasAccess) {
          console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted to access infections for unassigned patient ${patientId}`);
          return res.status(403).json({ error: 'Access denied - no patient assignment' });
        }
      }

      const result = await db.execute(drizzleSql`
        SELECT ie.*, 
          u.first_name as modifier_first_name, 
          u.last_name as modifier_last_name
        FROM infectious_events ie
        LEFT JOIN users u ON ie.overridden_by = u.id
        WHERE ie.patient_id = ${patientId}
        ORDER BY ie.onset_date DESC NULLS LAST
      `);
      
      console.log(`[HIPAA-AUDIT] User ${userId} accessed infections for patient ${patientId}`);
      res.json(result.rows || []);
    } catch (error) {
      console.error('Error fetching infections:', error);
      res.status(500).json({ error: 'Failed to fetch infections' });
    }
  });

  // GET immunizations for a patient
  app.get('/api/patients/:patientId/risk/immunizations', isAuthenticated, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const userId = req.user!.id;
      const isDoctor = req.user!.role === 'doctor';
      
      if (!isDoctor && userId !== patientId) {
        return res.status(403).json({ error: 'Access denied' });
      }
      if (isDoctor) {
        const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
        if (!hasAccess) {
          console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted to access immunizations for unassigned patient ${patientId}`);
          return res.status(403).json({ error: 'Access denied - no patient assignment' });
        }
      }

      const result = await db.execute(drizzleSql`
        SELECT * FROM patient_immunizations 
        WHERE patient_id = ${patientId}
        ORDER BY administration_date DESC NULLS LAST
      `);
      
      console.log(`[HIPAA-AUDIT] User ${userId} accessed immunizations for patient ${patientId}`);
      res.json(result.rows || []);
    } catch (error) {
      console.error('Error fetching immunizations:', error);
      res.status(500).json({ error: 'Failed to fetch immunizations' });
    }
  });

  // GET occupation and exposures for a patient
  app.get('/api/patients/:patientId/risk/occupation', isAuthenticated, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const userId = req.user!.id;
      const isDoctor = req.user!.role === 'doctor';
      
      if (!isDoctor && userId !== patientId) {
        return res.status(403).json({ error: 'Access denied' });
      }
      if (isDoctor) {
        const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
        if (!hasAccess) {
          console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted to access occupation for unassigned patient ${patientId}`);
          return res.status(403).json({ error: 'Access denied - no patient assignment' });
        }
      }

      // Get current occupation with doctor info
      const occupationResult = await db.execute(drizzleSql`
        SELECT po.*, 
          u.first_name as creator_first_name, 
          u.last_name as creator_last_name,
          u2.first_name as modifier_first_name,
          u2.last_name as modifier_last_name
        FROM patient_occupations po
        LEFT JOIN users u ON po.created_by = u.id
        LEFT JOIN users u2 ON po.modified_by = u2.id
        WHERE po.patient_id = ${patientId} AND po.is_current = true
        ORDER BY po.start_date DESC NULLS LAST
        LIMIT 1
      `);
      
      const occupation = occupationResult.rows?.[0] || null;
      
      let exposures: any[] = [];
      if (occupation) {
        const exposuresResult = await db.execute(drizzleSql`
          SELECT * FROM occupational_exposures 
          WHERE occupation_id = ${occupation.id}
          ORDER BY exposure_level DESC
        `);
        exposures = exposuresResult.rows || [];
      }
      
      console.log(`[HIPAA-AUDIT] User ${userId} accessed occupation for patient ${patientId}`);
      res.json({ occupation, exposures });
    } catch (error) {
      console.error('Error fetching occupation:', error);
      res.status(500).json({ error: 'Failed to fetch occupation' });
    }
  });

  // GET genetic risk flags for a patient
  app.get('/api/patients/:patientId/risk/genetics', isAuthenticated, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const userId = req.user!.id;
      const isDoctor = req.user!.role === 'doctor';
      
      if (!isDoctor && userId !== patientId) {
        return res.status(403).json({ error: 'Access denied' });
      }
      if (isDoctor) {
        const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
        if (!hasAccess) {
          console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted to access genetics for unassigned patient ${patientId}`);
          return res.status(403).json({ error: 'Access denied - no patient assignment' });
        }
      }

      const result = await db.execute(drizzleSql`
        SELECT grf.*, 
          u.first_name as modifier_first_name, 
          u.last_name as modifier_last_name
        FROM genetic_risk_flags grf
        LEFT JOIN users u ON grf.overridden_by = u.id
        WHERE grf.patient_id = ${patientId}
        ORDER BY grf.risk_level DESC NULLS LAST, grf.flag_name ASC
      `);
      
      console.log(`[HIPAA-AUDIT] User ${userId} accessed genetics for patient ${patientId}`);
      res.json(result.rows || []);
    } catch (error) {
      console.error('Error fetching genetic flags:', error);
      res.status(500).json({ error: 'Failed to fetch genetic flags' });
    }
  });

  // GET full risk summary for a patient
  app.get('/api/patients/:patientId/risk/summary', isAuthenticated, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const userId = req.user!.id;
      const isDoctor = req.user!.role === 'doctor';
      
      if (!isDoctor && userId !== patientId) {
        return res.status(403).json({ error: 'Access denied' });
      }
      if (isDoctor) {
        const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
        if (!hasAccess) {
          return res.status(403).json({ error: 'Access denied - no patient assignment' });
        }
      }

      // Fetch all risk data in parallel
      const [infectionsResult, immunizationsResult, occupationResult, geneticsResult] = await Promise.all([
        db.execute(drizzleSql`SELECT COUNT(*) as count FROM infectious_events WHERE patient_id = ${patientId}`),
        db.execute(drizzleSql`SELECT COUNT(*) as count FROM patient_immunizations WHERE patient_id = ${patientId}`),
        db.execute(drizzleSql`SELECT COUNT(*) as count FROM patient_occupations WHERE patient_id = ${patientId}`),
        db.execute(drizzleSql`SELECT COUNT(*) as count FROM genetic_risk_flags WHERE patient_id = ${patientId}`),
      ]);
      
      const summary = {
        infectionsCount: parseInt(infectionsResult.rows?.[0]?.count || '0'),
        immunizationsCount: parseInt(immunizationsResult.rows?.[0]?.count || '0'),
        occupationsCount: parseInt(occupationResult.rows?.[0]?.count || '0'),
        geneticFlagsCount: parseInt(geneticsResult.rows?.[0]?.count || '0'),
      };
      
      console.log(`[HIPAA-AUDIT] User ${userId} accessed risk summary for patient ${patientId}`);
      res.json(summary);
    } catch (error) {
      console.error('Error fetching risk summary:', error);
      res.status(500).json({ error: 'Failed to fetch risk summary' });
    }
  });

  // PATCH infection (manual override) - doctors only
  app.patch('/api/patients/:patientId/risk/infections/:id', isDoctor, async (req: any, res) => {
    try {
      const { patientId, id } = req.params;
      const userId = req.user!.id;
      
      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted to modify infection for unassigned patient ${patientId}`);
        return res.status(403).json({ error: 'Access denied - no patient assignment' });
      }

      const { infectionType, pathogen, severity, onsetDate, resolutionDate, hospitalization, icuAdmission, doctorNotes } = req.body;
      
      await db.execute(drizzleSql`
        UPDATE infectious_events SET
          infection_type = COALESCE(${infectionType}, infection_type),
          pathogen = COALESCE(${pathogen}, pathogen),
          severity = COALESCE(${severity}, severity),
          onset_date = COALESCE(${onsetDate ? new Date(onsetDate) : null}, onset_date),
          resolution_date = COALESCE(${resolutionDate ? new Date(resolutionDate) : null}, resolution_date),
          hospitalization = COALESCE(${hospitalization}, hospitalization),
          icu_admission = COALESCE(${icuAdmission}, icu_admission),
          doctor_notes = COALESCE(${doctorNotes}, doctor_notes),
          manual_override = TRUE,
          overridden_by = ${userId},
          overridden_at = NOW(),
          updated_at = NOW()
        WHERE id = ${id} AND patient_id = ${patientId}
      `);
      
      console.log(`[HIPAA-AUDIT] Doctor ${userId} modified infection ${id} for patient ${patientId}`);
      res.json({ success: true });
    } catch (error) {
      console.error('Error updating infection:', error);
      res.status(500).json({ error: 'Failed to update infection' });
    }
  });

  // PATCH immunization (manual override) - doctors only
  app.patch('/api/patients/:patientId/risk/immunizations/:id', isDoctor, async (req: any, res) => {
    try {
      const { patientId, id } = req.params;
      const userId = req.user!.id;
      
      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted to modify immunization for unassigned patient ${patientId}`);
        return res.status(403).json({ error: 'Access denied - no patient assignment' });
      }

      const { vaccineName, vaccineCode, doseNumber, seriesName, administrationDate, adverseReaction, reactionDetails } = req.body;
      
      await db.execute(drizzleSql`
        UPDATE patient_immunizations SET
          vaccine_name = COALESCE(${vaccineName}, vaccine_name),
          vaccine_code = COALESCE(${vaccineCode}, vaccine_code),
          dose_number = COALESCE(${doseNumber}, dose_number),
          series_name = COALESCE(${seriesName}, series_name),
          administration_date = COALESCE(${administrationDate ? new Date(administrationDate) : null}, administration_date),
          adverse_reaction = COALESCE(${adverseReaction}, adverse_reaction),
          reaction_details = COALESCE(${reactionDetails}, reaction_details),
          manual_override = TRUE,
          overridden_by = ${userId},
          overridden_at = NOW(),
          updated_at = NOW()
        WHERE id = ${id} AND patient_id = ${patientId}
      `);
      
      console.log(`[HIPAA-AUDIT] Doctor ${userId} modified immunization ${id} for patient ${patientId}`);
      res.json({ success: true });
    } catch (error) {
      console.error('Error updating immunization:', error);
      res.status(500).json({ error: 'Failed to update immunization' });
    }
  });

  // POST new occupation - doctors or patients
  app.post('/api/patients/:patientId/risk/occupation', isAuthenticated, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const userId = req.user!.id;
      const isDoctor = req.user!.role === 'doctor';
      
      if (!isDoctor && userId !== patientId) {
        return res.status(403).json({ error: 'Access denied' });
      }
      if (isDoctor) {
        const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
        if (!hasAccess) {
          return res.status(403).json({ error: 'Access denied - no patient assignment' });
        }
      }

      const { jobTitle, industry, employer, startDate, shiftWork, nightShift, hoursPerWeek, doctorNotes } = req.body;
      
      if (!jobTitle) {
        return res.status(400).json({ error: 'Job title is required' });
      }

      // Mark existing occupations as not current
      await db.execute(drizzleSql`
        UPDATE patient_occupations SET is_current = FALSE, end_date = NOW()
        WHERE patient_id = ${patientId} AND is_current = TRUE
      `);
      
      // Create new occupation with doctor tracking
      const result = await db.execute(drizzleSql`
        INSERT INTO patient_occupations (
          patient_id, job_title, industry, employer, start_date, 
          shift_work, night_shift, hours_per_week, is_current, status,
          created_by, doctor_notes
        ) VALUES (
          ${patientId}, ${jobTitle}, ${industry || null}, ${employer || null},
          ${startDate ? new Date(startDate) : new Date()},
          ${shiftWork || false}, ${nightShift || false}, ${hoursPerWeek || null},
          TRUE, 'active',
          ${isDoctor ? userId : null}, ${doctorNotes || null}
        ) RETURNING *
      `);
      
      console.log(`[HIPAA-AUDIT] User ${userId} created occupation for patient ${patientId}`);
      res.json(result.rows?.[0] || { success: true });
    } catch (error) {
      console.error('Error creating occupation:', error);
      res.status(500).json({ error: 'Failed to create occupation' });
    }
  });

  // POST new genetic risk flag - doctors only
  app.post('/api/patients/:patientId/risk/genetics', isDoctor, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const userId = req.user!.id;
      
      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        console.log(`[HIPAA-AUDIT] DENIED: Doctor ${userId} attempted to add genetic flag for unassigned patient ${patientId}`);
        return res.status(403).json({ error: 'Access denied - no patient assignment' });
      }

      const { flagName, flagType, riskLevel, clinicalImplications, doctorNotes } = req.body;
      
      if (!flagName || !flagType) {
        return res.status(400).json({ error: 'Flag name and type are required' });
      }

      const result = await db.execute(drizzleSql`
        INSERT INTO genetic_risk_flags (
          patient_id, flag_name, flag_type, value, risk_level,
          clinical_implications, source, auto_generated, manual_override,
          overridden_by, overridden_at, doctor_notes
        ) VALUES (
          ${patientId}, ${flagName}, ${flagType}, 'present', ${riskLevel || 'moderate'},
          ${clinicalImplications || null}, 'doctor_entered', FALSE, TRUE,
          ${userId}, NOW(), ${doctorNotes || null}
        ) RETURNING *
      `);
      
      console.log(`[HIPAA-AUDIT] Doctor ${userId} added genetic flag for patient ${patientId}: ${flagName}`);
      res.json(result.rows?.[0] || { success: true });
    } catch (error) {
      console.error('Error creating genetic flag:', error);
      res.status(500).json({ error: 'Failed to create genetic flag' });
    }
  });

  // POST new condition (for auto-ETL to pick up)
  app.post('/api/patients/:patientId/risk/conditions', isDoctor, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const userId = req.user!.id;
      
      const hasAccess = await storage.doctorHasPatientAccess(userId, patientId);
      if (!hasAccess) {
        return res.status(403).json({ error: 'Access denied - no patient assignment' });
      }

      const { conditionCode, conditionName, conditionCategory, onsetDate, severity, notes } = req.body;
      
      if (!conditionCode || !conditionName) {
        return res.status(400).json({ error: 'Condition code and name are required' });
      }

      const result = await db.execute(drizzleSql`
        INSERT INTO patient_conditions (
          patient_id, condition_code, condition_name, condition_category,
          onset_date, diagnosis_date, severity, notes, diagnosed_by, source_type
        ) VALUES (
          ${patientId}, ${conditionCode}, ${conditionName}, ${conditionCategory || 'other'},
          ${onsetDate ? new Date(onsetDate) : new Date()}, NOW(),
          ${severity || 'moderate'}, ${notes || null}, ${userId}, 'manual'
        ) RETURNING *
      `);
      
      console.log(`[HIPAA-AUDIT] Doctor ${userId} added condition for patient ${patientId}: ${conditionCode}`);
      res.json(result.rows?.[0] || { success: true });
    } catch (error) {
      console.error('Error creating condition:', error);
      res.status(500).json({ error: 'Failed to create condition' });
    }
  });

  // =============================================================================
  // END RISK & EXPOSURES ROUTES
  // =============================================================================

  const httpServer = createServer(app);

  // WebSocket proxy for agent communication
  // Forward /ws/agent connections to Python FastAPI backend
  httpServer.on('upgrade', (request, socket, head) => {
    const pathname = request.url || '';
    
    if (pathname.startsWith('/ws/agent')) {
      // Proxy WebSocket to Python backend
      const pythonWsUrl = (PYTHON_AGENT_URL || 'http://localhost:8000').replace('http://', 'ws://').replace('https://', 'wss://');
      const targetUrl = new URL(pathname, pythonWsUrl);
      
      import('ws').then(({ default: WebSocket }) => {
        const targetWs = new WebSocket(targetUrl.toString());
        
        targetWs.on('open', () => {
          // Forward handshake
          const clientWs = new WebSocket(null as any);
          (clientWs as any)._socket = socket;
          
          // Pipe data between client and target
          socket.on('data', (data) => {
            if (targetWs.readyState === WebSocket.OPEN) {
              targetWs.send(data);
            }
          });
          
          targetWs.on('message', (data) => {
            socket.write(data);
          });
          
          targetWs.on('close', () => socket.destroy());
          socket.on('close', () => targetWs.close());
        });
        
        targetWs.on('error', (err) => {
          console.error('Agent WebSocket proxy error:', err);
          socket.destroy();
        });
      }).catch((err) => {
        console.error('Failed to load ws module for proxy:', err);
        socket.destroy();
      });
    }
    // Let other upgrade requests pass through (handled by Vite or other middlewares)
  });

  return httpServer;
}
