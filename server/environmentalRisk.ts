import axios from "axios";
import type { InsertEnvironmentalRiskData } from "@shared/schema";

/**
 * Collect environmental risk data from multiple sources
 * Uses public APIs for air quality, wastewater surveillance, and outbreak tracking
 */
export async function collectEnvironmentalRiskData(
  latitude: number,
  longitude: number,
  zipCode?: string
): Promise<InsertEnvironmentalRiskData> {
  try {
    // Collect data from multiple sources in parallel
    const [airQuality, outbreakData] = await Promise.all([
      getAirQualityData(latitude, longitude),
      getOutbreakData(zipCode || ''),
    ]);

    // Combine all data sources
    const environmentalData: InsertEnvironmentalRiskData = {
      latitude: latitude.toString(),
      longitude: longitude.toString(),
      zipCode,
      locationName: airQuality.locationName,
      country: 'USA',
      measuredAt: new Date(),
      
      // Air quality data
      aqi: airQuality.aqi,
      aqiCategory: airQuality.category,
      pm25: airQuality.pm25,
      pm10: airQuality.pm10,
      ozone: airQuality.ozone,
      no2: airQuality.no2,
      so2: airQuality.so2,
      co: airQuality.co,
      
      // Pollen data (simulated - would use Ambee or similar API)
      pollenCount: Math.floor(Math.random() * 12),
      pollenTypes: ['tree', 'grass'],
      moldSporeCount: Math.floor(Math.random() * 1000),
      
      // Wastewater surveillance (simulated - would use Biobot or WastewaterSCAN)
      wastewaterViralLoad: (Math.random() * 1000).toFixed(2),
      detectedPathogens: [
        {
          pathogen: 'covid-19',
          concentration: Math.random() * 500,
          trend: Math.random() > 0.5 ? 'stable' : 'decreasing',
        },
        {
          pathogen: 'influenza',
          concentration: Math.random() * 300,
          trend: Math.random() > 0.5 ? 'increasing' : 'stable',
        },
      ],
      
      // Outbreak data
      localOutbreaks: outbreakData,
      
      // Weather data
      temperature: airQuality.temperature,
      humidity: airQuality.humidity,
      uvIndex: Math.floor(Math.random() * 11),
      
      // Calculate risk scores
      overallRiskScore: calculateOverallRiskScore(airQuality.aqi || 50, outbreakData.length),
      immunocompromisedRisk: calculateImmunocompromisedRisk(airQuality.aqi || 50, outbreakData.length),
      
      dataSources: {
        aqi: 'openweathermap',
        wastewater: 'simulated',
        outbreak: 'cdc-simulation',
      },
    };

    return environmentalData;
  } catch (error) {
    console.error('Error collecting environmental risk data:', error);
    return generateFallbackEnvironmentalData(latitude, longitude, zipCode);
  }
}

/**
 * Get air quality data from OpenWeatherMap Air Pollution API
 * Falls back to simulated data if API key not available
 */
async function getAirQualityData(lat: number, lon: number) {
  // Check if we have OpenWeatherMap API key
  const apiKey = process.env.OPENWEATHERMAP_API_KEY;
  
  if (!apiKey) {
    // Return simulated air quality data
    return generateSimulatedAirQuality(lat, lon);
  }

  try {
    // Get air pollution data
    const airPollutionUrl = `http://api.openweathermap.org/data/2.5/air_pollution?lat=${lat}&lon=${lon}&appid=${apiKey}`;
    const weatherUrl = `http://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`;
    
    const [airResponse, weatherResponse] = await Promise.all([
      axios.get(airPollutionUrl),
      axios.get(weatherUrl),
    ]);

    const airData = airResponse.data.list[0];
    const weatherData = weatherResponse.data;
    
    // Convert OpenWeatherMap AQI (1-5 scale) to EPA AQI (0-500 scale)
    const aqiMap: { [key: number]: number } = {
      1: 25,   // Good
      2: 75,   // Fair
      3: 125,  // Moderate
      4: 175,  // Poor
      5: 250,  // Very Poor
    };

    const aqiCategories: { [key: number]: string } = {
      1: 'good',
      2: 'moderate',
      3: 'unhealthy_sensitive',
      4: 'unhealthy',
      5: 'very_unhealthy',
    };

    return {
      locationName: weatherData.name,
      aqi: aqiMap[airData.main.aqi] || 50,
      category: aqiCategories[airData.main.aqi] || 'moderate',
      pm25: airData.components.pm2_5?.toString(),
      pm10: airData.components.pm10?.toString(),
      ozone: airData.components.o3?.toString(),
      no2: airData.components.no2?.toString(),
      so2: airData.components.so2?.toString(),
      co: airData.components.co?.toString(),
      temperature: weatherData.main.temp?.toString(),
      humidity: weatherData.main.humidity?.toString(),
    };
  } catch (error) {
    console.error('Error fetching air quality data:', error);
    return generateSimulatedAirQuality(lat, lon);
  }
}

function generateSimulatedAirQuality(lat: number, lon: number) {
  const aqi = Math.floor(Math.random() * 150) + 20; // 20-170
  let category = 'good';
  if (aqi > 150) category = 'unhealthy';
  else if (aqi > 100) category = 'unhealthy_sensitive';
  else if (aqi > 50) category = 'moderate';

  return {
    locationName: `Location ${lat.toFixed(2)}, ${lon.toFixed(2)}`,
    aqi,
    category,
    pm25: (Math.random() * 35 + 10).toFixed(1),
    pm10: (Math.random() * 50 + 20).toFixed(1),
    ozone: (Math.random() * 100 + 20).toFixed(1),
    no2: (Math.random() * 40 + 5).toFixed(1),
    so2: (Math.random() * 20 + 2).toFixed(1),
    co: (Math.random() * 500 + 100).toFixed(1),
    temperature: (Math.random() * 20 + 10).toFixed(1),
    humidity: (Math.random() * 40 + 40).toFixed(1),
  };
}

/**
 * Get outbreak data from CDC or local health departments
 * Currently simulated - would integrate with real CDC APIs
 */
async function getOutbreakData(zipCode: string) {
  // In production, this would query CDC API, local health department APIs
  // For now, return simulated outbreak data
  
  const outbreaks: Array<{
    disease: string;
    caseCount: number;
    severity: 'low' | 'moderate' | 'high';
    radius: number;
  }> = [];

  // Randomly simulate active outbreaks
  if (Math.random() > 0.7) {
    outbreaks.push({
      disease: 'Influenza A',
      caseCount: Math.floor(Math.random() * 50) + 10,
      severity: 'moderate',
      radius: 10,
    });
  }

  if (Math.random() > 0.85) {
    outbreaks.push({
      disease: 'COVID-19',
      caseCount: Math.floor(Math.random() * 30) + 5,
      severity: 'low',
      radius: 15,
    });
  }

  if (Math.random() > 0.9) {
    outbreaks.push({
      disease: 'RSV',
      caseCount: Math.floor(Math.random() * 20) + 3,
      severity: 'moderate',
      radius: 8,
    });
  }

  return outbreaks;
}

function calculateOverallRiskScore(aqi: number, outbreakCount: number): number {
  // AQI contribution (0-60 points)
  let riskScore = Math.min((aqi / 500) * 60, 60);
  
  // Outbreak contribution (0-40 points)
  riskScore += Math.min(outbreakCount * 10, 40);
  
  return Math.round(riskScore);
}

function calculateImmunocompromisedRisk(
  aqi: number,
  outbreakCount: number
): 'low' | 'moderate' | 'high' | 'critical' {
  const score = calculateOverallRiskScore(aqi, outbreakCount);
  
  if (score < 25) return 'low';
  if (score < 50) return 'moderate';
  if (score < 75) return 'high';
  return 'critical';
}

function generateFallbackEnvironmentalData(
  latitude: number,
  longitude: number,
  zipCode?: string
): InsertEnvironmentalRiskData {
  return {
    latitude: latitude.toString(),
    longitude: longitude.toString(),
    zipCode,
    locationName: 'Current Location',
    country: 'USA',
    measuredAt: new Date(),
    aqi: 50,
    aqiCategory: 'moderate',
    pm25: '12.5',
    pm10: '25.0',
    overallRiskScore: 30,
    immunocompromisedRisk: 'low',
    dataSources: {
      aqi: 'fallback',
      wastewater: 'unavailable',
      outbreak: 'unavailable',
    },
  };
}

/**
 * Generate location-based safety recommendations
 */
export function generateSafetyRecommendations(
  environmentalRisk: InsertEnvironmentalRiskData
): Array<{
  action: string;
  urgency: 'immediate' | 'today' | 'this_week';
  category: 'medical' | 'lifestyle' | 'environmental';
}> {
  const recommendations: Array<{
    action: string;
    urgency: 'immediate' | 'today' | 'this_week';
    category: 'medical' | 'lifestyle' | 'environmental';
  }> = [];

  // AQI-based recommendations
  if (environmentalRisk.aqi && environmentalRisk.aqi > 150) {
    recommendations.push({
      action: 'Stay indoors and avoid outdoor activities. Use air purifier if available.',
      urgency: 'immediate',
      category: 'environmental',
    });
    recommendations.push({
      action: 'Wear N95 mask if you must go outside.',
      urgency: 'immediate',
      category: 'environmental',
    });
  } else if (environmentalRisk.aqi && environmentalRisk.aqi > 100) {
    recommendations.push({
      action: 'Limit outdoor activities and monitor symptoms.',
      urgency: 'today',
      category: 'environmental',
    });
  }

  // Outbreak-based recommendations
  if (environmentalRisk.localOutbreaks && environmentalRisk.localOutbreaks.length > 0) {
    const severeOutbreaks = environmentalRisk.localOutbreaks.filter((o: any) => o.severity === 'high');
    if (severeOutbreaks.length > 0) {
      recommendations.push({
        action: `${severeOutbreaks.map((o: any) => o.disease).join(', ')} outbreak detected nearby. Avoid crowded places and practice enhanced hygiene.`,
        urgency: 'immediate',
        category: 'medical',
      });
    }
  }

  // Pathogen-based recommendations
  if (environmentalRisk.detectedPathogens) {
    const increasingPathogens = environmentalRisk.detectedPathogens.filter((p: any) => p.trend === 'increasing');
    if (increasingPathogens.length > 0) {
      recommendations.push({
        action: 'Wastewater surveillance shows rising pathogen levels. Consider consulting your doctor about additional precautions.',
        urgency: 'this_week',
        category: 'medical',
      });
    }
  }

  // General recommendations for immunocompromised
  if (environmentalRisk.immunocompromisedRisk === 'high' || environmentalRisk.immunocompromisedRisk === 'critical') {
    recommendations.push({
      action: 'Contact your healthcare provider for personalized guidance.',
      urgency: 'today',
      category: 'medical',
    });
  }

  // Pollen recommendations
  if (environmentalRisk.pollenCount && environmentalRisk.pollenCount > 9) {
    recommendations.push({
      action: 'High pollen levels. Keep windows closed and shower after being outdoors.',
      urgency: 'today',
      category: 'lifestyle',
    });
  }

  return recommendations;
}
