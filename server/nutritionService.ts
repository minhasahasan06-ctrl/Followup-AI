import OpenAI from "openai";
import { storage } from "./storage";
import type { 
  InsertMealPlan, 
  InsertMeal, 
  InsertNutritionEntry,
  InsertMedicationSchedule 
} from "@shared/schema";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface MealPlanRequest {
  dietaryPreferences: any;
  medications: any[];
  immuneGoals: string[];
  allergies: string[];
  calorieTarget?: number;
}

interface MealRecommendation {
  mealType: string;
  mealName: string;
  description: string;
  ingredients: Array<{ name: string; amount: string; unit: string }>;
  recipeSuggestion: string;
  immuneBenefits: string[];
  scheduledTime: string;
  nutrition: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    fiber: number;
    vitaminC?: number;
    vitaminD?: number;
    zinc?: number;
    omega3?: number;
    immuneSupportScore: number;
  };
}

interface MedicationTimingRecommendation {
  medicationId: string;
  medicationName: string;
  optimizedTime: string;
  withFood: boolean;
  reasoning: string;
  coordinateWithMeal?: string;
}

export async function generateWeeklyMealPlan(
  patientId: string
): Promise<{ planId: string; meals: number }> {
  try {
    // 1. Get patient dietary preferences
    const dietaryPrefs = await storage.getDietaryPreferences(patientId);
    if (!dietaryPrefs) {
      throw new Error("No dietary preferences found. Please set up preferences first.");
    }

    // 2. Get active medications
    const medications = await storage.getActiveMedications(patientId);

    // 3. Build meal plan request
    const request: MealPlanRequest = {
      dietaryPreferences: dietaryPrefs,
      medications: medications.map(m => ({
        name: m.name,
        dosage: m.dosage,
        frequency: m.frequency,
      })),
      immuneGoals: ["immune_support", "energy", "inflammation_reduction"],
      allergies: dietaryPrefs.allergies || [],
      calorieTarget: dietaryPrefs.calorieTarget || undefined,
    };

    // 4. Use AI to generate meal recommendations
    const mealRecommendations = await generateMealRecommendationsWithAI(request);
    
    if (mealRecommendations.length === 0) {
      throw new Error("Failed to generate meal recommendations. Please try again.");
    }

    // 5. Create meal plan in database
    const weekStartDate = getStartOfCurrentWeek();
    const mealPlan = await storage.createMealPlan({
      patientId,
      weekStartDate,
      planName: `Immune-Boosting Week Plan - ${formatDate(weekStartDate)}`,
      aiGeneratedSummary: `Personalized 7-day meal plan optimized for immune support, ${dietaryPrefs.dietType || 'balanced'} diet, ${dietaryPrefs.calorieTarget || 2000} daily calories.`,
      totalCalories: dietaryPrefs.calorieTarget ? dietaryPrefs.calorieTarget * 7 : undefined,
      focusAreas: ["immune_support", "energy", "inflammation_reduction"],
      considersMedications: medications.length > 0,
      active: true,
    });

    // 6. Create individual meals
    let mealsCreated = 0;
    for (const mealRec of mealRecommendations) {
      const meal = await storage.createMeal({
        patientId,
        mealPlanId: mealPlan.id,
        mealType: mealRec.mealType,
        mealName: mealRec.mealName,
        description: mealRec.description,
        ingredients: mealRec.ingredients,
        recipeSuggestion: mealRec.recipeSuggestion,
        scheduledTime: parseScheduledTime(mealRec.scheduledTime),
        status: "planned",
        aiNutritionAnalysis: `Immune support score: ${mealRec.nutrition.immuneSupportScore}/100`,
        immuneBenefits: mealRec.immuneBenefits,
      });

      // Create nutrition entry for the meal
      await storage.createNutritionEntry({
        mealId: meal.id,
        patientId,
        calories: mealRec.nutrition.calories,
        protein: mealRec.nutrition.protein.toString(),
        carbs: mealRec.nutrition.carbs.toString(),
        fat: mealRec.nutrition.fat.toString(),
        fiber: mealRec.nutrition.fiber.toString(),
        vitaminC: mealRec.nutrition.vitaminC?.toString(),
        vitaminD: mealRec.nutrition.vitaminD?.toString(),
        zinc: mealRec.nutrition.zinc?.toString(),
        omega3: mealRec.nutrition.omega3?.toString(),
        immuneSupportScore: mealRec.nutrition.immuneSupportScore,
        dataSource: "ai_estimation",
      });

      mealsCreated++;
    }

    console.log(`Generated meal plan with ${mealsCreated} meals for patient ${patientId}`);
    return { planId: mealPlan.id, meals: mealsCreated };
  } catch (error) {
    console.error("Error generating meal plan:", error);
    throw error;
  }
}

async function generateMealRecommendationsWithAI(
  request: MealPlanRequest
): Promise<MealRecommendation[]> {
  const prompt = buildMealPlanPrompt(request);

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: `You are a nutrition expert specializing in immune-boosting diets for immunocompromised patients. Generate personalized meal plans that:
1. Support immune function with nutrient-dense foods
2. Respect dietary restrictions and allergies
3. Coordinate with medication timing
4. Provide balanced nutrition
5. Include practical, achievable recipes

Return ONLY valid JSON matching the specified format.`,
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 0.7,
      response_format: { type: "json_object" },
    });

    const content = completion.choices[0]?.message?.content;
    if (!content) {
      console.error("Empty response from OpenAI for meal plan");
      return [];
    }

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch (parseError) {
      console.error("Failed to parse OpenAI response as JSON:", parseError);
      console.error("Raw response:", content);
      return [];
    }

    const meals = parsed.meals || parsed.recommendations || [];
    if (!Array.isArray(meals)) {
      console.error("AI response meals field is not an array:", meals);
      return [];
    }

    // Validate and sanitize each meal recommendation
    return meals.map((meal: any) => ({
      mealType: meal.mealType || "snack",
      mealName: meal.mealName || meal.name || "Healthy Meal",
      description: meal.description || "",
      ingredients: Array.isArray(meal.ingredients) ? meal.ingredients : [],
      recipeSuggestion: meal.recipeSuggestion || meal.recipe || "",
      immuneBenefits: Array.isArray(meal.immuneBenefits) ? meal.immuneBenefits : [],
      scheduledTime: meal.scheduledTime || meal.time || "12:00",
      nutrition: {
        calories: Number(meal.nutrition?.calories) || 500,
        protein: Number(meal.nutrition?.protein) || 20,
        carbs: Number(meal.nutrition?.carbs) || 50,
        fat: Number(meal.nutrition?.fat) || 15,
        fiber: Number(meal.nutrition?.fiber) || 5,
        vitaminC: meal.nutrition?.vitaminC !== undefined ? Number(meal.nutrition.vitaminC) : undefined,
        vitaminD: meal.nutrition?.vitaminD !== undefined ? Number(meal.nutrition.vitaminD) : undefined,
        zinc: meal.nutrition?.zinc !== undefined ? Number(meal.nutrition.zinc) : undefined,
        omega3: meal.nutrition?.omega3 !== undefined ? Number(meal.nutrition.omega3) : undefined,
        immuneSupportScore: Number(meal.nutrition?.immuneSupportScore) || 70,
      },
    }));
  } catch (error) {
    console.error("Error calling OpenAI for meal recommendations:", error);
    // Return empty array on error to prevent API crashes
    return [];
  }
}

export async function optimizeMedicationTiming(
  patientId: string
): Promise<MedicationTimingRecommendation[]> {
  try {
    // 1. Get active medications
    const medications = await storage.getActiveMedications(patientId);
    if (medications.length === 0) {
      return [];
    }

    // 2. Get dietary preferences for meal timing
    const dietaryPrefs = await storage.getDietaryPreferences(patientId);

    // 3. Use AI to optimize medication timing
    const recommendations = await optimizeTimingWithAI(medications, dietaryPrefs);
    
    if (recommendations.length === 0) {
      console.log("No medication timing recommendations generated");
      return [];
    }

    // 4. Create/update medication schedules
    for (const rec of recommendations) {
      const medication = medications.find(m => m.name === rec.medicationName);
      if (medication) {
        await storage.createMedicationSchedule({
          medicationId: medication.id,
          patientId,
          timeOfDay: rec.optimizedTime,
          withFood: rec.withFood,
          specialInstructions: rec.reasoning,
          aiOptimized: true,
          aiReasoning: rec.reasoning,
          reminderEnabled: true,
          active: true,
        });
      }
    }

    console.log(`Optimized timing for ${recommendations.length} medications for patient ${patientId}`);
    return recommendations;
  } catch (error) {
    console.error("Error optimizing medication timing:", error);
    throw error;
  }
}

async function optimizeTimingWithAI(
  medications: any[],
  dietaryPrefs: any
): Promise<MedicationTimingRecommendation[]> {
  const prompt = `Analyze these medications and recommend optimal timing:

Medications:
${medications.map(m => `- ${m.name} (${m.dosage}), frequency: ${m.frequency}`).join('\n')}

Meal Schedule: ${dietaryPrefs?.mealsPerDay || 3} meals per day

Provide timing recommendations that:
1. Maximize medication effectiveness
2. Minimize side effects
3. Coordinate with meal times
4. Consider drug interactions
5. Respect the prescribed frequency

Return JSON with format:
{
  "recommendations": [
    {
      "medicationName": "string",
      "optimizedTime": "HH:MM" (24-hour format),
      "withFood": boolean,
      "reasoning": "string explaining why this timing",
      "coordinateWithMeal": "breakfast|lunch|dinner" (optional)
    }
  ]
}`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are a clinical pharmacist specializing in medication timing optimization for immunocompromised patients.",
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
      console.error("Empty response from OpenAI for medication timing");
      return [];
    }

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch (parseError) {
      console.error("Failed to parse OpenAI medication timing response as JSON:", parseError);
      console.error("Raw response:", content);
      return [];
    }

    const recommendations = parsed.recommendations || [];
    if (!Array.isArray(recommendations)) {
      console.error("AI response recommendations field is not an array:", recommendations);
      return [];
    }

    // Validate and sanitize each recommendation
    return recommendations.map((rec: any) => ({
      medicationId: "", // Will be set in the calling function
      medicationName: rec.medicationName || rec.medication || "Unknown",
      optimizedTime: rec.optimizedTime || rec.time || "09:00",
      withFood: Boolean(rec.withFood),
      reasoning: rec.reasoning || rec.explanation || "Timing optimized for effectiveness",
      coordinateWithMeal: rec.coordinateWithMeal || undefined,
    }));
  } catch (error) {
    console.error("Error calling OpenAI for medication timing:", error);
    return [];
  }
}

export async function analyzeMealNutrition(
  mealDescription: string,
  patientId: string
): Promise<{
  calories: number;
  macros: { protein: number; carbs: number; fat: number };
  immuneScore: number;
  recommendations: string[];
}> {
  const prompt = `Analyze the nutritional content of this meal for an immunocompromised patient:

Meal: ${mealDescription}

Provide detailed nutrition analysis with immune support focus.

Return JSON with format:
{
  "calories": number,
  "protein": number (grams),
  "carbs": number (grams),
  "fat": number (grams),
  "fiber": number (grams),
  "vitaminC": number (mg),
  "vitaminD": number (mcg),
  "zinc": number (mg),
  "immuneSupportScore": number (1-100),
  "recommendations": ["string array of suggestions"]
}`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are a nutritionist specializing in immune-boosting diets.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 0.5,
      response_format: { type: "json_object" },
    });

    const content = completion.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No response from AI");
    }

    const parsed = JSON.parse(content);
    return {
      calories: parsed.calories || 0,
      macros: {
        protein: parsed.protein || 0,
        carbs: parsed.carbs || 0,
        fat: parsed.fat || 0,
      },
      immuneScore: parsed.immuneSupportScore || 50,
      recommendations: parsed.recommendations || [],
    };
  } catch (error) {
    console.error("Error analyzing meal nutrition:", error);
    throw error;
  }
}

// Helper functions

function buildMealPlanPrompt(request: MealPlanRequest): string {
  return `Create a 7-day meal plan for an immunocompromised patient with the following requirements:

Diet Type: ${request.dietaryPreferences.dietType || 'balanced'}
Daily Calorie Target: ${request.calorieTarget || 2000}
Meals Per Day: ${request.dietaryPreferences.mealsPerDay || 3}
Allergies: ${request.allergies.join(', ') || 'None'}
Medications: ${request.medications.map(m => m.name).join(', ') || 'None'}

Focus Areas:
- Immune system support (high vitamin C, D, zinc, antioxidants)
- Anti-inflammatory foods
- Energy and vitality
- Easy to prepare and digest

Return JSON with format:
{
  "meals": [
    {
      "mealType": "breakfast|lunch|dinner|snack",
      "mealName": "string",
      "description": "string",
      "ingredients": [{"name": "string", "amount": "string", "unit": "string"}],
      "recipeSuggestion": "string with simple cooking instructions",
      "immuneBenefits": ["high_vitamin_c", "probiotic", "anti_inflammatory"],
      "scheduledTime": "HH:MM",
      "nutrition": {
        "calories": number,
        "protein": number,
        "carbs": number,
        "fat": number,
        "fiber": number,
        "vitaminC": number,
        "vitaminD": number,
        "zinc": number,
        "omega3": number,
        "immuneSupportScore": number (1-100)
      }
    }
  ]
}

Generate ${(request.dietaryPreferences.mealsPerDay || 3) * 7} meals for the full week.`;
}

function getStartOfCurrentWeek(): Date {
  const now = new Date();
  const dayOfWeek = now.getDay();
  const diff = now.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1);
  const monday = new Date(now.setDate(diff));
  monday.setHours(0, 0, 0, 0);
  return monday;
}

function formatDate(date: Date): string {
  const month = date.toLocaleString('default', { month: 'short' });
  const day = date.getDate();
  return `${month} ${day}`;
}

function parseScheduledTime(timeString: string): Date {
  const [hours, minutes] = timeString.split(':').map(Number);
  const scheduledTime = new Date();
  scheduledTime.setHours(hours, minutes, 0, 0);
  return scheduledTime;
}
