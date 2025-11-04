import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import {
  Calendar,
  Clock,
  Utensils,
  Pill,
  ChefHat,
  Sparkles,
  CheckCircle2,
  XCircle,
  AlertCircle,
  TrendingUp,
  Heart,
  Leaf,
  RefreshCw,
} from "lucide-react";
import { format } from "date-fns";
import { useState } from "react";

interface MealPlan {
  id: string;
  patientId: string;
  weekStartDate: Date;
  planName: string;
  calorieTarget: number | null;
  proteinTarget: number | null;
  immuneGoals: string[] | null;
  status: string;
  aiGenerated: boolean;
  createdAt: Date;
}

interface Meal {
  id: string;
  patientId: string;
  mealPlanId: string | null;
  mealType: string;
  mealName: string;
  description: string | null;
  ingredients: string[] | null;
  recipeSuggestion: string | null;
  immuneBenefits: string[] | null;
  scheduledTime: string | null;
  actualTime: Date | null;
  status: string;
  companionLogged: boolean;
  createdAt: Date;
  nutrition: any;
}

interface MedicationSchedule {
  id: string;
  medicationId: string;
  patientId: string;
  scheduledTimes: string[];
  withFood: boolean;
  coordinateWithMeal: string | null;
  aiOptimized: boolean;
  optimizationReasoning: string | null;
  createdAt: Date;
}

interface MedicationAdherence {
  id: string;
  medicationId: string;
  patientId: string;
  scheduledTime: Date;
  takenAt: Date | null;
  status: string;
  notes: string | null;
  loggedBy: string;
  companionChecked: boolean;
  createdAt: Date;
}

export default function NutritionInsights() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("overview");

  const { data: activePlan, isLoading: planLoading } = useQuery<MealPlan>({
    queryKey: ["/api/nutrition/active-plan"],
  });

  const { data: todaysMeals, isLoading: mealsLoading } = useQuery<Meal[]>({
    queryKey: ["/api/nutrition/meals/today"],
  });

  const { data: schedules, isLoading: schedulesLoading } = useQuery<MedicationSchedule[]>({
    queryKey: ["/api/medications/schedules"],
  });

  const { data: pendingMeds, isLoading: pendingLoading } = useQuery<any[]>({
    queryKey: ["/api/medications/pending"],
  });

  const { data: adherence, isLoading: adherenceLoading } = useQuery<MedicationAdherence[]>({
    queryKey: ["/api/medications/adherence"],
  });

  const generateMealPlanMutation = useMutation({
    mutationFn: async () => {
      return await apiRequest("POST", "/api/nutrition/generate-meal-plan");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/nutrition/active-plan"] });
      queryClient.invalidateQueries({ queryKey: ["/api/nutrition/meals/today"] });
      toast({
        title: "Meal Plan Generated",
        description: "Your personalized weekly meal plan is ready!",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Generation Failed",
        description: error.message || "Unable to generate meal plan. Please try again.",
        variant: "destructive",
      });
    },
  });

  const optimizeMedicationTimingMutation = useMutation({
    mutationFn: async () => {
      return await apiRequest("POST", "/api/medications/optimize-timing");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/medications/schedules"] });
      toast({
        title: "Timing Optimized",
        description: "Your medication schedule has been optimized for effectiveness.",
      });
    },
    onError: () => {
      toast({
        title: "Optimization Failed",
        description: "Unable to optimize medication timing. Please try again.",
        variant: "destructive",
      });
    },
  });

  const markMedicationTakenMutation = useMutation({
    mutationFn: async (data: { medicationId: string; scheduledTime: Date }) => {
      return await apiRequest("POST", "/api/medications/adherence", {
        medicationId: data.medicationId,
        scheduledTime: data.scheduledTime,
        takenAt: new Date(),
        status: "taken",
        loggedBy: "patient",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/medications/pending"] });
      queryClient.invalidateQueries({ queryKey: ["/api/medications/adherence"] });
      toast({
        title: "Medication Logged",
        description: "Your dose has been recorded successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Failed to Log",
        description: "Unable to record medication. Please try again.",
        variant: "destructive",
      });
    },
  });

  const getMealTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case "breakfast":
        return <Utensils className="w-5 h-5 text-orange-600 dark:text-orange-400" />;
      case "lunch":
        return <ChefHat className="w-5 h-5 text-teal-600 dark:text-teal-400" />;
      case "dinner":
        return <Utensils className="w-5 h-5 text-purple-600 dark:text-purple-400" />;
      default:
        return <Leaf className="w-5 h-5 text-green-600 dark:text-green-400" />;
    }
  };

  const getMealStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      planned: "outline",
      eaten: "default",
      skipped: "secondary",
    };

    return (
      <Badge variant={variants[status.toLowerCase()] || "outline"} data-testid={`badge-meal-status-${status}`}>
        {status}
      </Badge>
    );
  };

  const getAdherenceStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case "taken":
        return <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />;
      case "missed":
        return <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />;
      default:
        return <Clock className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />;
    }
  };

  const calculateAdherenceRate = () => {
    if (!adherence || adherence.length === 0) return 0;
    const taken = adherence.filter(a => a.status === "taken").length;
    return Math.round((taken / adherence.length) * 100);
  };

  const calculateNutritionProgress = () => {
    if (!todaysMeals || todaysMeals.length === 0 || !activePlan) return 0;
    const eatenMeals = todaysMeals.filter(m => m.status === "eaten");
    const totalCalories = eatenMeals.reduce((sum, meal) => {
      return sum + (meal.nutrition?.calories || 0);
    }, 0);
    return activePlan.calorieTarget ? Math.min(100, Math.round((totalCalories / activePlan.calorieTarget) * 100)) : 0;
  };

  return (
    <div className="container mx-auto py-8 max-w-7xl" data-testid="page-nutrition-insights">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2" data-testid="heading-nutrition-insights">
          Nutrition & Medication Insights
        </h1>
        <p className="text-muted-foreground" data-testid="text-description">
          AI-powered meal planning and medication timing optimization for immune health
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-3 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Medication Adherence</CardTitle>
            <Pill className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-adherence-rate">
              {adherenceLoading ? "..." : `${calculateAdherenceRate()}%`}
            </div>
            <Progress value={calculateAdherenceRate()} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              {adherence?.filter(a => a.status === "taken").length || 0} of {adherence?.length || 0} doses taken
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Today's Nutrition</CardTitle>
            <Heart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-nutrition-progress">
              {mealsLoading ? "..." : `${calculateNutritionProgress()}%`}
            </div>
            <Progress value={calculateNutritionProgress()} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              {todaysMeals?.filter(m => m.status === "eaten").length || 0} meals eaten today
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Meal Plan</CardTitle>
            <ChefHat className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-plan-status">
              {planLoading ? "..." : activePlan ? "Active" : "None"}
            </div>
            {activePlan && (
              <>
                <p className="text-xs text-muted-foreground mt-2">{activePlan.planName}</p>
                {activePlan.aiGenerated && (
                  <Badge variant="outline" className="mt-2">
                    <Sparkles className="w-3 h-3 mr-1" />
                    AI Generated
                  </Badge>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview" data-testid="tab-overview">
            Overview
          </TabsTrigger>
          <TabsTrigger value="meals" data-testid="tab-meals">
            Meal Planning
          </TabsTrigger>
          <TabsTrigger value="medications" data-testid="tab-medications">
            Medications
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
                <CardDescription>AI-powered optimization tools</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button
                  onClick={() => generateMealPlanMutation.mutate()}
                  disabled={generateMealPlanMutation.isPending}
                  className="w-full"
                  data-testid="button-generate-meal-plan"
                >
                  {generateMealPlanMutation.isPending ? (
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4 mr-2" />
                  )}
                  Generate Weekly Meal Plan
                </Button>
                <Button
                  onClick={() => optimizeMedicationTimingMutation.mutate()}
                  disabled={optimizeMedicationTimingMutation.isPending}
                  variant="outline"
                  className="w-full"
                  data-testid="button-optimize-timing"
                >
                  {optimizeMedicationTimingMutation.isPending ? (
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Clock className="w-4 h-4 mr-2" />
                  )}
                  Optimize Medication Timing
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Pending Medications</CardTitle>
                <CardDescription>Doses due soon</CardDescription>
              </CardHeader>
              <CardContent>
                {pendingLoading ? (
                  <p className="text-sm text-muted-foreground">Loading...</p>
                ) : pendingMeds && pendingMeds.length > 0 ? (
                  <div className="space-y-3">
                    {pendingMeds.slice(0, 3).map((med: any) => (
                      <div
                        key={med.id}
                        className="flex items-center justify-between p-3 rounded-lg border"
                        data-testid={`pending-med-${med.id}`}
                      >
                        <div className="flex items-center gap-3">
                          <Pill className="w-5 h-5 text-primary" />
                          <div>
                            <p className="font-medium">{med.medicationName}</p>
                            <p className="text-xs text-muted-foreground">
                              Due at {format(new Date(med.scheduledTime), "h:mm a")}
                            </p>
                          </div>
                        </div>
                        <Button
                          size="sm"
                          data-testid={`button-take-${med.id}`}
                          onClick={() =>
                            markMedicationTakenMutation.mutate({
                              medicationId: med.medicationId,
                              scheduledTime: new Date(med.scheduledTime),
                            })
                          }
                          disabled={markMedicationTakenMutation.isPending}
                        >
                          {markMedicationTakenMutation.isPending ? "Logging..." : "Mark Taken"}
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No pending medications</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="meals" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Today's Meals</CardTitle>
              <CardDescription>Your personalized meal plan for today</CardDescription>
            </CardHeader>
            <CardContent>
              {mealsLoading ? (
                <p className="text-sm text-muted-foreground">Loading meals...</p>
              ) : todaysMeals && todaysMeals.length > 0 ? (
                <div className="space-y-4">
                  {todaysMeals.map((meal) => (
                    <div
                      key={meal.id}
                      className="border rounded-lg p-4 hover-elevate"
                      data-testid={`meal-card-${meal.id}`}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-3">
                          {getMealTypeIcon(meal.mealType)}
                          <div>
                            <h3 className="font-semibold" data-testid={`text-meal-name-${meal.id}`}>
                              {meal.mealName}
                            </h3>
                            <p className="text-xs text-muted-foreground">
                              {meal.scheduledTime || "Flexible timing"}
                            </p>
                          </div>
                        </div>
                        {getMealStatusBadge(meal.status)}
                      </div>

                      {meal.description && (
                        <p className="text-sm text-muted-foreground mb-3">{meal.description}</p>
                      )}

                      {meal.nutrition && (
                        <div className="grid grid-cols-4 gap-2 text-center">
                          <div className="p-2 bg-muted rounded">
                            <p className="text-xs text-muted-foreground">Calories</p>
                            <p className="font-semibold">{meal.nutrition.calories}</p>
                          </div>
                          <div className="p-2 bg-muted rounded">
                            <p className="text-xs text-muted-foreground">Protein</p>
                            <p className="font-semibold">{meal.nutrition.protein}g</p>
                          </div>
                          <div className="p-2 bg-muted rounded">
                            <p className="text-xs text-muted-foreground">Carbs</p>
                            <p className="font-semibold">{meal.nutrition.carbs}g</p>
                          </div>
                          <div className="p-2 bg-muted rounded">
                            <p className="text-xs text-muted-foreground">Fat</p>
                            <p className="font-semibold">{meal.nutrition.fat}g</p>
                          </div>
                        </div>
                      )}

                      {meal.immuneBenefits && meal.immuneBenefits.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {meal.immuneBenefits.map((benefit, idx) => (
                            <Badge key={idx} variant="outline" className="text-xs">
                              <Heart className="w-3 h-3 mr-1" />
                              {benefit}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Utensils className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
                  <p className="text-muted-foreground mb-4">No meals planned for today</p>
                  <Button
                    onClick={() => generateMealPlanMutation.mutate()}
                    data-testid="button-generate-plan-empty"
                  >
                    <Sparkles className="w-4 h-4 mr-2" />
                    Generate Meal Plan
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="medications" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Medication Schedules</CardTitle>
                <CardDescription>AI-optimized timing for maximum effectiveness</CardDescription>
              </CardHeader>
              <CardContent>
                {schedulesLoading ? (
                  <p className="text-sm text-muted-foreground">Loading schedules...</p>
                ) : schedules && schedules.length > 0 ? (
                  <div className="space-y-4">
                    {schedules.map((schedule) => (
                      <div
                        key={schedule.id}
                        className="border rounded-lg p-4"
                        data-testid={`schedule-card-${schedule.id}`}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Pill className="w-4 h-4 text-primary" />
                            <p className="font-medium">Medication Schedule</p>
                          </div>
                          {schedule.aiOptimized && (
                            <Badge variant="outline">
                              <Sparkles className="w-3 h-3 mr-1" />
                              AI Optimized
                            </Badge>
                          )}
                        </div>

                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-sm">
                            <Clock className="w-4 h-4 text-muted-foreground" />
                            <span>
                              {schedule.scheduledTimes.join(", ")}
                            </span>
                          </div>

                          {schedule.withFood && (
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <Utensils className="w-4 h-4" />
                              <span>Take with food</span>
                            </div>
                          )}

                          {schedule.coordinateWithMeal && (
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <Calendar className="w-4 h-4" />
                              <span>Best with {schedule.coordinateWithMeal}</span>
                            </div>
                          )}

                          {schedule.optimizationReasoning && (
                            <div className="mt-3 p-2 bg-muted rounded text-xs">
                              <p className="font-medium mb-1">Why this timing?</p>
                              <p className="text-muted-foreground">{schedule.optimizationReasoning}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Clock className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
                    <p className="text-muted-foreground mb-4">No medication schedules yet</p>
                    <Button
                      onClick={() => optimizeMedicationTimingMutation.mutate()}
                      variant="outline"
                      data-testid="button-optimize-empty"
                    >
                      <Sparkles className="w-4 h-4 mr-2" />
                      Optimize Timing
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recent Adherence</CardTitle>
                <CardDescription>Your medication tracking history</CardDescription>
              </CardHeader>
              <CardContent>
                {adherenceLoading ? (
                  <p className="text-sm text-muted-foreground">Loading adherence...</p>
                ) : adherence && adherence.length > 0 ? (
                  <div className="space-y-3">
                    {adherence.slice(0, 5).map((record) => (
                      <div
                        key={record.id}
                        className="flex items-center justify-between p-3 rounded-lg border"
                        data-testid={`adherence-record-${record.id}`}
                      >
                        <div className="flex items-center gap-3">
                          {getAdherenceStatusIcon(record.status)}
                          <div>
                            <p className="text-sm font-medium">
                              {format(new Date(record.scheduledTime), "MMM d, h:mm a")}
                            </p>
                            {record.companionChecked && (
                              <p className="text-xs text-muted-foreground">Logged by companion</p>
                            )}
                          </div>
                        </div>
                        <Badge
                          variant={record.status === "taken" ? "default" : "secondary"}
                          data-testid={`badge-adherence-${record.id}`}
                        >
                          {record.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No adherence records yet</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
