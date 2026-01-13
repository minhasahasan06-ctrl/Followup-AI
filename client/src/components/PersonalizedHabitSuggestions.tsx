import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { 
  Sparkles, Plus, Wind, Heart, Brain, Activity, Pill, 
  Scale, CheckCircle2, Bot, RefreshCw
} from "lucide-react";

interface HabitSuggestion {
  category: string;
  habits: Array<{
    name: string;
    description: string;
    frequency: string;
    priority: string;
    evidence_based: boolean;
    condition_link?: string;
  }>;
  condition_context?: string;
}

interface PersonalizedRecommendations {
  recommendations: HabitSuggestion[];
  generated_at: string;
  source?: "agent_clona" | "personalization";
}

const CATEGORY_ICONS: Record<string, React.ElementType> = {
  respiratory: Wind,
  cardiac: Heart,
  mental_health: Brain,
  pain: Activity,
  metabolic: Scale,
  immune: Pill,
  wellness: Sparkles,
  exercise: Activity,
  nutrition: Scale,
  sleep: Brain,
};

const PRIORITY_COLORS: Record<string, string> = {
  high: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
  medium: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400",
  low: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
};

interface PersonalizedHabitSuggestionsProps {
  patientId?: string;
  onAddHabit?: (habit: { name: string; description: string; category: string; frequency: string }) => void;
}

async function fetchRecommendations(patientId: string): Promise<PersonalizedRecommendations> {
  const pid = patientId || "me";
  
  try {
    const agentRes = await apiRequest(`/api/agent-clona/recommendations?patientId=${pid}`);
    if (agentRes.ok) {
      const data = await agentRes.json();
      return { ...data, source: "agent_clona" };
    }
  } catch {
    // Agent Clona unavailable, fall through to fallback
  }
  
  const fallbackRes = await apiRequest(`/api/v1/personalization/patient/${pid}/recommendations`);
  if (!fallbackRes.ok) {
    throw new Error("Failed to fetch recommendations");
  }
  const data = await fallbackRes.json();
  return { ...data, source: "personalization" };
}

export function PersonalizedHabitSuggestions({ patientId, onAddHabit }: PersonalizedHabitSuggestionsProps) {
  const { toast } = useToast();

  const { data, isLoading, error, refetch, isFetching } = useQuery<PersonalizedRecommendations>({
    queryKey: ["/api/personalization/recommendations", patientId || "me"],
    queryFn: () => fetchRecommendations(patientId || "me"),
    staleTime: 60 * 60 * 1000,
    retry: 1,
  });

  const addHabitMutation = useMutation({
    mutationFn: async (habitData: { name: string; description: string; category: string; frequency: string }) => {
      const res = await apiRequest("/api/habits", {
        method: "POST",
        json: habitData
      });
      return await res.json();
    },
    onMutate: async (newHabit) => {
      await queryClient.cancelQueries({ queryKey: ["/api/habits"] });
      await queryClient.cancelQueries({ queryKey: ["/api/v1/ml/habits"] });
      const prevHabits = queryClient.getQueryData(["/api/habits"]);
      const prevMlHabits = queryClient.getQueryData(["/api/v1/ml/habits"]);
      
      const optimisticHabit = { ...newHabit, id: `tmp-${Date.now()}`, currentStreak: 0, totalCompletions: 0 };
      
      queryClient.setQueryData(["/api/habits"], (old: unknown[]) => 
        old ? [...old, optimisticHabit] : [optimisticHabit]
      );
      queryClient.setQueryData(["/api/v1/ml/habits"], (old: unknown[]) => 
        old ? [...old, optimisticHabit] : [optimisticHabit]
      );
      
      return { prevHabits, prevMlHabits };
    },
    onError: (_err, _newHabit, context) => {
      if (context?.prevHabits) {
        queryClient.setQueryData(["/api/habits"], context.prevHabits);
      }
      if (context?.prevMlHabits) {
        queryClient.setQueryData(["/api/v1/ml/habits"], context.prevMlHabits);
      }
      toast({
        title: "Error",
        description: "Failed to add habit",
        variant: "destructive",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/habits"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/ml/habits"] });
      toast({
        title: "Habit added",
        description: "The suggested habit has been added to your tracking list.",
      });
    },
  });

  const handleAddHabit = (habit: { name: string; description: string; frequency: string }, category: string) => {
    if (onAddHabit) {
      onAddHabit({ ...habit, category });
    } else {
      addHabitMutation.mutate({ ...habit, category });
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-64" />
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error || !data?.recommendations?.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Personalized Suggestions
          </CardTitle>
          <CardDescription>
            No personalized suggestions available yet. Complete your health profile to receive recommendations.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Personalized for You
              {data.source === "agent_clona" && (
                <Badge variant="secondary" className="text-xs gap-1">
                  <Bot className="h-3 w-3" />
                  Agent Clona
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Habit suggestions based on your health profile and conditions
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => refetch()}
            disabled={isFetching}
            data-testid="button-refresh-suggestions"
          >
            <RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {data.recommendations.map((suggestion, idx) => {
          const IconComponent = CATEGORY_ICONS[suggestion.category] || Activity;
          
          return (
            <div key={idx} className="space-y-3">
              <div className="flex items-center gap-2">
                <IconComponent className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium capitalize">{suggestion.category.replace(/_/g, " ")}</span>
                {suggestion.condition_context && (
                  <Badge variant="outline" className="text-xs">
                    {suggestion.condition_context}
                  </Badge>
                )}
              </div>
              
              <div className="space-y-2 pl-6">
                {suggestion.habits.map((habit, habitIdx) => (
                  <div 
                    key={habitIdx}
                    className="flex items-start justify-between gap-3 p-3 border rounded-lg"
                    data-testid={`suggestion-${suggestion.category}-${habitIdx}`}
                  >
                    <div className="space-y-1 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium">{habit.name}</span>
                        <Badge className={PRIORITY_COLORS[habit.priority] || PRIORITY_COLORS.medium}>
                          {habit.priority}
                        </Badge>
                        {habit.evidence_based && (
                          <Badge variant="secondary" className="text-xs gap-1">
                            <CheckCircle2 className="h-3 w-3" />
                            Evidence-based
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">{habit.description}</p>
                      <p className="text-xs text-muted-foreground">
                        Recommended: {habit.frequency}
                      </p>
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleAddHabit(habit, suggestion.category)}
                      disabled={addHabitMutation.isPending}
                      data-testid={`button-add-habit-${habitIdx}`}
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

export default PersonalizedHabitSuggestions;
