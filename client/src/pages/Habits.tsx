import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Plus, Target, Flame, Trophy, Star, TrendingUp, CheckCircle2, Circle, Sparkles, Heart, Pill, Activity, Wind, Salad, Moon, Sparkle, Smile, Meh, Frown, SmilePlus } from "lucide-react";
import { Progress } from "@/components/ui/progress";

const createHabitSchema = z.object({
  name: z.string().min(1, "Name is required").max(100),
  description: z.string().max(500).optional(),
  category: z.enum(['health', 'medication', 'exercise', 'wellness', 'nutrition', 'sleep', 'other']),
  frequency: z.enum(['daily', 'weekly', 'custom']),
  goalCount: z.coerce.number().int().min(1).max(10).optional(),
});

const completeHabitSchema = z.object({
  mood: z.enum(['great', 'good', 'okay', 'struggling']).optional(),
  notes: z.string().max(500).optional(),
  difficultyLevel: z.coerce.number().int().min(1).max(5).optional(),
});

type Habit = {
  id: string;
  name: string;
  description?: string;
  category: string;
  frequency: string;
  currentStreak: number;
  longestStreak: number;
  totalCompletions: number;
  goalCount: number;
};

type Milestone = {
  id: string;
  type: string;
  title: string;
  description: string;
  achievedAt: string;
};

type Recommendation = {
  id: string;
  type: string;
  category: string;
  title: string;
  description: string;
  confidenceScore: string;
  priority: string;
};

export default function Habits() {
  const { toast } = useToast();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [selectedHabitForCompletion, setSelectedHabitForCompletion] = useState<Habit | null>(null);

  // Fetch user habits
  const { data: habits = [], isLoading: habitsLoading } = useQuery<Habit[]>({
    queryKey: ['/api/v1/ml/habits'],
  });

  // Fetch milestones
  const { data: milestones = [] } = useQuery<Milestone[]>({
    queryKey: ['/api/v1/ml/milestones'],
  });

  // Fetch recommendations
  const { data: recommendations = [] } = useQuery<Recommendation[]>({
    queryKey: ['/api/v1/ml/recommendations'],
  });

  // Create habit mutation
  const createHabitForm = useForm({
    resolver: zodResolver(createHabitSchema),
    defaultValues: {
      name: "",
      description: "",
      category: "health" as const,
      frequency: "daily" as const,
      goalCount: 1,
    },
  });

  const createHabitMutation = useMutation({
    mutationFn: async (data: z.infer<typeof createHabitSchema>) => {
      return await apiRequest('POST', '/api/v1/ml/habits', data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/ml/habits'] });
      setIsCreateDialogOpen(false);
      createHabitForm.reset();
      toast({
        title: "Habit created!",
        description: "Let's build this healthy habit together!",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error creating habit",
        description: error.message || "Please try again",
        variant: "destructive",
      });
    },
  });

  // Complete habit mutation
  const completeHabitForm = useForm({
    resolver: zodResolver(completeHabitSchema),
    defaultValues: {
      mood: "good" as const,
      notes: "",
      difficultyLevel: 3,
    },
  });

  const completeHabitMutation = useMutation({
    mutationFn: async ({ habitId, data }: { habitId: string; data: z.infer<typeof completeHabitSchema> }) => {
      return await apiRequest('POST', `/api/v1/ml/habits/${habitId}/complete`, data);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/ml/habits'] });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/ml/milestones'] });
      setSelectedHabitForCompletion(null);
      completeHabitForm.reset();
      
      if (data.newStreak && data.newStreak > 1) {
        toast({
          title: `${data.newStreak} Day Streak!`,
          description: "Keep up the amazing work!",
        });
      } else {
        toast({
          title: "Habit completed!",
          description: "Another step toward better health!",
        });
      }
    },
    onError: (error: any) => {
      toast({
        title: "Error completing habit",
        description: error.message || "Please try again",
        variant: "destructive",
      });
    },
  });

  const handleCompleteHabit = (habit: Habit) => {
    setSelectedHabitForCompletion(habit);
  };

  const getMoodColor = (mood?: string) => {
    switch (mood) {
      case 'great': return 'bg-green-500';
      case 'good': return 'bg-blue-500';
      case 'okay': return 'bg-yellow-500';
      case 'struggling': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getCategoryIcon = (category: string) => {
    const iconClass = "w-5 h-5";
    switch (category) {
      case 'health': return <Heart className={iconClass} />;
      case 'medication': return <Pill className={iconClass} />;
      case 'exercise': return <Activity className={iconClass} />;
      case 'wellness': return <Wind className={iconClass} />;
      case 'nutrition': return <Salad className={iconClass} />;
      case 'sleep': return <Moon className={iconClass} />;
      default: return <Target className={iconClass} />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold" data-testid="text-habits-title">Habit Tracker</h1>
          <p className="text-muted-foreground mt-1">
            Build healthy habits with Agent Clona's personalized support
          </p>
        </div>
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-create-habit">
              <Plus className="w-4 h-4 mr-2" />
              New Habit
            </Button>
          </DialogTrigger>
          <DialogContent data-testid="dialog-create-habit">
            <DialogHeader>
              <DialogTitle>Create New Habit</DialogTitle>
              <DialogDescription>
                Start tracking a new healthy habit with Agent Clona's support
              </DialogDescription>
            </DialogHeader>
            <Form {...createHabitForm}>
              <form onSubmit={createHabitForm.handleSubmit((data) => createHabitMutation.mutate(data))} className="space-y-4">
                <FormField
                  control={createHabitForm.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Habit Name</FormLabel>
                      <FormControl>
                        <Input placeholder="Take morning medication" data-testid="input-habit-name" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={createHabitForm.control}
                  name="description"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Description (Optional)</FormLabel>
                      <FormControl>
                        <Textarea placeholder="Details about this habit" data-testid="input-habit-description" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={createHabitForm.control}
                  name="category"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Category</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger data-testid="select-habit-category">
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="health">Health</SelectItem>
                          <SelectItem value="medication">Medication</SelectItem>
                          <SelectItem value="exercise">Exercise</SelectItem>
                          <SelectItem value="wellness">Wellness</SelectItem>
                          <SelectItem value="nutrition">Nutrition</SelectItem>
                          <SelectItem value="sleep">Sleep</SelectItem>
                          <SelectItem value="other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={createHabitForm.control}
                  name="frequency"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Frequency</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger data-testid="select-habit-frequency">
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="daily">Daily</SelectItem>
                          <SelectItem value="weekly">Weekly</SelectItem>
                          <SelectItem value="custom">Custom</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <div className="flex justify-end gap-2 pt-4">
                  <Button type="button" variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button type="submit" disabled={createHabitMutation.isPending} data-testid="button-submit-habit">
                    {createHabitMutation.isPending ? "Creating..." : "Create Habit"}
                  </Button>
                </div>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Milestones Section */}
      {milestones.length > 0 && (
        <Card className="border-primary bg-gradient-to-br from-primary/5 to-transparent">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Trophy className="w-5 h-5 text-primary" />
              <CardTitle className="text-lg">Recent Milestones</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2">
              {milestones.slice(0, 3).map((milestone) => (
                <div key={milestone.id} className="flex items-center gap-3 p-3 bg-background rounded-lg border" data-testid={`milestone-${milestone.id}`}>
                  <Star className="w-5 h-5 text-yellow-500 fill-yellow-500" />
                  <div className="flex-1">
                    <p className="font-medium">{milestone.title}</p>
                    <p className="text-sm text-muted-foreground">{milestone.description}</p>
                  </div>
                  <Badge variant="secondary">{new Date(milestone.achievedAt).toLocaleDateString()}</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Active Habits */}
      <div className="grid gap-4">
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5" />
          <h2 className="text-xl font-semibold">Your Habits</h2>
        </div>
        
        {habitsLoading ? (
          <Card>
            <CardContent className="py-12">
              <div className="text-center text-muted-foreground">Loading habits...</div>
            </CardContent>
          </Card>
        ) : habits.length === 0 ? (
          <Card>
            <CardContent className="py-12">
              <div className="text-center">
                <Target className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">No habits yet</h3>
                <p className="text-muted-foreground mb-4">
                  Create your first habit to start building healthier routines!
                </p>
                <Button onClick={() => setIsCreateDialogOpen(true)} data-testid="button-create-first-habit">
                  <Plus className="w-4 h-4 mr-2" />
                  Create Your First Habit
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {habits.map((habit) => (
              <Card key={habit.id} className="hover-elevate" data-testid={`habit-card-${habit.id}`}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <div className="p-2 rounded-lg bg-primary/10 text-primary">
                        {getCategoryIcon(habit.category)}
                      </div>
                      <div>
                        <CardTitle className="text-lg">{habit.name}</CardTitle>
                        <CardDescription className="capitalize">{habit.category}</CardDescription>
                      </div>
                    </div>
                    {habit.currentStreak > 0 && (
                      <div className="flex items-center gap-1 text-orange-500">
                        <Flame className="w-4 h-4" />
                        <span className="font-bold">{habit.currentStreak}</span>
                      </div>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {habit.description && (
                    <p className="text-sm text-muted-foreground">{habit.description}</p>
                  )}
                  
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-2xl font-bold text-orange-500">{habit.currentStreak}</div>
                      <div className="text-xs text-muted-foreground">Current Streak</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-blue-500">{habit.longestStreak}</div>
                      <div className="text-xs text-muted-foreground">Best Streak</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-green-500">{habit.totalCompletions}</div>
                      <div className="text-xs text-muted-foreground">Total</div>
                    </div>
                  </div>

                  <Button 
                    className="w-full" 
                    onClick={() => handleCompleteHabit(habit)}
                    data-testid={`button-complete-${habit.id}`}
                  >
                    <CheckCircle2 className="w-4 h-4 mr-2" />
                    Complete Today
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Agent Clona Recommendations */}
      {recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-primary" />
              <CardTitle>Personalized Recommendations from Agent Clona</CardTitle>
            </div>
            <CardDescription>
              AI-powered suggestions based on your progress and preferences
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3">
              {recommendations.slice(0, 4).map((rec) => (
                <div key={rec.id} className="p-4 border rounded-lg hover-elevate" data-testid={`recommendation-${rec.id}`}>
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium">{rec.title}</h4>
                        <Badge variant={rec.priority === 'high' ? 'default' : 'secondary'} className="text-xs">
                          {rec.priority}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{rec.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <TrendingUp className="w-3 h-3 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">
                      {Math.round(parseFloat(rec.confidenceScore) * 100)}% confidence
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Complete Habit Dialog */}
      <Dialog open={!!selectedHabitForCompletion} onOpenChange={(open) => !open && setSelectedHabitForCompletion(null)}>
        <DialogContent data-testid="dialog-complete-habit">
          <DialogHeader>
            <DialogTitle>Complete Habit</DialogTitle>
            <DialogDescription>
              {selectedHabitForCompletion?.name}
            </DialogDescription>
          </DialogHeader>
          <Form {...completeHabitForm}>
            <form onSubmit={completeHabitForm.handleSubmit((data) => {
              if (selectedHabitForCompletion) {
                completeHabitMutation.mutate({ habitId: selectedHabitForCompletion.id, data });
              }
            })} className="space-y-4">
              <FormField
                control={completeHabitForm.control}
                name="mood"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>How are you feeling?</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger data-testid="select-mood">
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="great">Great</SelectItem>
                        <SelectItem value="good">Good</SelectItem>
                        <SelectItem value="okay">Okay</SelectItem>
                        <SelectItem value="struggling">Struggling</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={completeHabitForm.control}
                name="notes"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Notes (Optional)</FormLabel>
                    <FormControl>
                      <Textarea placeholder="How did it go?" data-testid="input-completion-notes" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={completeHabitForm.control}
                name="difficultyLevel"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Difficulty Level (1-5)</FormLabel>
                    <FormControl>
                      <Input 
                        type="number" 
                        min="1" 
                        max="5" 
                        data-testid="input-difficulty-level"
                        {...field}
                        onChange={(e) => field.onChange(parseInt(e.target.value))}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <div className="flex justify-end gap-2 pt-4">
                <Button type="button" variant="outline" onClick={() => setSelectedHabitForCompletion(null)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={completeHabitMutation.isPending} data-testid="button-submit-completion">
                  {completeHabitMutation.isPending ? "Saving..." : "Complete"}
                </Button>
              </div>
            </form>
          </Form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
