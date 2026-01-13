import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { HabitGamificationPanel } from "@/components/habits/HabitGamificationPanel";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage, FormDescription } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { 
  Plus, Target, Flame, Trophy, Star, TrendingUp, CheckCircle2, Circle, 
  Sparkles, Heart, Pill, Activity, Wind, Salad, Moon, Sparkle, Smile, 
  Meh, Frown, SmilePlus, Calendar, Clock, Bell, Brain, MessageSquare,
  Users, Award, BookOpen, AlertTriangle, Zap, Flower2, TreeDeciduous,
  Send, RefreshCw, ChevronRight, ChevronDown, X, BarChart3, LineChart,
  Shield, AlertCircle, HelpCircle, Loader2, Wand2
} from "lucide-react";
import { useHabitSuggestions, type HabitSuggestion } from "@/hooks/usePatientAI";
import { HabitSuggestionsModal } from "@/components/ai/HabitSuggestionsModal";
import { useAuth } from "@/hooks/useAuth";
import { Progress } from "@/components/ui/progress";
import { format, startOfMonth, endOfMonth, eachDayOfInterval, isSameDay, parseISO, isToday } from "date-fns";

// Schemas
const createHabitSchema = z.object({
  name: z.string().min(1, "Name is required").max(100),
  description: z.string().max(500).optional(),
  category: z.enum(['health', 'medication', 'exercise', 'wellness', 'nutrition', 'sleep', 'other']),
  frequency: z.enum(['daily', 'weekly', 'custom']),
  goalCount: z.coerce.number().int().min(1).max(10).optional(),
  reminderEnabled: z.boolean().optional(),
  reminderTime: z.string().optional(),
});

const completeHabitSchema = z.object({
  mood: z.enum(['great', 'good', 'okay', 'struggling']).optional(),
  notes: z.string().max(500).optional(),
  difficultyLevel: z.coerce.number().int().min(1).max(5).optional(),
});

const moodEntrySchema = z.object({
  moodScore: z.number().min(1).max(10),
  moodLabel: z.string().optional(),
  energyLevel: z.number().min(1).max(10).optional(),
  stressLevel: z.number().min(1).max(10).optional(),
  journalText: z.string().max(2000).optional(),
});

const quitPlanSchema = z.object({
  habitName: z.string().min(1, "Name is required"),
  category: z.enum(['substance', 'behavioral', 'food', 'other']).optional(),
  quitMethod: z.enum(['cold_turkey', 'gradual_reduction', 'replacement']).optional(),
  dailyLimit: z.coerce.number().optional(),
  reasonsToQuit: z.string().optional(),
  moneySavedPerDay: z.coerce.number().optional(),
});

const journalSchema = z.object({
  title: z.string().optional(),
  content: z.string().min(1, "Journal content is required"),
  entryType: z.enum(['daily', 'reflection', 'gratitude', 'goal_setting']).optional(),
  mood: z.string().optional(),
});

// Types
type Habit = {
  id: string;
  name: string;
  description?: string;
  category: string;
  frequency: string;
  currentStreak: number;
  longestStreak: number;
  totalCompletions: number;
  goalCount?: number;
  reminderEnabled?: boolean;
  reminderTime?: string;
  isActive?: boolean;
  createdAt?: string;
};

type CalendarDay = {
  completions: Array<{
    habitId: string;
    habitName: string;
    completed: boolean;
    mood?: string;
  }>;
  totalCompleted: number;
  totalSkipped: number;
};

type QuitPlan = {
  id: string;
  habitName: string;
  category?: string;
  quitMethod?: string;
  daysClean: number;
  longestStreak: number;
  totalRelapses: number;
  status: string;
  reasonsToQuit?: string[];
  moneySavedPerDay?: number;
  startDate?: string;
};

type MoodEntry = {
  id: string;
  moodScore: number;
  moodLabel?: string;
  energyLevel?: number;
  stressLevel?: number;
  journalText?: string;
  sentimentScore?: number;
  recordedAt: string;
};

type JournalEntry = {
  id: string;
  title?: string;
  content: string;
  entryType?: string;
  aiSummary?: string;
  highlights?: string[];
  risks?: string[];
  recommendations?: string[];
  sentimentTrend?: string;
  mood?: string;
  recordedAt: string;
  isWeeklySummary?: boolean;
};

type RiskAlert = {
  id: string;
  alertType: string;
  severity: string;
  title: string;
  message: string;
  riskScore: number;
  contributingFactors?: Array<{ factor: string; weight: number; value: string }>;
  suggestedActions?: string[];
  status: string;
  createdAt: string;
};

type AiTrigger = {
  id: string;
  habitId?: string;
  habitName?: string;
  triggerType: string;
  pattern: string;
  correlatedFactor?: string;
  confidence: number;
  dataPoints: number;
};

type RewardState = {
  rewardType: string;
  currentLevel: number;
  growthStage: string;
  totalPoints: number;
  streakBonus: number;
  completionPoints: number;
  visualState?: { height?: number; leaves?: number; flowers?: number };
  unlockedBadges?: string[];
  daysActive: number;
  perfectDays: number;
  nextStage?: string;
  pointsToNextStage?: number;
};

type CoachMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  responseType?: string;
  createdAt: string;
};

// Helper functions
const getCategoryIcon = (category: string) => {
  const icons: Record<string, any> = {
    health: Heart,
    medication: Pill,
    exercise: Activity,
    wellness: Wind,
    nutrition: Salad,
    sleep: Moon,
    other: Sparkle,
  };
  return icons[category] || Sparkle;
};

const getMoodEmoji = (mood: string) => {
  const moods: Record<string, { icon: any; color: string }> = {
    great: { icon: SmilePlus, color: "text-green-500" },
    good: { icon: Smile, color: "text-emerald-500" },
    okay: { icon: Meh, color: "text-yellow-500" },
    struggling: { icon: Frown, color: "text-red-500" },
  };
  return moods[mood] || moods.okay;
};

const getGrowthStageVisual = (stage: string, points: number) => {
  const stages: Record<string, { emoji: string; description: string; color: string }> = {
    seed: { emoji: "🌱", description: "Seed", color: "text-amber-600" },
    sprout: { emoji: "🌿", description: "Sprout", color: "text-green-500" },
    growing: { emoji: "🌳", description: "Growing", color: "text-green-600" },
    blooming: { emoji: "🌸", description: "Blooming", color: "text-pink-500" },
    flourishing: { emoji: "🌻", description: "Flourishing", color: "text-yellow-500" },
  };
  return stages[stage] || stages.seed;
};

export default function Habits() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState("overview");
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [selectedHabitForCompletion, setSelectedHabitForCompletion] = useState<Habit | null>(null);
  const [isQuitPlanDialogOpen, setIsQuitPlanDialogOpen] = useState(false);
  const [isJournalDialogOpen, setIsJournalDialogOpen] = useState(false);
  const [isMoodDialogOpen, setIsMoodDialogOpen] = useState(false);
  const [isAISuggestionsOpen, setIsAISuggestionsOpen] = useState(false);
  const [coachMessage, setCoachMessage] = useState("");
  const [coachMessages, setCoachMessages] = useState<CoachMessage[]>([]);
  const [coachPersonality, setCoachPersonality] = useState<"supportive" | "motivational" | "analytical" | "tough_love" | "mindful">("supportive");
  const [coachSessionId, setCoachSessionId] = useState<string | null>(null);
  const [selectedMonth, setSelectedMonth] = useState(new Date());
  const [activeCbtSession, setActiveCbtSession] = useState<any>(null);
  const [cbtResponse, setCbtResponse] = useState("");
  
  const habitSuggestionsMutation = useHabitSuggestions(user?.id || '');
  
  const handleOpenAISuggestions = async () => {
    if (!user?.id) {
      toast({
        title: 'Sign in required',
        description: 'Please sign in to get AI-powered habit suggestions.',
        variant: 'destructive',
      });
      return;
    }
    setIsAISuggestionsOpen(true);
    try {
      await habitSuggestionsMutation.mutateAsync({});
    } catch {
      toast({
        title: 'Error',
        description: 'Failed to load AI suggestions. Please try again.',
        variant: 'destructive',
      });
    }
  };
  
  const handleAddSuggestedHabit = async (habit: HabitSuggestion) => {
    try {
      await apiRequest('/api/v1/ml/habits', {
        method: 'POST',
        body: JSON.stringify({
          name: habit.name,
          description: habit.description,
          category: habit.category,
          frequency: habit.frequency,
        }),
      });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/ml/habits'] });
      toast({
        title: 'Habit added',
        description: `"${habit.name}" has been added to your habits.`,
      });
    } catch {
      toast({
        title: 'Error',
        description: 'Failed to add habit. Please try again.',
        variant: 'destructive',
      });
      throw new Error('Failed to add habit');
    }
  };

  // Fetch habits from Express backend
  const { data: habits = [], isLoading: habitsLoading, error: habitsError } = useQuery<Habit[]>({
    queryKey: ['/api/v1/ml/habits'],
    retry: 1,
  });

  // Fetch calendar data
  const { data: calendarData } = useQuery<{ calendar: Record<string, CalendarDay> }>({
    queryKey: ['/api/habits/calendar', selectedMonth.getFullYear(), selectedMonth.getMonth() + 1],
    queryFn: async () => {
      const res = await fetch(
        `/api/habits/calendar?user_id=current&year=${selectedMonth.getFullYear()}&month=${selectedMonth.getMonth() + 1}`
      );
      if (!res.ok) return { calendar: {} };
      return res.json();
    },
    retry: 1,
  });

  // Fetch streaks summary
  const { data: streaksData } = useQuery({
    queryKey: ['/api/habits/streaks'],
    queryFn: async () => {
      const res = await fetch('/api/habits/streaks?user_id=current');
      if (!res.ok) return { habits: [], summary: {} };
      return res.json();
    },
    retry: 1,
  });

  // Fetch quit plans
  const { data: quitPlansData } = useQuery<{ quitPlans: QuitPlan[] }>({
    queryKey: ['/api/habits/quit-plans'],
    queryFn: async () => {
      const res = await fetch('/api/habits/quit-plans?user_id=current');
      if (!res.ok) return { quitPlans: [] };
      return res.json();
    },
    retry: 1,
  });

  // Fetch mood trends
  const { data: moodTrends } = useQuery({
    queryKey: ['/api/habits/mood/trends'],
    queryFn: async () => {
      const res = await fetch('/api/habits/mood/trends?user_id=current&days=30');
      if (!res.ok) return { trends: [] };
      return res.json();
    },
    retry: 1,
  });

  // Fetch AI triggers
  const { data: triggersData } = useQuery<{ triggers: AiTrigger[] }>({
    queryKey: ['/api/habits/triggers'],
    queryFn: async () => {
      const res = await fetch('/api/habits/triggers?user_id=current');
      if (!res.ok) return { triggers: [] };
      return res.json();
    },
    retry: 1,
  });

  // Fetch rewards state
  const { data: rewardsData } = useQuery<RewardState>({
    queryKey: ['/api/habits/rewards'],
    queryFn: async () => {
      const res = await fetch('/api/habits/rewards?user_id=current');
      if (!res.ok) return null;
      return res.json();
    },
    retry: 1,
  });

  // Fetch risk alerts
  const { data: alertsData } = useQuery<{ alerts: RiskAlert[] }>({
    queryKey: ['/api/habits/alerts'],
    queryFn: async () => {
      const res = await fetch('/api/habits/alerts?user_id=current&status=active');
      if (!res.ok) return { alerts: [] };
      return res.json();
    },
    retry: 1,
  });

  // Fetch journals
  const { data: journalsData } = useQuery<{ journals: JournalEntry[] }>({
    queryKey: ['/api/habits/journals'],
    queryFn: async () => {
      const res = await fetch('/api/habits/journals?user_id=current&limit=10');
      if (!res.ok) return { journals: [] };
      return res.json();
    },
    retry: 1,
  });

  // Fetch CBT flows
  const { data: cbtFlows } = useQuery({
    queryKey: ['/api/habits/cbt/flows'],
    queryFn: async () => {
      const res = await fetch('/api/habits/cbt/flows');
      if (!res.ok) return { flows: [] };
      return res.json();
    },
    retry: 1,
  });

  // Fetch AI recommendations
  const { data: recommendationsData } = useQuery({
    queryKey: ['/api/habits/recommendations'],
    queryFn: async () => {
      const res = await fetch('/api/habits/recommendations?user_id=current&status=pending');
      if (!res.ok) return { recommendations: [] };
      return res.json();
    },
    retry: 1,
  });

  // Create habit form
  const createHabitForm = useForm({
    resolver: zodResolver(createHabitSchema),
    defaultValues: {
      name: "",
      description: "",
      category: "health" as const,
      frequency: "daily" as const,
      goalCount: 1,
      reminderEnabled: true,
      reminderTime: "09:00",
    },
  });

  // Create habit mutation
  const createHabitMutation = useMutation({
    mutationFn: async (data: z.infer<typeof createHabitSchema>) => {
      const res = await apiRequest('/api/v1/ml/habits', { method: 'POST', json: data });
      return await res.json();
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

  // Complete habit form
  const completeHabitForm = useForm({
    resolver: zodResolver(completeHabitSchema),
    defaultValues: {
      mood: "good" as const,
      notes: "",
      difficultyLevel: 3,
    },
  });

  // Complete habit mutation
  const completeHabitMutation = useMutation({
    mutationFn: async ({ habitId, data }: { habitId: string; data: any }) => {
      const res = await apiRequest(`/api/v1/ml/habits/${habitId}/complete`, { method: 'POST', json: data });
      return await res.json();
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/ml/habits'] });
      queryClient.invalidateQueries({ queryKey: ['/api/habits/streaks'] });
      queryClient.invalidateQueries({ queryKey: ['/api/habits/rewards'] });
      setSelectedHabitForCompletion(null);
      completeHabitForm.reset();
      toast({
        title: "Great job!",
        description: "Keep building that streak!",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error completing habit",
        description: error.message || "Please try again",
        variant: "destructive",
      });
    },
  });

  // Mood entry form
  const moodForm = useForm({
    resolver: zodResolver(moodEntrySchema),
    defaultValues: {
      moodScore: 5,
      energyLevel: 5,
      stressLevel: 5,
      journalText: "",
    },
  });

  // Log mood mutation
  const logMoodMutation = useMutation({
    mutationFn: async (data: z.infer<typeof moodEntrySchema>) => {
      const res = await fetch('/api/habits/mood/log?user_id=current', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error('Failed to log mood');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/habits/mood/trends'] });
      setIsMoodDialogOpen(false);
      moodForm.reset();
      toast({ title: "Mood logged!", description: "Keep tracking for better insights." });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to log mood", variant: "destructive" });
    },
  });

  // Quit plan form
  const quitPlanForm = useForm({
    resolver: zodResolver(quitPlanSchema),
    defaultValues: {
      habitName: "",
      category: "behavioral" as const,
      quitMethod: "gradual_reduction" as const,
      dailyLimit: 5,
      reasonsToQuit: "",
      moneySavedPerDay: 0,
    },
  });

  // Create quit plan mutation
  const createQuitPlanMutation = useMutation({
    mutationFn: async (data: z.infer<typeof quitPlanSchema>) => {
      const payload = {
        ...data,
        reasons_to_quit: data.reasonsToQuit ? data.reasonsToQuit.split(',').map(r => r.trim()) : [],
        money_saved_per_day: data.moneySavedPerDay,
        daily_limit: data.dailyLimit,
        quit_method: data.quitMethod,
        habit_name: data.habitName,
      };
      const res = await fetch('/api/habits/quit-plans/create?user_id=current', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error('Failed to create quit plan');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/habits/quit-plans'] });
      setIsQuitPlanDialogOpen(false);
      quitPlanForm.reset();
      toast({ title: "Quit plan created!", description: "You've taken the first step. Stay strong!" });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to create quit plan", variant: "destructive" });
    },
  });

  // Journal form
  const journalForm = useForm({
    resolver: zodResolver(journalSchema),
    defaultValues: {
      title: "",
      content: "",
      entryType: "daily" as const,
      mood: "",
    },
  });

  // Create journal mutation
  const createJournalMutation = useMutation({
    mutationFn: async (data: z.infer<typeof journalSchema>) => {
      const res = await fetch('/api/habits/journals/create?user_id=current', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error('Failed to create journal');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/habits/journals'] });
      setIsJournalDialogOpen(false);
      journalForm.reset();
      toast({ title: "Journal saved!", description: "Reflecting is key to growth." });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to save journal", variant: "destructive" });
    },
  });

  // Coach chat mutation - using enhanced endpoint with personality and session memory
  const coachChatMutation = useMutation({
    mutationFn: async (message: string) => {
      const res = await fetch('/api/habits/coach/enhanced-chat?user_id=current', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message, 
          personality: coachPersonality,
          session_id: coachSessionId 
        }),
      });
      if (!res.ok) throw new Error('Failed to send message');
      return res.json();
    },
    onSuccess: (data) => {
      if (data.session_id && !coachSessionId) {
        setCoachSessionId(data.session_id);
      }
      setCoachMessages(prev => [
        ...prev,
        { id: Date.now().toString(), role: 'user', content: coachMessage, createdAt: new Date().toISOString() },
        { id: (Date.now() + 1).toString(), role: 'assistant', content: data.response, responseType: data.response_type, createdAt: new Date().toISOString() },
      ]);
      setCoachMessage("");
    },
    onError: () => {
      toast({ title: "Error", description: "Coach unavailable. Try again later.", variant: "destructive" });
    },
  });

  // Start CBT session mutation
  const startCbtMutation = useMutation({
    mutationFn: async (sessionType: string) => {
      const res = await fetch('/api/habits/cbt/start?user_id=current', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_type: sessionType }),
      });
      if (!res.ok) throw new Error('Failed to start session');
      return res.json();
    },
    onSuccess: (data) => {
      setActiveCbtSession(data);
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to start CBT session", variant: "destructive" });
    },
  });

  // CBT step response mutation
  const cbtRespondMutation = useMutation({
    mutationFn: async ({ sessionId, stepNumber, response }: { sessionId: string; stepNumber: number; response: string }) => {
      const res = await fetch('/api/habits/cbt/respond?user_id=current', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, step_number: stepNumber, response }),
      });
      if (!res.ok) throw new Error('Failed to submit response');
      return res.json();
    },
    onSuccess: (data) => {
      if (data.completed) {
        setActiveCbtSession(null);
        toast({ title: "Session Complete!", description: "Great work on this exercise." });
      } else {
        setActiveCbtSession((prev: any) => ({
          ...prev,
          current_step: data.current_step,
          current_prompt: data.next_prompt,
        }));
      }
      setCbtResponse("");
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to submit response", variant: "destructive" });
    },
  });

  // Analyze triggers mutation
  const analyzeTriggersMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch('/api/habits/triggers/analyze?user_id=current', {
        method: 'POST',
      });
      if (!res.ok) throw new Error('Failed to analyze');
      return res.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/habits/triggers'] });
      toast({ 
        title: "Analysis Complete", 
        description: `Found ${data.triggers?.length || 0} patterns in your habits.` 
      });
    },
    onError: () => {
      toast({ title: "Error", description: "Need more data for analysis", variant: "destructive" });
    },
  });

  // Generate recommendations mutation
  const generateRecommendationsMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch('/api/habits/recommendations/generate?user_id=current', {
        method: 'POST',
      });
      if (!res.ok) throw new Error('Failed to generate');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/habits/recommendations'] });
      toast({ title: "Recommendations Generated", description: "Check the insights tab for personalized tips." });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to generate recommendations", variant: "destructive" });
    },
  });

  // Generate weekly summary mutation
  const generateWeeklySummaryMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch('/api/habits/journals/weekly-summary?user_id=current', {
        method: 'POST',
      });
      if (!res.ok) throw new Error('Failed to generate summary');
      return res.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/habits/journals'] });
      toast({ 
        title: "Weekly Summary Generated", 
        description: data.summary || "Check journals for your AI-powered reflection." 
      });
    },
    onError: () => {
      toast({ title: "Error", description: "Need more journal entries for a summary", variant: "destructive" });
    },
  });

  // Render Calendar View
  const renderCalendar = () => {
    const monthStart = startOfMonth(selectedMonth);
    const monthEnd = endOfMonth(selectedMonth);
    const days = eachDayOfInterval({ start: monthStart, end: monthEnd });
    const calendar = calendarData?.calendar || {};

    return (
      <div className="grid grid-cols-7 gap-1">
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
          <div key={day} className="text-center text-xs font-medium text-muted-foreground py-2">
            {day}
          </div>
        ))}
        {days.map(day => {
          const dateKey = format(day, 'yyyy-MM-dd');
          const dayData = calendar[dateKey];
          const completed = dayData?.totalCompleted || 0;
          const total = completed + (dayData?.totalSkipped || 0);
          const completionRate = total > 0 ? completed / total : 0;
          
          let bgColor = "bg-muted/30";
          if (completionRate === 1 && total > 0) bgColor = "bg-green-500/30";
          else if (completionRate >= 0.5) bgColor = "bg-yellow-500/30";
          else if (total > 0) bgColor = "bg-red-500/30";

          return (
            <div
              key={dateKey}
              className={`aspect-square rounded-md p-1 text-center text-sm ${bgColor} ${
                isToday(day) ? 'ring-2 ring-primary' : ''
              }`}
            >
              <div className="font-medium">{format(day, 'd')}</div>
              {total > 0 && (
                <div className="text-xs text-muted-foreground">
                  {completed}/{total}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  // Render Rewards Visualization
  const renderRewardsVisualization = () => {
    const rewards = rewardsData;
    if (!rewards) return null;

    const stageInfo = getGrowthStageVisual(rewards.growthStage, rewards.totalPoints);
    const progressToNext = rewards.pointsToNextStage 
      ? Math.max(0, 100 - (rewards.pointsToNextStage / (rewards.totalPoints + rewards.pointsToNextStage) * 100))
      : 100;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Award className="h-5 w-5 text-primary" />
            Your Growth Journey
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-center">
            <div className="text-center">
              <div className="text-8xl mb-2">{stageInfo.emoji}</div>
              <Badge className={stageInfo.color}>{stageInfo.description}</Badge>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-primary">{rewards.totalPoints}</div>
              <div className="text-xs text-muted-foreground">Total Points</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">{rewards.daysActive}</div>
              <div className="text-xs text-muted-foreground">Days Active</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-amber-600">{rewards.perfectDays}</div>
              <div className="text-xs text-muted-foreground">Perfect Days</div>
            </div>
          </div>

          {rewards.nextStage && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress to {rewards.nextStage}</span>
                <span>{rewards.pointsToNextStage} pts to go</span>
              </div>
              <Progress value={progressToNext} className="h-2" />
            </div>
          )}

          {rewards.unlockedBadges && rewards.unlockedBadges.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Unlocked Badges</div>
              <div className="flex flex-wrap gap-2">
                {rewards.unlockedBadges.map((badge, i) => (
                  <Badge key={i} variant="outline">{badge}</Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  if (habitsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2">Loading your habits...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Target className="h-6 w-6 text-primary" />
            Habit Tracker
          </h1>
          <p className="text-muted-foreground">Build healthy habits with AI-powered insights</p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <Button 
            onClick={handleOpenAISuggestions} 
            variant="outline" 
            size="sm"
            disabled={habitSuggestionsMutation.isPending}
            data-testid="button-ai-optimize"
          >
            {habitSuggestionsMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
            ) : (
              <Wand2 className="h-4 w-4 mr-1" />
            )}
            AI Optimize
          </Button>
          <Button onClick={() => setIsMoodDialogOpen(true)} variant="outline" size="sm" data-testid="button-log-mood">
            <Smile className="h-4 w-4 mr-1" />
            Log Mood
          </Button>
          <Button onClick={() => setIsJournalDialogOpen(true)} variant="outline" size="sm" data-testid="button-add-journal">
            <BookOpen className="h-4 w-4 mr-1" />
            Journal
          </Button>
          <Button onClick={() => setIsCreateDialogOpen(true)} data-testid="button-create-habit">
            <Plus className="h-4 w-4 mr-1" />
            New Habit
          </Button>
        </div>
      </div>

      {/* Risk Alerts Banner */}
      {alertsData?.alerts && alertsData.alerts.length > 0 && (
        <Card className="border-amber-500/50 bg-amber-500/10">
          <CardContent className="py-3">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-amber-500 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="font-medium text-amber-700 dark:text-amber-400">
                  {alertsData.alerts[0].title}
                </div>
                <p className="text-sm text-muted-foreground">{alertsData.alerts[0].message}</p>
                {alertsData.alerts[0].suggestedActions && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {alertsData.alerts[0].suggestedActions.slice(0, 2).map((action, i) => (
                      <Badge key={i} variant="outline" className="text-xs">{action}</Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="flex flex-wrap h-auto gap-1">
          <TabsTrigger value="overview" className="flex items-center gap-1">
            <Target className="h-4 w-4" />
            <span className="hidden sm:inline">Overview</span>
          </TabsTrigger>
          <TabsTrigger value="calendar" className="flex items-center gap-1">
            <Calendar className="h-4 w-4" />
            <span className="hidden sm:inline">Calendar</span>
          </TabsTrigger>
          <TabsTrigger value="coach" className="flex items-center gap-1">
            <Brain className="h-4 w-4" />
            <span className="hidden sm:inline">AI Coach</span>
          </TabsTrigger>
          <TabsTrigger value="quit" className="flex items-center gap-1">
            <Shield className="h-4 w-4" />
            <span className="hidden sm:inline">Quit Plans</span>
          </TabsTrigger>
          <TabsTrigger value="cbt" className="flex items-center gap-1">
            <Heart className="h-4 w-4" />
            <span className="hidden sm:inline">CBT Tools</span>
          </TabsTrigger>
          <TabsTrigger value="insights" className="flex items-center gap-1">
            <TrendingUp className="h-4 w-4" />
            <span className="hidden sm:inline">Insights</span>
          </TabsTrigger>
          <TabsTrigger value="rewards" className="flex items-center gap-1">
            <Trophy className="h-4 w-4" />
            <span className="hidden sm:inline">Rewards</span>
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          {/* Quick Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Active Habits</p>
                    <p className="text-2xl font-bold">{habits.length}</p>
                  </div>
                  <Target className="h-8 w-8 text-primary/20" />
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Current Streaks</p>
                    <p className="text-2xl font-bold">{streaksData?.summary?.totalCurrentStreaks || 0}</p>
                  </div>
                  <Flame className="h-8 w-8 text-orange-500/20" />
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Points</p>
                    <p className="text-2xl font-bold">{rewardsData?.totalPoints || 0}</p>
                  </div>
                  <Star className="h-8 w-8 text-yellow-500/20" />
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Growth Stage</p>
                    <p className="text-2xl font-bold capitalize">{rewardsData?.growthStage || 'Seed'}</p>
                  </div>
                  <Flower2 className="h-8 w-8 text-green-500/20" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Habits Grid */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {habits.map((habit) => {
              const CategoryIcon = getCategoryIcon(habit.category);
              return (
                <Card key={habit.id} className="hover-elevate transition-all" data-testid={`card-habit-${habit.id}`}>
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <div className="p-2 rounded-lg bg-primary/10">
                          <CategoryIcon className="h-4 w-4 text-primary" />
                        </div>
                        <div>
                          <CardTitle className="text-base">{habit.name}</CardTitle>
                          <CardDescription className="text-xs capitalize">{habit.category}</CardDescription>
                        </div>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        <Flame className="h-3 w-3 mr-1 text-orange-500" />
                        {habit.currentStreak} day{habit.currentStreak !== 1 ? 's' : ''}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {habit.description && (
                      <p className="text-sm text-muted-foreground line-clamp-2">{habit.description}</p>
                    )}
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Best streak: {habit.longestStreak} days</span>
                      <span className="text-muted-foreground">{habit.totalCompletions} total</span>
                    </div>
                    <Button
                      onClick={() => setSelectedHabitForCompletion(habit)}
                      className="w-full"
                      size="sm"
                      data-testid={`button-complete-${habit.id}`}
                    >
                      <CheckCircle2 className="h-4 w-4 mr-1" />
                      Mark Complete
                    </Button>
                  </CardContent>
                </Card>
              );
            })}

            {habits.length === 0 && (
              <Card className="col-span-full">
                <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                  <Target className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="font-semibold text-lg">No habits yet</h3>
                  <p className="text-muted-foreground mb-4">Start building healthy routines today!</p>
                  <Button onClick={() => setIsCreateDialogOpen(true)} data-testid="button-create-first-habit">
                    <Plus className="h-4 w-4 mr-1" />
                    Create Your First Habit
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>

          {/* AI Recommendations */}
          {recommendationsData?.recommendations?.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  AI Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {recommendationsData.recommendations.slice(0, 3).map((rec: any) => (
                    <div key={rec.id} className="flex items-start gap-3 p-3 rounded-lg bg-muted/50">
                      <Zap className="h-5 w-5 text-amber-500 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="font-medium">{rec.title}</div>
                        <p className="text-sm text-muted-foreground">{rec.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Calendar Tab */}
        <TabsContent value="calendar" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Habit Calendar</CardTitle>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedMonth(new Date(selectedMonth.getFullYear(), selectedMonth.getMonth() - 1))}
                  >
                    <ChevronDown className="h-4 w-4 rotate-90" />
                  </Button>
                  <span className="font-medium">{format(selectedMonth, 'MMMM yyyy')}</span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedMonth(new Date(selectedMonth.getFullYear(), selectedMonth.getMonth() + 1))}
                  >
                    <ChevronDown className="h-4 w-4 -rotate-90" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {renderCalendar()}
              <div className="mt-4 flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-green-500/30" />
                  <span>All done</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-yellow-500/30" />
                  <span>Partial</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-red-500/30" />
                  <span>Mostly skipped</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Streaks Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Flame className="h-5 w-5 text-orange-500" />
                Current Streaks
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {habits.filter(h => h.currentStreak > 0).map(habit => (
                  <div key={habit.id} className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
                    <span className="font-medium">{habit.name}</span>
                    <Badge variant="outline" className="bg-orange-500/10">
                      <Flame className="h-3 w-3 mr-1 text-orange-500" />
                      {habit.currentStreak} days
                    </Badge>
                  </div>
                ))}
                {habits.filter(h => h.currentStreak > 0).length === 0 && (
                  <p className="text-center text-muted-foreground py-4">
                    Complete habits to start building streaks!
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* AI Coach Tab */}
        <TabsContent value="coach" className="space-y-4">
          <Card className="h-[600px] flex flex-col">
            <CardHeader>
              <CardTitle className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-primary" />
                  AI Habit Coach
                </div>
                <Select value={coachPersonality} onValueChange={(v: any) => setCoachPersonality(v)}>
                  <SelectTrigger className="w-[140px]" data-testid="select-coach-personality">
                    <SelectValue placeholder="Personality" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="supportive">Supportive</SelectItem>
                    <SelectItem value="motivational">Motivational</SelectItem>
                    <SelectItem value="analytical">Analytical</SelectItem>
                    <SelectItem value="tough_love">Tough Love</SelectItem>
                    <SelectItem value="mindful">Mindful</SelectItem>
                  </SelectContent>
                </Select>
              </CardTitle>
              <CardDescription>
                Get personalized advice, motivation, and CBT techniques
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <ScrollArea className="flex-1 pr-4 mb-4">
                <div className="space-y-4">
                  {coachMessages.length === 0 && (
                    <div className="text-center py-8 text-muted-foreground">
                      <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Start a conversation with your AI coach!</p>
                      <p className="text-sm mt-2">Try asking about motivation, habit tips, or dealing with setbacks.</p>
                    </div>
                  )}
                  {coachMessages.map(msg => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg px-4 py-2 ${
                          msg.role === 'user'
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted'
                        }`}
                      >
                        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                        {msg.responseType && (
                          <Badge variant="outline" className="mt-2 text-xs">{msg.responseType}</Badge>
                        )}
                      </div>
                    </div>
                  ))}
                  {coachChatMutation.isPending && (
                    <div className="flex justify-start">
                      <div className="bg-muted rounded-lg px-4 py-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
              <div className="flex gap-2">
                <Input
                  placeholder="Ask your coach anything..."
                  value={coachMessage}
                  onChange={(e) => setCoachMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && coachMessage && coachChatMutation.mutate(coachMessage)}
                  data-testid="input-coach-message"
                />
                <Button
                  onClick={() => coachMessage && coachChatMutation.mutate(coachMessage)}
                  disabled={!coachMessage || coachChatMutation.isPending}
                  data-testid="button-send-coach-message"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Quit Plans Tab */}
        <TabsContent value="quit" className="space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-lg font-semibold">Addiction-Mode: Quit Plans</h2>
              <p className="text-sm text-muted-foreground">Track your journey to break unhealthy habits</p>
            </div>
            <Button onClick={() => setIsQuitPlanDialogOpen(true)} data-testid="button-create-quit-plan">
              <Plus className="h-4 w-4 mr-1" />
              New Quit Plan
            </Button>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {quitPlansData?.quitPlans?.map(plan => (
              <Card key={plan.id} data-testid={`card-quit-plan-${plan.id}`}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5 text-green-500" />
                      {plan.habitName}
                    </CardTitle>
                    <Badge variant={plan.status === 'active' ? 'default' : 'secondary'}>
                      {plan.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-2xl font-bold text-green-600">{plan.daysClean}</div>
                      <div className="text-xs text-muted-foreground">Days Clean</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-amber-600">{plan.longestStreak}</div>
                      <div className="text-xs text-muted-foreground">Best Streak</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-red-600">{plan.totalRelapses}</div>
                      <div className="text-xs text-muted-foreground">Relapses</div>
                    </div>
                  </div>
                  {plan.moneySavedPerDay && plan.daysClean > 0 && (
                    <div className="text-center p-3 rounded-lg bg-green-500/10">
                      <div className="text-lg font-bold text-green-600">
                        ${(plan.moneySavedPerDay * plan.daysClean).toFixed(2)}
                      </div>
                      <div className="text-xs text-muted-foreground">Money Saved</div>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" className="flex-1">Log Craving</Button>
                    <Button variant="destructive" size="sm" className="flex-1">Log Setback</Button>
                  </div>
                </CardContent>
              </Card>
            ))}

            {(!quitPlansData?.quitPlans || quitPlansData.quitPlans.length === 0) && (
              <Card className="col-span-full">
                <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                  <Shield className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="font-semibold text-lg">No quit plans yet</h3>
                  <p className="text-muted-foreground mb-4">Create a plan to overcome unhealthy habits</p>
                  <Button onClick={() => setIsQuitPlanDialogOpen(true)}>
                    <Plus className="h-4 w-4 mr-1" />
                    Start a Quit Plan
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* CBT Tools Tab */}
        <TabsContent value="cbt" className="space-y-4">
          <div>
            <h2 className="text-lg font-semibold">Guided CBT Exercises</h2>
            <p className="text-sm text-muted-foreground">Evidence-based techniques for managing urges and thoughts</p>
          </div>

          {activeCbtSession ? (
            <Card>
              <CardHeader>
                <CardTitle>{activeCbtSession.title}</CardTitle>
                <CardDescription>
                  Step {activeCbtSession.current_step} of {activeCbtSession.total_steps}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Progress 
                  value={(activeCbtSession.current_step / activeCbtSession.total_steps) * 100} 
                  className="h-2"
                />
                <div className="p-4 bg-muted rounded-lg">
                  <p className="font-medium">{activeCbtSession.current_prompt?.prompt}</p>
                </div>
                {activeCbtSession.current_prompt?.type === 'text' && (
                  <Textarea
                    placeholder="Your response..."
                    value={cbtResponse}
                    onChange={(e) => setCbtResponse(e.target.value)}
                    className="min-h-[100px]"
                  />
                )}
                {activeCbtSession.current_prompt?.type === 'confirm' && (
                  <div className="flex gap-2">
                    <Button 
                      onClick={() => cbtRespondMutation.mutate({
                        sessionId: activeCbtSession.session_id,
                        stepNumber: activeCbtSession.current_step,
                        response: "confirmed"
                      })}
                      className="flex-1"
                    >
                      Continue
                    </Button>
                  </div>
                )}
                {activeCbtSession.current_prompt?.type === 'number' && (
                  <div className="space-y-2">
                    <Slider
                      value={[parseInt(cbtResponse) || 5]}
                      onValueChange={(v) => setCbtResponse(v[0].toString())}
                      min={1}
                      max={10}
                      step={1}
                    />
                    <div className="text-center font-bold text-2xl">{cbtResponse || 5}</div>
                  </div>
                )}
                <div className="flex gap-2">
                  <Button variant="outline" onClick={() => setActiveCbtSession(null)}>
                    Cancel
                  </Button>
                  {activeCbtSession.current_prompt?.type !== 'confirm' && (
                    <Button
                      onClick={() => cbtRespondMutation.mutate({
                        sessionId: activeCbtSession.session_id,
                        stepNumber: activeCbtSession.current_step,
                        response: cbtResponse
                      })}
                      disabled={!cbtResponse || cbtRespondMutation.isPending}
                      className="flex-1"
                    >
                      {cbtRespondMutation.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        'Next'
                      )}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {cbtFlows?.flows?.map((flow: any) => (
                <Card key={flow.id} className="hover-elevate cursor-pointer" onClick={() => startCbtMutation.mutate(flow.id)}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {flow.id === 'urge_surfing' && <Wind className="h-5 w-5 text-blue-500" />}
                      {flow.id === 'reframe_thought' && <Brain className="h-5 w-5 text-purple-500" />}
                      {flow.id === 'grounding' && <TreeDeciduous className="h-5 w-5 text-green-500" />}
                      {flow.id === 'breathing' && <Wind className="h-5 w-5 text-cyan-500" />}
                      {flow.title}
                    </CardTitle>
                    <CardDescription>{flow.description}</CardDescription>
                  </CardHeader>
                  <CardFooter>
                    <Badge variant="outline">{flow.totalSteps} steps</Badge>
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-lg font-semibold">AI-Powered Insights</h2>
              <p className="text-sm text-muted-foreground">Pattern detection and personalized recommendations</p>
            </div>
            <div className="flex gap-2">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => analyzeTriggersMutation.mutate()}
                disabled={analyzeTriggersMutation.isPending}
              >
                {analyzeTriggersMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-1" />
                ) : (
                  <RefreshCw className="h-4 w-4 mr-1" />
                )}
                Analyze Patterns
              </Button>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => generateRecommendationsMutation.mutate()}
                disabled={generateRecommendationsMutation.isPending}
              >
                {generateRecommendationsMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-1" />
                ) : (
                  <Sparkles className="h-4 w-4 mr-1" />
                )}
                Get Recommendations
              </Button>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {/* Detected Triggers */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-amber-500" />
                  Detected Patterns
                </CardTitle>
              </CardHeader>
              <CardContent>
                {triggersData?.triggers?.length > 0 ? (
                  <div className="space-y-3">
                    {triggersData.triggers.map(trigger => (
                      <div key={trigger.id} className="p-3 rounded-lg bg-muted/50">
                        <p className="font-medium text-sm">{trigger.pattern}</p>
                        <div className="flex items-center gap-2 mt-2">
                          <Badge variant="outline" className="text-xs">{trigger.triggerType}</Badge>
                          <span className="text-xs text-muted-foreground">
                            {Math.round(trigger.confidence * 100)}% confidence
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-center text-muted-foreground py-4">
                    Complete more habits to detect patterns
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Mood Trends */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <LineChart className="h-5 w-5 text-primary" />
                  Mood Trends
                </CardTitle>
              </CardHeader>
              <CardContent>
                {moodTrends?.trends?.length > 0 ? (
                  <div className="space-y-2">
                    {moodTrends.trends.slice(-7).map((trend: any) => (
                      <div key={trend.date} className="flex items-center justify-between">
                        <span className="text-sm">{format(parseISO(trend.date), 'EEE, MMM d')}</span>
                        <div className="flex items-center gap-2">
                          {trend.avgMood && (
                            <Badge variant="outline">Mood: {trend.avgMood}</Badge>
                          )}
                          {trend.avgEnergy && (
                            <Badge variant="outline" className="bg-yellow-500/10">Energy: {trend.avgEnergy}</Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-center text-muted-foreground py-4">
                    Log mood entries to see trends
                  </p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Journals */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5 text-primary" />
                Smart Journals
              </CardTitle>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => generateWeeklySummaryMutation.mutate()}
                disabled={generateWeeklySummaryMutation.isPending}
              >
                {generateWeeklySummaryMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-1" />
                ) : (
                  <Sparkles className="h-4 w-4 mr-1" />
                )}
                Generate Weekly Summary
              </Button>
            </CardHeader>
            <CardContent>
              {journalsData?.journals?.length > 0 ? (
                <div className="space-y-3">
                  {journalsData.journals.slice(0, 5).map(journal => (
                    <div key={journal.id} className="p-3 rounded-lg bg-muted/50">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">
                          {journal.title || format(parseISO(journal.recordedAt), 'MMMM d, yyyy')}
                        </span>
                        {journal.isWeeklySummary && (
                          <Badge variant="default" className="bg-primary/80">Weekly Summary</Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {journal.aiSummary || journal.content}
                      </p>
                      {journal.highlights && journal.highlights.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {journal.highlights.slice(0, 2).map((h, i) => (
                            <Badge key={i} variant="outline" className="text-xs bg-green-500/10">
                              {h.substring(0, 40)}...
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-center text-muted-foreground py-4">
                  Start journaling to get AI-powered reflections
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Rewards Tab */}
        <TabsContent value="rewards" className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
            <HabitGamificationPanel />
            {renderRewardsVisualization()}
          </div>
        </TabsContent>
      </Tabs>

      {/* Create Habit Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Create New Habit</DialogTitle>
            <DialogDescription>Build a healthy routine one habit at a time</DialogDescription>
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
                      <Input placeholder="Morning meditation" {...field} data-testid="input-habit-name" />
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
                    <FormLabel>Description (optional)</FormLabel>
                    <FormControl>
                      <Textarea placeholder="Why this habit matters to you..." {...field} data-testid="input-habit-description" />
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
                          <SelectValue placeholder="Select category" />
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
                          <SelectValue placeholder="Select frequency" />
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
              <FormField
                control={createHabitForm.control}
                name="reminderEnabled"
                render={({ field }) => (
                  <FormItem className="flex items-center justify-between rounded-lg border p-3">
                    <div className="space-y-0.5">
                      <FormLabel>Enable Reminders</FormLabel>
                      <FormDescription className="text-xs">Get notified to complete this habit</FormDescription>
                    </div>
                    <FormControl>
                      <Switch checked={field.value} onCheckedChange={field.onChange} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <Button type="submit" className="w-full" disabled={createHabitMutation.isPending} data-testid="button-submit-habit">
                {createHabitMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-1" />
                ) : (
                  <Plus className="h-4 w-4 mr-1" />
                )}
                Create Habit
              </Button>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      {/* Complete Habit Dialog */}
      <Dialog open={!!selectedHabitForCompletion} onOpenChange={(open) => !open && setSelectedHabitForCompletion(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Complete Habit</DialogTitle>
            <DialogDescription>
              {selectedHabitForCompletion?.name}
            </DialogDescription>
          </DialogHeader>
          <Form {...completeHabitForm}>
            <form 
              onSubmit={completeHabitForm.handleSubmit((data) => 
                selectedHabitForCompletion && completeHabitMutation.mutate({ habitId: selectedHabitForCompletion.id, data })
              )} 
              className="space-y-4"
            >
              <FormField
                control={completeHabitForm.control}
                name="mood"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>How are you feeling?</FormLabel>
                    <div className="flex gap-2 flex-wrap">
                      {['great', 'good', 'okay', 'struggling'].map(mood => {
                        const MoodIcon = getMoodEmoji(mood).icon;
                        return (
                          <Button
                            key={mood}
                            type="button"
                            variant={field.value === mood ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => field.onChange(mood)}
                            className="flex items-center gap-1"
                          >
                            <MoodIcon className={`h-4 w-4 ${getMoodEmoji(mood).color}`} />
                            <span className="capitalize">{mood}</span>
                          </Button>
                        );
                      })}
                    </div>
                  </FormItem>
                )}
              />
              <FormField
                control={completeHabitForm.control}
                name="difficultyLevel"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Difficulty (1-5)</FormLabel>
                    <FormControl>
                      <Slider
                        value={[field.value || 3]}
                        onValueChange={(v) => field.onChange(v[0])}
                        min={1}
                        max={5}
                        step={1}
                      />
                    </FormControl>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Easy</span>
                      <span>Hard</span>
                    </div>
                  </FormItem>
                )}
              />
              <FormField
                control={completeHabitForm.control}
                name="notes"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Notes (optional)</FormLabel>
                    <FormControl>
                      <Textarea placeholder="Any thoughts to capture?" {...field} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <Button type="submit" className="w-full" disabled={completeHabitMutation.isPending}>
                {completeHabitMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-1" />
                ) : (
                  <CheckCircle2 className="h-4 w-4 mr-1" />
                )}
                Mark Complete
              </Button>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      {/* Mood Entry Dialog */}
      <Dialog open={isMoodDialogOpen} onOpenChange={setIsMoodDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Log Your Mood</DialogTitle>
            <DialogDescription>Track how you're feeling for better insights</DialogDescription>
          </DialogHeader>
          <Form {...moodForm}>
            <form onSubmit={moodForm.handleSubmit((data) => logMoodMutation.mutate(data))} className="space-y-4">
              <FormField
                control={moodForm.control}
                name="moodScore"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Mood Score (1-10)</FormLabel>
                    <FormControl>
                      <Slider
                        value={[field.value]}
                        onValueChange={(v) => field.onChange(v[0])}
                        min={1}
                        max={10}
                        step={1}
                      />
                    </FormControl>
                    <div className="text-center font-bold text-2xl">{field.value}</div>
                  </FormItem>
                )}
              />
              <div className="grid grid-cols-2 gap-4">
                <FormField
                  control={moodForm.control}
                  name="energyLevel"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Energy (1-10)</FormLabel>
                      <FormControl>
                        <Slider
                          value={[field.value || 5]}
                          onValueChange={(v) => field.onChange(v[0])}
                          min={1}
                          max={10}
                          step={1}
                        />
                      </FormControl>
                      <div className="text-center">{field.value || 5}</div>
                    </FormItem>
                  )}
                />
                <FormField
                  control={moodForm.control}
                  name="stressLevel"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Stress (1-10)</FormLabel>
                      <FormControl>
                        <Slider
                          value={[field.value || 5]}
                          onValueChange={(v) => field.onChange(v[0])}
                          min={1}
                          max={10}
                          step={1}
                        />
                      </FormControl>
                      <div className="text-center">{field.value || 5}</div>
                    </FormItem>
                  )}
                />
              </div>
              <FormField
                control={moodForm.control}
                name="journalText"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Quick Journal (optional)</FormLabel>
                    <FormControl>
                      <Textarea placeholder="What's on your mind?" {...field} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <Button type="submit" className="w-full" disabled={logMoodMutation.isPending}>
                {logMoodMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                Log Mood
              </Button>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      {/* Quit Plan Dialog */}
      <Dialog open={isQuitPlanDialogOpen} onOpenChange={setIsQuitPlanDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Create Quit Plan</DialogTitle>
            <DialogDescription>Take the first step to overcome an unhealthy habit</DialogDescription>
          </DialogHeader>
          <Form {...quitPlanForm}>
            <form onSubmit={quitPlanForm.handleSubmit((data) => createQuitPlanMutation.mutate(data))} className="space-y-4">
              <FormField
                control={quitPlanForm.control}
                name="habitName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>What do you want to quit?</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., Smoking, Social media, Snacking" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={quitPlanForm.control}
                name="category"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Category</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select type" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="substance">Substance</SelectItem>
                        <SelectItem value="behavioral">Behavioral</SelectItem>
                        <SelectItem value="food">Food</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </FormItem>
                )}
              />
              <FormField
                control={quitPlanForm.control}
                name="quitMethod"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Approach</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select method" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="cold_turkey">Cold Turkey</SelectItem>
                        <SelectItem value="gradual_reduction">Gradual Reduction</SelectItem>
                        <SelectItem value="replacement">Replacement</SelectItem>
                      </SelectContent>
                    </Select>
                  </FormItem>
                )}
              />
              <FormField
                control={quitPlanForm.control}
                name="reasonsToQuit"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Reasons to quit (comma-separated)</FormLabel>
                    <FormControl>
                      <Textarea placeholder="Health, Save money, More time for family" {...field} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={quitPlanForm.control}
                name="moneySavedPerDay"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Money saved per day ($)</FormLabel>
                    <FormControl>
                      <Input type="number" step="0.01" placeholder="0.00" {...field} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <Button type="submit" className="w-full" disabled={createQuitPlanMutation.isPending}>
                {createQuitPlanMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                Start My Journey
              </Button>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      {/* Journal Dialog */}
      <Dialog open={isJournalDialogOpen} onOpenChange={setIsJournalDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>New Journal Entry</DialogTitle>
            <DialogDescription>Reflect on your day and track your thoughts</DialogDescription>
          </DialogHeader>
          <Form {...journalForm}>
            <form onSubmit={journalForm.handleSubmit((data) => createJournalMutation.mutate(data))} className="space-y-4">
              <FormField
                control={journalForm.control}
                name="title"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Title (optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="Today's reflection" {...field} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={journalForm.control}
                name="entryType"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Entry Type</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select type" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="daily">Daily Journal</SelectItem>
                        <SelectItem value="reflection">Reflection</SelectItem>
                        <SelectItem value="gratitude">Gratitude</SelectItem>
                        <SelectItem value="goal_setting">Goal Setting</SelectItem>
                      </SelectContent>
                    </Select>
                  </FormItem>
                )}
              />
              <FormField
                control={journalForm.control}
                name="content"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Your thoughts</FormLabel>
                    <FormControl>
                      <Textarea 
                        placeholder="What's on your mind today?" 
                        className="min-h-[150px]"
                        {...field} 
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <Button type="submit" className="w-full" disabled={createJournalMutation.isPending}>
                {createJournalMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                Save Journal
              </Button>
            </form>
          </Form>
        </DialogContent>
      </Dialog>
      
      {/* AI Habit Suggestions Modal */}
      <HabitSuggestionsModal
        open={isAISuggestionsOpen}
        onOpenChange={setIsAISuggestionsOpen}
        suggestions={habitSuggestionsMutation.data?.suggestions || []}
        experienceId={habitSuggestionsMutation.data?.experience_id || ''}
        patientId={user?.id || ''}
        isLoading={habitSuggestionsMutation.isPending}
        onAddHabit={handleAddSuggestedHabit}
      />
    </div>
  );
}
