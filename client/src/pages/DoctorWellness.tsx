import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Heart, TrendingUp, TrendingDown, AlertCircle, Battery, Brain, Calendar, Clock, Lightbulb, Sparkles, Plus } from "lucide-react";
import { Progress } from "@/components/ui/progress";

const logWellnessSchema = z.object({
  stressLevel: z.coerce.number().int().min(1).max(10),
  hoursWorked: z.coerce.number().min(0).max(24),
  patientsToday: z.coerce.number().int().min(0).max(100),
  burnoutRisk: z.enum(['low', 'moderate', 'high', 'critical']).optional(),
  notes: z.string().max(500).optional(),
});

type DoctorWellness = {
  id: string;
  stressLevel: number;
  burnoutRisk: string;
  workLifeBalance: string;
  lastUpdate: string;
  weeklyAvgStress: string;
  monthlyTrend: string;
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

export default function DoctorWellness() {
  const { toast } = useToast();
  const [isLogDialogOpen, setIsLogDialogOpen] = useState(false);

  // Fetch wellness data
  const { data: wellness, isLoading: wellnessLoading } = useQuery<DoctorWellness>({
    queryKey: ['/api/v1/ml/doctor-wellness'],
  });

  // Fetch recommendations
  const { data: recommendations = [] } = useQuery<Recommendation[]>({
    queryKey: ['/api/v1/ml/recommendations'],
  });

  const wellnessRecommendations = recommendations.filter(r => r.category === 'doctor_wellness');

  // Log wellness mutation
  const logWellnessForm = useForm({
    resolver: zodResolver(logWellnessSchema),
    defaultValues: {
      stressLevel: 5,
      hoursWorked: 8,
      patientsToday: 15,
      burnoutRisk: undefined,
      notes: "",
    },
  });

  const logWellnessMutation = useMutation({
    mutationFn: async (data: z.infer<typeof logWellnessSchema>) => {
      return await apiRequest('POST', '/api/v1/ml/doctor-wellness', data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/ml/doctor-wellness'] });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/ml/recommendations'] });
      setIsLogDialogOpen(false);
      logWellnessForm.reset();
      toast({
        title: "Wellness logged",
        description: "Your wellness data has been recorded. Assistant Lysa is analyzing your patterns.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error logging wellness",
        description: error.message || "Please try again",
        variant: "destructive",
      });
    },
  });

  const getBurnoutRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-500';
      case 'moderate': return 'text-yellow-500';
      case 'high': return 'text-orange-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getBurnoutRiskBg = (risk: string) => {
    switch (risk) {
      case 'low': return 'bg-green-500/10';
      case 'moderate': return 'bg-yellow-500/10';
      case 'high': return 'bg-orange-500/10';
      case 'critical': return 'bg-red-500/10';
      default: return 'bg-gray-500/10';
    }
  };

  const getWorkLifeBalanceColor = (balance: string) => {
    switch (balance) {
      case 'excellent': return 'text-green-500';
      case 'good': return 'text-blue-500';
      case 'fair': return 'text-yellow-500';
      case 'poor': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const stressLevelPercentage = wellness ? (wellness.stressLevel / 10) * 100 : 0;
  const weeklyAvgStress = wellness ? parseFloat(wellness.weeklyAvgStress) : 0;
  const weeklyAvgPercentage = (weeklyAvgStress / 10) * 100;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold" data-testid="text-wellness-title">Doctor Wellness Tracking</h1>
          <p className="text-muted-foreground mt-1">
            Monitor your well-being with Assistant Lysa's support
          </p>
        </div>
        <Dialog open={isLogDialogOpen} onOpenChange={setIsLogDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-log-wellness">
              <Plus className="w-4 h-4 mr-2" />
              Log Wellness
            </Button>
          </DialogTrigger>
          <DialogContent data-testid="dialog-log-wellness">
            <DialogHeader>
              <DialogTitle>Log Your Wellness</DialogTitle>
              <DialogDescription>
                Share how you're feeling today. Assistant Lysa will provide personalized recommendations.
              </DialogDescription>
            </DialogHeader>
            <Form {...logWellnessForm}>
              <form onSubmit={logWellnessForm.handleSubmit((data) => logWellnessMutation.mutate(data))} className="space-y-4">
                <FormField
                  control={logWellnessForm.control}
                  name="stressLevel"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Stress Level (1-10)</FormLabel>
                      <FormControl>
                        <div className="space-y-2">
                          <input
                            type="range"
                            min="1"
                            max="10"
                            className="w-full"
                            data-testid="input-stress-level"
                            {...field}
                            onChange={(e) => field.onChange(parseInt(e.target.value))}
                          />
                          <div className="text-center text-2xl font-bold">{field.value}</div>
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={logWellnessForm.control}
                  name="hoursWorked"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Hours Worked Today</FormLabel>
                      <FormControl>
                        <input
                          type="number"
                          min="0"
                          max="24"
                          step="0.5"
                          placeholder="8"
                          className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                          data-testid="input-hours-worked"
                          {...field}
                          onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={logWellnessForm.control}
                  name="patientsToday"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Patients Seen Today</FormLabel>
                      <FormControl>
                        <input
                          type="number"
                          min="0"
                          max="100"
                          placeholder="15"
                          className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                          data-testid="input-patients-today"
                          {...field}
                          onChange={(e) => field.onChange(parseInt(e.target.value) || 0)}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={logWellnessForm.control}
                  name="notes"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Notes (Optional)</FormLabel>
                      <FormControl>
                        <Textarea placeholder="How are you feeling today?" data-testid="input-wellness-notes" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <div className="flex justify-end gap-2 pt-4">
                  <Button type="button" variant="outline" onClick={() => setIsLogDialogOpen(false)} data-testid="button-cancel-wellness">
                    Cancel
                  </Button>
                  <Button type="submit" disabled={logWellnessMutation.isPending} data-testid="button-submit-wellness">
                    {logWellnessMutation.isPending ? "Saving..." : "Log Wellness"}
                  </Button>
                </div>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Wellness Overview */}
      {wellnessLoading ? (
        <div className="grid gap-4 md:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-6">
                <div className="space-y-3">
                  <div className="h-4 bg-muted rounded w-1/2" />
                  <div className="h-8 bg-muted rounded w-3/4" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : wellness ? (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            {/* Current Stress Level */}
            <Card data-testid="card-stress-level">
              <CardHeader className="pb-3">
                <CardDescription className="flex items-center gap-2">
                  <Brain className="w-4 h-4" />
                  Current Stress Level
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-3xl font-bold">{wellness.stressLevel}/10</div>
                  <Progress value={stressLevelPercentage} className="h-2" />
                  <p className="text-xs text-muted-foreground">
                    {wellness.stressLevel <= 3 ? 'Low stress' : wellness.stressLevel <= 6 ? 'Moderate stress' : 'High stress'}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Burnout Risk */}
            <Card data-testid="card-burnout-risk">
              <CardHeader className="pb-3">
                <CardDescription className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  Burnout Risk
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className={`text-3xl font-bold capitalize ${getBurnoutRiskColor(wellness.burnoutRisk)}`}>
                    {wellness.burnoutRisk}
                  </div>
                  <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm ${getBurnoutRiskBg(wellness.burnoutRisk)} ${getBurnoutRiskColor(wellness.burnoutRisk)}`}>
                    <Battery className="w-4 h-4" />
                    AI-detected risk level
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Work-Life Balance */}
            <Card data-testid="card-work-life-balance">
              <CardHeader className="pb-3">
                <CardDescription className="flex items-center gap-2">
                  <Heart className="w-4 h-4" />
                  Work-Life Balance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className={`text-3xl font-bold capitalize ${getWorkLifeBalanceColor(wellness.workLifeBalance)}`}>
                    {wellness.workLifeBalance}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {wellness.workLifeBalance === 'excellent' || wellness.workLifeBalance === 'good' 
                      ? 'Keep up the great balance!' 
                      : 'Consider taking breaks'}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Trends */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Calendar className="w-5 h-5" />
                  Weekly Average Stress
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">{weeklyAvgStress.toFixed(1)}/10</div>
                  <Progress value={weeklyAvgPercentage} className="h-2" />
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    {wellness.monthlyTrend === 'improving' ? (
                      <>
                        <TrendingDown className="w-4 h-4 text-green-500" />
                        <span className="text-green-500">Trending down (improving)</span>
                      </>
                    ) : wellness.monthlyTrend === 'stable' ? (
                      <>
                        <TrendingUp className="w-4 h-4 text-blue-500" />
                        <span className="text-blue-500">Stable</span>
                      </>
                    ) : (
                      <>
                        <TrendingUp className="w-4 h-4 text-orange-500" />
                        <span className="text-orange-500">Trending up (worsening)</span>
                      </>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Clock className="w-5 h-5" />
                  Last Updated
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">
                    {new Date(wellness.lastUpdate).toLocaleDateString()}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {new Date(wellness.lastUpdate).toLocaleTimeString()}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      ) : (
        <Card>
          <CardContent className="py-12">
            <div className="text-center">
              <Heart className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No wellness data yet</h3>
              <p className="text-muted-foreground mb-4">
                Start tracking your well-being to get personalized insights from Assistant Lysa
              </p>
              <Button onClick={() => setIsLogDialogOpen(true)} data-testid="button-start-tracking">
                <Plus className="w-4 h-4 mr-2" />
                Start Tracking
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Assistant Lysa Recommendations */}
      {wellnessRecommendations.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-primary" />
              <CardTitle>Wellness Recommendations from Assistant Lysa</CardTitle>
            </div>
            <CardDescription>
              AI-powered insights to support your well-being and prevent burnout
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3">
              {wellnessRecommendations.slice(0, 4).map((rec) => (
                <div key={rec.id} className="p-4 border rounded-lg hover-elevate" data-testid={`recommendation-${rec.id}`}>
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Lightbulb className="w-4 h-4 text-yellow-500" />
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
    </div>
  );
}
