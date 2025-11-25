import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'wouter';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Activity,
  AlertCircle,
  BarChart3,
  Brain,
  Calendar,
  Camera,
  CheckCircle2,
  ChevronRight,
  Clock,
  Eye,
  FileText,
  Hand,
  Heart,
  Loader2,
  Mic,
  Palette,
  Play,
  Smile,
  Thermometer,
  TrendingDown,
  TrendingUp,
  User,
  Video,
  Wind,
  Zap,
  Footprints,
  MessageSquare,
  AlertTriangle,
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { format, subDays, subMonths, subYears, isWithinInterval, startOfDay, endOfDay, parseISO } from 'date-fns';

const TIME_RANGES = [
  { value: '7', label: '7 Days', days: 7 },
  { value: '15', label: '15 Days', days: 15 },
  { value: '30', label: '30 Days', days: 30 },
  { value: '90', label: '3 Months', days: 90 },
  { value: '180', label: '6 Months', days: 180 },
  { value: '365', label: '1 Year', days: 365 },
  { value: 'all', label: 'All Time', days: Infinity },
];

const CHART_COLORS = {
  primary: 'hsl(var(--primary))',
  secondary: 'hsl(var(--chart-2))',
  tertiary: 'hsl(var(--chart-3))',
  quaternary: 'hsl(var(--chart-4))',
  quinary: 'hsl(var(--chart-5))',
};

function LegalDisclaimer() {
  return (
    <Alert className="border-amber-500/50 bg-amber-500/5">
      <AlertCircle className="h-4 w-4 text-amber-500" />
      <AlertTitle className="text-amber-700 dark:text-amber-400 font-semibold">
        Wellness Monitoring - Not Medical Advice
      </AlertTitle>
      <AlertDescription className="text-amber-600 dark:text-amber-300 text-sm">
        This data is for personal wellness tracking only. Consult your healthcare provider for medical decisions.
      </AlertDescription>
    </Alert>
  );
}

function EmptyStatePrompt({ 
  icon: Icon, 
  title, 
  description, 
  actionLabel, 
  actionHref 
}: { 
  icon: any; 
  title: string; 
  description: string; 
  actionLabel: string; 
  actionHref: string;
}) {
  return (
    <Card className="border-dashed">
      <CardContent className="p-8 text-center">
        <div className="mx-auto w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
          <Icon className="h-6 w-6 text-muted-foreground" />
        </div>
        <h3 className="font-semibold text-lg mb-2">{title}</h3>
        <p className="text-muted-foreground text-sm mb-4 max-w-md mx-auto">{description}</p>
        <Link href={actionHref}>
          <Button className="gap-2" data-testid="button-start-tracking">
            <Play className="h-4 w-4" />
            {actionLabel}
          </Button>
        </Link>
      </CardContent>
    </Card>
  );
}

function StatCard({ label, value, unit, icon: Icon, trend, color = "text-primary" }: {
  label: string;
  value: string | number;
  unit?: string;
  icon: any;
  trend?: 'up' | 'down' | 'stable';
  color?: string;
}) {
  return (
    <div className="p-4 rounded-lg border bg-card">
      <div className="flex items-center justify-between mb-2">
        <Icon className={`h-4 w-4 ${color}`} />
        {trend && (
          <div className={`flex items-center gap-1 text-xs ${
            trend === 'up' ? 'text-chart-2' : trend === 'down' ? 'text-destructive' : 'text-muted-foreground'
          }`}>
            {trend === 'up' ? <TrendingUp className="h-3 w-3" /> : 
             trend === 'down' ? <TrendingDown className="h-3 w-3" /> : null}
          </div>
        )}
      </div>
      <div className="text-2xl font-bold">{value}{unit && <span className="text-sm font-normal text-muted-foreground ml-1">{unit}</span>}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

export default function DailyFollowupHistory() {
  const { user } = useAuth();
  const [timeRange, setTimeRange] = useState('30');
  const [activeTab, setActiveTab] = useState('device');

  const selectedRange = TIME_RANGES.find(r => r.value === timeRange) || TIME_RANGES[2];
  const daysLimit = selectedRange.days === Infinity ? 3650 : selectedRange.days;

  const { data: deviceHistory, isLoading: deviceLoading } = useQuery<any[]>({
    queryKey: ['/api/daily-followup/history', { limit: daysLimit }],
    enabled: !!user,
  });

  const { data: symptomFeed, isLoading: symptomsLoading } = useQuery<any[]>({
    queryKey: ['/api/symptom-checkin/feed/unified', { days: daysLimit }],
    enabled: !!user,
  });

  const { data: paintrackSessions, isLoading: paintrackLoading } = useQuery<any[]>({
    queryKey: ['/api/paintrack/sessions', { limit: daysLimit }],
    enabled: !!user,
  });

  const { data: mentalHealthHistory, isLoading: mentalHealthLoading } = useQuery<any>({
    queryKey: ['/api/v1/mental-health/questionnaires/history', { limit: daysLimit }],
    enabled: !!user,
  });

  const { data: voiceFollowups, isLoading: audioLoading } = useQuery<any[]>({
    queryKey: ['/api/voice-followup/recent', { limit: daysLimit }],
    enabled: !!user,
  });

  const { data: videoMetrics, isLoading: videoLoading } = useQuery<any>({
    queryKey: ['/api/video-ai/latest-metrics'],
    enabled: !!user,
  });

  const filterByTimeRange = (data: any[], dateField: string = 'createdAt') => {
    if (!data || selectedRange.days === Infinity) return data || [];
    const cutoffDate = subDays(new Date(), selectedRange.days);
    return data.filter(item => {
      const itemDate = new Date(item[dateField] || item.timestamp || item.completed_at);
      return itemDate >= cutoffDate;
    });
  };

  const filteredDeviceHistory = useMemo(() => filterByTimeRange(deviceHistory || []), [deviceHistory, timeRange]);
  const filteredSymptoms = useMemo(() => {
    const filtered = filterByTimeRange(symptomFeed || [], 'timestamp');
    return filtered.filter((s: any) => s.dataSource === 'patient-reported');
  }, [symptomFeed, timeRange]);
  const filteredPaintrack = useMemo(() => filterByTimeRange(paintrackSessions || []), [paintrackSessions, timeRange]);
  const filteredMentalHealth = useMemo(() => {
    const history = mentalHealthHistory?.history || [];
    return filterByTimeRange(history, 'completed_at');
  }, [mentalHealthHistory, timeRange]);
  const filteredVoice = useMemo(() => filterByTimeRange(voiceFollowups || []), [voiceFollowups, timeRange]);

  const hasAnyData = (filteredDeviceHistory?.length > 0) || 
                     (filteredSymptoms?.length > 0) || 
                     (filteredPaintrack?.length > 0) || 
                     (filteredMentalHealth?.length > 0) ||
                     (filteredVoice?.length > 0) ||
                     videoMetrics;

  const isNewPatient = !hasAnyData && !deviceLoading && !symptomsLoading && !paintrackLoading && !mentalHealthLoading;

  const deviceChartData = useMemo(() => {
    if (!filteredDeviceHistory?.length) return [];
    return filteredDeviceHistory
      .slice()
      .reverse()
      .map((d: any) => ({
        date: format(new Date(d.createdAt || d.date), 'MMM d'),
        heartRate: d.heartRate,
        spo2: d.oxygenSaturation,
        temperature: d.temperature,
        steps: d.steps,
      }));
  }, [filteredDeviceHistory]);

  const symptomChartData = useMemo(() => {
    if (!filteredSymptoms?.length) return [];
    const grouped: Record<string, any> = {};
    filteredSymptoms.forEach((s: any) => {
      const date = format(new Date(s.timestamp), 'MMM d');
      if (!grouped[date]) {
        grouped[date] = { date, painTotal: 0, fatigueTotal: 0, count: 0 };
      }
      grouped[date].painTotal += s.painLevel || 0;
      grouped[date].fatigueTotal += s.fatigueLevel || 0;
      grouped[date].count++;
    });
    return Object.values(grouped).map((g: any) => ({
      date: g.date,
      painLevel: g.count > 0 ? (g.painTotal / g.count).toFixed(1) : 0,
      fatigueLevel: g.count > 0 ? (g.fatigueTotal / g.count).toFixed(1) : 0,
    }));
  }, [filteredSymptoms]);

  const painChartData = useMemo(() => {
    if (!filteredPaintrack?.length) return [];
    return filteredPaintrack
      .slice()
      .reverse()
      .map((p: any) => ({
        date: format(new Date(p.createdAt), 'MMM d'),
        vas: p.patientVas,
        joint: p.joint,
      }));
  }, [filteredPaintrack]);

  const mentalHealthChartData = useMemo(() => {
    if (!filteredMentalHealth?.length) return [];
    return filteredMentalHealth
      .slice()
      .reverse()
      .map((m: any) => ({
        date: format(new Date(m.completed_at), 'MMM d'),
        score: m.total_score,
        type: m.questionnaire_type,
        severity: m.severity_level,
      }));
  }, [filteredMentalHealth]);

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'none':
      case 'minimal':
        return 'bg-chart-2/20 text-chart-2';
      case 'mild':
        return 'bg-yellow-500/20 text-yellow-600';
      case 'moderate':
        return 'bg-orange-500/20 text-orange-600';
      case 'moderately_severe':
      case 'severe':
        return 'bg-destructive/20 text-destructive';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const calculateStats = (data: any[], field: string) => {
    if (!data?.length) return { avg: 0, min: 0, max: 0, trend: 'stable' as const };
    const values = data.map(d => d[field]).filter(v => v != null);
    if (!values.length) return { avg: 0, min: 0, max: 0, trend: 'stable' as const };
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const recentAvg = values.slice(-7).reduce((a, b) => a + b, 0) / Math.min(7, values.length);
    const olderAvg = values.slice(0, -7).reduce((a, b) => a + b, 0) / Math.max(1, values.length - 7);
    const trend = recentAvg > olderAvg * 1.05 ? 'up' : recentAvg < olderAvg * 0.95 ? 'down' : 'stable';
    return { avg: avg.toFixed(1), min, max, trend: trend as 'up' | 'down' | 'stable' };
  };

  if (!user) {
    return (
      <div className="container mx-auto p-6 max-w-7xl">
        <Card>
          <CardContent className="p-12 text-center">
            <AlertCircle className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h2 className="text-xl font-semibold mb-2">Sign In Required</h2>
            <p className="text-muted-foreground mb-4">Please sign in to view your health history.</p>
            <Link href="/auth">
              <Button data-testid="button-sign-in">Sign In</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3" data-testid="text-page-title">
            <Activity className="h-8 w-8 text-primary" />
            Daily Follow-up History
          </h1>
          <p className="text-muted-foreground">
            Track your health trends over time across all wellness categories
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[140px]" data-testid="select-time-range">
              <Calendar className="h-4 w-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TIME_RANGES.map(range => (
                <SelectItem key={range.value} value={range.value}>
                  {range.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <LegalDisclaimer />

      {isNewPatient && (
        <Card className="border-primary/20 bg-primary/5">
          <CardContent className="p-8 text-center">
            <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <Heart className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-bold mb-2">Welcome to Your Health Journey</h2>
            <p className="text-muted-foreground mb-6 max-w-lg mx-auto">
              Start tracking your daily wellness to build a comprehensive picture of your health. 
              Complete activities in any tab below to begin seeing your trends and insights.
            </p>
            <Link href="/">
              <Button size="lg" className="gap-2" data-testid="button-go-to-dashboard">
                <Play className="h-5 w-5" />
                Go to Dashboard to Start
              </Button>
            </Link>
          </CardContent>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="w-full flex gap-2 h-auto p-2 flex-wrap">
          <TabsTrigger value="device" data-testid="tab-device-history" className="flex-1 min-w-[100px]">
            <Heart className="h-3 w-3 mr-1" />
            Device Data
          </TabsTrigger>
          <TabsTrigger value="symptoms" data-testid="tab-symptoms-history" className="flex-1 min-w-[100px]">
            <Activity className="h-3 w-3 mr-1" />
            Symptoms
          </TabsTrigger>
          <TabsTrigger value="video-ai" data-testid="tab-video-history" className="flex-1 min-w-[100px]">
            <Video className="h-3 w-3 mr-1" />
            Video AI
          </TabsTrigger>
          <TabsTrigger value="audio-ai" data-testid="tab-audio-history" className="flex-1 min-w-[100px]">
            <Mic className="h-3 w-3 mr-1" />
            Audio AI
          </TabsTrigger>
          <TabsTrigger value="paintrack" data-testid="tab-paintrack-history" className="flex-1 min-w-[100px]">
            <Zap className="h-3 w-3 mr-1" />
            PainTrack
          </TabsTrigger>
          <TabsTrigger value="mental-health" data-testid="tab-mental-health-history" className="flex-1 min-w-[100px]">
            <Brain className="h-3 w-3 mr-1" />
            Mental Health
          </TabsTrigger>
        </TabsList>

        <TabsContent value="device" className="space-y-4">
          {deviceLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : filteredDeviceHistory?.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Avg Heart Rate" 
                  value={calculateStats(filteredDeviceHistory, 'heartRate').avg} 
                  unit="bpm"
                  icon={Heart}
                  trend={calculateStats(filteredDeviceHistory, 'heartRate').trend}
                  color="text-rose-500"
                />
                <StatCard 
                  label="Avg SpO2" 
                  value={calculateStats(filteredDeviceHistory, 'oxygenSaturation').avg} 
                  unit="%"
                  icon={Wind}
                  trend={calculateStats(filteredDeviceHistory, 'oxygenSaturation').trend}
                  color="text-blue-500"
                />
                <StatCard 
                  label="Avg Temperature" 
                  value={calculateStats(filteredDeviceHistory, 'temperature').avg} 
                  unit="°F"
                  icon={Thermometer}
                  trend={calculateStats(filteredDeviceHistory, 'temperature').trend}
                  color="text-orange-500"
                />
                <StatCard 
                  label="Total Steps" 
                  value={filteredDeviceHistory.reduce((sum, d) => sum + (d.steps || 0), 0).toLocaleString()} 
                  icon={Footprints}
                  color="text-chart-2"
                />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Device Data Trends
                  </CardTitle>
                  <CardDescription>
                    {filteredDeviceHistory.length} data points over {selectedRange.label.toLowerCase()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={deviceChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" className="text-xs" />
                        <YAxis className="text-xs" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px'
                          }} 
                        />
                        <Legend />
                        <Line type="monotone" dataKey="heartRate" name="Heart Rate (bpm)" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="spo2" name="SpO2 (%)" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Entries</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[200px]">
                    <div className="space-y-2">
                      {filteredDeviceHistory.slice(0, 10).map((entry: any, idx: number) => (
                        <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                          <div className="flex items-center gap-3">
                            <Clock className="h-4 w-4 text-muted-foreground" />
                            <span className="text-sm">{format(new Date(entry.createdAt || entry.date), 'MMM d, yyyy h:mm a')}</span>
                          </div>
                          <div className="flex items-center gap-4 text-sm">
                            {entry.heartRate && <span><Heart className="h-3 w-3 inline mr-1 text-rose-500" />{entry.heartRate} bpm</span>}
                            {entry.oxygenSaturation && <span><Wind className="h-3 w-3 inline mr-1 text-blue-500" />{entry.oxygenSaturation}%</span>}
                            {entry.steps && <span><Footprints className="h-3 w-3 inline mr-1 text-chart-2" />{entry.steps.toLocaleString()}</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Heart}
              title="No Device Data Yet"
              description="Sync your wearable device to start tracking heart rate, SpO2, temperature, and activity levels."
              actionLabel="Connect Wearable"
              actionHref="/wearables"
            />
          )}
        </TabsContent>

        <TabsContent value="symptoms" className="space-y-4">
          {symptomsLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : filteredSymptoms?.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Avg Pain Level" 
                  value={calculateStats(filteredSymptoms, 'painLevel').avg} 
                  unit="/10"
                  icon={Zap}
                  trend={calculateStats(filteredSymptoms, 'painLevel').trend}
                  color="text-orange-500"
                />
                <StatCard 
                  label="Avg Fatigue" 
                  value={calculateStats(filteredSymptoms, 'fatigueLevel').avg} 
                  unit="/10"
                  icon={Activity}
                  trend={calculateStats(filteredSymptoms, 'fatigueLevel').trend}
                  color="text-yellow-500"
                />
                <StatCard 
                  label="Check-ins" 
                  value={filteredSymptoms.length} 
                  icon={CheckCircle2}
                  color="text-chart-2"
                />
                <StatCard 
                  label="Avg Mood" 
                  value={(() => {
                    const moodValues = filteredSymptoms.filter((s: any) => s.mood).map((s: any) => {
                      const moodMap: Record<string, number> = { poor: 1, fair: 2, good: 3, great: 4, excellent: 5 };
                      return moodMap[s.mood?.toLowerCase()] || 3;
                    });
                    return moodValues.length > 0 ? (moodValues.reduce((a, b) => a + b, 0) / moodValues.length).toFixed(1) : '-';
                  })()} 
                  unit="/5"
                  icon={Smile}
                  color="text-primary"
                />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Symptom Trends
                  </CardTitle>
                  <CardDescription>
                    Pain and fatigue levels over {selectedRange.label.toLowerCase()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={symptomChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" className="text-xs" />
                        <YAxis domain={[0, 10]} className="text-xs" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px'
                          }} 
                        />
                        <Legend />
                        <Area type="monotone" dataKey="painLevel" name="Pain Level" stroke="hsl(var(--chart-1))" fill="hsl(var(--chart-1) / 0.2)" />
                        <Area type="monotone" dataKey="fatigueLevel" name="Fatigue Level" stroke="hsl(var(--chart-3))" fill="hsl(var(--chart-3) / 0.2)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Symptom Check-ins</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[250px]">
                    <div className="space-y-3">
                      {filteredSymptoms.slice(0, 15).map((symptom: any, idx: number) => (
                        <div key={idx} className="p-3 rounded-lg border bg-card">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium">{format(new Date(symptom.timestamp), 'MMM d, yyyy h:mm a')}</span>
                            <Badge variant="secondary" className="text-xs">Patient-reported</Badge>
                          </div>
                          <div className="grid grid-cols-3 gap-2 text-sm">
                            {symptom.painLevel != null && (
                              <div><span className="text-muted-foreground">Pain:</span> {symptom.painLevel}/10</div>
                            )}
                            {symptom.fatigueLevel != null && (
                              <div><span className="text-muted-foreground">Fatigue:</span> {symptom.fatigueLevel}/10</div>
                            )}
                            {symptom.mood && (
                              <div><span className="text-muted-foreground">Mood:</span> {symptom.mood}</div>
                            )}
                          </div>
                          {symptom.note && (
                            <p className="text-xs text-muted-foreground mt-2 italic">"{symptom.note}"</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Activity}
              title="No Symptom Check-ins Yet"
              description="Log your daily symptoms including pain, fatigue, mood, and other health indicators to track patterns over time."
              actionLabel="Start Symptom Check-in"
              actionHref="/"
            />
          )}
        </TabsContent>

        <TabsContent value="video-ai" className="space-y-4">
          {videoLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : videoMetrics ? (
            <>
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <Video className="h-5 w-5 text-primary" />
                        Latest Video AI Analysis
                      </CardTitle>
                      <CardDescription>
                        Last examination: {videoMetrics.created_at ? format(new Date(videoMetrics.created_at), 'MMM d, yyyy h:mm a') : 'N/A'}
                      </CardDescription>
                    </div>
                    <Link href="/ai-video">
                      <Button variant="outline" size="sm" className="gap-2" data-testid="button-view-full-video-analysis">
                        View Full Analysis
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </Link>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {videoMetrics.respiratory_rate_bpm && (
                      <div className="p-4 rounded-lg border bg-muted/30">
                        <div className="flex items-center gap-2 mb-2">
                          <Wind className="h-4 w-4 text-blue-500" />
                          <span className="text-sm text-muted-foreground">Respiratory Rate</span>
                        </div>
                        <div className="text-2xl font-bold">{videoMetrics.respiratory_rate_bpm.toFixed(1)} <span className="text-sm font-normal">bpm</span></div>
                      </div>
                    )}
                    {videoMetrics.skin_pallor_score != null && (
                      <div className="p-4 rounded-lg border bg-muted/30">
                        <div className="flex items-center gap-2 mb-2">
                          <Palette className="h-4 w-4 text-amber-500" />
                          <span className="text-sm text-muted-foreground">Skin Pallor</span>
                        </div>
                        <div className="text-2xl font-bold">{videoMetrics.skin_pallor_score.toFixed(1)}<span className="text-sm font-normal">/100</span></div>
                      </div>
                    )}
                    {videoMetrics.jaundice_risk_level && (
                      <div className="p-4 rounded-lg border bg-muted/30">
                        <div className="flex items-center gap-2 mb-2">
                          <Eye className="h-4 w-4 text-yellow-500" />
                          <span className="text-sm text-muted-foreground">Jaundice Risk</span>
                        </div>
                        <div className="text-2xl font-bold capitalize">{videoMetrics.jaundice_risk_level}</div>
                      </div>
                    )}
                    {videoMetrics.facial_swelling_score != null && (
                      <div className="p-4 rounded-lg border bg-muted/30">
                        <div className="flex items-center gap-2 mb-2">
                          <User className="h-4 w-4 text-rose-500" />
                          <span className="text-sm text-muted-foreground">Facial Swelling</span>
                        </div>
                        <div className="text-2xl font-bold">{videoMetrics.facial_swelling_score.toFixed(1)}<span className="text-sm font-normal">/100</span></div>
                      </div>
                    )}
                    {videoMetrics.tremor_detected !== undefined && (
                      <div className="p-4 rounded-lg border bg-muted/30">
                        <div className="flex items-center gap-2 mb-2">
                          <Hand className="h-4 w-4 text-purple-500" />
                          <span className="text-sm text-muted-foreground">Tremor</span>
                        </div>
                        <div className="text-2xl font-bold">{videoMetrics.tremor_detected ? 'Detected' : 'None'}</div>
                      </div>
                    )}
                    {videoMetrics.tongue_coating_detected !== undefined && (
                      <div className="p-4 rounded-lg border bg-muted/30">
                        <div className="flex items-center gap-2 mb-2">
                          <MessageSquare className="h-4 w-4 text-pink-500" />
                          <span className="text-sm text-muted-foreground">Tongue Analysis</span>
                        </div>
                        <div className="text-2xl font-bold">{videoMetrics.tongue_coating_detected ? 'Coating' : 'Normal'}</div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Start New Video Examination</CardTitle>
                  <CardDescription>Record a 7-stage AI-guided video examination to track changes</CardDescription>
                </CardHeader>
                <CardContent>
                  <Link href="/daily-followup">
                    <Button className="w-full gap-2" data-testid="button-start-video-exam">
                      <Camera className="h-4 w-4" />
                      Start Video Examination
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Video}
              title="No Video AI Analysis Yet"
              description="Complete a guided video examination to get AI-powered analysis of respiratory rate, skin pallor, jaundice indicators, and more."
              actionLabel="Start Video Examination"
              actionHref="/daily-followup"
            />
          )}
        </TabsContent>

        <TabsContent value="audio-ai" className="space-y-4">
          {audioLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : filteredVoice?.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Voice Recordings" 
                  value={filteredVoice.length} 
                  icon={Mic}
                  color="text-purple-500"
                />
                <StatCard 
                  label="Follow-ups Needed" 
                  value={filteredVoice.filter((v: any) => v.needsFollowup).length} 
                  icon={AlertTriangle}
                  color="text-amber-500"
                />
                <StatCard 
                  label="Avg Empathy Level" 
                  value={(() => {
                    const levels = filteredVoice.filter((v: any) => v.empathyLevel).map((v: any) => v.empathyLevel);
                    return levels.length > 0 ? (levels.reduce((a: number, b: number) => a + b, 0) / levels.length).toFixed(1) : '-';
                  })()} 
                  unit="/10"
                  icon={Heart}
                  color="text-rose-500"
                />
                <StatCard 
                  label="Symptoms Extracted" 
                  value={filteredVoice.reduce((sum: number, v: any) => sum + (v.extractedSymptoms?.length || 0), 0)} 
                  icon={Activity}
                  color="text-primary"
                />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Voice Follow-up History</CardTitle>
                  <CardDescription>AI analysis of your voice recordings</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[350px]">
                    <div className="space-y-4">
                      {filteredVoice.map((voice: any, idx: number) => (
                        <div key={idx} className="p-4 rounded-lg border bg-card">
                          <div className="flex items-center justify-between mb-3">
                            <span className="text-sm font-medium">{format(new Date(voice.createdAt), 'MMM d, yyyy h:mm a')}</span>
                            {voice.needsFollowup && (
                              <Badge variant="destructive" className="text-xs">Needs Follow-up</Badge>
                            )}
                          </div>
                          {voice.conversationSummary && (
                            <p className="text-sm mb-3">{voice.conversationSummary}</p>
                          )}
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            {voice.extractedMood && (
                              <div><span className="text-muted-foreground">Mood:</span> {voice.extractedMood}</div>
                            )}
                            {voice.empathyLevel && (
                              <div><span className="text-muted-foreground">Empathy:</span> {voice.empathyLevel}/10</div>
                            )}
                          </div>
                          {voice.extractedSymptoms?.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {voice.extractedSymptoms.map((symptom: string, sIdx: number) => (
                                <Badge key={sIdx} variant="secondary" className="text-xs">{symptom}</Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Mic}
              title="No Audio AI Analysis Yet"
              description="Record voice follow-ups to let AI analyze your speech patterns, extract symptoms, and track your emotional well-being."
              actionLabel="Start Voice Recording"
              actionHref="/"
            />
          )}
        </TabsContent>

        <TabsContent value="paintrack" className="space-y-4">
          {paintrackLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : filteredPaintrack?.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Avg Pain (VAS)" 
                  value={calculateStats(filteredPaintrack, 'patientVas').avg} 
                  unit="/10"
                  icon={Zap}
                  trend={calculateStats(filteredPaintrack, 'patientVas').trend}
                  color="text-orange-500"
                />
                <StatCard 
                  label="Sessions" 
                  value={filteredPaintrack.length} 
                  icon={FileText}
                  color="text-primary"
                />
                <StatCard 
                  label="Max Pain" 
                  value={calculateStats(filteredPaintrack, 'patientVas').max} 
                  unit="/10"
                  icon={TrendingUp}
                  color="text-destructive"
                />
                <StatCard 
                  label="Min Pain" 
                  value={calculateStats(filteredPaintrack, 'patientVas').min} 
                  unit="/10"
                  icon={TrendingDown}
                  color="text-chart-2"
                />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Pain Level Trends
                  </CardTitle>
                  <CardDescription>
                    VAS scores over {selectedRange.label.toLowerCase()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={painChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" className="text-xs" />
                        <YAxis domain={[0, 10]} className="text-xs" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px'
                          }}
                          formatter={(value: any, name: any, props: any) => [
                            `${value}/10 (${props.payload.joint})`,
                            'Pain Level'
                          ]}
                        />
                        <Bar dataKey="vas" name="Pain Level" fill="hsl(var(--chart-1))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Pain Assessments</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[250px]">
                    <div className="space-y-3">
                      {filteredPaintrack.slice(0, 15).map((session: any, idx: number) => (
                        <div key={idx} className="p-3 rounded-lg border bg-card">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Zap className="h-4 w-4 text-orange-500" />
                              <span className="font-medium capitalize">{session.laterality} {session.joint}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-2xl font-bold">{session.patientVas}</span>
                              <span className="text-sm text-muted-foreground">/10</span>
                            </div>
                          </div>
                          <div className="flex items-center justify-between text-sm text-muted-foreground">
                            <span>{format(new Date(session.createdAt), 'MMM d, yyyy h:mm a')}</span>
                            <span className="capitalize">{session.module}</span>
                          </div>
                          {session.patientNotes && (
                            <p className="text-xs text-muted-foreground mt-2 italic">"{session.patientNotes}"</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Zap}
              title="No Pain Tracking Data Yet"
              description="Track chronic pain with detailed assessments including joint selection, medication tracking, and video documentation."
              actionLabel="Start Pain Assessment"
              actionHref="/"
            />
          )}
        </TabsContent>

        <TabsContent value="mental-health" className="space-y-4">
          {mentalHealthLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : filteredMentalHealth?.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Assessments" 
                  value={filteredMentalHealth.length} 
                  icon={Brain}
                  color="text-purple-500"
                />
                <StatCard 
                  label="Avg Score" 
                  value={calculateStats(filteredMentalHealth, 'total_score').avg} 
                  icon={BarChart3}
                  color="text-primary"
                />
                <StatCard 
                  label="PHQ-9 Count" 
                  value={filteredMentalHealth.filter((m: any) => m.questionnaire_type === 'PHQ-9').length} 
                  icon={FileText}
                  color="text-blue-500"
                />
                <StatCard 
                  label="GAD-7 Count" 
                  value={filteredMentalHealth.filter((m: any) => m.questionnaire_type === 'GAD-7').length} 
                  icon={FileText}
                  color="text-amber-500"
                />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Mental Health Score Trends
                  </CardTitle>
                  <CardDescription>
                    Questionnaire scores over {selectedRange.label.toLowerCase()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={mentalHealthChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" className="text-xs" />
                        <YAxis className="text-xs" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px'
                          }}
                          formatter={(value: any, name: any, props: any) => [
                            `${value} (${props.payload.type})`,
                            'Score'
                          ]}
                        />
                        <Legend />
                        <Line type="monotone" dataKey="score" name="Score" stroke="hsl(var(--chart-4))" strokeWidth={2} dot />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Assessment History</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[250px]">
                    <div className="space-y-3">
                      {filteredMentalHealth.map((assessment: any, idx: number) => (
                        <div key={idx} className="p-3 rounded-lg border bg-card">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Brain className="h-4 w-4 text-purple-500" />
                              <span className="font-medium">{assessment.questionnaire_type}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-2xl font-bold">{assessment.total_score}</span>
                              <span className="text-sm text-muted-foreground">/{assessment.max_score}</span>
                            </div>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">
                              {format(new Date(assessment.completed_at), 'MMM d, yyyy h:mm a')}
                            </span>
                            <Badge className={getSeverityColor(assessment.severity_level)}>
                              {assessment.severity_level?.replace(/_/g, ' ')}
                            </Badge>
                          </div>
                          {assessment.crisis_detected && (
                            <Alert className="mt-2 p-2 border-destructive/50 bg-destructive/5">
                              <AlertTriangle className="h-3 w-3 text-destructive" />
                              <AlertDescription className="text-xs text-destructive">
                                Crisis indicators were detected in this assessment
                              </AlertDescription>
                            </Alert>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>

              <Link href="/mental-health">
                <Card className="hover-elevate cursor-pointer">
                  <CardContent className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Brain className="h-5 w-5 text-primary" />
                      <div>
                        <p className="font-medium">View Full Mental Health Dashboard</p>
                        <p className="text-sm text-muted-foreground">Access detailed analysis, clusters, and AI insights</p>
                      </div>
                    </div>
                    <ChevronRight className="h-5 w-5 text-muted-foreground" />
                  </CardContent>
                </Card>
              </Link>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Brain}
              title="No Mental Health Assessments Yet"
              description="Complete PHQ-9, GAD-7, or PSS-10 questionnaires to track your mental wellness indicators over time."
              actionLabel="Start Assessment"
              actionHref="/"
            />
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
