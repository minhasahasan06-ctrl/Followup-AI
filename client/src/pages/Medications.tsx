import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useAuth } from "@/contexts/AuthContext";
import {
  Pill,
  AlertTriangle,
  Clock,
  Activity,
  RefreshCw,
  Check,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Calendar,
  Shield,
  FileText,
  Infinity,
  Timer,
  PlayCircle,
  Stethoscope,
  Ban,
  ChevronDown,
  ChevronRight,
  Sparkles,
  User,
  ArrowRight,
  CalendarDays,
  Archive,
  TrendingUp,
  TrendingDown,
  Loader2,
  Target,
  Flame,
  Award,
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { format, differenceInDays } from "date-fns";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar as CalendarComponent } from "@/components/ui/calendar";

interface MedicationDashboard {
  activeMedications: ChronicMedication[];
  medicationsBySpecialty: Record<string, ChronicMedication[]>;
  prescriptions: any[];
  conflicts: MedicationConflict[];
  hasFreezeConflicts: boolean;
  interactionAlerts: any[];
  reminders: Reminder[];
  needsStartConfirmation: NeedsConfirmation[];
  archivedMedications: ChronicMedication[];
  doctors: Record<string, { id: string; name: string; email: string }>;
  summary: DashboardSummary;
}

interface DashboardSummary {
  total: number;
  continuous: number;
  duration: number;
  pendingConflicts: number;
  activeInteractions: number;
  expiringSoon: number;
  needsConfirmation: number;
}

interface Reminder {
  id: string;
  name: string;
  type: string;
  daysRemaining: number;
  message: string;
}

interface NeedsConfirmation {
  id: string;
  name: string;
  dosage: string;
  intendedStartDate: string;
  specialty: string;
}

interface MedicationConflict {
  id: string;
  patientId: string;
  medication1Id: string;
  medication2Id: string;
  doctor1Id: string;
  doctor2Id: string;
  specialty1: string;
  specialty2: string;
  conflictType: string;
  severity: string;
  description: string;
  status: 'pending' | 'resolved';
  doctor1Response?: string;
  doctor2Response?: string;
  resolution?: string;
  createdAt: string;
}

interface ChronicMedication {
  id: string;
  name: string;
  dosage: string;
  frequency: string;
  specialty?: string;
  isContinuous?: boolean;
  durationDays?: number;
  intendedStartDate?: string;
  actualStartDate?: string;
  computedEndDate?: string;
  status: string;
  conflictStatus?: string;
  conflictGroupId?: string;
  prescribingDoctorId?: string;
  prescribingDoctorName?: string;
  daysRemaining?: number | null;
  hasConflict?: boolean;
  active: boolean;
  drugId?: string;
  rxcui?: string;
  notes?: string;
  createdAt?: string;
}

interface AdherenceStats {
  overallAdherenceRate: number;
  weeklyAdherenceRate: number;
  currentStreak: number;
  bestStreak: number;
  totalDosesTaken: number;
  totalDosesMissed: number;
  lastLoggedAt?: string;
  medicationBreakdown: Array<{
    medicationId: string;
    medicationName: string;
    adherenceRate: number;
    dosesTaken: number;
    dosesScheduled: number;
  }>;
}

function getRemainingDays(endDate: string | undefined | null): number | null {
  if (!endDate) return null;
  const end = new Date(endDate);
  const today = new Date();
  return differenceInDays(end, today);
}

function formatDuration(med: ChronicMedication): string {
  if (med.isContinuous) {
    return "Continuous";
  }
  
  const remaining = med.daysRemaining ?? getRemainingDays(med.computedEndDate);
  if (remaining === null) {
    return med.durationDays ? `${med.durationDays} day course` : "Duration not set";
  }
  
  if (remaining < 0) {
    return "Completed";
  }
  
  if (remaining === 0) {
    return "Last day";
  }
  
  return `${remaining} day${remaining !== 1 ? 's' : ''} left`;
}

function getSpecialtyColor(specialty: string): string {
  const colors: Record<string, string> = {
    cardiology: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300 border-red-200 dark:border-red-800",
    oncology: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300 border-purple-200 dark:border-purple-800",
    neurology: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300 border-blue-200 dark:border-blue-800",
    rheumatology: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300 border-amber-200 dark:border-amber-800",
    immunology: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300 border-green-200 dark:border-green-800",
    endocrinology: "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-300 border-cyan-200 dark:border-cyan-800",
    gastroenterology: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300 border-orange-200 dark:border-orange-800",
    pulmonology: "bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-300 border-sky-200 dark:border-sky-800",
    nephrology: "bg-teal-100 text-teal-700 dark:bg-teal-900/40 dark:text-teal-300 border-teal-200 dark:border-teal-800",
    psychiatry: "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800",
    "general medicine": "bg-gray-100 text-gray-700 dark:bg-gray-800/60 dark:text-gray-300 border-gray-200 dark:border-gray-700",
    unspecified: "bg-gray-100 text-gray-700 dark:bg-gray-800/60 dark:text-gray-300 border-gray-200 dark:border-gray-700",
  };
  return colors[specialty?.toLowerCase()] || colors.unspecified;
}

function getSpecialtyIcon(specialty: string) {
  const icons: Record<string, any> = {
    cardiology: Activity,
    oncology: Shield,
    neurology: Sparkles,
    psychiatry: Sparkles,
  };
  return icons[specialty?.toLowerCase()] || Stethoscope;
}

export default function Medications() {
  const { toast } = useToast();
  const { user } = useAuth();
  
  const [confirmStartDialog, setConfirmStartDialog] = useState<{ open: boolean; medication: ChronicMedication | null }>({
    open: false,
    medication: null,
  });
  const [selectedStartDate, setSelectedStartDate] = useState<Date | undefined>(undefined);
  const [expandedSpecialties, setExpandedSpecialties] = useState<string[]>([]);
  const [showArchivedMeds, setShowArchivedMeds] = useState(false);
  const [activeTab, setActiveTab] = useState("medications");

  const { data: dashboard, isLoading: dashboardLoading, refetch } = useQuery<MedicationDashboard>({
    queryKey: ['/api/medications/dashboard'],
  });

  const { data: interactionAlerts } = useQuery<any[]>({
    queryKey: ['/api/drug-interactions/alerts'],
  });

  const { data: adherenceStats, isLoading: adherenceLoading } = useQuery<AdherenceStats>({
    queryKey: ['/api/v1/medication-adherence/stats'],
    queryFn: async () => {
      const res = await fetch('/api/v1/medication-adherence/stats');
      if (!res.ok) return null;
      return res.json();
    },
  });

  const logAdherenceMutation = useMutation({
    mutationFn: async ({ medicationId, status, notes }: { 
      medicationId: string; 
      status: 'taken' | 'missed' | 'skipped' | 'late';
      notes?: string;
    }) => {
      const now = new Date().toISOString();
      return apiRequest('POST', '/api/v1/medication-adherence/log', {
        medication_id: medicationId,
        scheduled_time: now,
        taken_at: status === 'taken' || status === 'late' ? now : null,
        status,
        notes,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/medication-adherence/stats'] });
      queryClient.invalidateQueries({ queryKey: ['/api/medications/dashboard'] });
      toast({
        title: "Adherence Logged",
        description: "Your medication intake has been recorded.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to log adherence. Please try again.",
        variant: "destructive",
      });
    },
  });

  const confirmStartMutation = useMutation({
    mutationFn: async ({ id, actualStartDate }: { id: string; actualStartDate: string }) => {
      return apiRequest('POST', `/api/medications/${id}/confirm-start`, { actualStartDate });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/medications/dashboard'] });
      queryClient.invalidateQueries({ queryKey: ['/api/medications'] });
      setConfirmStartDialog({ open: false, medication: null });
      setSelectedStartDate(undefined);
      toast({
        title: "Start Date Confirmed",
        description: "Your medication schedule has been updated.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to confirm start date. Please try again.",
        variant: "destructive",
      });
    },
  });

  const autoArchiveMutation = useMutation({
    mutationFn: async () => {
      return apiRequest('POST', '/api/medications/auto-archive', {});
    },
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['/api/medications/dashboard'] });
      toast({
        title: "Auto-Archive Complete",
        description: data.message || "Expired medications have been archived.",
      });
    },
  });

  const markTakenMutation = useMutation({
    mutationFn: async ({ id }: { id: string }) => {
      return apiRequest('POST', `/api/medications/${id}/mark-taken`, {
        takenAt: new Date().toISOString(),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/medications/dashboard'] });
      toast({
        title: "Dose Recorded",
        description: "Your medication dose has been recorded.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to record dose. Please try again.",
        variant: "destructive",
      });
    },
  });

  const requestRefillMutation = useMutation({
    mutationFn: async ({ id }: { id: string }) => {
      return apiRequest('POST', `/api/medications/${id}/request-refill`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/medications/dashboard'] });
      toast({
        title: "Refill Requested",
        description: "Your doctor has been notified of your refill request.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to request refill. Please try again.",
        variant: "destructive",
      });
    },
  });

  const toggleSpecialtyExpand = (specialty: string) => {
    setExpandedSpecialties(prev =>
      prev.includes(specialty)
        ? prev.filter(s => s !== specialty)
        : [...prev, specialty]
    );
  };

  const handleConfirmStart = () => {
    if (!confirmStartDialog.medication || !selectedStartDate) return;
    confirmStartMutation.mutate({
      id: confirmStartDialog.medication.id,
      actualStartDate: selectedStartDate.toISOString(),
    });
  };

  const specialties = dashboard?.medicationsBySpecialty ? Object.keys(dashboard.medicationsBySpecialty) : [];

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">My Medications</h1>
            <p className="text-muted-foreground">
              Unified view of all your prescriptions from multiple specialists
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => autoArchiveMutation.mutate()}
            disabled={autoArchiveMutation.isPending}
            data-testid="button-auto-archive"
          >
            {autoArchiveMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Archive className="h-4 w-4 mr-2" />
            )}
            Auto-Archive Expired
          </Button>
        </div>

        {dashboardLoading ? (
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-4">
              {[1, 2, 3, 4].map(i => <Skeleton key={i} className="h-24" />)}
            </div>
            <Skeleton className="h-64" />
          </div>
        ) : (
          <>
            {dashboard?.hasFreezeConflicts && (
              <Alert variant="destructive" className="border-2" data-testid="alert-freeze-conflict">
                <AlertTriangle className="h-5 w-5" />
                <AlertTitle className="text-lg">Cross-Specialty Conflict Detected</AlertTitle>
                <AlertDescription className="mt-2">
                  <p>
                    Your prescriptions from different specialists may have a dangerous interaction.
                    <strong> These medications are on hold</strong> until your doctors resolve the conflict.
                  </p>
                  <div className="mt-3 p-3 bg-destructive/10 rounded-md">
                    {dashboard.conflicts.map(conflict => (
                      <div key={conflict.id} className="flex items-center gap-2 text-sm">
                        <Ban className="h-4 w-4" />
                        <span>
                          <strong>{conflict.specialty1}</strong> vs <strong>{conflict.specialty2}</strong>:
                          {conflict.description}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-3 text-sm">
                    Your doctors have been notified and are working to resolve this. Do not start these medications until cleared.
                  </p>
                </AlertDescription>
              </Alert>
            )}

            {dashboard?.needsStartConfirmation && dashboard.needsStartConfirmation.length > 0 && (
              <Alert className="border-amber-500 bg-amber-50 dark:bg-amber-950/30" data-testid="alert-needs-confirmation">
                <CalendarDays className="h-5 w-5 text-amber-600" />
                <AlertTitle className="text-amber-800 dark:text-amber-300">When Will You Start?</AlertTitle>
                <AlertDescription className="mt-2">
                  <p className="text-amber-700 dark:text-amber-400 mb-3">
                    Please confirm when you will start taking these newly prescribed medications:
                  </p>
                  <div className="space-y-2">
                    {dashboard.needsStartConfirmation.map(med => (
                      <div
                        key={med.id}
                        className="flex items-center justify-between p-3 bg-white dark:bg-gray-900 rounded-md border border-amber-200 dark:border-amber-800"
                      >
                        <div>
                          <span className="font-medium">{med.name}</span>
                          <span className="text-sm text-muted-foreground ml-2">{med.dosage}</span>
                          <Badge variant="outline" className={`ml-2 text-xs ${getSpecialtyColor(med.specialty)}`}>
                            {med.specialty}
                          </Badge>
                        </div>
                        <Button
                          size="sm"
                          onClick={() => {
                            const fullMed = dashboard.activeMedications.find(m => m.id === med.id);
                            if (fullMed) {
                              setConfirmStartDialog({ open: true, medication: fullMed });
                              setSelectedStartDate(med.intendedStartDate ? new Date(med.intendedStartDate) : new Date());
                            }
                          }}
                          data-testid={`button-confirm-start-${med.id}`}
                        >
                          <PlayCircle className="h-4 w-4 mr-1" />
                          Set Start Date
                        </Button>
                      </div>
                    ))}
                  </div>
                </AlertDescription>
              </Alert>
            )}

            <div className="grid gap-4 md:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Medications</CardTitle>
                  <Pill className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold" data-testid="count-medications">
                    {dashboard?.summary.total || 0}
                  </div>
                  <div className="flex gap-2 mt-1">
                    <span className="text-xs text-muted-foreground flex items-center gap-1">
                      <Infinity className="h-3 w-3" /> {dashboard?.summary.continuous || 0} continuous
                    </span>
                    <span className="text-xs text-muted-foreground flex items-center gap-1">
                      <Timer className="h-3 w-3" /> {dashboard?.summary.duration || 0} timed
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Ending Soon</CardTitle>
                  <Clock className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold" data-testid="count-expiring">
                    {dashboard?.summary.expiringSoon || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">Medications ending within 7 days</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Interaction Alerts</CardTitle>
                  <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold" data-testid="count-interactions">
                    {dashboard?.summary.activeInteractions || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {(dashboard?.summary.activeInteractions || 0) > 0 ? "Review recommended" : "All clear"}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Pending Conflicts</CardTitle>
                  <Shield className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl font-bold ${(dashboard?.summary.pendingConflicts || 0) > 0 ? 'text-destructive' : ''}`} data-testid="count-conflicts">
                    {dashboard?.summary.pendingConflicts || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {(dashboard?.summary.pendingConflicts || 0) > 0 ? "Awaiting doctor resolution" : "No conflicts"}
                  </p>
                </CardContent>
              </Card>
            </div>

            {adherenceStats && (
              <Card className="border-green-200 dark:border-green-800 bg-green-50/50 dark:bg-green-950/20">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2 text-green-700 dark:text-green-300">
                    <Target className="h-5 w-5" />
                    Medication Adherence Tracking
                  </CardTitle>
                  <CardDescription className="text-green-600 dark:text-green-400">
                    Your medication-taking consistency helps your wellness monitoring
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-4">
                    <div className="flex items-center gap-3 p-3 bg-white dark:bg-gray-900 rounded-md border">
                      <div className="p-2 rounded-full bg-green-100 dark:bg-green-900/40">
                        <Target className="h-5 w-5 text-green-600 dark:text-green-400" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {Math.round(adherenceStats.overallAdherenceRate)}%
                        </p>
                        <p className="text-xs text-muted-foreground">Overall Adherence</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 bg-white dark:bg-gray-900 rounded-md border">
                      <div className="p-2 rounded-full bg-blue-100 dark:bg-blue-900/40">
                        <TrendingUp className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                          {Math.round(adherenceStats.weeklyAdherenceRate)}%
                        </p>
                        <p className="text-xs text-muted-foreground">This Week</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 bg-white dark:bg-gray-900 rounded-md border">
                      <div className="p-2 rounded-full bg-orange-100 dark:bg-orange-900/40">
                        <Flame className="h-5 w-5 text-orange-600 dark:text-orange-400" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                          {adherenceStats.currentStreak}
                        </p>
                        <p className="text-xs text-muted-foreground">Day Streak</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 bg-white dark:bg-gray-900 rounded-md border">
                      <div className="p-2 rounded-full bg-purple-100 dark:bg-purple-900/40">
                        <Award className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                          {adherenceStats.bestStreak}
                        </p>
                        <p className="text-xs text-muted-foreground">Best Streak</p>
                      </div>
                    </div>
                  </div>

                  {adherenceStats.medicationBreakdown && adherenceStats.medicationBreakdown.length > 0 && (
                    <div className="mt-4 space-y-3">
                      <h4 className="font-medium text-sm">Per-Medication Adherence</h4>
                      <div className="grid gap-2 md:grid-cols-2">
                        {adherenceStats.medicationBreakdown.map((med, idx) => (
                          <div
                            key={med.medicationId || idx}
                            className="flex items-center gap-3 p-3 bg-white dark:bg-gray-900 rounded-md border"
                            data-testid={`adherence-med-${med.medicationId}`}
                          >
                            <div className="flex-1 min-w-0">
                              <p className="font-medium text-sm truncate">{med.medicationName}</p>
                              <div className="flex items-center gap-2 mt-1">
                                <Progress
                                  value={med.adherenceRate}
                                  className="h-2 flex-1"
                                />
                                <span className={`text-xs font-medium ${
                                  med.adherenceRate >= 80 ? 'text-green-600' :
                                  med.adherenceRate >= 50 ? 'text-amber-600' : 'text-red-600'
                                }`}>
                                  {Math.round(med.adherenceRate)}%
                                </span>
                              </div>
                              <p className="text-xs text-muted-foreground mt-1">
                                {med.dosesTaken} of {med.dosesScheduled} doses taken
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <p className="text-xs text-muted-foreground mt-4 text-center italic">
                    Wellness monitoring - Not medical advice
                  </p>
                </CardContent>
              </Card>
            )}

            {dashboard?.reminders && dashboard.reminders.length > 0 && (
              <Card className="border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-950/20">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2 text-blue-700 dark:text-blue-300">
                    <Clock className="h-5 w-5" />
                    Medication Reminders
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
                    {dashboard.reminders.map(reminder => (
                      <div
                        key={reminder.id}
                        className="flex items-center gap-3 p-3 bg-white dark:bg-gray-900 rounded-md border"
                        data-testid={`reminder-${reminder.id}`}
                      >
                        <div className={`p-2 rounded-full ${reminder.daysRemaining <= 2 ? 'bg-red-100 dark:bg-red-900/40' : 'bg-amber-100 dark:bg-amber-900/40'}`}>
                          {reminder.daysRemaining <= 2 ? (
                            <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
                          ) : (
                            <Timer className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate">{reminder.name}</p>
                          <p className={`text-sm ${reminder.daysRemaining <= 2 ? 'text-red-600 dark:text-red-400 font-medium' : 'text-muted-foreground'}`}>
                            {reminder.message}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {interactionAlerts && interactionAlerts.length > 0 && (
              <Card className="border-orange-200 dark:border-orange-800 bg-orange-50/50 dark:bg-orange-950/20">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2 text-orange-700 dark:text-orange-300">
                    <AlertTriangle className="h-5 w-5" />
                    Drug Interaction Alerts
                  </CardTitle>
                  <CardDescription>
                    Potential interactions between your current medications
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {interactionAlerts.slice(0, 3).map((alert, idx) => (
                      <div
                        key={alert.id || idx}
                        className="flex items-start gap-3 p-3 bg-white dark:bg-gray-900 rounded-md border"
                        data-testid={`interaction-alert-${idx}`}
                      >
                        <Badge
                          variant={alert.severityLevel === 'severe' ? 'destructive' : 'secondary'}
                          className="shrink-0"
                        >
                          {alert.severityLevel}
                        </Badge>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium">
                            {alert.drug1Name || 'Drug 1'} + {alert.drug2Name || 'Drug 2'}
                          </p>
                          <p className="text-sm text-muted-foreground mt-1">
                            {alert.interactionDescription || 'Potential interaction detected'}
                          </p>
                        </div>
                      </div>
                    ))}
                    {interactionAlerts.length > 3 && (
                      <p className="text-sm text-muted-foreground text-center">
                        +{interactionAlerts.length - 3} more alerts
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}


            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Pill className="h-5 w-5" />
                  Current Medications by Specialty
                </CardTitle>
                <CardDescription>
                  All your active prescriptions organized by medical specialty
                </CardDescription>
              </CardHeader>
              <CardContent>
                {specialties.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <Pill className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p className="font-medium">No Active Medications</p>
                    <p className="text-sm mt-1">Your prescriptions will appear here once added by your doctor</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {specialties.map(specialty => {
                      const meds = dashboard?.medicationsBySpecialty[specialty] || [];
                      const isExpanded = expandedSpecialties.includes(specialty);
                      const SpecialtyIcon = getSpecialtyIcon(specialty);

                      return (
                        <Collapsible
                          key={specialty}
                          open={isExpanded || specialties.length <= 3}
                          onOpenChange={() => toggleSpecialtyExpand(specialty)}
                        >
                          <Card className="overflow-hidden">
                            <CollapsibleTrigger asChild>
                              <CardHeader
                                className={`cursor-pointer hover-elevate ${getSpecialtyColor(specialty)} border-b`}
                              >
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-3">
                                    <SpecialtyIcon className="h-5 w-5" />
                                    <div>
                                      <CardTitle className="text-lg capitalize">{specialty}</CardTitle>
                                      <CardDescription className="text-current opacity-80">
                                        {meds.length} medication{meds.length !== 1 ? 's' : ''}
                                      </CardDescription>
                                    </div>
                                  </div>
                                  {specialties.length > 3 && (
                                    isExpanded ? (
                                      <ChevronDown className="h-5 w-5" />
                                    ) : (
                                      <ChevronRight className="h-5 w-5" />
                                    )
                                  )}
                                </div>
                              </CardHeader>
                            </CollapsibleTrigger>
                            <CollapsibleContent>
                              <CardContent className="pt-4">
                                <div className="space-y-3">
                                  {meds.map((med) => (
                                    <div
                                      key={med.id}
                                      className={`p-4 rounded-lg border ${
                                        med.hasConflict
                                          ? 'border-red-300 bg-red-50/50 dark:border-red-800 dark:bg-red-950/20'
                                          : 'border-border bg-card'
                                      }`}
                                      data-testid={`medication-card-${med.id}`}
                                    >
                                      <div className="flex items-start justify-between gap-4">
                                        <div className="flex-1 min-w-0">
                                          <div className="flex items-center gap-2 flex-wrap">
                                            <h4 className="font-semibold text-lg">{med.name}</h4>
                                            {med.isContinuous ? (
                                              <Badge variant="secondary" className="text-xs">
                                                <Infinity className="h-3 w-3 mr-1" />
                                                Continuous
                                              </Badge>
                                            ) : (
                                              <Badge variant="outline" className="text-xs">
                                                <Timer className="h-3 w-3 mr-1" />
                                                {formatDuration(med)}
                                              </Badge>
                                            )}
                                            {med.hasConflict && (
                                              <Badge variant="destructive" className="text-xs">
                                                <Ban className="h-3 w-3 mr-1" />
                                                On Hold
                                              </Badge>
                                            )}
                                          </div>
                                          <p className="text-sm text-muted-foreground mt-1">
                                            {med.dosage} • {med.frequency}
                                          </p>
                                          {med.prescribingDoctorName && (
                                            <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1">
                                              <User className="h-3 w-3" />
                                              Prescribed by {med.prescribingDoctorName}
                                            </p>
                                          )}
                                          {med.actualStartDate && (
                                            <p className="text-xs text-muted-foreground flex items-center gap-1">
                                              <Calendar className="h-3 w-3" />
                                              Started: {format(new Date(med.actualStartDate), 'MMM d, yyyy')}
                                            </p>
                                          )}
                                          {!med.isContinuous && med.computedEndDate && (
                                            <p className="text-xs text-muted-foreground flex items-center gap-1">
                                              <Clock className="h-3 w-3" />
                                              Ends: {format(new Date(med.computedEndDate), 'MMM d, yyyy')}
                                            </p>
                                          )}
                                        </div>
                                        <div className="flex flex-col items-end gap-2">
                                          {!med.isContinuous && med.daysRemaining !== null && med.daysRemaining !== undefined && (
                                            <div
                                              className={`text-right ${
                                                med.daysRemaining <= 2
                                                  ? 'text-red-600 dark:text-red-400'
                                                  : med.daysRemaining <= 7
                                                  ? 'text-amber-600 dark:text-amber-400'
                                                  : 'text-green-600 dark:text-green-400'
                                              }`}
                                            >
                                              <p className="text-2xl font-bold">{med.daysRemaining}</p>
                                              <p className="text-xs">days left</p>
                                            </div>
                                          )}
                                          {med.isContinuous && (
                                            <div className="text-right text-muted-foreground">
                                              <Infinity className="h-6 w-6 mx-auto" />
                                              <p className="text-xs mt-1">Ongoing</p>
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                      {!med.actualStartDate && med.intendedStartDate && (
                                        <div className="mt-3 pt-3 border-t border-dashed">
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => {
                                              setConfirmStartDialog({ open: true, medication: med });
                                              setSelectedStartDate(new Date(med.intendedStartDate!));
                                            }}
                                            data-testid={`button-start-${med.id}`}
                                          >
                                            <PlayCircle className="h-4 w-4 mr-1" />
                                            Confirm Start Date
                                          </Button>
                                        </div>
                                      )}
                                      {med.actualStartDate && !med.hasConflict && (
                                        <div className="mt-3 pt-3 border-t border-dashed flex items-center gap-2 flex-wrap">
                                          <Button
                                            size="sm"
                                            variant="default"
                                            onClick={() => logAdherenceMutation.mutate({ 
                                              medicationId: med.id, 
                                              status: 'taken' 
                                            })}
                                            disabled={logAdherenceMutation.isPending}
                                            data-testid={`button-taken-${med.id}`}
                                          >
                                            {logAdherenceMutation.isPending ? (
                                              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                            ) : (
                                              <Check className="h-4 w-4 mr-1" />
                                            )}
                                            Mark Taken
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => logAdherenceMutation.mutate({ 
                                              medicationId: med.id, 
                                              status: 'skipped' 
                                            })}
                                            disabled={logAdherenceMutation.isPending}
                                            data-testid={`button-skipped-${med.id}`}
                                          >
                                            <XCircle className="h-4 w-4 mr-1" />
                                            Skip
                                          </Button>
                                          {med.isContinuous && (
                                            <Button
                                              size="sm"
                                              variant="outline"
                                              onClick={() => requestRefillMutation.mutate({ id: med.id })}
                                              disabled={requestRefillMutation.isPending}
                                              data-testid={`button-refill-${med.id}`}
                                            >
                                              {requestRefillMutation.isPending ? (
                                                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                              ) : (
                                                <RefreshCw className="h-4 w-4 mr-1" />
                                              )}
                                              Request Refill
                                            </Button>
                                          )}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </CardContent>
                            </CollapsibleContent>
                          </Card>
                        </Collapsible>
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>

            {dashboard?.archivedMedications && dashboard.archivedMedications.length > 0 && (
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <Archive className="h-5 w-5" />
                        Medication History
                      </CardTitle>
                      <CardDescription>
                        Previously completed, expired, or superseded medications
                      </CardDescription>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowArchivedMeds(!showArchivedMeds)}
                      data-testid="button-toggle-archived"
                    >
                      {showArchivedMeds ? 'Hide' : 'Show'} ({dashboard.archivedMedications.length})
                      {showArchivedMeds ? <ChevronDown className="h-4 w-4 ml-1" /> : <ChevronRight className="h-4 w-4 ml-1" />}
                    </Button>
                  </div>
                </CardHeader>
                {showArchivedMeds && (
                  <CardContent>
                    <ScrollArea className="h-64">
                      <div className="space-y-2">
                        {dashboard.archivedMedications.map((med) => (
                          <div
                            key={med.id}
                            className="flex items-center justify-between p-3 bg-muted/50 rounded-md opacity-75"
                            data-testid={`archived-medication-${med.id}`}
                          >
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-medium">{med.name}</span>
                                <Badge variant="outline" className="text-xs capitalize">
                                  {med.status}
                                </Badge>
                                {med.specialty && (
                                  <Badge variant="outline" className="text-xs">
                                    {med.specialty}
                                  </Badge>
                                )}
                              </div>
                              <p className="text-sm text-muted-foreground">
                                {med.dosage} • {med.frequency}
                              </p>
                            </div>
                            {med.computedEndDate && (
                              <p className="text-xs text-muted-foreground">
                                Ended: {format(new Date(med.computedEndDate), 'MMM d, yyyy')}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </CardContent>
                )}
              </Card>
            )}
          </>
        )}
      </div>

      <Dialog open={confirmStartDialog.open} onOpenChange={(open) => !open && setConfirmStartDialog({ open: false, medication: null })}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Medication Start Date</DialogTitle>
            <DialogDescription>
              When will you start taking <strong>{confirmStartDialog.medication?.name}</strong>?
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label className="mb-2 block">Select your start date:</Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className="w-full justify-start text-left font-normal" data-testid="button-date-picker">
                  <CalendarDays className="mr-2 h-4 w-4" />
                  {selectedStartDate ? format(selectedStartDate, 'PPP') : 'Pick a date'}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <CalendarComponent
                  mode="single"
                  selected={selectedStartDate}
                  onSelect={setSelectedStartDate}
                  initialFocus
                />
              </PopoverContent>
            </Popover>
            {confirmStartDialog.medication?.durationDays && selectedStartDate && (
              <p className="text-sm text-muted-foreground mt-3">
                This {confirmStartDialog.medication.durationDays}-day course will end on{' '}
                <strong>
                  {format(
                    new Date(selectedStartDate.getTime() + confirmStartDialog.medication.durationDays * 24 * 60 * 60 * 1000),
                    'MMM d, yyyy'
                  )}
                </strong>
              </p>
            )}
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmStartDialog({ open: false, medication: null })}
            >
              Cancel
            </Button>
            <Button
              onClick={handleConfirmStart}
              disabled={!selectedStartDate || confirmStartMutation.isPending}
              data-testid="button-confirm-date"
            >
              {confirmStartMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Check className="h-4 w-4 mr-2" />
              )}
              Confirm Start
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
