import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useLocation } from "wouter";
import { 
  Users, 
  Search, 
  TrendingUp, 
  AlertCircle, 
  Bot, 
  Eye, 
  EyeOff, 
  ChevronLeft, 
  User,
  Phone,
  Mail,
  Calendar,
  Heart,
  Droplets,
  Activity,
  Brain,
  Pill,
  Stethoscope,
  FileText,
  Clock,
  Sparkles,
  Lightbulb,
  BookOpen,
  Beaker,
  RefreshCw,
  X,
  LayoutGrid,
  List as ListIcon,
  MessageSquare,
  AlertTriangle,
  Circle,
  ChevronRight,
  TrendingDown
} from "lucide-react";
import type { User as UserType } from "@shared/schema";
import { AddPatientDialog } from "@/components/AddPatientDialog";
import { LysaChatPanel } from "@/components/LysaChatPanel";
import { PatientAIAlerts } from "@/components/patient-ai-alerts";
import { PatientMLTools } from "@/components/patient-ml-tools";
import { HealthMetricCard } from "@/components/HealthMetricCard";
import { PrescriptionHelper } from "@/components/PrescriptionHelper";
import { ClinicalDecisionSupport } from "@/components/ClinicalDecisionSupport";
import { PredictiveAnalyticsDashboard } from "@/components/PredictiveAnalyticsDashboard";
import { DraftDifferentialPanel } from "@/components/DraftDifferentialPanel";
import { CBTSessionPanel } from "@/components/CBTSessionPanel";
import { PersonalizedHabitSuggestions } from "@/components/PersonalizedHabitSuggestions";
import { format, parseISO, differenceInYears, formatDistanceToNow, isValid } from "date-fns";
import { cn } from "@/lib/utils";

function safeFormat(dateStr: string | undefined | null, formatStr: string, fallback: string = "—"): string {
  if (!dateStr) return fallback;
  try {
    const date = new Date(dateStr);
    if (!isValid(date)) return fallback;
    return format(date, formatStr);
  } catch {
    return fallback;
  }
}

function safeFormatDistance(dateStr: string | undefined | null, fallback: string = "—"): string {
  if (!dateStr) return fallback;
  try {
    const date = new Date(dateStr);
    if (!isValid(date)) return fallback;
    return formatDistanceToNow(date, { addSuffix: true });
  } catch {
    return fallback;
  }
}

type Recommendation = {
  id: string;
  type: string;
  category: string;
  title: string;
  description: string;
  confidenceScore: string;
  priority: string;
  reasoning?: string;
};

interface MonitoringStatus {
  patientId: string;
  isMonitored: boolean;
}

interface HealthMetric {
  name: string;
  value: number;
  unit: string;
  trend?: "up" | "down" | "stable";
  status?: "normal" | "warning" | "critical";
  lastUpdated?: string;
}

interface PatientProfile {
  dateOfBirth?: string;
  bloodType?: string;
  allergies?: string[];
  medicalConditions?: string[];
  emergencyContact?: string;
  emergencyPhone?: string;
  followupPatientId?: string;
}

interface EnrichedPatientSummary {
  id: string;
  name: string;
  email?: string;
  assignedAt?: string;
  accessLevel?: string;
  riskScore?: number;
  lastInteraction?: string;
  activeAlerts: number;
  medicationCount: number;
  pendingFollowups: number;
  isOnline: boolean;
}

interface PatientOverview {
  patient: {
    id: string;
    name: string;
    email?: string;
    dateOfBirth?: string;
    phone?: string;
    memberSince?: string;
  };
  assignment: {
    accessLevel: string;
    consentType?: string;
    expiresAt?: string;
  };
  healthContext: {
    recent_symptoms?: Array<{ type: string; severity: number; date: string }>;
    medications?: Array<{ name: string; dosage: string; frequency: string }>;
    risk_score?: { composite: number; respiratory: number; pain: number };
    active_alerts?: Array<{ type: string; severity: string; message: string }>;
    has_concerning_patterns?: boolean;
  };
  dailyFollowups: Array<{
    date?: string;
    overallStatus?: string;
    energyLevel?: number;
    painLevel?: number;
    symptomsNoted?: string;
    notes?: string;
    completedAt?: string;
  }>;
  conversationHistory: Array<{
    id: string;
    title: string;
    createdAt?: string;
    updatedAt?: string;
    messageCount: number;
    lastMessage?: string;
  }>;
  longTermInsights: Array<{
    content: string;
    type: string;
    createdAt?: string;
    importance: number;
  }>;
}

interface ConversationMessage {
  id: string;
  msgId: string;
  fromType: string;
  fromId: string;
  senderRole: string;
  senderName: string;
  senderSubtitle: string;
  senderAvatar?: string;
  isAI: boolean;
  isHuman: boolean;
  messageType: string;
  content: string;
  toolName?: string;
  toolStatus?: string;
  requiresApproval?: boolean;
  approvalStatus?: string;
  containsPhi?: boolean;
  createdAt?: string;
}

interface ConversationMessagesResponse {
  conversationId: string;
  title: string;
  conversationType: string;
  patientId: string;
  messages: ConversationMessage[];
  total: number;
}

interface PatientManagementPanelProps {
  onOpenLysa?: (patient: UserType) => void;
  className?: string;
}

function getRiskColor(score: number | undefined): string {
  if (!score) return "text-muted-foreground";
  if (score >= 10) return "text-destructive";
  if (score >= 6) return "text-orange-500";
  if (score >= 3) return "text-yellow-500";
  return "text-chart-2";
}

function getRiskBadgeVariant(score: number | undefined): "default" | "secondary" | "destructive" | "outline" {
  if (!score) return "outline";
  if (score >= 10) return "destructive";
  if (score >= 6) return "secondary";
  return "default";
}

export function PatientManagementPanel({ onOpenLysa, className }: PatientManagementPanelProps) {
  const [, setLocation] = useLocation();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<UserType | null>(null);
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [lysaDialogOpen, setLysaDialogOpen] = useState(false);
  const [monitoringStates, setMonitoringStates] = useState<Record<string, boolean>>({});
  const [detailTab, setDetailTab] = useState("overview");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [expandedConversationId, setExpandedConversationId] = useState<string | null>(null);
  const { toast } = useToast();

  const { data: patients = [], isLoading: patientsLoading } = useQuery<UserType[]>({
    queryKey: ["/api/doctor/patients"],
  });

  const { data: enrichedPatientsData, isLoading: enrichedLoading } = useQuery<{ patients: EnrichedPatientSummary[]; total: number }>({
    queryKey: ["/api/agent/patients"],
  });

  const enrichedPatients = enrichedPatientsData?.patients || [];

  const enrichedPatientMap = enrichedPatients.reduce((acc, p) => {
    acc[p.id] = p;
    return acc;
  }, {} as Record<string, EnrichedPatientSummary>);

  const { data: patientOverview, isLoading: overviewLoading } = useQuery<PatientOverview>({
    queryKey: ["/api/agent/patients", selectedPatientId, "overview"],
    enabled: !!selectedPatientId && !!selectedPatient,
  });

  const { data: monitoringStatuses } = useQuery<MonitoringStatus[]>({
    queryKey: ["/api/v1/lysa/monitoring/status"],
    queryFn: async () => {
      try {
        const res = await fetch('/api/v1/lysa/monitoring/status');
        if (!res.ok) return [];
        return res.json();
      } catch {
        return [];
      }
    }
  });

  const { data: recommendations = [], isLoading: recommendationsLoading } = useQuery<Recommendation[]>({
    queryKey: ['/api/v1/ml/recommendations', { agentType: 'lysa', limit: 6 }],
    queryFn: async () => {
      const res = await fetch('/api/v1/ml/recommendations?agentType=lysa&limit=6');
      if (!res.ok) return [];
      return res.json();
    },
  });

  const { data: patientProfile } = useQuery<PatientProfile>({
    queryKey: ['/api/v1/patients', selectedPatient?.id, 'profile'],
    queryFn: async () => {
      if (!selectedPatient?.id) return null;
      const res = await fetch(`/api/v1/patients/${selectedPatient.id}/profile`);
      if (!res.ok) return null;
      return res.json();
    },
    enabled: !!selectedPatient?.id
  });

  const { data: healthMetrics = [] } = useQuery<HealthMetric[]>({
    queryKey: ['/api/v1/patients', selectedPatient?.id, 'health-metrics'],
    queryFn: async () => {
      if (!selectedPatient?.id) return [];
      const res = await fetch(`/api/v1/patients/${selectedPatient.id}/health-metrics`);
      if (!res.ok) return [];
      return res.json();
    },
    enabled: !!selectedPatient?.id
  });

  const { data: conversationMessages, isLoading: messagesLoading } = useQuery<ConversationMessagesResponse>({
    queryKey: ['/api/agent/conversations', expandedConversationId, 'messages'],
    queryFn: async () => {
      if (!expandedConversationId) return null;
      const res = await fetch(`/api/agent/conversations/${expandedConversationId}/messages`);
      if (!res.ok) return null;
      return res.json();
    },
    enabled: !!expandedConversationId
  });

  useEffect(() => {
    if (monitoringStatuses) {
      const states: Record<string, boolean> = {};
      monitoringStatuses.forEach(s => {
        states[s.patientId] = s.isMonitored;
      });
      setMonitoringStates(states);
    }
  }, [monitoringStatuses]);

  const toggleMonitoringMutation = useMutation({
    mutationFn: async ({ patientId, enabled }: { patientId: string; enabled: boolean }) => {
      const response = await apiRequest('/api/v1/lysa/monitoring/toggle', {
        method: 'POST',
        json: { patientId, enabled }
      });
      return response;
    },
    onSuccess: (data: any, variables) => {
      setMonitoringStates(prev => ({ ...prev, [variables.patientId]: variables.enabled }));
      toast({
        title: variables.enabled ? "Monitoring Enabled" : "Monitoring Disabled",
        description: variables.enabled 
          ? "AI monitoring is now active for this patient"
          : "AI monitoring has been turned off for this patient"
      });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/lysa/monitoring/status'] });
    },
    onError: (error: any) => {
      toast({
        title: "Toggle Failed",
        description: error.message || "Failed to update monitoring status",
        variant: "destructive"
      });
    }
  });

  const handleMonitoringToggle = (patientId: string, enabled: boolean, e: React.MouseEvent) => {
    e.stopPropagation();
    toggleMonitoringMutation.mutate({ patientId, enabled });
  };

  const openLysaForPatient = (patient: UserType, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedPatient(patient);
    if (onOpenLysa) {
      onOpenLysa(patient);
    } else {
      setLysaDialogOpen(true);
    }
  };

  const handlePatientSelect = (patient: UserType) => {
    setSelectedPatient(patient);
    setSelectedPatientId(patient.id);
    setDetailTab("overview");
  };

  const filteredPatients = patients.filter((patient) =>
    `${patient.firstName} ${patient.lastName}`.toLowerCase().includes(searchQuery.toLowerCase()) ||
    patient.email?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const clinicalRecommendations = recommendations.filter(r => 
    r.category === 'clinical_decision_support' || r.category === 'protocol_suggestion'
  );
  
  const researchRecommendations = recommendations.filter(r => 
    r.category === 'research' || r.category === 'literature_review'
  );

  const getInitials = (firstName?: string | null, lastName?: string | null) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  const calculateAge = (dateOfBirth: string | undefined | null): string => {
    if (!dateOfBirth) return "—";
    try {
      const date = parseISO(dateOfBirth);
      if (!isValid(date)) return "—";
      return String(differenceInYears(new Date(), date));
    } catch {
      return "—";
    }
  };

  const getMetricIcon = (name: string) => {
    const icons: Record<string, any> = {
      heart_rate: Heart,
      blood_pressure: Activity,
      oxygen: Droplets,
      temperature: Activity,
      steps: Activity,
      sleep: Clock,
    };
    return icons[name.toLowerCase().replace(/\s+/g, '_')] || Activity;
  };

  if (selectedPatient) {
    const enrichedData = enrichedPatientMap[selectedPatient.id];
    const isLoadingEnrichedData = enrichedLoading || (!enrichedData && enrichedPatients.length === 0);
    
    if (isLoadingEnrichedData) {
      return (
        <div className={`space-y-4 ${className}`}>
          <div className="flex items-center gap-4 mb-6">
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => {
                setSelectedPatient(null);
                setSelectedPatientId(null);
                setDetailTab("overview");
              }}
              data-testid="button-back-to-patients"
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Back to Patients
            </Button>
          </div>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-4 mb-6">
                <Skeleton className="h-16 w-16 rounded-full" />
                <div className="flex-1 space-y-3">
                  <Skeleton className="h-6 w-1/3" />
                  <Skeleton className="h-4 w-1/2" />
                </div>
              </div>
              <div className="grid grid-cols-4 gap-4">
                {[1, 2, 3, 4].map(i => <Skeleton key={i} className="h-20" />)}
              </div>
            </CardContent>
          </Card>
        </div>
      );
    }
    
    return (
      <div className={`space-y-4 ${className}`}>
        <div className="flex items-center gap-4 mb-6">
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => {
              setSelectedPatient(null);
              setSelectedPatientId(null);
              setDetailTab("overview");
            }}
            data-testid="button-back-to-patients"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back to Patients
          </Button>
          <div className="flex-1" />
          {enrichedData && (
            <Badge 
              variant={getRiskBadgeVariant(enrichedData.riskScore)}
              className="text-sm"
            >
              Risk: {(enrichedData.riskScore ?? 0).toFixed(1)}
            </Badge>
          )}
          <Badge variant="secondary" className="text-sm">
            {monitoringStates[selectedPatient.id] ? (
              <>
                <Eye className="h-3 w-3 mr-1" />
                AI Monitoring Active
              </>
            ) : (
              <>
                <EyeOff className="h-3 w-3 mr-1" />
                Monitoring Off
              </>
            )}
          </Badge>
          <Switch
            checked={monitoringStates[selectedPatient.id] || false}
            onCheckedChange={(checked) => handleMonitoringToggle(selectedPatient.id, checked, { stopPropagation: () => {} } as React.MouseEvent)}
            disabled={toggleMonitoringMutation.isPending}
            data-testid={`switch-monitoring-detail-${selectedPatient.id}`}
          />
          <Button
            variant="outline"
            onClick={(e) => openLysaForPatient(selectedPatient, e)}
            data-testid="button-open-lysa-detail"
          >
            <Bot className="h-4 w-4 mr-2" />
            Ask Lysa
          </Button>
          <Button 
            onClick={() => setLocation(`/doctor/patient/${selectedPatient.id}`)}
            data-testid="button-full-profile"
          >
            <User className="h-4 w-4 mr-2" />
            Full Profile
          </Button>
        </div>

        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex items-start gap-4">
              <div className="relative">
                <Avatar className="h-16 w-16">
                  <AvatarFallback className="bg-primary text-primary-foreground text-xl">
                    {getInitials(selectedPatient.firstName, selectedPatient.lastName)}
                  </AvatarFallback>
                </Avatar>
                {enrichedData?.isOnline && (
                  <Circle className="absolute -bottom-0.5 -right-0.5 h-4 w-4 fill-chart-2 text-chart-2" />
                )}
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-semibold" data-testid="text-patient-name">
                  {selectedPatient.firstName} {selectedPatient.lastName}
                </h2>
                <div className="flex flex-wrap gap-4 mt-2 text-sm text-muted-foreground">
                  {selectedPatient.email && (
                    <span className="flex items-center gap-1">
                      <Mail className="h-3 w-3" />
                      {selectedPatient.email}
                    </span>
                  )}
                  {selectedPatient.phoneNumber && (
                    <span className="flex items-center gap-1">
                      <Phone className="h-3 w-3" />
                      {selectedPatient.phoneNumber}
                    </span>
                  )}
                  {patientProfile?.dateOfBirth && (
                    <span className="flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      {calculateAge(patientProfile.dateOfBirth)} years old
                    </span>
                  )}
                  {patientProfile?.bloodType && (
                    <Badge variant="outline" className="text-xs">
                      <Droplets className="h-3 w-3 mr-1" />
                      {patientProfile.bloodType}
                    </Badge>
                  )}
                </div>
                {patientProfile?.allergies && patientProfile.allergies.length > 0 && (
                  <div className="flex items-center gap-2 mt-3">
                    <AlertCircle className="h-4 w-4 text-destructive" />
                    <span className="text-sm text-muted-foreground">Allergies:</span>
                    {patientProfile.allergies.map((allergy, i) => (
                      <Badge key={i} variant="destructive" className="text-xs">{allergy}</Badge>
                    ))}
                  </div>
                )}
              </div>
              {enrichedData && (
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div className="p-3 rounded-lg border">
                    <div className={cn("text-2xl font-bold", getRiskColor(enrichedData.riskScore))}>
                      {(enrichedData.riskScore ?? 0).toFixed(1)}
                    </div>
                    <div className="text-xs text-muted-foreground">Risk Score</div>
                  </div>
                  <div className="p-3 rounded-lg border">
                    <div className={cn("text-2xl font-bold", enrichedData.activeAlerts > 0 ? "text-destructive" : "")}>
                      {enrichedData.activeAlerts}
                    </div>
                    <div className="text-xs text-muted-foreground">Active Alerts</div>
                  </div>
                  <div className="p-3 rounded-lg border">
                    <div className="text-2xl font-bold">
                      {enrichedData.medicationCount}
                    </div>
                    <div className="text-xs text-muted-foreground">Medications</div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Tabs value={detailTab} onValueChange={setDetailTab} className="w-full">
          <TabsList className="flex-wrap justify-start">
            <TabsTrigger value="overview" data-testid="tab-patient-overview">
              <User className="h-4 w-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="daily-followups" data-testid="tab-patient-daily-followups">
              <Calendar className="h-4 w-4 mr-2" />
              Daily Followups
            </TabsTrigger>
            <TabsTrigger value="conversations" data-testid="tab-patient-conversations">
              <MessageSquare className="h-4 w-4 mr-2" />
              Conversations
            </TabsTrigger>
            <TabsTrigger value="ai-insights" data-testid="tab-patient-ai-insights">
              <Brain className="h-4 w-4 mr-2" />
              AI Insights
            </TabsTrigger>
            <TabsTrigger value="metrics" data-testid="tab-patient-metrics">
              <Activity className="h-4 w-4 mr-2" />
              Health Metrics
            </TabsTrigger>
            <TabsTrigger value="ai-alerts" data-testid="tab-patient-ai-alerts">
              <AlertCircle className="h-4 w-4 mr-2" />
              AI Alerts
            </TabsTrigger>
            <TabsTrigger value="ml-tools" data-testid="tab-patient-ml-tools">
              <Brain className="h-4 w-4 mr-2" />
              ML Inference
            </TabsTrigger>
            <TabsTrigger value="analytics" data-testid="tab-patient-analytics">
              <TrendingUp className="h-4 w-4 mr-2" />
              Predictive Analytics
            </TabsTrigger>
            <TabsTrigger value="prescriptions" data-testid="tab-patient-prescriptions">
              <Pill className="h-4 w-4 mr-2" />
              Prescriptions
            </TabsTrigger>
            <TabsTrigger value="clinical" data-testid="tab-patient-clinical">
              <Stethoscope className="h-4 w-4 mr-2" />
              Clinical Support
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-4 space-y-4">
            {overviewLoading ? (
              <div className="grid gap-4 md:grid-cols-2">
                <Skeleton className="h-48" />
                <Skeleton className="h-48" />
              </div>
            ) : patientOverview ? (
              <>
                <div className="grid gap-4 md:grid-cols-2">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <Heart className="h-4 w-4 text-destructive" />
                        Patient Information
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Email</span>
                        <span>{patientOverview.patient.email || selectedPatient.email || "—"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Phone</span>
                        <span>{patientOverview.patient.phone || selectedPatient.phoneNumber || "—"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Date of Birth</span>
                        <span>{safeFormat(patientOverview.patient.dateOfBirth, "MMM d, yyyy")}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Member Since</span>
                        <span>{safeFormat(patientOverview.patient.memberSince, "MMM yyyy")}</span>
                      </div>
                      <Separator className="my-2" />
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Access Level</span>
                        <Badge variant="outline">{patientOverview.assignment.accessLevel}</Badge>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <TrendingUp className="h-4 w-4 text-chart-1" />
                        Risk Assessment
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-2xl font-bold">
                          {(patientOverview.healthContext.risk_score?.composite ?? 0).toFixed(1)}
                        </span>
                        <Badge variant={getRiskBadgeVariant(patientOverview.healthContext.risk_score?.composite)}>
                          {(patientOverview.healthContext.risk_score?.composite ?? 0) >= 10 ? "High Risk" :
                           (patientOverview.healthContext.risk_score?.composite ?? 0) >= 6 ? "Moderate" : "Low Risk"}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Respiratory</span>
                          <span>{(patientOverview.healthContext.risk_score?.respiratory ?? 0).toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Pain</span>
                          <span>{(patientOverview.healthContext.risk_score?.pain ?? 0).toFixed(1)}</span>
                        </div>
                      </div>
                      {patientOverview.healthContext.has_concerning_patterns && (
                        <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 p-2 rounded">
                          <AlertTriangle className="h-4 w-4" />
                          Concerning patterns detected
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>

                {patientOverview.healthContext.active_alerts && patientOverview.healthContext.active_alerts.length > 0 && (
                  <Card className="border-destructive/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-destructive" />
                        Active Alerts ({patientOverview.healthContext.active_alerts.length})
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      {patientOverview.healthContext.active_alerts.map((alert, idx) => (
                        <div key={idx} className="flex items-start gap-2 p-2 bg-destructive/5 rounded text-sm">
                          <Badge variant={alert.severity === "critical" ? "destructive" : "secondary"} className="flex-shrink-0">
                            {alert.severity}
                          </Badge>
                          <span>{alert.message}</span>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}

                <div className="grid gap-4 md:grid-cols-2">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <Pill className="h-4 w-4 text-blue-500" />
                        Current Medications ({patientOverview.healthContext.medications?.length || 0})
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {patientOverview.healthContext.medications && patientOverview.healthContext.medications.length > 0 ? (
                        <div className="space-y-2">
                          {patientOverview.healthContext.medications.map((med, idx) => (
                            <div key={idx} className="flex items-center justify-between text-sm p-2 bg-muted/50 rounded">
                              <span className="font-medium">{med.name}</span>
                              <span className="text-muted-foreground">{med.dosage} - {med.frequency}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">No active medications</p>
                      )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <Activity className="h-4 w-4 text-orange-500" />
                        Recent Symptoms
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {patientOverview.healthContext.recent_symptoms && patientOverview.healthContext.recent_symptoms.length > 0 ? (
                        <div className="space-y-2">
                          {patientOverview.healthContext.recent_symptoms.slice(0, 5).map((symptom, idx) => (
                            <div key={idx} className="flex items-center justify-between text-sm">
                              <span>{symptom.type}</span>
                              <div className="flex items-center gap-2">
                                <Badge variant="outline" className="text-xs">
                                  Severity: {symptom.severity}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  {safeFormat(symptom.date, "MMM d", "")}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">No recent symptoms recorded</p>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Contact Information</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {selectedPatient.email && (
                      <div className="flex items-center gap-3 p-3 rounded-lg border">
                        <Mail className="h-5 w-5 text-muted-foreground" />
                        <div>
                          <p className="text-sm text-muted-foreground">Email</p>
                          <p className="font-medium">{selectedPatient.email}</p>
                        </div>
                      </div>
                    )}
                    {selectedPatient.phoneNumber && (
                      <div className="flex items-center gap-3 p-3 rounded-lg border">
                        <Phone className="h-5 w-5 text-muted-foreground" />
                        <div>
                          <p className="text-sm text-muted-foreground">Phone</p>
                          <p className="font-medium">{selectedPatient.phoneNumber}</p>
                        </div>
                      </div>
                    )}
                    {patientProfile?.emergencyContact && (
                      <div className="flex items-center gap-3 p-3 rounded-lg border border-orange-200 bg-orange-50 dark:bg-orange-950/20 dark:border-orange-800">
                        <AlertCircle className="h-5 w-5 text-orange-500" />
                        <div>
                          <p className="text-sm text-muted-foreground">Emergency Contact</p>
                          <p className="font-medium">
                            {patientProfile.emergencyContact}
                            {patientProfile.emergencyPhone && ` - ${patientProfile.emergencyPhone}`}
                          </p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Medical Information</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {patientProfile?.bloodType && (
                      <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                        <Droplets className="h-6 w-6 text-red-500" />
                        <div>
                          <p className="text-sm text-muted-foreground">Blood Type</p>
                          <p className="text-lg font-bold text-red-500">{patientProfile.bloodType}</p>
                        </div>
                      </div>
                    )}
                    <div className="p-4 rounded-lg border">
                      <div className="flex items-center gap-2 mb-3">
                        <Heart className="h-5 w-5 text-pink-500" />
                        <p className="font-medium">Medical Conditions</p>
                      </div>
                      {patientProfile?.medicalConditions && patientProfile.medicalConditions.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {patientProfile.medicalConditions.map((condition, idx) => (
                            <Badge key={idx} variant="secondary">{condition}</Badge>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">No recorded conditions</p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          <TabsContent value="daily-followups" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Calendar className="h-5 w-5" />
                  Daily Followup History (Last 7 Days)
                </CardTitle>
                <CardDescription>
                  Track patient's daily health check-ins and wellness status
                </CardDescription>
              </CardHeader>
              <CardContent>
                {overviewLoading ? (
                  <div className="space-y-3">
                    {[1, 2, 3].map((i) => (
                      <Skeleton key={i} className="h-24" />
                    ))}
                  </div>
                ) : patientOverview?.dailyFollowups && patientOverview.dailyFollowups.length > 0 ? (
                  <div className="space-y-3">
                    {patientOverview.dailyFollowups.map((followup, idx) => (
                      <div key={idx} className="flex items-start gap-4 p-4 border rounded-lg hover-elevate">
                        <div className="text-center min-w-[60px]">
                          <div className="text-2xl font-bold">
                            {safeFormat(followup.date, "d")}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {safeFormat(followup.date, "MMM", "")}
                          </div>
                        </div>
                        <Separator orientation="vertical" className="h-auto self-stretch" />
                        <div className="flex-1 space-y-2">
                          <div className="flex items-center gap-2 flex-wrap">
                            <Badge variant={followup.overallStatus === "good" ? "default" : 
                                           followup.overallStatus === "fair" ? "secondary" : 
                                           followup.overallStatus === "poor" ? "destructive" : "outline"}>
                              {followup.overallStatus || "Not recorded"}
                            </Badge>
                            {followup.completedAt && (
                              <span className="text-xs text-muted-foreground">
                                Completed {safeFormat(followup.completedAt, "h:mm a", "")}
                              </span>
                            )}
                          </div>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div className="flex items-center gap-2">
                              <Activity className="h-4 w-4 text-yellow-500" />
                              <span className="text-muted-foreground">Energy:</span>
                              <span className="font-medium">{followup.energyLevel ?? "—"}/10</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Heart className="h-4 w-4 text-red-500" />
                              <span className="text-muted-foreground">Pain:</span>
                              <span className="font-medium">{followup.painLevel ?? "—"}/10</span>
                            </div>
                          </div>
                          {followup.symptomsNoted && (
                            <div className="text-sm">
                              <span className="text-muted-foreground">Symptoms: </span>
                              {followup.symptomsNoted}
                            </div>
                          )}
                          {followup.notes && (
                            <p className="text-sm italic text-muted-foreground border-l-2 pl-2">
                              {followup.notes}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="py-12 text-center">
                    <Calendar className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                    <h3 className="text-lg font-medium mb-2">No Daily Followups</h3>
                    <p className="text-muted-foreground">
                      No daily followup entries found for this patient in the past 7 days.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="conversations" className="mt-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <MessageSquare className="h-5 w-5" />
                      {expandedConversationId ? "Conversation Details" : "Lysa Conversation History"}
                    </CardTitle>
                    <CardDescription>
                      {expandedConversationId 
                        ? "View messages with sender identification (AI or Human)" 
                        : "Click a conversation to view detailed messages"}
                    </CardDescription>
                  </div>
                  {expandedConversationId && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => {
                        setExpandedConversationId(null);
                        queryClient.invalidateQueries({ 
                          queryKey: ["/api/agent/patients", selectedPatientId, "overview"] 
                        });
                      }}
                      data-testid="button-back-to-conversations"
                    >
                      <ChevronLeft className="h-4 w-4 mr-1" />
                      Back
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {expandedConversationId ? (
                  messagesLoading ? (
                    <div className="space-y-3">
                      {[1, 2, 3, 4].map((i) => (
                        <Skeleton key={i} className="h-16" />
                      ))}
                    </div>
                  ) : conversationMessages?.messages && conversationMessages.messages.length > 0 ? (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between px-2 py-1 bg-muted/50 rounded-lg mb-4">
                        <span className="text-sm font-medium">{conversationMessages.title}</span>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            {conversationMessages.total} messages
                          </Badge>
                        </div>
                      </div>
                      <ScrollArea className="h-[400px] pr-4">
                        <div className="space-y-3">
                          {conversationMessages.messages.map((message) => (
                            <div 
                              key={message.id}
                              className={cn(
                                "flex gap-3 p-3 rounded-lg border-l-4",
                                message.isAI 
                                  ? "bg-purple-100/80 dark:bg-purple-950/60 border-l-purple-500 border border-purple-200 dark:border-purple-700" 
                                  : "bg-green-100/80 dark:bg-green-950/60 border-l-green-500 border border-green-200 dark:border-green-700"
                              )}
                              data-testid={`message-${message.id}`}
                            >
                              <div className={cn(
                                "flex-shrink-0 p-2 rounded-full shadow-sm",
                                message.isAI 
                                  ? "bg-purple-200 dark:bg-purple-700" 
                                  : "bg-green-200 dark:bg-green-700"
                              )}>
                                {message.isAI ? (
                                  <Bot className="h-4 w-4 text-purple-600 dark:text-purple-300" />
                                ) : (
                                  <User className="h-4 w-4 text-green-600 dark:text-green-300" />
                                )}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap mb-1">
                                  <span className="font-medium text-sm">{message.senderName}</span>
                                  <Badge 
                                    variant={message.isAI ? "secondary" : "default"} 
                                    className={cn(
                                      "text-xs",
                                      message.isAI 
                                        ? "bg-purple-200 dark:bg-purple-700 text-purple-800 dark:text-purple-200" 
                                        : "bg-green-200 dark:bg-green-700 text-green-800 dark:text-green-200"
                                    )}
                                  >
                                    {message.isAI ? "AI Assistant" : "Human"}
                                  </Badge>
                                  <span className="text-xs text-muted-foreground">
                                    {message.senderSubtitle}
                                  </span>
                                  {message.containsPhi && (
                                    <Badge variant="outline" className="text-xs border-red-300 text-red-600">
                                      PHI
                                    </Badge>
                                  )}
                                </div>
                                <p className="text-sm whitespace-pre-wrap">
                                  {message.content || (message.toolName ? `[Tool: ${message.toolName}]` : "[No content]")}
                                </p>
                                {message.toolName && (
                                  <div className="flex items-center gap-2 mt-2">
                                    <Badge variant="outline" className="text-xs">
                                      Tool: {message.toolName}
                                    </Badge>
                                    {message.toolStatus && (
                                      <Badge 
                                        variant={message.toolStatus === "completed" ? "default" : "secondary"}
                                        className="text-xs"
                                      >
                                        {message.toolStatus}
                                      </Badge>
                                    )}
                                  </div>
                                )}
                                <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                                  <Clock className="h-3 w-3" />
                                  {safeFormat(message.createdAt, "MMM d, yyyy 'at' h:mm a")}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    </div>
                  ) : (
                    <div className="py-12 text-center">
                      <MessageSquare className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                      <h3 className="text-lg font-medium mb-2">No Messages</h3>
                      <p className="text-muted-foreground">
                        This conversation has no messages yet.
                      </p>
                    </div>
                  )
                ) : overviewLoading ? (
                  <div className="space-y-3">
                    {[1, 2, 3].map((i) => (
                      <Skeleton key={i} className="h-20" />
                    ))}
                  </div>
                ) : patientOverview?.conversationHistory && patientOverview.conversationHistory.length > 0 ? (
                  <div className="space-y-3">
                    <div className="flex items-center gap-4 mb-4 text-sm">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-purple-500" />
                        <span className="text-muted-foreground">AI Assistant</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-green-500" />
                        <span className="text-muted-foreground">Human</span>
                      </div>
                    </div>
                    {patientOverview.conversationHistory.map((conversation) => (
                      <div 
                        key={conversation.id} 
                        className="flex items-start gap-4 p-4 border rounded-lg hover-elevate cursor-pointer"
                        onClick={() => setExpandedConversationId(conversation.id)}
                        data-testid={`conversation-${conversation.id}`}
                      >
                        <div className="p-2 rounded-full bg-primary/10">
                          <Bot className="h-5 w-5 text-primary" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <h4 className="font-medium truncate">{conversation.title || "Conversation"}</h4>
                            <Badge variant="outline" className="flex-shrink-0">
                              {conversation.messageCount} messages
                            </Badge>
                          </div>
                          {conversation.lastMessage && (
                            <p className="text-sm text-muted-foreground line-clamp-2">
                              {conversation.lastMessage}
                            </p>
                          )}
                          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                            {conversation.createdAt && (
                              <span className="flex items-center gap-1">
                                <Clock className="h-3 w-3" />
                                Started {safeFormatDistance(conversation.createdAt)}
                              </span>
                            )}
                            {conversation.updatedAt && conversation.updatedAt !== conversation.createdAt && (
                              <span>
                                Last activity {safeFormatDistance(conversation.updatedAt)}
                              </span>
                            )}
                          </div>
                        </div>
                        <ChevronRight className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="py-12 text-center">
                    <MessageSquare className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                    <h3 className="text-lg font-medium mb-2">No Conversations Yet</h3>
                    <p className="text-muted-foreground mb-4">
                      No AI assistant conversations found for this patient.
                    </p>
                    <Button 
                      onClick={(e) => openLysaForPatient(selectedPatient, e)}
                      data-testid="button-start-conversation"
                    >
                      <Bot className="h-4 w-4 mr-2" />
                      Start Conversation
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="ai-insights" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Long-Term AI Insights
                </CardTitle>
                <CardDescription>
                  AI-generated insights and patterns from patient interactions over time
                </CardDescription>
              </CardHeader>
              <CardContent>
                {overviewLoading ? (
                  <div className="space-y-3">
                    {[1, 2, 3].map((i) => (
                      <Skeleton key={i} className="h-16" />
                    ))}
                  </div>
                ) : patientOverview?.longTermInsights && patientOverview.longTermInsights.length > 0 ? (
                  <div className="space-y-3">
                    {patientOverview.longTermInsights
                      .sort((a, b) => b.importance - a.importance)
                      .map((insight, idx) => (
                      <div 
                        key={idx} 
                        className={cn(
                          "flex items-start gap-3 p-4 border rounded-lg",
                          insight.importance >= 8 ? "border-primary/50 bg-primary/5" :
                          insight.importance >= 5 ? "border-yellow-500/50 bg-yellow-500/5" : ""
                        )}
                      >
                        <div className={cn(
                          "p-2 rounded-full",
                          insight.type === "health_pattern" ? "bg-red-100 dark:bg-red-900/30" :
                          insight.type === "behavioral" ? "bg-blue-100 dark:bg-blue-900/30" :
                          insight.type === "medication" ? "bg-green-100 dark:bg-green-900/30" :
                          "bg-muted"
                        )}>
                          {insight.type === "health_pattern" ? <Heart className="h-4 w-4 text-red-500" /> :
                           insight.type === "behavioral" ? <Activity className="h-4 w-4 text-blue-500" /> :
                           insight.type === "medication" ? <Pill className="h-4 w-4 text-green-500" /> :
                           <Sparkles className="h-4 w-4 text-muted-foreground" />}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <Badge variant="outline" className="text-xs">
                              {insight.type.replace(/_/g, " ")}
                            </Badge>
                            <div className="flex items-center gap-1">
                              <span className="text-xs text-muted-foreground">Importance:</span>
                              <Badge variant={insight.importance >= 8 ? "default" : "secondary"} className="text-xs">
                                {insight.importance}/10
                              </Badge>
                            </div>
                          </div>
                          <p className="text-sm">{insight.content}</p>
                          {insight.createdAt && (
                            <span className="text-xs text-muted-foreground mt-2 block">
                              Recorded {safeFormatDistance(insight.createdAt)}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="py-12 text-center">
                    <Brain className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                    <h3 className="text-lg font-medium mb-2">No AI Insights Yet</h3>
                    <p className="text-muted-foreground">
                      AI insights will appear here as the system learns from patient interactions and health data over time.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="metrics" className="mt-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {healthMetrics.length > 0 ? (
                healthMetrics.map((metric, i) => {
                  const Icon = getMetricIcon(metric.name);
                  return (
                    <HealthMetricCard
                      key={i}
                      title={metric.name}
                      value={String(metric.value)}
                      unit={metric.unit}
                      icon={Icon}
                      status={metric.status || "normal"}
                      trend={metric.trend}
                      lastUpdated={metric.lastUpdated}
                      testId={`metric-card-${metric.name.toLowerCase().replace(/\s+/g, '-')}`}
                    />
                  );
                })
              ) : (
                <>
                  <HealthMetricCard
                    title="Heart Rate"
                    value="--"
                    unit="bpm"
                    icon={Heart}
                    status="normal"
                    testId="metric-card-heart-rate"
                  />
                  <HealthMetricCard
                    title="Blood Pressure"
                    value="--/--"
                    unit="mmHg"
                    icon={Activity}
                    status="normal"
                    testId="metric-card-blood-pressure"
                  />
                  <HealthMetricCard
                    title="Oxygen Saturation"
                    value="--"
                    unit="%"
                    icon={Droplets}
                    status="normal"
                    testId="metric-card-oxygen"
                  />
                  <HealthMetricCard
                    title="Temperature"
                    value="--"
                    unit="°F"
                    icon={Activity}
                    status="normal"
                    testId="metric-card-temperature"
                  />
                </>
              )}
            </div>
            <Card className="mt-4">
              <CardContent className="py-8 text-center">
                <Activity className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                <p className="text-muted-foreground">
                  Real-time health metrics will appear here when connected to wearable devices or manual entry.
                </p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="ai-alerts" className="mt-4">
            <PatientAIAlerts 
              patientId={selectedPatient.id} 
              patientName={`${selectedPatient.firstName} ${selectedPatient.lastName}`}
            />
          </TabsContent>

          <TabsContent value="ml-tools" className="mt-4">
            <PatientMLTools 
              patientId={selectedPatient.id}
              patientName={`${selectedPatient.firstName} ${selectedPatient.lastName}`}
            />
          </TabsContent>

          <TabsContent value="analytics" className="mt-4">
            <PredictiveAnalyticsDashboard 
              patientContext={{
                id: selectedPatient.id,
                firstName: selectedPatient.firstName || "",
                lastName: selectedPatient.lastName || "",
                allergies: patientProfile?.allergies,
                comorbidities: patientProfile?.medicalConditions,
              }}
            />
          </TabsContent>

          <TabsContent value="prescriptions" className="mt-4">
            <PrescriptionHelper />
          </TabsContent>

          <TabsContent value="clinical" className="mt-4 space-y-6">
            <div className="flex flex-wrap gap-3">
              <DraftDifferentialPanel
                patientId={selectedPatient.id}
                patientName={`${selectedPatient.firstName} ${selectedPatient.lastName}`}
                onDraftApproved={(draftId) => {
                  console.log("Draft approved:", draftId);
                }}
              />
              <CBTSessionPanel
                patientId={selectedPatient.id}
              />
            </div>
            
            <ClinicalDecisionSupport 
              patientContext={{
                id: selectedPatient.id,
                firstName: selectedPatient.firstName || "",
                lastName: selectedPatient.lastName || "",
                allergies: patientProfile?.allergies,
                comorbidities: patientProfile?.medicalConditions,
              }}
            />
            
            <PersonalizedHabitSuggestions patientId={selectedPatient.id} />
          </TabsContent>
        </Tabs>

        <Dialog open={lysaDialogOpen} onOpenChange={setLysaDialogOpen}>
          <DialogContent className="max-w-4xl h-[80vh]" data-testid="dialog-lysa-assistant">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Bot className="h-5 w-5 text-primary" />
                Lysa AI Assistant
                <Badge variant="secondary" className="ml-2">
                  {selectedPatient.firstName} {selectedPatient.lastName}
                </Badge>
              </DialogTitle>
              <DialogDescription>
                AI-powered clinical assistant for patient care and decision support
              </DialogDescription>
            </DialogHeader>
            <div className="flex-1 overflow-hidden -mx-6 -mb-6">
              <LysaChatPanel 
                patientId={selectedPatient.id}
                patientName={`${selectedPatient.firstName} ${selectedPatient.lastName}`}
                className="h-full border-0 rounded-none"
              />
            </div>
          </DialogContent>
        </Dialog>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold flex items-center gap-2" data-testid="text-patients-title">
            <Users className="h-6 w-6" />
            All Patients
          </h2>
          <p className="text-muted-foreground">
            Manage patients, view health data, and access AI tools
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="secondary" className="text-sm">
            <Users className="h-3 w-3 mr-1" />
            {patients.length} Patients
          </Badge>
          <div className="flex border rounded-md">
            <Button
              variant={viewMode === "grid" ? "default" : "ghost"}
              size="sm"
              onClick={() => setViewMode("grid")}
              data-testid="button-view-grid"
            >
              <LayoutGrid className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === "list" ? "default" : "ghost"}
              size="sm"
              onClick={() => setViewMode("list")}
              data-testid="button-view-list"
            >
              <ListIcon className="h-4 w-4" />
            </Button>
          </div>
          <AddPatientDialog />
        </div>
      </div>

      {recommendationsLoading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {[1, 2].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-5 w-1/2 mb-2" />
                <Skeleton className="h-4 w-3/4" />
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[1, 2, 3].map((j) => (
                    <Skeleton key={j} className="h-20 w-full" />
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (clinicalRecommendations.length > 0 || researchRecommendations.length > 0) && (
        <div className="grid gap-4 md:grid-cols-2">
          {clinicalRecommendations.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-primary" />
                  <CardTitle className="text-lg">Clinical Insights</CardTitle>
                </div>
                <CardDescription>
                  AI-powered information from Assistant Lysa for discussion with patients
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {clinicalRecommendations.slice(0, 3).map((rec) => (
                    <div key={rec.id} className="p-3 border rounded-lg hover-elevate" data-testid={`clinical-recommendation-${rec.id}`}>
                      <div className="flex items-start justify-between mb-1">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <Lightbulb className="w-4 h-4 text-yellow-500" />
                            <h4 className="font-medium text-sm">{rec.title}</h4>
                          </div>
                          <p className="text-xs text-muted-foreground">{rec.description}</p>
                        </div>
                        <Badge variant={rec.priority === 'high' ? 'default' : 'secondary'} className="text-xs ml-2">
                          {rec.priority}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 mt-2">
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

          {researchRecommendations.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Beaker className="w-5 h-5 text-primary" />
                  <CardTitle className="text-lg">Research Insights</CardTitle>
                </div>
                <CardDescription>
                  Latest research and clinical literature
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {researchRecommendations.slice(0, 3).map((rec) => (
                    <div key={rec.id} className="p-3 border rounded-lg hover-elevate" data-testid={`research-recommendation-${rec.id}`}>
                      <div className="flex items-start justify-between mb-1">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <BookOpen className="w-4 h-4 text-blue-500" />
                            <h4 className="font-medium text-sm">{rec.title}</h4>
                          </div>
                          <p className="text-xs text-muted-foreground">{rec.description}</p>
                        </div>
                        <Badge variant={rec.priority === 'high' ? 'default' : 'secondary'} className="text-xs ml-2">
                          {rec.priority}
                        </Badge>
                      </div>
                      {rec.reasoning && (
                        <p className="text-xs text-muted-foreground mt-2 italic">
                          {rec.reasoning}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      <Card>
        <CardHeader>
          <div className="flex items-center gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search patients by name or email..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
                data-testid="input-search-patients"
              />
            </div>
          </div>
        </CardHeader>
      </Card>

      {patientsLoading || enrichedLoading ? (
        <div className={viewMode === "grid" ? "grid gap-4 md:grid-cols-2 lg:grid-cols-3" : "space-y-2"}>
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Card key={i}>
              <CardContent className="p-6">
                <div className="flex items-center gap-3">
                  <Skeleton className="h-12 w-12 rounded-full" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-3 w-1/2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : filteredPatients.length > 0 ? (
        <div className={viewMode === "grid" ? "grid gap-4 md:grid-cols-2 lg:grid-cols-3" : "space-y-2"}>
          {filteredPatients.map((patient) => {
            const enrichedData = enrichedPatientMap[patient.id];
            
            return (
              <Card
                key={patient.id}
                className="hover-elevate cursor-pointer"
                onClick={() => handlePatientSelect(patient)}
                data-testid={`card-patient-${patient.id}`}
              >
                <CardContent className={viewMode === "grid" ? "p-6" : "p-4"}>
                  <div className={`flex items-${viewMode === "grid" ? "start" : "center"} gap-3 ${viewMode === "grid" ? "mb-4" : ""}`}>
                    <div className="relative">
                      <Avatar className="h-12 w-12">
                        <AvatarFallback className="bg-primary text-primary-foreground">
                          {getInitials(patient.firstName, patient.lastName)}
                        </AvatarFallback>
                      </Avatar>
                      {enrichedData?.isOnline && (
                        <Circle className="absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 fill-chart-2 text-chart-2" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <h3 className="font-semibold truncate" data-testid={`text-patient-name-${patient.id}`}>
                          {patient.firstName} {patient.lastName}
                        </h3>
                        <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      </div>
                      <p className="text-sm text-muted-foreground truncate">{patient.email}</p>
                    </div>
                    {viewMode === "list" && (
                      <div className="flex items-center gap-2">
                        {enrichedData && (
                          <Badge variant={getRiskBadgeVariant(enrichedData.riskScore)} className="text-xs">
                            Risk: {(enrichedData.riskScore ?? 0).toFixed(1)}
                          </Badge>
                        )}
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex items-center gap-1.5 text-muted-foreground">
                              {monitoringStates[patient.id] ? (
                                <Eye className="h-3.5 w-3.5 text-green-500" />
                              ) : (
                                <EyeOff className="h-3.5 w-3.5" />
                              )}
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{monitoringStates[patient.id] 
                              ? "AI is actively monitoring this patient" 
                              : "AI monitoring is disabled"}</p>
                          </TooltipContent>
                        </Tooltip>
                        <Switch
                          checked={monitoringStates[patient.id] || false}
                          onCheckedChange={(checked) => handleMonitoringToggle(patient.id, checked, { stopPropagation: () => {} } as React.MouseEvent)}
                          onClick={(e) => e.stopPropagation()}
                          disabled={toggleMonitoringMutation.isPending}
                          data-testid={`switch-monitoring-${patient.id}`}
                        />
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={(e) => openLysaForPatient(patient, e)}
                          data-testid={`button-lysa-patient-${patient.id}`}
                        >
                          <Bot className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </div>

                  {viewMode === "grid" && (
                    <>
                      <Separator className="my-3" />
                      
                      <div className="grid grid-cols-3 gap-2 text-center mb-3">
                        <div>
                          <div className={cn("text-lg font-semibold", getRiskColor(enrichedData?.riskScore))}>
                            {enrichedData?.riskScore?.toFixed(1) || "—"}
                          </div>
                          <div className="text-xs text-muted-foreground">Risk</div>
                        </div>
                        <div>
                          <div className={cn("text-lg font-semibold", (enrichedData?.activeAlerts || 0) > 0 ? "text-destructive" : "")}>
                            {enrichedData?.activeAlerts ?? 0}
                          </div>
                          <div className="text-xs text-muted-foreground">Alerts</div>
                        </div>
                        <div>
                          <div className="text-lg font-semibold">
                            {enrichedData?.medicationCount ?? 0}
                          </div>
                          <div className="text-xs text-muted-foreground">Meds</div>
                        </div>
                      </div>

                      {enrichedData?.lastInteraction && (
                        <div className="flex items-center gap-1 text-xs text-muted-foreground mb-3">
                          <Clock className="h-3 w-3" />
                          Last: {safeFormatDistance(enrichedData.lastInteraction)}
                        </div>
                      )}

                      <div className="flex items-center justify-between text-sm pt-2 border-t">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex items-center gap-1.5 text-muted-foreground cursor-help">
                              {monitoringStates[patient.id] ? (
                                <Eye className="h-3.5 w-3.5 text-green-500" />
                              ) : (
                                <EyeOff className="h-3.5 w-3.5" />
                              )}
                              <span>AI Monitoring</span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{monitoringStates[patient.id] 
                              ? "AI is actively monitoring this patient for health changes" 
                              : "Enable AI monitoring to track health changes"}</p>
                          </TooltipContent>
                        </Tooltip>
                        <Switch
                          checked={monitoringStates[patient.id] || false}
                          onCheckedChange={(checked) => handleMonitoringToggle(patient.id, checked, { stopPropagation: () => {} } as React.MouseEvent)}
                          onClick={(e) => e.stopPropagation()}
                          disabled={toggleMonitoringMutation.isPending}
                          data-testid={`switch-monitoring-${patient.id}`}
                        />
                      </div>

                      <div className="flex gap-2 mt-4">
                        <Button 
                          className="flex-1" 
                          variant="outline" 
                          data-testid={`button-view-patient-${patient.id}`}
                        >
                          View Details
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={(e) => openLysaForPatient(patient, e)}
                          data-testid={`button-lysa-patient-${patient.id}`}
                          className="shrink-0"
                          title="Open Lysa AI Assistant"
                        >
                          <Bot className="h-4 w-4" />
                        </Button>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      ) : (
        <Card>
          <CardContent className="p-12 text-center">
            <Users className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
            <h3 className="text-lg font-semibold mb-2">No patients found</h3>
            <p className="text-muted-foreground">
              {searchQuery ? "Try adjusting your search" : "No patients have been added yet"}
            </p>
          </CardContent>
        </Card>
      )}

      <Dialog open={lysaDialogOpen} onOpenChange={setLysaDialogOpen}>
        <DialogContent className="max-w-4xl h-[80vh]" data-testid="dialog-lysa-assistant-list">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-primary" />
              Lysa AI Assistant
              {selectedPatient && (
                <Badge variant="secondary" className="ml-2">
                  {selectedPatient.firstName} {selectedPatient.lastName}
                </Badge>
              )}
            </DialogTitle>
            <DialogDescription>
              AI-powered clinical assistant for patient care and decision support
            </DialogDescription>
          </DialogHeader>
          <div className="flex-1 overflow-hidden -mx-6 -mb-6">
            {selectedPatient && (
              <LysaChatPanel 
                patientId={selectedPatient.id}
                patientName={`${selectedPatient.firstName} ${selectedPatient.lastName}`}
                className="h-full border-0 rounded-none"
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default PatientManagementPanel;
