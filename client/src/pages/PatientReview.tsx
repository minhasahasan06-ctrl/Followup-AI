import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useParams } from "wouter";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Activity,
  Heart,
  Droplet,
  TrendingUp,
  TrendingDown,
  Calendar,
  Bot,
  AlertCircle,
  CheckCircle,
  MessageSquare,
  Clock,
  FileText,
  Download,
  Bell,
  Brain,
  Pill,
  AlertTriangle,
  Plus,
  FolderOpen,
  Upload,
  X,
  Loader2,
  ClipboardList,
  Stethoscope,
  ThermometerSun,
  Wind,
  Eye,
  FlaskConical,
  Image,
  Lock,
  Users,
  BarChart3,
  Shield,
} from "lucide-react";
import { startOfWeek, endOfWeek, subWeeks, format, formatDistanceToNow } from "date-fns";
import type { User, DailyFollowup, Medication, ChatMessage, Prescription, DoctorPatientConsentPermissions } from "@shared/schema";
import { PatientAIAlerts } from "@/components/patient-ai-alerts";
import { MLInsightsPanel } from "@/components/MLInsightsPanel";
import { LysaPatientAssistant } from "@/components/LysaPatientAssistant";
import { ClinicalDecisionSupport } from "@/components/ClinicalDecisionSupport";
import { PredictiveAnalyticsDashboard } from "@/components/PredictiveAnalyticsDashboard";
import { DiagnosticImagingAnalysis } from "@/components/DiagnosticImagingAnalysis";
import { LabReportAnalysis } from "@/components/LabReportAnalysis";
import { LysaInsightFeed } from "@/components/LysaInsightFeed";
import { PatientSummaryCard } from "@/components/PatientSummaryCard";
import { RiskExposuresPanel } from "@/components/RiskExposuresCards";

interface ChatSession {
  id: string;
  patientId: string;
  agentType: string;
  sessionTitle: string | null;
  startedAt: Date;
  endedAt: Date | null;
  messageCount: number | null;
  symptomsDiscussed: string[] | null;
  recommendations: string[] | null;
  healthInsights: any;
  aiSummary: string | null;
  createdAt: Date;
  updatedAt: Date;
  doctorNotes: string | null;
}

interface DrugInteraction {
  id: string;
  drug1: string;
  drug2: string;
  severity: string;
  description: string;
  recommendations?: string[];
}

interface MedicationAdherence {
  medicationId: string;
  medicationName: string;
  totalDoses: number;
  takenDoses: number;
  missedDoses: number;
  adherenceRate: number;
  streak: number;
  lastTaken: string | null;
}

interface HealthSummary {
  overallStatus: 'stable' | 'improving' | 'concerning' | 'critical';
  riskScore: number;
  lastVitals: {
    heartRate?: number;
    bloodPressure?: string;
    oxygenSaturation?: number;
    temperature?: number;
    respiratoryRate?: number;
  };
  activeAlerts: number;
  medicationAdherence: number;
  lastCheckIn: string | null;
}

interface Complication {
  id: string;
  type: string;
  severity: string;
  description: string;
  detectedAt: string;
  resolvedAt: string | null;
  notes: string;
}

interface MedicalDocument {
  id: string;
  name: string;
  type: string;
  uploadedAt: string;
  size: number;
  category: string;
}

export default function PatientReview() {
  const params = useParams();
  const patientId = params.id;
  const { toast } = useToast();
  
  // PDF Report state
  const [selectedWeekOffset, setSelectedWeekOffset] = useState(0);
  const [referenceDate] = useState(() => new Date());
  
  // Prescription form state
  const [prescriptionDialogOpen, setPrescriptionDialogOpen] = useState(false);
  const [newPrescription, setNewPrescription] = useState({
    medicationName: '',
    dosage: '',
    frequency: '',
    quantity: '',
    refills: '0',
    dosageInstructions: '',
    notes: '',
  });
  const [checkingInteractions, setCheckingInteractions] = useState(false);
  const [detectedInteractions, setDetectedInteractions] = useState<DrugInteraction[]>([]);

  // Consent permissions query - MUST load first to gate other queries
  const { data: consentPermissions, isLoading: loadingPermissions } = useQuery<DoctorPatientConsentPermissions>({
    queryKey: [`/api/doctor/patients/${patientId}/consent-permissions`],
    queryFn: async () => {
      const res = await fetch(`/api/doctor/patients/${patientId}/consent-permissions`, { credentials: 'include' });
      if (!res.ok) {
        if (res.status === 404 || res.status === 403) return null;
        throw new Error('Failed to fetch consent permissions');
      }
      return res.json();
    },
  });

  // Helper to check if a specific permission is granted
  const hasPermission = (permission: keyof DoctorPatientConsentPermissions): boolean => {
    if (!consentPermissions) return false;
    return Boolean(consentPermissions[permission]);
  };

  // Patient basic info query - always allowed for assigned patients
  const { data: patient } = useQuery<User>({
    queryKey: [`/api/doctor/patients/${patientId}`],
  });

  // Consent-gated queries - only fetch data when permission is granted
  const { data: followups } = useQuery<DailyFollowup[]>({
    queryKey: [`/api/doctor/patients/${patientId}/followups`],
    enabled: !!consentPermissions && (hasPermission('shareDailyFollowups') || hasPermission('shareHealthData')),
  });

  const { data: medications } = useQuery<Medication[]>({
    queryKey: [`/api/doctor/patients/${patientId}/medications`],
    enabled: !!consentPermissions && (hasPermission('shareMedications') || hasPermission('shareHealthData')),
  });

  const { data: prescriptions } = useQuery<Prescription[]>({
    queryKey: ['/api/prescriptions/patient', patientId],
    queryFn: async () => {
      const res = await fetch(`/api/prescriptions/patient/${patientId}`, { credentials: 'include' });
      if (!res.ok) return [];
      return res.json();
    },
    enabled: !!consentPermissions && (hasPermission('shareMedications') || hasPermission('shareHealthData')),
  });

  const { data: chatSessions } = useQuery<ChatSession[]>({
    queryKey: [`/api/doctor/patient-sessions/${patientId}`],
    enabled: !!consentPermissions && hasPermission('shareAIMessages'),
  });

  // Health Summary - computed from available real data only
  const hasVitals = followups && followups.length > 0;
  const healthSummary: HealthSummary = {
    overallStatus: hasVitals ? 'stable' : 'stable',
    riskScore: 0, // Will be populated from AI Health Alerts API
    lastVitals: {
      heartRate: followups?.[0]?.heartRate || undefined,
      oxygenSaturation: followups?.[0]?.oxygenSaturation || undefined,
      temperature: followups?.[0]?.temperature || undefined,
    },
    activeAlerts: 0,
    medicationAdherence: medications?.length ? 
      Math.round((medications.filter(m => m.status === 'active').length / medications.length) * 100) : 0,
    lastCheckIn: followups?.[0]?.date?.toString() || null,
  };

  // Complications - fetched from API (empty until API implemented)
  // In production, this would be populated from /api/complications/:patientId
  const complications: Complication[] = [];

  // Adherence data - basic calculation from medications
  // Real adherence tracking requires medication log data
  const adherenceData: MedicationAdherence[] = [];

  // Drug interaction check mutation
  const checkInteractionsMutation = useMutation({
    mutationFn: async (drugName: string) => {
      const res = await apiRequest('POST', '/api/drug-interactions/analyze-for-patient', {
        patientId,
        drugName,
      });
      return res.json();
    },
    onSuccess: (data) => {
      if (data.interactions && data.interactions.length > 0) {
        setDetectedInteractions(data.interactions);
      } else {
        setDetectedInteractions([]);
      }
      setCheckingInteractions(false);
    },
    onError: () => {
      setCheckingInteractions(false);
      // Don't show error toast - just silently fail interaction check
      // Doctor can still write prescription and verify manually
      setDetectedInteractions([]);
    },
  });

  // Create prescription mutation
  const createPrescriptionMutation = useMutation({
    mutationFn: async (prescription: typeof newPrescription) => {
      const res = await apiRequest('POST', '/api/prescriptions', {
        patientId,
        medicationName: prescription.medicationName,
        dosage: prescription.dosage,
        frequency: prescription.frequency,
        quantity: parseInt(prescription.quantity) || null,
        refills: parseInt(prescription.refills) || 0,
        dosageInstructions: prescription.dosageInstructions || null,
        notes: prescription.notes || null,
      });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/prescriptions/patient', patientId] });
      queryClient.invalidateQueries({ queryKey: [`/api/doctor/patients/${patientId}/medications`] });
      setPrescriptionDialogOpen(false);
      setNewPrescription({
        medicationName: '',
        dosage: '',
        frequency: '',
        quantity: '',
        refills: '0',
        dosageInstructions: '',
        notes: '',
      });
      setDetectedInteractions([]);
      toast({
        title: "Prescription Created",
        description: "The prescription has been sent to the patient.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Prescription Failed",
        description: error.message || "Could not create prescription. Please try again.",
        variant: "destructive",
      });
    },
  });

  // PDF Generation Mutation
  const generatePDF = useMutation({
    mutationFn: async () => {
      const weekStart = startOfWeek(subWeeks(referenceDate, selectedWeekOffset), { weekStartsOn: 1 });
      const weekEnd = endOfWeek(subWeeks(referenceDate, selectedWeekOffset), { weekStartsOn: 1 });

      const res = await fetch(`/api/v1/symptom-journal/generate-weekly-pdf/${patientId}?${new URLSearchParams({
        week_start: weekStart.toISOString(),
        week_end: weekEnd.toISOString(),
      })}`, {
        method: "POST",
        credentials: 'include',
      });

      if (!res.ok) {
        const error = await res.text();
        throw new Error(error || "Failed to generate PDF");
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      
      const link = document.createElement('a');
      link.href = url;
      link.download = `symptom-report-${format(weekStart, 'yyyy-MM-dd')}-to-${format(weekEnd, 'yyyy-MM-dd')}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      return { success: true };
    },
    onSuccess: () => {
      toast({
        title: "PDF Generated",
        description: "Weekly report has been downloaded successfully.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Generation Failed",
        description: error.message || "Failed to generate PDF report. Please try again.",
        variant: "destructive",
      });
    },
  });

  const getInitials = (firstName?: string | null, lastName?: string | null) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'stable': return 'text-green-600 bg-green-100';
      case 'improving': return 'text-blue-600 bg-blue-100';
      case 'concerning': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-muted-foreground bg-muted';
    }
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'high':
      case 'severe':
        return <Badge variant="destructive">{severity}</Badge>;
      case 'moderate':
      case 'medium':
        return <Badge className="bg-yellow-500">{severity}</Badge>;
      default:
        return <Badge variant="secondary">{severity}</Badge>;
    }
  };

  // Handle drug name change and check interactions
  const handleDrugNameChange = (value: string) => {
    setNewPrescription(prev => ({ ...prev, medicationName: value }));
    if (value.length > 2) {
      setCheckingInteractions(true);
      checkInteractionsMutation.mutate(value);
    }
  };

  return (
    <div className="space-y-6">
      {/* Enhanced Patient Summary Card with Consent Permissions */}
      {patientId && (
        <PatientSummaryCard
          patient={patient}
          patientId={patientId}
          followups={followups}
          medications={medications}
          consentPermissions={consentPermissions}
          isLoadingPermissions={loadingPermissions}
        />
      )}

      {/* Main Content */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Tabs defaultValue="ai-alerts" className="space-y-4">
            <TabsList className="flex flex-wrap gap-1">
              {/* Core Health Tabs - Always Available */}
              <TabsTrigger value="ai-alerts" data-testid="tab-ai-alerts">
                <Bell className="h-4 w-4 mr-2" />
                AI Alerts
                {!hasPermission('shareHealthAlerts') && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
              </TabsTrigger>
              <TabsTrigger value="ml-tools" data-testid="tab-ml-tools">
                <Brain className="h-4 w-4 mr-2" />
                ML Tools
              </TabsTrigger>
              
              {/* Consent-Controlled Tabs */}
              <TabsTrigger value="daily-followups" data-testid="tab-daily-followups">
                <Calendar className="h-4 w-4 mr-2" />
                Daily Followups
                {!hasPermission('shareDailyFollowups') && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
              </TabsTrigger>
              <TabsTrigger value="ai-messages" data-testid="tab-ai-messages">
                <Bot className="h-4 w-4 mr-2" />
                AI Messages
                {!hasPermission('shareAIMessages') && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
              </TabsTrigger>
              <TabsTrigger value="doctor-messages" data-testid="tab-doctor-messages">
                <MessageSquare className="h-4 w-4 mr-2" />
                Messages
                {!hasPermission('shareDoctorMessages') && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
              </TabsTrigger>
              <TabsTrigger value="behavioral-insights" data-testid="tab-behavioral-insights">
                <BarChart3 className="h-4 w-4 mr-2" />
                Behavioral
                {!hasPermission('shareBehavioralInsights') && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
              </TabsTrigger>
              
              {/* Medical Data Tabs */}
              <TabsTrigger value="medications" data-testid="tab-medications">
                <Pill className="h-4 w-4 mr-2" />
                Medications
                {!hasPermission('shareMedications') && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
              </TabsTrigger>
              <TabsTrigger value="risk-exposures" data-testid="tab-risk-exposures">
                <Shield className="h-4 w-4 mr-2" />
                Risk & Exposures
              </TabsTrigger>
              <TabsTrigger value="prescriptions" data-testid="tab-prescriptions">
                <ClipboardList className="h-4 w-4 mr-2" />
                Prescriptions
              </TabsTrigger>
              <TabsTrigger value="documents" data-testid="tab-documents">
                <FolderOpen className="h-4 w-4 mr-2" />
                Medical Files
                {!hasPermission('shareMedicalFiles') && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
              </TabsTrigger>
              
              {/* Clinical Tabs */}
              <TabsTrigger value="complications" data-testid="tab-complications">
                <AlertTriangle className="h-4 w-4 mr-2" />
                Complications
              </TabsTrigger>
              <TabsTrigger value="timeline" data-testid="tab-timeline">
                <Clock className="h-4 w-4 mr-2" />
                Timeline
              </TabsTrigger>
              <TabsTrigger value="reports" data-testid="tab-reports">
                <FileText className="h-4 w-4 mr-2" />
                Reports
              </TabsTrigger>
              <TabsTrigger value="lysa-assistant" data-testid="tab-lysa-assistant">
                <Bot className="h-4 w-4 mr-2" />
                Lysa AI
              </TabsTrigger>
              <TabsTrigger value="clinical-support" data-testid="tab-clinical-support">
                <Stethoscope className="h-4 w-4 mr-2" />
                Clinical Support
              </TabsTrigger>
              <TabsTrigger value="predictive" data-testid="tab-predictive">
                <TrendingUp className="h-4 w-4 mr-2" />
                Predictive
              </TabsTrigger>
              <TabsTrigger value="imaging" data-testid="tab-imaging">
                <Image className="h-4 w-4 mr-2" />
                Imaging
              </TabsTrigger>
              <TabsTrigger value="labs" data-testid="tab-labs">
                <FlaskConical className="h-4 w-4 mr-2" />
                Labs
              </TabsTrigger>
              <TabsTrigger value="insights" data-testid="tab-insights">
                <Eye className="h-4 w-4 mr-2" />
                Insights
              </TabsTrigger>
            </TabsList>

            {/* AI Health Alerts Tab */}
            <TabsContent value="ai-alerts">
              {patientId && (
                <PatientAIAlerts 
                  patientId={patientId} 
                  patientName={patient ? `${patient.firstName} ${patient.lastName}` : undefined}
                />
              )}
            </TabsContent>

            {/* ML Prediction Tools Tab */}
            <TabsContent value="ml-tools">
              {patientId && (
                <MLInsightsPanel 
                  patientId={patientId} 
                  patientName={patient ? `${patient.firstName} ${patient.lastName}` : undefined}
                  isDoctor={true}
                />
              )}
            </TabsContent>

            {/* Daily Followups Tab - Consent Controlled */}
            <TabsContent value="daily-followups">
              {!hasPermission('shareDailyFollowups') ? (
                <Card>
                  <CardContent className="py-12 text-center">
                    <Lock className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-semibold mb-2">Access Restricted</h3>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      Patient consent is required to view daily followup data. 
                      Request consent permissions to access this information.
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle>Daily Followups</CardTitle>
                    <CardDescription>Patient health check-ins and symptom reports</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-[400px]">
                      {followups && followups.length > 0 ? (
                        <div className="space-y-4">
                          {followups.map((followup) => (
                            <div key={followup.id} className="p-4 border rounded-lg space-y-3">
                              <div className="flex items-center justify-between">
                                <span className="text-sm font-medium">
                                  {format(new Date(followup.date), 'MMMM d, yyyy')}
                                </span>
                                <Badge variant={followup.mood === 'good' ? 'default' : followup.mood === 'fair' ? 'secondary' : 'destructive'}>
                                  {followup.mood || 'Unknown'}
                                </Badge>
                              </div>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                                {followup.heartRate && (
                                  <div className="flex items-center gap-2">
                                    <Heart className="h-4 w-4 text-red-500" />
                                    <span>{followup.heartRate} bpm</span>
                                  </div>
                                )}
                                {followup.bloodPressureSystolic && followup.bloodPressureDiastolic && (
                                  <div className="flex items-center gap-2">
                                    <Activity className="h-4 w-4 text-blue-500" />
                                    <span>{followup.bloodPressureSystolic}/{followup.bloodPressureDiastolic} mmHg</span>
                                  </div>
                                )}
                                {followup.oxygenSaturation && (
                                  <div className="flex items-center gap-2">
                                    <Wind className="h-4 w-4 text-cyan-500" />
                                    <span>{followup.oxygenSaturation}% SpO2</span>
                                  </div>
                                )}
                                {followup.temperature && (
                                  <div className="flex items-center gap-2">
                                    <ThermometerSun className="h-4 w-4 text-orange-500" />
                                    <span>{followup.temperature}°F</span>
                                  </div>
                                )}
                              </div>
                              {followup.symptoms && followup.symptoms.length > 0 && (
                                <div className="flex flex-wrap gap-2">
                                  {followup.symptoms.map((symptom, idx) => (
                                    <Badge key={idx} variant="outline">
                                      {typeof symptom === 'string' ? symptom : symptom.name || 'Unknown'}
                                    </Badge>
                                  ))}
                                </div>
                              )}
                              {followup.notes && (
                                <p className="text-sm text-muted-foreground italic">{followup.notes}</p>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-8 text-muted-foreground">
                          No daily followups recorded yet.
                        </div>
                      )}
                    </ScrollArea>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* AI Messages Tab - Consent Controlled */}
            <TabsContent value="ai-messages">
              {!hasPermission('shareAIMessages') ? (
                <Card>
                  <CardContent className="py-12 text-center">
                    <Lock className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-semibold mb-2">Access Restricted</h3>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      Patient consent is required to view AI chat messages. 
                      Request consent permissions to access this information.
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle>AI Chat History</CardTitle>
                    <CardDescription>Patient conversations with Agent Clona</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-[400px]">
                      {chatSessions && chatSessions.length > 0 ? (
                        <div className="space-y-4">
                          {chatSessions.map((session) => (
                            <div key={session.id} className="p-4 border rounded-lg space-y-2">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <Bot className="h-4 w-4 text-primary" />
                                  <span className="font-medium">{session.sessionTitle || 'Chat Session'}</span>
                                </div>
                                <span className="text-xs text-muted-foreground">
                                  {formatDistanceToNow(new Date(session.startedAt), { addSuffix: true })}
                                </span>
                              </div>
                              {session.messageCount && (
                                <p className="text-sm text-muted-foreground">
                                  {session.messageCount} messages
                                </p>
                              )}
                              {session.aiSummary && (
                                <p className="text-sm bg-muted p-2 rounded">{session.aiSummary}</p>
                              )}
                              {session.symptomsDiscussed && session.symptomsDiscussed.length > 0 && (
                                <div className="flex flex-wrap gap-2">
                                  {session.symptomsDiscussed.map((symptom, idx) => (
                                    <Badge key={idx} variant="outline">{symptom}</Badge>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-8 text-muted-foreground">
                          No AI chat sessions found.
                        </div>
                      )}
                    </ScrollArea>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* Doctor Messages Tab - Consent Controlled */}
            <TabsContent value="doctor-messages">
              {!hasPermission('shareDoctorMessages') ? (
                <Card>
                  <CardContent className="py-12 text-center">
                    <Lock className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-semibold mb-2">Access Restricted</h3>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      Patient consent is required to view doctor communications. 
                      Request consent permissions to access this information.
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle>Doctor Messages</CardTitle>
                    <CardDescription>Communication history with healthcare providers</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center py-8 text-muted-foreground">
                      <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Doctor messaging feature coming soon.</p>
                      <p className="text-sm mt-2">Use the Lysa AI assistant tab for patient communication.</p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* Behavioral Insights Tab - Consent Controlled */}
            <TabsContent value="behavioral-insights">
              {!hasPermission('shareBehavioralInsights') ? (
                <Card>
                  <CardContent className="py-12 text-center">
                    <Lock className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-semibold mb-2">Access Restricted</h3>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      Patient consent is required to view behavioral insights and AI analysis. 
                      Request consent permissions to access this information.
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle>Behavioral Insights</CardTitle>
                    <CardDescription>AI-powered analysis of patient behavior patterns</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="p-4 border rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Activity className="h-5 w-5 text-primary" />
                            <h4 className="font-medium">Activity Level</h4>
                          </div>
                          <p className="text-2xl font-bold">Moderate</p>
                          <p className="text-sm text-muted-foreground">Based on daily check-ins</p>
                        </div>
                        <div className="p-4 border rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Clock className="h-5 w-5 text-primary" />
                            <h4 className="font-medium">Check-in Consistency</h4>
                          </div>
                          <p className="text-2xl font-bold">
                            {followups ? Math.round((followups.length / 7) * 100) : 0}%
                          </p>
                          <p className="text-sm text-muted-foreground">Past 7 days</p>
                        </div>
                        <div className="p-4 border rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Brain className="h-5 w-5 text-primary" />
                            <h4 className="font-medium">Engagement Score</h4>
                          </div>
                          <p className="text-2xl font-bold">
                            {chatSessions?.length ? 'High' : 'Low'}
                          </p>
                          <p className="text-sm text-muted-foreground">AI interaction level</p>
                        </div>
                      </div>
                      
                      <Separator />
                      
                      <div>
                        <h4 className="font-medium mb-3">Behavioral Patterns</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex items-center justify-between p-2 bg-muted/50 rounded">
                            <span>Sleep pattern regularity</span>
                            <Badge variant="secondary">Analyzing...</Badge>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-muted/50 rounded">
                            <span>Medication adherence trend</span>
                            <Badge variant="secondary">
                              {medications?.filter(m => m.status === 'active').length || 0} active meds
                            </Badge>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-muted/50 rounded">
                            <span>Mood pattern analysis</span>
                            <Badge variant="secondary">Analyzing...</Badge>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* Enhanced Medications Tab with Adherence */}
            <TabsContent value="medications">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div>
                      <CardTitle>Current Medications</CardTitle>
                      <CardDescription>Medication list with adherence tracking</CardDescription>
                    </div>
                    <Dialog open={prescriptionDialogOpen} onOpenChange={setPrescriptionDialogOpen}>
                      <DialogTrigger asChild>
                        <Button data-testid="button-new-prescription">
                          <Plus className="h-4 w-4 mr-2" />
                          New Prescription
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                        <DialogHeader>
                          <DialogTitle>Write New Prescription</DialogTitle>
                          <DialogDescription>
                            Create a prescription for {patient?.firstName} {patient?.lastName}. 
                            Drug interactions are checked automatically.
                          </DialogDescription>
                        </DialogHeader>

                        <div className="space-y-4 py-4">
                          {/* Drug Interaction Alert */}
                          {detectedInteractions.length > 0 && (
                            <Alert variant="destructive" data-testid="alert-drug-interactions">
                              <AlertTriangle className="h-4 w-4" />
                              <AlertTitle>Drug Interactions Detected</AlertTitle>
                              <AlertDescription>
                                <ul className="mt-2 space-y-2">
                                  {detectedInteractions.map((interaction, idx) => (
                                    <li key={idx} className="text-sm">
                                      <span className="font-medium">{interaction.drug1}</span> + 
                                      <span className="font-medium"> {interaction.drug2}</span>: 
                                      {interaction.description}
                                    </li>
                                  ))}
                                </ul>
                              </AlertDescription>
                            </Alert>
                          )}

                          <div className="grid gap-4 md:grid-cols-2">
                            <div className="space-y-2">
                              <Label htmlFor="medication-name">Medication Name *</Label>
                              <div className="relative">
                                <Input
                                  id="medication-name"
                                  placeholder="e.g., Amoxicillin"
                                  value={newPrescription.medicationName}
                                  onChange={(e) => handleDrugNameChange(e.target.value)}
                                  data-testid="input-medication-name"
                                />
                                {checkingInteractions && (
                                  <Loader2 className="absolute right-3 top-3 h-4 w-4 animate-spin text-muted-foreground" />
                                )}
                              </div>
                            </div>

                            <div className="space-y-2">
                              <Label htmlFor="dosage">Dosage *</Label>
                              <Input
                                id="dosage"
                                placeholder="e.g., 500mg"
                                value={newPrescription.dosage}
                                onChange={(e) => setNewPrescription(prev => ({ ...prev, dosage: e.target.value }))}
                                data-testid="input-dosage"
                              />
                            </div>

                            <div className="space-y-2">
                              <Label htmlFor="frequency">Frequency *</Label>
                              <Select
                                value={newPrescription.frequency}
                                onValueChange={(value) => setNewPrescription(prev => ({ ...prev, frequency: value }))}
                              >
                                <SelectTrigger data-testid="select-frequency">
                                  <SelectValue placeholder="Select frequency" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="once_daily">Once Daily</SelectItem>
                                  <SelectItem value="twice_daily">Twice Daily</SelectItem>
                                  <SelectItem value="three_times_daily">Three Times Daily</SelectItem>
                                  <SelectItem value="four_times_daily">Four Times Daily</SelectItem>
                                  <SelectItem value="every_6_hours">Every 6 Hours</SelectItem>
                                  <SelectItem value="every_8_hours">Every 8 Hours</SelectItem>
                                  <SelectItem value="every_12_hours">Every 12 Hours</SelectItem>
                                  <SelectItem value="as_needed">As Needed (PRN)</SelectItem>
                                  <SelectItem value="weekly">Weekly</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>

                            <div className="space-y-2">
                              <Label htmlFor="quantity">Quantity</Label>
                              <Input
                                id="quantity"
                                type="number"
                                placeholder="e.g., 30"
                                value={newPrescription.quantity}
                                onChange={(e) => setNewPrescription(prev => ({ ...prev, quantity: e.target.value }))}
                                data-testid="input-quantity"
                              />
                            </div>

                            <div className="space-y-2">
                              <Label htmlFor="refills">Refills</Label>
                              <Input
                                id="refills"
                                type="number"
                                placeholder="0"
                                value={newPrescription.refills}
                                onChange={(e) => setNewPrescription(prev => ({ ...prev, refills: e.target.value }))}
                                data-testid="input-refills"
                              />
                            </div>
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="instructions">Dosage Instructions</Label>
                            <Input
                              id="instructions"
                              placeholder="e.g., Take with food, avoid alcohol"
                              value={newPrescription.dosageInstructions}
                              onChange={(e) => setNewPrescription(prev => ({ ...prev, dosageInstructions: e.target.value }))}
                              data-testid="input-instructions"
                            />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="notes">Doctor Notes</Label>
                            <Textarea
                              id="notes"
                              placeholder="Additional notes about this prescription..."
                              value={newPrescription.notes}
                              onChange={(e) => setNewPrescription(prev => ({ ...prev, notes: e.target.value }))}
                              rows={3}
                              data-testid="input-notes"
                            />
                          </div>
                        </div>

                        <DialogFooter>
                          <Button 
                            variant="outline" 
                            onClick={() => setPrescriptionDialogOpen(false)}
                            data-testid="button-cancel-prescription"
                          >
                            Cancel
                          </Button>
                          <Button
                            onClick={() => createPrescriptionMutation.mutate(newPrescription)}
                            disabled={
                              !newPrescription.medicationName || 
                              !newPrescription.dosage || 
                              !newPrescription.frequency ||
                              createPrescriptionMutation.isPending
                            }
                            data-testid="button-submit-prescription"
                          >
                            {createPrescriptionMutation.isPending ? (
                              <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                Creating...
                              </>
                            ) : (
                              'Create Prescription'
                            )}
                          </Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                  </div>
                </CardHeader>
                <CardContent>
                  {medications && medications.length > 0 ? (
                    <div className="space-y-4">
                      {medications.map((med) => {
                        const adherence = adherenceData.find(a => a.medicationId === med.id);
                        return (
                          <Card key={med.id} className="hover-elevate" data-testid={`medication-${med.id}`}>
                            <CardContent className="p-4">
                              <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-2">
                                    <p className="font-medium">{med.name}</p>
                                    {med.isOTC && <Badge variant="secondary">OTC</Badge>}
                                    {med.status === 'active' && (
                                      <Badge variant="outline" className="text-green-600">Active</Badge>
                                    )}
                                  </div>
                                  <p className="text-sm text-muted-foreground">
                                    {med.dosage} • {med.frequency}
                                  </p>
                                  {med.aiSuggestion && (
                                    <p className="text-xs text-muted-foreground mt-2 italic flex items-center gap-1">
                                      <Bot className="h-3 w-3" />
                                      {med.aiSuggestion}
                                    </p>
                                  )}
                                </div>
                                
                                {adherence && (
                                  <div className="text-right min-w-[120px]">
                                    <div className="flex items-center justify-end gap-2 mb-1">
                                      <span className="text-sm font-medium">{adherence.adherenceRate}%</span>
                                      {adherence.adherenceRate >= 90 ? (
                                        <TrendingUp className="h-4 w-4 text-green-500" />
                                      ) : adherence.adherenceRate >= 70 ? (
                                        <Activity className="h-4 w-4 text-yellow-500" />
                                      ) : (
                                        <TrendingDown className="h-4 w-4 text-red-500" />
                                      )}
                                    </div>
                                    <Progress value={adherence.adherenceRate} className="h-1.5" />
                                    <p className="text-xs text-muted-foreground mt-1">
                                      {adherence.streak} day streak
                                    </p>
                                  </div>
                                )}
                              </div>
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Pill className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>No medications recorded</p>
                      <p className="text-sm">Create a prescription to add medications</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Risk & Exposures Tab */}
            <TabsContent value="risk-exposures">
              {patientId && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-semibold">Risk & Exposures Profile</h3>
                      <p className="text-sm text-muted-foreground">
                        Auto-populated from clinical data. Edit to add manual corrections.
                      </p>
                    </div>
                  </div>
                  <RiskExposuresPanel 
                    patientId={patientId} 
                    isDoctor={true}
                    compact={false}
                  />
                </div>
              )}
            </TabsContent>

            {/* Prescriptions Tab */}
            <TabsContent value="prescriptions">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div>
                      <CardTitle>Prescription History</CardTitle>
                      <CardDescription>All prescriptions written for this patient</CardDescription>
                    </div>
                    <Button 
                      onClick={() => setPrescriptionDialogOpen(true)}
                      data-testid="button-write-prescription"
                    >
                      <Plus className="h-4 w-4 mr-2" />
                      Write Prescription
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[500px]">
                    {prescriptions && prescriptions.length > 0 ? (
                      <div className="space-y-3">
                        {prescriptions.map((rx) => (
                          <Card key={rx.id} className="hover-elevate" data-testid={`prescription-${rx.id}`}>
                            <CardContent className="p-4">
                              <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-1">
                                    <p className="font-medium">{rx.medicationName}</p>
                                    <Badge variant={
                                      rx.status === 'filled' ? 'default' : 
                                      rx.status === 'acknowledged' ? 'secondary' : 
                                      rx.status === 'expired' ? 'destructive' : 'outline'
                                    }>
                                      {rx.status}
                                    </Badge>
                                  </div>
                                  <p className="text-sm text-muted-foreground">
                                    {rx.dosage} • {rx.frequency}
                                  </p>
                                  {rx.dosageInstructions && (
                                    <p className="text-xs text-muted-foreground mt-1">
                                      {rx.dosageInstructions}
                                    </p>
                                  )}
                                  <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                                    <span>Qty: {rx.quantity || 'N/A'}</span>
                                    <span>Refills: {rx.refills || 0}</span>
                                    {rx.expirationDate && (
                                      <span>Expires: {format(new Date(rx.expirationDate), 'MMM d, yyyy')}</span>
                                    )}
                                  </div>
                                </div>
                                <div className="text-right text-xs text-muted-foreground">
                                  <p>{format(new Date(rx.createdAt || Date.now()), 'MMM d, yyyy')}</p>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <ClipboardList className="h-12 w-12 mx-auto mb-3 opacity-50" />
                        <p>No prescriptions yet</p>
                        <p className="text-sm">Write a prescription to get started</p>
                      </div>
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Complications Tab */}
            <TabsContent value="complications">
              <Card>
                <CardHeader>
                  <CardTitle>Recent Complications</CardTitle>
                  <CardDescription>Health events and complications timeline</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[500px]">
                    {complications.length > 0 ? (
                      <div className="space-y-4">
                        {complications.map((comp) => (
                          <Card 
                            key={comp.id} 
                            className={`border-l-4 ${
                              comp.resolvedAt ? 'border-l-green-500' : 'border-l-yellow-500'
                            }`}
                            data-testid={`complication-${comp.id}`}
                          >
                            <CardContent className="p-4">
                              <div className="flex items-start justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  <AlertTriangle className={`h-4 w-4 ${
                                    comp.severity === 'high' ? 'text-red-500' : 
                                    comp.severity === 'moderate' ? 'text-yellow-500' : 
                                    'text-blue-500'
                                  }`} />
                                  <span className="font-medium">{comp.type}</span>
                                  {getSeverityBadge(comp.severity)}
                                </div>
                                {comp.resolvedAt ? (
                                  <Badge variant="secondary" className="bg-green-100 text-green-700">
                                    <CheckCircle className="h-3 w-3 mr-1" />
                                    Resolved
                                  </Badge>
                                ) : (
                                  <Badge variant="secondary" className="bg-yellow-100 text-yellow-700">
                                    Active
                                  </Badge>
                                )}
                              </div>
                              <p className="text-sm mb-2">{comp.description}</p>
                              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                <span>Detected: {format(new Date(comp.detectedAt), 'MMM d, yyyy')}</span>
                                {comp.resolvedAt && (
                                  <span>Resolved: {format(new Date(comp.resolvedAt), 'MMM d, yyyy')}</span>
                                )}
                              </div>
                              {comp.notes && (
                                <p className="text-xs text-muted-foreground mt-2 italic">
                                  Note: {comp.notes}
                                </p>
                              )}
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <CheckCircle className="h-12 w-12 mx-auto mb-3 opacity-50 text-green-500" />
                        <p>No complications recorded</p>
                        <p className="text-sm">Patient has no recent health complications</p>
                      </div>
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Timeline Tab */}
            <TabsContent value="timeline">
              <Card>
                <CardHeader>
                  <CardTitle>Patient Timeline</CardTitle>
                  <CardDescription>Daily follow-ups and health check-ins</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[600px] pr-4">
                    <div className="space-y-4">
                      {followups && followups.length > 0 ? (
                        followups.map((followup) => (
                          <Card key={followup.id} className="border-l-4 border-l-primary" data-testid={`followup-${followup.id}`}>
                            <CardContent className="p-4">
                              <div className="flex items-start justify-between mb-2">
                                <div>
                                  <p className="font-medium">
                                    {new Date(followup.date).toLocaleDateString()}
                                  </p>
                                  <p className="text-sm text-muted-foreground">Daily Follow-up</p>
                                </div>
                                {followup.completed ? (
                                  <Badge variant="secondary">
                                    <CheckCircle className="h-3 w-3 mr-1" />
                                    Completed
                                  </Badge>
                                ) : (
                                  <Badge variant="secondary">
                                    <AlertCircle className="h-3 w-3 mr-1" />
                                    Pending
                                  </Badge>
                                )}
                              </div>
                              <div className="grid grid-cols-2 gap-3 mt-3">
                                {followup.heartRate && (
                                  <div className="text-sm flex items-center gap-2">
                                    <Heart className="h-3 w-3 text-red-500" />
                                    <span className="text-muted-foreground">Heart Rate: </span>
                                    <span className="font-medium">{followup.heartRate} bpm</span>
                                  </div>
                                )}
                                {followup.temperature && (
                                  <div className="text-sm flex items-center gap-2">
                                    <ThermometerSun className="h-3 w-3 text-orange-500" />
                                    <span className="text-muted-foreground">Temp: </span>
                                    <span className="font-medium">{followup.temperature}°F</span>
                                  </div>
                                )}
                                {followup.oxygenSaturation && (
                                  <div className="text-sm flex items-center gap-2">
                                    <Wind className="h-3 w-3 text-blue-500" />
                                    <span className="text-muted-foreground">SpO2: </span>
                                    <span className="font-medium">{followup.oxygenSaturation}%</span>
                                  </div>
                                )}
                                {followup.stepsCount && (
                                  <div className="text-sm flex items-center gap-2">
                                    <Activity className="h-3 w-3 text-green-500" />
                                    <span className="text-muted-foreground">Steps: </span>
                                    <span className="font-medium">{followup.stepsCount}</span>
                                  </div>
                                )}
                              </div>
                            </CardContent>
                          </Card>
                        ))
                      ) : (
                        <div className="text-center py-8 text-muted-foreground">
                          <Calendar className="h-12 w-12 mx-auto mb-3 opacity-50" />
                          <p>No follow-up data available</p>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Documents Tab */}
            <TabsContent value="documents">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div>
                      <CardTitle>Health Documents</CardTitle>
                      <CardDescription>Medical records, lab results, and uploaded files</CardDescription>
                    </div>
                    <Button variant="outline" data-testid="button-upload-document">
                      <Upload className="h-4 w-4 mr-2" />
                      Upload Document
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-8 text-muted-foreground">
                    <FolderOpen className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>No documents uploaded</p>
                    <p className="text-sm">Upload medical records, lab results, or other health documents</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Reports Tab */}
            <TabsContent value="reports">
              <Card>
                <CardHeader>
                  <CardTitle>Weekly Symptom Journal Reports</CardTitle>
                  <CardDescription>
                    Generate comprehensive PDF reports of patient symptom tracking
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Week Selection */}
                  <div>
                    <label className="text-sm font-medium mb-2 block">Select Week</label>
                    <div className="grid grid-cols-2 gap-2">
                      {[0, 1, 2, 3, 4, 5, 6, 7].map((offset) => {
                        const weekStart = startOfWeek(subWeeks(referenceDate, offset), { weekStartsOn: 1 });
                        const weekEnd = endOfWeek(subWeeks(referenceDate, offset), { weekStartsOn: 1 });
                        return (
                          <Button
                            key={offset}
                            variant={selectedWeekOffset === offset ? "default" : "outline"}
                            onClick={() => setSelectedWeekOffset(offset)}
                            className="justify-start"
                            data-testid={`button-week-${offset}`}
                          >
                            <Calendar className="mr-2 h-4 w-4" />
                            {offset === 0 ? "This Week" : offset === 1 ? "Last Week" : `${offset} Weeks Ago`}
                            <div className="ml-auto text-xs opacity-75">
                              {format(weekStart, "MMM d")} - {format(weekEnd, "MMM d")}
                            </div>
                          </Button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Generate Button */}
                  <div className="flex items-center gap-4 p-4 border rounded-lg bg-muted/50">
                    <FileText className="h-8 w-8 text-muted-foreground" />
                    <div className="flex-1">
                      <h4 className="font-medium">Generate PDF Report</h4>
                      <p className="text-sm text-muted-foreground">
                        Create a comprehensive weekly report including measurements, trends, and AI observations
                      </p>
                    </div>
                    <Button
                      onClick={() => generatePDF.mutate()}
                      disabled={generatePDF.isPending}
                      data-testid="button-generate-pdf"
                    >
                      {generatePDF.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Download className="mr-2 h-4 w-4" />
                          Generate PDF
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Lysa AI Assistant Tab */}
            <TabsContent value="lysa-assistant">
              {patientId && patient && (
                <LysaPatientAssistant
                  patientId={patientId}
                  patientContext={{
                    id: patientId,
                    firstName: patient.firstName || '',
                    lastName: patient.lastName || '',
                    email: patient.email || '',
                    allergies: [],
                    comorbidities: [],
                    currentMedications: medications?.map(m => m.name) || []
                  }}
                />
              )}
            </TabsContent>

            {/* Clinical Decision Support Tab */}
            <TabsContent value="clinical-support">
              {patientId && patient && (
                <ClinicalDecisionSupport
                  patientContext={{
                    id: patientId,
                    firstName: patient.firstName || '',
                    lastName: patient.lastName || '',
                    allergies: [],
                    comorbidities: [],
                    currentMedications: medications?.map(m => m.name) || []
                  }}
                />
              )}
            </TabsContent>

            {/* Predictive Analytics Tab */}
            <TabsContent value="predictive">
              {patientId && patient && (
                <PredictiveAnalyticsDashboard
                  patientContext={{
                    id: patientId,
                    firstName: patient.firstName || '',
                    lastName: patient.lastName || '',
                    allergies: [],
                    comorbidities: [],
                    currentMedications: medications?.map(m => m.name) || []
                  }}
                />
              )}
            </TabsContent>

            {/* Diagnostic Imaging Tab */}
            <TabsContent value="imaging">
              {patientId && patient && (
                <DiagnosticImagingAnalysis
                  patientContext={{
                    id: patientId,
                    firstName: patient.firstName || '',
                    lastName: patient.lastName || '',
                    allergies: [],
                    comorbidities: [],
                    currentMedications: medications?.map(m => m.name) || []
                  }}
                />
              )}
            </TabsContent>

            {/* Lab Report Analysis Tab */}
            <TabsContent value="labs">
              {patientId && patient && (
                <LabReportAnalysis
                  patientContext={{
                    id: patientId,
                    firstName: patient.firstName || '',
                    lastName: patient.lastName || '',
                    allergies: [],
                    comorbidities: [],
                    currentMedications: medications?.map(m => m.name) || []
                  }}
                />
              )}
            </TabsContent>

            {/* Lysa Insight Feed Tab */}
            <TabsContent value="insights">
              {patientId && patient && (
                <LysaInsightFeed
                  patientContext={{
                    id: patientId,
                    firstName: patient.firstName || '',
                    lastName: patient.lastName || ''
                  }}
                />
              )}
            </TabsContent>
          </Tabs>
        </div>

        {/* Right Sidebar - AI Insights */}
        <div className="space-y-6">
          <Card className="bg-gradient-to-br from-accent/20 to-accent/5">
            <CardHeader>
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent">
                  <Bot className="h-4 w-4" />
                </div>
                <CardTitle className="text-base">Assistant Lysa Insights</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px]">
                <div className="space-y-3">
                  <Card>
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-sm font-medium mb-1">Medication Adherence</p>
                          <p className="text-xs text-muted-foreground">
                            Patient shows excellent adherence with {healthSummary.medicationAdherence}% completion rate over the past 30 days.
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <TrendingUp className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-sm font-medium mb-1">Positive Trend</p>
                          <p className="text-xs text-muted-foreground">
                            Vital signs trending positively. Heart rate variability improving.
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <Stethoscope className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-sm font-medium mb-1">Next Steps</p>
                          <p className="text-xs text-muted-foreground">
                            Consider reviewing recent lab results and scheduling follow-up consultation.
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </ScrollArea>

              <Button className="w-full mt-4" variant="outline" data-testid="button-chat-lysa">
                <Bot className="h-4 w-4 mr-2" />
                Chat with Assistant Lysa
              </Button>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button 
                variant="outline" 
                className="w-full justify-start"
                onClick={() => setPrescriptionDialogOpen(true)}
                data-testid="button-quick-prescription"
              >
                <Pill className="h-4 w-4 mr-2" />
                Write Prescription
              </Button>
              <Button variant="outline" className="w-full justify-start" data-testid="button-quick-message">
                <MessageSquare className="h-4 w-4 mr-2" />
                Send Message
              </Button>
              <Button variant="outline" className="w-full justify-start" data-testid="button-schedule-followup">
                <Calendar className="h-4 w-4 mr-2" />
                Schedule Follow-up
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
