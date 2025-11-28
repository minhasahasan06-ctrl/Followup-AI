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
  List as ListIcon
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
import { format, parseISO, differenceInYears } from "date-fns";

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

interface PatientManagementPanelProps {
  onOpenLysa?: (patient: UserType) => void;
  className?: string;
}

export function PatientManagementPanel({ onOpenLysa, className }: PatientManagementPanelProps) {
  const [, setLocation] = useLocation();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<UserType | null>(null);
  const [lysaDialogOpen, setLysaDialogOpen] = useState(false);
  const [monitoringStates, setMonitoringStates] = useState<Record<string, boolean>>({});
  const [detailTab, setDetailTab] = useState("overview");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const { toast } = useToast();

  const { data: patients = [], isLoading: patientsLoading } = useQuery<UserType[]>({
    queryKey: ["/api/doctor/patients"],
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

  const calculateAge = (dateOfBirth: string): number => {
    return differenceInYears(new Date(), parseISO(dateOfBirth));
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
    return (
      <div className={`space-y-4 ${className}`}>
        <div className="flex items-center gap-4 mb-6">
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => {
              setSelectedPatient(null);
              setDetailTab("overview");
            }}
            data-testid="button-back-to-patients"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back to Patients
          </Button>
          <div className="flex-1" />
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
              <Avatar className="h-16 w-16">
                <AvatarFallback className="bg-primary text-primary-foreground text-xl">
                  {getInitials(selectedPatient.firstName, selectedPatient.lastName)}
                </AvatarFallback>
              </Avatar>
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
            </div>
          </CardContent>
        </Card>

        <Tabs value={detailTab} onValueChange={setDetailTab} className="w-full">
          <TabsList className="flex-wrap justify-start">
            <TabsTrigger value="overview" data-testid="tab-patient-overview">
              <User className="h-4 w-4 mr-2" />
              Overview
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

          <TabsContent value="clinical" className="mt-4">
            <ClinicalDecisionSupport 
              patientContext={{
                id: selectedPatient.id,
                firstName: selectedPatient.firstName || "",
                lastName: selectedPatient.lastName || "",
                allergies: patientProfile?.allergies,
                comorbidities: patientProfile?.medicalConditions,
              }}
            />
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

      {patientsLoading ? (
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
          {filteredPatients.map((patient) => (
            <Card
              key={patient.id}
              className="hover-elevate cursor-pointer"
              onClick={() => setSelectedPatient(patient)}
              data-testid={`card-patient-${patient.id}`}
            >
              <CardContent className={viewMode === "grid" ? "p-6" : "p-4"}>
                <div className={`flex items-${viewMode === "grid" ? "start" : "center"} gap-3 ${viewMode === "grid" ? "mb-4" : ""}`}>
                  <Avatar className="h-12 w-12">
                    <AvatarFallback className="bg-primary text-primary-foreground">
                      {getInitials(patient.firstName, patient.lastName)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold truncate" data-testid={`text-patient-name-${patient.id}`}>
                      {patient.firstName} {patient.lastName}
                    </h3>
                    <p className="text-sm text-muted-foreground truncate">{patient.email}</p>
                  </div>
                  {viewMode === "list" && (
                    <div className="flex items-center gap-2">
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
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Last Follow-up</span>
                        <span className="font-medium">Today</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Adherence</span>
                        <Badge variant="secondary" className="text-xs">
                          <TrendingUp className="h-3 w-3 mr-1" />
                          92%
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Alerts</span>
                        <Badge variant="secondary" className="text-xs">
                          <AlertCircle className="h-3 w-3 mr-1" />
                          0
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between text-sm pt-1 border-t">
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
          ))}
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
