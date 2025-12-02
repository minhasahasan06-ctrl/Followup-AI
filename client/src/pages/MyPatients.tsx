import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Users,
  Search,
  AlertTriangle,
  Pill,
  Activity,
  Calendar,
  MessageSquare,
  ChevronRight,
  Clock,
  TrendingUp,
  Shield,
  Heart,
  Brain,
  X,
  Circle
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { format, formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";

interface PatientSummary {
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

export default function MyPatients() {
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);

  const { data: patientsData, isLoading: patientsLoading } = useQuery<{ patients: PatientSummary[]; total: number }>({
    queryKey: ["/api/agent/patients"],
    enabled: user?.role === "doctor",
  });

  const { data: patientOverview, isLoading: overviewLoading } = useQuery<PatientOverview>({
    queryKey: ["/api/agent/patients", selectedPatientId, "overview"],
    enabled: !!selectedPatientId && detailDialogOpen,
  });

  const patients = patientsData?.patients || [];
  const filteredPatients = patients.filter(p => 
    p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.email?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handlePatientClick = (patientId: string) => {
    setSelectedPatientId(patientId);
    setDetailDialogOpen(true);
  };

  if (user?.role !== "doctor") {
    return (
      <div className="flex items-center justify-center h-full">
        <Card className="max-w-md">
          <CardContent className="pt-6 text-center">
            <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h2 className="text-lg font-semibold mb-2">Access Restricted</h2>
            <p className="text-muted-foreground">
              This page is only available to healthcare providers.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Users className="h-6 w-6" />
            My Patients
          </h1>
          <p className="text-muted-foreground">
            View and manage your assigned patients
          </p>
        </div>
        <Badge variant="outline" className="text-sm">
          {patients.length} Active Patients
        </Badge>
      </div>

      <div className="relative mb-6">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search patients by name or email..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
          data-testid="input-patient-search"
        />
      </div>

      {patientsLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="h-12 w-12 rounded-full bg-muted" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-muted rounded w-3/4" />
                    <div className="h-3 bg-muted rounded w-1/2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : filteredPatients.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <Users className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No Patients Found</h3>
            <p className="text-muted-foreground">
              {searchQuery 
                ? "No patients match your search criteria." 
                : "You don't have any assigned patients yet."}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredPatients.map((patient) => (
            <Card 
              key={patient.id} 
              className="cursor-pointer hover-elevate transition-all"
              onClick={() => handlePatientClick(patient.id)}
              data-testid={`card-patient-${patient.id}`}
            >
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <div className="relative">
                    <Avatar className="h-12 w-12">
                      <AvatarFallback className="bg-primary/10 text-primary">
                        {patient.name.split(" ").map(n => n[0]).join("").toUpperCase().slice(0, 2)}
                      </AvatarFallback>
                    </Avatar>
                    {patient.isOnline && (
                      <Circle className="absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 fill-chart-2 text-chart-2" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <h3 className="font-medium truncate">{patient.name}</h3>
                      <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                    </div>
                    <p className="text-sm text-muted-foreground truncate">
                      {patient.email}
                    </p>
                  </div>
                </div>

                <Separator className="my-3" />

                <div className="grid grid-cols-3 gap-2 text-center">
                  <div>
                    <div className={cn("text-lg font-semibold", getRiskColor(patient.riskScore))}>
                      {patient.riskScore?.toFixed(1) || "—"}
                    </div>
                    <div className="text-xs text-muted-foreground">Risk</div>
                  </div>
                  <div>
                    <div className="text-lg font-semibold flex items-center justify-center gap-1">
                      {patient.activeAlerts > 0 ? (
                        <span className="text-destructive">{patient.activeAlerts}</span>
                      ) : (
                        <span className="text-muted-foreground">0</span>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground">Alerts</div>
                  </div>
                  <div>
                    <div className="text-lg font-semibold">
                      {patient.medicationCount}
                    </div>
                    <div className="text-xs text-muted-foreground">Meds</div>
                  </div>
                </div>

                {patient.lastInteraction && (
                  <div className="mt-3 flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    Last: {formatDistanceToNow(new Date(patient.lastInteraction), { addSuffix: true })}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Dialog open={detailDialogOpen} onOpenChange={setDetailDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader className="flex-shrink-0">
            <div className="flex items-center justify-between">
              <DialogTitle className="flex items-center gap-2">
                {patientOverview?.patient.name || "Patient Details"}
              </DialogTitle>
            </div>
          </DialogHeader>

          {overviewLoading ? (
            <div className="flex-1 flex items-center justify-center py-12">
              <div className="text-center space-y-3">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto" />
                <p className="text-muted-foreground">Loading patient data...</p>
              </div>
            </div>
          ) : patientOverview ? (
            <ScrollArea className="flex-1 pr-4">
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="overview" data-testid="tab-patient-overview">Overview</TabsTrigger>
                  <TabsTrigger value="followups" data-testid="tab-patient-followups">Daily Followups</TabsTrigger>
                  <TabsTrigger value="conversations" data-testid="tab-patient-conversations">Conversations</TabsTrigger>
                  <TabsTrigger value="insights" data-testid="tab-patient-insights">AI Insights</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="mt-4 space-y-4">
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
                          <span>{patientOverview.patient.email || "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Phone</span>
                          <span>{patientOverview.patient.phone || "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Date of Birth</span>
                          <span>{patientOverview.patient.dateOfBirth 
                            ? format(new Date(patientOverview.patient.dateOfBirth), "MMM d, yyyy") 
                            : "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Member Since</span>
                          <span>{patientOverview.patient.memberSince 
                            ? format(new Date(patientOverview.patient.memberSince), "MMM yyyy") 
                            : "—"}</span>
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
                            {patientOverview.healthContext.risk_score?.composite.toFixed(1) || "0.0"}
                          </span>
                          <Badge variant={getRiskBadgeVariant(patientOverview.healthContext.risk_score?.composite)}>
                            {(patientOverview.healthContext.risk_score?.composite || 0) >= 10 ? "High Risk" :
                             (patientOverview.healthContext.risk_score?.composite || 0) >= 6 ? "Moderate" : "Low Risk"}
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Respiratory</span>
                            <span>{patientOverview.healthContext.risk_score?.respiratory.toFixed(1) || "0.0"}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Pain</span>
                            <span>{patientOverview.healthContext.risk_score?.pain.toFixed(1) || "0.0"}</span>
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
                                <span className="text-muted-foreground">{med.dosage} • {med.frequency}</span>
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
                                    {symptom.date ? format(new Date(symptom.date), "MMM d") : ""}
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
                </TabsContent>

                <TabsContent value="followups" className="mt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <Calendar className="h-4 w-4" />
                        Daily Followup History (Last 7 Days)
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {patientOverview.dailyFollowups.length > 0 ? (
                        <div className="space-y-3">
                          {patientOverview.dailyFollowups.map((followup, idx) => (
                            <div key={idx} className="flex items-start gap-4 p-3 border rounded-lg">
                              <div className="text-center min-w-[60px]">
                                <div className="text-lg font-bold">
                                  {followup.date ? format(new Date(followup.date), "d") : "—"}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  {followup.date ? format(new Date(followup.date), "MMM") : ""}
                                </div>
                              </div>
                              <Separator orientation="vertical" className="h-auto" />
                              <div className="flex-1 space-y-2">
                                <div className="flex items-center gap-2">
                                  <Badge variant={followup.overallStatus === "good" ? "default" : "secondary"}>
                                    {followup.overallStatus || "Not recorded"}
                                  </Badge>
                                  {followup.completedAt && (
                                    <span className="text-xs text-muted-foreground">
                                      Completed {format(new Date(followup.completedAt), "h:mm a")}
                                    </span>
                                  )}
                                </div>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                  <div>
                                    <span className="text-muted-foreground">Energy: </span>
                                    <span>{followup.energyLevel ?? "—"}/10</span>
                                  </div>
                                  <div>
                                    <span className="text-muted-foreground">Pain: </span>
                                    <span>{followup.painLevel ?? "—"}/10</span>
                                  </div>
                                </div>
                                {followup.symptomsNoted && (
                                  <p className="text-sm text-muted-foreground">{followup.symptomsNoted}</p>
                                )}
                                {followup.notes && (
                                  <p className="text-sm italic">{followup.notes}</p>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-center text-muted-foreground py-8">
                          No daily followups recorded in the past 7 days
                        </p>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="conversations" className="mt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <MessageSquare className="h-4 w-4" />
                        Lysa Conversation History
                      </CardTitle>
                      <CardDescription>
                        Your past conversations with Lysa about this patient
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {patientOverview.conversationHistory.length > 0 ? (
                        <div className="space-y-2">
                          {patientOverview.conversationHistory.map((conv) => (
                            <div 
                              key={conv.id} 
                              className="flex items-start gap-3 p-3 border rounded-lg hover-elevate cursor-pointer"
                            >
                              <MessageSquare className="h-5 w-5 text-muted-foreground flex-shrink-0 mt-0.5" />
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between gap-2">
                                  <h4 className="font-medium truncate">{conv.title}</h4>
                                  <Badge variant="outline" className="flex-shrink-0">
                                    {conv.messageCount} msgs
                                  </Badge>
                                </div>
                                {conv.lastMessage && (
                                  <p className="text-sm text-muted-foreground truncate mt-1">
                                    {conv.lastMessage}
                                  </p>
                                )}
                                <div className="text-xs text-muted-foreground mt-1">
                                  {conv.updatedAt 
                                    ? formatDistanceToNow(new Date(conv.updatedAt), { addSuffix: true })
                                    : "Unknown date"}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-center text-muted-foreground py-8">
                          No conversations with Lysa about this patient yet
                        </p>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="insights" className="mt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <Brain className="h-4 w-4 text-purple-500" />
                        AI-Powered Insights
                      </CardTitle>
                      <CardDescription>
                        Long-term patterns and important observations from AI analysis
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {patientOverview.longTermInsights.length > 0 ? (
                        <div className="space-y-3">
                          {patientOverview.longTermInsights.map((insight, idx) => (
                            <div key={idx} className="p-3 bg-muted/50 rounded-lg">
                              <div className="flex items-start justify-between gap-2 mb-2">
                                <Badge variant="outline" className="text-xs">
                                  {insight.type}
                                </Badge>
                                <div className="flex items-center gap-1">
                                  <span className="text-xs text-muted-foreground">
                                    Importance: {(insight.importance * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </div>
                              <p className="text-sm">{insight.content}</p>
                              {insight.createdAt && (
                                <p className="text-xs text-muted-foreground mt-2">
                                  {format(new Date(insight.createdAt), "MMM d, yyyy")}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-center text-muted-foreground py-8">
                          No AI insights available for this patient yet
                        </p>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </ScrollArea>
          ) : (
            <div className="flex-1 flex items-center justify-center py-12">
              <p className="text-muted-foreground">Failed to load patient details</p>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
