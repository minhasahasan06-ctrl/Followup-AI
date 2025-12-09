import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Users, Search, TrendingUp, AlertCircle, Sparkles, Lightbulb, BookOpen, Beaker, Clock, CheckCircle, Bot, Eye, EyeOff, Bell, UserCheck, XCircle, Edit, ExternalLink, Activity, Pill, Brain, Loader2, History } from "lucide-react";
import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { User } from "@shared/schema";
import { AddPatientDialog } from "@/components/AddPatientDialog";
import { LysaChatPanel } from "@/components/LysaChatPanel";

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

interface PatientSummary {
  id: string;
  name: string;
  email?: string;
  risk_score?: number;
  risk_state?: string;
  last_checkin?: string;
  active_medications_count?: number;
}

interface PendingApproval {
  id: string;
  patient_id: string;
  patient: PatientSummary | null;
  action_type: string;
  status: string;
  priority: string;
  title: string;
  ai_recommendation: string;
  ai_reasoning?: string;
  confidence_score: number;
  risk_score?: number;
  risk_state?: string;
  patient_context?: Record<string, any>;
  expires_at?: string;
  created_at?: string;
}

interface ApprovalCount {
  count: number;
  critical: number;
  high: number;
}

interface PatientProfileSummary {
  patient: {
    id: string;
    name: string;
    email?: string;
    date_of_birth?: string;
    phone?: string;
    member_since?: string;
  };
  autopilot_state: {
    risk_score?: number;
    risk_state?: string;
    risk_components?: Record<string, number>;
    last_checkin?: string;
    next_followup?: string;
  };
  medications: Array<{
    id: string;
    name: string;
    dosage?: string;
    frequency?: string;
    status?: string;
  }>;
  recent_symptoms: Array<{
    type: string;
    severity?: number;
    date?: string;
  }>;
  mental_health: Array<{
    type: string;
    score?: number;
    severity?: string;
    date?: string;
  }>;
  adherence_rate_7d?: number;
}

export default function DoctorDashboard() {
  const [, setLocation] = useLocation();
  const [searchQuery, setSearchQuery] = useState("");
  const [lysaDialogOpen, setLysaDialogOpen] = useState(false);
  const [selectedPatientForLysa, setSelectedPatientForLysa] = useState<User | null>(null);
  const [monitoringStates, setMonitoringStates] = useState<Record<string, boolean>>({});
  const [approvalDialogOpen, setApprovalDialogOpen] = useState(false);
  const [selectedApproval, setSelectedApproval] = useState<PendingApproval | null>(null);
  const [decisionNotes, setDecisionNotes] = useState("");
  const [rejectionReason, setRejectionReason] = useState("");
  const [patientProfileOpen, setPatientProfileOpen] = useState(false);
  const [selectedPatientProfile, setSelectedPatientProfile] = useState<PatientProfileSummary | null>(null);
  const [showApprovalHistory, setShowApprovalHistory] = useState(false);
  const { toast } = useToast();

  const { data: patients, isLoading } = useQuery<User[]>({
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
        body: JSON.stringify({ patientId, enabled })
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

  const openLysaForPatient = (patient: User, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedPatientForLysa(patient);
    setLysaDialogOpen(true);
  };

  // Fetch Assistant Lysa recommendations for doctors
  const { data: recommendations = [], isLoading: recommendationsLoading, isError: recommendationsError } = useQuery<Recommendation[]>({
    queryKey: ['/api/v1/ml/recommendations', { agentType: 'lysa', limit: 6 }],
    queryFn: async () => {
      const res = await fetch('/api/v1/ml/recommendations?agentType=lysa&limit=6');
      if (!res.ok) throw new Error('Failed to fetch recommendations');
      return res.json();
    },
  });

  const clinicalRecommendations = recommendations.filter(r => 
    r.category === 'clinical_decision_support' || r.category === 'protocol_suggestion'
  );
  
  const researchRecommendations = recommendations.filter(r => 
    r.category === 'research' || r.category === 'literature_review'
  );

  // HITL Pending Approvals
  const { data: pendingApprovalsData, isLoading: approvalsLoading } = useQuery<{ approvals: PendingApproval[]; count: number }>({
    queryKey: ['/api/v1/hitl/pending'],
    queryFn: async () => {
      try {
        const res = await fetch('/api/v1/hitl/pending');
        if (!res.ok) return { approvals: [], count: 0 };
        return res.json();
      } catch {
        return { approvals: [], count: 0 };
      }
    },
    refetchInterval: 30000,
  });

  const { data: approvalCount } = useQuery<ApprovalCount>({
    queryKey: ['/api/v1/hitl/pending/count'],
    queryFn: async () => {
      try {
        const res = await fetch('/api/v1/hitl/pending/count');
        if (!res.ok) return { count: 0, critical: 0, high: 0 };
        return res.json();
      } catch {
        return { count: 0, critical: 0, high: 0 };
      }
    },
    refetchInterval: 15000,
  });

  const { data: approvalHistoryData } = useQuery<{ history: any[]; count: number }>({
    queryKey: ['/api/v1/hitl/history'],
    queryFn: async () => {
      try {
        const res = await fetch('/api/v1/hitl/history?days=30&limit=20');
        if (!res.ok) return { history: [], count: 0 };
        return res.json();
      } catch {
        return { history: [], count: 0 };
      }
    },
    enabled: showApprovalHistory,
  });

  const decisionMutation = useMutation({
    mutationFn: async ({ approvalId, action, notes, rejectionReason: reason }: { 
      approvalId: string; 
      action: 'approve' | 'modify' | 'reject';
      notes?: string;
      rejectionReason?: string;
    }) => {
      const body: any = { action, notes };
      if (action === 'reject') body.rejection_reason = reason;
      
      const res = await fetch(`/api/v1/hitl/approvals/${approvalId}/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error('Failed to submit decision');
      return res.json();
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/hitl/pending'] });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/hitl/pending/count'] });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/hitl/history'] });
      setApprovalDialogOpen(false);
      setSelectedApproval(null);
      setDecisionNotes("");
      setRejectionReason("");
      toast({
        title: `Recommendation ${variables.action === 'approve' ? 'Approved' : variables.action === 'reject' ? 'Rejected' : 'Modified'}`,
        description: "Your decision has been recorded and action will be executed.",
      });
    },
    onError: () => {
      toast({
        title: "Decision Failed",
        description: "Failed to submit your decision. Please try again.",
        variant: "destructive",
      });
    },
  });

  const fetchPatientProfileMutation = useMutation({
    mutationFn: async (patientId: string) => {
      const res = await fetch(`/api/v1/hitl/patients/${patientId}/profile-summary`);
      if (!res.ok) throw new Error('Failed to fetch patient profile');
      return res.json();
    },
    onSuccess: (data) => {
      setSelectedPatientProfile(data);
      setPatientProfileOpen(true);
    },
    onError: () => {
      toast({
        title: "Profile Load Failed",
        description: "Could not load patient profile. Please try again.",
        variant: "destructive",
      });
    },
  });

  const openApprovalDialog = (approval: PendingApproval) => {
    setSelectedApproval(approval);
    setApprovalDialogOpen(true);
    setDecisionNotes("");
    setRejectionReason("");
  };

  const handleDecision = (action: 'approve' | 'modify' | 'reject') => {
    if (!selectedApproval) return;
    decisionMutation.mutate({
      approvalId: selectedApproval.id,
      action,
      notes: decisionNotes,
      rejectionReason: action === 'reject' ? rejectionReason : undefined,
    });
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-500 text-white';
      case 'high': return 'bg-orange-500 text-white';
      case 'medium': return 'bg-yellow-500 text-black';
      default: return 'bg-gray-500 text-white';
    }
  };

  const getRiskStateColor = (riskState?: string) => {
    switch (riskState) {
      case 'Critical': return 'text-red-600';
      case 'Worsening': return 'text-orange-600';
      case 'AtRisk': return 'text-yellow-600';
      default: return 'text-green-600';
    }
  };

  const filteredPatients = patients?.filter((patient) =>
    `${patient.firstName} ${patient.lastName}`.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getInitials = (firstName?: string | null, lastName?: string | null) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-semibold mb-2" data-testid="text-dashboard-title">Doctor Dashboard</h1>
          <p className="text-muted-foreground">Clinical insights and patient overview powered by Assistant Lysa</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="secondary" className="text-sm">
            <Users className="h-3 w-3 mr-1" />
            {patients?.length || 0} Patients
          </Badge>
          <AddPatientDialog />
        </div>
      </div>

      {/* Assistant Lysa Recommendations Section */}
      {recommendationsLoading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {[1, 2].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-5 bg-muted rounded w-1/2 mb-2" />
                <div className="h-4 bg-muted rounded w-3/4" />
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[1, 2, 3].map((j) => (
                    <div key={j} className="h-20 bg-muted rounded" />
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : recommendationsError ? (
        <Card>
          <CardContent className="py-8 text-center">
            <AlertCircle className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">
              Unable to load recommendations. Please try again later.
            </p>
          </CardContent>
        </Card>
      ) : (clinicalRecommendations.length > 0 || researchRecommendations.length > 0) && (
        <div className="grid gap-4 md:grid-cols-2">
          {/* Clinical Insights */}
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

          {/* Research Recommendations */}
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

      {/* Pending Approvals Section - HITL */}
      <Card data-testid="card-pending-approvals">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bell className={`w-5 h-5 ${(approvalCount?.critical || 0) > 0 ? 'text-red-500' : 'text-primary'}`} />
              <CardTitle className="text-lg">Pending Approvals</CardTitle>
              {(approvalCount?.count || 0) > 0 && (
                <Badge variant="default" className={getPriorityColor(approvalCount?.critical ? 'critical' : approvalCount?.high ? 'high' : 'medium')}>
                  {approvalCount?.count}
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowApprovalHistory(!showApprovalHistory)}
                data-testid="button-toggle-approval-history"
              >
                <History className="w-4 h-4 mr-1" />
                {showApprovalHistory ? 'Hide History' : 'View History'}
              </Button>
            </div>
          </div>
          <CardDescription>
            AI wellness recommendations requiring your review. Actions will only execute after your approval.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {approvalsLoading ? (
            <div className="space-y-3">
              {[1, 2].map((i) => (
                <div key={i} className="h-24 bg-muted rounded animate-pulse" />
              ))}
            </div>
          ) : (pendingApprovalsData?.approvals?.length || 0) === 0 ? (
            <div className="py-8 text-center">
              <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">No pending approvals</p>
              <p className="text-xs text-muted-foreground mt-1">All AI recommendations have been reviewed</p>
            </div>
          ) : (
            <ScrollArea className="max-h-[400px]">
              <div className="space-y-3">
                {pendingApprovalsData?.approvals?.map((approval) => (
                  <div
                    key={approval.id}
                    className="p-4 border rounded-lg hover-elevate cursor-pointer"
                    onClick={() => openApprovalDialog(approval)}
                    data-testid={`approval-item-${approval.id}`}
                  >
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap mb-1">
                          <h4 className="font-medium text-sm">{approval.title}</h4>
                          <Badge className={`text-xs ${getPriorityColor(approval.priority)}`}>
                            {approval.priority}
                          </Badge>
                          {approval.risk_state && (
                            <span className={`text-xs font-medium ${getRiskStateColor(approval.risk_state)}`}>
                              {approval.risk_state}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground line-clamp-2">{approval.ai_recommendation}</p>
                      </div>
                      <div className="text-right shrink-0">
                        {approval.patient?.name && (
                          <p className="text-sm font-medium">{approval.patient.name}</p>
                        )}
                        <p className="text-xs text-muted-foreground">
                          {Math.round((approval.confidence_score || 0) * 100)}% confidence
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-3">
                        <span className="text-muted-foreground">
                          Action: {approval.action_type?.replace(/_/g, ' ')}
                        </span>
                        {approval.risk_score !== undefined && (
                          <span className="text-muted-foreground">
                            Risk Score: {Math.round(approval.risk_score)}
                          </span>
                        )}
                      </div>
                      {approval.expires_at && (
                        <span className="text-muted-foreground flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          Expires: {new Date(approval.expires_at).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>

        {/* Approval History */}
        {showApprovalHistory && (
          <CardFooter className="flex-col items-stretch border-t pt-4">
            <h4 className="font-medium text-sm mb-3">Recent Decisions (30 days)</h4>
            {!approvalHistoryData?.history?.length ? (
              <p className="text-xs text-muted-foreground text-center py-4">No decision history</p>
            ) : (
              <ScrollArea className="max-h-[200px]">
                <div className="space-y-2">
                  {approvalHistoryData.history.map((item: any) => (
                    <div key={item.id} className="flex items-center justify-between p-2 text-xs border rounded">
                      <div className="flex items-center gap-2">
                        {item.decision === 'approved' ? (
                          <CheckCircle className="w-3 h-3 text-green-500" />
                        ) : item.decision === 'rejected' ? (
                          <XCircle className="w-3 h-3 text-red-500" />
                        ) : (
                          <Edit className="w-3 h-3 text-blue-500" />
                        )}
                        <span>{item.approval?.title || 'Unknown'}</span>
                      </div>
                      <span className="text-muted-foreground">
                        {new Date(item.decided_at).toLocaleDateString()}
                      </span>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </CardFooter>
        )}
      </Card>

      {/* Patient List Section */}
      <div className="flex items-center gap-2">
        <Users className="w-5 h-5" />
        <h2 className="text-2xl font-semibold">All Patients</h2>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search patients by name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
                data-testid="input-search-patients"
              />
            </div>
          </div>
        </CardHeader>
      </Card>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-6">
                <div className="flex items-center gap-3">
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
      ) : filteredPatients && filteredPatients.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredPatients.map((patient) => (
            <Card
              key={patient.id}
              className="hover-elevate cursor-pointer"
              onClick={() => setLocation(`/doctor/patient/${patient.id}`)}
              data-testid={`card-patient-${patient.id}`}
            >
              <CardContent className="p-6">
                <div className="flex items-start gap-3 mb-4">
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
                </div>

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
                    View Patient
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

      {/* Lysa AI Assistant Dialog */}
      <Dialog open={lysaDialogOpen} onOpenChange={setLysaDialogOpen}>
        <DialogContent className="max-w-4xl h-[80vh]" data-testid="dialog-lysa-assistant">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-primary" />
              Lysa AI Assistant
              {selectedPatientForLysa && (
                <Badge variant="secondary" className="ml-2">
                  {selectedPatientForLysa.firstName} {selectedPatientForLysa.lastName}
                </Badge>
              )}
            </DialogTitle>
            <DialogDescription>
              AI-powered clinical assistant for patient care and decision support
            </DialogDescription>
          </DialogHeader>
          <div className="flex-1 overflow-hidden -mx-6 -mb-6">
            {selectedPatientForLysa && (
              <LysaChatPanel 
                patientId={selectedPatientForLysa.id}
                patientName={`${selectedPatientForLysa.firstName} ${selectedPatientForLysa.lastName}`}
                className="h-full border-0 rounded-none"
              />
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Approval Detail Dialog */}
      <Dialog open={approvalDialogOpen} onOpenChange={setApprovalDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col" data-testid="dialog-approval-detail">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5 text-primary" />
              Review Recommendation
              {selectedApproval?.priority && (
                <Badge className={`ml-2 ${getPriorityColor(selectedApproval.priority)}`}>
                  {selectedApproval.priority}
                </Badge>
              )}
            </DialogTitle>
            <DialogDescription>
              Review AI recommendation and decide whether to approve, modify, or reject.
            </DialogDescription>
          </DialogHeader>

          {selectedApproval && (
            <ScrollArea className="flex-1 pr-4">
              <div className="space-y-4">
                {/* Patient Info */}
                <div className="p-3 border rounded-lg bg-muted/30">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Avatar className="h-8 w-8">
                        <AvatarFallback className="text-xs">
                          {selectedApproval.patient?.name?.split(' ').map(n => n[0]).join('') || '?'}
                        </AvatarFallback>
                      </Avatar>
                      <div>
                        <p className="font-medium text-sm">{selectedApproval.patient?.name || 'Unknown Patient'}</p>
                        {selectedApproval.patient?.email && (
                          <p className="text-xs text-muted-foreground">{selectedApproval.patient.email}</p>
                        )}
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        fetchPatientProfileMutation.mutate(selectedApproval.patient_id);
                      }}
                      disabled={fetchPatientProfileMutation.isPending}
                      data-testid="button-view-patient-profile"
                    >
                      {fetchPatientProfileMutation.isPending ? (
                        <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                      ) : (
                        <ExternalLink className="w-4 h-4 mr-1" />
                      )}
                      View Profile
                    </Button>
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    {selectedApproval.risk_state && (
                      <span className={`font-medium ${getRiskStateColor(selectedApproval.risk_state)}`}>
                        Status: {selectedApproval.risk_state}
                      </span>
                    )}
                    {selectedApproval.risk_score !== undefined && (
                      <span className="text-muted-foreground">
                        Risk Score: {Math.round(selectedApproval.risk_score)}/100
                      </span>
                    )}
                  </div>
                </div>

                {/* Recommendation Details */}
                <div>
                  <h4 className="font-medium text-sm mb-2">{selectedApproval.title}</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    <strong>Recommended Action:</strong> {selectedApproval.action_type?.replace(/_/g, ' ')}
                  </p>
                </div>

                {/* AI Recommendation */}
                <div className="p-3 border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="w-4 h-4 text-primary" />
                    <h5 className="font-medium text-sm">AI Recommendation</h5>
                    <Badge variant="secondary" className="text-xs">
                      {Math.round((selectedApproval.confidence_score || 0) * 100)}% confidence
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{selectedApproval.ai_recommendation}</p>
                </div>

                {/* AI Reasoning */}
                {selectedApproval.ai_reasoning && (
                  <div className="p-3 border rounded-lg bg-muted/30">
                    <h5 className="font-medium text-sm mb-2">AI Reasoning</h5>
                    <p className="text-xs text-muted-foreground">{selectedApproval.ai_reasoning}</p>
                  </div>
                )}

                {/* Context Info */}
                {selectedApproval.patient_context && Object.keys(selectedApproval.patient_context).length > 0 && (
                  <div className="p-3 border rounded-lg">
                    <h5 className="font-medium text-sm mb-2">Patient Context</h5>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {Object.entries(selectedApproval.patient_context).map(([key, value]) => (
                        value !== null && value !== undefined && (
                          <div key={key} className="flex justify-between">
                            <span className="text-muted-foreground">{key.replace(/_/g, ' ')}:</span>
                            <span className="font-medium">
                              {typeof value === 'number' ? 
                                (key.includes('adherence') ? `${Math.round(value * 100)}%` : Math.round(value * 100) / 100) 
                                : String(value)}
                            </span>
                          </div>
                        )
                      ))}
                    </div>
                  </div>
                )}

                {/* Decision Notes */}
                <div>
                  <label className="text-sm font-medium mb-2 block">Decision Notes (optional)</label>
                  <Textarea
                    placeholder="Add notes about your decision..."
                    value={decisionNotes}
                    onChange={(e) => setDecisionNotes(e.target.value)}
                    className="min-h-[80px]"
                    data-testid="input-decision-notes"
                  />
                </div>

                {/* Rejection Reason */}
                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Rejection Reason (required if rejecting)
                  </label>
                  <Textarea
                    placeholder="Explain why you're rejecting this recommendation..."
                    value={rejectionReason}
                    onChange={(e) => setRejectionReason(e.target.value)}
                    className="min-h-[60px]"
                    data-testid="input-rejection-reason"
                  />
                </div>

                {/* Wellness Disclaimer */}
                <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded text-xs text-yellow-800 dark:text-yellow-200">
                  Wellness monitoring only - Not medical advice. All recommendations are for discussion and require your clinical judgment.
                </div>
              </div>
            </ScrollArea>
          )}

          <DialogFooter className="gap-2 sm:gap-2 flex-wrap border-t pt-4">
            <Button
              variant="outline"
              onClick={() => setApprovalDialogOpen(false)}
              disabled={decisionMutation.isPending}
              data-testid="button-cancel-approval"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => handleDecision('reject')}
              disabled={decisionMutation.isPending || !rejectionReason.trim()}
              data-testid="button-reject-approval"
            >
              {decisionMutation.isPending ? (
                <Loader2 className="w-4 h-4 mr-1 animate-spin" />
              ) : (
                <XCircle className="w-4 h-4 mr-1" />
              )}
              Reject
            </Button>
            <Button
              variant="secondary"
              onClick={() => handleDecision('modify')}
              disabled={decisionMutation.isPending}
              data-testid="button-modify-approval"
            >
              {decisionMutation.isPending ? (
                <Loader2 className="w-4 h-4 mr-1 animate-spin" />
              ) : (
                <Edit className="w-4 h-4 mr-1" />
              )}
              Modify
            </Button>
            <Button
              onClick={() => handleDecision('approve')}
              disabled={decisionMutation.isPending}
              data-testid="button-approve-approval"
            >
              {decisionMutation.isPending ? (
                <Loader2 className="w-4 h-4 mr-1 animate-spin" />
              ) : (
                <CheckCircle className="w-4 h-4 mr-1" />
              )}
              Approve
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Patient Profile Summary Dialog */}
      <Dialog open={patientProfileOpen} onOpenChange={setPatientProfileOpen}>
        <DialogContent className="max-w-xl max-h-[80vh] overflow-hidden flex flex-col" data-testid="dialog-patient-profile">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <UserCheck className="h-5 w-5 text-primary" />
              Patient Profile Summary
            </DialogTitle>
            <DialogDescription>
              Overview of patient condition to inform your decision
            </DialogDescription>
          </DialogHeader>

          {selectedPatientProfile && (
            <ScrollArea className="flex-1 pr-4">
              <div className="space-y-4">
                {/* Patient Info */}
                <div className="flex items-center gap-3 p-3 border rounded-lg">
                  <Avatar className="h-12 w-12">
                    <AvatarFallback>
                      {selectedPatientProfile.patient?.name?.split(' ').map(n => n[0]).join('') || '?'}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <h4 className="font-semibold">{selectedPatientProfile.patient?.name}</h4>
                    {selectedPatientProfile.patient?.email && (
                      <p className="text-sm text-muted-foreground">{selectedPatientProfile.patient.email}</p>
                    )}
                    {selectedPatientProfile.patient?.member_since && (
                      <p className="text-xs text-muted-foreground">
                        Member since {new Date(selectedPatientProfile.patient.member_since).toLocaleDateString()}
                      </p>
                    )}
                  </div>
                </div>

                {/* Autopilot State */}
                {selectedPatientProfile.autopilot_state && (
                  <div className="p-3 border rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-4 h-4 text-primary" />
                      <h5 className="font-medium text-sm">Wellness State</h5>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-muted-foreground">Status:</span>{' '}
                        <span className={`font-medium ${getRiskStateColor(selectedPatientProfile.autopilot_state.risk_state)}`}>
                          {selectedPatientProfile.autopilot_state.risk_state || 'Stable'}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Risk Score:</span>{' '}
                        <span className="font-medium">
                          {Math.round(selectedPatientProfile.autopilot_state.risk_score || 0)}/100
                        </span>
                      </div>
                      {selectedPatientProfile.autopilot_state.last_checkin && (
                        <div className="col-span-2">
                          <span className="text-muted-foreground">Last Check-in:</span>{' '}
                          <span className="font-medium">
                            {new Date(selectedPatientProfile.autopilot_state.last_checkin).toLocaleDateString()}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Medications */}
                {selectedPatientProfile.medications && selectedPatientProfile.medications.length > 0 && (
                  <div className="p-3 border rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Pill className="w-4 h-4 text-primary" />
                      <h5 className="font-medium text-sm">Active Medications</h5>
                      <Badge variant="secondary" className="text-xs">
                        {selectedPatientProfile.medications.length}
                      </Badge>
                    </div>
                    <div className="space-y-1">
                      {selectedPatientProfile.medications.slice(0, 5).map((med) => (
                        <div key={med.id} className="flex justify-between text-xs">
                          <span className="font-medium">{med.name}</span>
                          <span className="text-muted-foreground">{med.dosage} - {med.frequency}</span>
                        </div>
                      ))}
                      {selectedPatientProfile.medications.length > 5 && (
                        <p className="text-xs text-muted-foreground">
                          +{selectedPatientProfile.medications.length - 5} more
                        </p>
                      )}
                    </div>
                    {selectedPatientProfile.adherence_rate_7d !== undefined && (
                      <div className="mt-2 pt-2 border-t">
                        <span className="text-xs text-muted-foreground">7-day Adherence:</span>{' '}
                        <span className="text-xs font-medium">
                          {Math.round(selectedPatientProfile.adherence_rate_7d * 100)}%
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Mental Health */}
                {selectedPatientProfile.mental_health && selectedPatientProfile.mental_health.length > 0 && (
                  <div className="p-3 border rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Brain className="w-4 h-4 text-primary" />
                      <h5 className="font-medium text-sm">Mental Health Assessments</h5>
                    </div>
                    <div className="space-y-1">
                      {selectedPatientProfile.mental_health.map((mh, idx) => (
                        <div key={idx} className="flex justify-between text-xs">
                          <span className="font-medium">{mh.type}</span>
                          <span className={`${mh.severity === 'severe' ? 'text-red-500' : mh.severity === 'moderate' ? 'text-yellow-500' : 'text-green-500'}`}>
                            {mh.severity || 'Normal'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recent Symptoms */}
                {selectedPatientProfile.recent_symptoms && selectedPatientProfile.recent_symptoms.length > 0 && (
                  <div className="p-3 border rounded-lg">
                    <h5 className="font-medium text-sm mb-2">Recent Symptoms</h5>
                    <div className="flex flex-wrap gap-1">
                      {selectedPatientProfile.recent_symptoms.slice(0, 6).map((symptom, idx) => (
                        <Badge key={idx} variant="secondary" className="text-xs">
                          {symptom.type}
                          {symptom.severity !== undefined && ` (${symptom.severity}/10)`}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          )}

          <DialogFooter className="border-t pt-4">
            <Button
              variant="outline"
              onClick={() => {
                setPatientProfileOpen(false);
                if (selectedApproval?.patient_id) {
                  setLocation(`/doctor/patient/${selectedApproval.patient_id}`);
                }
              }}
              data-testid="button-view-full-profile"
            >
              <ExternalLink className="w-4 h-4 mr-1" />
              View Full Profile
            </Button>
            <Button onClick={() => setPatientProfileOpen(false)} data-testid="button-close-profile">
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
