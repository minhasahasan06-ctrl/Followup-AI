import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { DrugInteractionAlert } from "@/components/DrugInteractionAlert";
import { 
  AlertTriangle, 
  Pill, 
  Search, 
  Loader2, 
  CheckCircle2, 
  AlertCircle, 
  User, 
  Shield, 
  Clock, 
  FileText, 
  XCircle,
  RefreshCw,
  Activity
} from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { format } from "date-fns";
import type { User as UserType, DosageChangeRequest } from "@shared/schema";

export default function DrugInteractions() {
  const { toast } = useToast();
  const { user } = useAuth();
  const isDoctor = user?.role === "doctor";
  
  const [drugToCheck, setDrugToCheck] = useState("");
  const [checking, setChecking] = useState(false);
  const [checkResults, setCheckResults] = useState<any>(null);
  const [selectedPatient, setSelectedPatient] = useState("");
  const [showOverrideDialog, setShowOverrideDialog] = useState(false);
  const [selectedAlertForOverride, setSelectedAlertForOverride] = useState<any>(null);
  const [overrideReason, setOverrideReason] = useState("");
  const [activeTab, setActiveTab] = useState("active");

  const { data: patients } = useQuery<UserType[]>({
    queryKey: ["/api/doctor/patients"],
    enabled: isDoctor,
  });

  const { data: pendingRequests, isLoading: requestsLoading } = useQuery<DosageChangeRequest[]>({
    queryKey: ["/api/dosage-change-requests/pending"],
    enabled: isDoctor,
  });

  const { data: activeAlerts, isLoading: alertsLoading } = useQuery<any[]>({
    queryKey: ['/api/drug-interactions/alerts'],
  });

  const { data: allAlerts } = useQuery<any[]>({
    queryKey: ['/api/drug-interactions/alerts/all'],
  });

  const { data: patientMedications } = useQuery<any[]>({
    queryKey: ["/api/medications", selectedPatient],
    queryFn: async () => {
      if (!selectedPatient) return [];
      const res = await fetch(`/api/medications?patientId=${selectedPatient}`);
      if (!res.ok) return [];
      return res.json();
    },
    enabled: isDoctor && !!selectedPatient,
  });

  const acknowledgeMutation = useMutation({
    mutationFn: async (alertId: string) => {
      return apiRequest('POST', `/api/drug-interactions/alerts/${alertId}/acknowledge`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/drug-interactions/alerts'] });
      queryClient.invalidateQueries({ queryKey: ['/api/drug-interactions/alerts/all'] });
      toast({
        title: "Alert acknowledged",
        description: "You've acknowledged this drug interaction warning.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to acknowledge alert. Please try again.",
        variant: "destructive",
      });
    },
  });

  const overrideMutation = useMutation({
    mutationFn: async ({ alertId, reason }: { alertId: string; reason: string }) => {
      return apiRequest('POST', `/api/drug-interactions/alerts/${alertId}/override`, { reason });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/drug-interactions/alerts'] });
      queryClient.invalidateQueries({ queryKey: ['/api/drug-interactions/alerts/all'] });
      setShowOverrideDialog(false);
      setOverrideReason("");
      setSelectedAlertForOverride(null);
      toast({
        title: "Override Applied",
        description: "The interaction alert has been overridden with your clinical justification.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to override alert. Please try again.",
        variant: "destructive",
      });
    },
  });

  const approveDosageMutation = useMutation({
    mutationFn: async ({ requestId, notes }: { requestId: string; notes: string }) => {
      return apiRequest("POST", `/api/dosage-change-requests/${requestId}/approve`, { notes });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/dosage-change-requests/pending"] });
      toast({
        title: "Request Approved",
        description: "The dosage change has been applied to the patient's medication.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to approve request.",
        variant: "destructive",
      });
    },
  });

  const rejectDosageMutation = useMutation({
    mutationFn: async ({ requestId, notes }: { requestId: string; notes: string }) => {
      return apiRequest("POST", `/api/dosage-change-requests/${requestId}/reject`, { notes });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/dosage-change-requests/pending"] });
      toast({
        title: "Request Rejected",
        description: "The patient has been notified of the rejection.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to reject request.",
        variant: "destructive",
      });
    },
  });

  const handleCheckDrug = async () => {
    if (!drugToCheck.trim()) {
      toast({
        title: "Drug name required",
        description: "Please enter a medication name to check for interactions.",
        variant: "destructive",
      });
      return;
    }

    setChecking(true);
    try {
      const payload: any = { drugName: drugToCheck };
      if (isDoctor && selectedPatient) {
        payload.patientId = selectedPatient;
      }
      const result: any = await apiRequest('POST', '/api/drug-interactions/analyze', payload);
      setCheckResults(result);
      
      if (result.hasBlockingInteraction) {
        toast({
          title: "Severe interaction detected",
          description: "This medication has a severe interaction with current medications.",
          variant: "destructive",
        });
      } else if (result.hasInteractions) {
        toast({
          title: "Interactions found",
          description: `Found ${result.interactions.length} potential interaction(s).`,
        });
      } else {
        toast({
          title: "No interactions found",
          description: "This medication appears safe with current medications.",
        });
      }
    } catch (error) {
      toast({
        title: "Analysis failed",
        description: "Failed to analyze drug interactions. Please try again.",
        variant: "destructive",
      });
    } finally {
      setChecking(false);
    }
  };

  const handleOverride = (alert: any) => {
    setSelectedAlertForOverride(alert);
    setShowOverrideDialog(true);
  };

  const submitOverride = () => {
    if (!selectedAlertForOverride || !overrideReason.trim()) return;
    overrideMutation.mutate({ alertId: selectedAlertForOverride.id, reason: overrideReason });
  };

  const severeAlertsCount = activeAlerts?.filter(a => a.severityLevel === 'severe').length || 0;
  const moderateAlertsCount = activeAlerts?.filter(a => a.severityLevel === 'moderate').length || 0;

  return (
    <div className="container mx-auto p-6 max-w-6xl space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold" data-testid="heading-drug-interactions">
            {isDoctor ? "Drug Interaction Management" : "Drug Interaction Checker"}
          </h1>
          <p className="text-muted-foreground">
            {isDoctor 
              ? "Monitor, verify, and manage medication interactions for your patients"
              : "AI-powered drug interaction detection with clinical literature analysis"
            }
          </p>
        </div>
        {isDoctor && pendingRequests && pendingRequests.length > 0 && (
          <Badge variant="destructive" className="text-sm">
            <Clock className="h-3 w-3 mr-1" />
            {pendingRequests.length} Pending Requests
          </Badge>
        )}
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="count-active-alerts">{activeAlerts?.length || 0}</div>
            <p className="text-xs text-muted-foreground">Requires attention</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Severe Interactions</CardTitle>
            <AlertCircle className="h-4 w-4 text-destructive" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-destructive" data-testid="count-severe">{severeAlertsCount}</div>
            <p className="text-xs text-muted-foreground">High priority</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Moderate Interactions</CardTitle>
            <Pill className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="count-moderate">{moderateAlertsCount}</div>
            <p className="text-xs text-muted-foreground">Monitor carefully</p>
          </CardContent>
        </Card>
      </div>

      {isDoctor ? (
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="active" data-testid="tab-active">
              <AlertTriangle className="h-4 w-4 mr-2" />
              Active Alerts
              {activeAlerts && activeAlerts.length > 0 && (
                <Badge variant="destructive" className="ml-2">{activeAlerts.length}</Badge>
              )}
            </TabsTrigger>
            <TabsTrigger value="check" data-testid="tab-check">
              <Search className="h-4 w-4 mr-2" />
              Check Drug
            </TabsTrigger>
            <TabsTrigger value="regimen" data-testid="tab-regimen">
              <Pill className="h-4 w-4 mr-2" />
              Patient Regimen
            </TabsTrigger>
            <TabsTrigger value="requests" data-testid="tab-requests">
              <FileText className="h-4 w-4 mr-2" />
              Change Requests
              {pendingRequests && pendingRequests.length > 0 && (
                <Badge variant="destructive" className="ml-2">{pendingRequests.length}</Badge>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="space-y-4 mt-6">
            {alertsLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : activeAlerts && activeAlerts.length > 0 ? (
              activeAlerts.map((alert) => (
                <Card key={alert.id} className="border-amber-200 dark:border-amber-800" data-testid={`alert-${alert.id}`}>
                  <CardContent className="pt-6">
                    <DrugInteractionAlert
                      drug1={alert.medication1?.name || "Unknown"}
                      drug2={alert.medication2?.name || "Unknown"}
                      severityLevel={alert.interaction?.severityLevel || 'moderate'}
                      interactionType={alert.interaction?.interactionType || 'Unknown'}
                      clinicalEffects={alert.interaction?.clinicalEffects || ''}
                      mechanismDescription={alert.interaction?.mechanismDescription || ''}
                      managementRecommendations={alert.interaction?.managementRecommendations || ''}
                      alternativeSuggestions={alert.interaction?.alternativeSuggestions || []}
                      riskForImmunocompromised={alert.interaction?.riskForImmunocompromised}
                      requiresMonitoring={alert.interaction?.requiresMonitoring}
                      monitoringParameters={alert.interaction?.monitoringParameters || []}
                      evidenceLevel={alert.interaction?.evidenceLevel}
                      aiAnalysisConfidence={alert.interaction?.aiAnalysisConfidence}
                      status={alert.alertStatus}
                      onAcknowledge={() => acknowledgeMutation.mutate(alert.id)}
                      testId={`alert-content-${alert.id}`}
                    />
                  </CardContent>
                  <CardFooter className="flex justify-end gap-2 border-t pt-4">
                    <Button
                      variant="outline"
                      onClick={() => handleOverride(alert)}
                      data-testid={`button-override-${alert.id}`}
                    >
                      <Shield className="h-4 w-4 mr-2" />
                      Override with Justification
                    </Button>
                  </CardFooter>
                </Card>
              ))
            ) : (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <CheckCircle2 className="h-12 w-12 text-teal-500 mb-4" />
                  <p className="text-lg font-semibold mb-2">No Active Alerts</p>
                  <p className="text-sm text-muted-foreground text-center">
                    No active drug interaction warnings for your patients.
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="check" className="space-y-4 mt-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <User className="h-5 w-5" />
                  Select Patient
                </CardTitle>
                <CardDescription>
                  Choose a patient to check drug interactions for
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Select value={selectedPatient} onValueChange={setSelectedPatient}>
                  <SelectTrigger data-testid="select-patient">
                    <SelectValue placeholder="Select patient..." />
                  </SelectTrigger>
                  <SelectContent>
                    {patients?.map((patient) => (
                      <SelectItem key={patient.id} value={patient.id}>
                        {patient.firstName} {patient.lastName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="h-5 w-5" />
                  Check a New Medication
                </CardTitle>
                <CardDescription>
                  Analyze potential interactions before prescribing
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="drug-name">Medication Name</Label>
                  <div className="flex gap-2">
                    <Input
                      id="drug-name"
                      placeholder="e.g., Aspirin, Metformin, Lisinopril"
                      value={drugToCheck}
                      onChange={(e) => setDrugToCheck(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleCheckDrug()}
                      data-testid="input-drug-check"
                    />
                    <Button
                      onClick={handleCheckDrug}
                      disabled={checking || !drugToCheck.trim()}
                      data-testid="button-check-drug"
                    >
                      {checking ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Search className="h-4 w-4 mr-2" />
                          Check Interactions
                        </>
                      )}
                    </Button>
                  </div>
                </div>

                {checkResults && (
                  <div className="space-y-3 pt-3 border-t">
                    {checkResults.hasBlockingInteraction && (
                      <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>Severe Interaction Detected</AlertTitle>
                        <AlertDescription>
                          This medication has a severe interaction. Review carefully before prescribing.
                        </AlertDescription>
                      </Alert>
                    )}

                    {checkResults.hasInteractions && !checkResults.hasBlockingInteraction && (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Interactions Found</AlertTitle>
                        <AlertDescription>
                          Found {checkResults.interactions.length} potential interaction(s).
                        </AlertDescription>
                      </Alert>
                    )}

                    {!checkResults.hasInteractions && (
                      <Alert>
                        <CheckCircle2 className="h-4 w-4 text-teal-500" />
                        <AlertTitle>No Interactions Detected</AlertTitle>
                        <AlertDescription>
                          This medication appears safe with the patient's current regimen.
                        </AlertDescription>
                      </Alert>
                    )}

                    {checkResults.interactions && checkResults.interactions.length > 0 && (
                      <div className="space-y-3">
                        <h3 className="font-semibold">Detected Interactions:</h3>
                        {checkResults.interactions.map((interaction: any, index: number) => (
                          <DrugInteractionAlert
                            key={index}
                            drug1={interaction.drug1}
                            drug2={interaction.drug2}
                            severityLevel={interaction.interaction?.severityLevel || 'moderate'}
                            interactionType={interaction.interaction?.interactionType || 'Unknown'}
                            clinicalEffects={interaction.interaction?.clinicalEffects || ''}
                            mechanismDescription={interaction.interaction?.mechanismDescription || ''}
                            managementRecommendations={interaction.interaction?.managementRecommendations || ''}
                            alternativeSuggestions={interaction.interaction?.alternativeSuggestions || []}
                            riskForImmunocompromised={interaction.interaction?.riskForImmunocompromised}
                            requiresMonitoring={interaction.interaction?.requiresMonitoring}
                            monitoringParameters={interaction.interaction?.monitoringParameters || []}
                            evidenceLevel={interaction.interaction?.evidenceLevel}
                            aiAnalysisConfidence={interaction.interaction?.aiAnalysisConfidence}
                            testId={`check-result-${index}`}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="regimen" className="space-y-4 mt-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <User className="h-5 w-5" />
                  Select Patient
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Select value={selectedPatient} onValueChange={setSelectedPatient}>
                  <SelectTrigger data-testid="select-patient-regimen">
                    <SelectValue placeholder="Select patient..." />
                  </SelectTrigger>
                  <SelectContent>
                    {patients?.map((patient) => (
                      <SelectItem key={patient.id} value={patient.id}>
                        {patient.firstName} {patient.lastName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            {selectedPatient && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Current Medication Regimen
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {patientMedications && patientMedications.length > 0 ? (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Medication</TableHead>
                          <TableHead>Dosage</TableHead>
                          <TableHead>Frequency</TableHead>
                          <TableHead>Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {patientMedications.map((med: any) => (
                          <TableRow key={med.id} data-testid={`med-row-${med.id}`}>
                            <TableCell className="font-medium">{med.name}</TableCell>
                            <TableCell>{med.dosage}</TableCell>
                            <TableCell>{med.frequency}</TableCell>
                            <TableCell>
                              <Badge variant={med.status === "active" ? "default" : "secondary"}>
                                {med.status}
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <div className="text-center py-8">
                      <Pill className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
                      <p className="text-sm text-muted-foreground">No medications found for this patient</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="requests" className="space-y-4 mt-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Pending Dosage Change Requests
                </CardTitle>
                <CardDescription>
                  Review and approve or reject patient medication change requests
                </CardDescription>
              </CardHeader>
              <CardContent>
                {requestsLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : pendingRequests && pendingRequests.length > 0 ? (
                  <div className="space-y-4">
                    {pendingRequests.map((request) => (
                      <Card key={request.id} data-testid={`request-card-${request.id}`}>
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between gap-4">
                            <div className="space-y-2 flex-1">
                              <div className="flex items-center gap-2">
                                <Badge variant="outline">
                                  <Clock className="h-3 w-3 mr-1" />
                                  Pending Review
                                </Badge>
                                <span className="text-sm text-muted-foreground">
                                  {format(new Date(request.createdAt!), "MMM d, yyyy h:mm a")}
                                </span>
                              </div>
                              <div className="text-lg font-medium">Patient ID: {request.patientId}</div>
                              <div className="grid gap-2 md:grid-cols-2">
                                <div className="p-2 bg-muted rounded">
                                  <div className="text-xs text-muted-foreground">Current</div>
                                  <div className="font-medium">{request.currentDosage} - {request.currentFrequency}</div>
                                </div>
                                <div className="p-2 bg-primary/10 rounded">
                                  <div className="text-xs text-muted-foreground">Requested</div>
                                  <div className="font-medium text-primary">{request.requestedDosage} - {request.requestedFrequency}</div>
                                </div>
                              </div>
                              <div className="p-2 bg-muted rounded">
                                <div className="text-xs text-muted-foreground">Patient's Reason</div>
                                <div className="text-sm">{request.requestReason}</div>
                              </div>
                            </div>
                          </div>
                          <Separator className="my-4" />
                          <div className="flex justify-end gap-2">
                            <Button
                              variant="outline"
                              onClick={() => {
                                const notes = prompt("Reason for rejection:");
                                if (notes) {
                                  rejectDosageMutation.mutate({ requestId: request.id, notes });
                                }
                              }}
                              disabled={rejectDosageMutation.isPending}
                              data-testid={`button-reject-${request.id}`}
                            >
                              <XCircle className="h-4 w-4 mr-2" />
                              Reject
                            </Button>
                            <Button
                              onClick={() => {
                                const notes = prompt("Approval notes (optional):") || "";
                                approveDosageMutation.mutate({ requestId: request.id, notes });
                              }}
                              disabled={approveDosageMutation.isPending}
                              data-testid={`button-approve-${request.id}`}
                            >
                              <CheckCircle2 className="h-4 w-4 mr-2" />
                              Approve
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <CheckCircle2 className="h-12 w-12 mx-auto text-teal-500 mb-4" />
                    <h3 className="text-lg font-medium mb-1">No Pending Requests</h3>
                    <p className="text-sm text-muted-foreground">All dosage change requests have been processed</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5" />
                Check a New Medication
              </CardTitle>
              <CardDescription>
                Before adding a new medication, check if it interacts with your current medications
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="drug-name">Medication Name</Label>
                <div className="flex gap-2">
                  <Input
                    id="drug-name"
                    placeholder="e.g., Aspirin, Metformin, Lisinopril"
                    value={drugToCheck}
                    onChange={(e) => setDrugToCheck(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleCheckDrug()}
                    data-testid="input-drug-check"
                  />
                  <Button
                    onClick={handleCheckDrug}
                    disabled={checking || !drugToCheck.trim()}
                    data-testid="button-check-drug"
                  >
                    {checking ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Search className="h-4 w-4 mr-2" />
                        Check Interactions
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {checkResults && (
                <div className="space-y-3 pt-3 border-t">
                  {checkResults.hasBlockingInteraction && (
                    <Alert variant="destructive">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>Severe Interaction Detected</AlertTitle>
                      <AlertDescription>
                        This medication has a severe interaction with one or more of your current medications.
                        Do not take this medication without consulting your doctor first.
                      </AlertDescription>
                    </Alert>
                  )}

                  {checkResults.hasInteractions && !checkResults.hasBlockingInteraction && (
                    <Alert>
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>Interactions Found</AlertTitle>
                      <AlertDescription>
                        We found {checkResults.interactions.length} potential interaction(s). Review the details below.
                      </AlertDescription>
                    </Alert>
                  )}

                  {!checkResults.hasInteractions && (
                    <Alert>
                      <CheckCircle2 className="h-4 w-4 text-teal-500" />
                      <AlertTitle>No Interactions Detected</AlertTitle>
                      <AlertDescription>
                        This medication appears safe to take with your current medications. However, always consult
                        your doctor before starting any new medication.
                      </AlertDescription>
                    </Alert>
                  )}

                  {checkResults.interactions && checkResults.interactions.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="font-semibold">Detected Interactions:</h3>
                      {checkResults.interactions.map((interaction: any, index: number) => (
                        <DrugInteractionAlert
                          key={index}
                          drug1={interaction.drug1}
                          drug2={interaction.drug2}
                          severityLevel={interaction.interaction?.severityLevel || 'moderate'}
                          interactionType={interaction.interaction?.interactionType || 'Unknown'}
                          clinicalEffects={interaction.interaction?.clinicalEffects || ''}
                          mechanismDescription={interaction.interaction?.mechanismDescription || ''}
                          managementRecommendations={interaction.interaction?.managementRecommendations || ''}
                          alternativeSuggestions={interaction.interaction?.alternativeSuggestions || []}
                          riskForImmunocompromised={interaction.interaction?.riskForImmunocompromised}
                          requiresMonitoring={interaction.interaction?.requiresMonitoring}
                          monitoringParameters={interaction.interaction?.monitoringParameters || []}
                          evidenceLevel={interaction.interaction?.evidenceLevel}
                          aiAnalysisConfidence={interaction.interaction?.aiAnalysisConfidence}
                          testId={`check-result-${index}`}
                        />
                      ))}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <Tabs defaultValue="active" className="space-y-4">
            <TabsList data-testid="tabs-alerts">
              <TabsTrigger value="active" data-testid="tab-active">
                Active Alerts
                {activeAlerts && activeAlerts.length > 0 && (
                  <Badge variant="destructive" className="ml-2">{activeAlerts.length}</Badge>
                )}
              </TabsTrigger>
              <TabsTrigger value="all" data-testid="tab-all">All Alerts</TabsTrigger>
            </TabsList>

            <TabsContent value="active" className="space-y-4">
              {alertsLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : activeAlerts && activeAlerts.length > 0 ? (
                activeAlerts.map((alert) => (
                  <DrugInteractionAlert
                    key={alert.id}
                    drug1={alert.medication1?.name || "Unknown"}
                    drug2={alert.medication2?.name || "Unknown"}
                    severityLevel={alert.interaction?.severityLevel || 'moderate'}
                    interactionType={alert.interaction?.interactionType || 'Unknown'}
                    clinicalEffects={alert.interaction?.clinicalEffects || ''}
                    mechanismDescription={alert.interaction?.mechanismDescription || ''}
                    managementRecommendations={alert.interaction?.managementRecommendations || ''}
                    alternativeSuggestions={alert.interaction?.alternativeSuggestions || []}
                    riskForImmunocompromised={alert.interaction?.riskForImmunocompromised}
                    requiresMonitoring={alert.interaction?.requiresMonitoring}
                    monitoringParameters={alert.interaction?.monitoringParameters || []}
                    evidenceLevel={alert.interaction?.evidenceLevel}
                    aiAnalysisConfidence={alert.interaction?.aiAnalysisConfidence}
                    status={alert.alertStatus}
                    onAcknowledge={() => acknowledgeMutation.mutate(alert.id)}
                    testId={`alert-${alert.id}`}
                  />
                ))
              ) : (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <CheckCircle2 className="h-12 w-12 text-teal-500 mb-4" />
                    <p className="text-lg font-semibold mb-2">No Active Alerts</p>
                    <p className="text-sm text-muted-foreground text-center">
                      You don't have any active drug interaction warnings. Great job staying safe!
                    </p>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="all" className="space-y-4">
              {allAlerts && allAlerts.length > 0 ? (
                allAlerts.map((alert) => (
                  <DrugInteractionAlert
                    key={alert.id}
                    drug1={alert.medication1?.name || "Unknown"}
                    drug2={alert.medication2?.name || "Unknown"}
                    severityLevel={alert.interaction?.severityLevel || 'moderate'}
                    interactionType={alert.interaction?.interactionType || 'Unknown'}
                    clinicalEffects={alert.interaction?.clinicalEffects || ''}
                    mechanismDescription={alert.interaction?.mechanismDescription || ''}
                    managementRecommendations={alert.interaction?.managementRecommendations || ''}
                    alternativeSuggestions={alert.interaction?.alternativeSuggestions || []}
                    riskForImmunocompromised={alert.interaction?.riskForImmunocompromised}
                    requiresMonitoring={alert.interaction?.requiresMonitoring}
                    monitoringParameters={alert.interaction?.monitoringParameters || []}
                    evidenceLevel={alert.interaction?.evidenceLevel}
                    aiAnalysisConfidence={alert.interaction?.aiAnalysisConfidence}
                    status={alert.alertStatus}
                    testId={`all-alert-${alert.id}`}
                  />
                ))
              ) : (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <Pill className="h-12 w-12 text-muted-foreground mb-4" />
                    <p className="text-lg font-semibold mb-2">No Interaction History</p>
                    <p className="text-sm text-muted-foreground text-center">
                      No drug interaction alerts have been generated yet.
                    </p>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        </>
      )}

      <Dialog open={showOverrideDialog} onOpenChange={setShowOverrideDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Override Interaction Alert
            </DialogTitle>
            <DialogDescription>
              Provide clinical justification for overriding this drug interaction alert.
              This will be recorded in the patient's audit trail.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Clinical Responsibility</AlertTitle>
              <AlertDescription>
                By overriding this alert, you confirm that you have reviewed the interaction
                and determined that the benefits outweigh the risks for this patient.
              </AlertDescription>
            </Alert>
            <div className="space-y-2">
              <Label htmlFor="override-reason">Clinical Justification *</Label>
              <Textarea
                id="override-reason"
                placeholder="Provide detailed clinical justification for overriding this interaction..."
                value={overrideReason}
                onChange={(e) => setOverrideReason(e.target.value)}
                rows={4}
                data-testid="input-override-reason"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowOverrideDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={submitOverride}
              disabled={!overrideReason.trim() || overrideMutation.isPending}
            >
              {overrideMutation.isPending ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Shield className="h-4 w-4 mr-2" />
              )}
              Confirm Override
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
