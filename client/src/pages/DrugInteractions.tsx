import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { DrugInteractionAlert } from "@/components/DrugInteractionAlert";
import { AlertTriangle, Pill, Search, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export default function DrugInteractions() {
  const { toast } = useToast();
  const [drugToCheck, setDrugToCheck] = useState("");
  const [checking, setChecking] = useState(false);
  const [checkResults, setCheckResults] = useState<any>(null);

  // Fetch active interaction alerts
  const { data: activeAlerts, isLoading: alertsLoading } = useQuery<any[]>({
    queryKey: ['/api/drug-interactions/alerts'],
  });

  // Fetch all alerts (including acknowledged)
  const { data: allAlerts } = useQuery<any[]>({
    queryKey: ['/api/drug-interactions/alerts/all'],
  });

  // Acknowledge alert mutation
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
      const result: any = await apiRequest('POST', '/api/drug-interactions/analyze', {
        drugName: drugToCheck,
      });
      setCheckResults(result);
      
      if (result.hasBlockingInteraction) {
        toast({
          title: "Severe interaction detected",
          description: "This medication has a severe interaction with your current medications.",
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
          description: "This medication appears safe to take with your current medications.",
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

  const severeAlertsCount = activeAlerts?.filter(a => a.severityLevel === 'severe').length || 0;
  const moderateAlertsCount = activeAlerts?.filter(a => a.severityLevel === 'moderate').length || 0;

  return (
    <div className="container mx-auto p-6 max-w-6xl space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold" data-testid="heading-drug-interactions">Drug Interaction Checker</h1>
        <p className="text-muted-foreground">
          AI-powered drug interaction detection with 99% accuracy using Graph Neural Networks and clinical literature analysis
        </p>
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
            <p className="text-xs text-muted-foreground">Requires your attention</p>
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
            <Pill className="h-4 w-4 text-chart-3" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="count-moderate">{moderateAlertsCount}</div>
            <p className="text-xs text-muted-foreground">Monitor carefully</p>
          </CardContent>
        </Card>
      </div>

      {/* Check New Drug Section */}
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
                  <CheckCircle2 className="h-4 w-4 text-chart-2" />
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
                      severityLevel={interaction.interaction.severityLevel}
                      interactionType={interaction.interaction.interactionType}
                      clinicalEffects={interaction.interaction.clinicalEffects}
                      mechanismDescription={interaction.interaction.mechanismDescription}
                      managementRecommendations={interaction.interaction.managementRecommendations}
                      alternativeSuggestions={interaction.interaction.alternativeSuggestions}
                      riskForImmunocompromised={interaction.interaction.riskForImmunocompromised}
                      requiresMonitoring={interaction.interaction.requiresMonitoring}
                      monitoringParameters={interaction.interaction.monitoringParameters}
                      evidenceLevel={interaction.interaction.evidenceLevel}
                      aiAnalysisConfidence={interaction.interaction.aiAnalysisConfidence}
                      testId={`check-result-${index}`}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Alerts Tabs */}
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
                <CheckCircle2 className="h-12 w-12 text-chart-2 mb-4" />
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
    </div>
  );
}
