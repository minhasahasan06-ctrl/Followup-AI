import { useParams } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Download, AlertTriangle, TrendingUp, TrendingDown, Activity, Calendar, Pill, FileText } from "lucide-react";
import { format } from "date-fns";

type CorrelationStrength = "strong" | "likely" | "possible" | "unlikely";

interface MedicationCorrelation {
  symptom_name: string;
  correlation_strength: CorrelationStrength;
  confidence_score: number;
  time_to_onset_hours: number;
  patient_impact: string;
  action_recommended: string;
  temporal_pattern: string | null;
}

interface MedicationDetail {
  medication_name: string;
  generic_name: string | null;
  drug_class: string | null;
  dosage: string;
  frequency: string;
  route: string | null;
  started_at: string | null;
  prescribed_by: string | null;
  prescription_reason: string | null;
  correlation_count: number;
  strong_correlation_count: number;
  correlations: MedicationCorrelation[];
}

interface SymptomEntry {
  symptom_name: string;
  severity: number | null;
  reported_at: string | null;
  description: string | null;
}

interface CriticalFinding {
  id: number;
  medication_name: string;
  symptom_name: string;
  confidence_score: number;
  patient_impact: string;
  action_recommended: string;
  temporal_pattern: string | null;
  ai_reasoning: string | null;
}

interface ClinicalRecommendation {
  priority: string;
  medication: string;
  symptom: string;
  action: string;
  reasoning: string;
}

interface ReportData {
  patient_id: string;
  report_generated_at: string;
  analysis_period_days: number;
  doctor_id: string;
  doctor_name: string;
  active_medications_count: number;
  all_medications_in_period: number;
  medications: MedicationDetail[];
  total_symptoms_reported: number;
  symptoms_by_source: Record<string, SymptomEntry[]>;
  unique_symptoms: number;
  total_correlations: number;
  strong_correlations_count: number;
  likely_correlations_count: number;
  possible_correlations_count: number;
  critical_findings: CriticalFinding[];
  clinical_recommendations: ClinicalRecommendation[];
  last_analysis_date: string | null;
  analysis_status: "current" | "needs_update";
}

export default function DoctorMedicationReport() {
  const params = useParams();
  const patientId = params.id;

  const { data: report, isLoading } = useQuery<ReportData>({
    queryKey: ['/api/v1/medication-side-effects/doctor/patient', patientId, 'consultation-report'],
    enabled: !!patientId
  });

  if (isLoading) {
    return (
      <div className="container mx-auto p-6 space-y-6">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (!report) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive" data-testid="alert-report-not-found">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Report Not Found</AlertTitle>
          <AlertDescription>
            Unable to load medication effects report for this patient.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const getStrengthColor = (strength: CorrelationStrength): string => {
    const colors = {
      strong: "bg-rose-500 text-white",
      likely: "bg-coral-500 text-white",
      possible: "bg-amber-500 text-black",
      unlikely: "bg-slate-300 text-black"
    };
    return colors[strength] || "bg-slate-300";
  };

  const getImpactColor = (impact: string): string => {
    const impactLower = impact.toLowerCase();
    if (impactLower.includes("severe")) return "bg-rose-500 text-white";
    if (impactLower.includes("moderate")) return "bg-amber-500 text-black";
    return "bg-emerald-500 text-white";
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight" data-testid="heading-report-title">
            Medication Effects Report
          </h1>
          <p className="text-muted-foreground mt-1" data-testid="text-patient-id">
            Patient ID: {report.patient_id}
          </p>
          <p className="text-sm text-muted-foreground" data-testid="text-report-date">
            Generated: {format(new Date(report.report_generated_at), 'PPpp')} | Analysis Period: {report.analysis_period_days} days
          </p>
        </div>
        <Button variant="outline" data-testid="button-download-pdf">
          <Download className="mr-2 h-4 w-4" />
          Download PDF
        </Button>
      </div>

      {/* Analysis Status Alert */}
      {report.analysis_status === "needs_update" && (
        <Alert data-testid="alert-needs-update">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Analysis May Be Outdated</AlertTitle>
          <AlertDescription>
            The last correlation analysis was performed {report.last_analysis_date ? format(new Date(report.last_analysis_date), 'PPP') : 'more than a week ago'}.
            Consider triggering a new analysis for current insights.
          </AlertDescription>
        </Alert>
      )}

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card data-testid="card-medications">
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Medications</CardTitle>
            <Pill className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-medications-count">{report.active_medications_count}</div>
            <p className="text-xs text-muted-foreground">
              {report.all_medications_in_period} total in period
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-symptoms">
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Symptoms</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-symptoms-count">{report.total_symptoms_reported}</div>
            <p className="text-xs text-muted-foreground">
              {report.unique_symptoms} unique symptoms
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-correlations">
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Correlations Found</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-correlations-count">{report.total_correlations}</div>
            <p className="text-xs text-muted-foreground">
              {report.strong_correlations_count} strong, {report.likely_correlations_count} likely
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-critical-findings" className={report.strong_correlations_count > 0 ? "border-rose-500" : ""}>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Critical Findings</CardTitle>
            <AlertTriangle className={`h-4 w-4 ${report.strong_correlations_count > 0 ? "text-rose-500" : "text-muted-foreground"}`} />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${report.strong_correlations_count > 0 ? "text-rose-500" : ""}`} data-testid="text-critical-count">
              {report.strong_correlations_count}
            </div>
            <p className="text-xs text-muted-foreground">
              Require immediate attention
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="critical" className="space-y-4">
        <TabsList data-testid="tabs-report">
          <TabsTrigger value="critical" data-testid="tab-critical">Critical Findings</TabsTrigger>
          <TabsTrigger value="medications" data-testid="tab-medications">Medications</TabsTrigger>
          <TabsTrigger value="symptoms" data-testid="tab-symptoms">Symptoms</TabsTrigger>
          <TabsTrigger value="recommendations" data-testid="tab-recommendations">Recommendations</TabsTrigger>
        </TabsList>

        {/* Critical Findings Tab */}
        <TabsContent value="critical" className="space-y-4" data-testid="tabcontent-critical">
          {report.critical_findings.length === 0 ? (
            <Card>
              <CardContent className="pt-6">
                <p className="text-center text-muted-foreground" data-testid="text-no-critical">
                  No critical findings detected in this analysis period.
                </p>
              </CardContent>
            </Card>
          ) : (
            report.critical_findings.map((finding) => (
              <Card key={finding.id} data-testid={`card-finding-${finding.id}`}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="space-y-1 flex-1">
                      <CardTitle className="text-lg" data-testid={`text-finding-medication-${finding.id}`}>
                        {finding.medication_name} → {finding.symptom_name}
                      </CardTitle>
                      <CardDescription data-testid={`text-finding-pattern-${finding.id}`}>
                        {finding.temporal_pattern || "No temporal pattern specified"}
                      </CardDescription>
                    </div>
                    <Badge className="bg-rose-500 text-white" data-testid={`badge-finding-confidence-${finding.id}`}>
                      {Math.round(finding.confidence_score * 100)}% confidence
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Patient Impact</p>
                      <Badge className={getImpactColor(finding.patient_impact)} data-testid={`badge-finding-impact-${finding.id}`}>
                        {finding.patient_impact}
                      </Badge>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Recommended Action</p>
                      <p className="text-sm" data-testid={`text-finding-action-${finding.id}`}>{finding.action_recommended}</p>
                    </div>
                  </div>
                  {finding.ai_reasoning && (
                    <div className="bg-muted p-3 rounded-md">
                      <p className="text-sm font-medium mb-1">AI Analysis:</p>
                      <p className="text-sm text-muted-foreground" data-testid={`text-finding-reasoning-${finding.id}`}>
                        {finding.ai_reasoning}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        {/* Medications Tab */}
        <TabsContent value="medications" className="space-y-4" data-testid="tabcontent-medications">
          {report.medications.map((med, index) => (
            <Card key={index} data-testid={`card-medication-${index}`}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle data-testid={`text-medication-name-${index}`}>{med.medication_name}</CardTitle>
                    {med.generic_name && (
                      <CardDescription data-testid={`text-medication-generic-${index}`}>Generic: {med.generic_name}</CardDescription>
                    )}
                  </div>
                  <div className="text-right space-y-1">
                    <Badge variant="outline" data-testid={`badge-medication-correlations-${index}`}>
                      {med.correlation_count} correlations
                    </Badge>
                    {med.strong_correlation_count > 0 && (
                      <Badge className="bg-rose-500 text-white ml-2" data-testid={`badge-medication-strong-${index}`}>
                        {med.strong_correlation_count} strong
                      </Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-3">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Dosage</p>
                    <p className="text-sm" data-testid={`text-medication-dosage-${index}`}>{med.dosage} {med.frequency}</p>
                  </div>
                  {med.drug_class && (
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Drug Class</p>
                      <p className="text-sm" data-testid={`text-medication-class-${index}`}>{med.drug_class}</p>
                    </div>
                  )}
                  {med.started_at && (
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Started</p>
                      <p className="text-sm" data-testid={`text-medication-started-${index}`}>
                        {format(new Date(med.started_at), 'PP')}
                      </p>
                    </div>
                  )}
                </div>

                {med.correlations.length > 0 && (
                  <>
                    <Separator />
                    <div>
                      <p className="text-sm font-medium mb-2">Associated Symptoms:</p>
                      <div className="space-y-2">
                        {med.correlations.map((corr, corrIndex) => (
                          <div key={corrIndex} className="flex items-center justify-between p-2 bg-muted rounded" data-testid={`correlation-${index}-${corrIndex}`}>
                            <div className="flex items-center gap-2">
                              <Badge className={getStrengthColor(corr.correlation_strength)} data-testid={`badge-correlation-strength-${index}-${corrIndex}`}>
                                {corr.correlation_strength}
                              </Badge>
                              <span className="text-sm" data-testid={`text-correlation-symptom-${index}-${corrIndex}`}>{corr.symptom_name}</span>
                            </div>
                            <span className="text-xs text-muted-foreground" data-testid={`text-correlation-onset-${index}-${corrIndex}`}>
                              {corr.time_to_onset_hours}h onset
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        {/* Symptoms Tab */}
        <TabsContent value="symptoms" className="space-y-4" data-testid="tabcontent-symptoms">
          {Object.entries(report.symptoms_by_source).map(([source, symptoms]) => (
            <Card key={source} data-testid={`card-symptoms-${source}`}>
              <CardHeader>
                <CardTitle className="capitalize" data-testid={`heading-symptoms-source-${source}`}>
                  {source.replace('_', ' ')} ({symptoms.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {symptoms.slice(0, 10).map((symptom, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-muted rounded" data-testid={`symptom-${source}-${index}`}>
                      <div className="flex-1">
                        <p className="text-sm font-medium" data-testid={`text-symptom-name-${source}-${index}`}>{symptom.symptom_name}</p>
                        {symptom.description && (
                          <p className="text-xs text-muted-foreground" data-testid={`text-symptom-desc-${source}-${index}`}>
                            {symptom.description}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        {symptom.severity && (
                          <Badge variant="outline" data-testid={`badge-symptom-severity-${source}-${index}`}>
                            Severity: {symptom.severity}/10
                          </Badge>
                        )}
                        {symptom.reported_at && (
                          <span className="text-xs text-muted-foreground" data-testid={`text-symptom-date-${source}-${index}`}>
                            {format(new Date(symptom.reported_at), 'MM/dd/yy')}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                  {symptoms.length > 10 && (
                    <p className="text-sm text-muted-foreground text-center" data-testid={`text-symptoms-more-${source}`}>
                      And {symptoms.length - 10} more...
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        {/* Recommendations Tab */}
        <TabsContent value="recommendations" className="space-y-4" data-testid="tabcontent-recommendations">
          {report.clinical_recommendations.length === 0 ? (
            <Card>
              <CardContent className="pt-6">
                <p className="text-center text-muted-foreground" data-testid="text-no-recommendations">
                  No specific clinical recommendations at this time.
                </p>
              </CardContent>
            </Card>
          ) : (
            report.clinical_recommendations.map((rec, index) => (
              <Card key={index} data-testid={`card-recommendation-${index}`}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <CardTitle className="text-lg" data-testid={`text-recommendation-medication-${index}`}>
                        {rec.medication} - {rec.symptom}
                      </CardTitle>
                    </div>
                    <Badge 
                      className={rec.priority === "HIGH" ? "bg-rose-500 text-white" : "bg-amber-500"} 
                      data-testid={`badge-recommendation-priority-${index}`}
                    >
                      {rec.priority}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Recommended Action</p>
                    <p className="text-sm" data-testid={`text-recommendation-action-${index}`}>{rec.action}</p>
                  </div>
                  {rec.reasoning && (
                    <div className="bg-muted p-3 rounded-md">
                      <p className="text-sm font-medium mb-1">Clinical Reasoning:</p>
                      <p className="text-sm text-muted-foreground" data-testid={`text-recommendation-reasoning-${index}`}>
                        {rec.reasoning}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
