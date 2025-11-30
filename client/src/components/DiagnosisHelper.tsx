import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { 
  Stethoscope, 
  Brain,
  AlertTriangle,
  CheckCircle,
  Lightbulb,
  FileText,
  Plus,
  X,
  Loader2,
  Sparkles,
  ClipboardList,
  BookOpen,
  AlertCircle,
  TrendingUp,
  User,
  Pill,
  Activity,
  ChevronDown,
  ChevronRight,
  Shield,
  Eye,
  Mic,
  Heart,
  RefreshCw,
  Lock,
  Clock,
  Cpu
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

interface SymptomEntry {
  name: string;
  duration: string;
  severity: 'mild' | 'moderate' | 'severe';
}

interface ConsentedPatient {
  patient_id: string;
  patient_name: string;
  patient_email?: string;
  sharing_link_id: string;
  consent_status: string;
  access_level: string;
  consent_given_at?: string;
  share_vitals: boolean;
  share_symptoms: boolean;
  share_medications: boolean;
  share_activities: boolean;
  share_mental_health: boolean;
  share_video_exams: boolean;
  share_audio_exams: boolean;
}

interface MedicalFile {
  id: string;
  file_name: string;
  file_type: string;
  file_category?: string;
  description?: string;
  uploaded_at: string;
}

interface HealthAlert {
  id: string;
  alert_type: string;
  alert_category: string;
  severity: string;
  priority: number;
  title: string;
  message: string;
  status: string;
  created_at: string;
  contributing_metrics?: any[];
}

interface MLInferenceResult {
  model_name: string;
  prediction_type: string;
  risk_score?: number;
  risk_level?: string;
  confidence?: number;
  details?: any;
  computed_at: string;
}

interface CurrentMedication {
  id: number;
  medication_name: string;
  generic_name?: string;
  drug_class?: string;
  dosage: string;
  frequency: string;
  route?: string;
  started_at: string;
  prescribed_by?: string;
  prescription_reason?: string;
  is_active: boolean;
}

interface FollowupSummary {
  summary_date: string;
  vital_signs?: any;
  symptom_summary?: string;
  pain_level?: number;
  mental_health_score?: any;
  video_exam_findings?: any;
  audio_exam_findings?: any;
  overall_status?: string;
  risk_indicators?: string[];
}

interface PatientDataAggregation {
  patient_id: string;
  patient_name: string;
  patient_age?: number;
  patient_sex?: string;
  consent_info: ConsentedPatient;
  medical_files: MedicalFile[];
  health_alerts: HealthAlert[];
  ml_inference_results: MLInferenceResult[];
  current_medications: CurrentMedication[];
  last_followup?: FollowupSummary;
  aggregated_at: string;
  audit_id?: string;
}

interface DiagnosisSuggestion {
  condition: string;
  probability: number;
  matching_symptoms: string[];
  missing_symptoms: string[];
  urgency: 'low' | 'moderate' | 'high' | 'emergency';
  description: string;
  recommended_tests: string[];
  differential_diagnosis: string[];
}

interface ClinicalAssessmentResult {
  primary_diagnosis?: DiagnosisSuggestion;
  differential_diagnoses: DiagnosisSuggestion[];
  clinical_insights: string[];
  recommended_actions: string[];
  red_flags: string[];
  patient_context_summary?: string;
  medication_considerations: string[];
  references: string[];
  disclaimer: string;
  analyzed_at: string;
  audit_id?: string;
}

interface DiagnosisHelperProps {
  patientId?: string;
  patientName?: string;
  className?: string;
}

export function DiagnosisHelper({ patientId: propPatientId, patientName: propPatientName, className }: DiagnosisHelperProps) {
  const { toast } = useToast();
  const [selectedPatientId, setSelectedPatientId] = useState<string>(propPatientId || "");
  const [symptoms, setSymptoms] = useState<SymptomEntry[]>([]);
  const [newSymptom, setNewSymptom] = useState("");
  const [newDuration, setNewDuration] = useState("");
  const [newSeverity, setNewSeverity] = useState<'mild' | 'moderate' | 'severe'>('moderate');
  const [additionalNotes, setAdditionalNotes] = useState("");
  const [result, setResult] = useState<ClinicalAssessmentResult | null>(null);
  const [patientData, setPatientData] = useState<PatientDataAggregation | null>(null);
  const [expandedSections, setExpandedSections] = useState<{[key: string]: boolean}>({
    medications: true,
    alerts: true,
    mlInference: false,
    followup: true,
    files: false
  });

  // Fetch consented patients
  const { data: consentedPatients, isLoading: loadingPatients } = useQuery<ConsentedPatient[]>({
    queryKey: ['/api/v1/clinical-assessment/patients'],
  });

  // Fetch patient data when selected
  const { data: fetchedPatientData, isLoading: loadingPatientData, refetch: refetchPatientData } = useQuery<PatientDataAggregation>({
    queryKey: ['/api/v1/clinical-assessment/patient', selectedPatientId, 'data'],
    enabled: !!selectedPatientId,
  });

  useEffect(() => {
    if (fetchedPatientData) {
      setPatientData(fetchedPatientData);
    }
  }, [fetchedPatientData]);

  // AI analysis mutation
  const analysisMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('/api/v1/clinical-assessment/analyze', {
        method: 'POST',
        body: JSON.stringify({
          patient_id: selectedPatientId,
          symptoms: symptoms.map(s => ({ name: s.name, duration: s.duration, severity: s.severity })),
          additional_notes: additionalNotes,
          patient_data: patientData
        })
      });
      return response as ClinicalAssessmentResult;
    },
    onSuccess: (data) => {
      setResult(data);
      toast({
        title: "Analysis Complete",
        description: "Comprehensive clinical assessment with patient context has been generated.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to analyze. Please try again.",
        variant: "destructive",
      });
    }
  });

  const addSymptom = () => {
    if (newSymptom.trim()) {
      setSymptoms([...symptoms, {
        name: newSymptom.trim(),
        duration: newDuration || 'Not specified',
        severity: newSeverity
      }]);
      setNewSymptom("");
      setNewDuration("");
      setNewSeverity('moderate');
    }
  };

  const removeSymptom = (index: number) => {
    setSymptoms(symptoms.filter((_, i) => i !== index));
  };

  const handleAnalyze = () => {
    if (symptoms.length > 0 && selectedPatientId) {
      analysisMutation.mutate();
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'mild': return 'bg-green-500/10 text-green-600 border-green-500/20';
      case 'moderate': return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
      case 'severe': return 'bg-red-500/10 text-red-600 border-red-500/20';
      case 'low': return 'bg-green-500/10 text-green-600';
      case 'medium': return 'bg-yellow-500/10 text-yellow-600';
      case 'high': return 'bg-orange-500/10 text-orange-600';
      case 'critical': return 'bg-red-500/10 text-red-600';
      default: return 'bg-muted';
    }
  };

  const getUrgencyVariant = (urgency: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (urgency) {
      case 'low': return 'default';
      case 'moderate': return 'secondary';
      case 'high': return 'destructive';
      case 'emergency': return 'destructive';
      default: return 'outline';
    }
  };

  const clearForm = () => {
    setSymptoms([]);
    setAdditionalNotes("");
    setResult(null);
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const selectedPatient = consentedPatients?.find(p => p.patient_id === selectedPatientId);

  return (
    <div className={`grid gap-6 lg:grid-cols-2 ${className}`}>
      {/* Left Panel - Clinical Assessment Input */}
      <Card data-testid="card-diagnosis-input">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Stethoscope className="h-5 w-5 text-primary" />
            Clinical Assessment
          </CardTitle>
          <CardDescription>
            Select a patient with active consent to load their health data for AI-powered diagnosis support
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Patient Selection */}
          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <User className="h-4 w-4" />
              Select Patient (Consented)
            </Label>
            <Select 
              value={selectedPatientId} 
              onValueChange={setSelectedPatientId}
              data-testid="select-patient"
            >
              <SelectTrigger data-testid="select-patient-trigger">
                <SelectValue placeholder={loadingPatients ? "Loading patients..." : "Select a patient with consent..."} />
              </SelectTrigger>
              <SelectContent>
                {consentedPatients?.map((patient) => (
                  <SelectItem key={patient.patient_id} value={patient.patient_id} data-testid={`patient-option-${patient.patient_id}`}>
                    <div className="flex items-center gap-2">
                      <Shield className="h-3 w-3 text-green-500" />
                      <span>{patient.patient_name}</span>
                      <Badge variant="outline" className="text-xs ml-2">
                        {patient.access_level}
                      </Badge>
                    </div>
                  </SelectItem>
                ))}
                {(!consentedPatients || consentedPatients.length === 0) && (
                  <div className="p-4 text-center text-sm text-muted-foreground">
                    <Lock className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    No patients with active consent
                  </div>
                )}
              </SelectContent>
            </Select>
            {selectedPatient && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <CheckCircle className="h-3 w-3 text-green-500" />
                Consent active since {selectedPatient.consent_given_at ? new Date(selectedPatient.consent_given_at).toLocaleDateString() : 'N/A'}
              </div>
            )}
          </div>

          {/* Patient Data Summary (when loaded) */}
          {selectedPatientId && (
            <div className="space-y-3">
              {loadingPatientData ? (
                <div className="flex items-center justify-center p-6 border rounded-lg bg-muted/30">
                  <Loader2 className="h-6 w-6 animate-spin mr-2" />
                  <span className="text-sm text-muted-foreground">Loading patient data...</span>
                </div>
              ) : patientData ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">Patient Health Data</Label>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => refetchPatientData()}
                      data-testid="button-refresh-patient-data"
                    >
                      <RefreshCw className="h-3 w-3 mr-1" />
                      Refresh
                    </Button>
                  </div>

                  {/* Current Medications */}
                  <Collapsible open={expandedSections.medications} onOpenChange={() => toggleSection('medications')}>
                    <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-lg border hover-elevate" data-testid="collapsible-medications">
                      <div className="flex items-center gap-2">
                        <Pill className="h-4 w-4 text-blue-500" />
                        <span className="text-sm font-medium">Current Medications</span>
                        <Badge variant="secondary" className="text-xs">{patientData.current_medications.length}</Badge>
                      </div>
                      {expandedSections.medications ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mt-2 space-y-1">
                      {patientData.current_medications.length > 0 ? (
                        patientData.current_medications.map((med, idx) => (
                          <div key={idx} className="p-2 text-xs bg-muted/50 rounded border-l-2 border-blue-500" data-testid={`medication-${idx}`}>
                            <div className="font-medium">{med.medication_name}</div>
                            <div className="text-muted-foreground">{med.dosage} - {med.frequency}</div>
                            {med.prescription_reason && (
                              <div className="text-muted-foreground italic">For: {med.prescription_reason}</div>
                            )}
                          </div>
                        ))
                      ) : (
                        <div className="p-2 text-xs text-muted-foreground">No active medications</div>
                      )}
                    </CollapsibleContent>
                  </Collapsible>

                  {/* Health Alerts */}
                  <Collapsible open={expandedSections.alerts} onOpenChange={() => toggleSection('alerts')}>
                    <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-lg border hover-elevate" data-testid="collapsible-alerts">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-orange-500" />
                        <span className="text-sm font-medium">Health Alerts</span>
                        <Badge variant={patientData.health_alerts.length > 0 ? "destructive" : "secondary"} className="text-xs">
                          {patientData.health_alerts.length}
                        </Badge>
                      </div>
                      {expandedSections.alerts ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mt-2 space-y-1">
                      {patientData.health_alerts.length > 0 ? (
                        patientData.health_alerts.slice(0, 5).map((alert, idx) => (
                          <div key={idx} className={`p-2 text-xs rounded border-l-2 ${getSeverityColor(alert.severity)}`} data-testid={`alert-${idx}`}>
                            <div className="flex items-center justify-between">
                              <span className="font-medium">{alert.title}</span>
                              <Badge variant="outline" className="text-xs">{alert.severity}</Badge>
                            </div>
                            <div className="text-muted-foreground mt-1">{alert.message}</div>
                          </div>
                        ))
                      ) : (
                        <div className="p-2 text-xs text-muted-foreground">No active health alerts</div>
                      )}
                    </CollapsibleContent>
                  </Collapsible>

                  {/* ML Inference Results */}
                  <Collapsible open={expandedSections.mlInference} onOpenChange={() => toggleSection('mlInference')}>
                    <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-lg border hover-elevate" data-testid="collapsible-ml-inference">
                      <div className="flex items-center gap-2">
                        <Cpu className="h-4 w-4 text-purple-500" />
                        <span className="text-sm font-medium">ML Risk Predictions</span>
                        <Badge variant="secondary" className="text-xs">{patientData.ml_inference_results.length}</Badge>
                      </div>
                      {expandedSections.mlInference ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mt-2 space-y-1">
                      {patientData.ml_inference_results.length > 0 ? (
                        patientData.ml_inference_results.map((ml, idx) => (
                          <div key={idx} className="p-2 text-xs bg-muted/50 rounded border-l-2 border-purple-500" data-testid={`ml-result-${idx}`}>
                            <div className="flex items-center justify-between">
                              <span className="font-medium">{ml.model_name}</span>
                              {ml.risk_level && (
                                <Badge variant="outline" className={`text-xs ${getSeverityColor(ml.risk_level)}`}>
                                  {ml.risk_level}
                                </Badge>
                              )}
                            </div>
                            <div className="text-muted-foreground">
                              {ml.prediction_type}
                              {ml.risk_score !== undefined && ` - Score: ${(ml.risk_score * 100).toFixed(1)}%`}
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="p-2 text-xs text-muted-foreground">No recent ML predictions</div>
                      )}
                    </CollapsibleContent>
                  </Collapsible>

                  {/* Last Follow-up Summary */}
                  {patientData.last_followup && (
                    <Collapsible open={expandedSections.followup} onOpenChange={() => toggleSection('followup')}>
                      <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-lg border hover-elevate" data-testid="collapsible-followup">
                        <div className="flex items-center gap-2">
                          <Heart className="h-4 w-4 text-red-500" />
                          <span className="text-sm font-medium">Last Follow-up</span>
                          <Badge variant="outline" className="text-xs">
                            <Clock className="h-3 w-3 mr-1" />
                            {new Date(patientData.last_followup.summary_date).toLocaleDateString()}
                          </Badge>
                        </div>
                        {expandedSections.followup ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                      </CollapsibleTrigger>
                      <CollapsibleContent className="mt-2">
                        <div className="p-3 text-xs bg-muted/50 rounded space-y-2" data-testid="followup-summary">
                          {patientData.last_followup.overall_status && (
                            <div className="flex items-center gap-2">
                              <Activity className="h-3 w-3" />
                              <span>Status: {patientData.last_followup.overall_status}</span>
                            </div>
                          )}
                          {patientData.last_followup.pain_level !== undefined && (
                            <div className="flex items-center gap-2">
                              <span>Pain Level: {patientData.last_followup.pain_level}/10</span>
                            </div>
                          )}
                          {patientData.last_followup.symptom_summary && (
                            <div className="text-muted-foreground">{patientData.last_followup.symptom_summary}</div>
                          )}
                          {patientData.last_followup.risk_indicators && patientData.last_followup.risk_indicators.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {patientData.last_followup.risk_indicators.map((risk, idx) => (
                                <Badge key={idx} variant="destructive" className="text-xs">{risk}</Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  )}
                </div>
              ) : (
                <div className="p-4 text-center text-sm text-muted-foreground border rounded-lg">
                  Select a patient to load their health data
                </div>
              )}
            </div>
          )}

          <Separator />

          {/* Symptom Entry */}
          <div className="space-y-3">
            <Label>Presenting Symptoms</Label>
            <div className="flex gap-2 flex-wrap">
              <Input
                placeholder="Enter symptom..."
                value={newSymptom}
                onChange={(e) => setNewSymptom(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addSymptom()}
                className="flex-1 min-w-[150px]"
                data-testid="input-new-symptom"
              />
              <Input
                placeholder="Duration"
                value={newDuration}
                onChange={(e) => setNewDuration(e.target.value)}
                className="w-24"
                data-testid="input-symptom-duration"
              />
              <Select value={newSeverity} onValueChange={(v) => setNewSeverity(v as any)}>
                <SelectTrigger className="w-28" data-testid="select-symptom-severity">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mild">Mild</SelectItem>
                  <SelectItem value="moderate">Moderate</SelectItem>
                  <SelectItem value="severe">Severe</SelectItem>
                </SelectContent>
              </Select>
              <Button size="icon" onClick={addSymptom} data-testid="button-add-symptom">
                <Plus className="h-4 w-4" />
              </Button>
            </div>

            {symptoms.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {symptoms.map((symptom, index) => (
                  <Badge
                    key={index}
                    variant="outline"
                    className={`${getSeverityColor(symptom.severity)} px-3 py-1`}
                    data-testid={`symptom-badge-${index}`}
                  >
                    <span className="mr-1">{symptom.name}</span>
                    <span className="text-xs opacity-70">({symptom.duration})</span>
                    <button
                      onClick={() => removeSymptom(index)}
                      className="ml-2 hover:text-destructive"
                      data-testid={`button-remove-symptom-${index}`}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Additional Notes */}
          <div className="space-y-2">
            <Label htmlFor="notes">Additional Clinical Notes</Label>
            <Textarea
              id="notes"
              placeholder="Any other relevant observations..."
              value={additionalNotes}
              onChange={(e) => setAdditionalNotes(e.target.value)}
              className="min-h-[60px]"
              data-testid="textarea-additional-notes"
            />
          </div>
        </CardContent>
        <CardFooter className="flex justify-between gap-2">
          <Button variant="outline" onClick={clearForm} data-testid="button-clear-diagnosis">
            Clear Form
          </Button>
          <Button
            onClick={handleAnalyze}
            disabled={symptoms.length === 0 || !selectedPatientId || analysisMutation.isPending}
            data-testid="button-analyze-diagnosis"
          >
            {analysisMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing with AI...
              </>
            ) : (
              <>
                <Brain className="h-4 w-4 mr-2" />
                Analyze with Full Context
              </>
            )}
          </Button>
        </CardFooter>
      </Card>

      {/* Right Panel - AI Diagnosis Results */}
      <Card data-testid="card-diagnosis-results">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            AI Diagnosis Support
          </CardTitle>
          <CardDescription>
            Comprehensive clinical decision support with patient context (not a substitute for professional judgment)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[600px] pr-4">
            {analysisMutation.isPending ? (
              <div className="flex flex-col items-center justify-center py-16">
                <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
                <p className="text-muted-foreground">Analyzing symptoms with patient context...</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Integrating medications, health alerts, and ML predictions
                </p>
              </div>
            ) : result ? (
              <div className="space-y-6">
                {/* Red Flags */}
                {result.red_flags.length > 0 && (
                  <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20">
                    <div className="flex items-center gap-2 mb-3">
                      <AlertTriangle className="h-5 w-5 text-destructive" />
                      <span className="font-semibold text-destructive">Red Flags Identified</span>
                    </div>
                    <ul className="space-y-1 text-sm">
                      {result.red_flags.map((flag, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <AlertCircle className="h-4 w-4 text-destructive mt-0.5" />
                          <span>{flag}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Patient Context Summary */}
                {result.patient_context_summary && (
                  <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <User className="h-5 w-5 text-blue-500" />
                      <span className="font-semibold text-blue-700 dark:text-blue-300">Patient Context Summary</span>
                    </div>
                    <p className="text-sm">{result.patient_context_summary}</p>
                  </div>
                )}

                {/* Primary Diagnosis */}
                {result.primary_diagnosis && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5 text-primary" />
                      <span className="font-semibold">Primary Diagnosis Suggestion</span>
                    </div>
                    
                    <div className="p-4 rounded-lg border bg-primary/5">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h4 className="font-semibold text-lg">{result.primary_diagnosis.condition}</h4>
                          <p className="text-sm text-muted-foreground mt-1">
                            {result.primary_diagnosis.description}
                          </p>
                        </div>
                        <Badge variant={getUrgencyVariant(result.primary_diagnosis.urgency)}>
                          {result.primary_diagnosis.urgency} urgency
                        </Badge>
                      </div>
                      
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-sm text-muted-foreground">Confidence:</span>
                        <Progress value={result.primary_diagnosis.probability * 100} className="flex-1 h-2" />
                        <span className="text-sm font-medium">{(result.primary_diagnosis.probability * 100).toFixed(0)}%</span>
                      </div>

                      {result.primary_diagnosis.matching_symptoms.length > 0 && (
                        <div className="flex flex-wrap gap-1 mb-2">
                          <span className="text-xs text-muted-foreground mr-1">Matching:</span>
                          {result.primary_diagnosis.matching_symptoms.map((s, idx) => (
                            <Badge key={idx} variant="secondary" className="text-xs">
                              <CheckCircle className="h-3 w-3 mr-1" />
                              {s}
                            </Badge>
                          ))}
                        </div>
                      )}

                      {result.primary_diagnosis.recommended_tests.length > 0 && (
                        <div className="mt-3 pt-3 border-t">
                          <span className="text-sm font-medium flex items-center gap-1 mb-2">
                            <ClipboardList className="h-4 w-4" />
                            Recommended Tests
                          </span>
                          <div className="flex flex-wrap gap-1">
                            {result.primary_diagnosis.recommended_tests.map((test, idx) => (
                              <Badge key={idx} variant="outline">{test}</Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Medication Considerations */}
                {result.medication_considerations.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Pill className="h-5 w-5 text-blue-500" />
                      <span className="font-semibold">Medication Considerations</span>
                    </div>
                    <ul className="space-y-2">
                      {result.medication_considerations.map((consideration, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm p-2 bg-blue-500/10 rounded">
                          <AlertCircle className="h-4 w-4 text-blue-500 mt-0.5" />
                          <span>{consideration}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Differential Diagnoses */}
                {result.differential_diagnoses.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <FileText className="h-5 w-5 text-muted-foreground" />
                      <span className="font-semibold">Differential Diagnoses</span>
                    </div>
                    <div className="space-y-2">
                      {result.differential_diagnoses.map((dx, idx) => (
                        <div key={idx} className="p-3 rounded-lg border">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-medium">{dx.condition}</span>
                            <div className="flex items-center gap-2">
                              <Progress value={dx.probability * 100} className="w-16 h-2" />
                              <span className="text-sm text-muted-foreground">{(dx.probability * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                          <p className="text-sm text-muted-foreground">{dx.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Clinical Insights */}
                {result.clinical_insights.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Lightbulb className="h-5 w-5 text-yellow-500" />
                      <span className="font-semibold">Clinical Insights</span>
                    </div>
                    <ul className="space-y-2">
                      {result.clinical_insights.map((insight, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm">
                          <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Recommended Actions */}
                {result.recommended_actions.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <ClipboardList className="h-5 w-5 text-blue-500" />
                      <span className="font-semibold">Recommended Actions</span>
                    </div>
                    <ol className="space-y-2 list-decimal list-inside">
                      {result.recommended_actions.map((action, idx) => (
                        <li key={idx} className="text-sm">{action}</li>
                      ))}
                    </ol>
                  </div>
                )}

                {/* Disclaimer */}
                <div className="p-3 rounded-lg bg-muted/50 text-xs text-muted-foreground">
                  <AlertCircle className="h-4 w-4 inline mr-1" />
                  {result.disclaimer}
                </div>

                {/* Audit ID */}
                {result.audit_id && (
                  <div className="text-xs text-muted-foreground flex items-center gap-1">
                    <Shield className="h-3 w-3" />
                    Audit ID: {result.audit_id}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-center text-muted-foreground">
                <Brain className="h-16 w-16 mb-4 opacity-30" />
                <p className="text-lg font-medium mb-2">Ready to Analyze</p>
                <p className="text-sm max-w-md">
                  Select a patient with active consent, enter symptoms, and click "Analyze with Full Context" to receive AI-powered diagnosis suggestions based on their complete health data.
                </p>
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

export default DiagnosisHelper;
