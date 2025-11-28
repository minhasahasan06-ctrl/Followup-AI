import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
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
  TrendingUp
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

interface SymptomEntry {
  name: string;
  duration: string;
  severity: 'mild' | 'moderate' | 'severe';
}

interface DiagnosisSuggestion {
  condition: string;
  probability: number;
  matchingSymptoms: string[];
  missingSymptoms: string[];
  urgency: 'low' | 'moderate' | 'high' | 'emergency';
  description: string;
  recommendedTests: string[];
  differentialDiagnosis: string[];
}

interface DiagnosisResult {
  primaryDiagnosis: DiagnosisSuggestion;
  differentialDiagnoses: DiagnosisSuggestion[];
  clinicalInsights: string[];
  recommendedActions: string[];
  redFlags: string[];
  references: string[];
}

interface DiagnosisHelperProps {
  patientId?: string;
  patientName?: string;
  className?: string;
}

export function DiagnosisHelper({ patientId, patientName, className }: DiagnosisHelperProps) {
  const { toast } = useToast();
  const [symptoms, setSymptoms] = useState<SymptomEntry[]>([]);
  const [newSymptom, setNewSymptom] = useState("");
  const [newDuration, setNewDuration] = useState("");
  const [newSeverity, setNewSeverity] = useState<'mild' | 'moderate' | 'severe'>('moderate');
  const [patientAge, setPatientAge] = useState("");
  const [patientSex, setPatientSex] = useState("");
  const [medicalHistory, setMedicalHistory] = useState("");
  const [currentMedications, setCurrentMedications] = useState("");
  const [additionalNotes, setAdditionalNotes] = useState("");
  const [result, setResult] = useState<DiagnosisResult | null>(null);

  const diagnosisMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('/api/v1/lysa/diagnosis/analyze', {
        method: 'POST',
        body: JSON.stringify({
          patientId,
          symptoms,
          patientAge,
          patientSex,
          medicalHistory,
          currentMedications,
          additionalNotes
        })
      });
      return response;
    },
    onSuccess: (data: DiagnosisResult) => {
      setResult(data);
      toast({
        title: "Analysis Complete",
        description: "Clinical decision support analysis has been generated.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to analyze symptoms. Please try again.",
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
    if (symptoms.length > 0) {
      diagnosisMutation.mutate();
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'mild': return 'bg-green-500/10 text-green-600 border-green-500/20';
      case 'moderate': return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
      case 'severe': return 'bg-red-500/10 text-red-600 border-red-500/20';
      default: return 'bg-muted';
    }
  };

  const getUrgencyColor = (urgency: string) => {
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
    setPatientAge("");
    setPatientSex("");
    setMedicalHistory("");
    setCurrentMedications("");
    setAdditionalNotes("");
    setResult(null);
  };

  return (
    <div className={`grid gap-6 lg:grid-cols-2 ${className}`}>
      <Card data-testid="card-diagnosis-input">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Stethoscope className="h-5 w-5" />
            Clinical Assessment
          </CardTitle>
          <CardDescription>
            {patientName ? `Analyzing symptoms for ${patientName}` : 'Enter symptoms for AI-powered diagnosis support'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="patient-age">Patient Age</Label>
                <Input
                  id="patient-age"
                  placeholder="e.g., 45"
                  value={patientAge}
                  onChange={(e) => setPatientAge(e.target.value)}
                  data-testid="input-patient-age"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="patient-sex">Biological Sex</Label>
                <select
                  id="patient-sex"
                  value={patientSex}
                  onChange={(e) => setPatientSex(e.target.value)}
                  className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  data-testid="select-patient-sex"
                >
                  <option value="">Select...</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>
            </div>

            <Separator />

            <div className="space-y-3">
              <Label>Presenting Symptoms</Label>
              <div className="flex gap-2">
                <Input
                  placeholder="Enter symptom..."
                  value={newSymptom}
                  onChange={(e) => setNewSymptom(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && addSymptom()}
                  className="flex-1"
                  data-testid="input-new-symptom"
                />
                <Input
                  placeholder="Duration"
                  value={newDuration}
                  onChange={(e) => setNewDuration(e.target.value)}
                  className="w-28"
                  data-testid="input-symptom-duration"
                />
                <select
                  value={newSeverity}
                  onChange={(e) => setNewSeverity(e.target.value as 'mild' | 'moderate' | 'severe')}
                  className="flex h-9 w-24 rounded-md border border-input bg-background px-2 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  data-testid="select-symptom-severity"
                >
                  <option value="mild">Mild</option>
                  <option value="moderate">Moderate</option>
                  <option value="severe">Severe</option>
                </select>
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

            <div className="space-y-2">
              <Label htmlFor="medical-history">Relevant Medical History</Label>
              <Textarea
                id="medical-history"
                placeholder="Previous conditions, surgeries, family history..."
                value={medicalHistory}
                onChange={(e) => setMedicalHistory(e.target.value)}
                className="min-h-[80px]"
                data-testid="textarea-medical-history"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="medications">Current Medications</Label>
              <Input
                id="medications"
                placeholder="List current medications..."
                value={currentMedications}
                onChange={(e) => setCurrentMedications(e.target.value)}
                data-testid="input-medications"
              />
            </div>

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
          </div>
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button variant="outline" onClick={clearForm} data-testid="button-clear-diagnosis">
            Clear Form
          </Button>
          <Button
            onClick={handleAnalyze}
            disabled={symptoms.length === 0 || diagnosisMutation.isPending}
            data-testid="button-analyze-diagnosis"
          >
            {diagnosisMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Brain className="h-4 w-4 mr-2" />
                Analyze Symptoms
              </>
            )}
          </Button>
        </CardFooter>
      </Card>

      <Card data-testid="card-diagnosis-results">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            AI Diagnosis Support
          </CardTitle>
          <CardDescription>
            Clinical decision support (not a substitute for professional judgment)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[500px] pr-4">
            {diagnosisMutation.isPending ? (
              <div className="flex flex-col items-center justify-center py-16">
                <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
                <p className="text-muted-foreground">Analyzing symptoms...</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Cross-referencing with clinical knowledge base
                </p>
              </div>
            ) : result ? (
              <div className="space-y-6">
                {result.redFlags.length > 0 && (
                  <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20">
                    <div className="flex items-center gap-2 mb-3">
                      <AlertTriangle className="h-5 w-5 text-destructive" />
                      <span className="font-semibold text-destructive">Red Flags Identified</span>
                    </div>
                    <ul className="space-y-1 text-sm">
                      {result.redFlags.map((flag, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <AlertCircle className="h-4 w-4 text-destructive mt-0.5" />
                          <span>{flag}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-primary" />
                    <span className="font-semibold">Primary Diagnosis Suggestion</span>
                  </div>
                  
                  <div className="p-4 rounded-lg border bg-primary/5">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h4 className="font-semibold text-lg">{result.primaryDiagnosis.condition}</h4>
                        <p className="text-sm text-muted-foreground mt-1">
                          {result.primaryDiagnosis.description}
                        </p>
                      </div>
                      <Badge variant={getUrgencyColor(result.primaryDiagnosis.urgency) as any}>
                        {result.primaryDiagnosis.urgency} urgency
                      </Badge>
                    </div>
                    
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-sm text-muted-foreground">Confidence:</span>
                      <Progress value={result.primaryDiagnosis.probability} className="flex-1 h-2" />
                      <span className="text-sm font-medium">{result.primaryDiagnosis.probability}%</span>
                    </div>

                    {result.primaryDiagnosis.matchingSymptoms.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-2">
                        <span className="text-xs text-muted-foreground mr-1">Matching:</span>
                        {result.primaryDiagnosis.matchingSymptoms.map((s, idx) => (
                          <Badge key={idx} variant="secondary" className="text-xs">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            {s}
                          </Badge>
                        ))}
                      </div>
                    )}

                    {result.primaryDiagnosis.recommendedTests.length > 0 && (
                      <div className="mt-3 pt-3 border-t">
                        <span className="text-sm font-medium flex items-center gap-1 mb-2">
                          <ClipboardList className="h-4 w-4" />
                          Recommended Tests
                        </span>
                        <div className="flex flex-wrap gap-1">
                          {result.primaryDiagnosis.recommendedTests.map((test, idx) => (
                            <Badge key={idx} variant="outline">{test}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {result.differentialDiagnoses.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <FileText className="h-5 w-5 text-muted-foreground" />
                      <span className="font-semibold">Differential Diagnoses</span>
                    </div>
                    <div className="space-y-2">
                      {result.differentialDiagnoses.map((dx, idx) => (
                        <div key={idx} className="p-3 rounded-lg border">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-medium">{dx.condition}</span>
                            <div className="flex items-center gap-2">
                              <Progress value={dx.probability} className="w-16 h-2" />
                              <span className="text-sm text-muted-foreground">{dx.probability}%</span>
                            </div>
                          </div>
                          <p className="text-sm text-muted-foreground">{dx.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result.clinicalInsights.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Lightbulb className="h-5 w-5 text-yellow-500" />
                      <span className="font-semibold">Clinical Insights</span>
                    </div>
                    <ul className="space-y-2">
                      {result.clinicalInsights.map((insight, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm">
                          <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.recommendedActions.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <ClipboardList className="h-5 w-5 text-blue-500" />
                      <span className="font-semibold">Recommended Actions</span>
                    </div>
                    <ol className="space-y-2 list-decimal list-inside">
                      {result.recommendedActions.map((action, idx) => (
                        <li key={idx} className="text-sm">{action}</li>
                      ))}
                    </ol>
                  </div>
                )}

                {result.references.length > 0 && (
                  <div className="space-y-2 pt-4 border-t">
                    <div className="flex items-center gap-2">
                      <BookOpen className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium text-muted-foreground">References</span>
                    </div>
                    <ul className="text-xs text-muted-foreground space-y-1">
                      {result.references.map((ref, idx) => (
                        <li key={idx}>{ref}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="p-3 rounded-lg bg-muted/50 text-xs text-muted-foreground">
                  <AlertCircle className="h-4 w-4 inline mr-1" />
                  This AI analysis is for clinical decision support only. All diagnoses must be confirmed through proper clinical evaluation and diagnostic testing.
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-center text-muted-foreground">
                <Brain className="h-16 w-16 mb-4 opacity-30" />
                <p className="text-lg font-medium mb-2">Ready to Analyze</p>
                <p className="text-sm max-w-md">
                  Enter patient symptoms and clinical information on the left panel, then click "Analyze Symptoms" to receive AI-powered diagnosis suggestions.
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
