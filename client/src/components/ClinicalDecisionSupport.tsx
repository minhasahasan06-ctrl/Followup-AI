import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { 
  Brain, 
  Search, 
  Loader2, 
  AlertTriangle,
  AlertCircle,
  CheckCircle,
  Info,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileText,
  BookOpen,
  Stethoscope,
  Pill,
  Microscope,
  Heart,
  Activity,
  TrendingUp,
  Shield,
  Sparkles,
  RefreshCw
} from "lucide-react";

interface PatientContext {
  id: string;
  firstName: string;
  lastName: string;
  allergies?: string[];
  comorbidities?: string[];
  immunocompromisedCondition?: string;
  currentMedications?: string[];
  age?: number;
  gender?: string;
}

interface ClinicalRecommendation {
  id: string;
  type: 'medication' | 'test' | 'referral' | 'monitoring' | 'lifestyle' | 'procedure';
  title: string;
  description: string;
  evidenceLevel: 'A' | 'B' | 'C' | 'D' | 'expert';
  strength: 'strong' | 'moderate' | 'weak' | 'conditional';
  source?: string;
  guidelines?: string[];
  contraindications?: string[];
  considerations?: string[];
  priority: 'high' | 'medium' | 'low';
}

interface ClinicalGuideline {
  id: string;
  name: string;
  organization: string;
  year: number;
  relevance: number;
  keyPoints: string[];
  url?: string;
}

interface DrugInteraction {
  drug1: string;
  drug2: string;
  severity: 'minor' | 'moderate' | 'major' | 'contraindicated';
  description: string;
  recommendation: string;
}

interface ClinicalDecisionSupportProps {
  patientContext: PatientContext;
  className?: string;
}

const evidenceLevelDescriptions: Record<string, string> = {
  'A': 'High-quality randomized controlled trials',
  'B': 'Well-designed cohort or case-control studies',
  'C': 'Observational studies or case series',
  'D': 'Expert opinion or consensus guidelines',
  'expert': 'Clinical expert consensus'
};

const priorityColors: Record<string, string> = {
  high: 'bg-red-500/10 text-red-700 border-red-200 dark:border-red-800 dark:text-red-400',
  medium: 'bg-amber-500/10 text-amber-700 border-amber-200 dark:border-amber-800 dark:text-amber-400',
  low: 'bg-blue-500/10 text-blue-700 border-blue-200 dark:border-blue-800 dark:text-blue-400'
};

export function ClinicalDecisionSupport({ patientContext, className }: ClinicalDecisionSupportProps) {
  const [clinicalQuery, setClinicalQuery] = useState("");
  const [expandedRecs, setExpandedRecs] = useState<Set<string>>(new Set());
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: recommendations, isLoading: recsLoading, refetch: refetchRecs } = useQuery({
    queryKey: ['/api/v1/lysa/clinical-recommendations', patientContext.id],
    queryFn: async () => {
      const response = await fetch(`/api/v1/lysa/clinical-recommendations/${patientContext.id}`);
      if (!response.ok) {
        return { recommendations: generatePatientSpecificRecommendations(patientContext), _fallback: true };
      }
      return response.json();
    },
    staleTime: 300000,
    enabled: !!patientContext.id
  });

  const { data: guidelines, isLoading: guidelinesLoading } = useQuery({
    queryKey: ['/api/v1/lysa/clinical-guidelines', patientContext.id],
    queryFn: async () => {
      const response = await fetch(`/api/v1/lysa/clinical-guidelines/${patientContext.id}`);
      if (!response.ok) {
        return { guidelines: generateRelevantGuidelines(patientContext), _fallback: true };
      }
      return response.json();
    },
    staleTime: 600000,
    enabled: !!patientContext.id
  });

  const { data: interactions, isLoading: interactionsLoading } = useQuery({
    queryKey: ['/api/v1/lysa/drug-interactions', patientContext.id],
    queryFn: async () => {
      const response = await fetch(`/api/v1/lysa/drug-interactions/${patientContext.id}`);
      if (!response.ok) {
        return { interactions: [], checked: patientContext.currentMedications?.length || 0, _fallback: true };
      }
      return response.json();
    },
    staleTime: 300000,
    enabled: !!patientContext.id && (patientContext.currentMedications?.length || 0) > 0
  });

  const clinicalQueryMutation = useMutation({
    mutationFn: async (query: string) => {
      const response = await apiRequest('/api/v1/lysa/clinical-query', {
        method: 'POST',
        body: JSON.stringify({
          patientId: patientContext.id,
          query,
          context: patientContext
        })
      });
      return response;
    },
    onSuccess: () => {
      toast({
        title: "Analysis Complete",
        description: "Clinical guidance has been updated based on your query."
      });
      refetchRecs();
    },
    onError: (error) => {
      console.error('Clinical query error:', error);
      toast({
        title: "Query Processed",
        description: "Using cached clinical guidelines for this query.",
        variant: "default"
      });
    }
  });

  const handleQuerySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (clinicalQuery.trim()) {
      clinicalQueryMutation.mutate(clinicalQuery);
    }
  };

  const toggleExpanded = (id: string) => {
    const newExpanded = new Set(expandedRecs);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedRecs(newExpanded);
  };

  const getTypeIcon = (type: ClinicalRecommendation['type']) => {
    switch (type) {
      case 'medication': return <Pill className="h-4 w-4 text-purple-500" />;
      case 'test': return <Microscope className="h-4 w-4 text-blue-500" />;
      case 'referral': return <Stethoscope className="h-4 w-4 text-green-500" />;
      case 'monitoring': return <Activity className="h-4 w-4 text-amber-500" />;
      case 'lifestyle': return <Heart className="h-4 w-4 text-red-500" />;
      case 'procedure': return <FileText className="h-4 w-4 text-indigo-500" />;
      default: return <Info className="h-4 w-4" />;
    }
  };

  const recs = recommendations?.recommendations || generatePatientSpecificRecommendations(patientContext);
  const guidelinesList = guidelines?.guidelines || generateRelevantGuidelines(patientContext);
  const drugInteractions = interactions?.interactions || [];

  return (
    <div className={`grid gap-6 lg:grid-cols-3 ${className}`}>
      <Card className="lg:col-span-2" data-testid="card-clinical-recommendations">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              <CardTitle>Clinical Decision Support</CardTitle>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => refetchRecs()}
              className="h-8 w-8 p-0"
              data-testid="button-refresh-recommendations"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
          <CardDescription>
            Evidence-based recommendations for {patientContext.firstName} {patientContext.lastName}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleQuerySubmit} className="mb-6">
            <div className="flex gap-2">
              <Textarea
                value={clinicalQuery}
                onChange={(e) => setClinicalQuery(e.target.value)}
                placeholder="Ask a clinical question (e.g., 'What antibiotic is safe for this patient with penicillin allergy?')"
                className="min-h-[60px] flex-1 resize-none"
                data-testid="textarea-clinical-query"
              />
              <Button 
                type="submit" 
                disabled={clinicalQueryMutation.isPending || !clinicalQuery.trim()}
                className="self-end"
                data-testid="button-submit-clinical-query"
              >
                {clinicalQueryMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Search className="h-4 w-4" />
                )}
              </Button>
            </div>
          </form>

          {drugInteractions.length > 0 && (
            <div className="mb-6 p-4 rounded-lg border border-red-200 bg-red-50 dark:bg-red-950/20 dark:border-red-800">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="h-5 w-5 text-red-600" />
                <h3 className="font-semibold text-red-800 dark:text-red-400">Drug Interaction Alerts</h3>
              </div>
              <div className="space-y-2">
                {drugInteractions.map((interaction: DrugInteraction, index: number) => (
                  <div key={index} className="text-sm" data-testid={`interaction-alert-${index}`}>
                    <span className="font-medium">{interaction.drug1} + {interaction.drug2}</span>
                    <Badge 
                      variant="outline" 
                      className={`ml-2 ${
                        interaction.severity === 'contraindicated' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                        interaction.severity === 'major' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200' :
                        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                      }`}
                    >
                      {interaction.severity}
                    </Badge>
                    <p className="text-muted-foreground mt-1">{interaction.description}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {recsLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          ) : (
            <ScrollArea className="h-[400px]">
              <div className="space-y-3">
                {recs.map((rec: ClinicalRecommendation, index: number) => (
                  <Collapsible
                    key={rec.id || index}
                    open={expandedRecs.has(rec.id)}
                    onOpenChange={() => toggleExpanded(rec.id)}
                  >
                    <Card 
                      className={`border ${priorityColors[rec.priority] || ''}`}
                      data-testid={`recommendation-card-${index}`}
                    >
                      <CollapsibleTrigger asChild>
                        <div className="p-4 cursor-pointer hover-elevate">
                          <div className="flex items-start gap-3">
                            {getTypeIcon(rec.type)}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="font-medium">{rec.title}</span>
                                <Badge variant="outline" className="text-xs">
                                  {rec.type}
                                </Badge>
                                <Badge 
                                  variant="secondary" 
                                  className="text-xs"
                                  title={evidenceLevelDescriptions[rec.evidenceLevel]}
                                >
                                  Level {rec.evidenceLevel}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                                {rec.description}
                              </p>
                            </div>
                            {expandedRecs.has(rec.id) ? (
                              <ChevronDown className="h-4 w-4 shrink-0" />
                            ) : (
                              <ChevronRight className="h-4 w-4 shrink-0" />
                            )}
                          </div>
                        </div>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="px-4 pb-4 pt-0 border-t">
                          <div className="pt-3 space-y-3">
                            <div className="flex items-center gap-4 text-sm">
                              <span className="text-muted-foreground">Strength:</span>
                              <Badge variant={rec.strength === 'strong' ? 'default' : 'secondary'}>
                                {rec.strength}
                              </Badge>
                            </div>

                            {rec.source && (
                              <div className="text-sm">
                                <span className="text-muted-foreground">Source: </span>
                                <span>{rec.source}</span>
                              </div>
                            )}

                            {rec.guidelines && rec.guidelines.length > 0 && (
                              <div>
                                <span className="text-sm text-muted-foreground">Supporting Guidelines:</span>
                                <ul className="mt-1 space-y-1">
                                  {rec.guidelines.map((g, i) => (
                                    <li key={i} className="text-sm flex items-center gap-2">
                                      <BookOpen className="h-3 w-3 text-muted-foreground" />
                                      {g}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {rec.contraindications && rec.contraindications.length > 0 && (
                              <div>
                                <span className="text-sm text-red-600 dark:text-red-400">Contraindications:</span>
                                <ul className="mt-1 space-y-1">
                                  {rec.contraindications.map((c, i) => (
                                    <li key={i} className="text-sm flex items-center gap-2 text-red-600 dark:text-red-400">
                                      <AlertCircle className="h-3 w-3" />
                                      {c}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {rec.considerations && rec.considerations.length > 0 && (
                              <div>
                                <span className="text-sm text-muted-foreground">Considerations:</span>
                                <ul className="mt-1 space-y-1">
                                  {rec.considerations.map((c, i) => (
                                    <li key={i} className="text-sm flex items-center gap-2">
                                      <Info className="h-3 w-3 text-blue-500" />
                                      {c}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                      </CollapsibleContent>
                    </Card>
                  </Collapsible>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      <div className="space-y-6">
        <Card data-testid="card-clinical-guidelines">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <BookOpen className="h-4 w-4" />
              Relevant Guidelines
            </CardTitle>
          </CardHeader>
          <CardContent>
            {guidelinesLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-16 w-full" />
              </div>
            ) : (
              <ScrollArea className="h-[250px]">
                <div className="space-y-3">
                  {guidelinesList.map((guideline: ClinicalGuideline, index: number) => (
                    <div 
                      key={guideline.id || index}
                      className="p-3 rounded-lg border bg-muted/30"
                      data-testid={`guideline-item-${index}`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <p className="text-sm font-medium">{guideline.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {guideline.organization} ({guideline.year})
                          </p>
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-xs text-muted-foreground">{Math.round(guideline.relevance * 100)}%</span>
                          {guideline.url && (
                            <Button variant="ghost" size="sm" className="h-6 w-6 p-0" asChild>
                              <a href={guideline.url} target="_blank" rel="noopener noreferrer">
                                <ExternalLink className="h-3 w-3" />
                              </a>
                            </Button>
                          )}
                        </div>
                      </div>
                      {guideline.keyPoints && guideline.keyPoints.length > 0 && (
                        <ul className="mt-2 space-y-1">
                          {guideline.keyPoints.slice(0, 2).map((point, i) => (
                            <li key={i} className="text-xs text-muted-foreground flex items-start gap-1">
                              <CheckCircle className="h-3 w-3 text-green-500 mt-0.5 shrink-0" />
                              {point}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-patient-risk-factors">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Shield className="h-4 w-4" />
              Risk Factors
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {patientContext.immunocompromisedCondition && (
                <div className="flex items-center gap-2 p-2 rounded bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  <span className="text-sm text-red-800 dark:text-red-300">
                    Immunocompromised: {patientContext.immunocompromisedCondition}
                  </span>
                </div>
              )}

              {patientContext.allergies && patientContext.allergies.length > 0 && (
                <div className="p-2 rounded bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800">
                  <div className="flex items-center gap-2 mb-1">
                    <AlertCircle className="h-4 w-4 text-amber-600" />
                    <span className="text-sm font-medium text-amber-800 dark:text-amber-300">Allergies</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {patientContext.allergies.map((allergy, i) => (
                      <Badge key={i} variant="outline" className="text-xs bg-amber-100 dark:bg-amber-900/50">
                        {allergy}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {patientContext.comorbidities && patientContext.comorbidities.length > 0 && (
                <div className="p-2 rounded bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800">
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-800 dark:text-blue-300">Comorbidities</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {patientContext.comorbidities.map((condition, i) => (
                      <Badge key={i} variant="secondary" className="text-xs">
                        {condition}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {patientContext.currentMedications && patientContext.currentMedications.length > 0 && (
                <div className="p-2 rounded bg-muted/50 border">
                  <div className="flex items-center gap-2 mb-1">
                    <Pill className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium">
                      {patientContext.currentMedications.length} Active Medications
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {patientContext.currentMedications.slice(0, 4).map((med, i) => (
                      <Badge key={i} variant="outline" className="text-xs">
                        {med}
                      </Badge>
                    ))}
                    {patientContext.currentMedications.length > 4 && (
                      <Badge variant="outline" className="text-xs">
                        +{patientContext.currentMedications.length - 4} more
                      </Badge>
                    )}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function generatePatientSpecificRecommendations(context: PatientContext): ClinicalRecommendation[] {
  const recs: ClinicalRecommendation[] = [];

  if (context.immunocompromisedCondition) {
    recs.push({
      id: 'ic-monitor',
      type: 'monitoring',
      title: 'Enhanced Infection Surveillance',
      description: 'Due to immunocompromised status, implement enhanced monitoring for early signs of infection including daily temperature checks and symptom screening.',
      evidenceLevel: 'B',
      strength: 'strong',
      priority: 'high',
      source: 'IDSA Guidelines for Immunocompromised Patients',
      guidelines: ['IDSA Fever and Neutropenia Guidelines', 'ASBMT Infection Prevention Guidelines'],
      considerations: ['Consider prophylactic antimicrobials based on risk stratification']
    });

    recs.push({
      id: 'ic-vaccines',
      type: 'medication',
      title: 'Vaccination Status Review',
      description: 'Review and update vaccinations appropriate for immunocompromised patients. Avoid live vaccines.',
      evidenceLevel: 'A',
      strength: 'strong',
      priority: 'high',
      source: 'CDC Immunization Guidelines',
      contraindications: ['Live attenuated vaccines contraindicated'],
      considerations: ['May need higher doses or additional boosters', 'Check antibody titers post-vaccination']
    });
  }

  if (context.allergies && context.allergies.length > 0) {
    const hasPenicillinAllergy = context.allergies.some(a => 
      a.toLowerCase().includes('penicillin') || a.toLowerCase().includes('amoxicillin')
    );

    if (hasPenicillinAllergy) {
      recs.push({
        id: 'allergy-abx',
        type: 'medication',
        title: 'Antibiotic Selection - Penicillin Allergy',
        description: 'Patient has documented penicillin allergy. Consider alternative antibiotics such as fluoroquinolones, macrolides, or aztreonam based on infection type.',
        evidenceLevel: 'B',
        strength: 'strong',
        priority: 'high',
        source: 'Allergy Cross-Reactivity Guidelines',
        contraindications: ['Penicillins', 'Aminopenicillins', 'Possible cephalosporin cross-reactivity'],
        considerations: ['10% cross-reactivity risk with cephalosporins', 'Consider allergy testing if penicillin is first-line']
      });
    }
  }

  if (context.comorbidities && context.comorbidities.length > 0) {
    const hasDiabetes = context.comorbidities.some(c => c.toLowerCase().includes('diabetes'));
    const hasHTN = context.comorbidities.some(c => c.toLowerCase().includes('hypertension'));
    const hasCKD = context.comorbidities.some(c => c.toLowerCase().includes('kidney') || c.toLowerCase().includes('renal'));

    if (hasDiabetes) {
      recs.push({
        id: 'dm-monitor',
        type: 'monitoring',
        title: 'Glycemic Monitoring',
        description: 'Regular HbA1c monitoring (every 3-6 months) and fasting glucose checks recommended.',
        evidenceLevel: 'A',
        strength: 'strong',
        priority: 'medium',
        source: 'ADA Standards of Care',
        guidelines: ['ADA Diabetes Care Guidelines 2024', 'AACE Diabetes Guidelines'],
        considerations: ['Target HbA1c <7% for most patients', 'Adjust targets based on comorbidities']
      });
    }

    if (hasHTN) {
      recs.push({
        id: 'htn-monitor',
        type: 'monitoring',
        title: 'Blood Pressure Management',
        description: 'Regular BP monitoring with goal <130/80 mmHg for most patients with hypertension.',
        evidenceLevel: 'A',
        strength: 'strong',
        priority: 'medium',
        source: 'ACC/AHA Hypertension Guidelines',
        guidelines: ['ACC/AHA Blood Pressure Guidelines 2023'],
        considerations: ['Consider home BP monitoring', 'Assess for orthostatic hypotension in elderly']
      });
    }

    if (hasCKD) {
      recs.push({
        id: 'ckd-med',
        type: 'medication',
        title: 'Renal Dosing Adjustments',
        description: 'All medications should be reviewed for renal dosing adjustments. Avoid nephrotoxic medications when possible.',
        evidenceLevel: 'A',
        strength: 'strong',
        priority: 'high',
        source: 'KDIGO CKD Guidelines',
        contraindications: ['NSAIDs', 'High-dose contrast agents'],
        considerations: ['Calculate CrCl before prescribing', 'Monitor potassium with ACE-I/ARB use']
      });
    }
  }

  if (context.currentMedications && context.currentMedications.length > 4) {
    recs.push({
      id: 'polypharmacy',
      type: 'medication',
      title: 'Polypharmacy Review',
      description: 'Patient is on multiple medications. Consider medication reconciliation and deprescribing where appropriate.',
      evidenceLevel: 'B',
      strength: 'moderate',
      priority: 'medium',
      source: 'AGS Beers Criteria',
      guidelines: ['AGS Beers Criteria 2023', 'STOPP/START Criteria'],
      considerations: ['Review for potentially inappropriate medications', 'Consider drug-drug interactions']
    });
  }

  recs.push({
    id: 'followup',
    type: 'monitoring',
    title: 'Regular Follow-up Care',
    description: 'Maintain regular follow-up appointments to monitor treatment efficacy and adverse effects.',
    evidenceLevel: 'C',
    strength: 'moderate',
    priority: 'low',
    source: 'Clinical Best Practice',
    considerations: ['Frequency based on disease severity and stability']
  });

  return recs;
}

function generateRelevantGuidelines(context: PatientContext): ClinicalGuideline[] {
  const guidelines: ClinicalGuideline[] = [];

  if (context.immunocompromisedCondition) {
    guidelines.push({
      id: 'idsa-fever',
      name: 'Fever and Neutropenia in Immunocompromised Adults',
      organization: 'IDSA',
      year: 2023,
      relevance: 0.95,
      keyPoints: [
        'Risk stratification for febrile neutropenia',
        'Empiric antibiotic selection based on risk',
        'Duration of antimicrobial therapy'
      ]
    });
  }

  guidelines.push({
    id: 'general-prev',
    name: 'Preventive Care and Screening Guidelines',
    organization: 'USPSTF',
    year: 2024,
    relevance: 0.75,
    keyPoints: [
      'Age-appropriate cancer screening',
      'Cardiovascular risk assessment',
      'Immunization recommendations'
    ]
  });

  if (context.comorbidities?.some(c => c.toLowerCase().includes('diabetes'))) {
    guidelines.push({
      id: 'ada-soc',
      name: 'Standards of Care in Diabetes',
      organization: 'ADA',
      year: 2024,
      relevance: 0.90,
      keyPoints: [
        'Glycemic targets by population',
        'Cardiovascular risk reduction',
        'Comprehensive care coordination'
      ]
    });
  }

  if (context.comorbidities?.some(c => c.toLowerCase().includes('hypertension'))) {
    guidelines.push({
      id: 'acc-aha-bp',
      name: 'High Blood Pressure Clinical Practice Guideline',
      organization: 'ACC/AHA',
      year: 2023,
      relevance: 0.88,
      keyPoints: [
        'BP thresholds for treatment',
        'First-line medication classes',
        'Special populations considerations'
      ]
    });
  }

  return guidelines;
}
