import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Checkbox } from '@/components/ui/checkbox';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { apiRequest, queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { 
  Brain, Play, BarChart3, TrendingUp, Activity, Users,
  Settings2, FileJson, Clock, CheckCircle, AlertCircle,
  Loader2, ChevronRight, Sparkles, Target, GitBranch
} from 'lucide-react';

interface AnalysisConfig {
  type: 'descriptive' | 'risk_prediction' | 'survival' | 'causal';
  cohortId: string | null;
  outcomeVariable: string;
  exposureVariable?: string;
  covariates: string[];
  modelConfig: {
    algorithm?: string;
    crossValidation?: boolean;
    folds?: number;
    testSize?: number;
    randomSeed?: number;
  };
  title: string;
  description: string;
}

interface AnalysisJob {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  resultsJson?: any;
  createdAt: string;
  completedAt?: string;
  errorMessage?: string;
}

const ANALYSIS_TYPES = [
  {
    id: 'descriptive',
    label: 'Descriptive Statistics',
    icon: BarChart3,
    description: 'Generate baseline characteristics tables with means, SDs, medians, and proportions',
    color: 'text-blue-500',
  },
  {
    id: 'risk_prediction',
    label: 'Risk Prediction',
    icon: Target,
    description: 'Build predictive models with logistic regression, random forest, and feature importance',
    color: 'text-purple-500',
  },
  {
    id: 'survival',
    label: 'Survival Analysis',
    icon: TrendingUp,
    description: 'Generate Kaplan-Meier curves and Cox proportional hazards models',
    color: 'text-emerald-500',
  },
  {
    id: 'causal',
    label: 'Causal Inference',
    icon: GitBranch,
    description: 'Estimate treatment effects using propensity score matching and IPTW',
    color: 'text-amber-500',
  },
];

const AVAILABLE_VARIABLES = [
  { id: 'age', label: 'Age', type: 'continuous' },
  { id: 'sex', label: 'Sex', type: 'categorical' },
  { id: 'bmi', label: 'BMI', type: 'continuous' },
  { id: 'condition', label: 'Primary Condition', type: 'categorical' },
  { id: 'risk_score', label: 'Composite Risk Score', type: 'continuous' },
  { id: 'deterioration_index', label: 'Deterioration Index', type: 'continuous' },
  { id: 'medication_adherence', label: 'Medication Adherence %', type: 'continuous' },
  { id: 'hospitalization', label: 'Hospitalization Event', type: 'binary' },
  { id: 'mortality', label: 'Mortality Event', type: 'binary' },
  { id: 'readmission', label: '30-Day Readmission', type: 'binary' },
  { id: 'immunosuppressant', label: 'On Immunosuppressant', type: 'binary' },
  { id: 'followup_compliance', label: 'Follow-up Compliance', type: 'continuous' },
  { id: 'environmental_exposure', label: 'Environmental Risk Score', type: 'continuous' },
  { id: 'immune_status', label: 'Immune Status Score', type: 'continuous' },
];

const ALGORITHMS = {
  risk_prediction: [
    { id: 'logistic_regression', label: 'Logistic Regression' },
    { id: 'random_forest', label: 'Random Forest' },
    { id: 'gradient_boosting', label: 'Gradient Boosting (LightGBM)' },
    { id: 'xgboost', label: 'XGBoost' },
  ],
  survival: [
    { id: 'kaplan_meier', label: 'Kaplan-Meier' },
    { id: 'cox_ph', label: 'Cox Proportional Hazards' },
    { id: 'random_survival_forest', label: 'Random Survival Forest' },
  ],
  causal: [
    { id: 'psm', label: 'Propensity Score Matching' },
    { id: 'iptw', label: 'Inverse Probability Weighting' },
    { id: 'doubly_robust', label: 'Doubly Robust Estimation' },
  ],
};

export function AIAnalysisTab() {
  const { toast } = useToast();
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [config, setConfig] = useState<AnalysisConfig>({
    type: 'descriptive',
    cohortId: null,
    outcomeVariable: '',
    exposureVariable: '',
    covariates: [],
    modelConfig: {
      algorithm: 'logistic_regression',
      crossValidation: true,
      folds: 5,
      testSize: 0.2,
      randomSeed: 42,
    },
    title: '',
    description: '',
  });

  const { data: cohorts } = useQuery<any[]>({
    queryKey: ['/api/v1/research-center/cohorts'],
  });

  const { data: analysisJobs = [], isLoading: jobsLoading } = useQuery<AnalysisJob[]>({
    queryKey: ['/api/v1/research-center/analysis-jobs'],
  });

  const runAnalysisMutation = useMutation({
    mutationFn: async (analysisConfig: AnalysisConfig) => {
      const response = await apiRequest('POST', '/api/v1/research-center/analysis-jobs', {
        analysisType: analysisConfig.type,
        cohortId: analysisConfig.cohortId,
        analysisSpec: {
          outcome: analysisConfig.outcomeVariable,
          exposure: analysisConfig.exposureVariable,
          covariates: analysisConfig.covariates,
          modelConfig: analysisConfig.modelConfig,
        },
        title: analysisConfig.title,
        description: analysisConfig.description,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/analysis-jobs'] });
      toast({ title: 'Analysis Started', description: 'Your analysis job has been queued.' });
      setSelectedType(null);
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const toggleCovariate = (varId: string) => {
    setConfig(prev => ({
      ...prev,
      covariates: prev.covariates.includes(varId)
        ? prev.covariates.filter(c => c !== varId)
        : [...prev.covariates, varId],
    }));
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge className="bg-emerald-500"><CheckCircle className="h-3 w-3 mr-1" />Completed</Badge>;
      case 'running':
        return <Badge className="bg-blue-500"><Loader2 className="h-3 w-3 mr-1 animate-spin" />Running</Badge>;
      case 'failed':
        return <Badge variant="destructive"><AlertCircle className="h-3 w-3 mr-1" />Failed</Badge>;
      default:
        return <Badge variant="secondary"><Clock className="h-3 w-3 mr-1" />Pending</Badge>;
    }
  };

  const renderAnalysisConfig = () => {
    const analysisType = ANALYSIS_TYPES.find(t => t.id === selectedType);
    if (!analysisType) return null;

    const Icon = analysisType.icon;

    return (
      <Card data-testid="card-analysis-config">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className={`flex h-10 w-10 items-center justify-center rounded-full bg-accent ${analysisType.color}`}>
              <Icon className="h-5 w-5" />
            </div>
            <div>
              <CardTitle>{analysisType.label}</CardTitle>
              <CardDescription>{analysisType.description}</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="analysis-title">Analysis Title</Label>
              <Input
                id="analysis-title"
                placeholder="e.g., Transplant Outcomes Risk Prediction"
                value={config.title}
                onChange={(e) => setConfig({ ...config, title: e.target.value })}
                data-testid="input-analysis-title"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="cohort-select">Select Cohort</Label>
              <Select
                value={config.cohortId || ''}
                onValueChange={(v) => setConfig({ ...config, cohortId: v })}
              >
                <SelectTrigger id="cohort-select" data-testid="select-cohort">
                  <SelectValue placeholder="Choose a cohort" />
                </SelectTrigger>
                <SelectContent>
                  {cohorts?.map((cohort) => (
                    <SelectItem key={cohort.id} value={cohort.id}>
                      {cohort.name} ({cohort.patientCount || 0} patients)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              placeholder="Describe the analysis objectives..."
              value={config.description}
              onChange={(e) => setConfig({ ...config, description: e.target.value })}
              data-testid="input-analysis-description"
            />
          </div>

          <Separator />

          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="variables">
              <AccordionTrigger>
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Variable Selection
                </div>
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Outcome Variable</Label>
                    <Select
                      value={config.outcomeVariable}
                      onValueChange={(v) => setConfig({ ...config, outcomeVariable: v })}
                    >
                      <SelectTrigger data-testid="select-outcome-variable">
                        <SelectValue placeholder="Select outcome" />
                      </SelectTrigger>
                      <SelectContent>
                        {AVAILABLE_VARIABLES.filter(v => 
                          v.type === 'binary' || v.type === 'continuous'
                        ).map((v) => (
                          <SelectItem key={v.id} value={v.id}>
                            {v.label}
                            <Badge variant="outline" className="ml-2 text-xs">{v.type}</Badge>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {(selectedType === 'causal' || selectedType === 'survival') && (
                    <div className="space-y-2">
                      <Label>Exposure/Treatment Variable</Label>
                      <Select
                        value={config.exposureVariable}
                        onValueChange={(v) => setConfig({ ...config, exposureVariable: v })}
                      >
                        <SelectTrigger data-testid="select-exposure-variable">
                          <SelectValue placeholder="Select exposure" />
                        </SelectTrigger>
                        <SelectContent>
                          {AVAILABLE_VARIABLES.filter(v => v.type === 'binary').map((v) => (
                            <SelectItem key={v.id} value={v.id}>
                              {v.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <Label>Covariates</Label>
                  <p className="text-xs text-muted-foreground mb-2">
                    Select variables to include as covariates in the analysis
                  </p>
                  <div className="grid gap-2 md:grid-cols-3">
                    {AVAILABLE_VARIABLES.filter(v => 
                      v.id !== config.outcomeVariable && v.id !== config.exposureVariable
                    ).map((variable) => (
                      <div key={variable.id} className="flex items-center space-x-2">
                        <Checkbox
                          id={`covar-${variable.id}`}
                          checked={config.covariates.includes(variable.id)}
                          onCheckedChange={() => toggleCovariate(variable.id)}
                          data-testid={`checkbox-covariate-${variable.id}`}
                        />
                        <Label htmlFor={`covar-${variable.id}`} className="text-sm cursor-pointer">
                          {variable.label}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            {selectedType !== 'descriptive' && (
              <AccordionItem value="model-config">
                <AccordionTrigger>
                  <div className="flex items-center gap-2">
                    <Settings2 className="h-4 w-4" />
                    Model Configuration
                  </div>
                </AccordionTrigger>
                <AccordionContent className="space-y-4">
                  {selectedType && ALGORITHMS[selectedType as keyof typeof ALGORITHMS] && (
                    <div className="space-y-2">
                      <Label>Algorithm</Label>
                      <Select
                        value={config.modelConfig.algorithm}
                        onValueChange={(v) => setConfig({
                          ...config,
                          modelConfig: { ...config.modelConfig, algorithm: v }
                        })}
                      >
                        <SelectTrigger data-testid="select-algorithm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {ALGORITHMS[selectedType as keyof typeof ALGORITHMS]?.map((algo) => (
                            <SelectItem key={algo.id} value={algo.id}>
                              {algo.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  <div className="flex items-center justify-between p-4 border rounded-lg">
                    <div>
                      <Label>Cross-Validation</Label>
                      <p className="text-xs text-muted-foreground">
                        Use k-fold cross-validation for robust performance estimates
                      </p>
                    </div>
                    <div className="flex items-center gap-4">
                      {config.modelConfig.crossValidation && (
                        <div className="flex items-center gap-2">
                          <Label className="text-xs">Folds:</Label>
                          <Select
                            value={String(config.modelConfig.folds)}
                            onValueChange={(v) => setConfig({
                              ...config,
                              modelConfig: { ...config.modelConfig, folds: parseInt(v) }
                            })}
                          >
                            <SelectTrigger className="w-20" data-testid="select-cv-folds">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="3">3</SelectItem>
                              <SelectItem value="5">5</SelectItem>
                              <SelectItem value="10">10</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                      <Switch
                        checked={config.modelConfig.crossValidation}
                        onCheckedChange={(c) => setConfig({
                          ...config,
                          modelConfig: { ...config.modelConfig, crossValidation: c }
                        })}
                        data-testid="switch-cross-validation"
                      />
                    </div>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Test Set Size</Label>
                      <Select
                        value={String(config.modelConfig.testSize)}
                        onValueChange={(v) => setConfig({
                          ...config,
                          modelConfig: { ...config.modelConfig, testSize: parseFloat(v) }
                        })}
                      >
                        <SelectTrigger data-testid="select-test-size">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0.1">10%</SelectItem>
                          <SelectItem value="0.2">20%</SelectItem>
                          <SelectItem value="0.3">30%</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Random Seed</Label>
                      <Input
                        type="number"
                        value={config.modelConfig.randomSeed}
                        onChange={(e) => setConfig({
                          ...config,
                          modelConfig: { ...config.modelConfig, randomSeed: parseInt(e.target.value) || 42 }
                        })}
                        data-testid="input-random-seed"
                      />
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            )}
          </Accordion>

          <div className="flex justify-end gap-2 pt-4">
            <Button variant="outline" onClick={() => setSelectedType(null)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                setConfig(prev => ({ ...prev, type: selectedType as any }));
                runAnalysisMutation.mutate({ ...config, type: selectedType as any });
              }}
              disabled={!config.title || !config.cohortId || runAnalysisMutation.isPending}
              data-testid="button-run-analysis"
            >
              {runAnalysisMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Analysis
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6" data-testid="ai-analysis-tab">
      <Tabs defaultValue="new" className="w-full">
        <TabsList>
          <TabsTrigger value="new" data-testid="tab-new-analysis">
            <Sparkles className="h-4 w-4 mr-2" />
            New Analysis
          </TabsTrigger>
          <TabsTrigger value="jobs" data-testid="tab-analysis-jobs">
            <Clock className="h-4 w-4 mr-2" />
            Analysis Jobs
            {analysisJobs && analysisJobs.length > 0 && (
              <Badge variant="secondary" className="ml-2">{analysisJobs.length}</Badge>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="new" className="space-y-4 pt-4">
          {selectedType ? (
            renderAnalysisConfig()
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {ANALYSIS_TYPES.map((type) => {
                const Icon = type.icon;
                return (
                  <Card 
                    key={type.id} 
                    className="hover-elevate cursor-pointer"
                    onClick={() => setSelectedType(type.id)}
                    data-testid={`card-analysis-type-${type.id}`}
                  >
                    <CardContent className="pt-6">
                      <div className="flex items-start gap-4">
                        <div className={`flex h-12 w-12 items-center justify-center rounded-full bg-accent ${type.color}`}>
                          <Icon className="h-6 w-6" />
                        </div>
                        <div className="flex-1">
                          <h3 className="font-semibold">{type.label}</h3>
                          <p className="text-sm text-muted-foreground mt-1">
                            {type.description}
                          </p>
                          <Button variant="link" className="p-0 h-auto mt-2 text-sm">
                            Configure <ChevronRight className="h-4 w-4 ml-1" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </TabsContent>

        <TabsContent value="jobs" className="pt-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <FileJson className="h-4 w-4" />
                Analysis History
              </CardTitle>
              <CardDescription>
                View status and results of previous analysis jobs
              </CardDescription>
            </CardHeader>
            <CardContent>
              {jobsLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-16 bg-muted animate-pulse rounded-lg" />
                  ))}
                </div>
              ) : analysisJobs && analysisJobs.length > 0 ? (
                <ScrollArea className="h-[400px]">
                  <div className="space-y-3">
                    {analysisJobs.map((job) => (
                      <div 
                        key={job.id} 
                        className="flex items-center justify-between p-4 border rounded-lg hover-elevate"
                        data-testid={`job-${job.id}`}
                      >
                        <div className="flex items-center gap-3">
                          <Brain className="h-5 w-5 text-muted-foreground" />
                          <div>
                            <p className="font-medium">{job.type}</p>
                            <p className="text-xs text-muted-foreground">
                              {new Date(job.createdAt).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          {job.status === 'running' && job.progress !== undefined && (
                            <Progress value={job.progress} className="w-24 h-2" />
                          )}
                          {getStatusBadge(job.status)}
                          {job.status === 'completed' && (
                            <Button variant="ghost" size="sm" data-testid={`button-view-results-${job.id}`}>
                              View Results
                            </Button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-12">
                  <Brain className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
                  <h3 className="text-sm font-medium mb-2">No Analysis Jobs</h3>
                  <p className="text-xs text-muted-foreground">
                    Run your first analysis to see results here
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
