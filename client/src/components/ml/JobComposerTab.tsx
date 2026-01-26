import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Checkbox } from '@/components/ui/checkbox';
import { 
  Wand2, 
  Cpu, 
  Database, 
  Clock, 
  Loader2, 
  Sparkles,
  Play,
  Settings,
  GitBranch,
} from 'lucide-react';
import { useComposeJob, type JobComposition } from '@/hooks/useMLAI';
import { useToast } from '@/hooks/use-toast';

const MODEL_TYPES = [
  { value: 'deterioration_predictor', label: 'Deterioration Predictor' },
  { value: 'risk_scorer', label: 'Risk Scorer' },
  { value: 'symptom_classifier', label: 'Symptom Classifier' },
  { value: 'outcome_predictor', label: 'Outcome Predictor' },
  { value: 'anomaly_detector', label: 'Anomaly Detector' },
];

const DATA_SOURCES = [
  { id: 'vitals', label: 'Vital Signs' },
  { id: 'symptoms', label: 'Symptom Reports' },
  { id: 'medications', label: 'Medication Data' },
  { id: 'mental_health', label: 'Mental Health Questionnaires' },
  { id: 'wearables', label: 'Wearable Device Data' },
  { id: 'lab_results', label: 'Lab Results' },
];

export function JobComposerTab() {
  const { toast } = useToast();
  const [modelType, setModelType] = useState('');
  const [objective, setObjective] = useState('');
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [composedJob, setComposedJob] = useState<JobComposition | null>(null);
  
  const composeMutation = useComposeJob();
  
  const handleCompose = async () => {
    if (!modelType || !objective.trim() || selectedSources.length === 0) {
      toast({
        title: 'Missing information',
        description: 'Please select a model type, describe the objective, and choose at least one data source.',
        variant: 'destructive',
      });
      return;
    }
    
    try {
      const response = await composeMutation.mutateAsync({
        model_type: modelType,
        objective,
        data_sources: selectedSources,
      });
      setComposedJob(response.job);
      toast({
        title: 'Job composed',
        description: 'Your training job has been configured successfully.',
      });
    } catch {
      toast({
        title: 'Composition failed',
        description: 'Unable to compose training job. Please try again.',
        variant: 'destructive',
      });
    }
  };
  
  const toggleSource = (sourceId: string) => {
    setSelectedSources(prev =>
      prev.includes(sourceId)
        ? prev.filter(s => s !== sourceId)
        : [...prev, sourceId]
    );
  };
  
  return (
    <div className="space-y-6" data-testid="job-composer-tab">
      <Alert className="border-primary/50 bg-primary/5">
        <Wand2 className="h-4 w-4 text-primary" />
        <AlertTitle className="font-semibold">AI Job Composer</AlertTitle>
        <AlertDescription className="text-sm">
          Describe your ML objective and let AI configure the optimal training job,
          including hyperparameters, data pipeline, and validation strategy.
        </AlertDescription>
      </Alert>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-primary" />
            Job Configuration
          </CardTitle>
          <CardDescription>
            Define your training objectives
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Model Type</Label>
            <Select value={modelType} onValueChange={setModelType}>
              <SelectTrigger data-testid="select-model-type">
                <SelectValue placeholder="Select model type" />
              </SelectTrigger>
              <SelectContent>
                {MODEL_TYPES.map(type => (
                  <SelectItem key={type.value} value={type.value}>
                    {type.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <Label>Training Objective</Label>
            <Textarea
              placeholder="Describe what you want the model to predict or classify... e.g., 'Predict 30-day hospitalization risk for chronic care patients based on vital signs and symptom patterns'"
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="min-h-[100px]"
              data-testid="input-training-objective"
            />
          </div>
          
          <div className="space-y-2">
            <Label>Data Sources</Label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {DATA_SOURCES.map(source => (
                <div
                  key={source.id}
                  className="flex items-center space-x-2"
                >
                  <Checkbox
                    id={source.id}
                    checked={selectedSources.includes(source.id)}
                    onCheckedChange={() => toggleSource(source.id)}
                  />
                  <label
                    htmlFor={source.id}
                    className="text-sm cursor-pointer"
                  >
                    {source.label}
                  </label>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleCompose}
            disabled={composeMutation.isPending || !modelType || !objective.trim() || selectedSources.length === 0}
            className="w-full"
            data-testid="button-compose-job"
          >
            {composeMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Composing Job...
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4 mr-2" />
                Compose Training Job
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
      
      {composeMutation.isPending && (
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
          </CardHeader>
          <CardContent className="space-y-4">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-48 w-full" />
          </CardContent>
        </Card>
      )}
      
      {composedJob && !composeMutation.isPending && (
        <Card data-testid="card-composed-job">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Cpu className="h-5 w-5 text-primary" />
                {composedJob.job_name}
              </CardTitle>
              <Badge>{composedJob.model_type}</Badge>
            </div>
            <CardDescription>
              Job ID: {composedJob.job_id}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-muted/50">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">Estimated Duration</span>
                </div>
                <p className="text-2xl font-bold">{composedJob.estimated_duration}</p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50">
                <div className="flex items-center gap-2 mb-2">
                  <Database className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">Validation Strategy</span>
                </div>
                <p className="text-2xl font-bold">{composedJob.data_pipeline.validation_strategy}</p>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-3">Hyperparameters</h4>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(composedJob.hyperparameters).map(([key, value]) => (
                  <div key={key} className="flex justify-between p-2 rounded bg-muted/30">
                    <span className="text-sm text-muted-foreground">{key}</span>
                    <span className="text-sm font-mono">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-3 flex items-center gap-2">
                <GitBranch className="h-4 w-4" />
                Data Pipeline
              </h4>
              <div className="space-y-2">
                <div>
                  <span className="text-sm text-muted-foreground">Preprocessing:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {composedJob.data_pipeline.preprocessing.map((step, i) => (
                      <Badge key={i} variant="outline">{step}</Badge>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Feature Engineering:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {composedJob.data_pipeline.feature_engineering.map((step, i) => (
                      <Badge key={i} variant="secondary">{step}</Badge>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
          <CardFooter className="flex gap-2">
            <Button variant="outline" className="flex-1" data-testid="button-save-job">
              Save as Draft
            </Button>
            <Button className="flex-1" data-testid="button-start-job">
              <Play className="h-4 w-4 mr-2" />
              Start Training
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
}
