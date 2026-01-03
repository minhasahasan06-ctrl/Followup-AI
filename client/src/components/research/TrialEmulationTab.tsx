import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { 
  FlaskConical, 
  BarChart3, 
  Users, 
  TrendingUp,
  TrendingDown,
  Minus,
  Loader2, 
  Sparkles,
  Scale,
  AlertTriangle,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { useTrialEmulation, type TrialEmulation } from '@/hooks/useResearchAI';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '@/hooks/useAuth';

export function TrialEmulationTab() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [cohortId, setCohortId] = useState('');
  const [treatmentVar, setTreatmentVar] = useState('');
  const [outcomeVar, setOutcomeVar] = useState('');
  const [covariates, setCovariates] = useState('');
  const [emulation, setEmulation] = useState<TrialEmulation | null>(null);
  
  const emulateMutation = useTrialEmulation();
  
  const handleEmulate = async () => {
    if (!cohortId.trim() || !treatmentVar.trim() || !outcomeVar.trim()) {
      toast({
        title: 'Missing information',
        description: 'Please provide cohort ID, treatment variable, and outcome variable.',
        variant: 'destructive',
      });
      return;
    }
    
    if (!user?.id) {
      toast({
        title: 'Authentication required',
        description: 'Please sign in to run trial emulations.',
        variant: 'destructive',
      });
      return;
    }
    
    try {
      const response = await emulateMutation.mutateAsync({
        cohort_id: cohortId,
        doctor_id: user.id,
        treatment_variable: treatmentVar,
        outcome_variable: outcomeVar,
        covariates: covariates ? covariates.split(',').map(c => c.trim()) : undefined,
      });
      setEmulation(response.emulation);
      toast({
        title: 'Emulation complete',
        description: 'Target trial emulation has been completed.',
      });
    } catch {
      toast({
        title: 'Emulation failed',
        description: 'Unable to run trial emulation. Please check your inputs.',
        variant: 'destructive',
      });
    }
  };
  
  const getEffectInterpretation = (effect: number) => {
    if (effect > 1.2) return { icon: TrendingUp, color: 'text-red-500', text: 'Increased risk' };
    if (effect < 0.8) return { icon: TrendingDown, color: 'text-green-500', text: 'Reduced risk' };
    return { icon: Minus, color: 'text-yellow-500', text: 'No significant effect' };
  };
  
  const getSignificance = (pValue: number) => {
    if (pValue < 0.001) return { badge: '***', label: 'p < 0.001' };
    if (pValue < 0.01) return { badge: '**', label: 'p < 0.01' };
    if (pValue < 0.05) return { badge: '*', label: 'p < 0.05' };
    return { badge: 'ns', label: 'Not significant' };
  };
  
  return (
    <div className="space-y-6" data-testid="trial-emulation-tab">
      <Alert className="border-primary/50 bg-primary/5">
        <FlaskConical className="h-4 w-4 text-primary" />
        <AlertTitle className="font-semibold">Target Trial Emulation</AlertTitle>
        <AlertDescription className="text-sm">
          Emulate a randomized controlled trial using observational data with propensity score matching.
          This technique helps estimate causal effects while accounting for confounding.
        </AlertDescription>
      </Alert>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Scale className="h-5 w-5 text-primary" />
            Trial Configuration
          </CardTitle>
          <CardDescription>
            Define the treatment comparison for your target trial
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="cohort-id">Cohort ID</Label>
            <Input
              id="cohort-id"
              placeholder="Enter cohort ID from NL Cohort Builder..."
              value={cohortId}
              onChange={(e) => setCohortId(e.target.value)}
              data-testid="input-trial-cohort-id"
            />
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="treatment-var">Treatment Variable</Label>
              <Input
                id="treatment-var"
                placeholder="e.g., medication_x_prescribed"
                value={treatmentVar}
                onChange={(e) => setTreatmentVar(e.target.value)}
                data-testid="input-treatment-variable"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="outcome-var">Outcome Variable</Label>
              <Input
                id="outcome-var"
                placeholder="e.g., hospitalization_within_90_days"
                value={outcomeVar}
                onChange={(e) => setOutcomeVar(e.target.value)}
                data-testid="input-outcome-variable"
              />
            </div>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="covariates">Covariates (comma-separated)</Label>
            <Input
              id="covariates"
              placeholder="e.g., age, sex, comorbidity_count, baseline_egfr"
              value={covariates}
              onChange={(e) => setCovariates(e.target.value)}
              data-testid="input-covariates"
            />
            <p className="text-xs text-muted-foreground">
              Variables to adjust for in propensity score matching
            </p>
          </div>
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleEmulate}
            disabled={emulateMutation.isPending || !cohortId.trim() || !treatmentVar.trim() || !outcomeVar.trim()}
            className="w-full"
            data-testid="button-run-emulation"
          >
            {emulateMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Running Emulation...
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4 mr-2" />
                Run Trial Emulation
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
      
      {emulateMutation.isPending && (
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
      
      {emulation && !emulateMutation.isPending && (
        <div className="space-y-4">
          <Card data-testid="card-emulation-result">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                Emulation Results
              </CardTitle>
              <CardDescription>
                Trial ID: {emulation.trial_id}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg border">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Treatment Arm</span>
                    <Badge>N = {emulation.treatment_arm.size}</Badge>
                  </div>
                  <div className="space-y-2">
                    {Object.entries(emulation.treatment_arm.outcomes).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="text-muted-foreground">{key}</span>
                        <span className="font-medium">{typeof value === 'number' ? value.toFixed(2) : value}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="p-4 rounded-lg border">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Control Arm</span>
                    <Badge variant="secondary">N = {emulation.control_arm.size}</Badge>
                  </div>
                  <div className="space-y-2">
                    {Object.entries(emulation.control_arm.outcomes).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="text-muted-foreground">{key}</span>
                        <span className="font-medium">{typeof value === 'number' ? value.toFixed(2) : value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="p-6 rounded-lg bg-muted/50 text-center">
                {(() => {
                  const interpretation = getEffectInterpretation(emulation.effect_estimate);
                  const significance = getSignificance(emulation.p_value);
                  const EffectIcon = interpretation.icon;
                  return (
                    <>
                      <div className="flex items-center justify-center gap-3 mb-2">
                        <EffectIcon className={`h-8 w-8 ${interpretation.color}`} />
                        <span className="text-4xl font-bold">{emulation.effect_estimate.toFixed(2)}</span>
                        <Badge variant={emulation.p_value < 0.05 ? 'default' : 'secondary'}>
                          {significance.badge}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-1">
                        Effect Estimate (Hazard Ratio)
                      </p>
                      <p className={`text-sm font-medium ${interpretation.color}`}>
                        {interpretation.text}
                      </p>
                      <p className="text-xs text-muted-foreground mt-2">
                        95% CI: [{emulation.confidence_interval[0].toFixed(2)} - {emulation.confidence_interval[1].toFixed(2)}]
                        {' | '}
                        {significance.label}
                      </p>
                    </>
                  );
                })()}
              </div>
              
              <div>
                <h4 className="font-medium mb-3 flex items-center gap-2">
                  <Scale className="h-4 w-4" />
                  Covariate Balance (After Matching)
                </h4>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Covariate</TableHead>
                      <TableHead className="text-right">Standardized Difference</TableHead>
                      <TableHead>Balance</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(emulation.balance_metrics).map(([key, value]) => {
                      const isBalanced = Math.abs(value) < 0.1;
                      return (
                        <TableRow key={key}>
                          <TableCell>{key}</TableCell>
                          <TableCell className="text-right font-mono">
                            {value.toFixed(3)}
                          </TableCell>
                          <TableCell>
                            {isBalanced ? (
                              <Badge className="bg-green-500 gap-1">
                                <CheckCircle2 className="h-3 w-3" />
                                Balanced
                              </Badge>
                            ) : (
                              <Badge variant="destructive" className="gap-1">
                                <AlertTriangle className="h-3 w-3" />
                                Imbalanced
                              </Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
          
          <Alert>
            <Info className="h-4 w-4" />
            <AlertTitle>Interpretation Note</AlertTitle>
            <AlertDescription className="text-sm">
              Target trial emulation provides causal effect estimates under the assumption that
              all confounders are measured and appropriately adjusted. Results should be interpreted
              with caution and validated with clinical expertise.
            </AlertDescription>
          </Alert>
        </div>
      )}
    </div>
  );
}
