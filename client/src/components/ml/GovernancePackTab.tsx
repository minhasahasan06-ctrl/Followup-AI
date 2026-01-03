import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Shield, 
  Loader2, 
  Search,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  FileText,
  Database,
  GitBranch,
  Rocket,
  Download,
} from 'lucide-react';
import { useGovernancePack, type GovernancePack } from '@/hooks/useMLAI';
import { useToast } from '@/hooks/use-toast';

export function GovernancePackTab() {
  const { toast } = useToast();
  const [jobId, setJobId] = useState('');
  const [searchedJobId, setSearchedJobId] = useState('');
  
  const { data: pack, isLoading, error } = useGovernancePack(searchedJobId, !!searchedJobId);
  
  const handleSearch = () => {
    if (!jobId.trim()) {
      toast({
        title: 'Job ID required',
        description: 'Please enter a training job ID to view its governance pack.',
        variant: 'destructive',
      });
      return;
    }
    setSearchedJobId(jobId);
  };
  
  const getCheckStatusIcon = (status: 'pass' | 'fail' | 'warn') => {
    switch (status) {
      case 'pass':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'fail':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warn':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    }
  };
  
  return (
    <div className="space-y-6" data-testid="governance-pack-tab">
      <Alert className="border-primary/50 bg-primary/5">
        <Shield className="h-4 w-4 text-primary" />
        <AlertTitle className="font-semibold">Model Governance Pack</AlertTitle>
        <AlertDescription className="text-sm">
          Complete governance documentation including model card, datasheet, reproducibility info,
          and deployment gate checks for HIPAA-compliant ML deployment.
        </AlertDescription>
      </Alert>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5 text-primary" />
            Find Governance Pack
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter training job ID..."
              value={jobId}
              onChange={(e) => setJobId(e.target.value)}
              className="flex-1"
              data-testid="input-governance-job-id"
            />
            <Button onClick={handleSearch} disabled={isLoading} data-testid="button-search-governance">
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Search'}
            </Button>
          </div>
        </CardContent>
      </Card>
      
      {isLoading && (
        <Card>
          <CardContent className="py-8">
            <div className="flex items-center justify-center gap-2">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <span>Loading governance pack...</span>
            </div>
          </CardContent>
        </Card>
      )}
      
      {error && searchedJobId && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Pack Not Found</AlertTitle>
          <AlertDescription>
            No governance pack found for job ID: {searchedJobId}
          </AlertDescription>
        </Alert>
      )}
      
      {pack && !isLoading && (
        <div className="space-y-4">
          <Card data-testid="card-deploy-gate">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Rocket className="h-5 w-5 text-primary" />
                  Deployment Gate
                </CardTitle>
                {pack.deploy_gate.passed ? (
                  <Badge className="bg-green-500 gap-1">
                    <CheckCircle2 className="h-3 w-3" />
                    Passed
                  </Badge>
                ) : (
                  <Badge variant="destructive" className="gap-1">
                    <XCircle className="h-3 w-3" />
                    Blocked
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                {pack.deploy_gate.checks.map((check, i) => (
                  <div key={i} className="flex items-center justify-between p-3 rounded-lg border">
                    <div className="flex items-center gap-2">
                      {getCheckStatusIcon(check.status)}
                      <span className="font-medium">{check.name}</span>
                    </div>
                    <span className="text-sm text-muted-foreground">{check.message}</span>
                  </div>
                ))}
              </div>
              
              {pack.deploy_gate.blocking_issues.length > 0 && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Blocking Issues</AlertTitle>
                  <AlertDescription>
                    <ul className="list-disc list-inside text-sm mt-2">
                      {pack.deploy_gate.blocking_issues.map((issue, i) => (
                        <li key={i}>{issue}</li>
                      ))}
                    </ul>
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
          
          <div className="grid md:grid-cols-2 gap-4">
            <Card data-testid="card-model-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-primary" />
                  Model Card
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[300px] pr-4">
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Name</p>
                      <p className="font-medium">{pack.model_card.name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Version</p>
                      <Badge>{pack.model_card.version}</Badge>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Description</p>
                      <p className="text-sm">{pack.model_card.description}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Intended Use</p>
                      <p className="text-sm">{pack.model_card.intended_use}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Limitations</p>
                      <ul className="list-disc list-inside text-sm">
                        {pack.model_card.limitations.map((l, i) => (
                          <li key={i}>{l}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Ethical Considerations</p>
                      <ul className="list-disc list-inside text-sm">
                        {pack.model_card.ethical_considerations.map((e, i) => (
                          <li key={i}>{e}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Performance Metrics</p>
                      <div className="grid grid-cols-2 gap-2 mt-1">
                        {Object.entries(pack.model_card.performance_metrics).map(([key, value]) => (
                          <div key={key} className="flex justify-between p-2 rounded bg-muted/30">
                            <span className="text-xs">{key}</span>
                            <span className="text-xs font-mono">{value.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
            
            <Card data-testid="card-datasheet">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5 text-primary" />
                  Datasheet
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[300px] pr-4">
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Data Source</p>
                      <p className="font-medium">{pack.datasheet.data_source}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Collection Process</p>
                      <p className="text-sm">{pack.datasheet.collection_process}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Preprocessing</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {pack.datasheet.preprocessing.map((p, i) => (
                          <Badge key={i} variant="outline">{p}</Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Privacy Considerations</p>
                      <ul className="list-disc list-inside text-sm">
                        {pack.datasheet.privacy_considerations.map((p, i) => (
                          <li key={i}>{p}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">K-Anonymity Threshold</p>
                      <Badge className="bg-green-500">k ≥ {pack.datasheet.k_anonymity_threshold}</Badge>
                    </div>
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
          
          <Card data-testid="card-reproducibility">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitBranch className="h-5 w-5 text-primary" />
                Reproducibility Pack
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-4 gap-4">
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-xs text-muted-foreground mb-1">Code Hash</p>
                  <p className="font-mono text-xs truncate">{pack.reproducibility_pack.code_hash}</p>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-xs text-muted-foreground mb-1">Data Hash</p>
                  <p className="font-mono text-xs truncate">{pack.reproducibility_pack.data_hash}</p>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-xs text-muted-foreground mb-1">Environment Hash</p>
                  <p className="font-mono text-xs truncate">{pack.reproducibility_pack.env_hash}</p>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-xs text-muted-foreground mb-1">Random Seed</p>
                  <p className="font-mono text-xs">{pack.reproducibility_pack.random_seed}</p>
                </div>
              </div>
            </CardContent>
            <CardFooter>
              <Button variant="outline" className="w-full gap-2" data-testid="button-download-pack">
                <Download className="h-4 w-4" />
                Download Full Pack
              </Button>
            </CardFooter>
          </Card>
        </div>
      )}
    </div>
  );
}
