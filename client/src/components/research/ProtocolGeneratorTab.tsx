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
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  FileText, 
  Target, 
  Users, 
  Calendar, 
  CheckCircle2, 
  XCircle,
  Loader2, 
  Download,
  Sparkles,
  ClipboardList,
  Beaker,
} from 'lucide-react';
import { useGenerateProtocol, type StudyProtocol } from '@/hooks/useResearchAI';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '@/hooks/useAuth';

export function ProtocolGeneratorTab() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [cohortId, setCohortId] = useState('');
  const [studyObjective, setStudyObjective] = useState('');
  const [studyType, setStudyType] = useState('observational');
  const [protocol, setProtocol] = useState<StudyProtocol | null>(null);
  
  const generateMutation = useGenerateProtocol();
  
  const handleGenerate = async () => {
    if (!cohortId.trim() || !studyObjective.trim()) {
      toast({
        title: 'Missing information',
        description: 'Please provide both a cohort ID and study objective.',
        variant: 'destructive',
      });
      return;
    }
    
    if (!user?.id) {
      toast({
        title: 'Authentication required',
        description: 'Please sign in to generate protocols.',
        variant: 'destructive',
      });
      return;
    }
    
    try {
      const response = await generateMutation.mutateAsync({
        cohort_id: cohortId,
        doctor_id: user.id,
        study_objective: studyObjective,
        study_type: studyType,
      });
      setProtocol(response.protocol);
      toast({
        title: 'Protocol generated',
        description: 'Your study protocol has been created successfully.',
      });
    } catch {
      toast({
        title: 'Generation failed',
        description: 'Unable to generate protocol. Please try again.',
        variant: 'destructive',
      });
    }
  };
  
  const handleExport = () => {
    if (!protocol) return;
    
    const content = `
# ${protocol.title}

## Study Type
${protocol.study_type}

## Description
${protocol.description}

## Primary Endpoints
${protocol.primary_endpoints.map(e => `- ${e}`).join('\n')}

## Secondary Endpoints
${protocol.secondary_endpoints?.map(e => `- ${e}`).join('\n') || 'None specified'}

## Inclusion Criteria
${protocol.inclusion_criteria.map(c => `- ${c}`).join('\n')}

## Exclusion Criteria
${protocol.exclusion_criteria.map(c => `- ${c}`).join('\n')}

## Data Collection
${protocol.data_collection.map(d => `- ${d}`).join('\n')}

## Timeline
${protocol.timeline || 'To be determined'}

## Sample Size Estimate
${protocol.sample_size_estimate || 'To be calculated'}
    `.trim();
    
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `protocol-${protocol.id}.md`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast({
      title: 'Protocol exported',
      description: 'Your protocol has been downloaded as a Markdown file.',
    });
  };
  
  return (
    <div className="space-y-6" data-testid="protocol-generator-tab">
      <Alert className="border-primary/50 bg-primary/5">
        <FileText className="h-4 w-4 text-primary" />
        <AlertTitle className="font-semibold">AI Study Protocol Generator</AlertTitle>
        <AlertDescription className="text-sm">
          Generate comprehensive study protocols from your cohort definition and research objectives.
          Includes endpoints, criteria, and data collection requirements.
        </AlertDescription>
      </Alert>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Beaker className="h-5 w-5 text-primary" />
            Study Configuration
          </CardTitle>
          <CardDescription>
            Define your study parameters to generate a protocol
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
              data-testid="input-cohort-id"
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="study-type">Study Type</Label>
            <Select value={studyType} onValueChange={setStudyType}>
              <SelectTrigger data-testid="select-study-type">
                <SelectValue placeholder="Select study type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="observational">Observational</SelectItem>
                <SelectItem value="retrospective">Retrospective</SelectItem>
                <SelectItem value="prospective">Prospective</SelectItem>
                <SelectItem value="case-control">Case-Control</SelectItem>
                <SelectItem value="cohort">Cohort Study</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="objective">Study Objective</Label>
            <Textarea
              id="objective"
              placeholder="Describe your research objective... e.g., 'Evaluate the effectiveness of medication X in reducing hospitalization rates among elderly diabetic patients'"
              value={studyObjective}
              onChange={(e) => setStudyObjective(e.target.value)}
              className="min-h-[100px]"
              data-testid="input-study-objective"
            />
          </div>
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleGenerate}
            disabled={generateMutation.isPending || !cohortId.trim() || !studyObjective.trim()}
            className="w-full"
            data-testid="button-generate-protocol"
          >
            {generateMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Generating Protocol...
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4 mr-2" />
                Generate Protocol
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
      
      {generateMutation.isPending && (
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-64" />
          </CardHeader>
          <CardContent className="space-y-4">
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-40 w-full" />
          </CardContent>
        </Card>
      )}
      
      {protocol && !generateMutation.isPending && (
        <Card data-testid="card-protocol-result">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <ClipboardList className="h-5 w-5 text-primary" />
                  {protocol.title}
                </CardTitle>
                <CardDescription className="mt-1">
                  <Badge variant="secondary">{protocol.study_type}</Badge>
                  {protocol.sample_size_estimate && (
                    <span className="ml-2 text-xs">
                      Estimated N = {protocol.sample_size_estimate}
                    </span>
                  )}
                </CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleExport}
                className="gap-2"
                data-testid="button-export-protocol"
              >
                <Download className="h-4 w-4" />
                Export
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[500px] pr-4">
              <div className="space-y-6">
                <div>
                  <h4 className="font-medium mb-2">Description</h4>
                  <p className="text-sm text-muted-foreground">{protocol.description}</p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2 flex items-center gap-2">
                    <Target className="h-4 w-4 text-primary" />
                    Primary Endpoints
                  </h4>
                  <ul className="space-y-1">
                    {protocol.primary_endpoints.map((endpoint, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                        {endpoint}
                      </li>
                    ))}
                  </ul>
                </div>
                
                {protocol.secondary_endpoints && protocol.secondary_endpoints.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">Secondary Endpoints</h4>
                    <ul className="space-y-1">
                      {protocol.secondary_endpoints.map((endpoint, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                          <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                          {endpoint}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium mb-2 flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                      Inclusion Criteria
                    </h4>
                    <ul className="space-y-1 text-sm">
                      {protocol.inclusion_criteria.map((criterion, i) => (
                        <li key={i} className="pl-2 border-l-2 border-green-500">{criterion}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2 flex items-center gap-2">
                      <XCircle className="h-4 w-4 text-red-500" />
                      Exclusion Criteria
                    </h4>
                    <ul className="space-y-1 text-sm">
                      {protocol.exclusion_criteria.map((criterion, i) => (
                        <li key={i} className="pl-2 border-l-2 border-red-500">{criterion}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Data Collection</h4>
                  <div className="flex flex-wrap gap-2">
                    {protocol.data_collection.map((item, i) => (
                      <Badge key={i} variant="outline">{item}</Badge>
                    ))}
                  </div>
                </div>
                
                {protocol.timeline && (
                  <div>
                    <h4 className="font-medium mb-2 flex items-center gap-2">
                      <Calendar className="h-4 w-4" />
                      Timeline
                    </h4>
                    <p className="text-sm text-muted-foreground">{protocol.timeline}</p>
                  </div>
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
