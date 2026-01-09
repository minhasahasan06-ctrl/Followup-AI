import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Brain, 
  Search, 
  Users, 
  AlertTriangle, 
  CheckCircle2, 
  Loader2, 
  Code,
  Eye,
  Copy,
  Sparkles,
} from 'lucide-react';
import { useCohortTranslate, type CohortTranslation } from '@/hooks/useResearchAI';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '@/hooks/useAuth';

export function NLCohortBuilderTab() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<CohortTranslation | null>(null);
  const [showSQL, setShowSQL] = useState(false);
  
  const translateMutation = useCohortTranslate();
  
  const handleTranslate = async () => {
    if (!query.trim()) {
      toast({
        title: 'Query required',
        description: 'Please enter a natural language description of your cohort.',
        variant: 'destructive',
      });
      return;
    }
    
    if (!user?.id) {
      toast({
        title: 'Authentication required',
        description: 'Please sign in to use the cohort builder.',
        variant: 'destructive',
      });
      return;
    }
    
    try {
      const response = await translateMutation.mutateAsync({
        natural_language: query,
        doctor_id: user.id,
      });
      setResult(response.cohort);
      toast({
        title: 'Cohort translated',
        description: `Found ${response.cohort.patient_count} patients matching your criteria.`,
      });
    } catch {
      toast({
        title: 'Translation failed',
        description: 'Unable to translate your query. Please try rephrasing.',
        variant: 'destructive',
      });
    }
  };
  
  const handleCopySQL = () => {
    if (result?.sql_filter) {
      navigator.clipboard.writeText(result.sql_filter);
      toast({
        title: 'Copied',
        description: 'SQL filter copied to clipboard.',
      });
    }
  };
  
  const exampleQueries = [
    'Patients over 65 with diabetes and at least 2 hospitalizations in the past year',
    'Female patients with hypertension who are on ACE inhibitors',
    'Patients with chronic kidney disease stage 3 or higher',
    'Chronic care patients who had a COVID-19 diagnosis in 2024',
  ];
  
  return (
    <div className="space-y-6" data-testid="nl-cohort-builder-tab">
      <Alert className="border-primary/50 bg-primary/5">
        <Brain className="h-4 w-4 text-primary" />
        <AlertTitle className="font-semibold">Natural Language Cohort Builder</AlertTitle>
        <AlertDescription className="text-sm">
          Describe your research cohort in plain English. Our AI will translate it to a 
          k-anonymized patient selection with privacy protections built-in.
        </AlertDescription>
      </Alert>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5 text-primary" />
            Describe Your Cohort
          </CardTitle>
          <CardDescription>
            Use natural language to define your patient population
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Example: Patients over 50 with type 2 diabetes who have been hospitalized in the past 6 months..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="min-h-[120px]"
            data-testid="input-cohort-query"
          />
          
          <div className="space-y-2">
            <p className="text-sm font-medium text-muted-foreground">Example queries:</p>
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((example, i) => (
                <Badge
                  key={i}
                  variant="outline"
                  className="cursor-pointer hover-elevate text-xs"
                  onClick={() => setQuery(example)}
                  data-testid={`badge-example-query-${i}`}
                >
                  {example.slice(0, 50)}...
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleTranslate}
            disabled={translateMutation.isPending || !query.trim()}
            className="w-full"
            data-testid="button-translate-cohort"
          >
            {translateMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Translating...
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4 mr-2" />
                Build Cohort
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
      
      {translateMutation.isPending && (
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
          </CardHeader>
          <CardContent className="space-y-4">
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-40 w-full" />
          </CardContent>
        </Card>
      )}
      
      {result && !translateMutation.isPending && (
        <Card data-testid="card-cohort-result">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5 text-primary" />
                Cohort Results
              </CardTitle>
              <div className="flex items-center gap-2">
                {result.k_anonymous ? (
                  <Badge className="gap-1 bg-green-500">
                    <CheckCircle2 className="h-3 w-3" />
                    K-Anonymous
                  </Badge>
                ) : (
                  <Badge variant="destructive" className="gap-1">
                    <AlertTriangle className="h-3 w-3" />
                    Not K-Anonymous
                  </Badge>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-lg bg-muted/50 text-center">
                <p className="text-3xl font-bold text-primary">{result.patient_count}</p>
                <p className="text-sm text-muted-foreground">Matching Patients</p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50 text-center">
                <p className="text-3xl font-bold">{result.cohort_id.slice(0, 8)}...</p>
                <p className="text-sm text-muted-foreground">Cohort ID</p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50 text-center">
                <p className="text-3xl font-bold">{result.k_anonymous ? '≥25' : '<25'}</p>
                <p className="text-sm text-muted-foreground">K-Anonymity</p>
              </div>
            </div>
            
            {result.warnings && result.warnings.length > 0 && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Warnings</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc list-inside text-sm">
                    {result.warnings.map((warning, i) => (
                      <li key={i}>{warning}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="font-medium flex items-center gap-2">
                  <Eye className="h-4 w-4" />
                  Preview (De-identified)
                </h4>
              </div>
              <div className="border rounded-lg overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Patient Hash</TableHead>
                      <TableHead>Age Bucket</TableHead>
                      <TableHead>Condition Bucket</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.preview.slice(0, 5).map((row, i) => (
                      <TableRow key={i}>
                        <TableCell className="font-mono text-xs">
                          {row.patient_id_hash.slice(0, 12)}...
                        </TableCell>
                        <TableCell>{row.age_bucket}</TableCell>
                        <TableCell>{row.condition_bucket}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowSQL(!showSQL)}
                  className="gap-2"
                  data-testid="button-toggle-sql"
                >
                  <Code className="h-4 w-4" />
                  {showSQL ? 'Hide SQL' : 'Show SQL'}
                </Button>
                {showSQL && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopySQL}
                    className="gap-2"
                    data-testid="button-copy-sql"
                  >
                    <Copy className="h-4 w-4" />
                    Copy
                  </Button>
                )}
              </div>
              {showSQL && (
                <pre className="p-4 rounded-lg bg-muted overflow-x-auto text-xs font-mono">
                  {result.sql_filter}
                </pre>
              )}
            </div>
          </CardContent>
          <CardFooter className="flex gap-2">
            <Button variant="outline" className="flex-1" data-testid="button-save-cohort">
              Save Cohort
            </Button>
            <Button className="flex-1" data-testid="button-use-cohort">
              Use in Study
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
}
