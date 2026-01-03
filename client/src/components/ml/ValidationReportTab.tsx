import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { 
  Shield, 
  Calendar, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle,
  Loader2, 
  Search,
  TrendingUp,
  Clock,
} from 'lucide-react';
import { useValidationReport, type ValidationReport } from '@/hooks/useMLAI';
import { useToast } from '@/hooks/use-toast';

export function ValidationReportTab() {
  const { toast } = useToast();
  const [jobId, setJobId] = useState('');
  const [searchedJobId, setSearchedJobId] = useState('');
  
  const { data: report, isLoading, error, refetch } = useValidationReport(searchedJobId, !!searchedJobId);
  
  const handleSearch = () => {
    if (!jobId.trim()) {
      toast({
        title: 'Job ID required',
        description: 'Please enter a training job ID to view its validation report.',
        variant: 'destructive',
      });
      return;
    }
    setSearchedJobId(jobId);
  };
  
  const getRiskBadge = (risk: string) => {
    switch (risk) {
      case 'low':
        return <Badge className="bg-green-500 gap-1"><CheckCircle2 className="h-3 w-3" />Low Risk</Badge>;
      case 'medium':
        return <Badge className="bg-yellow-500 gap-1"><AlertTriangle className="h-3 w-3" />Medium Risk</Badge>;
      case 'high':
        return <Badge variant="destructive" className="gap-1"><XCircle className="h-3 w-3" />High Risk</Badge>;
      default:
        return <Badge variant="secondary">{risk}</Badge>;
    }
  };
  
  const getDriftStatus = (status: string) => {
    switch (status) {
      case 'ok':
        return <Badge className="bg-green-500">OK</Badge>;
      case 'warning':
        return <Badge className="bg-yellow-500">Warning</Badge>;
      case 'critical':
        return <Badge variant="destructive">Critical</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };
  
  return (
    <div className="space-y-6" data-testid="validation-report-tab">
      <Alert className="border-primary/50 bg-primary/5">
        <Shield className="h-4 w-4 text-primary" />
        <AlertTitle className="font-semibold">Temporal Validation Report</AlertTitle>
        <AlertDescription className="text-sm">
          View comprehensive validation including temporal split analysis, leakage detection,
          and train/test distribution comparisons.
        </AlertDescription>
      </Alert>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5 text-primary" />
            Find Validation Report
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter training job ID..."
              value={jobId}
              onChange={(e) => setJobId(e.target.value)}
              className="flex-1"
              data-testid="input-job-id"
            />
            <Button onClick={handleSearch} disabled={isLoading} data-testid="button-search-validation">
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
              <span>Loading validation report...</span>
            </div>
          </CardContent>
        </Card>
      )}
      
      {error && searchedJobId && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Report Not Found</AlertTitle>
          <AlertDescription>
            No validation report found for job ID: {searchedJobId}
          </AlertDescription>
        </Alert>
      )}
      
      {report && !isLoading && (
        <div className="space-y-4">
          <Card data-testid="card-temporal-split">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="h-5 w-5 text-primary" />
                Temporal Split
              </CardTitle>
              <CardDescription>
                Time-based train/test separation with gap period
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                  <p className="text-sm text-muted-foreground">Training Period</p>
                  <p className="font-medium">{report.temporal_split.train_start}</p>
                  <p className="font-medium">to {report.temporal_split.train_end}</p>
                </div>
                <div className="p-4 rounded-lg bg-muted/50 text-center">
                  <p className="text-sm text-muted-foreground">Gap Period</p>
                  <p className="text-3xl font-bold">{report.temporal_split.gap_days}</p>
                  <p className="text-xs text-muted-foreground">days</p>
                </div>
                <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                  <p className="text-sm text-muted-foreground">Testing Period</p>
                  <p className="font-medium">{report.temporal_split.test_start}</p>
                  <p className="font-medium">to {report.temporal_split.test_end}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card data-testid="card-leakage-report">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5 text-primary" />
                  Leakage Detection
                </CardTitle>
                {getRiskBadge(report.leakage_report.overall_risk)}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <span>Target Leakage</span>
                  {report.leakage_report.target_leakage ? (
                    <Badge variant="destructive">Detected</Badge>
                  ) : (
                    <Badge className="bg-green-500">None</Badge>
                  )}
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <span>Data Leakage</span>
                  {report.leakage_report.data_leakage ? (
                    <Badge variant="destructive">Detected</Badge>
                  ) : (
                    <Badge className="bg-green-500">None</Badge>
                  )}
                </div>
              </div>
              
              {report.leakage_report.feature_leakage.length > 0 && (
                <div>
                  <h4 className="font-medium mb-2">Feature Leakage Warnings</h4>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Feature</TableHead>
                        <TableHead>Risk Level</TableHead>
                        <TableHead>Reason</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {report.leakage_report.feature_leakage.map((leak, i) => (
                        <TableRow key={i}>
                          <TableCell className="font-mono">{leak.feature}</TableCell>
                          <TableCell>{getRiskBadge(leak.risk)}</TableCell>
                          <TableCell className="text-sm text-muted-foreground">{leak.reason}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
          
          <Card data-testid="card-distribution">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                Train/Test Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <h4 className="font-medium mb-3">Feature Drift (PSI)</h4>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead className="text-right">PSI Score</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {report.train_test_distribution.feature_drifts.map((drift, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-mono">{drift.feature}</TableCell>
                      <TableCell className="text-right">{drift.psi.toFixed(4)}</TableCell>
                      <TableCell>{getDriftStatus(drift.status)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
