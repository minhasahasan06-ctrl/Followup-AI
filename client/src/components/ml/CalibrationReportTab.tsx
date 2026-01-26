import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Progress } from '@/components/ui/progress';
import { 
  Target, 
  Loader2, 
  Search,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  BarChart3,
} from 'lucide-react';
import { useCalibrationReport, type CalibrationReport } from '@/hooks/useMLAI';
import { useToast } from '@/hooks/use-toast';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from 'recharts';

export function CalibrationReportTab() {
  const { toast } = useToast();
  const [jobId, setJobId] = useState('');
  const [searchedJobId, setSearchedJobId] = useState('');
  
  const { data: report, isLoading, error } = useCalibrationReport(searchedJobId, !!searchedJobId);
  
  const handleSearch = () => {
    if (!jobId.trim()) {
      toast({
        title: 'Job ID required',
        description: 'Please enter a training job ID to view its calibration report.',
        variant: 'destructive',
      });
      return;
    }
    setSearchedJobId(jobId);
  };
  
  const getCalibrationQuality = (ece: number) => {
    if (ece < 0.05) return { label: 'Excellent', color: 'bg-green-500' };
    if (ece < 0.1) return { label: 'Good', color: 'bg-blue-500' };
    if (ece < 0.2) return { label: 'Fair', color: 'bg-yellow-500' };
    return { label: 'Poor', color: 'bg-red-500' };
  };
  
  return (
    <div className="space-y-6" data-testid="calibration-report-tab">
      <Alert className="border-primary/50 bg-primary/5">
        <Target className="h-4 w-4 text-primary" />
        <AlertTitle className="font-semibold">Model Calibration Report</AlertTitle>
        <AlertDescription className="text-sm">
          Analyze model probability calibration with reliability diagrams, Brier scores,
          and threshold optimization for clinical decision support.
        </AlertDescription>
      </Alert>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5 text-primary" />
            Find Calibration Report
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter training job ID..."
              value={jobId}
              onChange={(e) => setJobId(e.target.value)}
              className="flex-1"
              data-testid="input-calibration-job-id"
            />
            <Button onClick={handleSearch} disabled={isLoading} data-testid="button-search-calibration">
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
              <span>Loading calibration report...</span>
            </div>
          </CardContent>
        </Card>
      )}
      
      {error && searchedJobId && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Report Not Found</AlertTitle>
          <AlertDescription>
            No calibration report found for job ID: {searchedJobId}
          </AlertDescription>
        </Alert>
      )}
      
      {report && !isLoading && (
        <div className="space-y-4">
          <div className="grid md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="pt-6 text-center">
                <p className="text-sm text-muted-foreground mb-1">Brier Score</p>
                <p className="text-3xl font-bold">{report.brier_score.toFixed(4)}</p>
                <p className="text-xs text-muted-foreground">Lower is better</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6 text-center">
                {(() => {
                  const quality = getCalibrationQuality(report.expected_calibration_error);
                  return (
                    <>
                      <p className="text-sm text-muted-foreground mb-1">Expected Calibration Error</p>
                      <p className="text-3xl font-bold">{report.expected_calibration_error.toFixed(4)}</p>
                      <Badge className={quality.color}>{quality.label}</Badge>
                    </>
                  );
                })()}
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6 text-center">
                <p className="text-sm text-muted-foreground mb-1">Optimal Threshold</p>
                <p className="text-3xl font-bold">{report.optimal_threshold.toFixed(2)}</p>
                <p className="text-xs text-muted-foreground">F1-optimized</p>
              </CardContent>
            </Card>
          </div>
          
          <Card data-testid="card-calibration-curve">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                Calibration Curve
              </CardTitle>
              <CardDescription>
                Predicted probability vs. actual frequency
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={report.calibration_curve}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis 
                      dataKey="predicted" 
                      label={{ value: 'Predicted Probability', position: 'bottom' }}
                      domain={[0, 1]}
                    />
                    <YAxis 
                      label={{ value: 'Actual Frequency', angle: -90, position: 'left' }}
                      domain={[0, 1]}
                    />
                    <Tooltip 
                      formatter={(value: number) => value.toFixed(3)}
                      labelFormatter={(value) => `Predicted: ${value}`}
                    />
                    <Legend />
                    <ReferenceLine 
                      stroke="hsl(var(--muted-foreground))" 
                      strokeDasharray="5 5"
                      segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="actual" 
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      name="Model"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
          
          <Card data-testid="card-threshold-analysis">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                Threshold Analysis
              </CardTitle>
              <CardDescription>
                Performance metrics at different decision thresholds
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Threshold</TableHead>
                    <TableHead className="text-right">Precision</TableHead>
                    <TableHead className="text-right">Recall</TableHead>
                    <TableHead className="text-right">F1 Score</TableHead>
                    <TableHead className="text-right">Specificity</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {report.threshold_analysis.map((row, i) => (
                    <TableRow 
                      key={i} 
                      className={row.threshold === report.optimal_threshold ? 'bg-primary/10' : ''}
                    >
                      <TableCell>
                        {row.threshold.toFixed(2)}
                        {row.threshold === report.optimal_threshold && (
                          <Badge className="ml-2 bg-green-500" variant="secondary">Optimal</Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-right">{row.precision.toFixed(3)}</TableCell>
                      <TableCell className="text-right">{row.recall.toFixed(3)}</TableCell>
                      <TableCell className="text-right font-medium">{row.f1.toFixed(3)}</TableCell>
                      <TableCell className="text-right">{row.specificity.toFixed(3)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Optimal Performance Metrics
              </CardTitle>
              <CardDescription>
                At threshold = {report.optimal_threshold.toFixed(2)}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Precision</span>
                    <span className="font-medium">{(report.optimal_metrics.precision * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={report.optimal_metrics.precision * 100} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Recall</span>
                    <span className="font-medium">{(report.optimal_metrics.recall * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={report.optimal_metrics.recall * 100} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">F1 Score</span>
                    <span className="font-medium">{(report.optimal_metrics.f1 * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={report.optimal_metrics.f1 * 100} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
