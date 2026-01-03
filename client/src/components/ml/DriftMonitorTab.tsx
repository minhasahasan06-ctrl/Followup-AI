import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { 
  Activity, 
  Loader2, 
  Search,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Bell,
  Calendar,
} from 'lucide-react';
import { useDriftReport, type DriftReport } from '@/hooks/useMLAI';
import { useToast } from '@/hooks/use-toast';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

export function DriftMonitorTab() {
  const { toast } = useToast();
  const [modelId, setModelId] = useState('');
  const [searchedModelId, setSearchedModelId] = useState('');
  
  const { data: report, isLoading, error } = useDriftReport(searchedModelId, !!searchedModelId);
  
  const handleSearch = () => {
    if (!modelId.trim()) {
      toast({
        title: 'Model ID required',
        description: 'Please enter a model ID to view its drift report.',
        variant: 'destructive',
      });
      return;
    }
    setSearchedModelId(modelId);
  };
  
  const getDriftStatusColor = (status: string) => {
    switch (status) {
      case 'ok':
        return 'hsl(var(--chart-2))';
      case 'warning':
        return 'hsl(45, 100%, 50%)';
      case 'critical':
        return 'hsl(var(--destructive))';
      default:
        return 'hsl(var(--muted-foreground))';
    }
  };
  
  const getDriftBadge = (status: string) => {
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
  
  const getAlertSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default:
        return <Bell className="h-4 w-4 text-blue-500" />;
    }
  };
  
  return (
    <div className="space-y-6" data-testid="drift-monitor-tab">
      <Alert className="border-primary/50 bg-primary/5">
        <Activity className="h-4 w-4 text-primary" />
        <AlertTitle className="font-semibold">Model Drift Monitor</AlertTitle>
        <AlertDescription className="text-sm">
          Monitor feature and prediction drift in deployed models using PSI and KL divergence metrics.
          Receive alerts when drift exceeds thresholds.
        </AlertDescription>
      </Alert>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5 text-primary" />
            Find Drift Report
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter deployed model ID..."
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="flex-1"
              data-testid="input-drift-model-id"
            />
            <Button onClick={handleSearch} disabled={isLoading} data-testid="button-search-drift">
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
              <span>Loading drift report...</span>
            </div>
          </CardContent>
        </Card>
      )}
      
      {error && searchedModelId && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Report Not Found</AlertTitle>
          <AlertDescription>
            No drift report found for model ID: {searchedModelId}
          </AlertDescription>
        </Alert>
      )}
      
      {report && !isLoading && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Report Date: {report.report_date}</span>
            </div>
            {getDriftBadge(report.prediction_drift.status)}
          </div>
          
          <Card data-testid="card-feature-drift">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                Feature Drift (PSI)
              </CardTitle>
              <CardDescription>
                Population Stability Index for each feature
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[250px] mb-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={report.feature_drifts} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis type="number" domain={[0, 'auto']} />
                    <YAxis type="category" dataKey="feature" width={100} fontSize={12} />
                    <Tooltip 
                      formatter={(value: number) => value.toFixed(4)}
                      labelFormatter={(value) => `Feature: ${value}`}
                    />
                    <ReferenceLine x={0.1} stroke="hsl(45, 100%, 50%)" strokeDasharray="3 3" />
                    <ReferenceLine x={0.25} stroke="hsl(var(--destructive))" strokeDasharray="3 3" />
                    <Bar dataKey="psi">
                      {report.feature_drifts.map((entry, index) => (
                        <Cell key={index} fill={getDriftStatusColor(entry.status)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead className="text-right">PSI</TableHead>
                    <TableHead className="text-right">KL Divergence</TableHead>
                    <TableHead className="text-right">Baseline μ</TableHead>
                    <TableHead className="text-right">Current μ</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {report.feature_drifts.map((drift, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-mono text-sm">{drift.feature}</TableCell>
                      <TableCell className="text-right">{drift.psi.toFixed(4)}</TableCell>
                      <TableCell className="text-right">{drift.kl_divergence.toFixed(4)}</TableCell>
                      <TableCell className="text-right">{drift.baseline_mean.toFixed(3)}</TableCell>
                      <TableCell className="text-right">{drift.current_mean.toFixed(3)}</TableCell>
                      <TableCell>{getDriftBadge(drift.status)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
          
          <Card data-testid="card-prediction-drift">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <TrendingDown className="h-5 w-5 text-primary" />
                  Prediction Drift
                </CardTitle>
                {getDriftBadge(report.prediction_drift.status)}
              </div>
              <CardDescription>
                PSI: {report.prediction_drift.psi.toFixed(4)}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg border">
                  <p className="text-sm font-medium mb-2">Baseline Distribution</p>
                  <div className="space-y-1">
                    {Object.entries(report.prediction_drift.baseline_distribution).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="text-muted-foreground">{key}</span>
                        <span className="font-mono">{value.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="p-4 rounded-lg border">
                  <p className="text-sm font-medium mb-2">Current Distribution</p>
                  <div className="space-y-1">
                    {Object.entries(report.prediction_drift.current_distribution).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="text-muted-foreground">{key}</span>
                        <span className="font-mono">{value.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {report.alerts.length > 0 && (
            <Card data-testid="card-drift-alerts">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="h-5 w-5 text-primary" />
                  Active Alerts
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {report.alerts.map((alert) => (
                    <div key={alert.id} className="flex items-start gap-3 p-3 rounded-lg border">
                      {getAlertSeverityIcon(alert.severity)}
                      <div className="flex-1">
                        <p className="font-medium">{alert.message}</p>
                        {alert.feature && (
                          <p className="text-sm text-muted-foreground">Feature: {alert.feature}</p>
                        )}
                        <p className="text-xs text-muted-foreground">{alert.created_at}</p>
                      </div>
                      <Badge variant={alert.severity === 'critical' ? 'destructive' : 'secondary'}>
                        {alert.severity}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
