import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { apiRequest, queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { 
  Download, FileJson, FileSpreadsheet, Plus, Loader2, 
  CheckCircle, XCircle, Clock, AlertTriangle, Shield, 
  RefreshCw, Eye, Lock, ExternalLink
} from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

interface Dataset {
  id: string;
  name: string;
  version: string;
  rowCount: number;
  createdAt: string;
}

interface Export {
  id: string;
  datasetId: string;
  datasetName?: string;
  format: 'csv' | 'json';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  includePhi: boolean;
  rowCount?: number;
  fileSizeBytes?: number;
  errorMessage?: string;
  createdAt: string;
  createdBy: string;
  createdByRole?: string;
}

const STATUS_CONFIG = {
  pending: { label: 'Pending', icon: Clock, color: 'text-blue-500', badge: 'secondary' as const },
  processing: { label: 'Processing', icon: Loader2, color: 'text-amber-500', badge: 'default' as const },
  completed: { label: 'Completed', icon: CheckCircle, color: 'text-emerald-500', badge: 'default' as const },
  failed: { label: 'Failed', icon: XCircle, color: 'text-red-500', badge: 'destructive' as const },
};

const FORMAT_CONFIG = {
  csv: { label: 'CSV', icon: FileSpreadsheet, description: 'Comma-separated values' },
  json: { label: 'JSON', icon: FileJson, description: 'JavaScript Object Notation' },
};

export function ExportsTab() {
  const { toast } = useToast();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [exportFormat, setExportFormat] = useState<'csv' | 'json'>('csv');
  const [includePhi, setIncludePhi] = useState(false);

  const { data: exports = [], isLoading: exportsLoading, refetch: refetchExports } = useQuery<Export[]>({
    queryKey: ['/python-api/v1/research-center/exports'],
    refetchInterval: 10000,
  });

  const { data: datasets = [], isLoading: datasetsLoading } = useQuery<Dataset[]>({
    queryKey: ['/python-api/v1/research-center/datasets'],
  });

  const createExportMutation = useMutation({
    mutationFn: async (data: { datasetId: string; format: string; includePhi: boolean }) => {
      const response = await apiRequest(`/python-api/v1/research-center/datasets/${data.datasetId}/export`, {
        method: 'POST',
        json: { format: data.format, include_phi: data.includePhi }
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/python-api/v1/research-center/exports'] });
      setCreateDialogOpen(false);
      resetForm();
      toast({ title: 'Export Started', description: 'Your export job has been queued for processing.' });
    },
    onError: (error: Error) => {
      toast({ title: 'Export Failed', description: error.message, variant: 'destructive' });
    },
  });

  const resetForm = () => {
    setSelectedDatasetId('');
    setExportFormat('csv');
    setIncludePhi(false);
  };

  const handleCreateExport = () => {
    if (!selectedDatasetId) {
      toast({ title: 'Select Dataset', description: 'Please select a dataset to export.', variant: 'destructive' });
      return;
    }
    createExportMutation.mutate({ datasetId: selectedDatasetId, format: exportFormat, includePhi });
  };

  const getDownloadUrlMutation = useMutation({
    mutationFn: async (exportId: string) => {
      const response = await fetch(`/python-api/v1/research-center/exports/${exportId}`);
      if (!response.ok) throw new Error('Failed to get download URL');
      const data = await response.json();
      return data.downloadUrl;
    },
    onSuccess: (downloadUrl: string | null) => {
      if (downloadUrl) {
        window.open(downloadUrl, '_blank');
      } else {
        toast({ title: 'Download Not Available', description: 'This export is not ready for download.', variant: 'destructive' });
      }
    },
    onError: (error: Error) => {
      toast({ title: 'Download Failed', description: error.message, variant: 'destructive' });
    },
  });

  const handleDownload = (exp: Export) => {
    // Security: ALL downloads go through secure endpoint for:
    // 1. Fresh short-lived URL generation (15-minute TTL)
    // 2. HIPAA audit logging with request context
    // 3. Ownership verification on each download request
    if (exp.status === 'completed') {
      getDownloadUrlMutation.mutate(exp.id);
    }
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const pendingExports = exports.filter(e => e.status === 'pending' || e.status === 'processing');
  const completedExports = exports.filter(e => e.status === 'completed');
  const failedExports = exports.filter(e => e.status === 'failed');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold" data-testid="text-exports-title">Data Exports</h2>
          <p className="text-muted-foreground">Export research datasets with HIPAA-compliant controls</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={() => refetchExports()} data-testid="button-refresh-exports">
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button onClick={() => setCreateDialogOpen(true)} data-testid="button-new-export">
            <Plus className="h-4 w-4 mr-2" />
            New Export
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card data-testid="card-pending-exports">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Clock className="h-4 w-4 text-blue-500" />
              In Progress
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{pendingExports.length}</div>
          </CardContent>
        </Card>
        <Card data-testid="card-completed-exports">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-emerald-500" />
              Completed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{completedExports.length}</div>
          </CardContent>
        </Card>
        <Card data-testid="card-failed-exports">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <XCircle className="h-4 w-4 text-red-500" />
              Failed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{failedExports.length}</div>
          </CardContent>
        </Card>
      </div>

      {pendingExports.length > 0 && (
        <Card data-testid="card-active-exports">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin text-amber-500" />
              Active Exports
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {pendingExports.map((exp) => (
                <div key={exp.id} className="flex items-center justify-between p-4 border rounded-lg" data-testid={`export-active-${exp.id}`}>
                  <div className="flex items-center gap-4">
                    {FORMAT_CONFIG[exp.format].icon === FileSpreadsheet ? 
                      <FileSpreadsheet className="h-8 w-8 text-green-600" /> : 
                      <FileJson className="h-8 w-8 text-amber-600" />
                    }
                    <div>
                      <div className="font-medium">{exp.datasetName || `Dataset ${exp.datasetId.slice(0, 8)}`}</div>
                      <div className="text-sm text-muted-foreground flex items-center gap-2">
                        <Badge variant={STATUS_CONFIG[exp.status].badge}>
                          {STATUS_CONFIG[exp.status].label}
                        </Badge>
                        {exp.includePhi && (
                          <Badge variant="outline" className="gap-1">
                            <Shield className="h-3 w-3" />
                            PHI Included
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="w-32">
                    <Progress value={exp.status === 'processing' ? 50 : 10} className="h-2" />
                    <div className="text-xs text-muted-foreground mt-1 text-center">
                      {exp.status === 'processing' ? 'Processing...' : 'Queued'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card data-testid="card-exports-list">
        <CardHeader>
          <CardTitle>Export History</CardTitle>
          <CardDescription>All completed and failed exports</CardDescription>
        </CardHeader>
        <CardContent>
          {exportsLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => <Skeleton key={i} className="h-16 w-full" />)}
            </div>
          ) : exports.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Download className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No exports yet. Create your first export to get started.</p>
            </div>
          ) : (
            <ScrollArea className="h-[400px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Dataset</TableHead>
                    <TableHead>Format</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Rows</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {exports.filter(e => e.status !== 'pending' && e.status !== 'processing').map((exp) => {
                    const StatusIcon = STATUS_CONFIG[exp.status].icon;
                    const FormatIcon = FORMAT_CONFIG[exp.format].icon;
                    return (
                      <TableRow key={exp.id} data-testid={`export-row-${exp.id}`}>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <FormatIcon className="h-4 w-4 text-muted-foreground" />
                            <span className="font-medium">{exp.datasetName || `Dataset ${exp.datasetId.slice(0, 8)}`}</span>
                            {exp.includePhi && (
                              <Shield className="h-3 w-3 text-amber-500" />
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{exp.format.toUpperCase()}</Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant={STATUS_CONFIG[exp.status].badge} className="gap-1">
                            <StatusIcon className="h-3 w-3" />
                            {STATUS_CONFIG[exp.status].label}
                          </Badge>
                        </TableCell>
                        <TableCell>{exp.rowCount?.toLocaleString() || 'N/A'}</TableCell>
                        <TableCell>{formatFileSize(exp.fileSizeBytes)}</TableCell>
                        <TableCell>
                          <div className="text-sm">
                            {format(new Date(exp.createdAt), 'MMM d, yyyy')}
                            <div className="text-xs text-muted-foreground">
                              {formatDistanceToNow(new Date(exp.createdAt), { addSuffix: true })}
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>
                          {exp.status === 'completed' ? (
                            <Button 
                              size="sm" 
                              onClick={() => handleDownload(exp)}
                              disabled={getDownloadUrlMutation.isPending}
                              data-testid={`button-download-${exp.id}`}
                            >
                              {getDownloadUrlMutation.isPending ? (
                                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                              ) : (
                                <Download className="h-4 w-4 mr-1" />
                              )}
                              {exp.includePhi && !exp.downloadUrl ? 'Request Download' : 'Download'}
                            </Button>
                          ) : exp.status === 'failed' ? (
                            <Button variant="ghost" size="sm" disabled>
                              <XCircle className="h-4 w-4 mr-1 text-red-500" />
                              Failed
                            </Button>
                          ) : (
                            <Button variant="ghost" size="sm" disabled>
                              <Clock className="h-4 w-4 mr-1" />
                              Pending
                            </Button>
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Create New Export</DialogTitle>
            <DialogDescription>
              Export a research dataset in your preferred format
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6 py-4">
            <div className="space-y-2">
              <Label>Select Dataset</Label>
              <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId}>
                <SelectTrigger data-testid="select-dataset">
                  <SelectValue placeholder="Choose a dataset to export" />
                </SelectTrigger>
                <SelectContent>
                  {datasetsLoading ? (
                    <SelectItem value="loading" disabled>Loading datasets...</SelectItem>
                  ) : datasets.length === 0 ? (
                    <SelectItem value="none" disabled>No datasets available</SelectItem>
                  ) : (
                    datasets.map((ds) => (
                      <SelectItem key={ds.id} value={ds.id}>
                        {ds.name} (v{ds.version}) - {ds.rowCount?.toLocaleString() || 0} rows
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Export Format</Label>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(FORMAT_CONFIG).map(([key, config]) => {
                  const FormatIcon = config.icon;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setExportFormat(key as 'csv' | 'json')}
                      className={`p-4 border rounded-lg text-left transition-colors hover-elevate ${
                        exportFormat === key ? 'border-primary bg-primary/5' : ''
                      }`}
                      data-testid={`button-format-${key}`}
                    >
                      <FormatIcon className={`h-6 w-6 mb-2 ${exportFormat === key ? 'text-primary' : 'text-muted-foreground'}`} />
                      <div className="font-medium">{config.label}</div>
                      <div className="text-xs text-muted-foreground">{config.description}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-amber-500" />
                  <Label htmlFor="include-phi">Include PHI Data</Label>
                </div>
                <Switch 
                  id="include-phi" 
                  checked={includePhi} 
                  onCheckedChange={setIncludePhi}
                  data-testid="switch-include-phi"
                />
              </div>
              {includePhi && (
                <Alert variant="destructive">
                  <Lock className="h-4 w-4" />
                  <AlertTitle>Admin Authorization Required</AlertTitle>
                  <AlertDescription>
                    PHI exports require admin privileges. This action will be logged for HIPAA compliance.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)} data-testid="button-cancel-export">
              Cancel
            </Button>
            <Button 
              onClick={handleCreateExport} 
              disabled={!selectedDatasetId || createExportMutation.isPending}
              data-testid="button-start-export"
            >
              {createExportMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <Download className="h-4 w-4 mr-2" />
                  Start Export
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
