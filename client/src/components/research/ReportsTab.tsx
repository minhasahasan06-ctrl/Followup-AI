import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { queryClient, apiRequest } from '@/lib/queryClient';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from '@/components/ui/dialog';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { useToast } from '@/hooks/use-toast';
import { 
  FileText, Download, Copy, History, Search, Filter, Calendar,
  ChevronDown, ChevronRight, Loader2, ExternalLink, Eye, Edit,
  Trash2, RefreshCw, BarChart, Table, BookOpen, PieChart, TrendingUp,
  AlertCircle, CheckCircle, Clock, Users
} from 'lucide-react';
import { format, formatDistanceToNow, parseISO } from 'date-fns';
import { ResponsiveContainer, BarChart as ReBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, LineChart, Line, PieChart as RePieChart, Pie, Cell } from 'recharts';

interface AIResearchReport {
  id: string;
  studyId?: string;
  studyName?: string;
  cohortId?: string;
  cohortName?: string;
  analysisType: 'descriptive' | 'risk_prediction' | 'survival' | 'causal';
  title: string;
  status: 'draft' | 'generating' | 'complete' | 'error';
  narrative?: {
    abstract?: string;
    methods?: string;
    results?: string;
    discussion?: string;
    limitations?: string;
  };
  resultsJson?: {
    tables?: Array<{
      title: string;
      columns: string[];
      rows: Array<Record<string, string | number>>;
    }>;
    charts?: Array<{
      type: 'bar' | 'line' | 'pie';
      title: string;
      data: Array<Record<string, string | number>>;
      xKey?: string;
      yKey?: string;
      nameKey?: string;
      valueKey?: string;
    }>;
    metrics?: Record<string, number | string>;
  };
  version: number;
  parentReportId?: string;
  createdAt: string;
  updatedAt: string;
  createdBy?: string;
}

interface ReportVersion {
  id: string;
  reportId: string;
  version: number;
  snapshotJson: any;
  createdAt: string;
  notes?: string;
}

const CHART_COLORS = ['#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];

const mockReports: AIResearchReport[] = [
  {
    id: 'rep-1',
    studyId: 'study-1',
    studyName: 'Immune Recovery Study',
    analysisType: 'descriptive',
    title: 'Baseline Characteristics of Transplant Recipients',
    status: 'complete',
    narrative: {
      abstract: 'This analysis examined baseline characteristics of 342 transplant recipients enrolled in the immune recovery study. Patients were predominantly male (58%) with a median age of 52 years. The most common primary diagnosis was acute myeloid leukemia (34%), followed by non-Hodgkin lymphoma (28%).',
      methods: 'We performed a cross-sectional descriptive analysis of all patients enrolled between January 2022 and December 2023. Continuous variables are presented as mean (SD) or median (IQR) based on distribution. Categorical variables are presented as counts and percentages.',
      results: 'A total of 342 patients met inclusion criteria. Mean age was 51.7 years (SD 14.2). The cohort was 58% male. Median time from diagnosis to transplant was 8.2 months (IQR 4.1-14.7). Baseline CD4 count was 185 cells/μL (SD 142).',
      discussion: 'Our cohort reflects typical demographics for allogeneic transplant recipients. The predominance of hematologic malignancies and relatively preserved baseline immune function suggest a cohort suitable for studying immune reconstitution trajectories.',
      limitations: 'This was a single-center study which may limit generalizability. Selection bias may exist as only patients with complete baseline data were included.',
    },
    resultsJson: {
      tables: [
        {
          title: 'Baseline Demographics',
          columns: ['Characteristic', 'Value', 'N'],
          rows: [
            { Characteristic: 'Age, years (mean ± SD)', Value: '51.7 ± 14.2', N: 342 },
            { Characteristic: 'Male, n (%)', Value: '198 (58%)', N: 342 },
            { Characteristic: 'White race, n (%)', Value: '256 (75%)', N: 342 },
            { Characteristic: 'BMI, kg/m² (mean ± SD)', Value: '26.4 ± 5.1', N: 340 },
          ]
        }
      ],
      charts: [
        {
          type: 'bar',
          title: 'Primary Diagnosis Distribution',
          data: [
            { diagnosis: 'AML', count: 116 },
            { diagnosis: 'NHL', count: 96 },
            { diagnosis: 'ALL', count: 55 },
            { diagnosis: 'MDS', count: 41 },
            { diagnosis: 'Other', count: 34 },
          ],
          xKey: 'diagnosis',
          yKey: 'count',
        },
        {
          type: 'pie',
          title: 'Gender Distribution',
          data: [
            { name: 'Male', value: 198 },
            { name: 'Female', value: 144 },
          ],
          nameKey: 'name',
          valueKey: 'value',
        }
      ],
      metrics: {
        totalPatients: 342,
        meanAge: 51.7,
        medianFollowup: 18.4,
        completenessRate: 94.7,
      }
    },
    version: 1,
    createdAt: '2024-01-15T10:30:00Z',
    updatedAt: '2024-01-15T12:45:00Z',
    createdBy: 'Dr. Smith',
  },
  {
    id: 'rep-2',
    studyId: 'study-2',
    studyName: 'Environmental Impact Study',
    analysisType: 'risk_prediction',
    title: 'Predictive Model for Infection Risk',
    status: 'complete',
    narrative: {
      abstract: 'We developed and validated a machine learning model to predict 90-day infection risk in chronic care patients. The XGBoost model achieved an AUROC of 0.82 (95% CI: 0.78-0.86) with good calibration.',
      methods: 'Using data from 1,247 patients, we trained XGBoost, random forest, and logistic regression models with 5-fold cross-validation. Features included demographics, laboratory values, environmental exposures, and prior infection history.',
      results: 'XGBoost outperformed other models with AUROC 0.82, sensitivity 0.74, specificity 0.78. Top predictive features were neutrophil count, prior 30-day hospitalization, environmental AQI score, and CD4 count.',
      discussion: 'Our model demonstrates strong predictive performance for infection risk. Integration of environmental factors improved model performance by 8% compared to clinical-only models.',
    },
    resultsJson: {
      charts: [
        {
          type: 'line',
          title: 'ROC Curves by Model',
          data: [
            { fpr: 0, tpr_xgb: 0, tpr_rf: 0, tpr_lr: 0 },
            { fpr: 0.1, tpr_xgb: 0.45, tpr_rf: 0.38, tpr_lr: 0.32 },
            { fpr: 0.2, tpr_xgb: 0.65, tpr_rf: 0.55, tpr_lr: 0.48 },
            { fpr: 0.3, tpr_xgb: 0.78, tpr_rf: 0.68, tpr_lr: 0.60 },
            { fpr: 0.5, tpr_xgb: 0.88, tpr_rf: 0.80, tpr_lr: 0.72 },
            { fpr: 0.7, tpr_xgb: 0.94, tpr_rf: 0.88, tpr_lr: 0.82 },
            { fpr: 1.0, tpr_xgb: 1.0, tpr_rf: 1.0, tpr_lr: 1.0 },
          ],
          xKey: 'fpr',
          yKey: 'tpr_xgb',
        },
        {
          type: 'bar',
          title: 'Feature Importance (Top 10)',
          data: [
            { feature: 'Neutrophil Count', importance: 0.18 },
            { feature: 'Prior Hospitalization', importance: 0.15 },
            { feature: 'AQI Score', importance: 0.12 },
            { feature: 'CD4 Count', importance: 0.11 },
            { feature: 'Age', importance: 0.09 },
            { feature: 'BMI', importance: 0.07 },
            { feature: 'Comorbidity Score', importance: 0.06 },
            { feature: 'Days Post-Transplant', importance: 0.05 },
          ],
          xKey: 'feature',
          yKey: 'importance',
        }
      ],
      metrics: {
        auroc: 0.82,
        auprc: 0.71,
        sensitivity: 0.74,
        specificity: 0.78,
        precision: 0.68,
        f1Score: 0.71,
      }
    },
    version: 2,
    parentReportId: 'rep-2-v1',
    createdAt: '2024-01-20T14:00:00Z',
    updatedAt: '2024-01-22T09:15:00Z',
    createdBy: 'Dr. Johnson',
  },
  {
    id: 'rep-3',
    analysisType: 'survival',
    title: 'Survival Analysis: CD4 Recovery Milestones',
    status: 'generating',
    version: 1,
    createdAt: '2024-01-25T08:00:00Z',
    updatedAt: '2024-01-25T08:00:00Z',
  }
];

export function ReportsTab() {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedReport, setSelectedReport] = useState<AIResearchReport | null>(null);
  const [viewMode, setViewMode] = useState<'list' | 'detail'>('list');
  const [activeSection, setActiveSection] = useState<string>('abstract');
  const [duplicateDialogOpen, setDuplicateDialogOpen] = useState(false);
  const [duplicateTitle, setDuplicateTitle] = useState('');
  const [versionHistoryOpen, setVersionHistoryOpen] = useState(false);

  const { data: reports = [], isLoading, error: reportsError } = useQuery<AIResearchReport[]>({
    queryKey: ['/api/v1/research-center/reports', { type: typeFilter, status: statusFilter }],
  });

  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState<'pdf' | 'docx'>('pdf');

  const { data: reportVersions, isLoading: versionsLoading } = useQuery<ReportVersion[]>({
    queryKey: ['/api/v1/research-center/reports', selectedReport?.id, 'versions'],
    enabled: !!selectedReport && versionHistoryOpen,
  });

  const duplicateMutation = useMutation({
    mutationFn: async ({ reportId, newTitle }: { reportId: string; newTitle: string }) => {
      const response = await apiRequest('POST', `/api/v1/research-center/reports/${reportId}/duplicate`, { title: newTitle });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/reports'] });
      setDuplicateDialogOpen(false);
      setDuplicateTitle('');
      toast({ title: 'Report duplicated', description: 'You can now modify the new copy' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const exportMutation = useMutation({
    mutationFn: async ({ reportId, format }: { reportId: string; format: 'pdf' | 'docx' }) => {
      const response = await apiRequest('POST', `/api/v1/research-center/reports/${reportId}/export`, { format });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report-${reportId}.${format}`;
      a.click();
      window.URL.revokeObjectURL(url);
    },
    onSuccess: () => {
      toast({ title: 'Export complete', description: 'Report downloaded successfully' });
    },
    onError: (error: Error) => {
      toast({ title: 'Export failed', description: error.message, variant: 'destructive' });
    },
  });

  const regenerateMutation = useMutation({
    mutationFn: async (reportId: string) => {
      const response = await apiRequest('POST', `/api/v1/research-center/reports/${reportId}/regenerate`, {});
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/reports'] });
      toast({ title: 'Regeneration started', description: 'AI is generating updated narratives...' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (reportId: string) => {
      await apiRequest('DELETE', `/api/v1/research-center/reports/${reportId}`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/reports'] });
      setSelectedReport(null);
      setViewMode('list');
      toast({ title: 'Report deleted' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const restoreVersionMutation = useMutation({
    mutationFn: async ({ reportId, versionId }: { reportId: string; versionId: string }) => {
      const response = await apiRequest('POST', `/api/v1/research-center/reports/${reportId}/restore`, { versionId });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/reports'] });
      setVersionHistoryOpen(false);
      toast({ title: 'Version restored', description: 'Report has been restored to the selected version' });
    },
    onError: (error: Error) => {
      toast({ title: 'Restore failed', description: error.message, variant: 'destructive' });
    },
  });

  const handleExport = () => {
    if (selectedReport) {
      exportMutation.mutate({ reportId: selectedReport.id, format: exportFormat });
      setExportDialogOpen(false);
    }
  };

  const filteredReports = reports?.filter(report => {
    if (typeFilter !== 'all' && report.analysisType !== typeFilter) return false;
    if (statusFilter !== 'all' && report.status !== statusFilter) return false;
    if (searchQuery && !report.title.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  }) || [];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'complete': return 'default';
      case 'generating': return 'secondary';
      case 'error': return 'destructive';
      default: return 'outline';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'descriptive': return <Table className="h-4 w-4" />;
      case 'risk_prediction': return <TrendingUp className="h-4 w-4" />;
      case 'survival': return <BarChart className="h-4 w-4" />;
      case 'causal': return <PieChart className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  const renderChart = (chart: NonNullable<AIResearchReport['resultsJson']>['charts'][0]) => {
    if (chart.type === 'bar') {
      return (
        <ResponsiveContainer width="100%" height={200}>
          <ReBarChart data={chart.data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={chart.xKey} tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <ReTooltip />
            <Bar dataKey={chart.yKey || 'value'} fill="#4f46e5" radius={[4, 4, 0, 0]} />
          </ReBarChart>
        </ResponsiveContainer>
      );
    }
    if (chart.type === 'line') {
      return (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chart.data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={chart.xKey} tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <ReTooltip />
            <Line type="monotone" dataKey={chart.yKey || 'value'} stroke="#4f46e5" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      );
    }
    if (chart.type === 'pie') {
      return (
        <ResponsiveContainer width="100%" height={200}>
          <RePieChart>
            <Pie
              data={chart.data}
              dataKey={chart.valueKey || 'value'}
              nameKey={chart.nameKey || 'name'}
              cx="50%"
              cy="50%"
              outerRadius={70}
              label
            >
              {chart.data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Pie>
            <ReTooltip />
          </RePieChart>
        </ResponsiveContainer>
      );
    }
    return null;
  };

  const handleViewReport = (report: AIResearchReport) => {
    setSelectedReport(report);
    setViewMode('detail');
    setActiveSection('abstract');
  };

  const handleDuplicate = () => {
    if (selectedReport && duplicateTitle) {
      duplicateMutation.mutate({ reportId: selectedReport.id, newTitle: duplicateTitle });
    }
  };

  if (viewMode === 'detail' && selectedReport) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={() => { setViewMode('list'); setSelectedReport(null); }}
            data-testid="button-back-to-list"
          >
            <ChevronRight className="h-4 w-4 mr-2 rotate-180" />
            Back to Reports
          </Button>
          <div className="flex gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => {
                    setDuplicateTitle(`${selectedReport.title} (Copy)`);
                    setDuplicateDialogOpen(true);
                  }}
                  data-testid="button-duplicate-report"
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Duplicate & Modify</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setVersionHistoryOpen(true)}
                  data-testid="button-version-history"
                >
                  <History className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Version History</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => regenerateMutation.mutate(selectedReport.id)}
                  disabled={regenerateMutation.isPending}
                  data-testid="button-regenerate"
                >
                  {regenerateMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>Regenerate Narrative</TooltipContent>
            </Tooltip>
            <Button
              variant="outline"
              onClick={() => setExportDialogOpen(true)}
              disabled={exportMutation.isPending}
              data-testid="button-export"
            >
              {exportMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Download className="h-4 w-4 mr-2" />
              )}
              Export
            </Button>
          </div>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  {getTypeIcon(selectedReport.analysisType)}
                  <Badge variant="secondary" className="capitalize">
                    {selectedReport.analysisType.replace('_', ' ')}
                  </Badge>
                  <Badge variant={getStatusColor(selectedReport.status)}>
                    {selectedReport.status}
                  </Badge>
                  {selectedReport.version > 1 && (
                    <Badge variant="outline">v{selectedReport.version}</Badge>
                  )}
                </div>
                <CardTitle data-testid="text-report-title">{selectedReport.title}</CardTitle>
                {selectedReport.studyName && (
                  <CardDescription className="mt-1">
                    Study: {selectedReport.studyName}
                  </CardDescription>
                )}
              </div>
              <div className="text-right text-sm text-muted-foreground">
                <p>Created {format(parseISO(selectedReport.createdAt), 'PPP')}</p>
                {selectedReport.createdBy && <p>By {selectedReport.createdBy}</p>}
              </div>
            </div>
          </CardHeader>
        </Card>

        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Sections</CardTitle>
              </CardHeader>
              <CardContent className="p-2">
                <div className="space-y-1">
                  {['abstract', 'methods', 'results', 'discussion', 'limitations', 'tables', 'charts'].map((section) => (
                    <Button
                      key={section}
                      variant={activeSection === section ? 'secondary' : 'ghost'}
                      className="w-full justify-start capitalize"
                      onClick={() => setActiveSection(section)}
                      data-testid={`button-section-${section}`}
                    >
                      {section === 'tables' ? <Table className="h-4 w-4 mr-2" /> :
                       section === 'charts' ? <BarChart className="h-4 w-4 mr-2" /> :
                       <BookOpen className="h-4 w-4 mr-2" />}
                      {section}
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>

            {selectedReport.resultsJson?.metrics && (
              <Card className="mt-4">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Key Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {Object.entries(selectedReport.resultsJson.metrics).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="text-muted-foreground capitalize">
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </span>
                        <span className="font-medium">
                          {typeof value === 'number' ? value.toFixed(2) : value}
                        </span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          <div className="col-span-9">
            <Card className="min-h-[500px]">
              <CardHeader>
                <CardTitle className="text-base capitalize">{activeSection}</CardTitle>
              </CardHeader>
              <CardContent>
                {activeSection === 'tables' ? (
                  <div className="space-y-6">
                    {selectedReport.resultsJson?.tables?.map((table, idx) => (
                      <div key={idx}>
                        <h4 className="font-medium mb-2">{table.title}</h4>
                        <div className="border rounded-lg overflow-hidden">
                          <table className="w-full text-sm">
                            <thead className="bg-muted">
                              <tr>
                                {table.columns.map((col) => (
                                  <th key={col} className="px-3 py-2 text-left font-medium">{col}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {table.rows.map((row, rowIdx) => (
                                <tr key={rowIdx} className="border-t">
                                  {table.columns.map((col) => (
                                    <td key={col} className="px-3 py-2">{row[col]}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )) || (
                      <p className="text-muted-foreground text-center py-8">No tables available</p>
                    )}
                  </div>
                ) : activeSection === 'charts' ? (
                  <div className="grid grid-cols-2 gap-4">
                    {selectedReport.resultsJson?.charts?.map((chart, idx) => (
                      <Card key={idx}>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">{chart.title}</CardTitle>
                        </CardHeader>
                        <CardContent>
                          {renderChart(chart)}
                        </CardContent>
                      </Card>
                    )) || (
                      <p className="col-span-2 text-muted-foreground text-center py-8">No charts available</p>
                    )}
                  </div>
                ) : (
                  <ScrollArea className="h-[400px]">
                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      {selectedReport.narrative?.[activeSection as keyof typeof selectedReport.narrative] ? (
                        <p className="whitespace-pre-wrap leading-relaxed">
                          {selectedReport.narrative[activeSection as keyof typeof selectedReport.narrative]}
                        </p>
                      ) : (
                        <p className="text-muted-foreground text-center py-8">
                          {selectedReport.status === 'generating' ? (
                            <span className="flex items-center justify-center gap-2">
                              <Loader2 className="h-4 w-4 animate-spin" />
                              AI is generating this section...
                            </span>
                          ) : (
                            'No content available for this section'
                          )}
                        </p>
                      )}
                    </div>
                  </ScrollArea>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        <Dialog open={duplicateDialogOpen} onOpenChange={setDuplicateDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Duplicate Report</DialogTitle>
              <DialogDescription>
                Create a copy of this report that you can modify independently.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="duplicate-title">New Report Title</Label>
                <Input
                  id="duplicate-title"
                  value={duplicateTitle}
                  onChange={(e) => setDuplicateTitle(e.target.value)}
                  placeholder="Enter title for the new report"
                  data-testid="input-duplicate-title"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDuplicateDialogOpen(false)}>Cancel</Button>
              <Button
                onClick={handleDuplicate}
                disabled={!duplicateTitle || duplicateMutation.isPending}
                data-testid="button-confirm-duplicate"
              >
                {duplicateMutation.isPending ? (
                  <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Duplicating...</>
                ) : 'Duplicate Report'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={versionHistoryOpen} onOpenChange={setVersionHistoryOpen}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Version History</DialogTitle>
              <DialogDescription>
                View and restore previous versions of this report.
              </DialogDescription>
            </DialogHeader>
            <ScrollArea className="h-64">
              {versionsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => <Skeleton key={i} className="h-16 w-full" />)}
                </div>
              ) : reportVersions && reportVersions.length > 0 ? (
                <div className="space-y-2">
                  {reportVersions.map((version) => (
                    <div
                      key={version.id}
                      className="p-3 border rounded-lg hover-elevate"
                      data-testid={`card-version-${version.id}`}
                    >
                      <div className="flex items-center justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <p className="font-medium">Version {version.version}</p>
                            {version.version === selectedReport?.version && (
                              <Badge variant="secondary" className="text-xs">Current</Badge>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {format(parseISO(version.createdAt), 'PPP p')}
                          </p>
                          {version.notes && (
                            <p className="text-sm text-muted-foreground mt-1 italic">{version.notes}</p>
                          )}
                        </div>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              if (selectedReport) {
                                restoreVersionMutation.mutate({ 
                                  reportId: selectedReport.id, 
                                  versionId: version.id 
                                });
                              }
                            }}
                            disabled={version.version === selectedReport?.version || restoreVersionMutation.isPending}
                            data-testid={`button-restore-version-${version.id}`}
                          >
                            {restoreVersionMutation.isPending ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              'Restore'
                            )}
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <History className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No previous versions available</p>
                  <p className="text-sm mt-1">Versions are created when reports are regenerated or modified</p>
                </div>
              )}
            </ScrollArea>
          </DialogContent>
        </Dialog>

        <Dialog open={exportDialogOpen} onOpenChange={setExportDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Export Report</DialogTitle>
              <DialogDescription>
                Choose a format to download this report.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Export Format</Label>
                <Select value={exportFormat} onValueChange={(v) => setExportFormat(v as 'pdf' | 'docx')}>
                  <SelectTrigger data-testid="select-export-format-dialog">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pdf">PDF Document</SelectItem>
                    <SelectItem value="docx">Word Document (.docx)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="bg-muted/50 p-3 rounded-lg text-sm">
                <p className="font-medium mb-1">Export includes:</p>
                <ul className="text-muted-foreground space-y-1">
                  <li>• All narrative sections</li>
                  <li>• Tables and data summaries</li>
                  <li>• Charts and visualizations</li>
                  <li>• Key metrics</li>
                </ul>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setExportDialogOpen(false)}>Cancel</Button>
              <Button
                onClick={handleExport}
                disabled={exportMutation.isPending}
                data-testid="button-confirm-export"
              >
                {exportMutation.isPending ? (
                  <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Exporting...</>
                ) : (
                  <><Download className="h-4 w-4 mr-2" /> Download {exportFormat.toUpperCase()}</>
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2 flex-1">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search reports..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8"
              data-testid="input-search-reports"
            />
          </div>
          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger className="w-40" data-testid="select-type-filter">
              <Filter className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="descriptive">Descriptive</SelectItem>
              <SelectItem value="risk_prediction">Risk Prediction</SelectItem>
              <SelectItem value="survival">Survival</SelectItem>
              <SelectItem value="causal">Causal</SelectItem>
            </SelectContent>
          </Select>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-36" data-testid="select-status-filter">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="complete">Complete</SelectItem>
              <SelectItem value="generating">Generating</SelectItem>
              <SelectItem value="draft">Draft</SelectItem>
              <SelectItem value="error">Error</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <p className="text-sm text-muted-foreground">
          {filteredReports.length} report{filteredReports.length !== 1 ? 's' : ''}
        </p>
      </div>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-48" />
          ))}
        </div>
      ) : filteredReports.length === 0 ? (
        <Card>
          <CardContent className="p-12 text-center">
            <FileText className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
            <h3 className="text-lg font-medium mb-2">No Reports Found</h3>
            <p className="text-muted-foreground mb-4">
              {searchQuery || typeFilter !== 'all' || statusFilter !== 'all'
                ? 'Try adjusting your filters'
                : 'Generate reports from your analyses to see them here'}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredReports.map((report) => (
            <Card
              key={report.id}
              className="hover-elevate cursor-pointer transition-all"
              onClick={() => handleViewReport(report)}
              data-testid={`card-report-${report.id}`}
            >
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2">
                    {getTypeIcon(report.analysisType)}
                    <Badge variant="secondary" className="capitalize text-xs">
                      {report.analysisType.replace('_', ' ')}
                    </Badge>
                  </div>
                  <Badge variant={getStatusColor(report.status)} className="text-xs">
                    {report.status === 'generating' && (
                      <Loader2 className="h-3 w-3 animate-spin mr-1" />
                    )}
                    {report.status}
                  </Badge>
                </div>
                <CardTitle className="text-base leading-tight mt-2 line-clamp-2">
                  {report.title}
                </CardTitle>
                {report.studyName && (
                  <CardDescription className="text-xs">
                    {report.studyName}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="pb-3">
                {report.narrative?.abstract && (
                  <p className="text-xs text-muted-foreground line-clamp-3">
                    {report.narrative.abstract}
                  </p>
                )}
              </CardContent>
              <CardFooter className="pt-0 flex items-center justify-between gap-2 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  {formatDistanceToNow(parseISO(report.createdAt), { addSuffix: true })}
                </div>
                {report.version > 1 && (
                  <Badge variant="outline" className="text-xs">v{report.version}</Badge>
                )}
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
