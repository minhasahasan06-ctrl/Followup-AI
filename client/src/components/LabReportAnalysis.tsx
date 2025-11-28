import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { 
  FlaskConical,
  TestTube,
  TrendingUp,
  TrendingDown,
  Minus,
  Loader2,
  AlertTriangle,
  CheckCircle,
  AlertCircle,
  Clock,
  Plus,
  Trash2,
  Activity,
  RefreshCw,
  ChevronDown,
  ChevronUp
} from "lucide-react";
import { format, parseISO } from "date-fns";

interface PatientContext {
  id: string;
  firstName: string;
  lastName: string;
  allergies?: string[];
  comorbidities?: string[];
  currentMedications?: string[];
}

interface LabResult {
  name: string;
  value: number;
  unit: string;
  normalRange: {
    low: number;
    high: number;
  };
}

interface AnalyzedResult extends LabResult {
  status: 'normal' | 'abnormal' | 'critical';
  isAbnormal: boolean;
  deviation?: 'low' | 'high' | null;
  deviationPercent?: number | null;
}

interface LabAnalysis {
  id: string;
  panelType: string;
  results: AnalyzedResult[];
  summary: {
    totalTests: number;
    abnormalCount: number;
    criticalCount: number;
    overallStatus: 'normal' | 'review' | 'critical';
  };
  interpretation: string;
  trends: any[];
  analyzedAt: string;
  _fallback?: boolean;
}

interface LabPanel {
  id: string;
  panelType: string;
  date: string;
  results: AnalyzedResult[];
  status: string;
}

interface LabReportAnalysisProps {
  patientContext: PatientContext;
  className?: string;
}

const panelTypes = [
  { value: 'cbc', label: 'Complete Blood Count (CBC)' },
  { value: 'cmp', label: 'Comprehensive Metabolic Panel' },
  { value: 'bmp', label: 'Basic Metabolic Panel' },
  { value: 'lipid', label: 'Lipid Panel' },
  { value: 'thyroid', label: 'Thyroid Panel' },
  { value: 'liver', label: 'Liver Function Tests' },
  { value: 'kidney', label: 'Kidney Function Tests' },
  { value: 'coagulation', label: 'Coagulation Panel' },
  { value: 'urinalysis', label: 'Urinalysis' },
  { value: 'custom', label: 'Custom Panel' }
];

const defaultLabTests: Record<string, LabResult[]> = {
  cbc: [
    { name: 'WBC', value: 0, unit: 'K/uL', normalRange: { low: 4.5, high: 11.0 } },
    { name: 'RBC', value: 0, unit: 'M/uL', normalRange: { low: 4.2, high: 5.4 } },
    { name: 'Hemoglobin', value: 0, unit: 'g/dL', normalRange: { low: 12.0, high: 16.0 } },
    { name: 'Hematocrit', value: 0, unit: '%', normalRange: { low: 36, high: 46 } },
    { name: 'Platelets', value: 0, unit: 'K/uL', normalRange: { low: 150, high: 400 } }
  ],
  cmp: [
    { name: 'Glucose', value: 0, unit: 'mg/dL', normalRange: { low: 70, high: 100 } },
    { name: 'BUN', value: 0, unit: 'mg/dL', normalRange: { low: 7, high: 20 } },
    { name: 'Creatinine', value: 0, unit: 'mg/dL', normalRange: { low: 0.7, high: 1.3 } },
    { name: 'Sodium', value: 0, unit: 'mEq/L', normalRange: { low: 136, high: 145 } },
    { name: 'Potassium', value: 0, unit: 'mEq/L', normalRange: { low: 3.5, high: 5.0 } },
    { name: 'Chloride', value: 0, unit: 'mEq/L', normalRange: { low: 98, high: 106 } },
    { name: 'CO2', value: 0, unit: 'mEq/L', normalRange: { low: 23, high: 29 } }
  ],
  lipid: [
    { name: 'Total Cholesterol', value: 0, unit: 'mg/dL', normalRange: { low: 0, high: 200 } },
    { name: 'LDL', value: 0, unit: 'mg/dL', normalRange: { low: 0, high: 100 } },
    { name: 'HDL', value: 0, unit: 'mg/dL', normalRange: { low: 40, high: 999 } },
    { name: 'Triglycerides', value: 0, unit: 'mg/dL', normalRange: { low: 0, high: 150 } }
  ]
};

export function LabReportAnalysis({ patientContext, className }: LabReportAnalysisProps) {
  const [selectedPanelType, setSelectedPanelType] = useState<string>('');
  const [labResults, setLabResults] = useState<LabResult[]>([]);
  const [clinicalContext, setClinicalContext] = useState('');
  const [currentAnalysis, setCurrentAnalysis] = useState<LabAnalysis | null>(null);
  const [expandedPanel, setExpandedPanel] = useState<string | null>(null);
  const { toast } = useToast();

  const { data: labHistory, isLoading: historyLoading, refetch: refetchHistory } = useQuery({
    queryKey: ['/api/v1/lysa/lab-history', patientContext.id],
    queryFn: async () => {
      const response = await fetch(`/api/v1/lysa/lab-history/${patientContext.id}`);
      if (!response.ok) {
        return { panels: [], totalCount: 0 };
      }
      return response.json();
    },
    enabled: !!patientContext.id
  });

  const analyzeLabMutation = useMutation({
    mutationFn: async (data: { labResults: LabResult[]; panelType: string; clinicalContext?: string }) => {
      const response = await apiRequest('/api/v1/lysa/lab-analysis', {
        method: 'POST',
        body: JSON.stringify({
          patientId: patientContext.id,
          ...data
        })
      });
      return response;
    },
    onSuccess: (data: any) => {
      setCurrentAnalysis(data.analysis);
      toast({
        title: "Analysis Complete",
        description: data.analysis._fallback 
          ? "AI interpretation temporarily unavailable - showing status analysis"
          : "AI-powered lab analysis generated successfully"
      });
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to analyze lab results",
        variant: "destructive"
      });
    }
  });

  const handlePanelTypeChange = (value: string) => {
    setSelectedPanelType(value);
    if (defaultLabTests[value]) {
      setLabResults([...defaultLabTests[value]]);
    } else {
      setLabResults([]);
    }
  };

  const updateLabValue = (index: number, value: string) => {
    const newResults = [...labResults];
    newResults[index] = { ...newResults[index], value: parseFloat(value) || 0 };
    setLabResults(newResults);
  };

  const addLabTest = () => {
    setLabResults([...labResults, { 
      name: '', 
      value: 0, 
      unit: '', 
      normalRange: { low: 0, high: 0 } 
    }]);
  };

  const removeLabTest = (index: number) => {
    setLabResults(labResults.filter((_, i) => i !== index));
  };

  const updateLabTest = (index: number, field: string, value: any) => {
    const newResults = [...labResults];
    if (field.startsWith('normalRange.')) {
      const rangeField = field.split('.')[1] as 'low' | 'high';
      newResults[index] = {
        ...newResults[index],
        normalRange: { ...newResults[index].normalRange, [rangeField]: parseFloat(value) || 0 }
      };
    } else {
      newResults[index] = { ...newResults[index], [field]: field === 'value' ? parseFloat(value) || 0 : value };
    }
    setLabResults(newResults);
  };

  const handleAnalyze = () => {
    if (!selectedPanelType) {
      toast({
        title: "Select Panel Type",
        description: "Please select a lab panel type",
        variant: "destructive"
      });
      return;
    }

    if (labResults.length === 0) {
      toast({
        title: "Add Lab Results",
        description: "Please add at least one lab result to analyze",
        variant: "destructive"
      });
      return;
    }

    const validResults = labResults.filter(r => r.name && r.value > 0);
    if (validResults.length === 0) {
      toast({
        title: "Enter Values",
        description: "Please enter values for the lab tests",
        variant: "destructive"
      });
      return;
    }

    analyzeLabMutation.mutate({
      labResults: validResults,
      panelType: panelTypes.find(t => t.value === selectedPanelType)?.label || selectedPanelType,
      clinicalContext
    });
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'normal':
        return <Badge variant="secondary" className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"><CheckCircle className="h-3 w-3 mr-1" />Normal</Badge>;
      case 'abnormal':
        return <Badge variant="secondary" className="bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"><AlertCircle className="h-3 w-3 mr-1" />Abnormal</Badge>;
      case 'critical':
        return <Badge variant="destructive"><AlertTriangle className="h-3 w-3 mr-1" />Critical</Badge>;
      case 'review':
        return <Badge variant="secondary" className="bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400"><AlertCircle className="h-3 w-3 mr-1" />Review</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing':
        return <TrendingUp className="h-4 w-4 text-red-500" />;
      case 'decreasing':
        return <TrendingDown className="h-4 w-4 text-blue-500" />;
      default:
        return <Minus className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <div className={className}>
      <Tabs defaultValue="analyze" className="space-y-4">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="analyze" data-testid="tab-lab-analyze">
            <FlaskConical className="h-4 w-4 mr-2" />
            New Analysis
          </TabsTrigger>
          <TabsTrigger value="history" data-testid="tab-lab-history">
            <TestTube className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="analyze" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                  <FlaskConical className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <CardTitle className="text-lg">AI-Powered Lab Analysis</CardTitle>
                  <CardDescription>
                    Enter lab results for AI interpretation for {patientContext.firstName} {patientContext.lastName}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="panel-type">Panel Type *</Label>
                  <Select value={selectedPanelType} onValueChange={handlePanelTypeChange}>
                    <SelectTrigger id="panel-type" data-testid="select-panel-type">
                      <SelectValue placeholder="Select panel type" />
                    </SelectTrigger>
                    <SelectContent>
                      {panelTypes.map(type => (
                        <SelectItem key={type.value} value={type.value}>{type.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="lab-context">Clinical Context</Label>
                  <Input
                    id="lab-context"
                    placeholder="e.g., Follow-up after medication change"
                    value={clinicalContext}
                    onChange={(e) => setClinicalContext(e.target.value)}
                    data-testid="input-lab-context"
                  />
                </div>
              </div>

              {labResults.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Lab Results</Label>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={addLabTest}
                      data-testid="button-add-lab-test"
                    >
                      <Plus className="h-4 w-4 mr-1" />
                      Add Test
                    </Button>
                  </div>
                  
                  <ScrollArea className="h-[300px] border rounded-md p-3">
                    <div className="space-y-3">
                      {labResults.map((result, index) => (
                        <div key={index} className="grid gap-2 grid-cols-12 items-end border-b pb-3">
                          <div className="col-span-3">
                            <Label className="text-xs">Test Name</Label>
                            <Input
                              value={result.name}
                              onChange={(e) => updateLabTest(index, 'name', e.target.value)}
                              placeholder="Test name"
                              className="h-8 text-sm"
                              data-testid={`input-lab-name-${index}`}
                            />
                          </div>
                          <div className="col-span-2">
                            <Label className="text-xs">Value *</Label>
                            <Input
                              type="number"
                              value={result.value || ''}
                              onChange={(e) => updateLabValue(index, e.target.value)}
                              placeholder="0"
                              className="h-8 text-sm"
                              data-testid={`input-lab-value-${index}`}
                            />
                          </div>
                          <div className="col-span-2">
                            <Label className="text-xs">Unit</Label>
                            <Input
                              value={result.unit}
                              onChange={(e) => updateLabTest(index, 'unit', e.target.value)}
                              placeholder="unit"
                              className="h-8 text-sm"
                              data-testid={`input-lab-unit-${index}`}
                            />
                          </div>
                          <div className="col-span-2">
                            <Label className="text-xs">Range Low</Label>
                            <Input
                              type="number"
                              value={result.normalRange.low || ''}
                              onChange={(e) => updateLabTest(index, 'normalRange.low', e.target.value)}
                              placeholder="0"
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="col-span-2">
                            <Label className="text-xs">Range High</Label>
                            <Input
                              type="number"
                              value={result.normalRange.high || ''}
                              onChange={(e) => updateLabTest(index, 'normalRange.high', e.target.value)}
                              placeholder="0"
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="col-span-1">
                            <Button 
                              variant="ghost" 
                              size="icon"
                              className="h-8 w-8"
                              onClick={() => removeLabTest(index)}
                              data-testid={`button-remove-lab-${index}`}
                            >
                              <Trash2 className="h-4 w-4 text-muted-foreground" />
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              )}

              <Button 
                onClick={handleAnalyze}
                disabled={analyzeLabMutation.isPending || !selectedPanelType || labResults.length === 0}
                className="w-full"
                data-testid="button-analyze-labs"
              >
                {analyzeLabMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity className="h-4 w-4 mr-2" />
                    Analyze Lab Results
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {currentAnalysis && (
            <Card className="border-primary/50">
              <CardHeader>
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div className="flex items-center gap-2">
                    <TestTube className="h-5 w-5 text-primary" />
                    <CardTitle className="text-lg">{currentAnalysis.panelType} Analysis</CardTitle>
                  </div>
                  {getStatusBadge(currentAnalysis.summary.overallStatus)}
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {currentAnalysis._fallback && (
                  <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
                    <div className="flex items-center gap-2 text-yellow-700 dark:text-yellow-400">
                      <AlertTriangle className="h-4 w-4" />
                      <span className="font-medium text-sm">AI Interpretation Temporarily Unavailable</span>
                    </div>
                    <p className="text-xs text-yellow-600 dark:text-yellow-500 mt-1">
                      Showing status analysis only. Full clinical interpretation unavailable.
                    </p>
                  </div>
                )}

                <div className="grid gap-4 md:grid-cols-4">
                  <Card className="bg-muted/50">
                    <CardContent className="p-3 text-center">
                      <p className="text-2xl font-bold">{currentAnalysis.summary.totalTests}</p>
                      <p className="text-xs text-muted-foreground">Total Tests</p>
                    </CardContent>
                  </Card>
                  <Card className="bg-green-50 dark:bg-green-900/20">
                    <CardContent className="p-3 text-center">
                      <p className="text-2xl font-bold text-green-600">{currentAnalysis.summary.totalTests - currentAnalysis.summary.abnormalCount}</p>
                      <p className="text-xs text-muted-foreground">Normal</p>
                    </CardContent>
                  </Card>
                  <Card className="bg-yellow-50 dark:bg-yellow-900/20">
                    <CardContent className="p-3 text-center">
                      <p className="text-2xl font-bold text-yellow-600">{currentAnalysis.summary.abnormalCount}</p>
                      <p className="text-xs text-muted-foreground">Abnormal</p>
                    </CardContent>
                  </Card>
                  <Card className="bg-red-50 dark:bg-red-900/20">
                    <CardContent className="p-3 text-center">
                      <p className="text-2xl font-bold text-red-600">{currentAnalysis.summary.criticalCount}</p>
                      <p className="text-xs text-muted-foreground">Critical</p>
                    </CardContent>
                  </Card>
                </div>

                <div className="space-y-2">
                  <h4 className="font-semibold text-sm">Results Summary</h4>
                  <div className="border rounded-md divide-y">
                    {currentAnalysis.results.map((result, index) => (
                      <div key={index} className="flex items-center justify-between p-3">
                        <div className="flex items-center gap-3">
                          {getTrendIcon(currentAnalysis.trends[index]?.trend || 'stable')}
                          <div>
                            <p className="font-medium text-sm">{result.name}</p>
                            <p className="text-xs text-muted-foreground">
                              Normal: {result.normalRange.low} - {result.normalRange.high} {result.unit}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className={`font-mono text-sm ${result.isAbnormal ? 'text-red-600 font-bold' : ''}`}>
                            {result.value} {result.unit}
                          </span>
                          {getStatusBadge(result.status)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="font-semibold text-sm">AI Interpretation</h4>
                  <div className="text-sm text-muted-foreground bg-muted/50 p-4 rounded-md whitespace-pre-wrap">
                    {currentAnalysis.interpretation}
                  </div>
                </div>

                <div className="border-t pt-3">
                  <p className="text-xs text-muted-foreground">
                    Analyzed: {format(parseISO(currentAnalysis.analyzedAt), 'MMM d, yyyy h:mm a')}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <CardTitle className="text-lg">Lab History</CardTitle>
                  <CardDescription>
                    Previous lab panels for {patientContext.firstName} {patientContext.lastName}
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={() => refetchHistory()} data-testid="button-refresh-labs">
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {historyLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map(i => (
                    <Skeleton key={i} className="h-24 w-full" />
                  ))}
                </div>
              ) : labHistory?.panels?.length > 0 ? (
                <ScrollArea className="h-[500px]">
                  <div className="space-y-3">
                    {labHistory.panels.map((panel: LabPanel) => (
                      <Card key={panel.id} className="hover-elevate">
                        <CardContent className="p-4">
                          <div 
                            className="flex items-center justify-between cursor-pointer"
                            onClick={() => setExpandedPanel(expandedPanel === panel.id ? null : panel.id)}
                          >
                            <div className="flex items-center gap-3">
                              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                                <FlaskConical className="h-5 w-5 text-primary" />
                              </div>
                              <div>
                                <p className="font-medium">{panel.panelType}</p>
                                <p className="text-sm text-muted-foreground">
                                  {format(parseISO(panel.date), 'MMM d, yyyy')} • {panel.results.length} tests
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              {getStatusBadge(panel.status)}
                              {expandedPanel === panel.id ? (
                                <ChevronUp className="h-4 w-4 text-muted-foreground" />
                              ) : (
                                <ChevronDown className="h-4 w-4 text-muted-foreground" />
                              )}
                            </div>
                          </div>
                          
                          {expandedPanel === panel.id && (
                            <div className="mt-4 pt-4 border-t">
                              <div className="grid gap-2">
                                {panel.results.map((result, idx) => (
                                  <div key={idx} className="flex items-center justify-between text-sm">
                                    <span>{result.name}</span>
                                    <div className="flex items-center gap-2">
                                      <span className={result.status !== 'normal' ? 'font-bold text-red-600' : ''}>
                                        {result.value} {result.unit}
                                      </span>
                                      <Badge variant="outline" className="text-xs">
                                        {result.normalRange.low}-{result.normalRange.high}
                                      </Badge>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <TestTube className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No lab panels found</p>
                  <p className="text-sm">Enter results using the "New Analysis" tab</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
