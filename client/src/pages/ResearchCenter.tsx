import { useState, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList, CommandSeparator } from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { 
  TrendingUp, 
  BarChart, 
  FileText, 
  Bot, 
  Beaker, 
  Users, 
  Download, 
  Upload,
  Activity,
  Brain,
  ChartPie,
  Calendar,
  CheckCircle2,
  Clock,
  AlertCircle,
  Plus,
  Filter,
  RefreshCw,
  Loader2,
  Eye,
  Database,
  CalendarDays,
  Search,
  Lock,
  ShieldAlert,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { AIResearchReport } from "@shared/schema";
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart as ReBarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, LineChart, Line, Legend } from "recharts";
import CohortBuilderTab from "@/components/research/CohortBuilderTab";
import { StudiesTab } from "@/components/research/StudiesTab";
import { AIAnalysisTab } from "@/components/research/AIAnalysisTab";
import { AlertsTab } from "@/components/research/AlertsTab";
import { DailyFollowupsTab } from "@/components/research/DailyFollowupsTab";
import { ReportsTab } from "@/components/research/ReportsTab";
import { PersonalResearchTab } from "@/components/research/PersonalResearchTab";
import { AdvancedAnalyticsTab } from "@/components/research/EpidemiologyTab";
import { ExportsTab } from "@/components/research/ExportsTab";
import { DatasetsTab } from "@/components/research/DatasetsTab";
import { NLPRedactionTab } from "@/components/research/NLPRedactionTab";
import { ResearchQATab } from "@/components/research/ResearchQATab";
import { NLCohortBuilderTab } from "@/components/research/NLCohortBuilderTab";
import { ProtocolGeneratorTab } from "@/components/research/ProtocolGeneratorTab";
import { TrialEmulationTab } from "@/components/research/TrialEmulationTab";
import { Sparkles, FlaskConical, ScrollText } from "lucide-react";

const COHORT_COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00C49F', '#FFBB28'];

interface CohortStats {
  total_patients: number;
  consenting_patients: number;
  average_age: number;
  gender_distribution: Record<string, number>;
  condition_distribution: Record<string, number>;
  data_completeness: number;
  active_studies: number;
}

interface ResearchStudy {
  id: string;
  title: string;
  description: string;
  status: 'draft' | 'active' | 'completed' | 'paused';
  cohort_size: number;
  start_date: string;
  end_date?: string;
  principal_investigator: string;
  data_types: string[];
}

interface ImportableDataType {
  type: string;
  label: string;
  requiredFields: string[];
  optionalFields: string[];
}

interface CSVPreview {
  headers: string[];
  sampleRows: string[][];
  totalRows: number;
}

interface ImportResult {
  success: boolean;
  imported: number;
  skipped: number;
  errors: { row: number; message: string }[];
}

function CSVImportTab() {
  const { toast } = useToast();
  const [step, setStep] = useState<'upload' | 'map' | 'review' | 'result'>('upload');
  const [csvData, setCsvData] = useState<string>('');
  const [selectedDataType, setSelectedDataType] = useState<string>('');
  const [selectedStudyId, setSelectedStudyId] = useState<string>('');
  const [preview, setPreview] = useState<CSVPreview | null>(null);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [validationResult, setValidationResult] = useState<{ valid: boolean; errors: string[]; warnings: string[] } | null>(null);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);

  const { data: dataTypes } = useQuery<ImportableDataType[]>({
    queryKey: ['/api/v1/research-center/import/data-types'],
  });

  const { data: studies } = useQuery<ResearchStudy[]>({
    queryKey: ['/api/research/studies'],
  });

  const previewMutation = useMutation({
    mutationFn: async (data: string) => {
      const response = await apiRequest('/api/v1/research-center/import/preview', { method: 'POST', json: { csvData: data } });
      return await response.json();
    },
    onSuccess: (data: CSVPreview) => {
      setPreview(data);
      if (data.headers.length > 0) {
        setStep('map');
      }
    },
    onError: (error: Error) => {
      toast({ title: 'Error parsing CSV', description: error.message, variant: 'destructive' });
    },
  });

  const validateMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('/api/v1/research-center/import/validate-mapping', {
        method: 'POST',
        json: { dataType: selectedDataType, mapping, headers: preview?.headers || [] }
      });
      return await response.json();
    },
    onSuccess: (data) => {
      setValidationResult(data);
      if (data.valid) {
        setStep('review');
      }
    },
    onError: (error: Error) => {
      toast({ title: 'Validation Error', description: error.message, variant: 'destructive' });
    },
  });

  const importMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('/api/v1/research-center/import/execute', {
        method: 'POST',
        json: { dataType: selectedDataType, csvData, mapping, studyId: selectedStudyId || null }
      });
      return await response.json();
    },
    onSuccess: (data: ImportResult) => {
      setImportResult(data);
      setStep('result');
      if (data.success) {
        toast({ title: 'Import Complete', description: `${data.imported} records imported successfully` });
      }
    },
    onError: (error: Error) => {
      toast({ title: 'Import Failed', description: error.message, variant: 'destructive' });
    },
  });

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        setCsvData(text);
        previewMutation.mutate(text);
      };
      reader.readAsText(file);
    }
  };

  const handleMappingChange = (field: string, csvColumn: string) => {
    setMapping(prev => ({ ...prev, [field]: csvColumn }));
  };

  const resetImport = () => {
    setStep('upload');
    setCsvData('');
    setSelectedDataType('');
    setSelectedStudyId('');
    setPreview(null);
    setMapping({});
    setValidationResult(null);
    setImportResult(null);
  };

  const selectedType = dataTypes?.find(t => t.type === selectedDataType);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" />
          CSV Data Import
        </CardTitle>
        <CardDescription>
          Import research data from CSV files with column mapping
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2 mb-6">
          <Badge variant={step === 'upload' ? 'default' : 'secondary'}>1. Upload</Badge>
          <Badge variant={step === 'map' ? 'default' : 'secondary'}>2. Map Columns</Badge>
          <Badge variant={step === 'review' ? 'default' : 'secondary'}>3. Review</Badge>
          <Badge variant={step === 'result' ? 'default' : 'secondary'}>4. Result</Badge>
        </div>

        {step === 'upload' && (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="dataType">Data Type</Label>
              <Select value={selectedDataType} onValueChange={setSelectedDataType}>
                <SelectTrigger data-testid="select-data-type">
                  <SelectValue placeholder="Select data type to import" />
                </SelectTrigger>
                <SelectContent>
                  {dataTypes?.map(dt => (
                    <SelectItem key={dt.type} value={dt.type}>{dt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {['study_enrollments', 'visits'].includes(selectedDataType) && (
              <div className="space-y-2">
                <Label htmlFor="studyId">Study (Optional)</Label>
                <Select value={selectedStudyId} onValueChange={setSelectedStudyId}>
                  <SelectTrigger data-testid="select-study">
                    <SelectValue placeholder="Select study" />
                  </SelectTrigger>
                  <SelectContent>
                    {studies?.map(s => (
                      <SelectItem key={s.id} value={s.id}>{s.title}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {selectedType && (
              <div className="bg-muted/50 p-4 rounded-lg space-y-2">
                <p className="text-sm font-medium">Required Fields:</p>
                <div className="flex flex-wrap gap-2">
                  {selectedType.requiredFields.map(f => (
                    <Badge key={f} variant="outline">{f}</Badge>
                  ))}
                </div>
                {selectedType.optionalFields.length > 0 && (
                  <>
                    <p className="text-sm font-medium mt-2">Optional Fields:</p>
                    <div className="flex flex-wrap gap-2">
                      {selectedType.optionalFields.map(f => (
                        <Badge key={f} variant="secondary">{f}</Badge>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="csvFile">Upload CSV File</Label>
              <Input
                id="csvFile"
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                disabled={!selectedDataType || previewMutation.isPending}
                data-testid="input-csv-file"
              />
            </div>

            {previewMutation.isPending && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Parsing CSV...
              </div>
            )}
          </div>
        )}

        {step === 'map' && preview && selectedType && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                {preview.totalRows} rows found with {preview.headers.length} columns
              </p>
              <Button variant="ghost" size="sm" onClick={resetImport} data-testid="button-reset-import">
                Start Over
              </Button>
            </div>

            <div className="border rounded-lg overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Field</TableHead>
                    <TableHead>CSV Column</TableHead>
                    <TableHead>Required</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...selectedType.requiredFields, ...selectedType.optionalFields].map(field => (
                    <TableRow key={field}>
                      <TableCell className="font-medium">{field}</TableCell>
                      <TableCell>
                        <Select 
                          value={mapping[field] || ''} 
                          onValueChange={(v) => handleMappingChange(field, v)}
                        >
                          <SelectTrigger className="w-48" data-testid={`select-map-${field}`}>
                            <SelectValue placeholder="Select column" />
                          </SelectTrigger>
                          <SelectContent>
                            {preview.headers.map(h => (
                              <SelectItem key={h} value={h}>{h}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </TableCell>
                      <TableCell>
                        {selectedType.requiredFields.includes(field) ? (
                          <Badge variant="destructive">Required</Badge>
                        ) : (
                          <Badge variant="secondary">Optional</Badge>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium">Sample Data Preview:</p>
              <div className="border rounded-lg overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      {preview.headers.map(h => (
                        <TableHead key={h}>{h}</TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {preview.sampleRows.slice(0, 3).map((row, i) => (
                      <TableRow key={i}>
                        {row.map((cell, j) => (
                          <TableCell key={j} className="text-sm">{cell}</TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>

            {validationResult && !validationResult.valid && (
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 space-y-2">
                <p className="text-sm font-medium text-destructive">Validation Errors:</p>
                {validationResult.errors.map((e, i) => (
                  <p key={i} className="text-sm text-destructive">{e}</p>
                ))}
              </div>
            )}

            <div className="flex gap-2">
              <Button variant="outline" onClick={resetImport} data-testid="button-back-upload">
                Back
              </Button>
              <Button 
                onClick={() => validateMutation.mutate()} 
                disabled={validateMutation.isPending}
                data-testid="button-validate-mapping"
              >
                {validateMutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
                Validate Mapping
              </Button>
            </div>
          </div>
        )}

        {step === 'review' && preview && (
          <div className="space-y-4">
            <div className="bg-muted/50 p-4 rounded-lg space-y-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                <p className="font-medium">Ready to Import</p>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div><span className="text-muted-foreground">Data Type:</span> {selectedType?.label}</div>
                <div><span className="text-muted-foreground">Total Rows:</span> {preview.totalRows}</div>
                <div><span className="text-muted-foreground">Fields Mapped:</span> {Object.keys(mapping).length}</div>
                {selectedStudyId && <div><span className="text-muted-foreground">Study:</span> {studies?.find(s => s.id === selectedStudyId)?.title}</div>}
              </div>
            </div>

            {validationResult?.warnings && validationResult.warnings.length > 0 && (
              <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4 space-y-2">
                <p className="text-sm font-medium text-yellow-700">Warnings:</p>
                {validationResult.warnings.map((w, i) => (
                  <p key={i} className="text-sm text-yellow-700">{w}</p>
                ))}
              </div>
            )}

            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setStep('map')} data-testid="button-back-map">
                Back to Mapping
              </Button>
              <Button 
                onClick={() => importMutation.mutate()} 
                disabled={importMutation.isPending}
                data-testid="button-execute-import"
              >
                {importMutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
                Import Data
              </Button>
            </div>
          </div>
        )}

        {step === 'result' && importResult && (
          <div className="space-y-4">
            <div className={`p-4 rounded-lg ${importResult.success ? 'bg-green-500/10 border border-green-500/20' : 'bg-destructive/10 border border-destructive/20'}`}>
              <div className="flex items-center gap-2 mb-3">
                {importResult.success ? (
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-destructive" />
                )}
                <p className="font-medium">{importResult.success ? 'Import Successful' : 'Import Completed with Issues'}</p>
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div><span className="text-muted-foreground">Imported:</span> <span className="font-medium text-green-600">{importResult.imported}</span></div>
                <div><span className="text-muted-foreground">Skipped:</span> <span className="font-medium text-yellow-600">{importResult.skipped}</span></div>
                <div><span className="text-muted-foreground">Errors:</span> <span className="font-medium text-red-600">{importResult.errors.length}</span></div>
              </div>
            </div>

            {importResult.errors.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm font-medium">Errors (first 10):</p>
                <ScrollArea className="h-40 border rounded-lg p-2">
                  {importResult.errors.slice(0, 10).map((e, i) => (
                    <p key={i} className="text-sm text-destructive">Row {e.row}: {e.message}</p>
                  ))}
                </ScrollArea>
              </div>
            )}

            <Button onClick={resetImport} data-testid="button-new-import">
              Start New Import
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

type ResearchRole = 'admin' | 'researcher' | 'viewer';

interface SearchResult {
  type: 'study' | 'patient' | 'report' | 'cohort';
  id: string;
  title: string;
  description?: string;
}

export default function ResearchCenter() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [reportTitle, setReportTitle] = useState("");
  const [analysisType, setAnalysisType] = useState<string>("");
  const [createStudyOpen, setCreateStudyOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [newStudy, setNewStudy] = useState({
    title: '',
    description: '',
    dataTypes: [] as string[],
  });

  const userRole: ResearchRole = useMemo(() => {
    if (!user) return 'viewer';
    if (user.role === 'admin') return 'admin';
    if (user.role === 'doctor') return 'researcher';
    return 'viewer';
  }, [user]);

  const permissions = useMemo(() => ({
    canCreateStudies: userRole === 'admin' || userRole === 'researcher',
    canImportData: userRole === 'admin' || userRole === 'researcher',
    canRunAnalysis: userRole === 'admin' || userRole === 'researcher',
    canManageAlerts: userRole === 'admin' || userRole === 'researcher',
    canViewReports: true,
    canExportData: userRole === 'admin' || userRole === 'researcher',
    canAccessPersonalResearch: userRole === 'admin' || userRole === 'researcher',
  }), [userRole]);

  const { data: reports, isLoading } = useQuery<AIResearchReport[]>({
    queryKey: ["/api/doctor/research-reports"],
  });

  const { data: cohortStats } = useQuery<CohortStats>({
    queryKey: ['/api/research/cohort-stats'],
  });

  const { data: studies, isLoading: studiesLoading } = useQuery<ResearchStudy[]>({
    queryKey: ['/api/research/studies'],
  });

  const { data: searchResults } = useQuery<SearchResult[]>({
    queryKey: ['/api/research/search', searchQuery],
    enabled: searchQuery.length > 2,
  });

  const filteredResults = useMemo(() => {
    if (!searchResults) return [];
    return searchResults.filter(r => 
      r.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.description?.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [searchResults, searchQuery]);

  const handleSearchSelect = (result: SearchResult) => {
    setSearchOpen(false);
    setSearchQuery("");
    toast({
      title: `Opening ${result.type}`,
      description: result.title,
    });
  };

  const generateReportMutation = useMutation({
    mutationFn: async (data: { title: string; analysisType: string }) => {
      const res = await apiRequest("/api/doctor/research-reports", { method: "POST", json: data });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/doctor/research-reports"] });
      toast({
        title: "Research report generated",
        description: "Your AI-powered research analysis is ready",
      });
      setReportTitle("");
      setAnalysisType("");
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to generate report",
        variant: "destructive",
      });
    },
  });

  const createStudyMutation = useMutation({
    mutationFn: async (data: typeof newStudy) => {
      const res = await apiRequest("/api/research/studies", { method: "POST", json: data });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/research/studies'] });
      toast({ title: 'Study Created', description: 'New research study has been created' });
      setCreateStudyOpen(false);
      setNewStudy({ title: '', description: '', dataTypes: [] });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const handleGenerateReport = () => {
    if (!reportTitle.trim() || !analysisType) {
      toast({
        title: "Missing information",
        description: "Please provide a title and select an analysis type",
        variant: "destructive",
      });
      return;
    }

    generateReportMutation.mutate({
      title: reportTitle,
      analysisType,
    });
  };

  const genderData = cohortStats?.gender_distribution 
    ? Object.entries(cohortStats.gender_distribution).map(([name, value]) => ({ name, value }))
    : [{ name: 'Male', value: 45 }, { name: 'Female', value: 52 }, { name: 'Other', value: 3 }];

  const conditionData = cohortStats?.condition_distribution
    ? Object.entries(cohortStats.condition_distribution).map(([name, value]) => ({ name, value }))
    : [
        { name: 'Post-Transplant', value: 35 },
        { name: 'Autoimmune', value: 28 },
        { name: 'Cancer Treatment', value: 22 },
        { name: 'Primary Immunodeficiency', value: 15 },
      ];

  const ageDistribution = [
    { range: '18-30', count: 25 },
    { range: '31-45', count: 40 },
    { range: '46-60', count: 55 },
    { range: '61-75', count: 30 },
  ];

  const enrollmentTrend = [
    { month: 'Aug', patients: 85 },
    { month: 'Sep', patients: 102 },
    { month: 'Oct', patients: 128 },
    { month: 'Nov', patients: 145 },
    { month: 'Dec', patients: 150 },
  ];

  return (
    <div className="space-y-6" data-testid="page-research-center">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Beaker className="h-6 w-6 text-purple-500" />
            Epidemiology Research Center
          </h1>
          <p className="text-muted-foreground">
            AI-powered research analysis and data visualization for chronic care patient populations
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge 
            variant={userRole === 'admin' ? 'default' : userRole === 'researcher' ? 'secondary' : 'outline'}
            className="gap-1"
            data-testid="badge-user-role"
          >
            {userRole === 'admin' && <ShieldAlert className="h-3 w-3" />}
            {userRole === 'viewer' && <Eye className="h-3 w-3" />}
            {userRole === 'researcher' && <Beaker className="h-3 w-3" />}
            {userRole.charAt(0).toUpperCase() + userRole.slice(1)}
          </Badge>
          
          <Popover open={searchOpen} onOpenChange={setSearchOpen}>
            <PopoverTrigger asChild>
              <Button variant="outline" className="w-64 justify-start gap-2" data-testid="button-global-search">
                <Search className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Search studies, patients, reports...</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80 p-0" align="end">
              <Command>
                <CommandInput 
                  placeholder="Search..." 
                  value={searchQuery}
                  onValueChange={setSearchQuery}
                  data-testid="input-global-search"
                />
                <CommandList>
                  <CommandEmpty>
                    {searchQuery.length < 3 
                      ? "Type at least 3 characters to search" 
                      : "No results found"
                    }
                  </CommandEmpty>
                  {filteredResults.length > 0 && (
                    <>
                      <CommandGroup heading="Studies">
                        {filteredResults.filter(r => r.type === 'study').slice(0, 3).map(r => (
                          <CommandItem key={r.id} onSelect={() => handleSearchSelect(r)} data-testid={`search-result-study-${r.id}`}>
                            <Beaker className="h-4 w-4 mr-2" />
                            {r.title}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                      <CommandSeparator />
                      <CommandGroup heading="Reports">
                        {filteredResults.filter(r => r.type === 'report').slice(0, 3).map(r => (
                          <CommandItem key={r.id} onSelect={() => handleSearchSelect(r)} data-testid={`search-result-report-${r.id}`}>
                            <FileText className="h-4 w-4 mr-2" />
                            {r.title}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                      <CommandSeparator />
                      <CommandGroup heading="Patients">
                        {filteredResults.filter(r => r.type === 'patient').slice(0, 3).map(r => (
                          <CommandItem key={r.id} onSelect={() => handleSearchSelect(r)} data-testid={`search-result-patient-${r.id}`}>
                            <Users className="h-4 w-4 mr-2" />
                            {r.title}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </>
                  )}
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>
        </div>
      </div>

      <div className="flex gap-2 flex-wrap">
        {permissions.canExportData && (
          <Button variant="outline" data-testid="button-export-data">
            <Download className="h-4 w-4 mr-2" />
            Export Data
          </Button>
        )}
        {permissions.canCreateStudies && (
          <Dialog open={createStudyOpen} onOpenChange={setCreateStudyOpen}>
            <DialogTrigger asChild>
              <Button data-testid="button-create-study">
                <Plus className="h-4 w-4 mr-2" />
                New Study
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Research Study</DialogTitle>
                <DialogDescription>
                  Define a new research study with specific data collection parameters
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="study-title">Study Title</Label>
                  <Input
                    id="study-title"
                    value={newStudy.title}
                    onChange={(e) => setNewStudy(s => ({ ...s, title: e.target.value }))}
                    placeholder="e.g., Long-term outcomes in post-transplant patients"
                    data-testid="input-study-title"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="study-desc">Description</Label>
                  <Textarea
                    id="study-desc"
                    value={newStudy.description}
                    onChange={(e) => setNewStudy(s => ({ ...s, description: e.target.value }))}
                    placeholder="Describe the research objectives and methodology..."
                    data-testid="input-study-description"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Data Types to Collect</Label>
                  <div className="flex flex-wrap gap-2">
                    {['Vitals', 'Medications', 'Lab Results', 'Symptoms', 'Wearable Data', 'Mental Health'].map((type) => (
                      <Badge
                        key={type}
                        variant={newStudy.dataTypes.includes(type) ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => setNewStudy(s => ({
                          ...s,
                          dataTypes: s.dataTypes.includes(type) 
                            ? s.dataTypes.filter(t => t !== type)
                            : [...s.dataTypes, type]
                        }))}
                        data-testid={`badge-datatype-${type.toLowerCase().replace(' ', '-')}`}
                      >
                        {type}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setCreateStudyOpen(false)}>Cancel</Button>
                <Button 
                  onClick={() => createStudyMutation.mutate(newStudy)}
                  disabled={!newStudy.title || createStudyMutation.isPending}
                  data-testid="button-confirm-create-study"
                >
                  {createStudyMutation.isPending ? (
                    <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Creating...</>
                  ) : (
                    'Create Study'
                  )}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card data-testid="card-total-patients">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Patients</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold" data-testid="text-total-patients">
              {cohortStats?.total_patients || 150}
            </p>
            <p className="text-xs text-muted-foreground">
              {cohortStats?.consenting_patients || 142} consenting to research
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-active-studies">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Active Studies</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold" data-testid="text-active-studies">
              {cohortStats?.active_studies || 3}
            </p>
            <p className="text-xs text-muted-foreground">Ongoing research projects</p>
          </CardContent>
        </Card>

        <Card data-testid="card-reports">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Reports Generated</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{reports?.length || 0}</p>
            <p className="text-xs text-muted-foreground">AI-powered analyses</p>
          </CardContent>
        </Card>

        <Card data-testid="card-data-completeness">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Data Completeness</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{(cohortStats?.data_completeness || 87).toFixed(0)}%</p>
            <Progress value={cohortStats?.data_completeness || 87} className="mt-2 h-2" />
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="cohort" className="space-y-6">
        <TabsList className="flex flex-wrap gap-1 w-full max-w-4xl h-auto p-1">
          <TabsTrigger value="cohort" className="gap-2" data-testid="tab-cohort">
            <Users className="h-4 w-4" />
            Cohort
          </TabsTrigger>
          <TabsTrigger value="studies" className="gap-2" data-testid="tab-studies">
            <Beaker className="h-4 w-4" />
            Studies
          </TabsTrigger>
          <TabsTrigger 
            value="import" 
            className="gap-2" 
            data-testid="tab-import"
            disabled={!permissions.canImportData}
          >
            <Upload className="h-4 w-4" />
            Import
            {!permissions.canImportData && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
          </TabsTrigger>
          <TabsTrigger 
            value="generate" 
            className="gap-2" 
            data-testid="tab-generate"
            disabled={!permissions.canRunAnalysis}
          >
            <Bot className="h-4 w-4" />
            AI Analysis
            {!permissions.canRunAnalysis && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
          </TabsTrigger>
          <TabsTrigger 
            value="alerts" 
            className="gap-2" 
            data-testid="tab-alerts"
            disabled={!permissions.canManageAlerts}
          >
            <AlertCircle className="h-4 w-4" />
            Alerts
            {!permissions.canManageAlerts && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
          </TabsTrigger>
          <TabsTrigger value="followups" className="gap-2" data-testid="tab-followups">
            <CalendarDays className="h-4 w-4" />
            Followups
          </TabsTrigger>
          <TabsTrigger value="reports" className="gap-2" data-testid="tab-reports">
            <FileText className="h-4 w-4" />
            Reports
          </TabsTrigger>
          <TabsTrigger 
            value="personal" 
            className="gap-2" 
            data-testid="tab-personal"
            disabled={!permissions.canAccessPersonalResearch}
          >
            <Brain className="h-4 w-4" />
            Personal Research
            {!permissions.canAccessPersonalResearch && <Lock className="h-3 w-3 ml-1 text-muted-foreground" />}
          </TabsTrigger>
          <TabsTrigger 
            value="advanced-analytics" 
            className="gap-2" 
            data-testid="tab-advanced-analytics"
          >
            <ShieldAlert className="h-4 w-4" />
            Advanced Analytics
          </TabsTrigger>
          <TabsTrigger value="datasets" className="gap-2" data-testid="tab-datasets">
            <Database className="h-4 w-4" />
            Datasets
          </TabsTrigger>
          <TabsTrigger value="exports" className="gap-2" data-testid="tab-exports">
            <Download className="h-4 w-4" />
            Exports
          </TabsTrigger>
          <TabsTrigger value="nlp-redaction" className="gap-2" data-testid="tab-nlp-redaction">
            <Lock className="h-4 w-4" />
            PHI Redaction
          </TabsTrigger>
          <TabsTrigger value="research-qa" className="gap-2" data-testid="tab-research-qa">
            <Bot className="h-4 w-4" />
            Research Q&A
          </TabsTrigger>
          <TabsTrigger value="nl-cohort" className="gap-2" data-testid="tab-nl-cohort">
            <Sparkles className="h-4 w-4" />
            NL Cohort
          </TabsTrigger>
          <TabsTrigger value="protocol-gen" className="gap-2" data-testid="tab-protocol-gen">
            <ScrollText className="h-4 w-4" />
            Protocol Gen
          </TabsTrigger>
          <TabsTrigger value="trial-emulation" className="gap-2" data-testid="tab-trial-emulation">
            <FlaskConical className="h-4 w-4" />
            Trial Emulation
          </TabsTrigger>
        </TabsList>

        <TabsContent value="cohort">
          <CohortBuilderTab />
        </TabsContent>

        <TabsContent value="studies">
          <StudiesTab />
        </TabsContent>

        <TabsContent value="generate">
          <AIAnalysisTab />
        </TabsContent>

        <TabsContent value="alerts">
          <AlertsTab />
        </TabsContent>

        <TabsContent value="followups">
          <DailyFollowupsTab />
        </TabsContent>

        <TabsContent value="import">
          <CSVImportTab />
        </TabsContent>

        <TabsContent value="reports">
          <ReportsTab />
        </TabsContent>

        <TabsContent value="personal">
          <PersonalResearchTab />
        </TabsContent>

        <TabsContent value="advanced-analytics">
          <AdvancedAnalyticsTab />
        </TabsContent>

        <TabsContent value="datasets">
          <DatasetsTab />
        </TabsContent>

        <TabsContent value="exports">
          <ExportsTab />
        </TabsContent>

        <TabsContent value="nlp-redaction">
          <NLPRedactionTab />
        </TabsContent>

        <TabsContent value="research-qa">
          <ResearchQATab />
        </TabsContent>

        <TabsContent value="nl-cohort">
          <NLCohortBuilderTab />
        </TabsContent>

        <TabsContent value="protocol-gen">
          <ProtocolGeneratorTab />
        </TabsContent>

        <TabsContent value="trial-emulation">
          <TrialEmulationTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
