import { useState } from "react";
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
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { AIResearchReport } from "@shared/schema";
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart as ReBarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, LineChart, Line, Legend } from "recharts";

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

export default function ResearchCenter() {
  const { toast } = useToast();
  const [reportTitle, setReportTitle] = useState("");
  const [analysisType, setAnalysisType] = useState<string>("");
  const [createStudyOpen, setCreateStudyOpen] = useState(false);
  const [newStudy, setNewStudy] = useState({
    title: '',
    description: '',
    dataTypes: [] as string[],
  });

  const { data: reports, isLoading } = useQuery<AIResearchReport[]>({
    queryKey: ["/api/doctor/research-reports"],
  });

  const { data: cohortStats } = useQuery<CohortStats>({
    queryKey: ['/api/research/cohort-stats'],
  });

  const { data: studies, isLoading: studiesLoading } = useQuery<ResearchStudy[]>({
    queryKey: ['/api/research/studies'],
  });

  const generateReportMutation = useMutation({
    mutationFn: async (data: { title: string; analysisType: string }) => {
      return await apiRequest("POST", "/api/doctor/research-reports", data);
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
      return await apiRequest("POST", "/api/research/studies", data);
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Beaker className="h-6 w-6 text-purple-500" />
            Epidemiology Research Center
          </h1>
          <p className="text-muted-foreground">
            AI-powered research analysis and data visualization for immunocompromised patient populations
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" data-testid="button-export-data">
            <Download className="h-4 w-4 mr-2" />
            Export Data
          </Button>
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
        </div>
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
        <TabsList className="grid w-full max-w-2xl grid-cols-4">
          <TabsTrigger value="cohort" className="gap-2" data-testid="tab-cohort">
            <Users className="h-4 w-4" />
            Cohort
          </TabsTrigger>
          <TabsTrigger value="studies" className="gap-2" data-testid="tab-studies">
            <Beaker className="h-4 w-4" />
            Studies
          </TabsTrigger>
          <TabsTrigger value="generate" className="gap-2" data-testid="tab-generate">
            <Bot className="h-4 w-4" />
            AI Analysis
          </TabsTrigger>
          <TabsTrigger value="reports" className="gap-2" data-testid="tab-reports">
            <FileText className="h-4 w-4" />
            Reports
          </TabsTrigger>
        </TabsList>

        <TabsContent value="cohort">
          <div className="grid gap-4 md:grid-cols-2">
            <Card data-testid="card-gender-distribution">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <ChartPie className="h-4 w-4 text-purple-500" />
                  Gender Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={genderData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      label={({ name, value }) => `${name}: ${value}%`}
                    >
                      {genderData.map((entry, index) => (
                        <Cell key={entry.name} fill={COHORT_COLORS[index % COHORT_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card data-testid="card-condition-distribution">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Activity className="h-4 w-4 text-blue-500" />
                  Condition Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <ReBarChart data={conditionData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#8884d8" radius={[0, 4, 4, 0]} />
                  </ReBarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card data-testid="card-age-distribution">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <BarChart className="h-4 w-4 text-green-500" />
                  Age Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <ReBarChart data={ageDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#82ca9d" radius={[4, 4, 0, 0]} />
                  </ReBarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card data-testid="card-enrollment-trend">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <TrendingUp className="h-4 w-4 text-amber-500" />
                  Enrollment Trend
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={enrollmentTrend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="patients" stroke="#8884d8" strokeWidth={2} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="studies">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Beaker className="h-5 w-5 text-purple-500" />
                Research Studies
              </CardTitle>
              <CardDescription>Manage and monitor ongoing research studies</CardDescription>
            </CardHeader>
            <CardContent>
              {studiesLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-20 bg-muted animate-pulse rounded-lg" />
                  ))}
                </div>
              ) : studies && studies.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Study</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Cohort</TableHead>
                      <TableHead>Start Date</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {studies.map((study) => (
                      <TableRow key={study.id} data-testid={`row-study-${study.id}`}>
                        <TableCell>
                          <div>
                            <p className="font-medium">{study.title}</p>
                            <p className="text-xs text-muted-foreground line-clamp-1">{study.description}</p>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant={
                            study.status === 'active' ? 'default' :
                            study.status === 'completed' ? 'secondary' :
                            study.status === 'paused' ? 'outline' : 'secondary'
                          }>
                            {study.status}
                          </Badge>
                        </TableCell>
                        <TableCell>{study.cohort_size} patients</TableCell>
                        <TableCell>{new Date(study.start_date).toLocaleDateString()}</TableCell>
                        <TableCell className="text-right">
                          <Button variant="ghost" size="sm" data-testid={`button-view-study-${study.id}`}>
                            <Eye className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <div className="text-center py-12">
                  <Beaker className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
                  <h3 className="text-sm font-medium mb-2">No Studies Yet</h3>
                  <p className="text-xs text-muted-foreground mb-4">
                    Create your first research study to start collecting data
                  </p>
                  <Button onClick={() => setCreateStudyOpen(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Create Study
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="generate">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-accent">
                  <Bot className="h-5 w-5" />
                </div>
                <div>
                  <CardTitle>AI Research Agent</CardTitle>
                  <CardDescription>
                    Generate comprehensive epidemiological research reports using AI analysis
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="title">Report Title</Label>
                  <Input
                    id="title"
                    placeholder="e.g., Correlation between medication adherence and vital stability"
                    value={reportTitle}
                    onChange={(e) => setReportTitle(e.target.value)}
                    data-testid="input-report-title"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="analysis-type">Analysis Type</Label>
                  <Select value={analysisType} onValueChange={setAnalysisType}>
                    <SelectTrigger id="analysis-type" data-testid="select-analysis-type">
                      <SelectValue placeholder="Select analysis type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="correlation">Correlation Analysis</SelectItem>
                      <SelectItem value="regression">Regression Analysis</SelectItem>
                      <SelectItem value="survival">Survival Analysis</SelectItem>
                      <SelectItem value="pattern">Pattern Recognition</SelectItem>
                      <SelectItem value="cohort">Cohort Comparison</SelectItem>
                      <SelectItem value="longitudinal">Longitudinal Study</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Card className="bg-muted/50">
                  <CardContent className="p-4">
                    <h4 className="font-medium mb-2 text-sm">Data Sources</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
                      <div>
                        <p className="font-medium text-foreground mb-1">Patient Data</p>
                        <ul className="space-y-1">
                          <li className="flex items-center gap-1">
                            <CheckCircle2 className="h-3 w-3 text-green-500" />
                            Daily follow-up assessments
                          </li>
                          <li className="flex items-center gap-1">
                            <CheckCircle2 className="h-3 w-3 text-green-500" />
                            Medication adherence records
                          </li>
                          <li className="flex items-center gap-1">
                            <CheckCircle2 className="h-3 w-3 text-green-500" />
                            Wearable device data
                          </li>
                          <li className="flex items-center gap-1">
                            <CheckCircle2 className="h-3 w-3 text-green-500" />
                            Behavioral insights
                          </li>
                        </ul>
                      </div>
                      <div>
                        <p className="font-medium text-foreground mb-1">Public Datasets</p>
                        <ul className="space-y-1">
                          <li className="flex items-center gap-1">
                            <Database className="h-3 w-3 text-blue-500" />
                            PubMed clinical studies
                          </li>
                          <li className="flex items-center gap-1">
                            <Database className="h-3 w-3 text-blue-500" />
                            PhysioNet datasets
                          </li>
                          <li className="flex items-center gap-1">
                            <Database className="h-3 w-3 text-blue-500" />
                            WHO Health Observatory
                          </li>
                          <li className="flex items-center gap-1">
                            <Database className="h-3 w-3 text-blue-500" />
                            MIMIC-IV clinical data
                          </li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Button
                onClick={handleGenerateReport}
                disabled={generateReportMutation.isPending}
                className="w-full"
                data-testid="button-generate-report"
              >
                {generateReportMutation.isPending ? (
                  <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Generating Report...</>
                ) : (
                  <><Brain className="h-4 w-4 mr-2" /> Generate Research Report</>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reports">
          <Card>
            <CardHeader>
              <CardTitle>Generated Research Reports</CardTitle>
              <CardDescription>View and manage your AI-generated research analyses</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <Card key={i} className="animate-pulse">
                      <CardContent className="p-4">
                        <div className="space-y-2">
                          <div className="h-4 bg-muted rounded w-3/4" />
                          <div className="h-3 bg-muted rounded w-1/2" />
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : reports && reports.length > 0 ? (
                <ScrollArea className="h-[600px] pr-4">
                  <div className="space-y-4">
                    {reports.map((report) => (
                      <Card key={report.id} className="hover-elevate" data-testid={`card-report-${report.id}`}>
                        <CardContent className="p-6">
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex-1">
                              <h3 className="font-semibold mb-1" data-testid={`text-report-title-${report.id}`}>
                                {report.title}
                              </h3>
                              <p className="text-sm text-muted-foreground line-clamp-2">{report.summary}</p>
                            </div>
                            <div className="flex gap-2 flex-shrink-0 ml-4">
                              <Badge variant="secondary">{report.analysisType}</Badge>
                              {report.publishReady && <Badge variant="secondary">Ready to Publish</Badge>}
                            </div>
                          </div>

                          <div className="grid grid-cols-2 gap-3 mb-4">
                            <div className="text-sm">
                              <span className="text-muted-foreground">Cohort Size: </span>
                              <span className="font-medium">{report.patientCohortSize} patients</span>
                            </div>
                            <div className="text-sm">
                              <span className="text-muted-foreground">Date: </span>
                              <span className="font-medium">
                                {new Date(report.createdAt).toLocaleDateString()}
                              </span>
                            </div>
                          </div>

                          {report.findings && report.findings.length > 0 && (
                            <div className="space-y-2 mt-4">
                              <p className="text-sm font-medium">Key Findings:</p>
                              {report.findings.slice(0, 2).map((finding, idx) => (
                                <Card key={idx} className="bg-muted/50">
                                  <CardContent className="p-3">
                                    <div className="flex items-start gap-2">
                                      <TrendingUp className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                                      <div className="text-sm">
                                        <p className="font-medium mb-1">{finding.finding}</p>
                                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                          <span>Significance: {finding.significance}</span>
                                          <span>•</span>
                                          <span>Confidence: {(finding.confidence * 100).toFixed(0)}%</span>
                                        </div>
                                      </div>
                                    </div>
                                  </CardContent>
                                </Card>
                              ))}
                            </div>
                          )}

                          <div className="flex gap-2 mt-4">
                            <Button
                              variant="outline"
                              className="flex-1"
                              data-testid={`button-view-report-${report.id}`}
                            >
                              <Eye className="h-4 w-4 mr-2" />
                              View Full Report
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              data-testid={`button-download-report-${report.id}`}
                            >
                              <Download className="h-4 w-4" />
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No research reports generated yet</p>
                  <p className="text-sm mt-1">Use the AI Research Agent to create your first report</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
