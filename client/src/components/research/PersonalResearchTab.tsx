import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { queryClient, apiRequest } from '@/lib/queryClient';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Switch } from '@/components/ui/switch';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToast } from '@/hooks/use-toast';
import { 
  FolderKanban, Plus, Loader2, Settings, Users, Lock, Globe, 
  FlaskConical, FileText, BarChart, MessageSquare, Sparkles, Brain,
  Lightbulb, Send, ChevronRight, Clock, Calendar, Trash2, Edit, 
  Play, Pause, Archive
} from 'lucide-react';
import { format, formatDistanceToNow, parseISO } from 'date-fns';

interface ResearchProject {
  id: string;
  ownerId: string;
  name: string;
  description?: string;
  projectType: 'personal' | 'institutional' | 'collaborative';
  status: 'draft' | 'active' | 'paused' | 'completed' | 'archived';
  collaboratorIds: string[];
  isPublic: boolean;
  allowDataSharing: boolean;
  createdAt: string;
  updatedAt: string;
  studyCount?: number;
  cohortCount?: number;
}

interface AnalysisSpec {
  analysisType: 'descriptive' | 'risk_prediction' | 'survival' | 'causal';
  primaryOutcome?: string;
  exposureVariable?: string;
  covariates: string[];
  timeWindow?: number;
  modelType?: string;
  cohortFilters?: Record<string, any>;
}

interface NLParseResult {
  success: boolean;
  analysisSpec?: AnalysisSpec;
  explanation: string;
  confidence: number;
  suggestions?: string[];
}

export function PersonalResearchTab() {
  const { toast } = useToast();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedProject, setSelectedProject] = useState<ResearchProject | null>(null);
  const [nlQuery, setNlQuery] = useState('');
  const [parsedSpec, setParsedSpec] = useState<NLParseResult | null>(null);
  
  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
    projectType: 'personal' as const,
    isPublic: false,
    allowDataSharing: false,
  });

  const { data: projects = [], isLoading } = useQuery<ResearchProject[]>({
    queryKey: ['/api/v1/research-center/projects'],
  });

  const { data: projectStudies } = useQuery<any[]>({
    queryKey: ['/api/v1/research-center/projects', selectedProject?.id, 'studies'],
    enabled: !!selectedProject,
  });

  const createProjectMutation = useMutation({
    mutationFn: async (data: typeof newProject) => {
      const response = await apiRequest('POST', '/api/v1/research-center/projects', data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/projects'] });
      setCreateDialogOpen(false);
      setNewProject({ name: '', description: '', projectType: 'personal', isPublic: false, allowDataSharing: false });
      toast({ title: 'Project created', description: 'Your personal research project has been created' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const updateProjectMutation = useMutation({
    mutationFn: async ({ id, data }: { id: string; data: Partial<ResearchProject> }) => {
      const response = await apiRequest('PATCH', `/api/v1/research-center/projects/${id}`, data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/projects'] });
      toast({ title: 'Project updated' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const parseNLQueryMutation = useMutation({
    mutationFn: async (query: string) => {
      const response = await apiRequest('POST', '/api/v1/research-center/analysis/parse-nl', { query });
      return response.json() as Promise<NLParseResult>;
    },
    onSuccess: (data) => {
      setParsedSpec(data);
      if (data.success) {
        toast({ title: 'Analysis parsed', description: 'Review the generated analysis specification below' });
      }
    },
    onError: (error: Error) => {
      toast({ title: 'Error parsing query', description: error.message, variant: 'destructive' });
    },
  });

  const runAnalysisMutation = useMutation({
    mutationFn: async (spec: AnalysisSpec) => {
      const response = await apiRequest('POST', '/api/v1/research-center/analysis-jobs', {
        projectId: selectedProject?.id,
        ...spec,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/analysis-jobs'] });
      toast({ title: 'Analysis started', description: 'Your analysis job has been queued' });
      setNlQuery('');
      setParsedSpec(null);
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'default';
      case 'completed': return 'secondary';
      case 'paused': return 'outline';
      case 'draft': return 'outline';
      case 'archived': return 'secondary';
      default: return 'outline';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'personal': return <Lock className="h-4 w-4" />;
      case 'institutional': return <FlaskConical className="h-4 w-4" />;
      case 'collaborative': return <Users className="h-4 w-4" />;
      default: return <FolderKanban className="h-4 w-4" />;
    }
  };

  const nlExamples = [
    "What is the average age and gender distribution of patients with lupus who have CD4 counts below 200?",
    "Can you predict which transplant patients are at risk of infection in the next 30 days based on their immune markers?",
    "Compare survival outcomes between patients on tacrolimus vs cyclosporine, adjusting for age and comorbidities",
    "What is the causal effect of environmental air quality on infection rates in chronic care patients?",
  ];

  if (selectedProject) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={() => setSelectedProject(null)}
            data-testid="button-back-to-projects"
          >
            <ChevronRight className="h-4 w-4 mr-2 rotate-180" />
            Back to Projects
          </Button>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => updateProjectMutation.mutate({ id: selectedProject.id, data: { status: selectedProject.status === 'paused' ? 'active' : 'paused' } })}
              disabled={updateProjectMutation.isPending}
            >
              {selectedProject.status === 'paused' ? <Play className="h-4 w-4 mr-2" /> : <Pause className="h-4 w-4 mr-2" />}
              {selectedProject.status === 'paused' ? 'Resume' : 'Pause'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => updateProjectMutation.mutate({ id: selectedProject.id, data: { status: 'archived' } })}
              disabled={updateProjectMutation.isPending}
            >
              <Archive className="h-4 w-4 mr-2" />
              Archive
            </Button>
          </div>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-3">
                {getTypeIcon(selectedProject.projectType)}
                <div>
                  <CardTitle data-testid="text-project-name">{selectedProject.name}</CardTitle>
                  {selectedProject.description && (
                    <CardDescription className="mt-1">{selectedProject.description}</CardDescription>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={getStatusColor(selectedProject.status)}>
                  {selectedProject.status}
                </Badge>
                {selectedProject.isPublic && (
                  <Badge variant="outline">
                    <Globe className="h-3 w-3 mr-1" />
                    Public
                  </Badge>
                )}
              </div>
            </div>
          </CardHeader>
        </Card>

        <Tabs defaultValue="nlanalysis" className="space-y-4">
          <TabsList>
            <TabsTrigger value="nlanalysis" data-testid="tab-nl-analysis">
              <Sparkles className="h-4 w-4 mr-2" />
              AI Analysis
            </TabsTrigger>
            <TabsTrigger value="studies" data-testid="tab-project-studies">
              <FlaskConical className="h-4 w-4 mr-2" />
              Studies
            </TabsTrigger>
            <TabsTrigger value="settings" data-testid="tab-project-settings">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="nlanalysis">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-primary" />
                  Natural Language Analysis
                </CardTitle>
                <CardDescription>
                  Describe your research question in plain language and AI will generate an analysis specification
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Your Research Question</Label>
                  <Textarea
                    placeholder="e.g., What factors predict infection risk in transplant patients within 90 days?"
                    value={nlQuery}
                    onChange={(e) => setNlQuery(e.target.value)}
                    className="min-h-[100px]"
                    data-testid="input-nl-query"
                  />
                </div>

                <div className="flex gap-2">
                  <Button
                    onClick={() => parseNLQueryMutation.mutate(nlQuery)}
                    disabled={!nlQuery.trim() || parseNLQueryMutation.isPending}
                    data-testid="button-parse-query"
                  >
                    {parseNLQueryMutation.isPending ? (
                      <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Analyzing...</>
                    ) : (
                      <><Sparkles className="h-4 w-4 mr-2" /> Parse Query</>
                    )}
                  </Button>
                </div>

                <div className="bg-muted/50 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Lightbulb className="h-4 w-4 text-amber-500" />
                    <p className="text-sm font-medium">Example Questions</p>
                  </div>
                  <div className="space-y-2">
                    {nlExamples.map((example, idx) => (
                      <Button
                        key={idx}
                        variant="ghost"
                        size="sm"
                        className="w-full justify-start text-left h-auto py-2 text-xs text-muted-foreground hover:text-foreground"
                        onClick={() => setNlQuery(example)}
                        data-testid={`button-example-${idx}`}
                      >
                        <MessageSquare className="h-3 w-3 mr-2 flex-shrink-0" />
                        <span className="line-clamp-2">{example}</span>
                      </Button>
                    ))}
                  </div>
                </div>

                {parsedSpec && (
                  <Card className="border-primary/20">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-base flex items-center gap-2">
                          <BarChart className="h-4 w-4" />
                          Generated Analysis Specification
                        </CardTitle>
                        <Badge variant={parsedSpec.success ? 'default' : 'destructive'}>
                          {parsedSpec.confidence ? `${(parsedSpec.confidence * 100).toFixed(0)}% confidence` : 'Parsed'}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-sm text-muted-foreground">{parsedSpec.explanation}</p>
                      
                      {parsedSpec.analysisSpec && (
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-muted-foreground">Analysis Type</p>
                            <p className="font-medium capitalize">{parsedSpec.analysisSpec.analysisType.replace('_', ' ')}</p>
                          </div>
                          {parsedSpec.analysisSpec.primaryOutcome && (
                            <div>
                              <p className="text-muted-foreground">Primary Outcome</p>
                              <p className="font-medium">{parsedSpec.analysisSpec.primaryOutcome}</p>
                            </div>
                          )}
                          {parsedSpec.analysisSpec.exposureVariable && (
                            <div>
                              <p className="text-muted-foreground">Exposure Variable</p>
                              <p className="font-medium">{parsedSpec.analysisSpec.exposureVariable}</p>
                            </div>
                          )}
                          {parsedSpec.analysisSpec.covariates && parsedSpec.analysisSpec.covariates.length > 0 && (
                            <div className="col-span-2">
                              <p className="text-muted-foreground">Covariates</p>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {parsedSpec.analysisSpec.covariates.map((cov) => (
                                  <Badge key={cov} variant="secondary" className="text-xs">{cov}</Badge>
                                ))}
                              </div>
                            </div>
                          )}
                          {parsedSpec.analysisSpec.timeWindow && (
                            <div>
                              <p className="text-muted-foreground">Time Window</p>
                              <p className="font-medium">{parsedSpec.analysisSpec.timeWindow} days</p>
                            </div>
                          )}
                        </div>
                      )}

                      {parsedSpec.suggestions && parsedSpec.suggestions.length > 0 && (
                        <div className="bg-amber-50 dark:bg-amber-950/30 p-3 rounded-lg">
                          <p className="text-sm font-medium mb-1">Suggestions</p>
                          <ul className="text-sm text-muted-foreground space-y-1">
                            {parsedSpec.suggestions.map((suggestion, idx) => (
                              <li key={idx}>• {suggestion}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </CardContent>
                    <CardFooter className="gap-2">
                      <Button
                        onClick={() => {
                          if (parsedSpec.analysisSpec) {
                            runAnalysisMutation.mutate(parsedSpec.analysisSpec);
                          }
                        }}
                        disabled={!parsedSpec.success || !parsedSpec.analysisSpec || runAnalysisMutation.isPending}
                        data-testid="button-run-analysis"
                      >
                        {runAnalysisMutation.isPending ? (
                          <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Starting...</>
                        ) : (
                          <><Play className="h-4 w-4 mr-2" /> Run Analysis</>
                        )}
                      </Button>
                      <Button variant="outline" onClick={() => setParsedSpec(null)}>
                        Clear
                      </Button>
                    </CardFooter>
                  </Card>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="studies">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between gap-4">
                <div>
                  <CardTitle>Project Studies</CardTitle>
                  <CardDescription>Studies created within this research project</CardDescription>
                </div>
                <Button size="sm" data-testid="button-create-study">
                  <Plus className="h-4 w-4 mr-2" />
                  New Study
                </Button>
              </CardHeader>
              <CardContent>
                {projectStudies && projectStudies.length > 0 ? (
                  <ScrollArea className="h-[400px]">
                    <div className="space-y-3">
                      {projectStudies.map((study: any) => (
                        <Card key={study.id} className="hover-elevate cursor-pointer">
                          <CardContent className="p-4">
                            <div className="flex items-start justify-between gap-4">
                              <div>
                                <p className="font-medium">{study.title}</p>
                                <p className="text-sm text-muted-foreground line-clamp-1">{study.description}</p>
                              </div>
                              <Badge variant={getStatusColor(study.status)}>{study.status}</Badge>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center py-12 text-muted-foreground">
                    <FlaskConical className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No studies in this project yet</p>
                    <p className="text-sm mt-1">Create a new study to start your research</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="settings">
            <Card>
              <CardHeader>
                <CardTitle>Project Settings</CardTitle>
                <CardDescription>Configure visibility and data sharing options</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Public Project</Label>
                    <p className="text-sm text-muted-foreground">
                      Allow other researchers to view this project
                    </p>
                  </div>
                  <Switch
                    checked={selectedProject.isPublic}
                    onCheckedChange={(checked) => 
                      updateProjectMutation.mutate({ id: selectedProject.id, data: { isPublic: checked } })
                    }
                    data-testid="switch-public-project"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Allow Data Sharing</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable sharing of anonymized data with collaborators
                    </p>
                  </div>
                  <Switch
                    checked={selectedProject.allowDataSharing}
                    onCheckedChange={(checked) => 
                      updateProjectMutation.mutate({ id: selectedProject.id, data: { allowDataSharing: checked } })
                    }
                    data-testid="switch-data-sharing"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Personal Research Projects</h2>
          <p className="text-sm text-muted-foreground">
            Create and manage your own research projects with AI-powered analysis
          </p>
        </div>
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-create-project">
              <Plus className="h-4 w-4 mr-2" />
              New Project
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Research Project</DialogTitle>
              <DialogDescription>
                Start a new personal research project to organize your studies and analyses
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="project-name">Project Name</Label>
                <Input
                  id="project-name"
                  value={newProject.name}
                  onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                  placeholder="e.g., Immune Recovery Analysis"
                  data-testid="input-project-name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="project-desc">Description</Label>
                <Textarea
                  id="project-desc"
                  value={newProject.description}
                  onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                  placeholder="Brief description of your research goals"
                  data-testid="input-project-description"
                />
              </div>
              <div className="space-y-2">
                <Label>Project Type</Label>
                <Select
                  value={newProject.projectType}
                  onValueChange={(v) => setNewProject({ ...newProject, projectType: v as any })}
                >
                  <SelectTrigger data-testid="select-project-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="personal">
                      <div className="flex items-center gap-2">
                        <Lock className="h-4 w-4" />
                        Personal
                      </div>
                    </SelectItem>
                    <SelectItem value="institutional">
                      <div className="flex items-center gap-2">
                        <FlaskConical className="h-4 w-4" />
                        Institutional
                      </div>
                    </SelectItem>
                    <SelectItem value="collaborative">
                      <div className="flex items-center gap-2">
                        <Users className="h-4 w-4" />
                        Collaborative
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Public Project</Label>
                  <p className="text-xs text-muted-foreground">Visible to other researchers</p>
                </div>
                <Switch
                  checked={newProject.isPublic}
                  onCheckedChange={(checked) => setNewProject({ ...newProject, isPublic: checked })}
                  data-testid="switch-new-project-public"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
              <Button
                onClick={() => createProjectMutation.mutate(newProject)}
                disabled={!newProject.name || createProjectMutation.isPending}
                data-testid="button-confirm-create-project"
              >
                {createProjectMutation.isPending ? (
                  <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Creating...</>
                ) : 'Create Project'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-48" />
          ))}
        </div>
      ) : projects.length === 0 ? (
        <Card>
          <CardContent className="p-12 text-center">
            <FolderKanban className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
            <h3 className="text-lg font-medium mb-2">No Research Projects</h3>
            <p className="text-muted-foreground mb-4">
              Create your first personal research project to start exploring your data
            </p>
            <Button onClick={() => setCreateDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Project
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <Card
              key={project.id}
              className="hover-elevate cursor-pointer"
              onClick={() => setSelectedProject(project)}
              data-testid={`card-project-${project.id}`}
            >
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2">
                    {getTypeIcon(project.projectType)}
                    <Badge variant={getStatusColor(project.status)} className="text-xs">
                      {project.status}
                    </Badge>
                  </div>
                  {project.isPublic && (
                    <Globe className="h-4 w-4 text-muted-foreground" />
                  )}
                </div>
                <CardTitle className="text-base mt-2">{project.name}</CardTitle>
                {project.description && (
                  <CardDescription className="line-clamp-2">{project.description}</CardDescription>
                )}
              </CardHeader>
              <CardContent className="pb-3">
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  {project.studyCount !== undefined && (
                    <span className="flex items-center gap-1">
                      <FlaskConical className="h-3 w-3" />
                      {project.studyCount} studies
                    </span>
                  )}
                  {project.cohortCount !== undefined && (
                    <span className="flex items-center gap-1">
                      <Users className="h-3 w-3" />
                      {project.cohortCount} cohorts
                    </span>
                  )}
                </div>
              </CardContent>
              <CardFooter className="pt-0 text-xs text-muted-foreground">
                <Clock className="h-3 w-3 mr-1" />
                Updated {formatDistanceToNow(parseISO(project.updatedAt), { addSuffix: true })}
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
