import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { apiRequest, queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { 
  Beaker, Plus, Eye, Edit, Play, Pause, CheckCircle, Clock, 
  Users, CalendarDays, BarChart3, Settings, RefreshCw, 
  TrendingUp, AlertTriangle, FileText, Filter
} from 'lucide-react';
import { 
  LineChart, Line, AreaChart, Area, BarChart as ReBarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';

interface Study {
  id: string;
  title: string;
  description: string;
  status: 'planning' | 'enrolling' | 'follow_up' | 'analysis' | 'completed' | 'paused';
  cohortId?: string;
  startDate?: string;
  endDate?: string;
  targetEnrollment: number;
  currentEnrollment: number;
  inclusionCriteria?: string;
  exclusionCriteria?: string;
  autoReanalysis?: boolean;
  reanalysisFrequency?: string;
  lastReanalysisAt?: string;
  createdAt: string;
  updatedAt: string;
}

const STATUS_CONFIG = {
  planning: { label: 'Planning', icon: Clock, color: 'text-blue-500', badge: 'secondary' as const },
  enrolling: { label: 'Enrolling', icon: Users, color: 'text-green-500', badge: 'default' as const },
  follow_up: { label: 'Follow-up', icon: CalendarDays, color: 'text-purple-500', badge: 'default' as const },
  analysis: { label: 'Analysis', icon: BarChart3, color: 'text-amber-500', badge: 'default' as const },
  completed: { label: 'Completed', icon: CheckCircle, color: 'text-emerald-500', badge: 'secondary' as const },
  paused: { label: 'Paused', icon: Pause, color: 'text-gray-500', badge: 'outline' as const },
};

const LIFECYCLE_STAGES = ['planning', 'enrolling', 'follow_up', 'analysis', 'completed'];

export function StudiesTab() {
  const { toast } = useToast();
  const [selectedStudy, setSelectedStudy] = useState<Study | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  
  const [newStudy, setNewStudy] = useState({
    title: '',
    description: '',
    targetEnrollment: 100,
    inclusionCriteria: '',
    exclusionCriteria: '',
    autoReanalysis: false,
    reanalysisFrequency: 'weekly',
  });

  const { data: studies, isLoading } = useQuery<Study[]>({
    queryKey: ['/api/v1/research-center/studies', statusFilter],
  });

  const { data: cohorts } = useQuery<any[]>({
    queryKey: ['/api/v1/research-center/cohorts'],
  });

  const createMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiRequest('POST', '/api/v1/research-center/studies', data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/studies'] });
      setCreateDialogOpen(false);
      setNewStudy({
        title: '',
        description: '',
        targetEnrollment: 100,
        inclusionCriteria: '',
        exclusionCriteria: '',
        autoReanalysis: false,
        reanalysisFrequency: 'weekly',
      });
      toast({ title: 'Study Created', description: 'Your research study has been created successfully.' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const updateStatusMutation = useMutation({
    mutationFn: async ({ id, status }: { id: string; status: string }) => {
      const response = await apiRequest('PATCH', `/api/v1/research-center/studies/${id}`, { status });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/studies'] });
      toast({ title: 'Status Updated', description: 'Study status has been updated.' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const enrollmentData = [
    { week: 'Week 1', enrolled: 12, target: 15 },
    { week: 'Week 2', enrolled: 28, target: 30 },
    { week: 'Week 3', enrolled: 45, target: 45 },
    { week: 'Week 4', enrolled: 58, target: 60 },
    { week: 'Week 5', enrolled: 72, target: 75 },
    { week: 'Week 6', enrolled: 85, target: 90 },
  ];

  const followupData = [
    { month: 'M1', retention: 98, compliance: 95 },
    { month: 'M2', retention: 95, compliance: 92 },
    { month: 'M3', retention: 92, compliance: 88 },
    { month: 'M6', retention: 88, compliance: 85 },
    { month: 'M9', retention: 84, compliance: 82 },
    { month: 'M12', retention: 80, compliance: 78 },
  ];

  const sitePerformanceData = [
    { site: 'Site A', enrolled: 45, target: 50, compliance: 92 },
    { site: 'Site B', enrolled: 38, target: 50, compliance: 88 },
    { site: 'Site C', enrolled: 52, target: 50, compliance: 95 },
    { site: 'Site D', enrolled: 28, target: 50, compliance: 85 },
  ];

  const filteredStudies = studies?.filter(s => 
    statusFilter === 'all' || s.status === statusFilter
  ) || [];

  const getNextStatus = (current: string) => {
    const idx = LIFECYCLE_STAGES.indexOf(current);
    return idx < LIFECYCLE_STAGES.length - 1 ? LIFECYCLE_STAGES[idx + 1] : null;
  };

  const renderLifecycleProgress = (study: Study) => {
    const stages = LIFECYCLE_STAGES;
    const currentIdx = stages.indexOf(study.status);
    
    return (
      <div className="flex items-center gap-1 mt-2" data-testid="lifecycle-progress">
        {stages.map((stage, idx) => {
          const isCompleted = idx < currentIdx;
          const isCurrent = idx === currentIdx;
          const StatusIcon = STATUS_CONFIG[stage as keyof typeof STATUS_CONFIG].icon;
          
          return (
            <div key={stage} className="flex items-center">
              <div 
                className={`flex items-center justify-center w-6 h-6 rounded-full text-xs font-medium transition-colors
                  ${isCompleted ? 'bg-emerald-500 text-white' : 
                    isCurrent ? 'bg-primary text-primary-foreground' : 
                    'bg-muted text-muted-foreground'}`}
              >
                {isCompleted ? <CheckCircle className="w-4 h-4" /> : <StatusIcon className="w-3 h-3" />}
              </div>
              {idx < stages.length - 1 && (
                <div className={`w-6 h-0.5 ${idx < currentIdx ? 'bg-emerald-500' : 'bg-muted'}`} />
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="space-y-4" data-testid="studies-tab">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-40" data-testid="select-status-filter">
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Studies</SelectItem>
              <SelectItem value="planning">Planning</SelectItem>
              <SelectItem value="enrolling">Enrolling</SelectItem>
              <SelectItem value="follow_up">Follow-up</SelectItem>
              <SelectItem value="analysis">Analysis</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="paused">Paused</SelectItem>
            </SelectContent>
          </Select>
        </div>
        
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-create-study">
              <Plus className="h-4 w-4 mr-2" />
              New Study
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Create New Research Study</DialogTitle>
              <DialogDescription>
                Define your study parameters and eligibility criteria
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="title">Study Title</Label>
                  <Input
                    id="title"
                    placeholder="e.g., Long-term Outcomes in Transplant Patients"
                    value={newStudy.title}
                    onChange={(e) => setNewStudy({ ...newStudy, title: e.target.value })}
                    data-testid="input-study-title"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="target">Target Enrollment</Label>
                  <Input
                    id="target"
                    type="number"
                    value={newStudy.targetEnrollment}
                    onChange={(e) => setNewStudy({ ...newStudy, targetEnrollment: parseInt(e.target.value) || 0 })}
                    data-testid="input-target-enrollment"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  placeholder="Describe the study objectives and methodology..."
                  value={newStudy.description}
                  onChange={(e) => setNewStudy({ ...newStudy, description: e.target.value })}
                  data-testid="input-study-description"
                />
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="inclusion">Inclusion Criteria</Label>
                  <Textarea
                    id="inclusion"
                    placeholder="e.g., Age 18-65, Transplant within 5 years..."
                    value={newStudy.inclusionCriteria}
                    onChange={(e) => setNewStudy({ ...newStudy, inclusionCriteria: e.target.value })}
                    data-testid="input-inclusion-criteria"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="exclusion">Exclusion Criteria</Label>
                  <Textarea
                    id="exclusion"
                    placeholder="e.g., Active malignancy, Multiple organ transplant..."
                    value={newStudy.exclusionCriteria}
                    onChange={(e) => setNewStudy({ ...newStudy, exclusionCriteria: e.target.value })}
                    data-testid="input-exclusion-criteria"
                  />
                </div>
              </div>
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div className="space-y-0.5">
                  <Label>Auto Re-analysis</Label>
                  <p className="text-xs text-muted-foreground">
                    Automatically re-run analyses when new data arrives
                  </p>
                </div>
                <div className="flex items-center gap-4">
                  <Select
                    value={newStudy.reanalysisFrequency}
                    onValueChange={(v) => setNewStudy({ ...newStudy, reanalysisFrequency: v })}
                    disabled={!newStudy.autoReanalysis}
                  >
                    <SelectTrigger className="w-32" data-testid="select-reanalysis-frequency">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="weekly">Weekly</SelectItem>
                      <SelectItem value="monthly">Monthly</SelectItem>
                    </SelectContent>
                  </Select>
                  <Switch
                    checked={newStudy.autoReanalysis}
                    onCheckedChange={(c) => setNewStudy({ ...newStudy, autoReanalysis: c })}
                    data-testid="switch-auto-reanalysis"
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
                Cancel
              </Button>
              <Button 
                onClick={() => createMutation.mutate(newStudy)}
                disabled={!newStudy.title || createMutation.isPending}
                data-testid="button-submit-study"
              >
                {createMutation.isPending ? 'Creating...' : 'Create Study'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-32 bg-muted animate-pulse rounded-lg" />
          ))}
        </div>
      ) : filteredStudies.length > 0 ? (
        <div className="grid gap-4">
          {filteredStudies.map((study) => {
            const config = STATUS_CONFIG[study.status];
            const StatusIcon = config.icon;
            const enrollmentPercent = study.targetEnrollment > 0 
              ? Math.round((study.currentEnrollment / study.targetEnrollment) * 100) 
              : 0;
            const nextStatus = getNextStatus(study.status);
            
            return (
              <Card key={study.id} data-testid={`card-study-${study.id}`} className="hover-elevate">
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <StatusIcon className={`h-5 w-5 ${config.color}`} />
                        <CardTitle className="text-lg truncate">{study.title}</CardTitle>
                        <Badge variant={config.badge}>{config.label}</Badge>
                      </div>
                      <CardDescription className="line-clamp-2 mt-1">
                        {study.description || 'No description provided'}
                      </CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                      {study.status !== 'completed' && study.status !== 'paused' && nextStatus && (
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => updateStatusMutation.mutate({ id: study.id, status: nextStatus })}
                          data-testid={`button-advance-${study.id}`}
                        >
                          <Play className="h-3 w-3 mr-1" />
                          Advance
                        </Button>
                      )}
                      <Button 
                        variant="ghost" 
                        size="icon"
                        onClick={() => {
                          setSelectedStudy(study);
                          setDetailsOpen(true);
                        }}
                        data-testid={`button-view-${study.id}`}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Enrollment</span>
                        <span className="font-medium">{study.currentEnrollment}/{study.targetEnrollment}</span>
                      </div>
                      <Progress value={enrollmentPercent} className="h-2" />
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Started: </span>
                        <span>{study.startDate ? new Date(study.startDate).toLocaleDateString() : 'Not started'}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      {study.autoReanalysis && (
                        <Badge variant="outline" className="text-xs">
                          <RefreshCw className="h-3 w-3 mr-1" />
                          Auto-reanalysis ({study.reanalysisFrequency})
                        </Badge>
                      )}
                    </div>
                  </div>
                  {renderLifecycleProgress(study)}
                </CardContent>
              </Card>
            );
          })}
        </div>
      ) : (
        <Card>
          <CardContent className="text-center py-12">
            <Beaker className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
            <h3 className="text-sm font-medium mb-2">No Studies Found</h3>
            <p className="text-xs text-muted-foreground mb-4">
              {statusFilter !== 'all' ? 'No studies match the selected filter' : 'Create your first research study to start collecting data'}
            </p>
            <Button onClick={() => setCreateDialogOpen(true)} data-testid="button-create-first-study">
              <Plus className="h-4 w-4 mr-2" />
              Create Study
            </Button>
          </CardContent>
        </Card>
      )}

      <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Beaker className="h-5 w-5 text-purple-500" />
              {selectedStudy?.title}
            </DialogTitle>
            <DialogDescription>
              Study details, enrollment metrics, and performance analytics
            </DialogDescription>
          </DialogHeader>
          
          <ScrollArea className="max-h-[60vh]">
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="enrollment">Enrollment</TabsTrigger>
                <TabsTrigger value="followup">Follow-up</TabsTrigger>
                <TabsTrigger value="settings">Settings</TabsTrigger>
              </TabsList>
              
              <TabsContent value="overview" className="space-y-4 pt-4">
                <div className="grid gap-4 md:grid-cols-3">
                  <Card>
                    <CardContent className="pt-4">
                      <div className="flex items-center gap-2">
                        <Users className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">Enrolled</span>
                      </div>
                      <p className="text-2xl font-bold mt-1">
                        {selectedStudy?.currentEnrollment || 0}
                        <span className="text-sm font-normal text-muted-foreground">
                          /{selectedStudy?.targetEnrollment}
                        </span>
                      </p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-4">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">Retention Rate</span>
                      </div>
                      <p className="text-2xl font-bold mt-1">88%</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-4">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">Data Completeness</span>
                      </div>
                      <p className="text-2xl font-bold mt-1">92%</p>
                    </CardContent>
                  </Card>
                </div>
                
                <Accordion type="single" collapsible className="w-full">
                  <AccordionItem value="criteria">
                    <AccordionTrigger>Eligibility Criteria</AccordionTrigger>
                    <AccordionContent>
                      <div className="grid gap-4 md:grid-cols-2">
                        <div>
                          <h4 className="text-sm font-medium text-green-600 mb-2">Inclusion Criteria</h4>
                          <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                            {selectedStudy?.inclusionCriteria || 'No inclusion criteria specified'}
                          </p>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-red-600 mb-2">Exclusion Criteria</h4>
                          <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                            {selectedStudy?.exclusionCriteria || 'No exclusion criteria specified'}
                          </p>
                        </div>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </TabsContent>
              
              <TabsContent value="enrollment" className="space-y-4 pt-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Enrollment Progress</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={enrollmentData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="week" fontSize={12} />
                        <YAxis fontSize={12} />
                        <Tooltip />
                        <Legend />
                        <Area 
                          type="monotone" 
                          dataKey="target" 
                          stroke="#94a3b8" 
                          fill="#f1f5f9" 
                          name="Target"
                        />
                        <Area 
                          type="monotone" 
                          dataKey="enrolled" 
                          stroke="#8b5cf6" 
                          fill="#c4b5fd" 
                          name="Enrolled"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Site Performance</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Site</TableHead>
                          <TableHead>Enrolled</TableHead>
                          <TableHead>Target</TableHead>
                          <TableHead>Progress</TableHead>
                          <TableHead>Compliance</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sitePerformanceData.map((site) => (
                          <TableRow key={site.site}>
                            <TableCell className="font-medium">{site.site}</TableCell>
                            <TableCell>{site.enrolled}</TableCell>
                            <TableCell>{site.target}</TableCell>
                            <TableCell>
                              <Progress value={(site.enrolled / site.target) * 100} className="h-2 w-20" />
                            </TableCell>
                            <TableCell>
                              <Badge variant={site.compliance >= 90 ? 'default' : 'secondary'}>
                                {site.compliance}%
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="followup" className="space-y-4 pt-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Retention & Compliance Over Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={followupData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" fontSize={12} />
                        <YAxis fontSize={12} domain={[0, 100]} />
                        <Tooltip />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="retention" 
                          stroke="#22c55e" 
                          strokeWidth={2}
                          name="Retention %"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="compliance" 
                          stroke="#8b5cf6" 
                          strokeWidth={2}
                          name="Compliance %"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
                
                <div className="grid gap-4 md:grid-cols-2">
                  <Card>
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-muted-foreground">Upcoming Visits</p>
                          <p className="text-2xl font-bold">23</p>
                        </div>
                        <CalendarDays className="h-8 w-8 text-muted-foreground" />
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-muted-foreground">Overdue Visits</p>
                          <p className="text-2xl font-bold text-amber-500">5</p>
                        </div>
                        <AlertTriangle className="h-8 w-8 text-amber-500" />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
              
              <TabsContent value="settings" className="space-y-4 pt-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <RefreshCw className="h-4 w-4" />
                      Auto Re-analysis Configuration
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>Enable Auto Re-analysis</Label>
                        <p className="text-xs text-muted-foreground">
                          Automatically run analyses when new data is collected
                        </p>
                      </div>
                      <Switch checked={selectedStudy?.autoReanalysis} />
                    </div>
                    <div className="space-y-2">
                      <Label>Frequency</Label>
                      <Select defaultValue={selectedStudy?.reanalysisFrequency || 'weekly'}>
                        <SelectTrigger data-testid="select-detail-reanalysis-freq">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="daily">Daily</SelectItem>
                          <SelectItem value="weekly">Weekly</SelectItem>
                          <SelectItem value="monthly">Monthly</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    {selectedStudy?.lastReanalysisAt && (
                      <p className="text-xs text-muted-foreground">
                        Last re-analysis: {new Date(selectedStudy.lastReanalysisAt).toLocaleString()}
                      </p>
                    )}
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Settings className="h-4 w-4" />
                      Status Management
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Current Status</Label>
                      <Select defaultValue={selectedStudy?.status}>
                        <SelectTrigger data-testid="select-study-status">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="planning">Planning</SelectItem>
                          <SelectItem value="enrolling">Enrolling</SelectItem>
                          <SelectItem value="follow_up">Follow-up</SelectItem>
                          <SelectItem value="analysis">Analysis</SelectItem>
                          <SelectItem value="completed">Completed</SelectItem>
                          <SelectItem value="paused">Paused</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Button variant="outline" className="w-full">
                      Save Changes
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </ScrollArea>
        </DialogContent>
      </Dialog>
    </div>
  );
}
