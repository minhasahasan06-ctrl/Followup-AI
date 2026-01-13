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
import { apiRequest, queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { 
  CalendarCheck, Plus, Edit, Trash2, Copy, Users, Clock,
  CheckCircle, AlertCircle, TrendingUp, BarChart3, FileText,
  Settings, Play, Pause, Eye, ChevronRight
} from 'lucide-react';
import { 
  LineChart, Line, AreaChart, Area, BarChart as ReBarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';

interface FollowupTemplate {
  id: string;
  name: string;
  description: string;
  frequency: 'daily' | 'weekly' | 'biweekly' | 'monthly';
  questions: {
    id: string;
    text: string;
    type: 'scale' | 'boolean' | 'text' | 'multiple_choice';
    options?: string[];
    required: boolean;
  }[];
  isActive: boolean;
  assignedPatients: number;
  responseRate: number;
  createdAt: string;
}

interface PatientAssignment {
  id: string;
  patientId: string;
  patientName: string;
  templateId: string;
  frequency: string;
  lastResponse?: string;
  completionRate: number;
  status: 'active' | 'paused' | 'completed';
}

const COLORS = ['#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#6366f1'];

const mockTemplates: FollowupTemplate[] = [
  {
    id: '1',
    name: 'Daily Health Check',
    description: 'Standard daily health and symptom assessment',
    frequency: 'daily',
    questions: [
      { id: 'q1', text: 'How are you feeling today?', type: 'scale', required: true },
      { id: 'q2', text: 'Any new symptoms?', type: 'boolean', required: true },
      { id: 'q3', text: 'Did you take all your medications?', type: 'boolean', required: true },
      { id: 'q4', text: 'Additional notes', type: 'text', required: false },
    ],
    isActive: true,
    assignedPatients: 45,
    responseRate: 87,
    createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '2',
    name: 'Weekly Pain Assessment',
    description: 'Track pain levels and patterns weekly',
    frequency: 'weekly',
    questions: [
      { id: 'q1', text: 'Rate your average pain level this week', type: 'scale', required: true },
      { id: 'q2', text: 'Pain location', type: 'multiple_choice', options: ['Head', 'Back', 'Joints', 'Chest', 'Other'], required: true },
      { id: 'q3', text: 'Has your pain affected daily activities?', type: 'boolean', required: true },
    ],
    isActive: true,
    assignedPatients: 28,
    responseRate: 92,
    createdAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '3',
    name: 'Mental Health Check-in',
    description: 'Monthly mental health and wellbeing assessment',
    frequency: 'monthly',
    questions: [
      { id: 'q1', text: 'Rate your overall mood', type: 'scale', required: true },
      { id: 'q2', text: 'Have you felt anxious or stressed?', type: 'scale', required: true },
      { id: 'q3', text: 'Sleep quality rating', type: 'scale', required: true },
    ],
    isActive: false,
    assignedPatients: 15,
    responseRate: 78,
    createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

const responseRateData = [
  { day: 'Mon', rate: 85 },
  { day: 'Tue', rate: 88 },
  { day: 'Wed', rate: 82 },
  { day: 'Thu', rate: 90 },
  { day: 'Fri', rate: 87 },
  { day: 'Sat', rate: 75 },
  { day: 'Sun', rate: 72 },
];

const symptomTrendData = [
  { date: 'Week 1', fatigue: 6.5, pain: 4.2, mood: 7.0 },
  { date: 'Week 2', fatigue: 5.8, pain: 4.0, mood: 7.2 },
  { date: 'Week 3', fatigue: 5.2, pain: 3.8, mood: 7.5 },
  { date: 'Week 4', fatigue: 4.5, pain: 3.5, mood: 8.0 },
];

const adherenceData = [
  { name: 'Completed', value: 78 },
  { name: 'Partial', value: 15 },
  { name: 'Missed', value: 7 },
];

export function DailyFollowupsTab() {
  const { toast } = useToast();
  const [selectedTemplate, setSelectedTemplate] = useState<FollowupTemplate | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editMode, setEditMode] = useState(false);
  
  const [newTemplate, setNewTemplate] = useState({
    name: '',
    description: '',
    frequency: 'daily',
    questions: [] as FollowupTemplate['questions'],
  });

  const [newQuestion, setNewQuestion] = useState({
    text: '',
    type: 'scale' as const,
    required: true,
  });

  const { data: templatesData, isLoading } = useQuery<FollowupTemplate[]>({
    queryKey: ['/api/v1/research-center/followup-templates'],
  });
  
  const templates = templatesData && templatesData.length > 0 ? templatesData : mockTemplates;

  const createMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiRequest('/api/v1/research-center/followup-templates', { method: 'POST', json: data });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/followup-templates'] });
      setCreateDialogOpen(false);
      setNewTemplate({ name: '', description: '', frequency: 'daily', questions: [] });
      toast({ title: 'Template Created', description: 'Your follow-up template has been created.' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const addQuestion = () => {
    if (!newQuestion.text) return;
    setNewTemplate(prev => ({
      ...prev,
      questions: [...prev.questions, { 
        id: `q${prev.questions.length + 1}`, 
        ...newQuestion,
      }],
    }));
    setNewQuestion({ text: '', type: 'scale', required: true });
  };

  const removeQuestion = (index: number) => {
    setNewTemplate(prev => ({
      ...prev,
      questions: prev.questions.filter((_, i) => i !== index),
    }));
  };

  const totalAssignedPatients = templates.reduce((sum, t) => sum + t.assignedPatients, 0);
  const avgResponseRate = templates.reduce((sum, t) => sum + t.responseRate, 0) / templates.length;

  return (
    <div className="space-y-4" data-testid="daily-followups-tab">
      <Tabs defaultValue="templates" className="w-full">
        <TabsList>
          <TabsTrigger value="templates" data-testid="tab-templates">
            <FileText className="h-4 w-4 mr-2" />
            Templates
          </TabsTrigger>
          <TabsTrigger value="monitoring" data-testid="tab-monitoring">
            <BarChart3 className="h-4 w-4 mr-2" />
            Monitoring
          </TabsTrigger>
          <TabsTrigger value="assignments" data-testid="tab-assignments">
            <Users className="h-4 w-4 mr-2" />
            Assignments
          </TabsTrigger>
        </TabsList>

        <TabsContent value="templates" className="space-y-4 pt-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <Card className="px-4 py-2">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Templates:</span>
                  <span className="font-semibold">{templates.length}</span>
                </div>
              </Card>
              <Card className="px-4 py-2">
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Patients:</span>
                  <span className="font-semibold">{totalAssignedPatients}</span>
                </div>
              </Card>
            </div>

            <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
              <DialogTrigger asChild>
                <Button data-testid="button-create-template">
                  <Plus className="h-4 w-4 mr-2" />
                  New Template
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl max-h-[90vh]">
                <DialogHeader>
                  <DialogTitle>Create Follow-up Template</DialogTitle>
                  <DialogDescription>
                    Design a custom follow-up questionnaire for patient monitoring
                  </DialogDescription>
                </DialogHeader>
                <ScrollArea className="max-h-[60vh]">
                  <div className="space-y-4 py-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label htmlFor="name">Template Name</Label>
                        <Input
                          id="name"
                          placeholder="e.g., Daily Health Check"
                          value={newTemplate.name}
                          onChange={(e) => setNewTemplate({ ...newTemplate, name: e.target.value })}
                          data-testid="input-template-name"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="frequency">Frequency</Label>
                        <Select
                          value={newTemplate.frequency}
                          onValueChange={(v) => setNewTemplate({ ...newTemplate, frequency: v })}
                        >
                          <SelectTrigger id="frequency" data-testid="select-frequency">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="daily">Daily</SelectItem>
                            <SelectItem value="weekly">Weekly</SelectItem>
                            <SelectItem value="biweekly">Bi-weekly</SelectItem>
                            <SelectItem value="monthly">Monthly</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="description">Description</Label>
                      <Textarea
                        id="description"
                        placeholder="Describe the purpose of this follow-up..."
                        value={newTemplate.description}
                        onChange={(e) => setNewTemplate({ ...newTemplate, description: e.target.value })}
                        data-testid="input-template-description"
                      />
                    </div>

                    <div className="border rounded-lg p-4 space-y-4">
                      <Label>Questions</Label>
                      {newTemplate.questions.length > 0 && (
                        <div className="space-y-2">
                          {newTemplate.questions.map((q, idx) => (
                            <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-medium">{idx + 1}.</span>
                                <span className="text-sm">{q.text}</span>
                                <Badge variant="outline" className="text-xs">{q.type}</Badge>
                                {q.required && <Badge variant="secondary" className="text-xs">Required</Badge>}
                              </div>
                              <Button 
                                variant="ghost" 
                                size="icon"
                                onClick={() => removeQuestion(idx)}
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                          ))}
                        </div>
                      )}

                      <div className="space-y-3 pt-2">
                        <div className="grid gap-3 md:grid-cols-3">
                          <div className="md:col-span-2">
                            <Input
                              placeholder="Question text..."
                              value={newQuestion.text}
                              onChange={(e) => setNewQuestion({ ...newQuestion, text: e.target.value })}
                              data-testid="input-question-text"
                            />
                          </div>
                          <Select
                            value={newQuestion.type}
                            onValueChange={(v: any) => setNewQuestion({ ...newQuestion, type: v })}
                          >
                            <SelectTrigger data-testid="select-question-type">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="scale">Scale (1-10)</SelectItem>
                              <SelectItem value="boolean">Yes/No</SelectItem>
                              <SelectItem value="text">Free Text</SelectItem>
                              <SelectItem value="multiple_choice">Multiple Choice</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <Switch
                              id="required"
                              checked={newQuestion.required}
                              onCheckedChange={(c) => setNewQuestion({ ...newQuestion, required: c })}
                            />
                            <Label htmlFor="required" className="text-sm">Required</Label>
                          </div>
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={addQuestion}
                            disabled={!newQuestion.text}
                            data-testid="button-add-question"
                          >
                            <Plus className="h-4 w-4 mr-1" />
                            Add Question
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </ScrollArea>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={() => createMutation.mutate(newTemplate)}
                    disabled={!newTemplate.name || newTemplate.questions.length === 0 || createMutation.isPending}
                    data-testid="button-submit-template"
                  >
                    {createMutation.isPending ? 'Creating...' : 'Create Template'}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>

          <div className="grid gap-4">
            {templates.map((template) => (
              <Card 
                key={template.id} 
                className="hover-elevate"
                data-testid={`card-template-${template.id}`}
              >
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold">{template.name}</h3>
                        <Badge variant={template.isActive ? 'default' : 'secondary'}>
                          {template.isActive ? 'Active' : 'Inactive'}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {template.frequency.charAt(0).toUpperCase() + template.frequency.slice(1)}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-3">{template.description}</p>
                      <div className="flex items-center gap-4 text-sm">
                        <div className="flex items-center gap-1">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span>{template.questions.length} questions</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Users className="h-4 w-4 text-muted-foreground" />
                          <span>{template.assignedPatients} patients</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <TrendingUp className="h-4 w-4 text-emerald-500" />
                          <span className="text-emerald-600">{template.responseRate}% response rate</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button variant="ghost" size="icon" data-testid={`button-edit-${template.id}`}>
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" data-testid={`button-duplicate-${template.id}`}>
                        <Copy className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" data-testid={`button-toggle-${template.id}`}>
                        {template.isActive ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-4 pt-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Response Rate</p>
                    <p className="text-2xl font-bold">{avgResponseRate.toFixed(0)}%</p>
                  </div>
                  <TrendingUp className="h-8 w-8 text-emerald-500" />
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Today's Responses</p>
                    <p className="text-2xl font-bold">38/45</p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-blue-500" />
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Pending</p>
                    <p className="text-2xl font-bold text-amber-500">7</p>
                  </div>
                  <Clock className="h-8 w-8 text-amber-500" />
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Weekly Response Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={responseRateData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day" fontSize={12} />
                    <YAxis fontSize={12} domain={[0, 100]} />
                    <Tooltip />
                    <Area 
                      type="monotone" 
                      dataKey="rate" 
                      stroke="#8b5cf6" 
                      fill="#c4b5fd"
                      name="Response Rate %"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Adherence Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={adherenceData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}%`}
                    >
                      {adherenceData.map((entry, index) => (
                        <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Symptom Trends (Cohort Average)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={symptomTrendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" fontSize={12} />
                  <YAxis fontSize={12} domain={[0, 10]} />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="fatigue" 
                    stroke="#f59e0b" 
                    strokeWidth={2}
                    name="Fatigue"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="pain" 
                    stroke="#ef4444" 
                    strokeWidth={2}
                    name="Pain"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="mood" 
                    stroke="#22c55e" 
                    strokeWidth={2}
                    name="Mood"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="assignments" className="pt-4">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Users className="h-4 w-4" />
                    Patient Assignments
                  </CardTitle>
                  <CardDescription>
                    Manage which patients receive which follow-up templates
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm" data-testid="button-assign-patients">
                  <Plus className="h-4 w-4 mr-1" />
                  Assign Patients
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Patient</TableHead>
                    <TableHead>Template</TableHead>
                    <TableHead>Frequency</TableHead>
                    <TableHead>Last Response</TableHead>
                    <TableHead>Completion</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[
                    { id: '1', patientId: 'P001', patientName: 'John D.', templateId: '1', frequency: 'daily', lastResponse: '2 hours ago', completionRate: 92, status: 'active' },
                    { id: '2', patientId: 'P002', patientName: 'Sarah M.', templateId: '1', frequency: 'daily', lastResponse: '1 day ago', completionRate: 78, status: 'active' },
                    { id: '3', patientId: 'P003', patientName: 'Robert K.', templateId: '2', frequency: 'weekly', lastResponse: '3 days ago', completionRate: 95, status: 'active' },
                    { id: '4', patientId: 'P004', patientName: 'Emily T.', templateId: '1', frequency: 'daily', lastResponse: '5 days ago', completionRate: 45, status: 'paused' },
                  ].map((assignment) => (
                    <TableRow key={assignment.id} data-testid={`row-assignment-${assignment.id}`}>
                      <TableCell className="font-medium">{assignment.patientName}</TableCell>
                      <TableCell>
                        {templates.find(t => t.id === assignment.templateId)?.name || 'Unknown'}
                      </TableCell>
                      <TableCell className="capitalize">{assignment.frequency}</TableCell>
                      <TableCell className="text-muted-foreground">{assignment.lastResponse}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Progress value={assignment.completionRate} className="h-2 w-16" />
                          <span className="text-sm">{assignment.completionRate}%</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant={assignment.status === 'active' ? 'default' : 'secondary'}>
                          {assignment.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="icon">
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="icon">
                          <Settings className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
