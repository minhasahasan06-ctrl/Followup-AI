import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { 
  Play, Pause, RefreshCw, XCircle, RotateCcw, 
  Clock, CheckCircle2, AlertCircle, Loader2,
  Activity, Database, Server, Cpu
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";

interface TrainingJob {
  job_id: string;
  job_type: string;
  model_name: string;
  status: string;
  priority: number;
  progress_percent: number;
  current_step: string;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  metrics: Record<string, number | string>;
  artifact_path: string | null;
  consent_verified: boolean;
  governance_approved: boolean;
}

interface QueueStats {
  queue_size: number;
  status_counts: Record<string, number>;
  total_jobs: number;
  avg_duration_seconds: number;
  completed_count: number;
  failed_count: number;
  success_rate: number;
  redis_connected: boolean;
}

interface WorkerStatus {
  worker_id: string;
  is_running: boolean;
  current_job: TrainingJob | null;
}

const statusColors: Record<string, string> = {
  pending: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
  queued: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  running: "bg-purple-500/10 text-purple-500 border-purple-500/20",
  completed: "bg-green-500/10 text-green-500 border-green-500/20",
  failed: "bg-red-500/10 text-red-500 border-red-500/20",
  cancelled: "bg-gray-500/10 text-gray-500 border-gray-500/20",
  retrying: "bg-orange-500/10 text-orange-500 border-orange-500/20",
};

const statusIcons: Record<string, JSX.Element> = {
  pending: <Clock className="h-3 w-3" />,
  queued: <Database className="h-3 w-3" />,
  running: <Loader2 className="h-3 w-3 animate-spin" />,
  completed: <CheckCircle2 className="h-3 w-3" />,
  failed: <AlertCircle className="h-3 w-3" />,
  cancelled: <XCircle className="h-3 w-3" />,
  retrying: <RotateCcw className="h-3 w-3" />,
};

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "-";
  return new Date(dateStr).toLocaleString();
}

export function TrainingJobsPanel() {
  const { toast } = useToast();
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newJobType, setNewJobType] = useState("risk_model");
  const [newModelName, setNewModelName] = useState("");

  const { data: jobs = [], isLoading: jobsLoading, refetch: refetchJobs } = useQuery<TrainingJob[]>({
    queryKey: ['/python-api/ml-training/jobs'],
    refetchInterval: 5000,
  });

  const { data: queueStats } = useQuery<QueueStats>({
    queryKey: ['/python-api/ml-training/queue/stats'],
    refetchInterval: 10000,
  });

  const { data: workerStatus } = useQuery<WorkerStatus>({
    queryKey: ['/python-api/ml-training/worker/status'],
    refetchInterval: 5000,
  });

  const createJobMutation = useMutation({
    mutationFn: async (data: { job_type: string; model_name: string; config: Record<string, unknown> }) => {
      return apiRequest('/python-api/ml-training/jobs', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    onSuccess: () => {
      toast({ title: "Job created", description: "Training job has been queued" });
      queryClient.invalidateQueries({ queryKey: ['/python-api/ml-training/jobs'] });
      setCreateDialogOpen(false);
      setNewModelName("");
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  const cancelJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      return apiRequest(`/python-api/ml-training/jobs/${jobId}/cancel`, { method: 'POST' });
    },
    onSuccess: () => {
      toast({ title: "Job cancelled" });
      queryClient.invalidateQueries({ queryKey: ['/python-api/ml-training/jobs'] });
    },
  });

  const retryJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      return apiRequest(`/python-api/ml-training/jobs/${jobId}/retry`, { method: 'POST' });
    },
    onSuccess: () => {
      toast({ title: "Job queued for retry" });
      queryClient.invalidateQueries({ queryKey: ['/python-api/ml-training/jobs'] });
    },
  });

  const startWorkerMutation = useMutation({
    mutationFn: async () => {
      return apiRequest('/python-api/ml-training/worker/start', { method: 'POST' });
    },
    onSuccess: () => {
      toast({ title: "Worker started" });
      queryClient.invalidateQueries({ queryKey: ['/python-api/ml-training/worker/status'] });
    },
  });

  const stopWorkerMutation = useMutation({
    mutationFn: async () => {
      return apiRequest('/python-api/ml-training/worker/stop', { method: 'POST' });
    },
    onSuccess: () => {
      toast({ title: "Worker stopped" });
      queryClient.invalidateQueries({ queryKey: ['/python-api/ml-training/worker/status'] });
    },
  });

  const filteredJobs = statusFilter === "all" 
    ? jobs 
    : jobs.filter(job => job.status === statusFilter);

  const handleCreateJob = () => {
    if (!newModelName.trim()) {
      toast({ title: "Error", description: "Model name is required", variant: "destructive" });
      return;
    }
    createJobMutation.mutate({
      job_type: newJobType,
      model_name: newModelName,
      config: {
        require_consent: true,
        require_governance: false,
        auto_approve_governance: true,
      },
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold" data-testid="text-training-jobs-title">Training Jobs</h2>
          <p className="text-muted-foreground">Manage ML model training jobs and monitor progress</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetchJobs()}
            data-testid="button-refresh-jobs"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm" data-testid="button-create-job">
                <Play className="h-4 w-4 mr-2" />
                New Job
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Training Job</DialogTitle>
                <DialogDescription>
                  Configure and start a new ML model training job
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label>Job Type</Label>
                  <Select value={newJobType} onValueChange={setNewJobType}>
                    <SelectTrigger data-testid="select-job-type">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="risk_model">Risk Prediction (LSTM)</SelectItem>
                      <SelectItem value="adherence_model">Adherence Prediction (XGBoost)</SelectItem>
                      <SelectItem value="engagement_model">Engagement Prediction</SelectItem>
                      <SelectItem value="anomaly_model">Anomaly Detection (IsolationForest)</SelectItem>
                      <SelectItem value="custom">Custom Model</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Model Name</Label>
                  <Input
                    placeholder="e.g., risk_model_v2"
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                    data-testid="input-model-name"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
                  Cancel
                </Button>
                <Button 
                  onClick={handleCreateJob} 
                  disabled={createJobMutation.isPending}
                  data-testid="button-submit-job"
                >
                  {createJobMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  ) : null}
                  Create Job
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Database className="h-4 w-4" />
              Queue Size
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-queue-size">
              {queueStats?.queue_size ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {queueStats?.redis_connected ? "Redis connected" : "In-memory queue"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Success Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-success-rate">
              {(queueStats?.success_rate ?? 0).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              {queueStats?.completed_count ?? 0} completed, {queueStats?.failed_count ?? 0} failed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Avg Duration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-avg-duration">
              {formatDuration(queueStats?.avg_duration_seconds ?? 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              {queueStats?.total_jobs ?? 0} total jobs
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Server className="h-4 w-4" />
              Worker Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={`h-2 w-2 rounded-full ${workerStatus?.is_running ? 'bg-green-500' : 'bg-gray-400'}`} />
                <span className="text-sm font-medium" data-testid="text-worker-status">
                  {workerStatus?.is_running ? "Running" : "Stopped"}
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => workerStatus?.is_running ? stopWorkerMutation.mutate() : startWorkerMutation.mutate()}
                disabled={startWorkerMutation.isPending || stopWorkerMutation.isPending}
                data-testid="button-toggle-worker"
              >
                {workerStatus?.is_running ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
              </Button>
            </div>
            {workerStatus?.current_job && (
              <p className="text-xs text-muted-foreground mt-1 truncate">
                Processing: {workerStatus.current_job.model_name}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Job Queue</CardTitle>
              <CardDescription>View and manage training jobs</CardDescription>
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[150px]" data-testid="select-status-filter">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="queued">Queued</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
                <SelectItem value="cancelled">Cancelled</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          {jobsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : filteredJobs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No training jobs found
            </div>
          ) : (
            <ScrollArea className="h-[400px]">
              <div className="space-y-3">
                {filteredJobs.map((job) => (
                  <JobCard 
                    key={job.job_id} 
                    job={job}
                    onCancel={() => cancelJobMutation.mutate(job.job_id)}
                    onRetry={() => retryJobMutation.mutate(job.job_id)}
                  />
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function JobCard({ 
  job, 
  onCancel, 
  onRetry 
}: { 
  job: TrainingJob; 
  onCancel: () => void;
  onRetry: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card className="overflow-hidden">
      <div 
        className="p-4 cursor-pointer hover-elevate"
        onClick={() => setExpanded(!expanded)}
        data-testid={`card-job-${job.job_id}`}
      >
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-medium truncate" data-testid={`text-model-name-${job.job_id}`}>
                {job.model_name}
              </span>
              <Badge variant="outline" className={statusColors[job.status]}>
                {statusIcons[job.status]}
                <span className="ml-1 capitalize">{job.status}</span>
              </Badge>
              <Badge variant="secondary" className="text-xs">
                {job.job_type.replace('_', ' ')}
              </Badge>
            </div>
            <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
              <span>Created: {formatDate(job.created_at)}</span>
              {job.consent_verified && (
                <span className="flex items-center gap-1 text-green-500">
                  <CheckCircle2 className="h-3 w-3" /> Consent
                </span>
              )}
              {job.governance_approved && (
                <span className="flex items-center gap-1 text-green-500">
                  <CheckCircle2 className="h-3 w-3" /> Governance
                </span>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {job.status === "running" && (
              <div className="w-32">
                <Progress value={job.progress_percent} className="h-2" />
                <span className="text-xs text-muted-foreground">{job.progress_percent}%</span>
              </div>
            )}
            
            {(job.status === "pending" || job.status === "queued" || job.status === "running") && (
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => { e.stopPropagation(); onCancel(); }}
                data-testid={`button-cancel-${job.job_id}`}
              >
                <XCircle className="h-4 w-4" />
              </Button>
            )}
            
            {job.status === "failed" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => { e.stopPropagation(); onRetry(); }}
                data-testid={`button-retry-${job.job_id}`}
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
        
        {job.status === "running" && job.current_step && (
          <p className="text-xs text-muted-foreground mt-2 truncate">
            {job.current_step}
          </p>
        )}
      </div>
      
      {expanded && (
        <div className="px-4 pb-4 border-t pt-3 space-y-3">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Job ID:</span>
              <span className="ml-2 font-mono text-xs">{job.job_id}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Priority:</span>
              <span className="ml-2">{job.priority}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Started:</span>
              <span className="ml-2">{formatDate(job.started_at)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Completed:</span>
              <span className="ml-2">{formatDate(job.completed_at)}</span>
            </div>
          </div>
          
          {job.error_message && (
            <div className="p-2 rounded bg-red-500/10 text-red-500 text-sm">
              <strong>Error:</strong> {job.error_message}
            </div>
          )}
          
          {job.artifact_path && (
            <div className="text-sm">
              <span className="text-muted-foreground">Artifact:</span>
              <span className="ml-2 font-mono text-xs">{job.artifact_path}</span>
            </div>
          )}
          
          {Object.keys(job.metrics).length > 0 && (
            <div>
              <span className="text-sm text-muted-foreground block mb-2">Metrics:</span>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(job.metrics).map(([key, value]) => (
                  <div key={key} className="p-2 rounded bg-muted text-sm">
                    <div className="text-muted-foreground text-xs">{key}</div>
                    <div className="font-medium">
                      {typeof value === 'number' ? value.toFixed(4) : String(value)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
