import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { InputOTP, InputOTPGroup, InputOTPSlot, InputOTPSeparator } from "@/components/ui/input-otp";
import { 
  Database, 
  Brain, 
  Play, 
  Loader2, 
  FileText, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  AlertCircle,
  RefreshCw,
  Rocket,
  Users,
  ShieldCheck,
  TrendingUp,
  Package,
  Settings,
  History,
  Cpu,
  HardDrive,
  Lock,
  Unlock,
  Watch,
  Download,
  Calendar,
  Filter,
  Droplets,
  Activity,
  KeyRound,
  QrCode,
  Smartphone,
  Copy,
  Sparkles,
  Network,
  GitBranch,
  Target,
  BarChart3,
  TestTube,
  Shield,
  MapPin,
} from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { format } from "date-fns";
import { AutopilotAdminDashboard } from "@/components/autopilot/AutopilotAdminDashboard";
import { TrainingJobsPanel } from "@/components/ml-training/TrainingJobsPanel";
import { JobComposerTab } from "@/components/ml/JobComposerTab";
import { ValidationReportTab } from "@/components/ml/ValidationReportTab";
import { CalibrationReportTab } from "@/components/ml/CalibrationReportTab";
import { GovernancePackTab } from "@/components/ml/GovernancePackTab";
import { DriftMonitorTab } from "@/components/ml/DriftMonitorTab";
import { FeedbackDatasetTab } from "@/components/ml/FeedbackDatasetTab";
import { Wand2, MessageSquare } from "lucide-react";

interface Dataset {
  name: string;
  source: string;
  description: string;
  record_count: number;
  patient_count: number;
  requires_credentials: boolean;
  is_available: boolean;
}

interface TrainedModel {
  model_id: string;
  name: string;
  model_type: string;
  version: string;
  status: string;
  is_active: boolean;
  metrics: {
    accuracy?: number;
    auc?: number;
    f1?: number;
    precision?: number;
    recall?: number;
  } | null;
  created_at: string;
}

interface TrainingJob {
  job_id: string;
  status: string;
  model_name: string;
  version: string;
  progress_percent: number;
  current_phase: string | null;
  message: string | null;
}

interface ConsentStats {
  total_patients_with_consent_record: number;
  consenting_patients: number;
  consent_rate: number;
  data_type_breakdown: {
    vitals: number;
    symptoms: number;
    mental_health: number;
    medications: number;
    wearable: number;
  };
}

interface ContributionsSummary {
  unique_contributors: number;
  total_contributions: number;
  total_records_contributed: number;
  training_jobs_with_patient_data: number;
}

const CONSENT_COLORS = ["#22c55e", "#3b82f6", "#f97316", "#8b5cf6", "#ec4899"];

interface TOTPStatus {
  isSetup: boolean;
  enabled: boolean;
  isLocked?: boolean;
  lockedUntil?: string | null;
}

interface TOTPSession {
  isVerified: boolean;
  verifiedAt?: string | null;
}

function TOTPGate({ children }: { children: React.ReactNode }) {
  const { toast } = useToast();
  const [otpValue, setOtpValue] = useState("");
  const [showSecret, setShowSecret] = useState(false);
  const [qrCode, setQrCode] = useState<string | null>(null);
  const [secretKey, setSecretKey] = useState<string | null>(null);

  const { data: totpStatus, isLoading: statusLoading, refetch: refetchStatus } = useQuery<TOTPStatus>({
    queryKey: ['/api/admin/totp/status'],
  });

  const { data: sessionStatus, isLoading: sessionLoading, refetch: refetchSession } = useQuery<TOTPSession>({
    queryKey: ['/api/admin/totp/session'],
  });

  const setupMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('/api/admin/totp/setup', {
        method: 'POST',
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to generate QR code');
      }
      return response.json();
    },
    onSuccess: (data: any) => {
      setQrCode(data.qrCode);
      setSecretKey(data.secret);
      toast({
        title: "QR Code Generated",
        description: "Scan with Google Authenticator to set up 2FA",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Setup Failed",
        description: error.message || "Failed to generate QR code",
        variant: "destructive",
      });
    },
  });

  const verifySetupMutation = useMutation({
    mutationFn: async (token: string) => {
      const response = await apiRequest('/api/admin/totp/verify-setup', {
        method: 'POST',
        body: JSON.stringify({ token }),
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Verification failed');
      }
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "2FA Enabled",
        description: "Google Authenticator is now required for access",
      });
      setQrCode(null);
      setSecretKey(null);
      setOtpValue("");
      refetchStatus();
      refetchSession();
    },
    onError: (error: any) => {
      toast({
        title: "Verification Failed",
        description: error.message || "Invalid code. Please try again.",
        variant: "destructive",
      });
      setOtpValue("");
    },
  });

  const authenticateMutation = useMutation({
    mutationFn: async (token: string) => {
      const response = await apiRequest('/api/admin/totp/authenticate', {
        method: 'POST',
        body: JSON.stringify({ token }),
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Authentication failed');
      }
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Access Granted",
        description: "Authentication successful",
      });
      setOtpValue("");
      refetchSession();
    },
    onError: (error: any) => {
      toast({
        title: "Authentication Failed",
        description: error.message || "Invalid code. Please try again.",
        variant: "destructive",
      });
      setOtpValue("");
    },
  });

  const handleCopySecret = () => {
    if (secretKey) {
      navigator.clipboard.writeText(secretKey);
      toast({
        title: "Copied",
        description: "Secret key copied to clipboard",
      });
    }
  };

  const handleOtpComplete = (value: string) => {
    if (value.length === 6) {
      if (qrCode && !totpStatus?.enabled) {
        verifySetupMutation.mutate(value);
      } else {
        authenticateMutation.mutate(value);
      }
    }
  };

  useEffect(() => {
    if (otpValue.length === 6) {
      handleOtpComplete(otpValue);
    }
  }, [otpValue]);

  if (statusLoading || sessionLoading) {
    return (
      <div className="container mx-auto py-8 px-4">
        <Card className="max-w-md mx-auto">
          <CardContent className="pt-6">
            <div className="flex items-center justify-center space-x-2">
              <Loader2 className="h-6 w-6 animate-spin" />
              <span className="text-muted-foreground">Loading security status...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (sessionStatus?.isVerified) {
    return <>{children}</>;
  }

  if (totpStatus?.isLocked) {
    return (
      <div className="container mx-auto py-8 px-4">
        <Card className="max-w-md mx-auto border-red-500/50" data-testid="card-totp-locked">
          <CardHeader className="text-center">
            <div className="mx-auto w-16 h-16 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center mb-4">
              <Lock className="h-8 w-8 text-red-500" />
            </div>
            <CardTitle className="text-red-600 dark:text-red-400">Access Locked</CardTitle>
            <CardDescription>
              Too many failed attempts. Access is temporarily locked.
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Please try again later or contact an administrator.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!totpStatus?.isSetup || !totpStatus?.enabled) {
    return (
      <div className="container mx-auto py-8 px-4">
        <Card className="max-w-lg mx-auto" data-testid="card-totp-setup">
          <CardHeader className="text-center">
            <div className="mx-auto w-16 h-16 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center mb-4">
              <KeyRound className="h-8 w-8 text-purple-500" />
            </div>
            <CardTitle>Set Up Two-Factor Authentication</CardTitle>
            <CardDescription>
              Protect the ML Training Hub with Google Authenticator
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {!qrCode ? (
              <div className="text-center space-y-4">
                <div className="p-6 bg-muted/30 rounded-lg space-y-3">
                  <Smartphone className="h-12 w-12 mx-auto text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    You'll need the Google Authenticator app installed on your phone.
                    Download it from the App Store or Google Play if you don't have it.
                  </p>
                </div>
                <Button
                  onClick={() => setupMutation.mutate()}
                  disabled={setupMutation.isPending}
                  className="w-full"
                  data-testid="button-generate-qr"
                >
                  {setupMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  ) : (
                    <QrCode className="h-4 w-4 mr-2" />
                  )}
                  Generate QR Code
                </Button>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground mb-4">
                    Scan this QR code with Google Authenticator:
                  </p>
                  <div className="bg-white p-4 rounded-lg inline-block">
                    <img 
                      src={qrCode} 
                      alt="TOTP QR Code" 
                      className="w-48 h-48"
                      data-testid="img-qr-code"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm text-muted-foreground">Manual entry key:</Label>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowSecret(!showSecret)}
                      data-testid="button-toggle-secret"
                    >
                      {showSecret ? "Hide" : "Show"}
                    </Button>
                  </div>
                  {showSecret && secretKey && (
                    <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
                      <code className="flex-1 text-xs font-mono break-all" data-testid="text-secret-key">
                        {secretKey}
                      </code>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={handleCopySecret}
                        data-testid="button-copy-secret"
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                </div>

                <div className="space-y-3">
                  <Label className="text-center block">Enter the 6-digit code from your app:</Label>
                  <div className="flex justify-center">
                    <InputOTP
                      maxLength={6}
                      value={otpValue}
                      onChange={setOtpValue}
                      disabled={verifySetupMutation.isPending}
                      data-testid="input-otp-setup"
                    >
                      <InputOTPGroup>
                        <InputOTPSlot index={0} />
                        <InputOTPSlot index={1} />
                        <InputOTPSlot index={2} />
                      </InputOTPGroup>
                      <InputOTPSeparator />
                      <InputOTPGroup>
                        <InputOTPSlot index={3} />
                        <InputOTPSlot index={4} />
                        <InputOTPSlot index={5} />
                      </InputOTPGroup>
                    </InputOTP>
                  </div>
                  {verifySetupMutation.isPending && (
                    <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Verifying...
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="max-w-md mx-auto" data-testid="card-totp-auth">
        <CardHeader className="text-center">
          <div className="mx-auto w-16 h-16 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mb-4">
            <ShieldCheck className="h-8 w-8 text-blue-500" />
          </div>
          <CardTitle>Two-Factor Authentication</CardTitle>
          <CardDescription>
            Enter the code from Google Authenticator
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <div className="flex justify-center">
              <InputOTP
                maxLength={6}
                value={otpValue}
                onChange={setOtpValue}
                disabled={authenticateMutation.isPending}
                data-testid="input-otp-auth"
              >
                <InputOTPGroup>
                  <InputOTPSlot index={0} />
                  <InputOTPSlot index={1} />
                  <InputOTPSlot index={2} />
                </InputOTPGroup>
                <InputOTPSeparator />
                <InputOTPGroup>
                  <InputOTPSlot index={3} />
                  <InputOTPSlot index={4} />
                  <InputOTPSlot index={5} />
                </InputOTPGroup>
              </InputOTP>
            </div>
            {authenticateMutation.isPending && (
              <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Authenticating...
              </div>
            )}
          </div>
          <p className="text-xs text-center text-muted-foreground">
            Open Google Authenticator on your phone and enter the 6-digit code for "Followup AI ML Training Hub"
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function LegalDisclaimer() {
  return (
    <Alert className="border-amber-500/50 bg-amber-500/5 mb-4" data-testid="alert-ml-disclaimer">
      <ShieldCheck className="h-4 w-4 text-amber-500" />
      <AlertTitle className="text-amber-700 dark:text-amber-400 font-semibold">
        HIPAA-Compliant ML Training Platform
      </AlertTitle>
      <AlertDescription className="text-amber-600 dark:text-amber-300 text-sm">
        All training data is fully anonymized with k-anonymity enforcement. Patient consent is verified 
        before any data extraction. HIPAA audit logs are maintained for all operations.
      </AlertDescription>
    </Alert>
  );
}

function DatasetsTab() {
  const { data: datasets, isLoading } = useQuery<Dataset[]>({
    queryKey: ['/api/ml/training/datasets'],
  });

  const registerMutation = useMutation({
    mutationFn: async (datasetKey: string) => {
      return apiRequest(`/api/ml/training/register-dataset/${datasetKey}`, {
        method: 'POST',
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/ml/training/datasets'] });
    },
  });

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        {[1, 2, 3, 4].map(i => (
          <Card key={i}>
            <CardHeader><Skeleton className="h-6 w-32" /></CardHeader>
            <CardContent><Skeleton className="h-24 w-full" /></CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {datasets?.map((dataset) => (
        <Card key={dataset.name} data-testid={`card-dataset-${dataset.name.toLowerCase().replace(/\s+/g, '-')}`}>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between gap-2 text-base">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-blue-500" />
                {dataset.name}
              </div>
              {dataset.is_available ? (
                <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                  <Unlock className="h-3 w-3 mr-1" /> Available
                </Badge>
              ) : (
                <Badge className="bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
                  <Lock className="h-3 w-3 mr-1" /> Credentials Required
                </Badge>
              )}
            </CardTitle>
            <CardDescription>{dataset.source}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">{dataset.description}</p>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-2 rounded bg-muted/50 text-center">
                <p className="text-xs text-muted-foreground">Records</p>
                <p className="text-lg font-semibold" data-testid={`text-dataset-records-${dataset.name}`}>
                  {dataset.record_count.toLocaleString()}
                </p>
              </div>
              <div className="p-2 rounded bg-muted/50 text-center">
                <p className="text-xs text-muted-foreground">Patients</p>
                <p className="text-lg font-semibold" data-testid={`text-dataset-patients-${dataset.name}`}>
                  {dataset.patient_count.toLocaleString()}
                </p>
              </div>
            </div>
          </CardContent>
          <CardFooter>
            {dataset.is_available ? (
              <Button 
                variant="default" 
                size="sm" 
                className="w-full"
                onClick={() => registerMutation.mutate(dataset.name.toLowerCase().replace(/\s+/g, '_'))}
                disabled={registerMutation.isPending}
                data-testid={`button-register-dataset-${dataset.name}`}
              >
                {registerMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                Register for Training
              </Button>
            ) : (
              <Button variant="outline" size="sm" className="w-full" disabled>
                <Lock className="h-4 w-4 mr-2" />
                Add PhysioNet Credentials
              </Button>
            )}
          </CardFooter>
        </Card>
      ))}
    </div>
  );
}

function ModelsTab() {
  const { toast } = useToast();
  const { data: models, isLoading, refetch } = useQuery<TrainedModel[]>({
    queryKey: ['/api/ml/training/models'],
  });

  const deployMutation = useMutation({
    mutationFn: async (modelId: string) => {
      return apiRequest(`/api/ml/training/models/${modelId}/deploy`, {
        method: 'POST',
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/ml/training/models'] });
      toast({ title: 'Model Deployed', description: 'Model is now active for predictions' });
    },
    onError: (error: Error) => {
      toast({ title: 'Deployment Failed', description: error.message, variant: 'destructive' });
    },
  });

  const getStatusBadge = (status: string, isActive: boolean) => {
    if (isActive) {
      return <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"><CheckCircle2 className="h-3 w-3 mr-1" />Active</Badge>;
    }
    switch (status) {
      case 'trained':
        return <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"><Package className="h-3 w-3 mr-1" />Trained</Badge>;
      case 'training':
        return <Badge className="bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"><Loader2 className="h-3 w-3 mr-1 animate-spin" />Training</Badge>;
      case 'failed':
        return <Badge className="bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"><XCircle className="h-3 w-3 mr-1" />Failed</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map(i => (
          <Card key={i}>
            <CardHeader><Skeleton className="h-6 w-48" /></CardHeader>
            <CardContent><Skeleton className="h-16 w-full" /></CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (!models?.length) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <Brain className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
          <h3 className="text-lg font-medium mb-2">No Trained Models</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Create a training job to train your first ML model.
          </p>
          <Button variant="outline" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" /> Refresh
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" /> Refresh
        </Button>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Version</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Metrics</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((model) => (
            <TableRow key={model.model_id} data-testid={`row-model-${model.model_id}`}>
              <TableCell className="font-medium">{model.name}</TableCell>
              <TableCell><Badge variant="outline">{model.model_type}</Badge></TableCell>
              <TableCell className="text-sm text-muted-foreground">{model.version}</TableCell>
              <TableCell>{getStatusBadge(model.status, model.is_active)}</TableCell>
              <TableCell>
                {model.metrics ? (
                  <div className="flex gap-2 text-xs">
                    {model.metrics.accuracy && (
                      <span className="bg-green-100 dark:bg-green-900/30 px-2 py-0.5 rounded">
                        Acc: {(model.metrics.accuracy * 100).toFixed(1)}%
                      </span>
                    )}
                    {model.metrics.auc && (
                      <span className="bg-blue-100 dark:bg-blue-900/30 px-2 py-0.5 rounded">
                        AUC: {model.metrics.auc.toFixed(3)}
                      </span>
                    )}
                    {model.metrics.f1 && (
                      <span className="bg-purple-100 dark:bg-purple-900/30 px-2 py-0.5 rounded">
                        F1: {model.metrics.f1.toFixed(3)}
                      </span>
                    )}
                  </div>
                ) : (
                  <span className="text-muted-foreground text-sm">—</span>
                )}
              </TableCell>
              <TableCell className="text-sm text-muted-foreground">
                {format(new Date(model.created_at), 'MMM d, yyyy')}
              </TableCell>
              <TableCell className="text-right">
                {model.status === 'trained' && !model.is_active && (
                  <Button 
                    size="sm" 
                    onClick={() => deployMutation.mutate(model.model_id)}
                    disabled={deployMutation.isPending}
                    data-testid={`button-deploy-${model.model_id}`}
                  >
                    {deployMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Rocket className="h-4 w-4 mr-1" />}
                    Deploy
                  </Button>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function TrainingJobsTab() {
  const { toast } = useToast();
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [jobForm, setJobForm] = useState({
    model_name: '',
    model_type: 'random_forest',
  });

  const { data: jobs, isLoading, refetch } = useQuery<TrainingJob[]>({
    queryKey: ['/api/ml/training/jobs'],
    refetchInterval: 5000,
  });

  const createJobMutation = useMutation({
    mutationFn: async () => {
      return apiRequest('/api/ml/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: jobForm.model_name,
          model_type: jobForm.model_type,
          data_sources: {},
          hyperparameters: {},
        }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/ml/training/jobs'] });
      setShowCreateDialog(false);
      setJobForm({ model_name: '', model_type: 'random_forest' });
      toast({ title: 'Training Job Created', description: 'Job has been queued for processing' });
    },
    onError: (error: Error) => {
      toast({ title: 'Failed to Create Job', description: error.message, variant: 'destructive' });
    },
  });

  const startJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      return apiRequest(`/api/ml/training/jobs/${jobId}/start`, {
        method: 'POST',
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/ml/training/jobs'] });
      toast({ title: 'Job Started', description: 'Training has begun' });
    },
  });

  const getStatusBadge = (status: string, progress: number) => {
    switch (status) {
      case 'queued':
        return <Badge variant="outline"><Clock className="h-3 w-3 mr-1" />Queued</Badge>;
      case 'running':
        return (
          <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />{progress}%
          </Badge>
        );
      case 'completed':
        return <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"><CheckCircle2 className="h-3 w-3 mr-1" />Completed</Badge>;
      case 'failed':
        return <Badge className="bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"><XCircle className="h-3 w-3 mr-1" />Failed</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-40" />
        {[1, 2, 3].map(i => (
          <Card key={i}>
            <CardContent className="py-4"><Skeleton className="h-12 w-full" /></CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogTrigger asChild>
            <Button data-testid="button-create-training-job">
              <Play className="h-4 w-4 mr-2" /> Create Training Job
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create ML Training Job</DialogTitle>
              <DialogDescription>Configure a new model training job with hyperparameters.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="model_name">Model Name</Label>
                <Input 
                  id="model_name"
                  placeholder="e.g., Deterioration Predictor v2"
                  value={jobForm.model_name}
                  onChange={(e) => setJobForm(prev => ({ ...prev, model_name: e.target.value }))}
                  data-testid="input-model-name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="model_type">Model Type</Label>
                <Select 
                  value={jobForm.model_type} 
                  onValueChange={(v) => setJobForm(prev => ({ ...prev, model_type: v }))}
                >
                  <SelectTrigger data-testid="select-model-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="random_forest">Random Forest</SelectItem>
                    <SelectItem value="gradient_boosting">Gradient Boosting</SelectItem>
                    <SelectItem value="kmeans">K-Means Clustering</SelectItem>
                    <SelectItem value="lstm">LSTM Neural Network</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>Cancel</Button>
              <Button 
                onClick={() => createJobMutation.mutate()}
                disabled={!jobForm.model_name || createJobMutation.isPending}
                data-testid="button-submit-training-job"
              >
                {createJobMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Create Job
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" /> Refresh
        </Button>
      </div>

      {!jobs?.length ? (
        <Card>
          <CardContent className="py-12 text-center">
            <Cpu className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
            <h3 className="text-lg font-medium mb-2">No Training Jobs</h3>
            <p className="text-sm text-muted-foreground">Create your first training job to get started.</p>
          </CardContent>
        </Card>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Job ID</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Version</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Phase</TableHead>
              <TableHead>Progress</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {jobs.map((job) => (
              <TableRow key={job.job_id} data-testid={`row-job-${job.job_id}`}>
                <TableCell className="font-mono text-xs">{job.job_id.slice(0, 8)}...</TableCell>
                <TableCell className="font-medium">{job.model_name}</TableCell>
                <TableCell className="text-sm text-muted-foreground">{job.version}</TableCell>
                <TableCell>{getStatusBadge(job.status, job.progress_percent)}</TableCell>
                <TableCell className="text-sm text-muted-foreground">{job.current_phase || '—'}</TableCell>
                <TableCell>
                  {job.status === 'running' && (
                    <Progress value={job.progress_percent} className="w-24 h-2" />
                  )}
                </TableCell>
                <TableCell className="text-right">
                  {job.status === 'queued' && (
                    <Button 
                      size="sm" 
                      onClick={() => startJobMutation.mutate(job.job_id)}
                      disabled={startJobMutation.isPending}
                      data-testid={`button-start-job-${job.job_id}`}
                    >
                      {startJobMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4 mr-1" />}
                      Start
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}

interface DeviceExtractionResult {
  job_id: string;
  status: string;
  patients_extracted: number;
  total_readings: number;
  device_types_included: string[];
  consent_verified: boolean;
  hipaa_audit_logged: boolean;
}

const DEVICE_TYPES = [
  { id: 'smartwatch', label: 'Smartwatch', icon: Watch },
  { id: 'bp_monitor', label: 'Blood Pressure Monitor', icon: Activity },
  { id: 'glucose_meter', label: 'Glucose Meter', icon: Droplets },
  { id: 'scale', label: 'Smart Scale', icon: TrendingUp },
  { id: 'thermometer', label: 'Thermometer', icon: Activity },
  { id: 'pulse_oximeter', label: 'Pulse Oximeter', icon: Activity },
];

function DeviceExtractionTab() {
  const { toast } = useToast();
  const [dateRange, setDateRange] = useState(30);
  const [selectedDevices, setSelectedDevices] = useState<string[]>(DEVICE_TYPES.map(d => d.id));
  const [extractionResult, setExtractionResult] = useState<DeviceExtractionResult | null>(null);

  const { data: consentStats } = useQuery<ConsentStats>({
    queryKey: ['/api/ml/training/consent/stats'],
  });

  const extractMutation = useMutation({
    mutationFn: async () => {
      return apiRequest<DeviceExtractionResult>('/api/ml/training/device-data/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_ids: null,
          device_types: selectedDevices.length > 0 ? selectedDevices : null,
          date_range_days: dateRange,
        }),
      });
    },
    onSuccess: (data) => {
      setExtractionResult(data);
      queryClient.invalidateQueries({ queryKey: ['/api/ml/training/contributions/summary'] });
      toast({ 
        title: 'Device Data Extracted', 
        description: `Extracted ${data.total_readings.toLocaleString()} readings from ${data.patients_extracted} patients` 
      });
    },
    onError: (error: Error) => {
      toast({ title: 'Extraction Failed', description: error.message, variant: 'destructive' });
    },
  });

  const toggleDevice = (deviceId: string) => {
    setSelectedDevices(prev => 
      prev.includes(deviceId) 
        ? prev.filter(d => d !== deviceId)
        : [...prev, deviceId]
    );
  };

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card data-testid="card-extraction-config">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Filter className="h-4 w-4 text-blue-500" />
            Extraction Configuration
          </CardTitle>
          <CardDescription>
            Configure which device data to extract for ML training
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label className="flex items-center justify-between">
              <span>Date Range</span>
              <span className="text-sm text-muted-foreground">{dateRange} days</span>
            </Label>
            <Select 
              value={String(dateRange)} 
              onValueChange={(v) => setDateRange(Number(v))}
            >
              <SelectTrigger data-testid="select-date-range">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="7">Last 7 days</SelectItem>
                <SelectItem value="30">Last 30 days</SelectItem>
                <SelectItem value="90">Last 90 days</SelectItem>
                <SelectItem value="180">Last 180 days</SelectItem>
                <SelectItem value="365">Last 365 days</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Device Types</Label>
            <div className="grid grid-cols-2 gap-2">
              {DEVICE_TYPES.map((device) => (
                <Button
                  key={device.id}
                  variant={selectedDevices.includes(device.id) ? "default" : "outline"}
                  size="sm"
                  className="justify-start gap-2"
                  onClick={() => toggleDevice(device.id)}
                  data-testid={`button-device-${device.id}`}
                >
                  <device.icon className="h-4 w-4" />
                  {device.label}
                </Button>
              ))}
            </div>
            <div className="flex gap-2 mt-2">
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => setSelectedDevices(DEVICE_TYPES.map(d => d.id))}
                data-testid="button-select-all-devices"
              >
                Select All
              </Button>
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => setSelectedDevices([])}
                data-testid="button-clear-devices"
              >
                Clear All
              </Button>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-muted/50">
            <div className="flex justify-between items-center text-sm mb-1">
              <span className="text-muted-foreground">Consenting Patients</span>
              <span className="font-semibold" data-testid="text-consenting-count">
                {consentStats?.consenting_patients || 0}
              </span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-muted-foreground">Wearable Consent</span>
              <span className="font-semibold" data-testid="text-wearable-consent">
                {consentStats?.data_type_breakdown?.wearable || 0} patients
              </span>
            </div>
          </div>

          <Button 
            className="w-full"
            onClick={() => extractMutation.mutate()}
            disabled={extractMutation.isPending || selectedDevices.length === 0}
            data-testid="button-extract-device-data"
          >
            {extractMutation.isPending ? (
              <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Extracting...</>
            ) : (
              <><Download className="h-4 w-4 mr-2" /> Extract Device Data</>
            )}
          </Button>
        </CardContent>
      </Card>

      <Card data-testid="card-extraction-results">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <HardDrive className="h-4 w-4 text-green-500" />
            Extraction Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          {extractionResult ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2 p-2 rounded-lg bg-green-100/50 dark:bg-green-900/20">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                <span className="text-sm font-medium">Extraction Complete</span>
                <Badge className="ml-auto bg-green-100 text-green-700 dark:bg-green-900/30">
                  {extractionResult.status}
                </Badge>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-lg bg-muted/50 text-center">
                  <p className="text-xs text-muted-foreground">Patients</p>
                  <p className="text-2xl font-bold" data-testid="text-extracted-patients">
                    {extractionResult.patients_extracted}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-muted/50 text-center">
                  <p className="text-xs text-muted-foreground">Readings</p>
                  <p className="text-2xl font-bold" data-testid="text-extracted-readings">
                    {extractionResult.total_readings.toLocaleString()}
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                <p className="text-sm font-medium">Device Types Included</p>
                <div className="flex flex-wrap gap-1">
                  {extractionResult.device_types_included.map((type) => (
                    <Badge key={type} variant="outline">{type}</Badge>
                  ))}
                </div>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex items-center gap-2">
                  {extractionResult.consent_verified ? (
                    <><CheckCircle2 className="h-4 w-4 text-green-500" /> Consent Verified</>
                  ) : (
                    <><AlertCircle className="h-4 w-4 text-amber-500" /> Consent Pending</>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {extractionResult.hipaa_audit_logged ? (
                    <><CheckCircle2 className="h-4 w-4 text-green-500" /> HIPAA Audit Logged</>
                  ) : (
                    <><AlertCircle className="h-4 w-4 text-amber-500" /> Audit Pending</>
                  )}
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                Job ID: <span className="font-mono">{extractionResult.job_id.slice(0, 12)}...</span>
              </p>
            </div>
          ) : (
            <div className="py-12 text-center">
              <Database className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
              <h3 className="text-sm font-medium mb-2">No Recent Extraction</h3>
              <p className="text-xs text-muted-foreground">
                Configure and run an extraction to see results here.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="md:col-span-2" data-testid="card-consent-verification">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <ShieldCheck className="h-4 w-4 text-purple-500" />
            Device Data Consent Verification
          </CardTitle>
          <CardDescription>
            Only patients with explicit device data consent will have their data extracted
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="p-4 rounded-lg bg-green-100/50 dark:bg-green-900/20">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                <span className="font-medium">Pre-Extraction</span>
              </div>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• ML training consent required</li>
                <li>• Wearable data consent verified</li>
                <li>• Per-device type consent checked</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg bg-blue-100/50 dark:bg-blue-900/20">
              <div className="flex items-center gap-2 mb-2">
                <Lock className="h-5 w-5 text-blue-500" />
                <span className="font-medium">During Extraction</span>
              </div>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Patient IDs anonymized (SHA-256)</li>
                <li>• k-Anonymity (k=5) enforced</li>
                <li>• HIPAA audit log created</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg bg-purple-100/50 dark:bg-purple-900/20">
              <div className="flex items-center gap-2 mb-2">
                <Database className="h-5 w-5 text-purple-500" />
                <span className="font-medium">Post-Extraction</span>
              </div>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Contribution tracked per patient</li>
                <li>• Data linked to training job</li>
                <li>• Withdrawal honored retroactively</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function ConsentAnalyticsTab() {
  const { data: consentStats, isLoading: loadingConsent } = useQuery<ConsentStats>({
    queryKey: ['/api/ml/training/consent/stats'],
  });

  const { data: contributionsSummary, isLoading: loadingContributions } = useQuery<ContributionsSummary>({
    queryKey: ['/api/ml/training/contributions/summary'],
  });

  if (loadingConsent || loadingContributions) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        {[1, 2, 3, 4].map(i => (
          <Card key={i}>
            <CardHeader><Skeleton className="h-6 w-32" /></CardHeader>
            <CardContent><Skeleton className="h-24 w-full" /></CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const consentPieData = consentStats?.data_type_breakdown 
    ? Object.entries(consentStats.data_type_breakdown).map(([key, value]) => ({
        name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        value: value,
      }))
    : [];

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card data-testid="card-consent-overview">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Users className="h-4 w-4 text-blue-500" />
            Consent Overview
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded-lg bg-muted/50 text-center">
              <p className="text-xs text-muted-foreground">Total Patients</p>
              <p className="text-2xl font-bold" data-testid="text-total-patients">
                {consentStats?.total_patients_with_consent_record || 0}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-green-100 dark:bg-green-900/30 text-center">
              <p className="text-xs text-muted-foreground">Consenting</p>
              <p className="text-2xl font-bold text-green-700 dark:text-green-400" data-testid="text-consenting-patients">
                {consentStats?.consenting_patients || 0}
              </p>
            </div>
          </div>
          <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900/30">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">Overall Consent Rate</span>
              <span className="text-lg font-bold text-blue-700 dark:text-blue-400" data-testid="text-consent-rate">
                {consentStats?.consent_rate?.toFixed(1) || 0}%
              </span>
            </div>
            <Progress value={consentStats?.consent_rate || 0} className="h-2" />
          </div>
        </CardContent>
      </Card>

      <Card data-testid="card-consent-breakdown">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <TrendingUp className="h-4 w-4 text-purple-500" />
            Consent by Data Type
          </CardTitle>
        </CardHeader>
        <CardContent>
          {consentPieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={consentPieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {consentPieData.map((entry, index) => (
                    <Cell key={entry.name} fill={CONSENT_COLORS[index % CONSENT_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-48 flex items-center justify-center text-muted-foreground">
              No consent data available
            </div>
          )}
        </CardContent>
      </Card>

      <Card data-testid="card-contributions">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <HardDrive className="h-4 w-4 text-cyan-500" />
            Data Contributions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded-lg bg-muted/50 text-center">
              <p className="text-xs text-muted-foreground">Unique Contributors</p>
              <p className="text-2xl font-bold" data-testid="text-contributors">
                {contributionsSummary?.unique_contributors || 0}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 text-center">
              <p className="text-xs text-muted-foreground">Total Contributions</p>
              <p className="text-2xl font-bold" data-testid="text-contributions">
                {contributionsSummary?.total_contributions || 0}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 text-center">
              <p className="text-xs text-muted-foreground">Records Contributed</p>
              <p className="text-2xl font-bold" data-testid="text-records">
                {contributionsSummary?.total_records_contributed?.toLocaleString() || 0}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 text-center">
              <p className="text-xs text-muted-foreground">Training Jobs</p>
              <p className="text-2xl font-bold" data-testid="text-training-jobs">
                {contributionsSummary?.training_jobs_with_patient_data || 0}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card data-testid="card-hipaa-audit">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <ShieldCheck className="h-4 w-4 text-green-500" />
            HIPAA Compliance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-2 rounded bg-green-100/50 dark:bg-green-900/20">
              <span className="flex items-center gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                Anonymization Enforced
              </span>
              <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30">Active</Badge>
            </div>
            <div className="flex items-center justify-between p-2 rounded bg-green-100/50 dark:bg-green-900/20">
              <span className="flex items-center gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                Consent Verification
              </span>
              <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30">Active</Badge>
            </div>
            <div className="flex items-center justify-between p-2 rounded bg-green-100/50 dark:bg-green-900/20">
              <span className="flex items-center gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                Audit Logging
              </span>
              <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30">Active</Badge>
            </div>
            <div className="flex items-center justify-between p-2 rounded bg-green-100/50 dark:bg-green-900/20">
              <span className="flex items-center gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                k-Anonymity (k=5)
              </span>
              <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30">Enforced</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function MLTrainingHubContent() {
  return (
    <div className="container mx-auto py-8 px-4 space-y-6" data-testid="page-admin-ml-training">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="h-6 w-6 text-purple-500" />
            ML Training Hub
          </h1>
          <p className="text-muted-foreground">
            Manage datasets, training jobs, and model deployments
          </p>
        </div>
        <Badge variant="outline" className="text-purple-500 border-purple-300">
          <Lock className="h-3 w-3 mr-1" />
          2FA Protected
        </Badge>
      </div>

      <LegalDisclaimer />

      <Tabs defaultValue="datasets" className="w-full">
        <TabsList className="flex flex-wrap gap-1 w-full max-w-5xl h-auto p-1">
          <TabsTrigger value="datasets" className="gap-2" data-testid="tab-datasets">
            <Database className="h-4 w-4" />
            Datasets
          </TabsTrigger>
          <TabsTrigger value="jobs" className="gap-2" data-testid="tab-jobs">
            <Cpu className="h-4 w-4" />
            Jobs
          </TabsTrigger>
          <TabsTrigger value="models" className="gap-2" data-testid="tab-models">
            <Brain className="h-4 w-4" />
            Models
          </TabsTrigger>
          <TabsTrigger value="devices" className="gap-2" data-testid="tab-devices">
            <Watch className="h-4 w-4" />
            Devices
          </TabsTrigger>
          <TabsTrigger value="consent" className="gap-2" data-testid="tab-consent">
            <ShieldCheck className="h-4 w-4" />
            Consent
          </TabsTrigger>
          <TabsTrigger value="autopilot" className="gap-2" data-testid="tab-autopilot">
            <Activity className="h-4 w-4" />
            Autopilot
          </TabsTrigger>
          <TabsTrigger value="advanced" className="gap-2" data-testid="tab-advanced">
            <Sparkles className="h-4 w-4" />
            Advanced
          </TabsTrigger>
          <TabsTrigger value="ai-compose" className="gap-2" data-testid="tab-ai-compose">
            <Wand2 className="h-4 w-4" />
            AI Compose
          </TabsTrigger>
          <TabsTrigger value="validation" className="gap-2" data-testid="tab-validation">
            <Shield className="h-4 w-4" />
            Validation
          </TabsTrigger>
          <TabsTrigger value="calibration" className="gap-2" data-testid="tab-calibration">
            <Target className="h-4 w-4" />
            Calibration
          </TabsTrigger>
          <TabsTrigger value="governance" className="gap-2" data-testid="tab-governance">
            <ShieldCheck className="h-4 w-4" />
            Governance
          </TabsTrigger>
          <TabsTrigger value="drift" className="gap-2" data-testid="tab-drift">
            <Activity className="h-4 w-4" />
            Drift
          </TabsTrigger>
          <TabsTrigger value="rlhf" className="gap-2" data-testid="tab-rlhf">
            <MessageSquare className="h-4 w-4" />
            RLHF
          </TabsTrigger>
        </TabsList>

        <TabsContent value="datasets" className="mt-6">
          <DatasetsTab />
        </TabsContent>

        <TabsContent value="jobs" className="mt-6">
          <TrainingJobsPanel />
        </TabsContent>

        <TabsContent value="models" className="mt-6">
          <ModelsTab />
        </TabsContent>

        <TabsContent value="devices" className="mt-6">
          <DeviceExtractionTab />
        </TabsContent>

        <TabsContent value="consent" className="mt-6">
          <ConsentAnalyticsTab />
        </TabsContent>

        <TabsContent value="autopilot" className="mt-6">
          <AutopilotAdminDashboard />
        </TabsContent>

        <TabsContent value="advanced" className="mt-6">
          <AdvancedMLTab />
        </TabsContent>

        <TabsContent value="ai-compose" className="mt-6">
          <JobComposerTab />
        </TabsContent>

        <TabsContent value="validation" className="mt-6">
          <ValidationReportTab />
        </TabsContent>

        <TabsContent value="calibration" className="mt-6">
          <CalibrationReportTab />
        </TabsContent>

        <TabsContent value="governance" className="mt-6">
          <GovernancePackTab />
        </TabsContent>

        <TabsContent value="drift" className="mt-6">
          <DriftMonitorTab />
        </TabsContent>

        <TabsContent value="rlhf" className="mt-6">
          <FeedbackDatasetTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function AdvancedMLTab() {
  const { toast } = useToast();
  const [selectedSection, setSelectedSection] = useState("outbreak");

  const { data: outbreakData, isLoading: outbreakLoading, refetch: refetchOutbreak } = useQuery<{
    model_info: { model_type: string; ensemble_size: number; hidden_units: number };
    prediction: { confidence: number; trend_direction: string };
  }>({
    queryKey: ['/api/v1/ml/advanced/outbreak/status'],
    enabled: selectedSection === "outbreak",
  });

  const { data: embeddingsData, isLoading: embeddingsLoading, refetch: refetchEmbeddings } = useQuery<{
    patient_embeddings_count: number;
    drug_embeddings_count: number;
    location_embeddings_count: number;
    model_info: { architecture: string; embedding_dim: number };
  }>({
    queryKey: ['/api/v1/ml/advanced/embeddings/status'],
    enabled: selectedSection === "embeddings",
  });

  const { data: governanceData, isLoading: governanceLoading } = useQuery<{
    total_protocols: number;
    active_protocols: number;
    total_versions: number;
    pre_specified_count: number;
    exploratory_count: number;
  }>({
    queryKey: ['/api/v1/ml/advanced/governance/stats'],
    enabled: selectedSection === "governance",
  });

  const { data: robustnessData, isLoading: robustnessLoading } = useQuery<{
    reports_count: number;
    latest_report?: {
      overall_status: string;
      checks_passed: number;
      checks_failed: number;
    };
  }>({
    queryKey: ['/api/v1/ml/advanced/robustness/latest'],
    enabled: selectedSection === "robustness",
  });

  const trainOutbreakMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('/api/v1/ml/advanced/outbreak/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          region: 'US',
          lookback_days: 90,
          prediction_horizon: 14
        })
      });
      if (!response.ok) throw new Error('Training failed');
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Training Started", description: "Outbreak prediction model training initiated" });
      refetchOutbreak();
    },
    onError: (error: Error) => {
      toast({ title: "Training Failed", description: error.message, variant: "destructive" });
    }
  });

  const trainEmbeddingsMutation = useMutation({
    mutationFn: async (entityType: string) => {
      const response = await apiRequest('/api/v1/ml/advanced/embeddings/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_type: entityType, embedding_dim: 64 })
      });
      if (!response.ok) throw new Error('Training failed');
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Embedding Training Started", description: "Entity embedding training initiated" });
      refetchEmbeddings();
    },
    onError: (error: Error) => {
      toast({ title: "Training Failed", description: error.message, variant: "destructive" });
    }
  });

  const { data: schedulerData, isLoading: schedulerLoading, refetch: refetchScheduler } = useQuery<{
    running: boolean;
    job_count: number;
    jobs: Array<{
      id: string;
      name: string;
      next_run_time: string | null;
      trigger: string;
    }>;
    last_checked: string;
    error?: string;
  }>({
    queryKey: ['/api/v1/ml/advanced/scheduler/status'],
    enabled: selectedSection === "scheduler",
    refetchInterval: 30000,
  });

  const triggerJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      const response = await apiRequest(`/api/v1/ml/advanced/scheduler/trigger/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) throw new Error('Trigger failed');
      return response.json();
    },
    onSuccess: (data) => {
      toast({ title: "Job Triggered", description: data.message || "Job scheduled for immediate execution" });
      refetchScheduler();
    },
    onError: (error: Error) => {
      toast({ title: "Trigger Failed", description: error.message, variant: "destructive" });
    }
  });

  const sections = [
    { id: "scheduler", label: "Background Scheduler", icon: Clock, color: "text-cyan-500" },
    { id: "outbreak", label: "Outbreak Prediction", icon: Activity, color: "text-red-500" },
    { id: "embeddings", label: "Entity Embeddings", icon: Network, color: "text-blue-500" },
    { id: "governance", label: "Research Governance", icon: Shield, color: "text-purple-500" },
    { id: "robustness", label: "Robustness Checks", icon: TestTube, color: "text-orange-500" },
    { id: "temporal", label: "Temporal Validation", icon: Calendar, color: "text-green-500" },
  ];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-500" />
            Advanced ML Capabilities
          </CardTitle>
          <CardDescription>
            Production-grade ML pipelines for epidemiology research including DeepSurv, trial emulation, and outbreak prediction
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {sections.map((section) => (
              <Button
                key={section.id}
                variant={selectedSection === section.id ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedSection(section.id)}
                className="gap-2"
                data-testid={`btn-section-${section.id}`}
              >
                <section.icon className={`h-4 w-4 ${selectedSection === section.id ? "" : section.color}`} />
                {section.label}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {selectedSection === "scheduler" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-cyan-500" />
              Background Scheduler Status
            </CardTitle>
            <CardDescription>
              APScheduler-based background jobs for auto-reanalysis, risk scoring, ETL pipelines, and ML feature materialization
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {schedulerLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-40 w-full" />
              </div>
            ) : schedulerData?.error ? (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Scheduler Error</AlertTitle>
                <AlertDescription>{schedulerData.error}</AlertDescription>
              </Alert>
            ) : (
              <>
                <div className="flex items-center gap-4 p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-2">
                    {schedulerData?.running ? (
                      <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30">
                        <CheckCircle2 className="h-3 w-3 mr-1" />
                        Running
                      </Badge>
                    ) : (
                      <Badge variant="destructive">
                        <XCircle className="h-3 w-3 mr-1" />
                        Stopped
                      </Badge>
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    <span className="font-medium">{schedulerData?.job_count || 0}</span> active jobs
                  </div>
                  <div className="text-sm text-muted-foreground ml-auto">
                    Last checked: {schedulerData?.last_checked ? format(new Date(schedulerData.last_checked), 'PPpp') : 'N/A'}
                  </div>
                </div>

                {schedulerData?.jobs && schedulerData.jobs.length > 0 && (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Job Name</TableHead>
                        <TableHead>Trigger</TableHead>
                        <TableHead>Next Run</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {schedulerData.jobs.map((job) => (
                        <TableRow key={job.id} data-testid={`row-scheduler-job-${job.id}`}>
                          <TableCell className="font-medium">{job.name}</TableCell>
                          <TableCell>
                            <Badge variant="outline" className="font-mono text-xs">
                              {job.trigger}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {job.next_run_time ? format(new Date(job.next_run_time), 'PPpp') : 'Not scheduled'}
                          </TableCell>
                          <TableCell className="text-right">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => triggerJobMutation.mutate(job.id)}
                              disabled={triggerJobMutation.isPending}
                              data-testid={`btn-trigger-job-${job.id}`}
                            >
                              {triggerJobMutation.isPending ? (
                                <Loader2 className="h-3 w-3 animate-spin" />
                              ) : (
                                <><Play className="h-3 w-3 mr-1" />Run Now</>
                              )}
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}

                {(!schedulerData?.jobs || schedulerData.jobs.length === 0) && (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>No Jobs Scheduled</AlertTitle>
                    <AlertDescription>
                      The scheduler is {schedulerData?.running ? 'running but has no active jobs' : 'not running'}. 
                      Check the Python backend logs for details.
                    </AlertDescription>
                  </Alert>
                )}
              </>
            )}
          </CardContent>
          <CardFooter className="gap-2 flex-wrap">
            <Button variant="outline" onClick={() => refetchScheduler()} data-testid="btn-refresh-scheduler">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh Status
            </Button>
          </CardFooter>
        </Card>
      )}

      {selectedSection === "outbreak" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-red-500" />
              Outbreak Prediction Pipeline
            </CardTitle>
            <CardDescription>
              RNN/Transformer-based models for epidemic curve prediction, R₀ trends, and environmental factors
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {outbreakLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-20 w-full" />
              </div>
            ) : outbreakData ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-2 mb-3">
                    <Brain className="h-4 w-4 text-purple-500" />
                    <span className="font-medium">Model Architecture</span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Type</span>
                      <Badge variant="secondary">{outbreakData.model_info?.model_type || "RNN Ensemble"}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Ensemble Size</span>
                      <span>{outbreakData.model_info?.ensemble_size || 5}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Hidden Units</span>
                      <span>{outbreakData.model_info?.hidden_units || 128}</span>
                    </div>
                  </div>
                </div>
                <div className="p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-2 mb-3">
                    <TrendingUp className="h-4 w-4 text-green-500" />
                    <span className="font-medium">Latest Prediction</span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Confidence</span>
                      <Badge className={outbreakData.prediction?.confidence > 0.7 ? "bg-green-100 text-green-700" : "bg-yellow-100 text-yellow-700"}>
                        {((outbreakData.prediction?.confidence || 0) * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Trend</span>
                      <span className="capitalize">{outbreakData.prediction?.trend_direction || "stable"}</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>No Model Available</AlertTitle>
                <AlertDescription>
                  Train an outbreak prediction model to start generating forecasts
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
          <CardFooter className="gap-2 flex-wrap">
            <Button 
              onClick={() => trainOutbreakMutation.mutate()}
              disabled={trainOutbreakMutation.isPending}
              data-testid="btn-train-outbreak"
            >
              {trainOutbreakMutation.isPending ? (
                <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Training...</>
              ) : (
                <><Rocket className="h-4 w-4 mr-2" />Train Model</>
              )}
            </Button>
            <Button variant="outline" onClick={() => refetchOutbreak()} data-testid="btn-refresh-outbreak">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </CardFooter>
        </Card>
      )}

      {selectedSection === "embeddings" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5 text-blue-500" />
              Entity Embeddings
            </CardTitle>
            <CardDescription>
              Learn dense vector representations for patients, drugs, and locations to improve rare scenario modeling
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {embeddingsLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[1,2,3].map((i) => <Skeleton key={i} className="h-32 w-full" />)}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-2 mb-3">
                    <Users className="h-4 w-4 text-blue-500" />
                    <span className="font-medium">Patient Embeddings</span>
                  </div>
                  <div className="text-3xl font-bold">{embeddingsData?.patient_embeddings_count || 0}</div>
                  <p className="text-sm text-muted-foreground mt-1">Vectors learned</p>
                  <Button 
                    size="sm" 
                    className="mt-3 w-full"
                    onClick={() => trainEmbeddingsMutation.mutate('patient')}
                    disabled={trainEmbeddingsMutation.isPending}
                    data-testid="btn-train-patient-embeddings"
                  >
                    <Play className="h-3 w-3 mr-1" />
                    Train
                  </Button>
                </div>
                <div className="p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-2 mb-3">
                    <Droplets className="h-4 w-4 text-purple-500" />
                    <span className="font-medium">Drug Embeddings</span>
                  </div>
                  <div className="text-3xl font-bold">{embeddingsData?.drug_embeddings_count || 0}</div>
                  <p className="text-sm text-muted-foreground mt-1">Vectors learned</p>
                  <Button 
                    size="sm" 
                    className="mt-3 w-full"
                    onClick={() => trainEmbeddingsMutation.mutate('drug')}
                    disabled={trainEmbeddingsMutation.isPending}
                    data-testid="btn-train-drug-embeddings"
                  >
                    <Play className="h-3 w-3 mr-1" />
                    Train
                  </Button>
                </div>
                <div className="p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-2 mb-3">
                    <MapPin className="h-4 w-4 text-green-500" />
                    <span className="font-medium">Location Embeddings</span>
                  </div>
                  <div className="text-3xl font-bold">{embeddingsData?.location_embeddings_count || 0}</div>
                  <p className="text-sm text-muted-foreground mt-1">Vectors learned</p>
                  <Button 
                    size="sm" 
                    className="mt-3 w-full"
                    onClick={() => trainEmbeddingsMutation.mutate('location')}
                    disabled={trainEmbeddingsMutation.isPending}
                    data-testid="btn-train-location-embeddings"
                  >
                    <Play className="h-3 w-3 mr-1" />
                    Train
                  </Button>
                </div>
              </div>
            )}
            {embeddingsData?.model_info && (
              <div className="p-3 rounded bg-muted/50 text-sm">
                <span className="text-muted-foreground">Architecture: </span>
                <span>{embeddingsData.model_info.architecture}</span>
                <span className="mx-2">•</span>
                <span className="text-muted-foreground">Dimension: </span>
                <span>{embeddingsData.model_info.embedding_dim}</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {selectedSection === "governance" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-purple-500" />
              Research Governance
            </CardTitle>
            <CardDescription>
              Protocol management with versioned specs, snapshot linking, and pre-specified vs exploratory marking
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {governanceLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-lg border bg-card text-center">
                  <div className="text-3xl font-bold text-purple-500">{governanceData?.total_protocols || 0}</div>
                  <p className="text-sm text-muted-foreground">Total Protocols</p>
                </div>
                <div className="p-4 rounded-lg border bg-card text-center">
                  <div className="text-3xl font-bold text-green-500">{governanceData?.active_protocols || 0}</div>
                  <p className="text-sm text-muted-foreground">Active</p>
                </div>
                <div className="p-4 rounded-lg border bg-card text-center">
                  <div className="text-3xl font-bold text-blue-500">{governanceData?.pre_specified_count || 0}</div>
                  <p className="text-sm text-muted-foreground">Pre-Specified</p>
                </div>
                <div className="p-4 rounded-lg border bg-card text-center">
                  <div className="text-3xl font-bold text-orange-500">{governanceData?.exploratory_count || 0}</div>
                  <p className="text-sm text-muted-foreground">Exploratory</p>
                </div>
              </div>
            )}
            <div className="flex items-center gap-2 p-3 rounded bg-green-100/50 dark:bg-green-900/20">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              <span className="text-sm">All protocols include audit trail and version history</span>
            </div>
          </CardContent>
        </Card>
      )}

      {selectedSection === "robustness" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TestTube className="h-5 w-5 text-orange-500" />
              Automated Robustness Checks
            </CardTitle>
            <CardDescription>
              Sample size sufficiency, missingness patterns, causal diagnostics, and subgroup performance analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {robustnessLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : robustnessData?.latest_report ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 rounded-lg border">
                  <div className="flex items-center gap-3">
                    {robustnessData.latest_report.overall_status === "passed" ? (
                      <CheckCircle2 className="h-8 w-8 text-green-500" />
                    ) : (
                      <AlertCircle className="h-8 w-8 text-yellow-500" />
                    )}
                    <div>
                      <div className="font-medium">Latest Report Status</div>
                      <Badge 
                        className={robustnessData.latest_report.overall_status === "passed" 
                          ? "bg-green-100 text-green-700" 
                          : "bg-yellow-100 text-yellow-700"
                        }
                      >
                        {robustnessData.latest_report.overall_status.toUpperCase()}
                      </Badge>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-green-500">{robustnessData.latest_report.checks_passed}</div>
                    <p className="text-sm text-muted-foreground">Checks Passed</p>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {["Sample Size", "Missingness", "Causal Diagnostics", "Subgroup Analysis"].map((check, i) => (
                    <div key={check} className="p-2 rounded border text-center text-sm">
                      <CheckCircle2 className="h-4 w-4 mx-auto mb-1 text-green-500" />
                      {check}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>No Reports</AlertTitle>
                <AlertDescription>Run robustness checks on a trained model to generate a report</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {selectedSection === "temporal" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5 text-green-500" />
              Temporal & Geographic Validation
            </CardTitle>
            <CardDescription>
              Train on earlier periods, validate on recent data. Multi-site validation framework for external generalizability.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-lg border bg-card">
                <div className="flex items-center gap-2 mb-3">
                  <Clock className="h-4 w-4 text-blue-500" />
                  <span className="font-medium">Temporal Splits</span>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Training Period</span>
                    <span>2020-01 to 2023-06</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Validation Period</span>
                    <span>2023-07 to 2024-01</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Test Period</span>
                    <span>2024-02 to Present</span>
                  </div>
                </div>
              </div>
              <div className="p-4 rounded-lg border bg-card">
                <div className="flex items-center gap-2 mb-3">
                  <MapPin className="h-4 w-4 text-green-500" />
                  <span className="font-medium">Geographic Validation</span>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Primary Sites</span>
                    <Badge variant="secondary">3 regions</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Holdout Sites</span>
                    <Badge variant="outline">2 regions</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Cross-validation</span>
                    <Badge className="bg-green-100 text-green-700">Enabled</Badge>
                  </div>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2 p-3 rounded bg-blue-100/50 dark:bg-blue-900/20">
              <GitBranch className="h-4 w-4 text-blue-500" />
              <span className="text-sm">Temporal validation prevents data leakage and ensures real-world model performance</span>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">HIPAA Compliance Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="flex items-center justify-between p-2 rounded bg-green-100/50 dark:bg-green-900/20">
              <span className="flex items-center gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                Consent Verification
              </span>
              <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30">Active</Badge>
            </div>
            <div className="flex items-center justify-between p-2 rounded bg-green-100/50 dark:bg-green-900/20">
              <span className="flex items-center gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                Differential Privacy
              </span>
              <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30">Enabled</Badge>
            </div>
            <div className="flex items-center justify-between p-2 rounded bg-green-100/50 dark:bg-green-900/20">
              <span className="flex items-center gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                Audit Logging
              </span>
              <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30">Active</Badge>
            </div>
            <div className="flex items-center justify-between p-2 rounded bg-green-100/50 dark:bg-green-900/20">
              <span className="flex items-center gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                k-Anonymity (k=10)
              </span>
              <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30">Enforced</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function AdminMLTrainingHub() {
  return (
    <TOTPGate>
      <MLTrainingHubContent />
    </TOTPGate>
  );
}
