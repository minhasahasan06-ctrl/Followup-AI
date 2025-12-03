import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  ChevronDown,
  ChevronUp,
  Heart,
  Activity,
  Thermometer,
  Wind,
  Droplet,
  Shield,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Clock,
  Calendar,
  Phone,
  Mail,
  MapPin,
  AlertCircle,
  CheckCircle2,
  FileText,
  MessageSquare,
  Brain,
  Pill,
  TrendingUp,
  TrendingDown,
  Eye,
  EyeOff,
  Lock,
  Unlock,
} from "lucide-react";
import { format, formatDistanceToNow } from "date-fns";
import type { User, DailyFollowup, DoctorPatientConsentPermissions } from "@shared/schema";

interface PatientSummaryCardProps {
  patient: User | undefined;
  patientId: string;
  followups?: DailyFollowup[];
  medications?: any[];
  consentPermissions?: DoctorPatientConsentPermissions | null;
  isLoadingPermissions?: boolean;
}

interface VitalSign {
  label: string;
  value: number | string | undefined;
  unit: string;
  icon: any;
  normalRange?: { min: number; max: number };
  status?: "normal" | "warning" | "critical";
}

export function PatientSummaryCard({
  patient,
  patientId,
  followups = [],
  medications = [],
  consentPermissions,
  isLoadingPermissions = false,
}: PatientSummaryCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showConsentDetails, setShowConsentDetails] = useState(false);

  const getInitials = (firstName?: string | null, lastName?: string | null) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  const latestFollowup = followups[0];
  
  const getVitalStatus = (value: number | undefined, min: number, max: number): "normal" | "warning" | "critical" => {
    if (value === undefined) return "normal";
    if (value < min * 0.8 || value > max * 1.2) return "critical";
    if (value < min || value > max) return "warning";
    return "normal";
  };

  const vitals: VitalSign[] = [
    {
      label: "Heart Rate",
      value: latestFollowup?.heartRate,
      unit: "bpm",
      icon: Heart,
      normalRange: { min: 60, max: 100 },
      status: getVitalStatus(latestFollowup?.heartRate ?? undefined, 60, 100),
    },
    {
      label: "Blood Pressure",
      value: latestFollowup?.bloodPressureSystolic && latestFollowup?.bloodPressureDiastolic
        ? `${latestFollowup.bloodPressureSystolic}/${latestFollowup.bloodPressureDiastolic}`
        : undefined,
      unit: "mmHg",
      icon: Activity,
    },
    {
      label: "SpO2",
      value: latestFollowup?.oxygenSaturation,
      unit: "%",
      icon: Wind,
      normalRange: { min: 95, max: 100 },
      status: getVitalStatus(latestFollowup?.oxygenSaturation ?? undefined, 95, 100),
    },
    {
      label: "Temperature",
      value: latestFollowup?.temperature,
      unit: "°F",
      icon: Thermometer,
      normalRange: { min: 97, max: 99.5 },
      status: getVitalStatus(latestFollowup?.temperature ?? undefined, 97, 99.5),
    },
    {
      label: "Resp Rate",
      value: latestFollowup?.respiratoryRate,
      unit: "/min",
      icon: Wind,
      normalRange: { min: 12, max: 20 },
      status: getVitalStatus(latestFollowup?.respiratoryRate ?? undefined, 12, 20),
    },
  ];

  const getStatusColor = (status?: "normal" | "warning" | "critical") => {
    switch (status) {
      case "critical": return "text-red-600 bg-red-100 dark:bg-red-900/30";
      case "warning": return "text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30";
      default: return "text-green-600 bg-green-100 dark:bg-green-900/30";
    }
  };

  const overallStatus = vitals.some(v => v.status === "critical") ? "critical"
    : vitals.some(v => v.status === "warning") ? "warning" : "stable";

  const getOverallStatusBadge = () => {
    switch (overallStatus) {
      case "critical": return <Badge variant="destructive">Critical</Badge>;
      case "warning": return <Badge className="bg-yellow-500">Concerning</Badge>;
      default: return <Badge className="bg-green-500">Stable</Badge>;
    }
  };

  const activeMedications = medications?.filter(m => m.status === 'active').length || 0;

  const consentPermissionItems = consentPermissions ? [
    { key: "shareHealthData", label: "Health Data", icon: Heart, granted: consentPermissions.shareHealthData },
    { key: "shareMedicalFiles", label: "Medical Files", icon: FileText, granted: consentPermissions.shareMedicalFiles },
    { key: "shareAIMessages", label: "AI Messages", icon: Brain, granted: consentPermissions.shareAIMessages },
    { key: "shareDoctorMessages", label: "Messages", icon: MessageSquare, granted: consentPermissions.shareDoctorMessages },
    { key: "shareDailyFollowups", label: "Daily Followups", icon: Calendar, granted: consentPermissions.shareDailyFollowups },
    { key: "shareHealthAlerts", label: "Health Alerts", icon: AlertCircle, granted: consentPermissions.shareHealthAlerts },
    { key: "shareBehavioralInsights", label: "Behavioral Insights", icon: Brain, granted: consentPermissions.shareBehavioralInsights },
    { key: "sharePainTracking", label: "Pain Tracking", icon: Activity, granted: consentPermissions.sharePainTracking },
    { key: "shareVitalSigns", label: "Vital Signs", icon: Heart, granted: consentPermissions.shareVitalSigns },
    { key: "consentEpidemiologicalResearch", label: "Research", icon: FileText, granted: consentPermissions.consentEpidemiologicalResearch },
  ] : [];

  const grantedPermissions = consentPermissionItems.filter(p => p.granted).length;
  const totalPermissions = consentPermissionItems.length;

  return (
    <Card className="overflow-hidden" data-testid="card-patient-summary">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16 border-2 border-primary/20">
              <AvatarFallback className="bg-primary text-primary-foreground text-xl font-semibold">
                {getInitials(patient?.firstName, patient?.lastName)}
              </AvatarFallback>
            </Avatar>
            <div>
              <CardTitle className="text-xl" data-testid="text-patient-name">
                {patient?.firstName} {patient?.lastName}
              </CardTitle>
              <CardDescription className="flex items-center gap-2 mt-1">
                <Mail className="h-3 w-3" />
                {patient?.email}
              </CardDescription>
              <div className="flex items-center gap-2 mt-2">
                <Badge variant="outline" className="text-xs">
                  ID: {patientId.slice(0, 8)}...
                </Badge>
                {getOverallStatusBadge()}
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowConsentDetails(!showConsentDetails)}
                    className="gap-2"
                    data-testid="button-toggle-consent"
                  >
                    {isLoadingPermissions ? (
                      <Skeleton className="h-4 w-4 rounded-full" />
                    ) : consentPermissions ? (
                      <ShieldCheck className="h-4 w-4 text-green-500" />
                    ) : (
                      <ShieldX className="h-4 w-4 text-red-500" />
                    )}
                    <span className="text-xs">
                      {grantedPermissions}/{totalPermissions}
                    </span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {consentPermissions 
                    ? `${grantedPermissions} of ${totalPermissions} permissions granted`
                    : "No consent permissions found"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              data-testid="button-expand-summary"
            >
              {isExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {vitals.map((vital, idx) => (
            <div
              key={idx}
              className={`p-3 rounded-lg border ${getStatusColor(vital.status)}`}
              data-testid={`vital-${vital.label.toLowerCase().replace(/\s/g, '-')}`}
            >
              <div className="flex items-center gap-2 mb-1">
                <vital.icon className="h-3 w-3" />
                <span className="text-xs font-medium">{vital.label}</span>
              </div>
              <p className="text-lg font-bold">
                {vital.value ?? "—"} 
                <span className="text-xs font-normal ml-1">{vital.unit}</span>
              </p>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4" />
            <span>Last check-in: {latestFollowup 
              ? formatDistanceToNow(new Date(latestFollowup.date), { addSuffix: true })
              : "No data"}</span>
          </div>
          <div className="flex items-center gap-2">
            <Pill className="h-4 w-4" />
            <span>{activeMedications} active medications</span>
          </div>
        </div>

        <Collapsible open={showConsentDetails} onOpenChange={setShowConsentDetails}>
          <CollapsibleContent className="space-y-3 pt-2">
            <Separator />
            <div className="space-y-2">
              <h4 className="text-sm font-semibold flex items-center gap-2">
                <Shield className="h-4 w-4 text-primary" />
                Consent Permissions
              </h4>
              {isLoadingPermissions ? (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                  {[1,2,3,4,5].map(i => <Skeleton key={i} className="h-8 w-full" />)}
                </div>
              ) : consentPermissions ? (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                  {consentPermissionItems.map((perm) => (
                    <div
                      key={perm.key}
                      className={`flex items-center gap-2 p-2 rounded-md text-xs ${
                        perm.granted 
                          ? "bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400" 
                          : "bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-400"
                      }`}
                      data-testid={`consent-${perm.key}`}
                    >
                      {perm.granted ? <Unlock className="h-3 w-3" /> : <Lock className="h-3 w-3" />}
                      <span>{perm.label}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  No consent permissions configured for this patient.
                </p>
              )}
              {consentPermissions?.termsAgreedAt && (
                <p className="text-xs text-muted-foreground">
                  Consent granted on {format(new Date(consentPermissions.termsAgreedAt), "MMM d, yyyy 'at' h:mm a")}
                  {consentPermissions.termsVersion && ` • Terms v${consentPermissions.termsVersion}`}
                </p>
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
          <CollapsibleContent className="space-y-4 pt-2">
            <Separator />
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-3">
                <h4 className="text-sm font-semibold">Contact Information</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Mail className="h-4 w-4" />
                    <span>{patient?.email || "Not provided"}</span>
                  </div>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Phone className="h-4 w-4" />
                    <span>{patient?.phone || "Not provided"}</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="text-sm font-semibold">Recent Activity</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Followups this week</span>
                    <Badge variant="secondary">{followups.length}</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Active medications</span>
                    <Badge variant="secondary">{activeMedications}</Badge>
                  </div>
                </div>
              </div>
            </div>

            {followups.length > 0 && (
              <>
                <Separator />
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold">Recent Symptoms</h4>
                  <div className="flex flex-wrap gap-2">
                    {followups.slice(0, 3).flatMap(f => 
                      f.symptoms?.slice(0, 3).map((symptom, idx) => (
                        <Badge key={`${f.id}-${idx}`} variant="outline" className="text-xs">
                          {typeof symptom === 'string' ? symptom : symptom.name || 'Unknown'}
                        </Badge>
                      )) || []
                    )}
                    {followups[0]?.symptoms?.length === 0 && (
                      <span className="text-sm text-muted-foreground">No symptoms reported</span>
                    )}
                  </div>
                </div>
              </>
            )}
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
}
