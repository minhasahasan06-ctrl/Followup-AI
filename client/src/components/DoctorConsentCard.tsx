import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Shield,
  ShieldCheck,
  ShieldX,
  ChevronDown,
  ChevronUp,
  Lock,
  Unlock,
  Heart,
  FileText,
  Brain,
  MessageSquare,
  Calendar,
  AlertCircle,
  Activity,
  Pill,
  FlaskConical,
  Eye,
  Loader2,
} from "lucide-react";
import { format } from "date-fns";
import type { DoctorPatientConsentPermissions } from "@shared/schema";

interface Doctor {
  id: string;
  firstName?: string;
  lastName?: string;
  first_name?: string;
  last_name?: string;
  email: string;
  specialty?: string;
}

interface DoctorConsentCardProps {
  doctor: Doctor;
  consentPermissions?: DoctorPatientConsentPermissions | null;
  onConsentUpdated?: () => void;
}

interface PermissionItem {
  key: keyof DoctorPatientConsentPermissions;
  label: string;
  description: string;
  icon: any;
}

const permissionItems: PermissionItem[] = [
  { 
    key: "shareHealthData", 
    label: "Health Data", 
    description: "General health information and vitals",
    icon: Heart 
  },
  { 
    key: "shareMedicalFiles", 
    label: "Medical Files", 
    description: "Medical documents and records",
    icon: FileText 
  },
  { 
    key: "shareAIMessages", 
    label: "AI Messages", 
    description: "Conversations with AI health assistant",
    icon: Brain 
  },
  { 
    key: "shareDoctorMessages", 
    label: "Doctor Messages", 
    description: "Communication with healthcare providers",
    icon: MessageSquare 
  },
  { 
    key: "shareDailyFollowups", 
    label: "Daily Followups", 
    description: "Daily health check-ins and symptom reports",
    icon: Calendar 
  },
  { 
    key: "shareHealthAlerts", 
    label: "Health Alerts", 
    description: "AI-generated health warnings and alerts",
    icon: AlertCircle 
  },
  { 
    key: "shareBehavioralInsights", 
    label: "Behavioral Insights", 
    description: "AI analysis of health behavior patterns",
    icon: Activity 
  },
  { 
    key: "sharePainTracking", 
    label: "Pain Tracking", 
    description: "Chronic pain logs and assessments",
    icon: Activity 
  },
  { 
    key: "shareVitalSigns", 
    label: "Vital Signs", 
    description: "Heart rate, blood pressure, temperature",
    icon: Heart 
  },
  { 
    key: "shareMedications", 
    label: "Medications", 
    description: "Current and past medication records",
    icon: Pill 
  },
  { 
    key: "shareLabResults", 
    label: "Lab Results", 
    description: "Laboratory test results and reports",
    icon: FlaskConical 
  },
  { 
    key: "consentEpidemiologicalResearch", 
    label: "Research Data", 
    description: "Anonymous data for medical research",
    icon: Eye 
  },
];

export function DoctorConsentCard({
  doctor,
  consentPermissions,
  onConsentUpdated,
}: DoctorConsentCardProps) {
  const { toast } = useToast();
  const [isExpanded, setIsExpanded] = useState(false);
  const [showWithdrawDialog, setShowWithdrawDialog] = useState(false);
  const [showUpdateDialog, setShowUpdateDialog] = useState(false);
  const [withdrawReason, setWithdrawReason] = useState("");
  const [pendingChanges, setPendingChanges] = useState<Record<string, boolean>>({});

  const doctorName = `Dr. ${doctor.firstName || doctor.first_name || ''} ${doctor.lastName || doctor.last_name || ''}`.trim() || doctor.email;

  const updateConsentMutation = useMutation({
    mutationFn: async (updates: Partial<DoctorPatientConsentPermissions>) => {
      const res = await apiRequest(`/api/patient/doctors/${doctor.id}/consent`, { method: "PATCH", json: updates });
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Consent updated",
        description: "Your data sharing preferences have been updated.",
      });
      queryClient.invalidateQueries({ queryKey: [`/api/patient/doctors/${doctor.id}/consent`] });
      queryClient.invalidateQueries({ queryKey: ["/api/patient/doctors/consents"] });
      onConsentUpdated?.();
      setShowUpdateDialog(false);
      setPendingChanges({});
    },
    onError: (error: any) => {
      toast({
        title: "Update failed",
        description: error.message || "Failed to update consent preferences.",
        variant: "destructive",
      });
    },
  });

  const withdrawAllConsentMutation = useMutation({
    mutationFn: async (reason: string) => {
      const res = await apiRequest(`/api/patient/doctors/${doctor.id}/withdraw-consent`, { method: "POST", json: { reason } });
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Consent withdrawn",
        description: `All data sharing with ${doctorName} has been revoked.`,
      });
      queryClient.invalidateQueries({ queryKey: [`/api/patient/doctors/${doctor.id}/consent`] });
      queryClient.invalidateQueries({ queryKey: ["/api/patient/doctors/consents"] });
      onConsentUpdated?.();
      setShowWithdrawDialog(false);
      setWithdrawReason("");
    },
    onError: (error: any) => {
      toast({
        title: "Withdrawal failed",
        description: error.message || "Failed to withdraw consent. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handlePermissionChange = (key: string, value: boolean) => {
    setPendingChanges(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  const getCurrentValue = (key: keyof DoctorPatientConsentPermissions): boolean => {
    if (key in pendingChanges) {
      return pendingChanges[key];
    }
    return Boolean(consentPermissions?.[key]);
  };

  const hasPendingChanges = Object.keys(pendingChanges).length > 0;

  const handleSaveChanges = () => {
    if (hasPendingChanges) {
      updateConsentMutation.mutate(pendingChanges);
    }
  };

  const handleWithdrawAllConsent = () => {
    withdrawAllConsentMutation.mutate(withdrawReason);
  };

  const grantedCount = permissionItems.filter(p => getCurrentValue(p.key)).length;

  if (!consentPermissions) {
    return (
      <Card className="border-dashed" data-testid={`card-consent-${doctor.id}`}>
        <CardContent className="py-6 text-center">
          <ShieldX className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">
            No consent record found for this doctor.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid={`card-consent-${doctor.id}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-green-600" />
            <CardTitle className="text-base">Data Sharing with {doctorName}</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs">
              {grantedCount}/{permissionItems.length} permissions
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              data-testid={`button-expand-consent-${doctor.id}`}
            >
              {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
          </div>
        </div>
        <CardDescription>
          Manage what health information this doctor can access
        </CardDescription>
      </CardHeader>

      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleContent>
          <CardContent className="pt-0 space-y-4">
            <Separator />
            
            <div className="grid gap-4 sm:grid-cols-2">
              {permissionItems.map((item) => {
                const Icon = item.icon;
                const isGranted = getCurrentValue(item.key);
                const hasChange = item.key in pendingChanges;
                
                return (
                  <div
                    key={item.key}
                    className={`flex items-start gap-3 p-3 rounded-lg border transition-colors ${
                      hasChange 
                        ? "border-primary bg-primary/5" 
                        : isGranted 
                          ? "bg-green-50 border-green-200 dark:bg-green-950/20 dark:border-green-800" 
                          : "bg-muted/50"
                    }`}
                    data-testid={`permission-${item.key}`}
                  >
                    <div className={`mt-0.5 ${isGranted ? "text-green-600" : "text-muted-foreground"}`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <Label 
                          htmlFor={`switch-${item.key}`}
                          className="text-sm font-medium cursor-pointer"
                        >
                          {item.label}
                        </Label>
                        <Switch
                          id={`switch-${item.key}`}
                          checked={isGranted}
                          onCheckedChange={(checked) => handlePermissionChange(item.key, checked)}
                          data-testid={`switch-${item.key}`}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {item.description}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            <Separator />

            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="text-xs text-muted-foreground">
                {consentPermissions.termsAgreedAt && (
                  <span>
                    Consent granted: {format(new Date(consentPermissions.termsAgreedAt), "MMM d, yyyy")}
                    {consentPermissions.termsVersion && ` (v${consentPermissions.termsVersion})`}
                  </span>
                )}
              </div>
              <div className="flex gap-2">
                {hasPendingChanges && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPendingChanges({})}
                    data-testid="button-cancel-changes"
                  >
                    Cancel
                  </Button>
                )}
                <Button
                  variant={hasPendingChanges ? "default" : "outline"}
                  size="sm"
                  onClick={handleSaveChanges}
                  disabled={!hasPendingChanges || updateConsentMutation.isPending}
                  data-testid="button-save-consent"
                >
                  {updateConsentMutation.isPending ? (
                    <>
                      <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    "Save Changes"
                  )}
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => setShowWithdrawDialog(true)}
                  data-testid={`button-withdraw-consent-${doctor.id}`}
                >
                  Withdraw All
                </Button>
              </div>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>

      <AlertDialog open={showWithdrawDialog} onOpenChange={setShowWithdrawDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Withdraw All Consent</AlertDialogTitle>
            <AlertDialogDescription>
              This will revoke all data sharing permissions with {doctorName}. 
              They will no longer be able to access your health records, followups, 
              or any other medical information.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="py-4">
            <Label htmlFor="withdraw-reason">Reason (optional)</Label>
            <Textarea
              id="withdraw-reason"
              value={withdrawReason}
              onChange={(e) => setWithdrawReason(e.target.value)}
              placeholder="Why are you withdrawing consent?"
              className="mt-2"
              data-testid="input-withdraw-reason"
            />
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel data-testid="button-cancel-withdraw">
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleWithdrawAllConsent}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={withdrawAllConsentMutation.isPending}
              data-testid="button-confirm-withdraw"
            >
              {withdrawAllConsentMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Withdrawing...
                </>
              ) : (
                "Withdraw All Consent"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  );
}
