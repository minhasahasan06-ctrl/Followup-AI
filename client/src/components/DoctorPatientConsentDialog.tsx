/**
 * Doctor-Patient Consent Terms Dialog
 * HIPAA-compliant consent management with granular permissions
 * 
 * This dialog appears when a patient approves a doctor's access request,
 * presenting comprehensive terms including:
 * - Health data sharing permissions
 * - Doctor-patient confidentiality agreement
 * - Granular data access controls
 * - Epidemiological research consent
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Shield,
  FileText,
  Pill,
  MessageSquare,
  Activity,
  Brain,
  Heart,
  BarChart3,
  Lock,
  UserCheck,
  AlertTriangle,
  CheckCircle2,
  Info,
  Microscope,
  Loader2,
} from "lucide-react";

interface ConsentPermissions {
  shareHealthData: boolean;
  confidentialityAgreed: boolean;
  shareMedicalFiles: boolean;
  shareMedications: boolean;
  shareAIMessages: boolean;
  shareDoctorMessages: boolean;
  shareDailyFollowups: boolean;
  shareHealthAlerts: boolean;
  shareBehavioralInsights: boolean;
  sharePainTracking: boolean;
  shareVitalSigns: boolean;
  consentEpidemiologicalResearch: boolean;
}

interface DoctorInfo {
  id: string;
  firstName: string;
  lastName: string;
  specialty?: string;
  institution?: string;
}

interface DoctorPatientConsentDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  requestId: string;
  doctor: DoctorInfo;
  requestMessage?: string;
  onConsentGranted?: () => void;
  onConsentDenied?: () => void;
}

const CURRENT_TERMS_VERSION = "1.0";

const defaultPermissions: ConsentPermissions = {
  shareHealthData: false,
  confidentialityAgreed: false,
  shareMedicalFiles: false,
  shareMedications: true,
  shareAIMessages: false,
  shareDoctorMessages: true,
  shareDailyFollowups: true,
  shareHealthAlerts: true,
  shareBehavioralInsights: false,
  sharePainTracking: true,
  shareVitalSigns: true,
  consentEpidemiologicalResearch: false,
};

export function DoctorPatientConsentDialog({
  open,
  onOpenChange,
  requestId,
  doctor,
  requestMessage,
  onConsentGranted,
  onConsentDenied,
}: DoctorPatientConsentDialogProps) {
  const { toast } = useToast();
  const [permissions, setPermissions] = useState<ConsentPermissions>(defaultPermissions);
  const [typedSignature, setTypedSignature] = useState("");
  const [hasReadTerms, setHasReadTerms] = useState(false);
  const [step, setStep] = useState<"terms" | "permissions" | "signature">("terms");

  const grantConsentMutation = useMutation({
    mutationFn: async (data: {
      requestId: string;
      permissions: ConsentPermissions;
      digitalSignature: string;
      termsVersion: string;
    }) => {
      const res = await apiRequest(`/api/patient/consent-requests/${data.requestId}/approve`, {
        method: "POST",
        json: {
          permissions: data.permissions,
          digitalSignature: data.digitalSignature,
          signatureMethod: "typed",
          termsVersion: data.termsVersion,
        }
      });
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Consent Granted",
        description: `Dr. ${doctor.firstName} ${doctor.lastName} now has access to your health information based on your selected permissions.`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/patient/consent-requests"] });
      queryClient.invalidateQueries({ queryKey: ["/api/patient/connected-doctors"] });
      onConsentGranted?.();
      handleClose();
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to grant consent. Please try again.",
        variant: "destructive",
      });
    },
  });

  const denyConsentMutation = useMutation({
    mutationFn: async (requestId: string) => {
      const res = await apiRequest(`/api/patient/consent-requests/${requestId}/deny`, {
        method: "POST",
        json: { reason: "Patient declined consent" }
      });
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Request Denied",
        description: "The doctor's access request has been denied.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/patient/consent-requests"] });
      onConsentDenied?.();
      handleClose();
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to deny request. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleClose = () => {
    onOpenChange(false);
    setPermissions(defaultPermissions);
    setTypedSignature("");
    setHasReadTerms(false);
    setStep("terms");
  };

  const handlePermissionChange = (key: keyof ConsentPermissions, value: boolean) => {
    setPermissions(prev => ({ ...prev, [key]: value }));
  };

  const handleGrantConsent = () => {
    if (!permissions.shareHealthData || !permissions.confidentialityAgreed) {
      toast({
        title: "Required Permissions",
        description: "You must agree to share health data and the confidentiality agreement to proceed.",
        variant: "destructive",
      });
      return;
    }

    if (!typedSignature.trim()) {
      toast({
        title: "Signature Required",
        description: "Please type your full name as your digital signature.",
        variant: "destructive",
      });
      return;
    }

    grantConsentMutation.mutate({
      requestId,
      permissions,
      digitalSignature: typedSignature,
      termsVersion: CURRENT_TERMS_VERSION,
    });
  };

  const canProceedToPermissions = hasReadTerms;
  const canProceedToSignature = permissions.shareHealthData && permissions.confidentialityAgreed;
  const canSubmit = canProceedToSignature && typedSignature.trim().length >= 2;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            Doctor-Patient Consent Agreement
          </DialogTitle>
          <DialogDescription>
            Review and agree to the terms for sharing your health information with{" "}
            <span className="font-semibold">
              Dr. {doctor.firstName} {doctor.lastName}
            </span>
            {doctor.specialty && <span className="text-muted-foreground"> ({doctor.specialty})</span>}
          </DialogDescription>
        </DialogHeader>

        {requestMessage && (
          <Alert className="mx-1">
            <Info className="h-4 w-4" />
            <AlertDescription>
              <span className="font-medium">Doctor's message:</span> {requestMessage}
            </AlertDescription>
          </Alert>
        )}

        <div className="flex gap-2 justify-center my-2">
          <Badge 
            variant={step === "terms" ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setStep("terms")}
          >
            1. Terms
          </Badge>
          <Badge 
            variant={step === "permissions" ? "default" : "outline"}
            className={!canProceedToPermissions ? "opacity-50" : "cursor-pointer"}
            onClick={() => canProceedToPermissions && setStep("permissions")}
          >
            2. Permissions
          </Badge>
          <Badge 
            variant={step === "signature" ? "default" : "outline"}
            className={!canProceedToSignature ? "opacity-50" : "cursor-pointer"}
            onClick={() => canProceedToSignature && setStep("signature")}
          >
            3. Signature
          </Badge>
        </div>

        <ScrollArea className="flex-1 pr-4">
          {step === "terms" && (
            <div className="space-y-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Lock className="h-4 w-4" />
                    HIPAA Privacy Notice
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-2">
                  <p>
                    This consent authorizes the sharing of your Protected Health Information (PHI) 
                    with the healthcare provider listed above. Under the Health Insurance Portability 
                    and Accountability Act (HIPAA), you have the right to:
                  </p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Know how your health information is used and shared</li>
                    <li>Get a copy of your health records</li>
                    <li>Request corrections to your health information</li>
                    <li>Receive notification if there is a breach of your information</li>
                    <li>Revoke this consent at any time</li>
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <UserCheck className="h-4 w-4" />
                    Doctor-Patient Confidentiality
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-2">
                  <p>
                    By granting consent, both you and your healthcare provider agree to the following 
                    confidentiality terms:
                  </p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>
                      <strong>Healthcare Provider Obligations:</strong> The doctor agrees to maintain 
                      strict confidentiality of your health information and use it only for your 
                      medical care and treatment.
                    </li>
                    <li>
                      <strong>Secure Access:</strong> Your information will be accessed through 
                      secure, HIPAA-compliant systems with full audit logging.
                    </li>
                    <li>
                      <strong>Limited Disclosure:</strong> Your information will not be shared with 
                      third parties without your explicit additional consent, except as required by law.
                    </li>
                    <li>
                      <strong>Professional Standards:</strong> The healthcare provider will adhere to 
                      all applicable medical ethics and professional standards.
                    </li>
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4" />
                    Important Information
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-2">
                  <ul className="list-disc pl-5 space-y-1">
                    <li>You can withdraw consent at any time from your profile settings</li>
                    <li>Withdrawing consent will immediately revoke the doctor's access</li>
                    <li>Historical access logs are maintained for audit purposes</li>
                    <li>You will receive notifications when your data is accessed</li>
                  </ul>
                </CardContent>
              </Card>

              <div className="flex items-center space-x-2 pt-2">
                <Checkbox
                  id="read-terms"
                  checked={hasReadTerms}
                  onCheckedChange={(checked) => setHasReadTerms(checked === true)}
                  data-testid="checkbox-read-terms"
                />
                <Label htmlFor="read-terms" className="text-sm font-medium">
                  I have read and understand the terms above
                </Label>
              </div>
            </div>
          )}

          {step === "permissions" && (
            <div className="space-y-4">
              {/* Mandatory Permissions */}
              <Card className="border-primary/50">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2 text-primary">
                    <Shield className="h-4 w-4" />
                    Required Permissions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <Checkbox
                      id="share-health-data"
                      checked={permissions.shareHealthData}
                      onCheckedChange={(checked) => handlePermissionChange("shareHealthData", checked === true)}
                      data-testid="checkbox-share-health-data"
                    />
                    <div className="space-y-1">
                      <Label htmlFor="share-health-data" className="font-medium flex items-center gap-2">
                        <Heart className="h-4 w-4 text-red-500" />
                        Share Health Data
                        <Badge variant="destructive" className="ml-2">Required</Badge>
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Allow the doctor to view your basic health information including conditions and diagnoses
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-3">
                    <Checkbox
                      id="confidentiality-agreed"
                      checked={permissions.confidentialityAgreed}
                      onCheckedChange={(checked) => handlePermissionChange("confidentialityAgreed", checked === true)}
                      data-testid="checkbox-confidentiality-agreed"
                    />
                    <div className="space-y-1">
                      <Label htmlFor="confidentiality-agreed" className="font-medium flex items-center gap-2">
                        <Lock className="h-4 w-4 text-blue-500" />
                        Confidentiality Agreement
                        <Badge variant="destructive" className="ml-2">Required</Badge>
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        I acknowledge the doctor-patient confidentiality terms and agree to this protected relationship
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Data Access Permissions */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    Data Access Permissions
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Accordion type="single" collapsible className="w-full">
                    <AccordionItem value="medical">
                      <AccordionTrigger className="text-sm">
                        <span className="flex items-center gap-2">
                          <FileText className="h-4 w-4" />
                          Medical Records
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-3 pt-2">
                        <div className="flex items-start space-x-3">
                          <Checkbox
                            id="share-medical-files"
                            checked={permissions.shareMedicalFiles}
                            onCheckedChange={(checked) => handlePermissionChange("shareMedicalFiles", checked === true)}
                            data-testid="checkbox-share-medical-files"
                          />
                          <div className="space-y-1">
                            <Label htmlFor="share-medical-files" className="font-medium">Medical Files</Label>
                            <p className="text-xs text-muted-foreground">
                              Lab results, imaging reports, and uploaded documents
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start space-x-3">
                          <Checkbox
                            id="share-medications"
                            checked={permissions.shareMedications}
                            onCheckedChange={(checked) => handlePermissionChange("shareMedications", checked === true)}
                            data-testid="checkbox-share-medications"
                          />
                          <div className="space-y-1">
                            <Label htmlFor="share-medications" className="font-medium flex items-center gap-2">
                              <Pill className="h-4 w-4 text-green-500" />
                              Medications
                            </Label>
                            <p className="text-xs text-muted-foreground">
                              Current and past medications, dosages, and adherence
                            </p>
                          </div>
                        </div>
                      </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="communications">
                      <AccordionTrigger className="text-sm">
                        <span className="flex items-center gap-2">
                          <MessageSquare className="h-4 w-4" />
                          Communications
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-3 pt-2">
                        <div className="flex items-start space-x-3">
                          <Checkbox
                            id="share-ai-messages"
                            checked={permissions.shareAIMessages}
                            onCheckedChange={(checked) => handlePermissionChange("shareAIMessages", checked === true)}
                            data-testid="checkbox-share-ai-messages"
                          />
                          <div className="space-y-1">
                            <Label htmlFor="share-ai-messages" className="font-medium flex items-center gap-2">
                              <Brain className="h-4 w-4 text-purple-500" />
                              AI Assistant Messages
                            </Label>
                            <p className="text-xs text-muted-foreground">
                              Conversations with Clona and Lysa AI assistants
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start space-x-3">
                          <Checkbox
                            id="share-doctor-messages"
                            checked={permissions.shareDoctorMessages}
                            onCheckedChange={(checked) => handlePermissionChange("shareDoctorMessages", checked === true)}
                            data-testid="checkbox-share-doctor-messages"
                          />
                          <div className="space-y-1">
                            <Label htmlFor="share-doctor-messages" className="font-medium">Doctor Messages</Label>
                            <p className="text-xs text-muted-foreground">
                              Previous communications with healthcare providers
                            </p>
                          </div>
                        </div>
                      </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="monitoring">
                      <AccordionTrigger className="text-sm">
                        <span className="flex items-center gap-2">
                          <Activity className="h-4 w-4" />
                          Health Monitoring
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-3 pt-2">
                        <div className="flex items-start space-x-3">
                          <Checkbox
                            id="share-daily-followups"
                            checked={permissions.shareDailyFollowups}
                            onCheckedChange={(checked) => handlePermissionChange("shareDailyFollowups", checked === true)}
                            data-testid="checkbox-share-daily-followups"
                          />
                          <div className="space-y-1">
                            <Label htmlFor="share-daily-followups" className="font-medium">Daily Followups</Label>
                            <p className="text-xs text-muted-foreground">
                              Daily health check-ins, symptoms, and wellness logs
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start space-x-3">
                          <Checkbox
                            id="share-vital-signs"
                            checked={permissions.shareVitalSigns}
                            onCheckedChange={(checked) => handlePermissionChange("shareVitalSigns", checked === true)}
                            data-testid="checkbox-share-vital-signs"
                          />
                          <div className="space-y-1">
                            <Label htmlFor="share-vital-signs" className="font-medium">Vital Signs</Label>
                            <p className="text-xs text-muted-foreground">
                              Blood pressure, heart rate, temperature, and other vitals
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start space-x-3">
                          <Checkbox
                            id="share-pain-tracking"
                            checked={permissions.sharePainTracking}
                            onCheckedChange={(checked) => handlePermissionChange("sharePainTracking", checked === true)}
                            data-testid="checkbox-share-pain-tracking"
                          />
                          <div className="space-y-1">
                            <Label htmlFor="share-pain-tracking" className="font-medium">Pain Tracking</Label>
                            <p className="text-xs text-muted-foreground">
                              Pain levels, locations, and patterns over time
                            </p>
                          </div>
                        </div>
                      </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="ai-insights">
                      <AccordionTrigger className="text-sm">
                        <span className="flex items-center gap-2">
                          <BarChart3 className="h-4 w-4" />
                          AI Insights
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-3 pt-2">
                        <div className="flex items-start space-x-3">
                          <Checkbox
                            id="share-health-alerts"
                            checked={permissions.shareHealthAlerts}
                            onCheckedChange={(checked) => handlePermissionChange("shareHealthAlerts", checked === true)}
                            data-testid="checkbox-share-health-alerts"
                          />
                          <div className="space-y-1">
                            <Label htmlFor="share-health-alerts" className="font-medium">AI Health Alerts</Label>
                            <p className="text-xs text-muted-foreground">
                              Disease risk predictions, deterioration alerts, and AI-generated health insights
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start space-x-3">
                          <Checkbox
                            id="share-behavioral-insights"
                            checked={permissions.shareBehavioralInsights}
                            onCheckedChange={(checked) => handlePermissionChange("shareBehavioralInsights", checked === true)}
                            data-testid="checkbox-share-behavioral-insights"
                          />
                          <div className="space-y-1">
                            <Label htmlFor="share-behavioral-insights" className="font-medium">Behavioral Insights</Label>
                            <p className="text-xs text-muted-foreground">
                              Habit tracking, mental health scores, and behavioral patterns
                            </p>
                          </div>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </CardContent>
              </Card>

              {/* Research Consent */}
              <Card className="border-amber-500/30 bg-amber-500/5">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Microscope className="h-4 w-4 text-amber-600" />
                    Research Participation (Optional)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-start space-x-3">
                    <Checkbox
                      id="consent-research"
                      checked={permissions.consentEpidemiologicalResearch}
                      onCheckedChange={(checked) => handlePermissionChange("consentEpidemiologicalResearch", checked === true)}
                      data-testid="checkbox-consent-research"
                    />
                    <div className="space-y-1">
                      <Label htmlFor="consent-research" className="font-medium">
                        Epidemiological Research Consent
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        I consent to my <strong>anonymized</strong> health data being used for 
                        epidemiological research to advance medical knowledge. My personal 
                        identity will never be disclosed, and I can withdraw this consent at any time.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {step === "signature" && (
            <div className="space-y-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    Summary of Permissions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex items-center gap-2">
                      {permissions.shareHealthData ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <AlertTriangle className="h-4 w-4 text-red-500" />}
                      <span>Health Data</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {permissions.shareMedicalFiles ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <span className="h-4 w-4 text-muted-foreground">—</span>}
                      <span>Medical Files</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {permissions.shareMedications ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <span className="h-4 w-4 text-muted-foreground">—</span>}
                      <span>Medications</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {permissions.shareAIMessages ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <span className="h-4 w-4 text-muted-foreground">—</span>}
                      <span>AI Messages</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {permissions.shareDoctorMessages ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <span className="h-4 w-4 text-muted-foreground">—</span>}
                      <span>Doctor Messages</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {permissions.shareDailyFollowups ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <span className="h-4 w-4 text-muted-foreground">—</span>}
                      <span>Daily Followups</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {permissions.shareHealthAlerts ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <span className="h-4 w-4 text-muted-foreground">—</span>}
                      <span>Health Alerts</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {permissions.shareBehavioralInsights ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <span className="h-4 w-4 text-muted-foreground">—</span>}
                      <span>Behavioral Insights</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {permissions.shareVitalSigns ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <span className="h-4 w-4 text-muted-foreground">—</span>}
                      <span>Vital Signs</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {permissions.sharePainTracking ? 
                        <CheckCircle2 className="h-4 w-4 text-green-500" /> : 
                        <span className="h-4 w-4 text-muted-foreground">—</span>}
                      <span>Pain Tracking</span>
                    </div>
                  </div>
                  <Separator className="my-2" />
                  <div className="flex items-center gap-2 text-sm">
                    {permissions.consentEpidemiologicalResearch ? 
                      <CheckCircle2 className="h-4 w-4 text-amber-500" /> : 
                      <span className="h-4 w-4 text-muted-foreground">—</span>}
                    <span>Research Participation</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    Digital Signature
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <p className="text-sm text-muted-foreground">
                    By typing your full name below, you acknowledge that you have read, understood, 
                    and agree to the terms of this consent agreement and the permissions selected above.
                  </p>
                  <div className="space-y-2">
                    <Label htmlFor="signature">Type your full legal name</Label>
                    <Input
                      id="signature"
                      placeholder="Your Full Name"
                      value={typedSignature}
                      onChange={(e) => setTypedSignature(e.target.value)}
                      className="font-serif text-lg"
                      data-testid="input-signature"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Terms Version: {CURRENT_TERMS_VERSION} | Date: {new Date().toLocaleDateString()}
                  </p>
                </CardContent>
              </Card>
            </div>
          )}
        </ScrollArea>

        <DialogFooter className="flex-shrink-0 gap-2 sm:gap-0">
          {step === "terms" && (
            <>
              <Button
                variant="outline"
                onClick={() => denyConsentMutation.mutate(requestId)}
                disabled={denyConsentMutation.isPending}
                data-testid="button-deny-consent"
              >
                Deny Request
              </Button>
              <Button
                onClick={() => setStep("permissions")}
                disabled={!canProceedToPermissions}
                data-testid="button-continue-to-permissions"
              >
                Continue to Permissions
              </Button>
            </>
          )}

          {step === "permissions" && (
            <>
              <Button
                variant="outline"
                onClick={() => setStep("terms")}
                data-testid="button-back-to-terms"
              >
                Back
              </Button>
              <Button
                onClick={() => setStep("signature")}
                disabled={!canProceedToSignature}
                data-testid="button-continue-to-signature"
              >
                Continue to Signature
              </Button>
            </>
          )}

          {step === "signature" && (
            <>
              <Button
                variant="outline"
                onClick={() => setStep("permissions")}
                data-testid="button-back-to-permissions"
              >
                Back
              </Button>
              <Button
                onClick={handleGrantConsent}
                disabled={!canSubmit || grantConsentMutation.isPending}
                data-testid="button-grant-consent"
              >
                {grantConsentMutation.isPending && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                Grant Consent
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
