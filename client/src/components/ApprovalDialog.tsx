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
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  Edit3,
  Pill,
  User,
  Clock,
  Shield,
  Loader2,
  FileText
} from "lucide-react";
import { cn } from "@/lib/utils";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { format } from "date-fns";

interface ApprovalRequest {
  id: string;
  toolName: string;
  reason: string;
  parameters?: Record<string, unknown>;
  timestamp: string;
  status: "pending" | "approved" | "rejected" | "modified";
  patientId?: string;
  patientName?: string;
  doctorId?: string;
  urgency?: "routine" | "urgent" | "stat";
  expiresAt?: string;
}

interface ApprovalDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  approval: ApprovalRequest | null;
  onApprove?: (id: string, notes?: string) => void;
  onReject?: (id: string, reason: string) => void;
  onModify?: (id: string, modifications: Record<string, unknown>) => void;
}

export function ApprovalDialog({
  open,
  onOpenChange,
  approval,
  onApprove,
  onReject,
  onModify
}: ApprovalDialogProps) {
  const [mode, setMode] = useState<"review" | "modify" | "reject">("review");
  const [rejectReason, setRejectReason] = useState("");
  const [approvalNotes, setApprovalNotes] = useState("");
  const [modifications, setModifications] = useState<Record<string, string>>({});

  const submitDecisionMutation = useMutation({
    mutationFn: async (decision: { action: "approve" | "reject" | "modify"; notes?: string; reason?: string; modifications?: Record<string, unknown> }) => {
      if (!approval) throw new Error("No approval request");
      
      return apiRequest(`/api/agent/approvals/${approval.id}/decision`, {
        method: "POST",
        body: JSON.stringify({
          decision: decision.action,
          notes: decision.notes,
          rejection_reason: decision.reason,
          modifications: decision.modifications
        })
      });
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["/api/agent/approvals"] });
      queryClient.invalidateQueries({ queryKey: ["/api/agent/messages"] });
      
      if (variables.action === "approve" && onApprove) {
        onApprove(approval!.id, variables.notes);
      } else if (variables.action === "reject" && onReject) {
        onReject(approval!.id, variables.reason || "");
      } else if (variables.action === "modify" && onModify) {
        onModify(approval!.id, variables.modifications || {});
      }
      
      handleClose();
    }
  });

  const handleClose = () => {
    setMode("review");
    setRejectReason("");
    setApprovalNotes("");
    setModifications({});
    onOpenChange(false);
  };

  const handleApprove = () => {
    submitDecisionMutation.mutate({
      action: "approve",
      notes: approvalNotes
    });
  };

  const handleReject = () => {
    if (!rejectReason.trim()) return;
    submitDecisionMutation.mutate({
      action: "reject",
      reason: rejectReason
    });
  };

  const handleModify = () => {
    submitDecisionMutation.mutate({
      action: "modify",
      modifications
    });
  };

  if (!approval) return null;

  const prescription = approval.parameters?.prescription as Record<string, unknown> | undefined;
  const interactions = approval.parameters?.interactions as Record<string, unknown>[] | undefined;
  const hasInteractions = interactions && interactions.length > 0;

  const getUrgencyBadge = (urgency?: string) => {
    switch (urgency) {
      case "stat":
        return <Badge variant="destructive" className="gap-1"><AlertTriangle className="h-3 w-3" />STAT</Badge>;
      case "urgent":
        return <Badge className="gap-1 bg-yellow-500"><Clock className="h-3 w-3" />Urgent</Badge>;
      default:
        return <Badge variant="outline" className="gap-1"><Clock className="h-3 w-3" />Routine</Badge>;
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              Human-in-the-Loop Approval Required
            </DialogTitle>
            {getUrgencyBadge(approval.urgency)}
          </div>
          <DialogDescription>
            {approval.reason || "This action requires your explicit approval before proceeding."}
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="flex-1 -mx-6 px-6">
          <div className="space-y-4 py-4">
            {approval.patientName && (
              <Card data-testid="approval-patient-info">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <User className="h-4 w-4 text-muted-foreground" />
                    <span className="font-medium">Patient</span>
                    <Badge variant="outline" className="ml-auto text-xs gap-1" data-testid="badge-phi-indicator">
                      <Shield className="h-3 w-3" />
                      PHI Access
                    </Badge>
                  </div>
                  <p className="text-sm" data-testid="text-patient-name">{approval.patientName}</p>
                  {approval.patientId && (
                    <p className="text-xs text-muted-foreground mt-1" data-testid="text-patient-id">
                      ID: {approval.patientId}
                    </p>
                  )}
                </CardContent>
              </Card>
            )}

            {prescription && (
              <Card data-testid="approval-prescription-details">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Pill className="h-4 w-4 text-primary" />
                    <span className="font-medium">Prescription Details</span>
                  </div>

                  {mode === "modify" ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <Label htmlFor="drug-name">Drug Name</Label>
                          <Input
                            id="drug-name"
                            defaultValue={prescription.drugName as string}
                            onChange={(e) => setModifications(prev => ({ ...prev, drugName: e.target.value }))}
                            data-testid="input-modify-drug-name"
                          />
                        </div>
                        <div>
                          <Label htmlFor="dosage">Dosage</Label>
                          <Input
                            id="dosage"
                            defaultValue={prescription.dosage as string}
                            onChange={(e) => setModifications(prev => ({ ...prev, dosage: e.target.value }))}
                            data-testid="input-modify-dosage"
                          />
                        </div>
                        <div>
                          <Label htmlFor="frequency">Frequency</Label>
                          <Input
                            id="frequency"
                            defaultValue={prescription.frequency as string}
                            onChange={(e) => setModifications(prev => ({ ...prev, frequency: e.target.value }))}
                            data-testid="input-modify-frequency"
                          />
                        </div>
                        <div>
                          <Label htmlFor="duration">Duration</Label>
                          <Input
                            id="duration"
                            defaultValue={prescription.duration as string}
                            onChange={(e) => setModifications(prev => ({ ...prev, duration: e.target.value }))}
                            data-testid="input-modify-duration"
                          />
                        </div>
                      </div>
                      <div>
                        <Label htmlFor="instructions">Instructions</Label>
                        <Textarea
                          id="instructions"
                          defaultValue={prescription.instructions as string}
                          onChange={(e) => setModifications(prev => ({ ...prev, instructions: e.target.value }))}
                          data-testid="input-modify-instructions"
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-2 text-sm" data-testid="prescription-review-details">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                        <div>
                          <span className="text-muted-foreground">Drug:</span>
                          <span className="ml-2 font-medium" data-testid="text-drug-name">{prescription.drugName as string}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Dosage:</span>
                          <span className="ml-2 font-medium" data-testid="text-dosage">{prescription.dosage as string}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Frequency:</span>
                          <span className="ml-2 font-medium" data-testid="text-frequency">{prescription.frequency as string}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Duration:</span>
                          <span className="ml-2 font-medium" data-testid="text-duration">{prescription.duration as string}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Refills:</span>
                          <span className="ml-2 font-medium" data-testid="text-refills">{prescription.refills as number}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Route:</span>
                          <span className="ml-2 font-medium" data-testid="text-route">{prescription.route as string || "Oral"}</span>
                        </div>
                      </div>
                      {prescription.instructions && (
                        <div className="mt-2 p-2 bg-muted rounded" data-testid="text-instructions">
                          <span className="text-muted-foreground">Instructions: </span>
                          {prescription.instructions as string}
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {hasInteractions && (
              <Card className="border-red-200 dark:border-red-900" data-testid="approval-drug-interactions">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-3 text-red-600 dark:text-red-400">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="font-medium">Drug Interactions Detected</span>
                  </div>
                  <div className="space-y-2" data-testid="drug-interactions-list">
                    {interactions!.map((int, i) => (
                      <div key={i} className="p-2 bg-red-50 dark:bg-red-950/30 rounded text-sm" data-testid={`drug-interaction-${i}`}>
                        <div className="flex items-center justify-between">
                          <span className="font-medium" data-testid={`text-interaction-drugs-${i}`}>
                            {int.drug1 as string} + {int.drug2 as string}
                          </span>
                          <Badge variant={int.severity === "severe" ? "destructive" : "outline"} data-testid={`badge-severity-${i}`}>
                            {int.severity as string}
                          </Badge>
                        </div>
                        {int.description && (
                          <p className="text-red-600 dark:text-red-400 mt-1" data-testid={`text-interaction-desc-${i}`}>
                            {int.description as string}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {mode === "reject" && (
              <Card data-testid="approval-reject-form">
                <CardContent className="p-4">
                  <Label htmlFor="reject-reason">Rejection Reason (Required)</Label>
                  <Textarea
                    id="reject-reason"
                    value={rejectReason}
                    onChange={(e) => setRejectReason(e.target.value)}
                    placeholder="Please provide a reason for rejection..."
                    className="mt-2"
                    data-testid="input-reject-reason"
                  />
                </CardContent>
              </Card>
            )}

            {mode === "review" && (
              <div>
                <Label htmlFor="approval-notes">Notes (Optional)</Label>
                <Textarea
                  id="approval-notes"
                  value={approvalNotes}
                  onChange={(e) => setApprovalNotes(e.target.value)}
                  placeholder="Add any notes for the record..."
                  className="mt-2"
                  data-testid="input-approval-notes"
                />
              </div>
            )}

            <Separator />

            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <Shield className="h-3 w-3" />
                <span>HIPAA-compliant audit logging enabled</span>
              </div>
              <span>Requested: {format(new Date(approval.timestamp), "PPp")}</span>
            </div>

            {approval.expiresAt && (
              <div className="p-2 bg-yellow-50 dark:bg-yellow-950/30 rounded text-sm text-yellow-600 dark:text-yellow-400 flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Expires: {format(new Date(approval.expiresAt), "PPp")}
              </div>
            )}
          </div>
        </ScrollArea>

        <DialogFooter className="flex-shrink-0 gap-2 sm:gap-2">
          {mode === "review" && (
            <>
              <Button
                variant="outline"
                onClick={() => setMode("reject")}
                className="gap-2"
                data-testid="button-start-reject"
              >
                <XCircle className="h-4 w-4" />
                Reject
              </Button>
              <Button
                variant="outline"
                onClick={() => setMode("modify")}
                className="gap-2"
                data-testid="button-start-modify"
              >
                <Edit3 className="h-4 w-4" />
                Modify
              </Button>
              <Button
                onClick={handleApprove}
                disabled={submitDecisionMutation.isPending}
                className="gap-2"
                data-testid="button-approve"
              >
                {submitDecisionMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <CheckCircle className="h-4 w-4" />
                )}
                Approve
              </Button>
            </>
          )}

          {mode === "reject" && (
            <>
              <Button
                variant="ghost"
                onClick={() => setMode("review")}
                data-testid="button-cancel-reject"
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleReject}
                disabled={!rejectReason.trim() || submitDecisionMutation.isPending}
                className="gap-2"
                data-testid="button-confirm-reject"
              >
                {submitDecisionMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <XCircle className="h-4 w-4" />
                )}
                Confirm Rejection
              </Button>
            </>
          )}

          {mode === "modify" && (
            <>
              <Button
                variant="ghost"
                onClick={() => setMode("review")}
                data-testid="button-cancel-modify"
              >
                Cancel
              </Button>
              <Button
                onClick={handleModify}
                disabled={Object.keys(modifications).length === 0 || submitDecisionMutation.isPending}
                className="gap-2"
                data-testid="button-confirm-modify"
              >
                {submitDecisionMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Edit3 className="h-4 w-4" />
                )}
                Submit Modifications
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function ApprovalBadge({ 
  count, 
  onClick 
}: { 
  count: number; 
  onClick?: () => void;
}) {
  if (count === 0) return null;
  
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className="gap-2 border-yellow-500 text-yellow-600 hover:bg-yellow-50 dark:hover:bg-yellow-950/30"
      data-testid="button-pending-approvals"
    >
      <AlertTriangle className="h-4 w-4" />
      {count} Pending Approval{count > 1 ? "s" : ""}
    </Button>
  );
}

export default ApprovalDialog;
