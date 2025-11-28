import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
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
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Shield,
  CheckCircle,
  XCircle,
  Clock,
  Stethoscope,
  Building2,
  Mail,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import { useState } from "react";
import { formatDistanceToNow } from "date-fns";

type ConsentRequest = {
  id: string;
  doctorId: string;
  patientId: string;
  requestMessage: string | null;
  accessLevel: string;
  status: string;
  createdAt: string;
  expiresAt: string;
  doctor: {
    id: string;
    firstName: string | null;
    lastName: string | null;
    email: string;
  } | null;
  doctorProfile: {
    specialty: string | null;
    licenseNumber: string | null;
    hospitalAffiliation: string | null;
  } | null;
};

export function PatientConsentRequests() {
  const { toast } = useToast();
  const [selectedRequest, setSelectedRequest] = useState<ConsentRequest | null>(null);
  const [showApproveDialog, setShowApproveDialog] = useState(false);
  const [showDenyDialog, setShowDenyDialog] = useState(false);
  const [responseMessage, setResponseMessage] = useState("");

  const { data: pendingRequests = [], isLoading } = useQuery<ConsentRequest[]>({
    queryKey: ["/api/patient/consent-requests/pending"],
  });

  const respondToRequest = useMutation({
    mutationFn: async (data: { id: string; approved: boolean; responseMessage?: string }) => {
      const res = await apiRequest("POST", `/api/patient/consent-requests/${data.id}/respond`, {
        approved: data.approved,
        responseMessage: data.responseMessage,
      });
      return res.json();
    },
    onSuccess: (_, variables) => {
      const action = variables.approved ? "approved" : "denied";
      toast({
        title: `Access ${action.charAt(0).toUpperCase() + action.slice(1)}`,
        description: variables.approved 
          ? "The doctor can now view your health records."
          : "The doctor's request has been denied.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/patient/consent-requests/pending"] });
      setSelectedRequest(null);
      setShowApproveDialog(false);
      setShowDenyDialog(false);
      setResponseMessage("");
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to respond to the request.",
        variant: "destructive",
      });
    },
  });

  const handleApprove = () => {
    if (!selectedRequest) return;
    respondToRequest.mutate({
      id: selectedRequest.id,
      approved: true,
      responseMessage,
    });
  };

  const handleDeny = () => {
    if (!selectedRequest) return;
    respondToRequest.mutate({
      id: selectedRequest.id,
      approved: false,
      responseMessage,
    });
  };

  const getInitials = (firstName?: string | null, lastName?: string | null) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  const getDoctorName = (request: ConsentRequest) => {
    if (request.doctor?.firstName && request.doctor?.lastName) {
      return `Dr. ${request.doctor.firstName} ${request.doctor.lastName}`;
    }
    return "A Doctor";
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            <div className="h-6 bg-muted rounded w-1/3" />
            <div className="h-20 bg-muted rounded" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (pendingRequests.length === 0) {
    return null;
  }

  return (
    <>
      <Card className="border-primary/20 bg-primary/5">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Access Requests</CardTitle>
          </div>
          <CardDescription>
            Healthcare providers are requesting access to your health records
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {pendingRequests.map((request) => (
            <Card key={request.id} className="hover-elevate" data-testid={`consent-request-${request.id}`}>
              <CardContent className="p-4">
                <div className="flex items-start gap-4">
                  <Avatar className="h-12 w-12">
                    <AvatarFallback className="bg-primary text-primary-foreground">
                      {getInitials(request.doctor?.firstName, request.doctor?.lastName)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold">{getDoctorName(request)}</h4>
                      <Badge variant="outline" className="text-xs">
                        <Clock className="h-3 w-3 mr-1" />
                        {formatDistanceToNow(new Date(request.createdAt), { addSuffix: true })}
                      </Badge>
                    </div>
                    {request.doctorProfile?.specialty && (
                      <div className="flex items-center gap-1 text-sm text-muted-foreground mb-1">
                        <Stethoscope className="h-3 w-3" />
                        <span>{request.doctorProfile.specialty}</span>
                      </div>
                    )}
                    {request.doctorProfile?.hospitalAffiliation && (
                      <div className="flex items-center gap-1 text-sm text-muted-foreground mb-1">
                        <Building2 className="h-3 w-3" />
                        <span>{request.doctorProfile.hospitalAffiliation}</span>
                      </div>
                    )}
                    {request.doctor?.email && (
                      <div className="flex items-center gap-1 text-sm text-muted-foreground">
                        <Mail className="h-3 w-3" />
                        <span>{request.doctor.email}</span>
                      </div>
                    )}
                    {request.requestMessage && (
                      <p className="mt-2 text-sm bg-muted/50 rounded-md p-2 italic">
                        "{request.requestMessage}"
                      </p>
                    )}
                  </div>
                </div>
                <Separator className="my-3" />
                <div className="flex gap-2">
                  <Button
                    variant="default"
                    className="flex-1"
                    onClick={() => {
                      setSelectedRequest(request);
                      setShowApproveDialog(true);
                    }}
                    data-testid={`button-approve-${request.id}`}
                  >
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Approve Access
                  </Button>
                  <Button
                    variant="outline"
                    className="flex-1"
                    onClick={() => {
                      setSelectedRequest(request);
                      setShowDenyDialog(true);
                    }}
                    data-testid={`button-deny-${request.id}`}
                  >
                    <XCircle className="h-4 w-4 mr-2" />
                    Deny
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </CardContent>
      </Card>

      <Dialog open={showApproveDialog} onOpenChange={setShowApproveDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              Approve Access
            </DialogTitle>
            <DialogDescription>
              You are about to grant {selectedRequest ? getDoctorName(selectedRequest) : "this doctor"} access 
              to view your health records.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="bg-muted/50 rounded-lg p-4 text-sm">
              <h4 className="font-medium mb-2">What this means:</h4>
              <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                <li>The doctor can view your health history and records</li>
                <li>They can see your symptom journals and daily followups</li>
                <li>They can prescribe medications and manage your treatment</li>
                <li>You can revoke access at any time</li>
              </ul>
            </div>
            <div className="space-y-2">
              <Label htmlFor="approveMessage">Optional Message</Label>
              <Textarea
                id="approveMessage"
                placeholder="Add a message for the doctor (optional)..."
                value={responseMessage}
                onChange={(e) => setResponseMessage(e.target.value)}
                rows={2}
                data-testid="input-approve-message"
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowApproveDialog(false);
                setResponseMessage("");
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleApprove}
              disabled={respondToRequest.isPending}
              data-testid="button-confirm-approve"
            >
              {respondToRequest.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <CheckCircle className="h-4 w-4 mr-2" />
              )}
              Grant Access
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={showDenyDialog} onOpenChange={setShowDenyDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
              Deny Access Request
            </AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to deny {selectedRequest ? getDoctorName(selectedRequest) : "this doctor"}'s 
              request to access your health records? They will not be able to view your data.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="space-y-2 my-4">
            <Label htmlFor="denyMessage">Reason (Optional)</Label>
            <Textarea
              id="denyMessage"
              placeholder="Provide a reason for denying access (optional)..."
              value={responseMessage}
              onChange={(e) => setResponseMessage(e.target.value)}
              rows={2}
              data-testid="input-deny-message"
            />
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setResponseMessage("")}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeny}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={respondToRequest.isPending}
              data-testid="button-confirm-deny"
            >
              {respondToRequest.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <XCircle className="h-4 w-4 mr-2" />
              )}
              Deny Access
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

export function PatientFollowupId() {
  const { toast } = useToast();
  const { data, isLoading } = useQuery<{ followupPatientId: string }>({
    queryKey: ["/api/patient/followup-id"],
  });

  const copyToClipboard = () => {
    if (data?.followupPatientId) {
      navigator.clipboard.writeText(data.followupPatientId);
      toast({
        title: "Copied!",
        description: "Your Patient ID has been copied to clipboard.",
      });
    }
  };

  if (isLoading || !data) {
    return null;
  }

  return (
    <Card className="bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20">
      <CardContent className="p-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Shield className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Your Followup Patient ID</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Share this ID with your doctor to connect with them
            </p>
          </div>
          <Button
            variant="outline"
            onClick={copyToClipboard}
            className="font-mono text-lg tracking-wider"
            data-testid="button-copy-patient-id"
          >
            {data.followupPatientId}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
