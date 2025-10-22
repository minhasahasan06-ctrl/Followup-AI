import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { Shield, CheckCircle, XCircle, Calendar, Building, Globe } from "lucide-react";
import { useState } from "react";

export default function AdminVerification() {
  const { toast } = useToast();
  const [selectedDoctor, setSelectedDoctor] = useState<any>(null);
  const [showVerifyDialog, setShowVerifyDialog] = useState(false);
  const [showRejectDialog, setShowRejectDialog] = useState(false);
  const [notes, setNotes] = useState("");

  const { data: pendingDoctors, isLoading } = useQuery<any[]>({
    queryKey: ["/api/admin/pending-doctors"],
  });

  const verifyMutation = useMutation({
    mutationFn: async ({ id, notes }: { id: string; notes: string }) => {
      return await apiRequest(`/api/admin/verify-doctor/${id}`, {
        method: "POST",
        body: JSON.stringify({ notes }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/admin/pending-doctors"] });
      setShowVerifyDialog(false);
      setSelectedDoctor(null);
      setNotes("");
      toast({
        title: "Doctor Verified",
        description: "The doctor's license has been successfully verified.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to verify doctor. Please try again.",
        variant: "destructive",
      });
    },
  });

  const rejectMutation = useMutation({
    mutationFn: async ({ id, reason }: { id: string; reason: string }) => {
      return await apiRequest(`/api/admin/reject-doctor/${id}`, {
        method: "POST",
        body: JSON.stringify({ reason }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/admin/pending-doctors"] });
      setShowRejectDialog(false);
      setSelectedDoctor(null);
      setNotes("");
      toast({
        title: "Verification Rejected",
        description: "The doctor's license verification has been rejected.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to reject verification. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleVerify = () => {
    if (selectedDoctor) {
      verifyMutation.mutate({ id: selectedDoctor.id, notes });
    }
  };

  const handleReject = () => {
    if (selectedDoctor && notes) {
      rejectMutation.mutate({ id: selectedDoctor.id, reason: notes });
    } else {
      toast({
        title: "Reason Required",
        description: "Please provide a reason for rejection.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Shield className="h-8 w-8 text-primary" />
          <div>
            <h1 className="text-3xl font-bold">Admin: Doctor License Verification</h1>
            <p className="text-muted-foreground">
              Review and verify doctor license applications
            </p>
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="grid gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-4 bg-muted rounded w-1/3" />
                <div className="h-3 bg-muted rounded w-1/2 mt-2" />
              </CardHeader>
            </Card>
          ))}
        </div>
      ) : pendingDoctors && pendingDoctors.length > 0 ? (
        <div className="grid gap-4">
          {pendingDoctors.map((doctor: any) => (
            <Card key={doctor.id} className="hover-elevate" data-testid={`card-doctor-${doctor.id}`}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle data-testid={`text-doctor-name-${doctor.id}`}>
                      Dr. {doctor.firstName} {doctor.lastName}
                    </CardTitle>
                    <CardDescription className="mt-1" data-testid={`text-doctor-email-${doctor.id}`}>
                      {doctor.email}
                    </CardDescription>
                  </div>
                  <Badge variant="secondary" data-testid={`badge-status-${doctor.id}`}>
                    Pending Verification
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4 mb-4">
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <Building className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Organization:</span>
                      <span className="font-medium" data-testid={`text-organization-${doctor.id}`}>
                        {doctor.doctorProfile?.organization || "N/A"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Globe className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">License Country:</span>
                      <span className="font-medium" data-testid={`text-license-country-${doctor.id}`}>
                        {doctor.doctorProfile?.licenseCountry || "N/A"}
                      </span>
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <Shield className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">License Number:</span>
                      <span className="font-mono font-medium" data-testid={`text-license-number-${doctor.id}`}>
                        {doctor.doctorProfile?.medicalLicenseNumber || "N/A"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Applied:</span>
                      <span data-testid={`text-applied-date-${doctor.id}`}>
                        {new Date(doctor.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>

                {doctor.doctorProfile?.kycPhotoUrl && (
                  <div className="mb-4">
                    <p className="text-sm text-muted-foreground mb-2">KYC Document:</p>
                    <a
                      href={doctor.doctorProfile.kycPhotoUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline text-sm"
                      data-testid={`link-kyc-${doctor.id}`}
                    >
                      View uploaded document
                    </a>
                  </div>
                )}

                <div className="flex gap-3 pt-2 border-t">
                  <Button
                    variant="default"
                    className="flex-1"
                    onClick={() => {
                      setSelectedDoctor(doctor);
                      setShowVerifyDialog(true);
                    }}
                    data-testid={`button-verify-${doctor.id}`}
                  >
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Verify License
                  </Button>
                  <Button
                    variant="destructive"
                    className="flex-1"
                    onClick={() => {
                      setSelectedDoctor(doctor);
                      setShowRejectDialog(true);
                    }}
                    data-testid={`button-reject-${doctor.id}`}
                  >
                    <XCircle className="h-4 w-4 mr-2" />
                    Reject
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Shield className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Pending Verifications</h3>
            <p className="text-muted-foreground text-center">
              All doctor license verifications have been processed
            </p>
          </CardContent>
        </Card>
      )}

      {/* Verify Dialog */}
      <Dialog open={showVerifyDialog} onOpenChange={setShowVerifyDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Verify Doctor License</DialogTitle>
            <DialogDescription>
              Confirm that you have verified the medical license for Dr. {selectedDoctor?.firstName}{" "}
              {selectedDoctor?.lastName}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="verify-notes">Verification Notes (Optional)</Label>
              <Textarea
                id="verify-notes"
                placeholder="Add any notes about the verification process..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={4}
                data-testid="textarea-verify-notes"
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowVerifyDialog(false);
                setNotes("");
              }}
              data-testid="button-cancel-verify"
            >
              Cancel
            </Button>
            <Button
              onClick={handleVerify}
              disabled={verifyMutation.isPending}
              data-testid="button-confirm-verify"
            >
              {verifyMutation.isPending ? "Verifying..." : "Verify License"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Reject Dialog */}
      <Dialog open={showRejectDialog} onOpenChange={setShowRejectDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject License Verification</DialogTitle>
            <DialogDescription>
              Please provide a reason for rejecting the license verification for Dr.{" "}
              {selectedDoctor?.firstName} {selectedDoctor?.lastName}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="reject-reason">Rejection Reason *</Label>
              <Textarea
                id="reject-reason"
                placeholder="Explain why the verification is being rejected..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={4}
                data-testid="textarea-reject-reason"
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowRejectDialog(false);
                setNotes("");
              }}
              data-testid="button-cancel-reject"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleReject}
              disabled={rejectMutation.isPending || !notes}
              data-testid="button-confirm-reject"
            >
              {rejectMutation.isPending ? "Rejecting..." : "Reject Verification"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
