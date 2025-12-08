import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Calendar, Clock, User, FileText, Plus, CheckCircle, XCircle, AlertCircle, Video } from "lucide-react";
import { Link } from "wouter";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useAuth } from "@/hooks/useAuth";
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

interface Doctor {
  id: string;
  firstName: string;
  lastName: string;
  specialty: string;
  hospital: string;
}

interface ConsultationRequest {
  id: number;
  doctor_id: string;
  consultation_reason: string;
  symptoms: string | null;
  urgency: string;
  mode: string;
  status: string;
  created_at: string;
  scheduled_for: string | null;
}

export default function ConsultationRequests() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [selectedDoctor, setSelectedDoctor] = useState("");
  const [consultationReason, setConsultationReason] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [urgency, setUrgency] = useState("routine");
  const [dialogOpen, setDialogOpen] = useState(false);

  const { data: connectedDoctorsResponse } = useQuery<{ connections: Array<{ doctor: Doctor }> }>({
    queryKey: ["/api/doctors/my-doctors/list"],
    enabled: user?.role === "patient",
  });

  const connectedDoctors = connectedDoctorsResponse?.connections.map(c => c.doctor) || [];

  const { data: consultationRequests } = useQuery<{ requests: ConsultationRequest[] }>({
    queryKey: ["/api/v1/consultations/patient/my-requests"],
    enabled: user?.role === "patient",
  });

  const createRequestMutation = useMutation({
    mutationFn: async (data: {
      doctor_id: string;
      consultation_reason: string;
      symptoms?: string;
      urgency?: string;
    }) => {
      return await apiRequest("POST", "/api/v1/consultations/patient/request", data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/consultations/patient/my-requests"] });
      toast({
        title: "Request submitted",
        description: "Your consultation request has been sent to the doctor",
      });
      setDialogOpen(false);
      setSelectedDoctor("");
      setConsultationReason("");
      setSymptoms("");
      setUrgency("routine");
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to submit consultation request",
        variant: "destructive",
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDoctor || !consultationReason) {
      toast({
        title: "Missing information",
        description: "Please select a doctor and provide a reason",
        variant: "destructive",
      });
      return;
    }

    createRequestMutation.mutate({
      doctor_id: selectedDoctor,
      consultation_reason: consultationReason,
      symptoms: symptoms || undefined,
      urgency: urgency || "routine",
    });
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "requested":
        return (
          <Badge variant="secondary" data-testid="badge-status-requested">
            <AlertCircle className="h-3 w-3 mr-1" />
            Requested
          </Badge>
        );
      case "approved":
        return (
          <Badge variant="default" data-testid="badge-status-approved">
            <CheckCircle className="h-3 w-3 mr-1" />
            Approved
          </Badge>
        );
      case "declined":
        return (
          <Badge variant="destructive" data-testid="badge-status-declined">
            <XCircle className="h-3 w-3 mr-1" />
            Declined
          </Badge>
        );
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getDoctorName = (doctorId: string) => {
    const doctor = connectedDoctors.find((d) => d.id === doctorId);
    return doctor ? `Dr. ${doctor.firstName} ${doctor.lastName}` : "Unknown Doctor";
  };

  if (user?.role !== "patient") {
    return (
      <div className="flex items-center justify-center h-screen">
        <Card>
          <CardHeader>
            <CardTitle>Access Denied</CardTitle>
            <CardDescription>Only patients can view consultation requests</CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-semibold mb-2">Consultation Requests</h1>
          <p className="text-muted-foreground">Request and manage consultations with your doctors</p>
        </div>
        
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-new-request">
              <Plus className="h-4 w-4 mr-2" />
              New Request
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Request Consultation</DialogTitle>
              <DialogDescription>
                Submit a consultation request to one of your connected doctors
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="doctor">Select Doctor *</Label>
                <Select value={selectedDoctor} onValueChange={setSelectedDoctor}>
                  <SelectTrigger data-testid="select-doctor">
                    <SelectValue placeholder="Choose a doctor" />
                  </SelectTrigger>
                  <SelectContent>
                    {connectedDoctors.length === 0 ? (
                      <div className="p-2 text-sm text-muted-foreground">
                        No connected doctors. Please connect with a doctor first.
                      </div>
                    ) : (
                      connectedDoctors.map((doctor) => (
                        <SelectItem key={doctor.id} value={doctor.id}>
                          Dr. {doctor.firstName} {doctor.lastName} - {doctor.specialty}
                        </SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="consultation-reason">Reason for Consultation *</Label>
                <Textarea
                  id="consultation-reason"
                  placeholder="Why do you need to see the doctor?"
                  value={consultationReason}
                  onChange={(e) => setConsultationReason(e.target.value)}
                  data-testid="input-consultation-reason"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="symptoms">Symptoms</Label>
                <Textarea
                  id="symptoms"
                  placeholder="Describe your symptoms..."
                  value={symptoms}
                  onChange={(e) => setSymptoms(e.target.value)}
                  data-testid="input-symptoms"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="urgency">Urgency</Label>
                <Select value={urgency} onValueChange={setUrgency}>
                  <SelectTrigger data-testid="select-urgency">
                    <SelectValue placeholder="Select urgency" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="routine">Routine</SelectItem>
                    <SelectItem value="urgent">Urgent</SelectItem>
                    <SelectItem value="emergency">Emergency</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex gap-2">
                <Button
                  type="submit"
                  disabled={createRequestMutation.isPending}
                  data-testid="button-submit-request"
                >
                  {createRequestMutation.isPending ? "Submitting..." : "Submit Request"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setDialogOpen(false)}
                  data-testid="button-cancel"
                >
                  Cancel
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {consultationRequests?.requests.length === 0 ? (
        <Card>
          <CardContent className="p-12 text-center">
            <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-lg font-semibold mb-2">No Consultation Requests</h3>
            <p className="text-muted-foreground mb-4">
              You haven't submitted any consultation requests yet
            </p>
            <Button onClick={() => setDialogOpen(true)} data-testid="button-create-first">
              <Plus className="h-4 w-4 mr-2" />
              Create Your First Request
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {consultationRequests?.requests.map((request) => (
            <Card key={request.id} data-testid={`card-request-${request.id}`}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="flex items-center gap-2">
                      <User className="h-5 w-5" />
                      {getDoctorName(request.doctor_id)}
                    </CardTitle>
                    <CardDescription>Request #{request.id}</CardDescription>
                  </div>
                  {getStatusBadge(request.status)}
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <FileText className="h-4 w-4" />
                      <span className="font-medium">Reason:</span>
                    </div>
                    <p className="text-sm" data-testid={`text-consultation-reason-${request.id}`}>
                      {request.consultation_reason}
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Calendar className="h-4 w-4" />
                      <span className="font-medium">Requested:</span>
                    </div>
                    <p className="text-sm">
                      {new Date(request.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>

                {request.symptoms && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <FileText className="h-4 w-4" />
                      <span className="font-medium">Symptoms:</span>
                    </div>
                    <p className="text-sm text-muted-foreground">{request.symptoms}</p>
                  </div>
                )}

                <div className="flex items-center gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <AlertCircle className="h-4 w-4" />
                      <span className="font-medium">Urgency:</span>
                    </div>
                    <Badge variant="secondary">{request.urgency}</Badge>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock className="h-4 w-4" />
                      <span className="font-medium">Mode:</span>
                    </div>
                    <Badge variant="outline">{request.mode}</Badge>
                  </div>
                </div>

                {request.status === "approved" && (
                  <div className="p-4 bg-muted rounded-md">
                    <div className="flex items-center justify-between">
                      <div>
                        {request.scheduled_for && (
                          <>
                            <div className="flex items-center gap-2 text-sm font-medium mb-1">
                              <CheckCircle className="h-4 w-4 text-primary" />
                              Scheduled For:
                            </div>
                            <p className="text-sm font-semibold" data-testid={`text-scheduled-${request.id}`}>
                              {new Date(request.scheduled_for).toLocaleString()}
                            </p>
                          </>
                        )}
                      </div>
                      <Link href={`/video-consultation/${request.id}`}>
                        <Button data-testid={`button-video-call-${request.id}`}>
                          <Video className="h-4 w-4 mr-2" />
                          Join Video Call
                        </Button>
                      </Link>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
