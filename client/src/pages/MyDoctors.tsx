import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useAuth } from "@/hooks/useAuth";
import { Link } from "wouter";
import {
  MapPin,
  Building2,
  Stethoscope,
  Linkedin,
  Loader2,
  UserMinus,
  Search,
  MessageSquare,
  Users,
  Calendar,
  FileText,
  Plus,
  CheckCircle,
  XCircle,
  AlertCircle,
  Clock,
  User,
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

interface Doctor {
  id: string;
  first_name?: string;
  last_name?: string;
  firstName?: string;
  lastName?: string;
  email: string;
  specialty?: string;
  years_of_experience?: number;
  hospital_name?: string;
  hospital?: string;
  location_city?: string;
  location_state?: string;
  linkedin_url?: string;
  availability_status?: string;
  bio?: string;
}

interface Connection {
  connection_id: number;
  connection_type: string;
  connected_at?: string;
  doctor: Doctor;
}

interface MyDoctorsResponse {
  connections: Connection[];
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

export default function MyDoctors() {
  const { user } = useAuth();
  const { toast } = useToast();
  
  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [locationCity, setLocationCity] = useState("");
  const [locationState, setLocationState] = useState("");

  // Consultation request state
  const [selectedDoctor, setSelectedDoctor] = useState("");
  const [consultationReason, setConsultationReason] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [urgency, setUrgency] = useState("routine");
  const [dialogOpen, setDialogOpen] = useState(false);

  // My connected doctors query
  const { data: myDoctorsData, isLoading: isLoadingMyDoctors } = useQuery<MyDoctorsResponse>({
    queryKey: ["/api/doctors/my-doctors/list"],
  });

  // Search doctors query
  const { data: searchResults, isLoading: isLoadingSearch } = useQuery<Doctor[]>({
    queryKey: ["/api/doctors/search", { searchQuery, specialty, locationCity, locationState }],
    enabled: true,
  });

  // Consultation requests query
  const { data: consultationRequests, isLoading: isLoadingRequests } = useQuery<{ requests: ConsultationRequest[] }>({
    queryKey: ["/api/v1/consultations/patient/my-requests"],
    enabled: user?.role === "patient",
  });

  const connectedDoctors = myDoctorsData?.connections.map(c => c.doctor) || [];

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

  const disconnectMutation = useMutation({
    mutationFn: async (doctorId: string) => {
      return await apiRequest("DELETE", `/api/doctors/disconnect/${doctorId}`, {});
    },
    onSuccess: () => {
      toast({
        title: "Disconnected",
        description: "You have been disconnected from this doctor.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/doctors/my-doctors/list"] });
    },
    onError: (error: any) => {
      toast({
        title: "Disconnection Failed",
        description: error.message || "Unable to disconnect from this doctor.",
        variant: "destructive",
      });
    },
  });

  const getInitials = (firstName?: string, lastName?: string) => {
    const first = firstName?.charAt(0) || "";
    const last = lastName?.charAt(0) || "";
    return (first + last).toUpperCase() || "DR";
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return "";
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const handleSubmitRequest = (e: React.FormEvent) => {
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
    const firstName = doctor?.first_name || doctor?.firstName || "";
    const lastName = doctor?.last_name || doctor?.lastName || "";
    return doctor ? `Dr. ${firstName} ${lastName}` : "Unknown Doctor";
  };

  const renderDoctorCard = (doctor: Doctor, connectionData?: { connectionId: number; connectedAt?: string; connectionType?: string }) => {
    const firstName = doctor.first_name || doctor.firstName || "";
    const lastName = doctor.last_name || doctor.lastName || "";
    const hospital = doctor.hospital_name || doctor.hospital || "";

    return (
      <Card
        key={doctor.id}
        data-testid={connectionData ? `card-connection-${connectionData.connectionId}` : `card-doctor-${doctor.id}`}
        className="hover-elevate"
      >
        <CardHeader>
          <div className="flex items-start gap-4">
            <Avatar className="h-12 w-12">
              <AvatarFallback>
                {getInitials(firstName, lastName)}
              </AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-lg">
                Dr. {firstName} {lastName}
              </CardTitle>
              {doctor.specialty && (
                <CardDescription className="flex items-center gap-1 mt-1">
                  <Stethoscope className="h-3 w-3" />
                  {doctor.specialty}
                </CardDescription>
              )}
              {connectionData?.connectionType && (
                <Badge variant="secondary" className="mt-2">
                  {connectionData.connectionType.replace("_", " ")}
                </Badge>
              )}
              {!connectionData && doctor.availability_status === "available" && (
                <Badge variant="default" className="bg-green-500 mt-2">
                  Available
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {hospital && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Building2 className="h-4 w-4" />
              {hospital}
            </div>
          )}
          {(doctor.location_city || doctor.location_state) && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <MapPin className="h-4 w-4" />
              {doctor.location_city}
              {doctor.location_city && doctor.location_state && ", "}
              {doctor.location_state}
            </div>
          )}
          {doctor.years_of_experience && !connectionData && (
            <p className="text-sm text-muted-foreground">
              {doctor.years_of_experience} years of experience
            </p>
          )}
          {doctor.bio && !connectionData && (
            <p className="text-sm line-clamp-2">{doctor.bio}</p>
          )}
          {connectionData?.connectedAt && (
            <p className="text-xs text-muted-foreground">
              Connected since {formatDate(connectionData.connectedAt)}
            </p>
          )}
          <div className="flex gap-2 pt-2">
            <Button
              asChild
              size="sm"
              variant={connectionData ? "outline" : "default"}
              data-testid={`button-view-${doctor.id}`}
            >
              <Link href={`/doctor/${doctor.id}`}>View Profile</Link>
            </Button>
            {connectionData && (
              <Button
                size="sm"
                variant="outline"
                data-testid={`button-message-${doctor.id}`}
              >
                <MessageSquare className="h-4 w-4" />
              </Button>
            )}
            {doctor.linkedin_url && (
              <Button
                asChild
                variant="outline"
                size="sm"
                data-testid={`button-linkedin-${doctor.id}`}
              >
                <a
                  href={doctor.linkedin_url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Linkedin className="h-4 w-4" />
                </a>
              </Button>
            )}
            {connectionData && (
              <Button
                size="sm"
                variant="destructive"
                onClick={() => disconnectMutation.mutate(doctor.id)}
                disabled={disconnectMutation.isPending}
                data-testid={`button-disconnect-${doctor.id}`}
              >
                {disconnectMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <UserMinus className="h-4 w-4" />
                )}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">My Doctors</h1>
          <p className="text-muted-foreground">
            Manage your healthcare providers, discover new ones, and request consultations
          </p>
        </div>

        <Tabs defaultValue="connected" className="space-y-6">
          <TabsList className="grid w-full max-w-2xl grid-cols-3">
            <TabsTrigger value="connected" data-testid="tab-connected-doctors">
              <Users className="h-4 w-4 mr-2" />
              Connected
            </TabsTrigger>
            <TabsTrigger value="search" data-testid="tab-find-doctors">
              <Search className="h-4 w-4 mr-2" />
              Find Doctors
            </TabsTrigger>
            <TabsTrigger value="requests" data-testid="tab-consultation-requests">
              <Calendar className="h-4 w-4 mr-2" />
              Requests
            </TabsTrigger>
          </TabsList>

          {/* Connected Doctors Tab */}
          <TabsContent value="connected" className="space-y-4">
            {isLoadingMyDoctors && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            )}

            {!isLoadingMyDoctors && myDoctorsData && myDoctorsData.connections.length === 0 && (
              <Card>
                <CardContent className="py-12 text-center space-y-4">
                  <Users className="h-12 w-12 mx-auto text-muted-foreground" />
                  <div>
                    <p className="font-medium">No connected doctors yet</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Search for healthcare providers in the "Find Doctors" tab
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {!isLoadingMyDoctors && myDoctorsData && myDoctorsData.connections.length > 0 && (
              <div>
                <p className="text-sm text-muted-foreground mb-4">
                  You're connected with {myDoctorsData.connections.length} healthcare provider{myDoctorsData.connections.length !== 1 ? 's' : ''}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {myDoctorsData.connections.map((connection) => 
                    renderDoctorCard(connection.doctor, {
                      connectionId: connection.connection_id,
                      connectedAt: connection.connected_at,
                      connectionType: connection.connection_type,
                    })
                  )}
                </div>
              </div>
            )}
          </TabsContent>

          {/* Find Doctors Tab */}
          <TabsContent value="search" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Search for Healthcare Providers</CardTitle>
                <CardDescription>
                  Find doctors by name, specialty, or location
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Search by Name or Email</label>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        data-testid="input-doctor-search"
                        placeholder="Dr. Smith, john@example.com..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-9"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Specialty</label>
                    <div className="relative">
                      <Stethoscope className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        data-testid="input-specialty"
                        placeholder="Cardiology, Oncology..."
                        value={specialty}
                        onChange={(e) => setSpecialty(e.target.value)}
                        className="pl-9"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">City</label>
                    <div className="relative">
                      <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        data-testid="input-city"
                        placeholder="New York, Boston..."
                        value={locationCity}
                        onChange={(e) => setLocationCity(e.target.value)}
                        className="pl-9"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">State</label>
                    <Input
                      data-testid="input-state"
                      placeholder="NY, CA, MA..."
                      value={locationState}
                      onChange={(e) => setLocationState(e.target.value)}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="space-y-4">
              <h2 className="text-xl font-semibold">
                {isLoadingSearch ? "Searching..." : `Found ${searchResults?.length || 0} doctors`}
              </h2>

              {isLoadingSearch && (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              )}

              {!isLoadingSearch && searchResults && searchResults.length === 0 && (
                <Card>
                  <CardContent className="py-12 text-center">
                    <Search className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="font-medium">No doctors found</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Try adjusting your search filters
                    </p>
                  </CardContent>
                </Card>
              )}

              {!isLoadingSearch && searchResults && searchResults.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {searchResults.map((doctor) => renderDoctorCard(doctor))}
                </div>
              )}
            </div>
          </TabsContent>

          {/* Consultation Requests Tab */}
          <TabsContent value="requests" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold">Consultation Requests</h2>
                <p className="text-sm text-muted-foreground">
                  Request and manage consultations with your doctors
                </p>
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
                  <form onSubmit={handleSubmitRequest} className="space-y-4">
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
                            connectedDoctors.map((doctor) => {
                              const firstName = doctor.first_name || doctor.firstName || "";
                              const lastName = doctor.last_name || doctor.lastName || "";
                              return (
                                <SelectItem key={doctor.id} value={doctor.id}>
                                  Dr. {firstName} {lastName} - {doctor.specialty}
                                </SelectItem>
                              );
                            })
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

            {isLoadingRequests && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            )}

            {!isLoadingRequests && consultationRequests?.requests.length === 0 && (
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
            )}

            {!isLoadingRequests && consultationRequests && consultationRequests.requests.length > 0 && (
              <div className="grid gap-4">
                {consultationRequests.requests.map((request) => (
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

                      {request.scheduled_for && request.status === "approved" && (
                        <div className="p-4 bg-muted rounded-md">
                          <div className="flex items-center gap-2 text-sm font-medium mb-1">
                            <CheckCircle className="h-4 w-4 text-primary" />
                            Scheduled For:
                          </div>
                          <p className="text-sm font-semibold" data-testid={`text-scheduled-${request.id}`}>
                            {new Date(request.scheduled_for).toLocaleString()}
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
