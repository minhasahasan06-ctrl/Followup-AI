import { useQuery, useMutation } from "@tanstack/react-query";
import { useParams, useLocation } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  MapPin,
  Building2,
  Stethoscope,
  Linkedin,
  Award,
  Loader2,
  UserPlus,
  MessageSquare,
  ArrowLeft,
  UserCheck,
} from "lucide-react";
import { Link } from "wouter";
import { useAuth } from "@/hooks/useAuth";
import api from "@/lib/api";

interface DoctorProfileData {
  id: string;
  first_name?: string;
  last_name?: string;
  email: string;
  specialty?: string;
  years_of_experience?: number;
  medical_license_number?: string;
  hospital_name?: string;
  location_city?: string;
  location_state?: string;
  location_country?: string;
  linkedin_url?: string;
  availability_status?: string;
  bio?: string;
  created_at?: string;
}

export default function DoctorProfile() {
  const { doctorId } = useParams();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const { user } = useAuth();
  const isPatient = user?.role === "patient";

  const { data: doctor, isLoading } = useQuery<DoctorProfileData>({
    queryKey: [`/api/doctors/${doctorId}`],
    enabled: !!doctorId,
  });

  const connectMutation = useMutation({
    mutationFn: async () => {
      return await apiRequest("POST", "/api/doctors/connect", {
        doctor_id: doctorId,
        connection_type: "specialist",
      });
    },
    onSuccess: () => {
      toast({
        title: "Connected!",
        description: "You have successfully connected with this doctor.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/doctors/my-doctors/list"] });
      navigate("/my-doctors");
    },
    onError: (error: any) => {
      toast({
        title: "Connection Failed",
        description: error.message || "Unable to connect with this doctor.",
        variant: "destructive",
      });
    },
  });

  const assignDoctorMutation = useMutation({
    mutationFn: async () => {
      const response = await api.post("/api/patient/assign-doctor", {
        doctor_id: doctorId,
      });
      return response.data;
    },
    onSuccess: () => {
      toast({
        title: "Doctor Assigned",
        description: "This doctor has been assigned as your primary doctor.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/patient/profile/extended"] });
    },
    onError: (error: any) => {
      toast({
        title: "Assignment Failed",
        description: error.response?.data?.detail || "Unable to assign this doctor.",
        variant: "destructive",
      });
    },
  });

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!doctor) {
    return (
      <div className="h-full overflow-auto p-6">
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">Doctor not found.</p>
            <Button asChild className="mt-4">
              <Link href="/doctor-search">Back to Search</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const getInitials = (firstName?: string, lastName?: string) => {
    const first = firstName?.charAt(0) || "";
    const last = lastName?.charAt(0) || "";
    return (first + last).toUpperCase() || "DR";
  };

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <Button
          variant="ghost"
          asChild
          data-testid="button-back"
        >
          <Link href="/doctor-search">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Search
          </Link>
        </Button>

        <Card>
          <CardHeader>
            <div className="flex items-start gap-6">
              <Avatar className="h-20 w-20">
                <AvatarFallback className="text-2xl">
                  {getInitials(doctor.first_name, doctor.last_name)}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1">
                <CardTitle className="text-2xl" data-testid="text-doctor-name">
                  Dr. {doctor.first_name} {doctor.last_name}
                </CardTitle>
                {doctor.specialty && (
                  <CardDescription className="flex items-center gap-2 mt-2 text-base">
                    <Stethoscope className="h-4 w-4" />
                    {doctor.specialty}
                  </CardDescription>
                )}
                {doctor.availability_status === "available" && (
                  <Badge variant="default" className="bg-green-500 mt-2">
                    Available
                  </Badge>
                )}
              </div>
              <div className="flex gap-2">
                {doctor.linkedin_url && (
                  <Button
                    asChild
                    variant="outline"
                    size="icon"
                    data-testid="button-linkedin"
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
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              {doctor.hospital_name && (
                <div className="flex items-center gap-3">
                  <Building2 className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">Hospital</p>
                    <p className="font-medium">{doctor.hospital_name}</p>
                  </div>
                </div>
              )}

              {(doctor.location_city || doctor.location_state) && (
                <div className="flex items-center gap-3">
                  <MapPin className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">Location</p>
                    <p className="font-medium">
                      {doctor.location_city}
                      {doctor.location_city && doctor.location_state && ", "}
                      {doctor.location_state}
                      {doctor.location_country && `, ${doctor.location_country}`}
                    </p>
                  </div>
                </div>
              )}

              {doctor.years_of_experience !== undefined && (
                <div className="flex items-center gap-3">
                  <Award className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">Experience</p>
                    <p className="font-medium">
                      {doctor.years_of_experience} years
                    </p>
                  </div>
                </div>
              )}

              {doctor.medical_license_number && (
                <div className="flex items-center gap-3">
                  <Stethoscope className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">
                      Medical License
                    </p>
                    <p className="font-medium">{doctor.medical_license_number}</p>
                  </div>
                </div>
              )}
            </div>

            {doctor.bio && (
              <>
                <Separator />
                <div>
                  <h3 className="font-semibold mb-2">About</h3>
                  <p className="text-muted-foreground">{doctor.bio}</p>
                </div>
              </>
            )}

            <Separator />

            <div className="flex gap-3 flex-wrap">
              {isPatient && (
                <Button
                  onClick={() => assignDoctorMutation.mutate()}
                  disabled={assignDoctorMutation.isPending}
                  data-testid="button-assign-primary-doctor"
                  className="flex-1"
                  variant="default"
                >
                  {assignDoctorMutation.isPending ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <UserCheck className="mr-2 h-4 w-4" />
                  )}
                  Assign as my primary doctor
                </Button>
              )}
              <Button
                onClick={() => connectMutation.mutate()}
                disabled={connectMutation.isPending}
                data-testid="button-connect"
                className="flex-1"
                variant={isPatient ? "outline" : "default"}
              >
                {connectMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <UserPlus className="mr-2 h-4 w-4" />
                )}
                Connect with Doctor
              </Button>
              <Button
                variant="outline"
                data-testid="button-message"
                className="flex-1"
              >
                <MessageSquare className="mr-2 h-4 w-4" />
                Request Consultation
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
