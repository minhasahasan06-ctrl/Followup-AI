import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { UserPlus, Shield, Phone, Mail, X, Stethoscope } from "lucide-react";
import { Link } from "wouter";

interface AssignedDoctor {
  id: string;
  first_name: string;
  last_name: string;
  email: string;
  organization?: string;
  specialty?: string;
  license_verified: boolean;
}

interface PatientProfileExtended {
  user_id: string;
  assigned_doctor_id?: string;
  assigned_doctor?: AssignedDoctor;
  emergency_contacts: any[];
  medications: any[];
  allergies: string[];
}

export function AssignedDoctorCard() {
  const { toast } = useToast();

  const { data: profile, isLoading } = useQuery<PatientProfileExtended>({
    queryKey: ["/api/patient/profile/extended"],
  });

  const unassignMutation = useMutation({
    mutationFn: async () => {
      return await apiRequest("DELETE", "/api/patient/unassign-doctor");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/patient/profile/extended"] });
      toast({
        title: "Doctor Unassigned",
        description: "You can assign a new doctor at any time.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to unassign doctor.",
        variant: "destructive",
      });
    },
  });

  const getInitials = (firstName?: string, lastName?: string) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  if (isLoading) {
    return (
      <Card data-testid="assigned-doctor-card-loading">
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-muted rounded w-1/3" />
            <div className="h-12 bg-muted rounded" />
          </div>
        </CardContent>
      </Card>
    );
  }

  const doctor = profile?.assigned_doctor;

  return (
    <Card data-testid="assigned-doctor-card">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Stethoscope className="h-5 w-5 text-primary" />
              Your Doctor
            </CardTitle>
            <CardDescription>
              {doctor ? "Your assigned primary care doctor" : "Assign a doctor for better care coordination"}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {doctor ? (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <Avatar className="h-16 w-16">
                <AvatarFallback className="bg-primary/10 text-primary text-lg">
                  {getInitials(doctor.first_name, doctor.last_name)}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1">
                <h4 className="font-semibold text-lg" data-testid="text-doctor-name">
                  Dr. {doctor.first_name} {doctor.last_name}
                </h4>
                {doctor.specialty && (
                  <p className="text-sm text-muted-foreground">{doctor.specialty}</p>
                )}
                {doctor.organization && (
                  <p className="text-sm text-muted-foreground">{doctor.organization}</p>
                )}
                <div className="flex items-center gap-2 mt-2">
                  {doctor.license_verified && (
                    <Badge variant="secondary">
                      <Shield className="h-3 w-3 mr-1" />
                      Verified
                    </Badge>
                  )}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 pt-4 border-t">
              <Button
                variant="outline"
                size="sm"
                onClick={() => unassignMutation.mutate()}
                disabled={unassignMutation.isPending}
                data-testid="button-unassign-doctor"
              >
                <X className="h-4 w-4 mr-2" />
                Remove
              </Button>
              <Link href="/consultation-requests">
                <Button size="sm" data-testid="button-request-consultation">
                  Request Consultation
                </Button>
              </Link>
            </div>
          </div>
        ) : (
          <div className="text-center py-6">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted mx-auto mb-4">
              <UserPlus className="h-8 w-8 text-muted-foreground" />
            </div>
            <p className="text-muted-foreground mb-4">
              No doctor assigned yet. Find and assign a doctor for personalized care.
            </p>
            <Link href="/doctor-search">
              <Button data-testid="button-find-doctor">
                <UserPlus className="h-4 w-4 mr-2" />
                Find a Doctor
              </Button>
            </Link>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default AssignedDoctorCard;
