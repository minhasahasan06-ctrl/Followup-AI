import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
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
} from "lucide-react";

interface Doctor {
  id: string;
  first_name?: string;
  last_name?: string;
  email: string;
  specialty?: string;
  hospital_name?: string;
  location_city?: string;
  linkedin_url?: string;
  availability_status?: string;
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

export default function MyDoctors() {
  const { toast } = useToast();

  const { data, isLoading } = useQuery<MyDoctorsResponse>({
    queryKey: ["/api/doctors/my-doctors/list"],
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

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">My Doctors</h1>
            <p className="text-muted-foreground">
              Healthcare providers you're connected with
            </p>
          </div>
          <Button asChild data-testid="button-find-doctors">
            <Link href="/doctor-search">
              <Search className="mr-2 h-4 w-4" />
              Find More Doctors
            </Link>
          </Button>
        </div>

        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}

        {!isLoading && data && data.connections.length === 0 && (
          <Card>
            <CardContent className="py-12 text-center space-y-4">
              <p className="text-muted-foreground">
                You haven't connected with any doctors yet.
              </p>
              <Button asChild>
                <Link href="/doctor-search">
                  <Search className="mr-2 h-4 w-4" />
                  Find Doctors
                </Link>
              </Button>
            </CardContent>
          </Card>
        )}

        {!isLoading && data && data.connections.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.connections.map((connection) => {
              const doctor = connection.doctor;
              return (
                <Card
                  key={connection.connection_id}
                  data-testid={`card-connection-${connection.connection_id}`}
                  className="hover-elevate"
                >
                  <CardHeader>
                    <div className="flex items-start gap-4">
                      <Avatar className="h-12 w-12">
                        <AvatarFallback>
                          {getInitials(doctor.first_name, doctor.last_name)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1 min-w-0">
                        <CardTitle className="text-lg">
                          Dr. {doctor.first_name} {doctor.last_name}
                        </CardTitle>
                        {doctor.specialty && (
                          <CardDescription className="flex items-center gap-1 mt-1">
                            <Stethoscope className="h-3 w-3" />
                            {doctor.specialty}
                          </CardDescription>
                        )}
                        <Badge variant="secondary" className="mt-2">
                          {connection.connection_type.replace("_", " ")}
                        </Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {doctor.hospital_name && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Building2 className="h-4 w-4" />
                        {doctor.hospital_name}
                      </div>
                    )}
                    {doctor.location_city && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <MapPin className="h-4 w-4" />
                        {doctor.location_city}
                      </div>
                    )}
                    {connection.connected_at && (
                      <p className="text-xs text-muted-foreground">
                        Connected since {formatDate(connection.connected_at)}
                      </p>
                    )}
                    <div className="flex gap-2 pt-2">
                      <Button
                        asChild
                        size="sm"
                        variant="outline"
                        data-testid={`button-view-${doctor.id}`}
                      >
                        <Link href={`/doctor/${doctor.id}`}>View Profile</Link>
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        data-testid={`button-message-${doctor.id}`}
                      >
                        <MessageSquare className="h-4 w-4" />
                      </Button>
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
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
