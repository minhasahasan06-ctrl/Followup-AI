import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  Users,
} from "lucide-react";

interface Doctor {
  id: string;
  first_name?: string;
  last_name?: string;
  email: string;
  specialty?: string;
  years_of_experience?: number;
  hospital_name?: string;
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

export default function MyDoctors() {
  const { toast } = useToast();
  
  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [locationCity, setLocationCity] = useState("");
  const [locationState, setLocationState] = useState("");

  // My connected doctors query
  const { data: myDoctorsData, isLoading: isLoadingMyDoctors } = useQuery<MyDoctorsResponse>({
    queryKey: ["/api/doctors/my-doctors/list"],
  });

  // Search doctors query
  const { data: searchResults, isLoading: isLoadingSearch } = useQuery<Doctor[]>({
    queryKey: ["/api/doctors/search", { searchQuery, specialty, locationCity, locationState }],
    enabled: true,
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

  const renderDoctorCard = (doctor: Doctor, connectionData?: { connectionId: number; connectedAt?: string; connectionType?: string }) => (
    <Card
      key={doctor.id}
      data-testid={connectionData ? `card-connection-${connectionData.connectionId}` : `card-doctor-${doctor.id}`}
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
        {doctor.hospital_name && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Building2 className="h-4 w-4" />
            {doctor.hospital_name}
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

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">My Doctors</h1>
          <p className="text-muted-foreground">
            Manage your healthcare providers and discover new ones
          </p>
        </div>

        <Tabs defaultValue="connected" className="space-y-6">
          <TabsList className="grid w-full max-w-md grid-cols-2">
            <TabsTrigger value="connected" data-testid="tab-connected-doctors">
              <Users className="h-4 w-4 mr-2" />
              Connected Doctors
            </TabsTrigger>
            <TabsTrigger value="search" data-testid="tab-find-doctors">
              <Search className="h-4 w-4 mr-2" />
              Find Doctors
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
        </Tabs>
      </div>
    </div>
  );
}
