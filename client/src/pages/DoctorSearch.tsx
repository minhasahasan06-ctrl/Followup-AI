import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Search, MapPin, Building2, Stethoscope, Linkedin, Loader2 } from "lucide-react";
import { Link } from "wouter";

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

export default function DoctorSearch() {
  const [searchQuery, setSearchQuery] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [locationCity, setLocationCity] = useState("");
  const [locationState, setLocationState] = useState("");

  const { data: doctors, isLoading } = useQuery<Doctor[]>({
    queryKey: ["/api/doctors/search", { searchQuery, specialty, locationCity, locationState }],
    enabled: true,
  });

  const handleSearch = () => {
    // Query will automatically refetch due to queryKey changes
  };

  const getInitials = (firstName?: string, lastName?: string) => {
    const first = firstName?.charAt(0) || "";
    const last = lastName?.charAt(0) || "";
    return (first + last).toUpperCase() || "DR";
  };

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Find Your Doctor</h1>
          <p className="text-muted-foreground">
            Search for healthcare providers by name, specialty, or location
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Search Filters</CardTitle>
            <CardDescription>
              Refine your search to find the right healthcare provider
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

            <Button
              data-testid="button-search"
              onClick={handleSearch}
              className="w-full md:w-auto"
            >
              <Search className="mr-2 h-4 w-4" />
              Search Doctors
            </Button>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <h2 className="text-xl font-semibold">
            {isLoading ? "Searching..." : `Found ${doctors?.length || 0} doctors`}
          </h2>

          {isLoading && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          )}

          {!isLoading && doctors && doctors.length === 0 && (
            <Card>
              <CardContent className="py-12 text-center">
                <p className="text-muted-foreground">
                  No doctors found. Try adjusting your search filters.
                </p>
              </CardContent>
            </Card>
          )}

          {!isLoading && doctors && doctors.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {doctors.map((doctor) => (
                <Card
                  key={doctor.id}
                  data-testid={`card-doctor-${doctor.id}`}
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
                      </div>
                      {doctor.availability_status === "available" && (
                        <Badge variant="default" className="bg-green-500">
                          Available
                        </Badge>
                      )}
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
                    {doctor.years_of_experience && (
                      <p className="text-sm text-muted-foreground">
                        {doctor.years_of_experience} years of experience
                      </p>
                    )}
                    {doctor.bio && (
                      <p className="text-sm line-clamp-2">{doctor.bio}</p>
                    )}
                    <div className="flex gap-2 pt-2">
                      <Button
                        asChild
                        size="sm"
                        data-testid={`button-view-${doctor.id}`}
                      >
                        <Link href={`/doctor/${doctor.id}`}>View Profile</Link>
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
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
