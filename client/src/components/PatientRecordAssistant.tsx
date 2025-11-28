import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Search, 
  User, 
  Phone, 
  Mail, 
  Calendar, 
  AlertCircle,
  Heart,
  Droplets,
  Clock,
  ChevronRight,
  Loader2,
  Users,
  FileText,
  Stethoscope
} from "lucide-react";
import { format, parseISO, differenceInYears } from "date-fns";

interface PatientResult {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
  phoneNumber: string;
  followupPatientId?: string;
  dateOfBirth?: string;
  bloodType?: string;
  allergies?: string[];
  medicalConditions?: string[];
  matchScore: number;
}

interface PatientRecord {
  patient: {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    phoneNumber: string;
  };
  profile: {
    followupPatientId?: string;
    dateOfBirth?: string;
    bloodType?: string;
    allergies?: string[];
    medicalConditions?: string[];
    emergencyContact?: string;
    emergencyPhone?: string;
  } | null;
  recentAppointments: Array<{
    id: string;
    date: string;
    startTime: string;
    endTime: string;
    reason: string;
    status: string;
    type: string;
  }>;
  upcomingAppointments: Array<{
    id: string;
    date: string;
    startTime: string;
    endTime: string;
    reason: string;
    status: string;
    type: string;
  }>;
}

interface PatientRecordAssistantProps {
  onOpenLysa?: () => void;
  className?: string;
}

export function PatientRecordAssistant({ onOpenLysa, className }: PatientRecordAssistantProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  const { data: searchResults, isLoading: searchLoading, refetch: refetchSearch } = useQuery({
    queryKey: ['/api/v1/lysa/patients/search', searchQuery],
    queryFn: async () => {
      if (!searchQuery.trim()) return { results: [], count: 0 };
      const response = await fetch(`/api/v1/lysa/patients/search?query=${encodeURIComponent(searchQuery)}&limit=10`);
      if (!response.ok) throw new Error('Search failed');
      return response.json();
    },
    enabled: searchQuery.length >= 2,
    staleTime: 30000
  });

  const { data: allPatients, isLoading: patientsLoading } = useQuery({
    queryKey: ['/api/v1/lysa/patients'],
    staleTime: 60000
  });

  const { data: patientRecord, isLoading: recordLoading } = useQuery<{ success: boolean } & PatientRecord>({
    queryKey: ['/api/v1/lysa/patients', selectedPatientId],
    queryFn: async () => {
      if (!selectedPatientId) return null;
      const response = await fetch(`/api/v1/lysa/patients/${selectedPatientId}`);
      if (!response.ok) throw new Error('Failed to fetch patient record');
      return response.json();
    },
    enabled: !!selectedPatientId
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      setIsSearching(true);
      refetchSearch().finally(() => setIsSearching(false));
    }
  };

  const patients: PatientResult[] = searchQuery.length >= 2 && searchResults?.results 
    ? searchResults.results 
    : (allPatients?.patients || []);

  const calculateAge = (dateOfBirth: string): number => {
    return differenceInYears(new Date(), parseISO(dateOfBirth));
  };

  const getInitials = (firstName: string, lastName: string) => {
    return `${firstName?.charAt(0) || ''}${lastName?.charAt(0) || ''}`.toUpperCase();
  };

  return (
    <div className={`grid gap-6 lg:grid-cols-2 ${className}`}>
      <Card data-testid="card-patient-search">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Patient Search
          </CardTitle>
          <CardDescription>Search for patients by name, email, or ID</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSearch} className="flex gap-2 mb-4">
            <Input
              placeholder="Search patients..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1"
              data-testid="input-patient-search"
            />
            <Button 
              type="submit" 
              disabled={isSearching || searchLoading}
              data-testid="button-search-patients"
            >
              {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            </Button>
          </form>

          <ScrollArea className="h-[400px]">
            {patientsLoading || searchLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : patients.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Users className="h-12 w-12 mx-auto mb-4 opacity-30" />
                <p>No patients found</p>
                {searchQuery && <p className="text-sm mt-2">Try a different search term</p>}
              </div>
            ) : (
              <div className="space-y-2">
                {patients.map((patient) => (
                  <div
                    key={patient.id}
                    className={`p-3 rounded-lg border hover-elevate active-elevate-2 cursor-pointer transition-all ${
                      selectedPatientId === patient.id ? 'border-primary bg-primary/5' : ''
                    }`}
                    onClick={() => setSelectedPatientId(patient.id)}
                    data-testid={`patient-card-${patient.id}`}
                  >
                    <div className="flex items-center gap-3">
                      <Avatar className="h-10 w-10">
                        <AvatarFallback className="bg-primary/10 text-primary">
                          {getInitials(patient.firstName, patient.lastName)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-medium truncate">
                            {patient.firstName} {patient.lastName}
                          </p>
                          {patient.followupPatientId && (
                            <Badge variant="outline" className="text-xs">
                              {patient.followupPatientId}
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-sm text-muted-foreground">
                          {patient.email && (
                            <span className="flex items-center gap-1 truncate">
                              <Mail className="h-3 w-3" />
                              {patient.email}
                            </span>
                          )}
                        </div>
                      </div>
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>

      <Card data-testid="card-patient-details">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Patient Record
          </CardTitle>
          <CardDescription>
            {selectedPatientId ? 'Viewing patient details' : 'Select a patient to view details'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!selectedPatientId ? (
            <div className="text-center py-12 text-muted-foreground">
              <User className="h-16 w-16 mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium mb-2">No Patient Selected</p>
              <p className="text-sm mb-4">
                Search and select a patient to view their medical record
              </p>
              {onOpenLysa && (
                <Button variant="outline" onClick={onOpenLysa} data-testid="button-ask-lysa">
                  <Stethoscope className="h-4 w-4 mr-2" />
                  Ask Lysa About Patients
                </Button>
              )}
            </div>
          ) : recordLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : patientRecord ? (
            <Tabs defaultValue="info" className="w-full">
              <TabsList className="grid w-full grid-cols-3 mb-4">
                <TabsTrigger value="info" data-testid="tab-patient-info">Info</TabsTrigger>
                <TabsTrigger value="medical" data-testid="tab-patient-medical">Medical</TabsTrigger>
                <TabsTrigger value="appointments" data-testid="tab-patient-appointments">Appointments</TabsTrigger>
              </TabsList>

              <TabsContent value="info" className="space-y-4 mt-0">
                <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                  <Avatar className="h-16 w-16">
                    <AvatarFallback className="bg-primary/10 text-primary text-xl">
                      {getInitials(patientRecord.patient.firstName, patientRecord.patient.lastName)}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <h3 className="text-xl font-semibold">
                      {patientRecord.patient.firstName} {patientRecord.patient.lastName}
                    </h3>
                    {patientRecord.profile?.followupPatientId && (
                      <Badge variant="secondary">{patientRecord.profile.followupPatientId}</Badge>
                    )}
                  </div>
                </div>

                <div className="space-y-3">
                  {patientRecord.patient.email && (
                    <div className="flex items-center gap-3 p-3 rounded-lg border">
                      <Mail className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <p className="text-sm text-muted-foreground">Email</p>
                        <p className="font-medium">{patientRecord.patient.email}</p>
                      </div>
                    </div>
                  )}
                  {patientRecord.patient.phoneNumber && (
                    <div className="flex items-center gap-3 p-3 rounded-lg border">
                      <Phone className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <p className="text-sm text-muted-foreground">Phone</p>
                        <p className="font-medium">{patientRecord.patient.phoneNumber}</p>
                      </div>
                    </div>
                  )}
                  {patientRecord.profile?.dateOfBirth && (
                    <div className="flex items-center gap-3 p-3 rounded-lg border">
                      <Calendar className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <p className="text-sm text-muted-foreground">Date of Birth</p>
                        <p className="font-medium">
                          {format(parseISO(patientRecord.profile.dateOfBirth), "MMMM d, yyyy")}
                          <span className="text-muted-foreground ml-2">
                            ({calculateAge(patientRecord.profile.dateOfBirth)} years old)
                          </span>
                        </p>
                      </div>
                    </div>
                  )}
                  {patientRecord.profile?.emergencyContact && (
                    <div className="flex items-center gap-3 p-3 rounded-lg border">
                      <AlertCircle className="h-5 w-5 text-orange-500" />
                      <div>
                        <p className="text-sm text-muted-foreground">Emergency Contact</p>
                        <p className="font-medium">
                          {patientRecord.profile.emergencyContact}
                          {patientRecord.profile.emergencyPhone && ` - ${patientRecord.profile.emergencyPhone}`}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="medical" className="space-y-4 mt-0">
                {patientRecord.profile?.bloodType && (
                  <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                    <Droplets className="h-6 w-6 text-red-500" />
                    <div>
                      <p className="text-sm text-muted-foreground">Blood Type</p>
                      <p className="text-lg font-bold text-red-500">{patientRecord.profile.bloodType}</p>
                    </div>
                  </div>
                )}

                <div className="space-y-3">
                  <div className="p-4 rounded-lg border">
                    <div className="flex items-center gap-2 mb-3">
                      <AlertCircle className="h-5 w-5 text-orange-500" />
                      <p className="font-medium">Allergies</p>
                    </div>
                    {patientRecord.profile?.allergies && patientRecord.profile.allergies.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {patientRecord.profile.allergies.map((allergy, idx) => (
                          <Badge key={idx} variant="destructive">{allergy}</Badge>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No known allergies</p>
                    )}
                  </div>

                  <div className="p-4 rounded-lg border">
                    <div className="flex items-center gap-2 mb-3">
                      <Heart className="h-5 w-5 text-pink-500" />
                      <p className="font-medium">Medical Conditions</p>
                    </div>
                    {patientRecord.profile?.medicalConditions && patientRecord.profile.medicalConditions.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {patientRecord.profile.medicalConditions.map((condition, idx) => (
                          <Badge key={idx} variant="secondary">{condition}</Badge>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No recorded conditions</p>
                    )}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="appointments" className="space-y-4 mt-0">
                {patientRecord.upcomingAppointments.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-3 flex items-center gap-2">
                      <Clock className="h-4 w-4 text-blue-500" />
                      Upcoming Appointments
                    </h4>
                    <div className="space-y-2">
                      {patientRecord.upcomingAppointments.map((apt) => (
                        <div key={apt.id} className="p-3 rounded-lg border bg-blue-500/5">
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium">{apt.reason || 'General Visit'}</p>
                              <p className="text-sm text-muted-foreground">
                                {format(parseISO(apt.date), "EEEE, MMMM d")} at {apt.startTime}
                              </p>
                            </div>
                            <Badge variant={apt.status === 'confirmed' ? 'default' : 'secondary'}>
                              {apt.status}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {patientRecord.recentAppointments.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-3 flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      Recent Appointments
                    </h4>
                    <div className="space-y-2">
                      {patientRecord.recentAppointments.map((apt) => (
                        <div key={apt.id} className="p-3 rounded-lg border">
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium">{apt.reason || 'General Visit'}</p>
                              <p className="text-sm text-muted-foreground">
                                {format(parseISO(apt.date), "MMMM d, yyyy")} at {apt.startTime}
                              </p>
                            </div>
                            <Badge variant="outline">{apt.status}</Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {patientRecord.upcomingAppointments.length === 0 && 
                 patientRecord.recentAppointments.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <Calendar className="h-12 w-12 mx-auto mb-4 opacity-30" />
                    <p>No appointments on record</p>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-30" />
              <p>Failed to load patient record</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default PatientRecordAssistant;
