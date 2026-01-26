import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { PatientEnvironmentRiskCard } from "./PatientEnvironmentRiskCard";
import { 
  Search, 
  User, 
  Phone, 
  Mail, 
  Calendar, 
  AlertCircle,
  AlertTriangle,
  Heart,
  Droplets,
  Clock,
  ChevronRight,
  Loader2,
  Users,
  FileText,
  Stethoscope,
  Bell,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Thermometer,
  Brain,
  Zap,
  CheckCircle2,
  ClipboardList,
  CloudSun
} from "lucide-react";
import { format, parseISO, differenceInYears, formatDistanceToNow } from "date-fns";

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

interface HealthAlert {
  id: string;
  patientId: string;
  alertType: string;
  severity: "critical" | "high" | "moderate" | "low";
  message: string;
  triggeredAt: string;
  acknowledged: boolean;
  acknowledgedAt?: string;
  acknowledgedBy?: string;
  dismissed: boolean;
  escalationProbability?: number;
  contributingFactors?: string[];
}

interface DailyFollowup {
  id: string;
  patientId: string;
  date: string;
  overallFeeling: number;
  painLevel?: number;
  energyLevel?: number;
  sleepQuality?: number;
  symptoms?: string[];
  notes?: string;
  vitalSigns?: {
    heartRate?: number;
    bloodPressure?: string;
    temperature?: number;
    oxygenSaturation?: number;
    respiratoryRate?: number;
  };
  createdAt: string;
}

interface RiskScore {
  compositeScore: number;
  riskLevel: "low" | "moderate" | "high" | "critical";
  factors: Array<{
    name: string;
    score: number;
    weight: number;
  }>;
  trend: "improving" | "stable" | "deteriorating";
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

  const { data: healthAlerts, isLoading: alertsLoading, error: alertsError } = useQuery<HealthAlert[]>({
    queryKey: ['/api/ai-health-alerts/alerts', selectedPatientId],
    queryFn: async () => {
      if (!selectedPatientId) return [];
      const response = await fetch(`/api/ai-health-alerts/alerts/${selectedPatientId}?limit=20`);
      if (!response.ok) {
        if (response.status === 404) return [];
        throw new Error('Failed to fetch health alerts');
      }
      return response.json();
    },
    enabled: !!selectedPatientId
  });

  const { data: riskData, isLoading: riskLoading, error: riskError } = useQuery<RiskScore | null>({
    queryKey: ['/api/ai-health-alerts/risk-score', selectedPatientId],
    queryFn: async () => {
      if (!selectedPatientId) return null;
      const response = await fetch(`/api/ai-health-alerts/risk-score/${selectedPatientId}`);
      if (!response.ok) {
        if (response.status === 404) return null;
        throw new Error('Failed to fetch risk score');
      }
      return response.json();
    },
    enabled: !!selectedPatientId
  });

  const { data: dailyFollowups, isLoading: followupsLoading, error: followupsError } = useQuery<DailyFollowup[]>({
    queryKey: ['/api/daily-followup/patient', selectedPatientId],
    queryFn: async () => {
      if (!selectedPatientId) return [];
      const response = await fetch(`/api/daily-followup/patient/${selectedPatientId}?limit=14`);
      if (!response.ok) {
        if (response.status === 404) return [];
        throw new Error('Failed to fetch daily followups');
      }
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

  const getSeverityStyles = (severity: string) => {
    switch (severity) {
      case "critical":
        return { bg: "bg-rose-500/10 border-rose-500/30", text: "text-rose-600 dark:text-rose-400", badge: "bg-rose-100 text-rose-700 dark:bg-rose-900 dark:text-rose-300" };
      case "high":
        return { bg: "bg-orange-500/10 border-orange-500/30", text: "text-orange-600 dark:text-orange-400", badge: "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300" };
      case "moderate":
        return { bg: "bg-yellow-500/10 border-yellow-500/30", text: "text-yellow-600 dark:text-yellow-400", badge: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300" };
      default:
        return { bg: "bg-blue-500/10 border-blue-500/30", text: "text-blue-600 dark:text-blue-400", badge: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300" };
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case "critical": return "text-rose-500";
      case "high": return "text-orange-500";
      case "moderate": return "text-yellow-500";
      default: return "text-green-500";
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving": return { Icon: TrendingDown, color: "text-green-500" };
      case "deteriorating": return { Icon: TrendingUp, color: "text-rose-500" };
      default: return { Icon: Minus, color: "text-muted-foreground" };
    }
  };

  const getFeelingEmoji = (level: number) => {
    if (level >= 8) return { icon: "text-green-500", label: "Great" };
    if (level >= 6) return { icon: "text-lime-500", label: "Good" };
    if (level >= 4) return { icon: "text-yellow-500", label: "Fair" };
    if (level >= 2) return { icon: "text-orange-500", label: "Poor" };
    return { icon: "text-rose-500", label: "Bad" };
  };

  const activeAlerts = healthAlerts?.filter(a => !a.dismissed) || [];
  const criticalAlerts = activeAlerts.filter(a => a.severity === "critical" || a.severity === "high");

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
              <TabsList className="grid w-full grid-cols-6 mb-4">
                <TabsTrigger value="info" data-testid="tab-patient-info">Info</TabsTrigger>
                <TabsTrigger value="medical" data-testid="tab-patient-medical">Medical</TabsTrigger>
                <TabsTrigger value="alerts" data-testid="tab-patient-alerts" className="relative">
                  Alerts
                  {criticalAlerts.length > 0 && (
                    <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-rose-500 text-[10px] text-white flex items-center justify-center">
                      {criticalAlerts.length}
                    </span>
                  )}
                </TabsTrigger>
                <TabsTrigger value="environment" data-testid="tab-patient-environment">
                  <CloudSun className="h-4 w-4 mr-1" />
                  Environ
                </TabsTrigger>
                <TabsTrigger value="followups" data-testid="tab-patient-followups">Followups</TabsTrigger>
                <TabsTrigger value="appointments" data-testid="tab-patient-appointments">Appts</TabsTrigger>
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

              <TabsContent value="alerts" className="space-y-4 mt-0">
                {alertsLoading || riskLoading ? (
                  <div className="space-y-3">
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-16 w-full" />
                    <Skeleton className="h-16 w-full" />
                  </div>
                ) : alertsError || riskError ? (
                  <div className="text-center py-8">
                    <AlertCircle className="h-12 w-12 mx-auto mb-4 text-destructive opacity-50" />
                    <p className="font-medium text-destructive">Failed to Load Alerts</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {alertsError?.message || riskError?.message || 'Unable to fetch health alert data'}
                    </p>
                  </div>
                ) : (
                  <>
                    {riskData && (
                      <div className="p-4 rounded-lg border bg-muted/30">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <Brain className="h-5 w-5 text-primary" />
                            <span className="font-medium">Risk Assessment</span>
                          </div>
                          <div className="flex items-center gap-2">
                            {(() => {
                              const { Icon, color } = getTrendIcon(riskData.trend);
                              return (
                                <>
                                  <Icon className={`h-4 w-4 ${color}`} />
                                  <span className={`text-sm ${color}`}>{riskData.trend}</span>
                                </>
                              );
                            })()}
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="text-center">
                            <p className={`text-3xl font-bold ${getRiskLevelColor(riskData.riskLevel)}`}>
                              {riskData.compositeScore.toFixed(1)}
                            </p>
                            <p className="text-xs text-muted-foreground">Risk Score</p>
                          </div>
                          <div className="flex-1">
                            <Progress 
                              value={(riskData.compositeScore / 15) * 100} 
                              className="h-2"
                            />
                            <p className={`text-sm mt-1 ${getRiskLevelColor(riskData.riskLevel)}`}>
                              {riskData.riskLevel.charAt(0).toUpperCase() + riskData.riskLevel.slice(1)} Risk
                            </p>
                          </div>
                        </div>
                      </div>
                    )}

                    {activeAlerts.length > 0 ? (
                      <ScrollArea className="h-[280px]">
                        <div className="space-y-2 pr-4">
                          {activeAlerts.map((alert) => {
                            const styles = getSeverityStyles(alert.severity);
                            return (
                              <div
                                key={alert.id}
                                className={`p-3 rounded-lg border ${styles.bg}`}
                                data-testid={`alert-item-${alert.id}`}
                              >
                                <div className="flex items-start gap-3">
                                  <AlertTriangle className={`h-5 w-5 mt-0.5 ${styles.text}`} />
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 flex-wrap mb-1">
                                      <Badge className={styles.badge}>
                                        {alert.severity.toUpperCase()}
                                      </Badge>
                                      <Badge variant="outline" className="text-xs">
                                        {alert.alertType}
                                      </Badge>
                                      {alert.escalationProbability && alert.escalationProbability > 0.5 && (
                                        <Badge variant="secondary" className="text-xs">
                                          <Zap className="h-3 w-3 mr-1" />
                                          {(alert.escalationProbability * 100).toFixed(0)}% escalation
                                        </Badge>
                                      )}
                                    </div>
                                    <p className="text-sm">{alert.message}</p>
                                    <p className="text-xs text-muted-foreground mt-1">
                                      {formatDistanceToNow(parseISO(alert.triggeredAt), { addSuffix: true })}
                                    </p>
                                    {alert.contributingFactors && alert.contributingFactors.length > 0 && (
                                      <div className="flex flex-wrap gap-1 mt-2">
                                        {alert.contributingFactors.slice(0, 3).map((factor, idx) => (
                                          <Badge key={idx} variant="outline" className="text-xs">
                                            {factor}
                                          </Badge>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                  {alert.acknowledged && (
                                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </ScrollArea>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <Bell className="h-12 w-12 mx-auto mb-4 opacity-30" />
                        <p className="font-medium">No Active Alerts</p>
                        <p className="text-sm">This patient has no current health alerts</p>
                      </div>
                    )}
                  </>
                )}
              </TabsContent>

              <TabsContent value="environment" className="space-y-4 mt-0">
                {selectedPatientId ? (
                  <PatientEnvironmentRiskCard patientId={selectedPatientId} />
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <CloudSun className="h-12 w-12 mx-auto mb-4 opacity-30" />
                    <p className="font-medium">No Patient Selected</p>
                    <p className="text-sm">Select a patient to view environmental risk data</p>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="followups" className="space-y-4 mt-0">
                {followupsLoading ? (
                  <div className="space-y-3">
                    <Skeleton className="h-24 w-full" />
                    <Skeleton className="h-24 w-full" />
                    <Skeleton className="h-24 w-full" />
                  </div>
                ) : followupsError ? (
                  <div className="text-center py-8">
                    <AlertCircle className="h-12 w-12 mx-auto mb-4 text-destructive opacity-50" />
                    <p className="font-medium text-destructive">Failed to Load Followups</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {followupsError?.message || 'Unable to fetch daily followup data'}
                    </p>
                  </div>
                ) : dailyFollowups && dailyFollowups.length > 0 ? (
                  <ScrollArea className="h-[350px]">
                    <div className="space-y-3 pr-4">
                      {dailyFollowups.map((followup) => {
                        const feeling = getFeelingEmoji(followup.overallFeeling);
                        return (
                          <div
                            key={followup.id}
                            className="p-4 rounded-lg border"
                            data-testid={`followup-item-${followup.id}`}
                          >
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex items-center gap-2">
                                <ClipboardList className="h-4 w-4 text-primary" />
                                <span className="font-medium">
                                  {format(parseISO(followup.date), "EEEE, MMM d")}
                                </span>
                              </div>
                              <div className="flex items-center gap-2">
                                <Activity className={`h-4 w-4 ${feeling.icon}`} />
                                <span className={`text-sm font-medium ${feeling.icon}`}>
                                  {feeling.label} ({followup.overallFeeling}/10)
                                </span>
                              </div>
                            </div>

                            <div className="grid grid-cols-2 gap-2 text-sm">
                              {followup.painLevel !== undefined && (
                                <div className="flex items-center gap-2">
                                  <span className="text-muted-foreground">Pain:</span>
                                  <span className={followup.painLevel > 5 ? "text-rose-500" : ""}>
                                    {followup.painLevel}/10
                                  </span>
                                </div>
                              )}
                              {followup.energyLevel !== undefined && (
                                <div className="flex items-center gap-2">
                                  <span className="text-muted-foreground">Energy:</span>
                                  <span>{followup.energyLevel}/10</span>
                                </div>
                              )}
                              {followup.sleepQuality !== undefined && (
                                <div className="flex items-center gap-2">
                                  <span className="text-muted-foreground">Sleep:</span>
                                  <span>{followup.sleepQuality}/10</span>
                                </div>
                              )}
                            </div>

                            {followup.vitalSigns && (
                              <div className="mt-3 pt-3 border-t">
                                <div className="flex flex-wrap gap-3 text-xs">
                                  {followup.vitalSigns.heartRate && (
                                    <div className="flex items-center gap-1">
                                      <Heart className="h-3 w-3 text-rose-500" />
                                      <span>{followup.vitalSigns.heartRate} bpm</span>
                                    </div>
                                  )}
                                  {followup.vitalSigns.bloodPressure && (
                                    <div className="flex items-center gap-1">
                                      <Activity className="h-3 w-3 text-blue-500" />
                                      <span>{followup.vitalSigns.bloodPressure}</span>
                                    </div>
                                  )}
                                  {followup.vitalSigns.temperature && (
                                    <div className="flex items-center gap-1">
                                      <Thermometer className="h-3 w-3 text-orange-500" />
                                      <span>{followup.vitalSigns.temperature}°F</span>
                                    </div>
                                  )}
                                  {followup.vitalSigns.oxygenSaturation && (
                                    <div className="flex items-center gap-1">
                                      <Droplets className="h-3 w-3 text-cyan-500" />
                                      <span>{followup.vitalSigns.oxygenSaturation}% SpO2</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                            {followup.symptoms && followup.symptoms.length > 0 && (
                              <div className="mt-3 pt-3 border-t">
                                <div className="flex flex-wrap gap-1">
                                  {followup.symptoms.map((symptom, idx) => (
                                    <Badge key={idx} variant="secondary" className="text-xs">
                                      {symptom}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}

                            {followup.notes && (
                              <p className="mt-3 pt-3 border-t text-sm text-muted-foreground italic">
                                "{followup.notes}"
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <ClipboardList className="h-12 w-12 mx-auto mb-4 opacity-30" />
                    <p className="font-medium">No Daily Followups</p>
                    <p className="text-sm">This patient hasn't submitted any daily check-ins yet</p>
                  </div>
                )}
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
