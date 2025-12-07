import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { 
  Shield, 
  Syringe, 
  Briefcase, 
  Dna, 
  AlertTriangle,
  Calendar,
  Building,
  Plus,
  Edit,
  Save
} from "lucide-react";
import { useState } from "react";
import { format } from "date-fns";

export interface InfectiousEvent {
  id: string;
  patient_id: string;
  infection_type: string;
  pathogen?: string;
  pathogen_category?: string;
  severity: string;
  onset_date?: string;
  resolution_date?: string;
  duration_days?: number;
  hospitalization: boolean;
  icu_admission: boolean;
  ventilator_required: boolean;
  auto_generated: boolean;
  manual_override: boolean;
}

export interface Immunization {
  id: string;
  patient_id: string;
  vaccine_code?: string;
  vaccine_name: string;
  vaccine_manufacturer?: string;
  dose_number?: number;
  series_name?: string;
  administration_date: string;
  adverse_reaction: boolean;
  reaction_details?: string;
  auto_generated: boolean;
}

export interface Occupation {
  id: string;
  patient_id: string;
  job_title: string;
  industry?: string;
  employer?: string;
  physical_demand_level?: string;
  shift_work: boolean;
  night_shift: boolean;
  hours_per_week?: number;
  start_date?: string;
  end_date?: string;
  is_current: boolean;
  auto_enriched: boolean;
}

export interface OccupationalExposure {
  id: string;
  occupation_id: string;
  exposure_type: string;
  exposure_agent?: string;
  exposure_level: string;
  auto_generated: boolean;
}

export interface GeneticRiskFlag {
  id: string;
  patient_id: string;
  flag_name: string;
  flag_type: string;
  value: string;
  risk_level?: string;
  clinical_implications?: string;
  affected_medications?: string[];
  affected_conditions?: string[];
  source: string;
  auto_generated: boolean;
}

function getSeverityColor(severity: string): string {
  switch (severity?.toLowerCase()) {
    case 'mild': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    case 'moderate': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    case 'severe': return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200';
    case 'critical': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
    default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
  }
}

function getRiskLevelColor(level: string): string {
  switch (level?.toLowerCase()) {
    case 'low': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    case 'moderate': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    case 'high': return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200';
    case 'critical': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
    default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
  }
}

function getExposureLevelColor(level: string): string {
  switch (level?.toLowerCase()) {
    case 'low': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    case 'high': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
    default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
  }
}

interface CardProps {
  patientId: string;
  isDoctor: boolean;
  compact?: boolean;
}

export function InfectionHistoryCard({ patientId, isDoctor, compact = false }: CardProps) {
  const { toast } = useToast();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editSeverity, setEditSeverity] = useState('');
  const [editNotes, setEditNotes] = useState('');

  const { data: infections, isLoading } = useQuery<InfectiousEvent[]>({
    queryKey: ['/api/patients', patientId, 'risk', 'infections'],
    queryFn: () => fetch(`/api/patients/${patientId}/risk/infections`, { credentials: 'include' }).then(r => r.json()),
    enabled: !!patientId,
  });

  const updateMutation = useMutation({
    mutationFn: (data: { id: string; severity?: string; notes?: string }) =>
      apiRequest(`/api/patients/${patientId}/risk/infections/${data.id}`, {
        method: 'PATCH',
        json: { severity: data.severity, notes: data.notes },
      }),
    onSuccess: () => {
      toast({ title: 'Infection record updated' });
      queryClient.invalidateQueries({ queryKey: ['/api/patients', patientId, 'risk', 'infections'] });
      setEditingId(null);
    },
    onError: () => {
      toast({ title: 'Failed to update', variant: 'destructive' });
    },
  });

  if (isLoading) {
    return (
      <Card data-testid="card-infections">
        <CardHeader className={compact ? "pb-2" : ""}>
          <CardTitle className="flex items-center gap-2 text-base">
            <Shield className="h-5 w-5 text-red-500" />
            Infection History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="card-infections">
      <CardHeader className={compact ? "pb-2" : ""}>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Shield className="h-5 w-5 text-red-500" />
            Infection History
          </CardTitle>
          <Badge variant="outline" data-testid="badge-infection-count">
            {infections?.length || 0} records
          </Badge>
        </div>
        {!compact && <CardDescription>Auto-generated from diagnoses and hospitalizations</CardDescription>}
      </CardHeader>
      <CardContent>
        {!infections || infections.length === 0 ? (
          <div className="text-center py-6 text-muted-foreground" data-testid="text-no-infections">
            <Shield className="h-10 w-10 mx-auto mb-2 opacity-30" />
            <p className="text-sm">No infection history recorded</p>
          </div>
        ) : (
          <div className="space-y-2" data-testid="list-infections">
            {infections.slice(0, compact ? 3 : 5).map((infection) => (
              <div 
                key={infection.id} 
                className="flex items-start justify-between p-3 bg-muted/50 rounded-lg"
                data-testid={`item-infection-${infection.id}`}
              >
                <div className="space-y-1 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-sm">{infection.infection_type}</span>
                    <Badge className={getSeverityColor(infection.severity)} variant="secondary">
                      {infection.severity}
                    </Badge>
                    {infection.hospitalization && (
                      <Badge variant="destructive" className="text-xs">Hospitalized</Badge>
                    )}
                    {infection.icu_admission && (
                      <Badge variant="destructive" className="text-xs">ICU</Badge>
                    )}
                  </div>
                  {infection.pathogen && (
                    <p className="text-xs text-muted-foreground">
                      Pathogen: {infection.pathogen}
                    </p>
                  )}
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    <Calendar className="h-3 w-3" />
                    {infection.onset_date 
                      ? format(new Date(infection.onset_date), 'MMM d, yyyy')
                      : 'Date unknown'}
                    {infection.duration_days && ` (${infection.duration_days} days)`}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {infection.auto_generated && !infection.manual_override && (
                    <Badge variant="outline" className="text-xs">Auto</Badge>
                  )}
                  {isDoctor && (
                    <Dialog open={editingId === infection.id} onOpenChange={(open) => {
                      if (open) {
                        setEditingId(infection.id);
                        setEditSeverity(infection.severity);
                      } else {
                        setEditingId(null);
                      }
                    }}>
                      <DialogTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-7 w-7" data-testid={`button-edit-infection-${infection.id}`}>
                          <Edit className="h-3 w-3" />
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Edit Infection Record</DialogTitle>
                        </DialogHeader>
                        <div className="space-y-4 mt-4">
                          <div>
                            <Label>Infection Type</Label>
                            <Input value={infection.infection_type} disabled className="bg-muted" />
                          </div>
                          <div>
                            <Label>Severity</Label>
                            <Select value={editSeverity} onValueChange={setEditSeverity}>
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="mild">Mild</SelectItem>
                                <SelectItem value="moderate">Moderate</SelectItem>
                                <SelectItem value="severe">Severe</SelectItem>
                                <SelectItem value="critical">Critical</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div>
                            <Label>Clinical Notes</Label>
                            <Textarea
                              value={editNotes}
                              onChange={(e) => setEditNotes(e.target.value)}
                              placeholder="Add clinical notes..."
                            />
                          </div>
                        </div>
                        <DialogFooter className="mt-4">
                          <Button
                            onClick={() => updateMutation.mutate({ id: infection.id, severity: editSeverity, notes: editNotes })}
                            disabled={updateMutation.isPending}
                          >
                            <Save className="h-4 w-4 mr-2" />
                            {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
                          </Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                  )}
                </div>
              </div>
            ))}
            {infections.length > (compact ? 3 : 5) && (
              <p className="text-xs text-muted-foreground text-center">
                +{infections.length - (compact ? 3 : 5)} more infections
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function VaccinationsCard({ patientId, isDoctor, compact = false }: CardProps) {
  const { data: immunizations, isLoading } = useQuery<Immunization[]>({
    queryKey: ['/api/patients', patientId, 'risk', 'immunizations'],
    queryFn: () => fetch(`/api/patients/${patientId}/risk/immunizations`, { credentials: 'include' }).then(r => r.json()),
    enabled: !!patientId,
  });

  if (isLoading) {
    return (
      <Card data-testid="card-immunizations">
        <CardHeader className={compact ? "pb-2" : ""}>
          <CardTitle className="flex items-center gap-2 text-base">
            <Syringe className="h-5 w-5 text-blue-500" />
            Vaccinations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  const vaccineGroups = immunizations?.reduce((acc, imm) => {
    const name = imm.vaccine_name;
    if (!acc[name]) acc[name] = [];
    acc[name].push(imm);
    return acc;
  }, {} as Record<string, Immunization[]>) || {};

  return (
    <Card data-testid="card-immunizations">
      <CardHeader className={compact ? "pb-2" : ""}>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Syringe className="h-5 w-5 text-blue-500" />
            Vaccinations
          </CardTitle>
          <Badge variant="outline" data-testid="badge-immunization-count">
            {immunizations?.length || 0} doses
          </Badge>
        </div>
        {!compact && <CardDescription>Immunization records from prescriptions and imports</CardDescription>}
      </CardHeader>
      <CardContent>
        {!immunizations || immunizations.length === 0 ? (
          <div className="text-center py-6 text-muted-foreground" data-testid="text-no-immunizations">
            <Syringe className="h-10 w-10 mx-auto mb-2 opacity-30" />
            <p className="text-sm">No vaccination records</p>
          </div>
        ) : (
          <div className="space-y-2" data-testid="list-immunizations">
            {Object.entries(vaccineGroups).slice(0, compact ? 3 : 5).map(([name, doses]) => (
              <div 
                key={name} 
                className="flex items-start justify-between p-3 bg-muted/50 rounded-lg"
                data-testid={`item-vaccine-${name.replace(/\s+/g, '-')}`}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-sm">{name}</span>
                    <Badge variant="secondary">{doses.length} dose{doses.length > 1 ? 's' : ''}</Badge>
                    {doses[0].series_name && (
                      <Badge variant="outline" className="text-xs capitalize">
                        {doses[0].series_name}
                      </Badge>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    <Calendar className="h-3 w-3" />
                    Last: {format(new Date(doses[0].administration_date), 'MMM d, yyyy')}
                  </p>
                  {doses.some(d => d.adverse_reaction) && (
                    <Badge variant="destructive" className="text-xs">Adverse reaction reported</Badge>
                  )}
                </div>
              </div>
            ))}
            {Object.keys(vaccineGroups).length > (compact ? 3 : 5) && (
              <p className="text-xs text-muted-foreground text-center">
                +{Object.keys(vaccineGroups).length - (compact ? 3 : 5)} more vaccine types
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function OccupationCard({ patientId, isDoctor, compact = false }: CardProps) {
  const { toast } = useToast();
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [jobTitle, setJobTitle] = useState('');
  const [industry, setIndustry] = useState('');

  const { data, isLoading } = useQuery<{ occupation: Occupation | null; exposures: OccupationalExposure[] }>({
    queryKey: ['/api/patients', patientId, 'risk', 'occupation'],
    queryFn: () => fetch(`/api/patients/${patientId}/risk/occupation`, { credentials: 'include' }).then(r => r.json()),
    enabled: !!patientId,
  });

  const addOccupationMutation = useMutation({
    mutationFn: (data: { jobTitle: string; industry?: string }) =>
      apiRequest(`/api/patients/${patientId}/risk/occupation`, {
        method: 'POST',
        json: data,
      }),
    onSuccess: () => {
      toast({ title: 'Occupation added successfully' });
      queryClient.invalidateQueries({ queryKey: ['/api/patients', patientId, 'risk', 'occupation'] });
      setShowAddDialog(false);
      setJobTitle('');
      setIndustry('');
    },
    onError: () => {
      toast({ title: 'Failed to add occupation', variant: 'destructive' });
    },
  });

  if (isLoading) {
    return (
      <Card data-testid="card-occupation">
        <CardHeader className={compact ? "pb-2" : ""}>
          <CardTitle className="flex items-center gap-2 text-base">
            <Briefcase className="h-5 w-5 text-amber-500" />
            Occupation & Exposures
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  const { occupation, exposures } = data || { occupation: null, exposures: [] };

  return (
    <Card data-testid="card-occupation">
      <CardHeader className={compact ? "pb-2" : ""}>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Briefcase className="h-5 w-5 text-amber-500" />
            Occupation & Exposures
          </CardTitle>
          {isDoctor && (
            <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm" data-testid="button-add-occupation">
                  <Plus className="h-4 w-4 mr-1" />
                  {occupation ? 'Change' : 'Add'}
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{occupation ? 'Update' : 'Add'} Occupation</DialogTitle>
                </DialogHeader>
                <div className="space-y-4 mt-4">
                  <div>
                    <Label htmlFor="jobTitle">Job Title</Label>
                    <Input
                      id="jobTitle"
                      value={jobTitle}
                      onChange={(e) => setJobTitle(e.target.value)}
                      placeholder="e.g., Nurse, Construction Worker"
                      data-testid="input-job-title"
                    />
                  </div>
                  <div>
                    <Label htmlFor="industry">Industry</Label>
                    <Select value={industry} onValueChange={setIndustry}>
                      <SelectTrigger data-testid="select-industry">
                        <SelectValue placeholder="Select industry" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="healthcare">Healthcare</SelectItem>
                        <SelectItem value="construction">Construction</SelectItem>
                        <SelectItem value="manufacturing">Manufacturing</SelectItem>
                        <SelectItem value="office">Office/Administrative</SelectItem>
                        <SelectItem value="education">Education</SelectItem>
                        <SelectItem value="transportation">Transportation</SelectItem>
                        <SelectItem value="food_service">Food Service</SelectItem>
                        <SelectItem value="retail">Retail</SelectItem>
                        <SelectItem value="agriculture">Agriculture</SelectItem>
                        <SelectItem value="first_responders">First Responders</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button 
                    onClick={() => addOccupationMutation.mutate({ jobTitle, industry })}
                    disabled={!jobTitle || addOccupationMutation.isPending}
                    className="w-full"
                    data-testid="button-submit-occupation"
                  >
                    {addOccupationMutation.isPending ? 'Saving...' : 'Save Occupation'}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          )}
        </div>
        {!compact && <CardDescription>Job-based risk exposures (auto-inferred from job title)</CardDescription>}
      </CardHeader>
      <CardContent>
        {!occupation ? (
          <div className="text-center py-6 text-muted-foreground" data-testid="text-no-occupation">
            <Briefcase className="h-10 w-10 mx-auto mb-2 opacity-30" />
            <p className="text-sm">No occupation recorded</p>
            {isDoctor && <p className="text-xs mt-1">Add job to see inferred risk exposures</p>}
          </div>
        ) : (
          <div className="space-y-3">
            <div className="p-3 bg-muted/50 rounded-lg" data-testid="item-current-occupation">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-sm">{occupation.job_title}</p>
                  {occupation.industry && (
                    <p className="text-xs text-muted-foreground capitalize">
                      {occupation.industry.replace(/_/g, ' ')}
                    </p>
                  )}
                  {occupation.employer && (
                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                      <Building className="h-3 w-3" /> {occupation.employer}
                    </p>
                  )}
                </div>
                <div className="text-right flex flex-wrap gap-1 justify-end">
                  {occupation.physical_demand_level && (
                    <Badge variant="outline" className="capitalize text-xs">
                      {occupation.physical_demand_level.replace(/_/g, ' ')}
                    </Badge>
                  )}
                  {occupation.shift_work && (
                    <Badge variant="secondary" className="text-xs">Shift Work</Badge>
                  )}
                </div>
              </div>
            </div>

            {exposures.length > 0 && !compact && (
              <div>
                <p className="text-xs font-medium mb-2">Inferred Exposures</p>
                <div className="space-y-1" data-testid="list-exposures">
                  {exposures.slice(0, 4).map((exp) => (
                    <div 
                      key={exp.id}
                      className="flex items-center justify-between text-xs"
                      data-testid={`item-exposure-${exp.id}`}
                    >
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-3 w-3 text-muted-foreground" />
                        <span className="capitalize">{exp.exposure_type}</span>
                        {exp.exposure_agent && (
                          <span className="text-muted-foreground">({exp.exposure_agent})</span>
                        )}
                      </div>
                      <Badge className={getExposureLevelColor(exp.exposure_level)} variant="secondary">
                        {exp.exposure_level}
                      </Badge>
                    </div>
                  ))}
                  {exposures.length > 4 && (
                    <p className="text-xs text-muted-foreground text-center">
                      +{exposures.length - 4} more exposures
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function GeneticRiskCard({ patientId, isDoctor, compact = false }: CardProps) {
  const { toast } = useToast();
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [flagName, setFlagName] = useState('');
  const [flagType, setFlagType] = useState('');
  const [riskLevel, setRiskLevel] = useState('');
  const [notes, setNotes] = useState('');

  const { data: flags, isLoading } = useQuery<GeneticRiskFlag[]>({
    queryKey: ['/api/patients', patientId, 'risk', 'genetics'],
    queryFn: () => fetch(`/api/patients/${patientId}/risk/genetics`, { credentials: 'include' }).then(r => r.json()),
    enabled: !!patientId,
  });

  const addFlagMutation = useMutation({
    mutationFn: (data: { flagName: string; flagType: string; riskLevel: string; notes?: string }) =>
      apiRequest(`/api/patients/${patientId}/risk/genetics`, {
        method: 'POST',
        json: data,
      }),
    onSuccess: () => {
      toast({ title: 'Genetic flag added' });
      queryClient.invalidateQueries({ queryKey: ['/api/patients', patientId, 'risk', 'genetics'] });
      setShowAddDialog(false);
      setFlagName('');
      setFlagType('');
      setRiskLevel('');
      setNotes('');
    },
    onError: () => {
      toast({ title: 'Failed to add flag', variant: 'destructive' });
    },
  });

  if (isLoading) {
    return (
      <Card data-testid="card-genetics">
        <CardHeader className={compact ? "pb-2" : ""}>
          <CardTitle className="flex items-center gap-2 text-base">
            <Dna className="h-5 w-5 text-purple-500" />
            Genetic & Family Risk
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="card-genetics">
      <CardHeader className={compact ? "pb-2" : ""}>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Dna className="h-5 w-5 text-purple-500" />
            Genetic & Family Risk
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" data-testid="badge-genetic-count">
              {flags?.length || 0} flags
            </Badge>
            {isDoctor && (
              <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm" data-testid="button-add-genetic">
                    <Plus className="h-4 w-4" />
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Add Genetic/Family Risk Flag</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4 mt-4">
                    <div>
                      <Label>Flag Name</Label>
                      <Input
                        value={flagName}
                        onChange={(e) => setFlagName(e.target.value)}
                        placeholder="e.g., BRCA1 Mutation, Family History - Diabetes"
                      />
                    </div>
                    <div>
                      <Label>Flag Type</Label>
                      <Select value={flagType} onValueChange={setFlagType}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="genetic_marker">Genetic Marker</SelectItem>
                          <SelectItem value="family_history">Family History</SelectItem>
                          <SelectItem value="pharmacogenomic">Pharmacogenomic</SelectItem>
                          <SelectItem value="hereditary_condition">Hereditary Condition</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label>Risk Level</Label>
                      <Select value={riskLevel} onValueChange={setRiskLevel}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select risk level" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="low">Low</SelectItem>
                          <SelectItem value="moderate">Moderate</SelectItem>
                          <SelectItem value="high">High</SelectItem>
                          <SelectItem value="critical">Critical</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label>Clinical Notes</Label>
                      <Textarea
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        placeholder="Add clinical implications..."
                      />
                    </div>
                    <Button 
                      onClick={() => addFlagMutation.mutate({ flagName, flagType, riskLevel, notes })}
                      disabled={!flagName || !flagType || !riskLevel || addFlagMutation.isPending}
                      className="w-full"
                    >
                      {addFlagMutation.isPending ? 'Saving...' : 'Add Flag'}
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            )}
          </div>
        </div>
        {!compact && <CardDescription>Auto-derived from genetic panels and lab results</CardDescription>}
      </CardHeader>
      <CardContent>
        {!flags || flags.length === 0 ? (
          <div className="text-center py-6 text-muted-foreground" data-testid="text-no-genetics">
            <Dna className="h-10 w-10 mx-auto mb-2 opacity-30" />
            <p className="text-sm">No genetic risk flags</p>
            {isDoctor && <p className="text-xs mt-1">Add flags manually or they auto-generate from lab results</p>}
          </div>
        ) : (
          <div className="space-y-2" data-testid="list-genetics">
            {flags.slice(0, compact ? 3 : 5).map((flag) => (
              <div 
                key={flag.id} 
                className="p-3 bg-muted/50 rounded-lg"
                data-testid={`item-genetic-${flag.id}`}
              >
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-medium text-sm">{flag.flag_name}</span>
                      {flag.risk_level && (
                        <Badge className={getRiskLevelColor(flag.risk_level)} variant="secondary">
                          {flag.risk_level} risk
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground capitalize">
                      Type: {flag.flag_type.replace(/_/g, ' ')}
                    </p>
                    {!compact && flag.clinical_implications && (
                      <p className="text-xs text-muted-foreground line-clamp-2">
                        {flag.clinical_implications}
                      </p>
                    )}
                    {!compact && flag.affected_medications && flag.affected_medications.length > 0 && (
                      <div className="flex items-center gap-1 flex-wrap mt-1">
                        <span className="text-xs text-muted-foreground">Affects:</span>
                        {flag.affected_medications.slice(0, 3).map((med) => (
                          <Badge key={med} variant="outline" className="text-xs">
                            {med}
                          </Badge>
                        ))}
                        {flag.affected_medications.length > 3 && (
                          <span className="text-xs text-muted-foreground">
                            +{flag.affected_medications.length - 3}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {flags.length > (compact ? 3 : 5) && (
              <p className="text-xs text-muted-foreground text-center">
                +{flags.length - (compact ? 3 : 5)} more genetic flags
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function RiskExposuresPanel({ patientId, isDoctor, compact = false }: CardProps) {
  return (
    <div className={compact ? "space-y-4" : "grid grid-cols-1 md:grid-cols-2 gap-4"}>
      <InfectionHistoryCard patientId={patientId} isDoctor={isDoctor} compact={compact} />
      <VaccinationsCard patientId={patientId} isDoctor={isDoctor} compact={compact} />
      <OccupationCard patientId={patientId} isDoctor={isDoctor} compact={compact} />
      <GeneticRiskCard patientId={patientId} isDoctor={isDoctor} compact={compact} />
    </div>
  );
}
