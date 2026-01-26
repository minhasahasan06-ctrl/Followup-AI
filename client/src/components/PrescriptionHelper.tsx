import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { 
  Pill, 
  AlertTriangle,
  AlertCircle,
  CheckCircle,
  Plus,
  X,
  Loader2,
  Shield,
  Beaker,
  Info,
  Heart
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

interface Medication {
  name: string;
  dosage: string;
  frequency: string;
  duration: string;
  route: string;
  instructions: string;
}

interface DrugInteraction {
  drug1: string;
  drug2: string;
  severity: 'minor' | 'moderate' | 'major' | 'contraindicated';
  description: string;
  clinicalEffect: string;
  recommendation: string;
}

interface InteractionCheckResult {
  hasInteractions: boolean;
  interactions: DrugInteraction[];
  allergicRisks: string[];
  contraindications: string[];
  warnings: string[];
  safeToPresrcibe: boolean;
  crossSpecialtyCount?: number;
  sameSpecialtyCount?: number;
  allMedicationsCount?: number;
  checkedMedications?: string[];
  _note?: string;
  _allMedicationsMode?: boolean;
  _specialty?: string | null;
  _fallback?: boolean;
  // Error response fields
  error?: string;
  message?: string;
  hint?: string;
}

interface PrescriptionHelperProps {
  patientId?: string;
  patientName?: string;
  patientAllergies?: string[];
  currentMedications?: string[];
  doctorSpecialty?: string;
  className?: string;
}

export function PrescriptionHelper({ 
  patientId, 
  patientName, 
  patientAllergies = [], 
  currentMedications = [],
  doctorSpecialty,
  className 
}: PrescriptionHelperProps) {
  const [medications, setMedications] = useState<Medication[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const { toast } = useToast();
  const [selectedMedication, setSelectedMedication] = useState<string>("");
  const [dosage, setDosage] = useState("");
  const [frequency, setFrequency] = useState("");
  const [duration, setDuration] = useState("");
  const [route, setRoute] = useState("oral");
  const [instructions, setInstructions] = useState("");
  const [interactionResult, setInteractionResult] = useState<InteractionCheckResult | null>(null);

  const frequencyOptions = [
    { value: "once_daily", label: "Once daily" },
    { value: "twice_daily", label: "Twice daily (BID)" },
    { value: "three_times_daily", label: "Three times daily (TID)" },
    { value: "four_times_daily", label: "Four times daily (QID)" },
    { value: "every_4_hours", label: "Every 4 hours" },
    { value: "every_6_hours", label: "Every 6 hours" },
    { value: "every_8_hours", label: "Every 8 hours" },
    { value: "every_12_hours", label: "Every 12 hours" },
    { value: "as_needed", label: "As needed (PRN)" },
    { value: "at_bedtime", label: "At bedtime (HS)" },
    { value: "weekly", label: "Weekly" }
  ];

  const routeOptions = [
    { value: "oral", label: "Oral (PO)" },
    { value: "sublingual", label: "Sublingual (SL)" },
    { value: "topical", label: "Topical" },
    { value: "inhalation", label: "Inhalation" },
    { value: "injection_im", label: "Intramuscular (IM)" },
    { value: "injection_iv", label: "Intravenous (IV)" },
    { value: "injection_sc", label: "Subcutaneous (SC)" },
    { value: "rectal", label: "Rectal (PR)" },
    { value: "ophthalmic", label: "Ophthalmic (Eye)" },
    { value: "otic", label: "Otic (Ear)" },
    { value: "nasal", label: "Nasal" },
    { value: "transdermal", label: "Transdermal (Patch)" }
  ];

  const interactionCheckMutation = useMutation({
    mutationFn: async () => {
      const allMedications = [
        ...medications.map(m => m.name),
        ...currentMedications
      ];

      const response = await apiRequest('/api/v1/lysa/prescriptions/check-interactions', {
        method: 'POST',
        body: JSON.stringify({
          patientId,
          doctorSpecialty,
          medications: allMedications,
          allergies: patientAllergies,
          newPrescriptions: medications.map(m => m.name)
        })
      });
      return response;
    },
    onSuccess: (data: InteractionCheckResult) => {
      setInteractionResult(data);
      
      // Determine toast variant and messaging based on response state
      if (data._fallback) {
        toast({
          title: "AI Analysis Unavailable",
          description: "Please verify drug interactions manually using clinical references.",
          variant: "destructive",
        });
      } else if (data._allMedicationsMode) {
        toast({
          title: "All-Medications Mode Active",
          description: `Checked against ALL ${data.allMedicationsCount || 0} patient medication(s). Cross-specialty filtering was bypassed.`,
          variant: "destructive",
        });
      } else {
        const crossSpecialtyNote = data.crossSpecialtyCount !== undefined 
          ? ` (checked against ${data.crossSpecialtyCount} cross-specialty medication${data.crossSpecialtyCount !== 1 ? 's' : ''})`
          : '';
        toast({
          title: "Cross-Specialty Safety Check Complete",
          description: data.safeToPresrcibe 
            ? `No critical drug interactions found${crossSpecialtyNote}.`
            : `Safety concerns identified${crossSpecialtyNote}. Please review.`,
          variant: data.safeToPresrcibe ? "default" : "destructive",
        });
      }
    },
    onError: (error: Error) => {
      toast({
        title: "Safety Check Failed",
        description: error.message || "Failed to check drug interactions. Please try again.",
        variant: "destructive",
      });
    }
  });

  const addMedication = () => {
    if (selectedMedication && dosage && frequency) {
      const newMed: Medication = {
        name: selectedMedication,
        dosage,
        frequency,
        duration: duration || "As directed",
        route,
        instructions: instructions || "Take as directed"
      };
      setMedications([...medications, newMed]);
      
      setSelectedMedication("");
      setDosage("");
      setFrequency("");
      setDuration("");
      setRoute("oral");
      setInstructions("");
      setInteractionResult(null);
    }
  };

  const removeMedication = (index: number) => {
    setMedications(medications.filter((_, i) => i !== index));
    setInteractionResult(null);
  };

  const checkInteractions = () => {
    if (medications.length > 0) {
      interactionCheckMutation.mutate();
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'minor': return 'bg-blue-500/10 text-blue-600 border-blue-500/20';
      case 'moderate': return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
      case 'major': return 'bg-orange-500/10 text-orange-600 border-orange-500/20';
      case 'contraindicated': return 'bg-red-500/10 text-red-600 border-red-500/20';
      default: return 'bg-muted';
    }
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case 'minor': return 'secondary';
      case 'moderate': return 'default';
      case 'major': return 'destructive';
      case 'contraindicated': return 'destructive';
      default: return 'outline';
    }
  };

  const getFrequencyLabel = (value: string) => {
    return frequencyOptions.find(f => f.value === value)?.label || value;
  };

  const getRouteLabel = (value: string) => {
    return routeOptions.find(r => r.value === value)?.label || value;
  };

  const clearPrescription = () => {
    setMedications([]);
    setInteractionResult(null);
    setSelectedMedication("");
    setDosage("");
    setFrequency("");
    setDuration("");
    setRoute("oral");
    setInstructions("");
  };

  return (
    <div className={`grid gap-6 lg:grid-cols-2 ${className}`}>
      <Card data-testid="card-prescription-builder">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Pill className="h-5 w-5" />
            Prescription Builder
          </CardTitle>
          <CardDescription>
            {patientName ? `Creating prescription for ${patientName}` : 'Build prescriptions with safety checks'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {patientAllergies.length > 0 && (
            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="h-4 w-4 text-red-500" />
                <span className="font-medium text-red-600">Patient Allergies</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {patientAllergies.map((allergy, idx) => (
                  <Badge key={idx} variant="destructive">{allergy}</Badge>
                ))}
              </div>
            </div>
          )}

          {currentMedications.length > 0 && (
            <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
              <div className="flex items-center gap-2 mb-2">
                <Pill className="h-4 w-4 text-blue-500" />
                <span className="font-medium text-blue-600">Current Medications</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {currentMedications.map((med, idx) => (
                  <Badge key={idx} variant="secondary">{med}</Badge>
                ))}
              </div>
            </div>
          )}

          <Separator />

          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="medication-search">Medication Name</Label>
              <Input
                id="medication-search"
                placeholder="Enter medication name..."
                value={selectedMedication}
                onChange={(e) => setSelectedMedication(e.target.value)}
                data-testid="input-medication-name"
              />
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="dosage">Dosage</Label>
                <Input
                  id="dosage"
                  placeholder="e.g., 500mg"
                  value={dosage}
                  onChange={(e) => setDosage(e.target.value)}
                  data-testid="input-dosage"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="duration">Duration</Label>
                <Input
                  id="duration"
                  placeholder="e.g., 7 days"
                  value={duration}
                  onChange={(e) => setDuration(e.target.value)}
                  data-testid="input-duration"
                />
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="frequency">Frequency</Label>
                <Select value={frequency} onValueChange={setFrequency}>
                  <SelectTrigger data-testid="select-frequency">
                    <SelectValue placeholder="Select frequency" />
                  </SelectTrigger>
                  <SelectContent>
                    {frequencyOptions.map(opt => (
                      <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="route">Route</Label>
                <Select value={route} onValueChange={setRoute}>
                  <SelectTrigger data-testid="select-route">
                    <SelectValue placeholder="Select route" />
                  </SelectTrigger>
                  <SelectContent>
                    {routeOptions.map(opt => (
                      <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="instructions">Special Instructions</Label>
              <Textarea
                id="instructions"
                placeholder="e.g., Take with food, avoid alcohol..."
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                className="min-h-[60px]"
                data-testid="textarea-instructions"
              />
            </div>

            <Button 
              onClick={addMedication} 
              disabled={!selectedMedication || !dosage || !frequency}
              className="w-full"
              data-testid="button-add-medication"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add to Prescription
            </Button>
          </div>

          {medications.length > 0 && (
            <>
              <Separator />
              <div className="space-y-3">
                <Label>Prescription Items</Label>
                <ScrollArea className="h-[200px]">
                  <div className="space-y-2">
                    {medications.map((med, index) => (
                      <div
                        key={index}
                        className="p-3 rounded-lg border bg-card"
                        data-testid={`prescription-item-${index}`}
                      >
                        <div className="flex items-start justify-between">
                          <div>
                            <div className="flex items-center gap-2">
                              <Pill className="h-4 w-4 text-primary" />
                              <span className="font-medium">{med.name}</span>
                              <Badge variant="outline">{med.dosage}</Badge>
                            </div>
                            <div className="text-sm text-muted-foreground mt-1">
                              <span>{getFrequencyLabel(med.frequency)}</span>
                              <span className="mx-2">·</span>
                              <span>{getRouteLabel(med.route)}</span>
                              <span className="mx-2">·</span>
                              <span>{med.duration}</span>
                            </div>
                            {med.instructions && med.instructions !== "Take as directed" && (
                              <p className="text-xs text-muted-foreground mt-1 italic">
                                {med.instructions}
                              </p>
                            )}
                          </div>
                          <Button
                            size="icon"
                            variant="ghost"
                            onClick={() => removeMedication(index)}
                            data-testid={`button-remove-med-${index}`}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </>
          )}
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button variant="outline" onClick={clearPrescription} data-testid="button-clear-prescription">
            Clear All
          </Button>
          <Button
            onClick={checkInteractions}
            disabled={medications.length === 0 || interactionCheckMutation.isPending}
            data-testid="button-check-interactions"
          >
            {interactionCheckMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Checking...
              </>
            ) : (
              <>
                <Shield className="h-4 w-4 mr-2" />
                Check Interactions
              </>
            )}
          </Button>
        </CardFooter>
      </Card>

      <Card data-testid="card-interaction-results">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Beaker className="h-5 w-5" />
            Safety Analysis
          </CardTitle>
          <CardDescription>
            Cross-specialty drug interactions and safety checks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[500px] pr-4">
            {interactionCheckMutation.isPending ? (
              <div className="flex flex-col items-center justify-center py-16">
                <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
                <p className="text-muted-foreground">Analyzing drug interactions...</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Checking against known interactions and patient profile
                </p>
              </div>
            ) : interactionResult ? (
              <div className="space-y-6">
                <div className={`p-4 rounded-lg ${
                  interactionResult.safeToPresrcibe 
                    ? 'bg-green-500/10 border border-green-500/20' 
                    : 'bg-red-500/10 border border-red-500/20'
                }`}>
                  <div className="flex items-center gap-2">
                    {interactionResult.safeToPresrcibe ? (
                      <>
                        <CheckCircle className="h-5 w-5 text-green-500" />
                        <span className="font-semibold text-green-600">No Critical Issues Found</span>
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="h-5 w-5 text-red-500" />
                        <span className="font-semibold text-red-600">Safety Concerns Identified</span>
                      </>
                    )}
                  </div>
                  {/* All-Medications Mode Warning Banner */}
                  {interactionResult._allMedicationsMode && (
                    <div className="mt-3 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-yellow-600" />
                        <span className="font-medium text-yellow-700 text-sm">All-Medications Mode</span>
                      </div>
                      <p className="text-xs text-yellow-600 mt-1">
                        Cross-specialty filtering bypassed. Checked against ALL {interactionResult.allMedicationsCount || 0} patient medications.
                      </p>
                    </div>
                  )}
                  
                  {/* Fallback Warning */}
                  {interactionResult._fallback && (
                    <div className="mt-3 p-3 rounded-lg bg-orange-500/10 border border-orange-500/20">
                      <div className="flex items-center gap-2">
                        <AlertCircle className="h-4 w-4 text-orange-600" />
                        <span className="font-medium text-orange-700 text-sm">AI Analysis Unavailable</span>
                      </div>
                      <p className="text-xs text-orange-600 mt-1">Please verify drug interactions manually using clinical references.</p>
                    </div>
                  )}

                  {/* Cross-specialty context */}
                  {(interactionResult.crossSpecialtyCount !== undefined || interactionResult._note) && !interactionResult._allMedicationsMode && (
                    <div className="mt-3 pt-3 border-t border-current/10">
                      <div className="flex items-center gap-4 text-sm">
                        {interactionResult.crossSpecialtyCount !== undefined && (
                          <div className="flex items-center gap-1">
                            <Info className="h-3 w-3" />
                            <span>{interactionResult.crossSpecialtyCount} cross-specialty med{interactionResult.crossSpecialtyCount !== 1 ? 's' : ''} checked</span>
                          </div>
                        )}
                        {interactionResult.sameSpecialtyCount !== undefined && interactionResult.sameSpecialtyCount > 0 && (
                          <div className="flex items-center gap-1 text-muted-foreground">
                            <span>{interactionResult.sameSpecialtyCount} same-specialty (supersession rules)</span>
                          </div>
                        )}
                      </div>
                      {interactionResult._note && (
                        <p className="text-xs text-muted-foreground mt-2">{interactionResult._note}</p>
                      )}
                    </div>
                  )}
                </div>

                {interactionResult.contraindications.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="h-5 w-5 text-red-500" />
                      <span className="font-semibold">Contraindications</span>
                    </div>
                    <ul className="space-y-2">
                      {interactionResult.contraindications.map((item, idx) => (
                        <li key={idx} className="flex items-start gap-2 p-2 rounded-lg bg-red-500/5 text-sm">
                          <X className="h-4 w-4 text-red-500 mt-0.5" />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {interactionResult.allergicRisks.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Heart className="h-5 w-5 text-red-500" />
                      <span className="font-semibold">Allergy Alerts</span>
                    </div>
                    <ul className="space-y-2">
                      {interactionResult.allergicRisks.map((risk, idx) => (
                        <li key={idx} className="flex items-start gap-2 p-2 rounded-lg bg-red-500/5 text-sm">
                          <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5" />
                          <span>{risk}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {interactionResult.interactions.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Beaker className="h-5 w-5 text-orange-500" />
                      <span className="font-semibold">Drug Interactions</span>
                    </div>
                    <div className="space-y-3">
                      {interactionResult.interactions.map((interaction, idx) => (
                        <div 
                          key={idx} 
                          className={`p-3 rounded-lg border ${getSeverityColor(interaction.severity)}`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Badge variant={getSeverityBadge(interaction.severity) as any}>
                                {interaction.severity}
                              </Badge>
                              <span className="font-medium">
                                {interaction.drug1} + {interaction.drug2}
                              </span>
                            </div>
                          </div>
                          <p className="text-sm mb-2">{interaction.description}</p>
                          <div className="text-sm">
                            <span className="font-medium">Clinical Effect: </span>
                            <span className="text-muted-foreground">{interaction.clinicalEffect}</span>
                          </div>
                          <div className="text-sm mt-1">
                            <span className="font-medium">Recommendation: </span>
                            <span className="text-muted-foreground">{interaction.recommendation}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {interactionResult.warnings.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Info className="h-5 w-5 text-yellow-500" />
                      <span className="font-semibold">Warnings</span>
                    </div>
                    <ul className="space-y-2">
                      {interactionResult.warnings.map((warning, idx) => (
                        <li key={idx} className="flex items-start gap-2 p-2 rounded-lg bg-yellow-500/5 text-sm">
                          <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5" />
                          <span>{warning}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {!interactionResult.hasInteractions && 
                 interactionResult.allergicRisks.length === 0 && 
                 interactionResult.contraindications.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500 opacity-50" />
                    <p className="font-medium">All Clear</p>
                    <p className="text-sm mt-1">No drug interactions or safety concerns detected</p>
                  </div>
                )}

                <div className="p-3 rounded-lg bg-muted/50 text-xs text-muted-foreground">
                  <Info className="h-4 w-4 inline mr-1" />
                  This analysis is for clinical decision support. Always verify prescriptions against complete patient records and current clinical guidelines.
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-center text-muted-foreground">
                <Shield className="h-16 w-16 mb-4 opacity-30" />
                <p className="text-lg font-medium mb-2">Safety Check Ready</p>
                <p className="text-sm max-w-md">
                  Add medications to the prescription and click "Check Interactions" to analyze drug interactions, allergies, and contraindications.
                </p>
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

export default PrescriptionHelper;
