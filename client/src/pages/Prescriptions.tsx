import { useState, useMemo, useEffect } from "react";
import { useLocation } from "wouter";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useAuth } from "@/hooks/useAuth";
import {
  Pill,
  Search,
  User,
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileText,
  Plus,
  Send,
  RefreshCw,
  ChevronRight,
  Calendar,
  Package,
  Activity,
  X,
  Edit3,
  MessageSquare,
  Infinity,
  Timer,
  ArrowRightLeft,
  Stethoscope,
  Shield,
  Ban,
  ClipboardList,
  Brain,
  Loader2,
  Save,
  Trash2,
} from "lucide-react";
import { format, addDays } from "date-fns";
import type { User as UserType, Prescription, Drug, DosageChangeRequest } from "@shared/schema";

interface DrugInteractionCheck {
  hasInteractions: boolean;
  severity: string | null;
  interactions: Array<{
    drug1: string;
    drug2: string;
    severity: string;
    description: string;
    recommendation: string;
  }>;
}

interface PatientMedication {
  id: string;
  name: string;
  dosage: string;
  frequency: string;
  status: string;
}

interface SOAPNote {
  id: string;
  patientId: string;
  doctorId: string;
  encounterDate: string;
  chiefComplaint?: string;
  subjective?: string;
  historyPresentIllness?: string;
  reviewOfSystems?: Record<string, any>;
  objective?: string;
  vitalSigns?: Record<string, any>;
  physicalExam?: string;
  labResults?: Array<Record<string, any>>;
  assessment?: string;
  primaryDiagnosis?: string;
  primaryIcd10?: string;
  secondaryDiagnoses?: Array<{ diagnosis: string; icd10: string }>;
  differentialDiagnoses?: string[];
  plan?: string;
  medicationsPrescribed?: Array<Record<string, any>>;
  proceduresOrdered?: string[];
  referrals?: string[];
  patientEducation?: string;
  followUpInstructions?: string;
  followUpDate?: string;
  linkedAppointmentId?: string;
  status: string;
  createdAt: string;
  updatedAt: string;
}

interface ICD10Suggestion {
  code: string;
  description: string;
  confidence: number;
  category?: string;
}

export default function Prescriptions() {
  const { user } = useAuth();
  const { toast } = useToast();
  const isDoctor = user?.role === "doctor";
  
  const [activeTab, setActiveTab] = useState(isDoctor ? "write" : "active");
  const [selectedPatient, setSelectedPatient] = useState<string>("");
  const [drugSearchQuery, setDrugSearchQuery] = useState("");
  const [selectedDrug, setSelectedDrug] = useState<Drug | null>(null);
  const [showInteractionDialog, setShowInteractionDialog] = useState(false);
  const [interactionResult, setInteractionResult] = useState<DrugInteractionCheck | null>(null);
  const [showDosageRequestDialog, setShowDosageRequestDialog] = useState(false);
  const [selectedMedicationForChange, setSelectedMedicationForChange] = useState<any>(null);
  const [dosageChangeForm, setDosageChangeForm] = useState({
    requestedDosage: "",
    requestedFrequency: "",
    reason: "",
  });
  
  const [prescriptionForm, setPrescriptionForm] = useState({
    medicationName: "",
    dosage: "",
    frequency: "",
    quantity: "",
    refills: "0",
    dosageInstructions: "",
    notes: "",
    startDate: format(new Date(), "yyyy-MM-dd"),
    specialty: "",
    isContinuous: false,
    durationDays: "",
    supersedes: "",
  });

  const [showSupersessionDialog, setShowSupersessionDialog] = useState(false);
  const [supersessionCandidates, setSupersessionCandidates] = useState<any[]>([]);
  const [conflictResponse, setConflictResponse] = useState<{ id: string; response: string; notes: string } | null>(null);

  // SOAP Notes state
  const [soapNotePatient, setSoapNotePatient] = useState<string>("");
  const [selectedSoapNote, setSelectedSoapNote] = useState<SOAPNote | null>(null);
  const [icd10Suggestions, setIcd10Suggestions] = useState<ICD10Suggestion[]>([]);
  const [icd10Loading, setIcd10Loading] = useState(false);
  const [soapForm, setSoapForm] = useState({
    chiefComplaint: "",
    subjective: "",
    historyPresentIllness: "",
    objective: "",
    physicalExam: "",
    assessment: "",
    primaryDiagnosis: "",
    primaryIcd10: "",
    plan: "",
    followUpInstructions: "",
  });

  const { data: patients, isLoading: patientsLoading } = useQuery<UserType[]>({
    queryKey: ["/api/doctor/patients"],
    enabled: isDoctor,
  });

  const { data: doctorPrescriptions, isLoading: doctorPrescriptionsLoading } = useQuery<Prescription[]>({
    queryKey: ["/api/prescriptions/doctor"],
    queryFn: async () => {
      const res = await fetch("/api/prescriptions/doctor");
      if (!res.ok) throw new Error("Failed to fetch prescriptions");
      return res.json();
    },
    enabled: isDoctor,
  });

  const { data: patientPrescriptions, isLoading: patientPrescriptionsLoading } = useQuery<Prescription[]>({
    queryKey: ["/api/prescriptions"],
    enabled: !isDoctor,
  });

  const { data: myDosageRequests } = useQuery<DosageChangeRequest[]>({
    queryKey: ["/api/dosage-change-requests"],
    enabled: !isDoctor,
  });

  const { data: drugSearchResults, isLoading: drugsLoading } = useQuery<Drug[]>({
    queryKey: ["/api/drugs/search", drugSearchQuery],
    queryFn: async () => {
      if (drugSearchQuery.length < 2) return [];
      const res = await fetch(`/api/drugs/search?q=${encodeURIComponent(drugSearchQuery)}`);
      if (!res.ok) throw new Error("Failed to search drugs");
      return res.json();
    },
    enabled: drugSearchQuery.length >= 2 && isDoctor,
  });

  const { data: patientMedications } = useQuery<PatientMedication[]>({
    queryKey: ["/api/medications", selectedPatient],
    queryFn: async () => {
      if (!selectedPatient) return [];
      const res = await fetch(`/api/medications?patientId=${selectedPatient}`);
      if (!res.ok) return [];
      return res.json();
    },
    enabled: isDoctor && !!selectedPatient,
  });

  interface MedicationConflict {
    id: string;
    patientId: string;
    medication1Id: string;
    medication2Id: string;
    doctor1Id: string;
    doctor2Id: string;
    specialty1: string;
    specialty2: string;
    conflictType: string;
    severity: string;
    description: string;
    status: 'pending' | 'resolved';
    doctor1Response?: string;
    doctor2Response?: string;
    resolution?: string;
    createdAt: string;
    medication1Name?: string;
    medication2Name?: string;
    patientName?: string;
  }

  const { data: pendingConflicts, isLoading: conflictsLoading } = useQuery<MedicationConflict[]>({
    queryKey: ["/api/doctor/medication-conflicts/pending"],
    enabled: isDoctor,
  });

  // SOAP Notes queries
  const { data: soapNotes, isLoading: soapNotesLoading } = useQuery<SOAPNote[]>({
    queryKey: ["/api/v1/rx-builder/soap-notes", soapNotePatient],
    queryFn: async () => {
      if (!soapNotePatient) return [];
      const res = await fetch(`/api/v1/rx-builder/soap-notes?patient_id=${soapNotePatient}`);
      if (!res.ok) return [];
      return res.json();
    },
    enabled: isDoctor && !!soapNotePatient,
  });

  const createSoapNoteMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/v1/rx-builder/soap-notes", {
        patient_id: soapNotePatient,
        encounter_date: new Date().toISOString(),
        chief_complaint: soapForm.chiefComplaint,
        subjective: soapForm.subjective,
        history_present_illness: soapForm.historyPresentIllness,
        objective: soapForm.objective,
        physical_exam: soapForm.physicalExam,
        assessment: soapForm.assessment,
        primary_diagnosis: soapForm.primaryDiagnosis,
        primary_icd10: soapForm.primaryIcd10,
        plan: soapForm.plan,
        follow_up_instructions: soapForm.followUpInstructions,
      });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/rx-builder/soap-notes", soapNotePatient] });
      resetSoapForm();
      toast({
        title: "SOAP Note Created",
        description: "Clinical documentation saved successfully",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to create SOAP note",
        variant: "destructive",
      });
    },
  });

  const getIcd10SuggestionsMutation = useMutation({
    mutationFn: async (symptoms: string) => {
      setIcd10Loading(true);
      const res = await apiRequest("POST", "/api/v1/rx-builder/icd10/suggest", {
        symptoms,
        include_related: true,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setIcd10Suggestions(data.suggestions || []);
      setIcd10Loading(false);
    },
    onError: () => {
      setIcd10Loading(false);
      toast({
        title: "Error",
        description: "Failed to get ICD-10 suggestions",
        variant: "destructive",
      });
    },
  });

  const resetSoapForm = () => {
    setSoapForm({
      chiefComplaint: "",
      subjective: "",
      historyPresentIllness: "",
      objective: "",
      physicalExam: "",
      assessment: "",
      primaryDiagnosis: "",
      primaryIcd10: "",
      plan: "",
      followUpInstructions: "",
    });
    setSelectedSoapNote(null);
    setIcd10Suggestions([]);
  };

  const handleGetIcd10Suggestions = () => {
    const symptoms = [soapForm.chiefComplaint, soapForm.subjective, soapForm.assessment]
      .filter(Boolean)
      .join(". ");
    if (symptoms.length > 5) {
      getIcd10SuggestionsMutation.mutate(symptoms);
    }
  };

  const handleSelectIcd10 = (suggestion: ICD10Suggestion) => {
    setSoapForm(prev => ({
      ...prev,
      primaryDiagnosis: suggestion.description,
      primaryIcd10: suggestion.code,
    }));
  };

  const respondToConflictMutation = useMutation({
    mutationFn: async ({ conflictId, response, notes }: { conflictId: string; response: string; notes: string }) => {
      const res = await apiRequest("POST", `/api/doctor/medication-conflicts/${conflictId}/respond`, {
        response,
        notes,
      });
      return res.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["/api/doctor/medication-conflicts/pending"] });
      setConflictResponse(null);
      if (data.resolved) {
        toast({
          title: "Conflict Resolved",
          description: "Both doctors agreed. The medications are now active.",
        });
      } else {
        toast({
          title: "Response Recorded",
          description: "Waiting for the other doctor to respond.",
        });
      }
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to submit response",
        variant: "destructive",
      });
    },
  });

  const checkSupersessionMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("GET", `/api/medications/supersession-check?patientId=${selectedPatient}&specialty=${prescriptionForm.specialty}`, {});
      return res.json();
    },
    onSuccess: (data) => {
      if (data.supersessionCandidates && data.supersessionCandidates.length > 0) {
        setSupersessionCandidates(data.supersessionCandidates);
        setShowSupersessionDialog(true);
      }
    },
  });

  const { data: myMedications } = useQuery<PatientMedication[]>({
    queryKey: ["/api/medications"],
    enabled: !isDoctor,
  });

  const checkInteractionsMutation = useMutation({
    mutationFn: async (drugName: string) => {
      const res = await apiRequest("POST", "/api/drug-interactions/analyze", {
        drugName,
        patientId: selectedPatient,
        createAlerts: false,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setInteractionResult(data);
      if (data.hasInteractions) {
        setShowInteractionDialog(true);
      }
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to check drug interactions",
        variant: "destructive",
      });
    },
  });

  const createPrescriptionMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/prescriptions", {
        patientId: selectedPatient,
        drugId: selectedDrug?.id,
        rxcui: selectedDrug?.rxcui,
        medicationName: prescriptionForm.medicationName,
        dosage: prescriptionForm.dosage,
        frequency: prescriptionForm.frequency,
        quantity: parseInt(prescriptionForm.quantity) || null,
        refills: parseInt(prescriptionForm.refills) || 0,
        dosageInstructions: prescriptionForm.dosageInstructions || null,
        notes: prescriptionForm.notes || null,
        startDate: new Date(prescriptionForm.startDate),
        specialty: prescriptionForm.specialty || null,
        isContinuous: prescriptionForm.isContinuous,
        durationDays: prescriptionForm.isContinuous ? null : (parseInt(prescriptionForm.durationDays) || null),
        intendedStartDate: prescriptionForm.startDate ? new Date(prescriptionForm.startDate) : null,
        supersedes: prescriptionForm.supersedes || null,
      });
      return res.json();
    },
    onSuccess: (data) => {
      if (data.conflictDetected) {
        toast({
          title: "Prescription Created with Conflict Warning",
          description: `Cross-specialty conflict detected: ${data.conflictDetected.description}. The medication is on hold until resolved.`,
          variant: "destructive",
        });
      } else if (data.supersessionTarget) {
        toast({
          title: "Prescription Created",
          description: "The previous medication has been superseded by this new prescription.",
        });
      } else {
        toast({
          title: "Prescription Created",
          description: "The prescription has been sent to the patient",
        });
      }
      queryClient.invalidateQueries({ queryKey: ["/api/prescriptions/doctor"] });
      resetForm();
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to create prescription",
        variant: "destructive",
      });
    },
  });

  const acknowledgePrescriptionMutation = useMutation({
    mutationFn: async (prescriptionId: string) => {
      const res = await apiRequest("POST", `/api/prescriptions/${prescriptionId}/acknowledge`, {});
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Prescription Acknowledged",
        description: "You have acknowledged receiving this prescription",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/prescriptions"] });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to acknowledge prescription",
        variant: "destructive",
      });
    },
  });

  const createDosageRequestMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/dosage-change-requests", {
        medicationId: selectedMedicationForChange?.id,
        currentDosage: selectedMedicationForChange?.dosage,
        currentFrequency: selectedMedicationForChange?.frequency,
        requestedDosage: dosageChangeForm.requestedDosage,
        requestedFrequency: dosageChangeForm.requestedFrequency,
        requestReason: dosageChangeForm.reason,
      });
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Request Submitted",
        description: "Your dosage change request has been sent to your doctor for review",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/dosage-change-requests"] });
      setShowDosageRequestDialog(false);
      resetDosageForm();
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to submit dosage change request",
        variant: "destructive",
      });
    },
  });

  const resetForm = () => {
    setSelectedPatient("");
    setSelectedDrug(null);
    setDrugSearchQuery("");
    setPrescriptionForm({
      medicationName: "",
      dosage: "",
      frequency: "",
      quantity: "",
      refills: "0",
      dosageInstructions: "",
      notes: "",
      startDate: format(new Date(), "yyyy-MM-dd"),
      specialty: "",
      isContinuous: false,
      durationDays: "",
      supersedes: "",
    });
    setInteractionResult(null);
    setSupersessionCandidates([]);
  };

  const MEDICAL_SPECIALTIES = [
    "cardiology",
    "oncology",
    "neurology",
    "rheumatology",
    "immunology",
    "endocrinology",
    "gastroenterology",
    "pulmonology",
    "nephrology",
    "psychiatry",
    "general medicine",
  ];

  const resetDosageForm = () => {
    setSelectedMedicationForChange(null);
    setDosageChangeForm({
      requestedDosage: "",
      requestedFrequency: "",
      reason: "",
    });
  };

  const handleDrugSelect = (drug: Drug) => {
    setSelectedDrug(drug);
    setPrescriptionForm({
      ...prescriptionForm,
      medicationName: drug.brandName || drug.genericName || "",
    });
    setDrugSearchQuery("");
    
    if (selectedPatient) {
      checkInteractionsMutation.mutate(drug.brandName || drug.genericName || "");
    }
  };

  const handleSubmit = () => {
    if (!selectedPatient || !prescriptionForm.medicationName || !prescriptionForm.dosage || !prescriptionForm.frequency) {
      toast({
        title: "Missing Information",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }
    createPrescriptionMutation.mutate();
  };

  const handleRequestDosageChange = (medication: any) => {
    setSelectedMedicationForChange(medication);
    setDosageChangeForm({
      requestedDosage: medication.dosage,
      requestedFrequency: medication.frequency,
      reason: "",
    });
    setShowDosageRequestDialog(true);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "sent":
        return <Badge variant="secondary"><Clock className="h-3 w-3 mr-1" />Pending</Badge>;
      case "acknowledged":
        return <Badge variant="default"><CheckCircle2 className="h-3 w-3 mr-1" />Acknowledged</Badge>;
      case "filled":
        return <Badge className="bg-teal-500"><Package className="h-3 w-3 mr-1" />Filled</Badge>;
      case "expired":
        return <Badge variant="outline"><X className="h-3 w-3 mr-1" />Expired</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getRequestStatusBadge = (status: string) => {
    switch (status) {
      case "pending":
        return <Badge variant="secondary"><Clock className="h-3 w-3 mr-1" />Pending</Badge>;
      case "approved":
        return <Badge className="bg-teal-500"><CheckCircle2 className="h-3 w-3 mr-1" />Approved</Badge>;
      case "rejected":
        return <Badge variant="destructive"><X className="h-3 w-3 mr-1" />Rejected</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const selectedPatientData = patients?.find(p => p.id === selectedPatient);
  const [, setLocation] = useLocation();

  useEffect(() => {
    if (!isDoctor) {
      setLocation("/patient-records");
    }
  }, [isDoctor, setLocation]);

  if (!isDoctor) {
    return null;
  }


  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight" data-testid="text-page-title">
            Prescription Management
          </h1>
          <p className="text-muted-foreground mt-1">
            Write new prescriptions and manage patient medications
          </p>
        </div>
        <Badge variant="outline" className="text-sm">
          <FileText className="h-3 w-3 mr-1" />
          {doctorPrescriptions?.length || 0} Total Prescriptions
        </Badge>
      </div>

      {pendingConflicts && pendingConflicts.length > 0 && (
        <Alert variant="destructive" className="border-2" data-testid="alert-pending-conflicts">
          <Ban className="h-5 w-5" />
          <AlertTitle className="text-lg">Cross-Specialty Conflicts Require Your Attention</AlertTitle>
          <AlertDescription className="mt-2">
            <p className="mb-3">
              The following medication conflicts are on hold waiting for your response. Both prescribing doctors must respond before medications can be released.
            </p>
            <div className="space-y-3">
              {pendingConflicts.map(conflict => (
                <div key={conflict.id} className="p-4 bg-background rounded-md border">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant="outline">{conflict.specialty1}</Badge>
                        <span className="text-muted-foreground">vs</span>
                        <Badge variant="outline">{conflict.specialty2}</Badge>
                        <Badge variant={conflict.severity === 'severe' ? 'destructive' : 'secondary'}>
                          {conflict.severity}
                        </Badge>
                      </div>
                      <p className="text-sm font-medium mb-1">
                        {conflict.medication1Name || 'Medication 1'} + {conflict.medication2Name || 'Medication 2'}
                      </p>
                      <p className="text-sm text-muted-foreground">{conflict.description}</p>
                      {conflict.patientName && (
                        <p className="text-xs text-muted-foreground mt-1">Patient: {conflict.patientName}</p>
                      )}
                    </div>
                    <div className="flex flex-col gap-2">
                      {conflictResponse?.id === conflict.id ? (
                        <div className="space-y-2 min-w-[200px]">
                          <RadioGroup
                            value={conflictResponse.response}
                            onValueChange={(val) => setConflictResponse({ ...conflictResponse, response: val })}
                          >
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="approve" id={`approve-${conflict.id}`} />
                              <Label htmlFor={`approve-${conflict.id}`} className="text-sm cursor-pointer">Approve my medication</Label>
                            </div>
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="withdraw" id={`withdraw-${conflict.id}`} />
                              <Label htmlFor={`withdraw-${conflict.id}`} className="text-sm cursor-pointer">Withdraw my prescription</Label>
                            </div>
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="modify" id={`modify-${conflict.id}`} />
                              <Label htmlFor={`modify-${conflict.id}`} className="text-sm cursor-pointer">Suggest modification</Label>
                            </div>
                          </RadioGroup>
                          <Input
                            placeholder="Notes (optional)"
                            value={conflictResponse.notes}
                            onChange={(e) => setConflictResponse({ ...conflictResponse, notes: e.target.value })}
                            className="text-sm"
                          />
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => setConflictResponse(null)}
                            >
                              Cancel
                            </Button>
                            <Button
                              size="sm"
                              onClick={() => respondToConflictMutation.mutate({
                                conflictId: conflict.id,
                                response: conflictResponse.response,
                                notes: conflictResponse.notes,
                              })}
                              disabled={!conflictResponse.response || respondToConflictMutation.isPending}
                            >
                              Submit
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <Button
                          size="sm"
                          onClick={() => setConflictResponse({ id: conflict.id, response: '', notes: '' })}
                          data-testid={`button-respond-conflict-${conflict.id}`}
                        >
                          <Shield className="h-4 w-4 mr-1" />
                          Respond
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="write" data-testid="tab-write">
            <Plus className="h-4 w-4 mr-2" />
            Write Prescription
          </TabsTrigger>
          <TabsTrigger value="soap-notes" data-testid="tab-soap-notes">
            <ClipboardList className="h-4 w-4 mr-2" />
            SOAP Notes
          </TabsTrigger>
          <TabsTrigger value="history" data-testid="tab-history">
            <FileText className="h-4 w-4 mr-2" />
            Prescription History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="write" className="space-y-6 mt-6">
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <User className="h-5 w-5" />
                    Select Patient
                  </CardTitle>
                  <CardDescription>
                    Choose the patient to prescribe medication for
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {patientsLoading ? (
                    <Skeleton className="h-10 w-full" />
                  ) : (
                    <Select value={selectedPatient} onValueChange={setSelectedPatient}>
                      <SelectTrigger data-testid="select-patient">
                        <SelectValue placeholder="Select a patient..." />
                      </SelectTrigger>
                      <SelectContent>
                        {patients?.map((patient) => (
                          <SelectItem key={patient.id} value={patient.id}>
                            {patient.firstName} {patient.lastName} ({patient.email})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Pill className="h-5 w-5" />
                    Medication Details
                  </CardTitle>
                  <CardDescription>
                    Search for medications and configure dosage
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="drug-search">Search Medication</Label>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        id="drug-search"
                        placeholder="Search by drug name or RxNorm code..."
                        value={drugSearchQuery}
                        onChange={(e) => setDrugSearchQuery(e.target.value)}
                        className="pl-10"
                        data-testid="input-drug-search"
                      />
                    </div>
                    {drugsLoading && <Skeleton className="h-20 w-full" />}
                    {drugSearchResults && drugSearchResults.length > 0 && (
                      <div className="border rounded-md max-h-48 overflow-y-auto">
                        {drugSearchResults.map((drug) => (
                          <button
                            key={drug.id}
                            className="w-full p-3 text-left hover-elevate border-b last:border-b-0"
                            onClick={() => handleDrugSelect(drug)}
                            data-testid={`drug-option-${drug.id}`}
                          >
                            <div className="font-medium">{drug.brandName || drug.genericName}</div>
                            <div className="text-sm text-muted-foreground">
                              {drug.genericName && drug.brandName && `Generic: ${drug.genericName} • `}
                              {drug.drugClass && `Class: ${drug.drugClass}`}
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {selectedDrug && (
                    <Alert>
                      <Pill className="h-4 w-4" />
                      <AlertTitle>Selected Medication</AlertTitle>
                      <AlertDescription>
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="font-medium">{selectedDrug.brandName || selectedDrug.genericName}</span>
                            {selectedDrug.rxcui && (
                              <Badge variant="outline" className="ml-2">RxCUI: {selectedDrug.rxcui}</Badge>
                            )}
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setSelectedDrug(null)}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      </AlertDescription>
                    </Alert>
                  )}

                  <Card className="mb-4 border-primary/20 bg-primary/5">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base flex items-center gap-2">
                        <Stethoscope className="h-4 w-4" />
                        Chronic Care Settings
                      </CardTitle>
                      <CardDescription>
                        Configure specialty and duration for multi-specialist coordination
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid gap-4 md:grid-cols-2">
                        <div className="space-y-2">
                          <Label htmlFor="specialty">Medical Specialty *</Label>
                          <Select
                            value={prescriptionForm.specialty}
                            onValueChange={(val) => {
                              setPrescriptionForm({ ...prescriptionForm, specialty: val });
                              if (selectedPatient && val) {
                                checkSupersessionMutation.mutate();
                              }
                            }}
                          >
                            <SelectTrigger data-testid="select-specialty">
                              <SelectValue placeholder="Select your specialty..." />
                            </SelectTrigger>
                            <SelectContent>
                              {MEDICAL_SPECIALTIES.map((spec) => (
                                <SelectItem key={spec} value={spec} className="capitalize">
                                  {spec.charAt(0).toUpperCase() + spec.slice(1)}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-2">
                          <Label>Duration Type</Label>
                          <div className="flex items-center gap-4 mt-2">
                            <div className="flex items-center gap-2">
                              <Switch
                                id="continuous"
                                checked={prescriptionForm.isContinuous}
                                onCheckedChange={(checked) =>
                                  setPrescriptionForm({ ...prescriptionForm, isContinuous: checked, durationDays: "" })
                                }
                                data-testid="switch-continuous"
                              />
                              <Label htmlFor="continuous" className="cursor-pointer">
                                {prescriptionForm.isContinuous ? (
                                  <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
                                    <Infinity className="h-4 w-4" /> Continuous
                                  </span>
                                ) : (
                                  <span className="flex items-center gap-1 text-blue-600 dark:text-blue-400">
                                    <Timer className="h-4 w-4" /> Fixed Duration
                                  </span>
                                )}
                              </Label>
                            </div>
                          </div>
                        </div>
                      </div>

                      {!prescriptionForm.isContinuous && (
                        <div className="space-y-2">
                          <Label htmlFor="duration-days">Duration (Days)</Label>
                          <div className="flex items-center gap-2">
                            <Input
                              id="duration-days"
                              type="number"
                              value={prescriptionForm.durationDays}
                              onChange={(e) => setPrescriptionForm({ ...prescriptionForm, durationDays: e.target.value })}
                              placeholder="e.g., 14"
                              className="w-32"
                              data-testid="input-duration-days"
                            />
                            <span className="text-sm text-muted-foreground">days</span>
                            {prescriptionForm.durationDays && prescriptionForm.startDate && (
                              <Badge variant="outline" className="ml-2">
                                Ends: {format(addDays(new Date(prescriptionForm.startDate), parseInt(prescriptionForm.durationDays)), "MMM d, yyyy")}
                              </Badge>
                            )}
                          </div>
                        </div>
                      )}

                      {supersessionCandidates.length > 0 && (
                        <Alert className="border-amber-500 bg-amber-50 dark:bg-amber-950/30">
                          <ArrowRightLeft className="h-4 w-4 text-amber-600" />
                          <AlertTitle className="text-amber-800 dark:text-amber-300">Supersession Available</AlertTitle>
                          <AlertDescription>
                            <p className="text-amber-700 dark:text-amber-400 mb-2">
                              This patient has {supersessionCandidates.length} existing {prescriptionForm.specialty} medication(s). 
                              Would you like this prescription to supersede any of them?
                            </p>
                            <Select
                              value={prescriptionForm.supersedes}
                              onValueChange={(val) => setPrescriptionForm({ ...prescriptionForm, supersedes: val })}
                            >
                              <SelectTrigger data-testid="select-supersedes" className="bg-white dark:bg-gray-900">
                                <SelectValue placeholder="Select medication to supersede (optional)..." />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="">None (Add as new)</SelectItem>
                                {supersessionCandidates.map((med) => (
                                  <SelectItem key={med.id} value={med.id}>
                                    {med.name} ({med.dosage}) - {med.frequency}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </AlertDescription>
                        </Alert>
                      )}
                    </CardContent>
                  </Card>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="medication-name">Medication Name *</Label>
                      <Input
                        id="medication-name"
                        value={prescriptionForm.medicationName}
                        onChange={(e) => setPrescriptionForm({ ...prescriptionForm, medicationName: e.target.value })}
                        placeholder="e.g., Lisinopril"
                        data-testid="input-medication-name"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="dosage">Dosage *</Label>
                      <Input
                        id="dosage"
                        value={prescriptionForm.dosage}
                        onChange={(e) => setPrescriptionForm({ ...prescriptionForm, dosage: e.target.value })}
                        placeholder="e.g., 10mg"
                        data-testid="input-dosage"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="frequency">Frequency *</Label>
                      <Select
                        value={prescriptionForm.frequency}
                        onValueChange={(val) => setPrescriptionForm({ ...prescriptionForm, frequency: val })}
                      >
                        <SelectTrigger data-testid="select-frequency">
                          <SelectValue placeholder="Select frequency..." />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="once daily">Once Daily</SelectItem>
                          <SelectItem value="twice daily">Twice Daily</SelectItem>
                          <SelectItem value="three times daily">Three Times Daily</SelectItem>
                          <SelectItem value="four times daily">Four Times Daily</SelectItem>
                          <SelectItem value="every 4 hours">Every 4 Hours</SelectItem>
                          <SelectItem value="every 6 hours">Every 6 Hours</SelectItem>
                          <SelectItem value="every 8 hours">Every 8 Hours</SelectItem>
                          <SelectItem value="every 12 hours">Every 12 Hours</SelectItem>
                          <SelectItem value="as needed">As Needed (PRN)</SelectItem>
                          <SelectItem value="weekly">Weekly</SelectItem>
                          <SelectItem value="monthly">Monthly</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="quantity">Quantity</Label>
                      <Input
                        id="quantity"
                        type="number"
                        value={prescriptionForm.quantity}
                        onChange={(e) => setPrescriptionForm({ ...prescriptionForm, quantity: e.target.value })}
                        placeholder="e.g., 30"
                        data-testid="input-quantity"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="refills">Refills</Label>
                      <Select
                        value={prescriptionForm.refills}
                        onValueChange={(val) => setPrescriptionForm({ ...prescriptionForm, refills: val })}
                      >
                        <SelectTrigger data-testid="select-refills">
                          <SelectValue placeholder="Number of refills..." />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0">No Refills</SelectItem>
                          <SelectItem value="1">1 Refill</SelectItem>
                          <SelectItem value="2">2 Refills</SelectItem>
                          <SelectItem value="3">3 Refills</SelectItem>
                          <SelectItem value="5">5 Refills</SelectItem>
                          <SelectItem value="11">11 Refills (1 year)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="start-date">Start Date</Label>
                      <Input
                        id="start-date"
                        type="date"
                        value={prescriptionForm.startDate}
                        onChange={(e) => setPrescriptionForm({ ...prescriptionForm, startDate: e.target.value })}
                        data-testid="input-start-date"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="instructions">Dosage Instructions</Label>
                    <Textarea
                      id="instructions"
                      value={prescriptionForm.dosageInstructions}
                      onChange={(e) => setPrescriptionForm({ ...prescriptionForm, dosageInstructions: e.target.value })}
                      placeholder="e.g., Take with food, Avoid alcohol..."
                      data-testid="input-instructions"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="notes">Doctor's Notes</Label>
                    <Textarea
                      id="notes"
                      value={prescriptionForm.notes}
                      onChange={(e) => setPrescriptionForm({ ...prescriptionForm, notes: e.target.value })}
                      placeholder="Additional notes for the prescription..."
                      data-testid="input-notes"
                    />
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between gap-4">
                  <Button variant="outline" onClick={resetForm} data-testid="button-reset">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Reset Form
                  </Button>
                  <div className="flex gap-2">
                    {selectedPatient && prescriptionForm.medicationName && (
                      <Button
                        variant="outline"
                        onClick={() => checkInteractionsMutation.mutate(prescriptionForm.medicationName)}
                        disabled={checkInteractionsMutation.isPending}
                        data-testid="button-check-interactions"
                      >
                        <AlertTriangle className="h-4 w-4 mr-2" />
                        Check Interactions
                      </Button>
                    )}
                    <Button
                      onClick={handleSubmit}
                      disabled={createPrescriptionMutation.isPending || !selectedPatient || !prescriptionForm.medicationName}
                      data-testid="button-submit"
                    >
                      {createPrescriptionMutation.isPending ? (
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Send className="h-4 w-4 mr-2" />
                      )}
                      Send Prescription
                    </Button>
                  </div>
                </CardFooter>
              </Card>
            </div>

            <div className="space-y-6">
              {selectedPatientData && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Patient Info</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <div className="text-sm text-muted-foreground">Name</div>
                      <div className="font-medium">{selectedPatientData.firstName} {selectedPatientData.lastName}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Email</div>
                      <div className="font-medium">{selectedPatientData.email}</div>
                    </div>
                    {selectedPatientData.phoneNumber && (
                      <div>
                        <div className="text-sm text-muted-foreground">Phone</div>
                        <div className="font-medium">{selectedPatientData.phoneNumber}</div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {selectedPatient && patientMedications && patientMedications.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Activity className="h-4 w-4" />
                      Current Medications
                    </CardTitle>
                    <CardDescription>
                      Review for potential interactions
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {patientMedications.slice(0, 5).map((med) => (
                        <div key={med.id} className="p-2 border rounded-md text-sm">
                          <div className="font-medium">{med.name}</div>
                          <div className="text-muted-foreground">{med.dosage} • {med.frequency}</div>
                        </div>
                      ))}
                      {patientMedications.length > 5 && (
                        <div className="text-sm text-muted-foreground text-center">
                          +{patientMedications.length - 5} more medications
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {interactionResult && (
                <Card className={interactionResult.hasInteractions ? "border-amber-500" : "border-teal-500"}>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      {interactionResult.hasInteractions ? (
                        <AlertTriangle className="h-4 w-4 text-amber-500" />
                      ) : (
                        <CheckCircle2 className="h-4 w-4 text-teal-500" />
                      )}
                      Interaction Check
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {interactionResult.hasInteractions ? (
                      <div className="space-y-2">
                        <Badge variant="outline" className="text-amber-600">
                          {interactionResult.interactions.length} interaction(s) found
                        </Badge>
                        <p className="text-sm text-muted-foreground">
                          Review interactions before prescribing
                        </p>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setShowInteractionDialog(true)}
                          data-testid="button-view-interactions"
                        >
                          View Details
                        </Button>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        No significant interactions detected with current medications
                      </p>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="soap-notes" className="space-y-6 mt-6">
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <User className="h-5 w-5" />
                    Select Patient for SOAP Note
                  </CardTitle>
                  <CardDescription>
                    Choose the patient to create clinical documentation for
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {patientsLoading ? (
                    <Skeleton className="h-10 w-full" />
                  ) : (
                    <Select value={soapNotePatient} onValueChange={setSoapNotePatient}>
                      <SelectTrigger data-testid="select-soap-patient">
                        <SelectValue placeholder="Select a patient..." />
                      </SelectTrigger>
                      <SelectContent>
                        {patients?.map((patient) => (
                          <SelectItem key={patient.id} value={patient.id}>
                            {patient.firstName} {patient.lastName} ({patient.email})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </CardContent>
              </Card>

              {soapNotePatient && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <ClipboardList className="h-5 w-5" />
                      SOAP Note Documentation
                    </CardTitle>
                    <CardDescription>
                      Subjective, Objective, Assessment, and Plan
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="chief-complaint">Chief Complaint</Label>
                        <Input
                          id="chief-complaint"
                          placeholder="Patient's primary reason for visit..."
                          value={soapForm.chiefComplaint}
                          onChange={(e) => setSoapForm(prev => ({ ...prev, chiefComplaint: e.target.value }))}
                          data-testid="input-chief-complaint"
                        />
                      </div>

                      <Separator />
                      <h4 className="font-semibold text-sm text-muted-foreground">SUBJECTIVE</h4>

                      <div>
                        <Label htmlFor="subjective">Patient History & Symptoms</Label>
                        <Textarea
                          id="subjective"
                          placeholder="Patient's description of symptoms, history, pain levels..."
                          value={soapForm.subjective}
                          onChange={(e) => setSoapForm(prev => ({ ...prev, subjective: e.target.value }))}
                          className="min-h-[100px]"
                          data-testid="input-subjective"
                        />
                      </div>

                      <div>
                        <Label htmlFor="hpi">History of Present Illness</Label>
                        <Textarea
                          id="hpi"
                          placeholder="Onset, location, duration, character, aggravating/relieving factors..."
                          value={soapForm.historyPresentIllness}
                          onChange={(e) => setSoapForm(prev => ({ ...prev, historyPresentIllness: e.target.value }))}
                          className="min-h-[80px]"
                          data-testid="input-hpi"
                        />
                      </div>

                      <Separator />
                      <h4 className="font-semibold text-sm text-muted-foreground">OBJECTIVE</h4>

                      <div>
                        <Label htmlFor="objective">Clinical Findings</Label>
                        <Textarea
                          id="objective"
                          placeholder="Vital signs, lab results, imaging findings..."
                          value={soapForm.objective}
                          onChange={(e) => setSoapForm(prev => ({ ...prev, objective: e.target.value }))}
                          className="min-h-[80px]"
                          data-testid="input-objective"
                        />
                      </div>

                      <div>
                        <Label htmlFor="physical-exam">Physical Examination</Label>
                        <Textarea
                          id="physical-exam"
                          placeholder="Physical exam findings, observations..."
                          value={soapForm.physicalExam}
                          onChange={(e) => setSoapForm(prev => ({ ...prev, physicalExam: e.target.value }))}
                          className="min-h-[80px]"
                          data-testid="input-physical-exam"
                        />
                      </div>

                      <Separator />
                      <h4 className="font-semibold text-sm text-muted-foreground">ASSESSMENT</h4>

                      <div>
                        <Label htmlFor="assessment">Clinical Assessment</Label>
                        <Textarea
                          id="assessment"
                          placeholder="Clinical reasoning and impressions..."
                          value={soapForm.assessment}
                          onChange={(e) => setSoapForm(prev => ({ ...prev, assessment: e.target.value }))}
                          className="min-h-[80px]"
                          data-testid="input-assessment"
                        />
                      </div>

                      <div className="grid gap-4 md:grid-cols-2">
                        <div>
                          <Label htmlFor="primary-diagnosis">Primary Diagnosis</Label>
                          <Input
                            id="primary-diagnosis"
                            placeholder="Main diagnosis..."
                            value={soapForm.primaryDiagnosis}
                            onChange={(e) => setSoapForm(prev => ({ ...prev, primaryDiagnosis: e.target.value }))}
                            data-testid="input-primary-diagnosis"
                          />
                        </div>
                        <div>
                          <Label htmlFor="primary-icd10">ICD-10 Code</Label>
                          <div className="flex gap-2">
                            <Input
                              id="primary-icd10"
                              placeholder="e.g., J06.9"
                              value={soapForm.primaryIcd10}
                              onChange={(e) => setSoapForm(prev => ({ ...prev, primaryIcd10: e.target.value }))}
                              data-testid="input-primary-icd10"
                            />
                            <Button
                              type="button"
                              variant="outline"
                              size="icon"
                              onClick={handleGetIcd10Suggestions}
                              disabled={icd10Loading}
                              data-testid="button-get-icd10"
                            >
                              {icd10Loading ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Brain className="h-4 w-4" />
                              )}
                            </Button>
                          </div>
                        </div>
                      </div>

                      <Separator />
                      <h4 className="font-semibold text-sm text-muted-foreground">PLAN</h4>

                      <div>
                        <Label htmlFor="plan">Treatment Plan</Label>
                        <Textarea
                          id="plan"
                          placeholder="Medications, procedures, referrals, patient education..."
                          value={soapForm.plan}
                          onChange={(e) => setSoapForm(prev => ({ ...prev, plan: e.target.value }))}
                          className="min-h-[100px]"
                          data-testid="input-plan"
                        />
                      </div>

                      <div>
                        <Label htmlFor="follow-up">Follow-up Instructions</Label>
                        <Textarea
                          id="follow-up"
                          placeholder="When to return, warning signs, self-care..."
                          value={soapForm.followUpInstructions}
                          onChange={(e) => setSoapForm(prev => ({ ...prev, followUpInstructions: e.target.value }))}
                          className="min-h-[60px]"
                          data-testid="input-follow-up"
                        />
                      </div>
                    </div>
                  </CardContent>
                  <CardFooter className="flex justify-between gap-4">
                    <Button
                      variant="outline"
                      onClick={resetSoapForm}
                      data-testid="button-reset-soap"
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Clear Form
                    </Button>
                    <Button
                      onClick={() => createSoapNoteMutation.mutate()}
                      disabled={createSoapNoteMutation.isPending || !soapForm.chiefComplaint}
                      data-testid="button-save-soap"
                    >
                      {createSoapNoteMutation.isPending ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Save className="h-4 w-4 mr-2" />
                      )}
                      Save SOAP Note
                    </Button>
                  </CardFooter>
                </Card>
              )}
            </div>

            <div className="space-y-6">
              {icd10Suggestions.length > 0 && (
                <Card className="border-primary/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Brain className="h-5 w-5 text-primary" />
                      AI ICD-10 Suggestions
                    </CardTitle>
                    <CardDescription>
                      Click to apply a suggested diagnosis code
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 max-h-[400px] overflow-y-auto">
                      {icd10Suggestions.map((suggestion, idx) => (
                        <div
                          key={idx}
                          className="p-3 border rounded-md cursor-pointer hover-elevate"
                          onClick={() => handleSelectIcd10(suggestion)}
                          data-testid={`icd10-suggestion-${idx}`}
                        >
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <Badge variant="outline" className="font-mono">
                              {suggestion.code}
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              {Math.round(suggestion.confidence * 100)}% match
                            </Badge>
                          </div>
                          <p className="text-sm">{suggestion.description}</p>
                          {suggestion.category && (
                            <p className="text-xs text-muted-foreground mt-1">
                              {suggestion.category}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {soapNotePatient && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Previous SOAP Notes</CardTitle>
                    <CardDescription>
                      Recent clinical documentation
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {soapNotesLoading ? (
                      <div className="space-y-2">
                        {[1, 2, 3].map(i => <Skeleton key={i} className="h-16" />)}
                      </div>
                    ) : soapNotes && soapNotes.length > 0 ? (
                      <div className="space-y-2 max-h-[300px] overflow-y-auto">
                        {soapNotes.map((note) => (
                          <div
                            key={note.id}
                            className="p-3 border rounded-md hover-elevate cursor-pointer"
                            onClick={() => {
                              setSelectedSoapNote(note);
                              setSoapForm({
                                chiefComplaint: note.chiefComplaint || "",
                                subjective: note.subjective || "",
                                historyPresentIllness: note.historyPresentIllness || "",
                                objective: note.objective || "",
                                physicalExam: note.physicalExam || "",
                                assessment: note.assessment || "",
                                primaryDiagnosis: note.primaryDiagnosis || "",
                                primaryIcd10: note.primaryIcd10 || "",
                                plan: note.plan || "",
                                followUpInstructions: note.followUpInstructions || "",
                              });
                            }}
                            data-testid={`soap-note-${note.id}`}
                          >
                            <div className="flex items-center justify-between gap-2 mb-1">
                              <span className="text-sm font-medium truncate">
                                {note.chiefComplaint || "No chief complaint"}
                              </span>
                              <Badge variant="outline" className="text-xs shrink-0">
                                {note.status}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Calendar className="h-3 w-3" />
                              {format(new Date(note.encounterDate), "MMM d, yyyy")}
                              {note.primaryIcd10 && (
                                <Badge variant="secondary" className="font-mono text-xs">
                                  {note.primaryIcd10}
                                </Badge>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-6">
                        <ClipboardList className="h-8 w-8 mx-auto text-muted-foreground opacity-50 mb-2" />
                        <p className="text-sm text-muted-foreground">
                          No previous SOAP notes for this patient
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="history" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Prescription History</CardTitle>
              <CardDescription>
                All prescriptions you have written
              </CardDescription>
            </CardHeader>
            <CardContent>
              {doctorPrescriptionsLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-24 w-full" />
                  ))}
                </div>
              ) : doctorPrescriptions && doctorPrescriptions.length > 0 ? (
                <div className="space-y-4">
                  {doctorPrescriptions.map((rx) => (
                    <div
                      key={rx.id}
                      className="p-4 border rounded-lg hover-elevate"
                      data-testid={`prescription-card-${rx.id}`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold">{rx.medicationName}</span>
                            {getStatusBadge(rx.status)}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {rx.dosage} • {rx.frequency}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            Patient ID: {rx.patientId}
                          </div>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground mt-2">
                            <span className="flex items-center gap-1">
                              <Calendar className="h-3 w-3" />
                              Created: {format(new Date(rx.createdAt!), "MMM d, yyyy")}
                            </span>
                            {rx.acknowledgedAt && (
                              <span className="flex items-center gap-1">
                                <CheckCircle2 className="h-3 w-3 text-teal-500" />
                                Acknowledged: {format(new Date(rx.acknowledgedAt), "MMM d, yyyy")}
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="text-right">
                          {rx.quantity && (
                            <div className="text-sm">Qty: {rx.quantity}</div>
                          )}
                          {rx.refills !== null && rx.refills > 0 && (
                            <div className="text-sm text-muted-foreground">
                              {rx.refills} refill(s)
                            </div>
                          )}
                        </div>
                      </div>
                      {rx.notes && (
                        <div className="mt-2 p-2 bg-muted rounded-md text-sm">
                          <span className="font-medium">Notes:</span> {rx.notes}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <FileText className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
                  <h3 className="text-lg font-medium mb-1">No Prescriptions Yet</h3>
                  <p className="text-sm text-muted-foreground">
                    Prescriptions you write will appear here
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Dialog open={showInteractionDialog} onOpenChange={setShowInteractionDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
              Drug Interactions Detected
            </DialogTitle>
            <DialogDescription>
              Review the following potential interactions before prescribing
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {interactionResult?.interactions.map((interaction, idx) => (
              <div key={idx} className="p-4 border rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Badge
                    className={
                      interaction.severity === "severe"
                        ? "bg-rose-500"
                        : interaction.severity === "moderate"
                        ? "bg-amber-500"
                        : "bg-slate-400"
                    }
                  >
                    {interaction.severity?.toUpperCase()}
                  </Badge>
                  <span className="font-medium">
                    {interaction.drug1} + {interaction.drug2}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground mb-2">
                  {interaction.description}
                </p>
                <div className="text-sm bg-muted p-2 rounded">
                  <span className="font-medium">Recommendation:</span> {interaction.recommendation}
                </div>
              </div>
            ))}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowInteractionDialog(false)}>
              Cancel Prescription
            </Button>
            <Button
              variant="default"
              onClick={() => {
                setShowInteractionDialog(false);
                handleSubmit();
              }}
            >
              Proceed with Caution
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
