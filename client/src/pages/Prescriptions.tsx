import { useState, useMemo } from "react";
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
} from "lucide-react";
import { format } from "date-fns";
import type { User as UserType, Prescription, Drug } from "@shared/schema";

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

export default function Prescriptions() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("write");
  const [selectedPatient, setSelectedPatient] = useState<string>("");
  const [drugSearchQuery, setDrugSearchQuery] = useState("");
  const [selectedDrug, setSelectedDrug] = useState<Drug | null>(null);
  const [showInteractionDialog, setShowInteractionDialog] = useState(false);
  const [interactionResult, setInteractionResult] = useState<DrugInteractionCheck | null>(null);
  
  const [prescriptionForm, setPrescriptionForm] = useState({
    medicationName: "",
    dosage: "",
    frequency: "",
    quantity: "",
    refills: "0",
    dosageInstructions: "",
    notes: "",
    startDate: format(new Date(), "yyyy-MM-dd"),
  });

  const { data: patients, isLoading: patientsLoading } = useQuery<UserType[]>({
    queryKey: ["/api/doctor/patients"],
  });

  const { data: prescriptions, isLoading: prescriptionsLoading } = useQuery<Prescription[]>({
    queryKey: ["/api/prescriptions/doctor"],
    queryFn: async () => {
      const res = await fetch("/api/prescriptions/doctor");
      if (!res.ok) throw new Error("Failed to fetch prescriptions");
      return res.json();
    },
  });

  const { data: drugSearchResults, isLoading: drugsLoading } = useQuery<Drug[]>({
    queryKey: ["/api/drugs/search", drugSearchQuery],
    queryFn: async () => {
      if (drugSearchQuery.length < 2) return [];
      const res = await fetch(`/api/drugs/search?q=${encodeURIComponent(drugSearchQuery)}`);
      if (!res.ok) throw new Error("Failed to search drugs");
      return res.json();
    },
    enabled: drugSearchQuery.length >= 2,
  });

  const { data: patientMedications } = useQuery<PatientMedication[]>({
    queryKey: ["/api/medications", selectedPatient],
    queryFn: async () => {
      if (!selectedPatient) return [];
      const res = await fetch(`/api/medications?patientId=${selectedPatient}`);
      if (!res.ok) return [];
      return res.json();
    },
    enabled: !!selectedPatient,
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
      });
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Prescription Created",
        description: "The prescription has been sent to the patient",
      });
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
    });
    setInteractionResult(null);
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

  const selectedPatientData = patients?.find(p => p.id === selectedPatient);

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
          {prescriptions?.length || 0} Total Prescriptions
        </Badge>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="write" data-testid="tab-write">
            <Plus className="h-4 w-4 mr-2" />
            Write Prescription
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

        <TabsContent value="history" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Prescription History</CardTitle>
              <CardDescription>
                All prescriptions you have written
              </CardDescription>
            </CardHeader>
            <CardContent>
              {prescriptionsLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-24 w-full" />
                  ))}
                </div>
              ) : prescriptions && prescriptions.length > 0 ? (
                <div className="space-y-4">
                  {prescriptions.map((rx) => (
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
