import { useState, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import {
  Upload,
  FileText,
  Loader2,
  CheckCircle2,
  XCircle,
  Trash2,
  Eye,
  Activity,
  Plus,
  RefreshCw,
  AlertCircle,
  Image,
  Lock,
  Calendar,
  Database,
  Pill,
  ArrowRightLeft,
  Clock,
  Ban,
  Infinity,
  Stethoscope,
  Send,
  Edit3,
  ChevronRight,
  Shield,
  Timer,
  Archive,
  User,
  History,
} from "lucide-react";
import { format, formatDistanceToNow, differenceInDays } from "date-fns";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const EHR_SYSTEMS = [
  { id: "epic", name: "Epic MyChart", description: "Access your Epic health records" },
  { id: "cerner", name: "Oracle Cerner", description: "Connect to Oracle Health records" },
  { id: "athena", name: "Athena Health", description: "Sync with Athena Practice" },
  { id: "eclinicalworks", name: "eClinicalWorks", description: "Link eClinicalWorks EHR" },
  { id: "allscripts", name: "Allscripts", description: "Connect Allscripts system" },
  { id: "advancedmd", name: "AdvancedMD", description: "Sync AdvancedMD records" },
  { id: "meditech", name: "Meditech", description: "Access Meditech data" },
  { id: "nextgen", name: "NextGen Healthcare", description: "Link NextGen EHR" },
  { id: "drchrono", name: "DrChrono", description: "Connect DrChrono platform" },
];

const SPECIALTY_COLORS: Record<string, string> = {
  cardiology: "border-red-500 text-red-700 bg-red-50 dark:bg-red-950/30 dark:text-red-400",
  oncology: "border-purple-500 text-purple-700 bg-purple-50 dark:bg-purple-950/30 dark:text-purple-400",
  neurology: "border-blue-500 text-blue-700 bg-blue-50 dark:bg-blue-950/30 dark:text-blue-400",
  endocrinology: "border-amber-500 text-amber-700 bg-amber-50 dark:bg-amber-950/30 dark:text-amber-400",
  pulmonology: "border-teal-500 text-teal-700 bg-teal-50 dark:bg-teal-950/30 dark:text-teal-400",
  gastroenterology: "border-orange-500 text-orange-700 bg-orange-50 dark:bg-orange-950/30 dark:text-orange-400",
  rheumatology: "border-indigo-500 text-indigo-700 bg-indigo-50 dark:bg-indigo-950/30 dark:text-indigo-400",
  nephrology: "border-cyan-500 text-cyan-700 bg-cyan-50 dark:bg-cyan-950/30 dark:text-cyan-400",
  hematology: "border-rose-500 text-rose-700 bg-rose-50 dark:bg-rose-950/30 dark:text-rose-400",
  infectious_disease: "border-lime-500 text-lime-700 bg-lime-50 dark:bg-lime-950/30 dark:text-lime-400",
  general: "border-gray-500 text-gray-700 bg-gray-50 dark:bg-gray-950/30 dark:text-gray-400",
};

type MedicalDocument = {
  id: string;
  userId: string;
  fileName: string;
  fileType: string;
  fileSize: number;
  fileUrl: string;
  source?: "upload" | "ehr";
  extractedText?: string;
  extractedData?: {
    patientName?: string;
    dateOfBirth?: string;
    diagnosis?: string[];
    medications?: string[];
    labResults?: { test: string; value: string; unit?: string }[];
    vitalSigns?: { type: string; value: string; unit?: string }[];
    allergies?: string[];
    procedures?: string[];
    notes?: string;
    medicalEntities?: Array<{ text: string; type: string; category: string; score: number }>;
    icdCodes?: Array<{ code: string; description: string; score: number }>;
  };
  documentType?: string;
  documentDate?: string;
  processingStatus?: string;
  errorMessage?: string;
  createdAt: string;
  updatedAt: string;
};

type Prescription = {
  id: string;
  patientId: string;
  doctorId: string;
  medicationName: string;
  dosage: string;
  frequency: string;
  quantity?: number;
  refills?: number;
  startDate?: string;
  expirationDate?: string;
  isContinuous?: boolean;
  durationDays?: number;
  intendedStartDate?: string;
  specialty?: string;
  supersedes?: string;
  supersededBy?: string;
  hasConflict?: boolean;
  conflictGroupId?: string;
  status: string;
  notes?: string;
  createdAt: string;
  doctorName?: string;
};

type ArchivedMedication = {
  id: string;
  name: string;
  dosage: string;
  frequency: string;
  specialty?: string;
  status?: string;
  isContinuous?: boolean;
  durationDays?: number;
  supersededBy?: string;
  supersededAt?: string;
  supersessionReason?: string;
  conflictStatus?: string;
  computedEndDate?: string;
  archivedAt?: string;
  archiveReason?: string;
  createdAt?: string;
};

type EHRConnection = {
  id: string;
  ehrSystem: string;
  ehrSystemName: string;
  facilityName: string;
  patientExternalId: string;
  connectionStatus: string;
  lastSyncedAt?: string;
  documentsCount?: number;
};

export default function PatientRecords() {
  const { toast } = useToast();
  const { user } = useAuth();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState<string>("other");
  const [documentDate, setDocumentDate] = useState<string>("");
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [selectedDocument, setSelectedDocument] = useState<MedicalDocument | null>(null);

  const [showAddEHRDialog, setShowAddEHRDialog] = useState(false);
  const [selectedSystem, setSelectedSystem] = useState("");
  const [facilityName, setFacilityName] = useState("");
  const [patientId, setPatientId] = useState("");

  const [showDosageRequestDialog, setShowDosageRequestDialog] = useState(false);
  const [selectedPrescriptionForChange, setSelectedPrescriptionForChange] = useState<Prescription | null>(null);
  const [dosageChangeForm, setDosageChangeForm] = useState({
    requestedDosage: "",
    requestedFrequency: "",
    reason: "",
  });

  const { data: documents = [], isLoading: isLoadingDocs } = useQuery<MedicalDocument[]>({
    queryKey: ["/api/medical-documents"],
  });

  const { data: ehrConnections = [], isLoading: isLoadingEHR } = useQuery<EHRConnection[]>({
    queryKey: ["/api/ehr/connections"],
  });

  const { data: prescriptions = [], isLoading: isLoadingRx } = useQuery<Prescription[]>({
    queryKey: ["/api/prescriptions"],
  });

  const { data: medicationHistory, isLoading: isLoadingHistory, error: historyError } = useQuery<{
    medications: ArchivedMedication[];
    total: number;
    hasMore: boolean;
  }>({
    queryKey: ["/api/medications/history"],
  });

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await fetch("/api/medical-documents/upload", {
        method: "POST",
        body: formData,
        credentials: "include",
      });
      if (!response.ok) throw new Error("Upload failed");
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/medical-documents"] });
      setSelectedFile(null);
      setDocumentType("other");
      setDocumentDate("");
      setUploadProgress(0);
      toast({ title: "Document Uploaded", description: "Your document is being processed with AI analysis." });
    },
    onError: () => {
      setUploadProgress(0);
      toast({ title: "Upload Failed", description: "Please try again.", variant: "destructive" });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (documentId: string) => apiRequest(`/api/medical-documents/${documentId}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/medical-documents"] });
      toast({ title: "Document Deleted", description: "The document has been removed." });
    },
  });

  const addEHRMutation = useMutation({
    mutationFn: async (data: any) => apiRequest("/api/ehr/connections", { method: "POST", body: JSON.stringify(data) }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ehr/connections"] });
      queryClient.invalidateQueries({ queryKey: ["/api/medical-documents"] });
      setShowAddEHRDialog(false);
      setSelectedSystem("");
      setFacilityName("");
      setPatientId("");
      toast({ title: "EHR Connected", description: "Your health records will sync automatically." });
    },
  });

  const syncEHRMutation = useMutation({
    mutationFn: async (id: string) => apiRequest(`/api/ehr/connections/${id}/sync`, { method: "POST" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ehr/connections"] });
      queryClient.invalidateQueries({ queryKey: ["/api/medical-documents"] });
      toast({ title: "Sync Complete", description: "Health records synchronized." });
    },
  });

  const deleteEHRMutation = useMutation({
    mutationFn: async (id: string) => apiRequest(`/api/ehr/connections/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ehr/connections"] });
      toast({ title: "Connection Removed", description: "The EHR connection has been deleted." });
    },
  });

  const createDosageRequestMutation = useMutation({
    mutationFn: async () => {
      if (!selectedPrescriptionForChange) throw new Error("No prescription selected");
      return apiRequest("/api/dosage-change-requests", {
        method: "POST",
        body: JSON.stringify({
          prescriptionId: selectedPrescriptionForChange.id,
          currentDosage: selectedPrescriptionForChange.dosage,
          currentFrequency: selectedPrescriptionForChange.frequency,
          requestedDosage: dosageChangeForm.requestedDosage || selectedPrescriptionForChange.dosage,
          requestedFrequency: dosageChangeForm.requestedFrequency || selectedPrescriptionForChange.frequency,
          reason: dosageChangeForm.reason,
        }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/dosage-change-requests"] });
      setShowDosageRequestDialog(false);
      setSelectedPrescriptionForChange(null);
      setDosageChangeForm({ requestedDosage: "", requestedFrequency: "", reason: "" });
      toast({ title: "Request Submitted", description: "Your doctor will review your dosage change request." });
    },
  });

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 20 * 1024 * 1024) {
        toast({ title: "File Too Large", description: "Maximum file size is 20MB", variant: "destructive" });
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("documentType", documentType);
    if (documentDate) formData.append("documentDate", documentDate);
    setUploadProgress(50);
    uploadMutation.mutate(formData);
  };

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file) {
      if (file.size > 20 * 1024 * 1024) {
        toast({ title: "File Too Large", description: "Maximum file size is 20MB", variant: "destructive" });
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleAddEHRConnection = () => {
    if (!selectedSystem || !facilityName || !patientId) {
      toast({ title: "Missing Information", description: "Please fill in all fields.", variant: "destructive" });
      return;
    }
    const system = EHR_SYSTEMS.find((s) => s.id === selectedSystem);
    addEHRMutation.mutate({
      ehrSystem: selectedSystem,
      ehrSystemName: system?.name || "",
      facilityName,
      patientExternalId: patientId,
      connectionStatus: "pending",
    });
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case "completed": case "connected": case "active": case "filled":
        return <CheckCircle2 className="h-4 w-4 text-primary" />;
      case "processing": case "pending": case "sent":
        return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
      case "failed": case "error": case "conflict_hold":
        return <XCircle className="h-4 w-4 text-destructive" />;
      default:
        return <FileText className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case "completed": case "connected": case "active": case "filled":
        return <Badge variant="default">Active</Badge>;
      case "processing": case "pending": case "sent":
        return <Badge variant="secondary">Pending</Badge>;
      case "failed": case "error":
        return <Badge variant="destructive">Failed</Badge>;
      case "conflict_hold":
        return <Badge variant="destructive">On Hold</Badge>;
      case "superseded":
        return <Badge variant="outline">Superseded</Badge>;
      case "expired":
        return <Badge variant="outline">Expired</Badge>;
      default:
        return <Badge variant="outline">{status || "Unknown"}</Badge>;
    }
  };

  const getSpecialtyColor = (specialty?: string) => SPECIALTY_COLORS[specialty || "general"] || SPECIALTY_COLORS.general;

  const sortedDocuments = useMemo(() => [...documents].sort((a, b) => {
    const dateA = new Date(a.documentDate || a.createdAt).getTime();
    const dateB = new Date(b.documentDate || b.createdAt).getTime();
    return dateB - dateA;
  }), [documents]);

  const uploadedDocs = sortedDocuments.filter(d => !d.source || d.source === "upload");
  const ehrDocs = sortedDocuments.filter(d => d.source === "ehr");
  const imagingDocs = sortedDocuments.filter(d => d.documentType === "imaging" || d.fileType?.includes("image"));

  const activePrescriptions = prescriptions.filter(p => ["sent", "acknowledged", "filled", "active"].includes(p.status));
  const prescriptionsWithConflict = activePrescriptions.filter(p => p.hasConflict);

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2" data-testid="text-page-title">Patient Records</h1>
            <p className="text-muted-foreground">
              All prescriptions, medical documents, and health records in one unified view
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-sm">
              <Pill className="h-3 w-3 mr-1" />
              {activePrescriptions.length} Active Rx
            </Badge>
            <Badge variant="outline" className="text-sm">
              <FileText className="h-3 w-3 mr-1" />
              {documents.length} Documents
            </Badge>
          </div>
        </div>

        {prescriptionsWithConflict.length > 0 && (
          <Alert variant="destructive" className="border-2" data-testid="alert-prescription-conflicts">
            <Ban className="h-5 w-5" />
            <AlertTitle>Prescription Conflicts Detected</AlertTitle>
            <AlertDescription>
              {prescriptionsWithConflict.length} prescription(s) are on hold due to cross-specialty conflicts. 
              Your doctors are reviewing the situation.
            </AlertDescription>
          </Alert>
        )}

        <div className="flex items-center gap-2 p-3 rounded-md bg-muted/50 border">
          <Lock className="h-4 w-4 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            HIPAA-compliant storage • AWS Textract OCR • Comprehend Medical analysis
          </p>
        </div>

        <Tabs defaultValue="prescriptions" className="space-y-6">
          <TabsList className="grid w-full max-w-5xl grid-cols-5">
            <TabsTrigger value="prescriptions" data-testid="tab-prescriptions">
              <Pill className="h-4 w-4 mr-2" />
              Current Rx
            </TabsTrigger>
            <TabsTrigger value="documents" data-testid="tab-documents">
              <Upload className="h-4 w-4 mr-2" />
              Documents
            </TabsTrigger>
            <TabsTrigger value="ehr" data-testid="tab-ehr">
              <Activity className="h-4 w-4 mr-2" />
              EHR
            </TabsTrigger>
            <TabsTrigger value="imaging" data-testid="tab-imaging">
              <Image className="h-4 w-4 mr-2" />
              Imaging
            </TabsTrigger>
            <TabsTrigger value="history" data-testid="tab-history">
              <History className="h-4 w-4 mr-2" />
              Rx History
            </TabsTrigger>
          </TabsList>

          <TabsContent value="prescriptions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Pill className="h-5 w-5" />
                  Current Prescriptions
                </CardTitle>
                <CardDescription>
                  Active prescriptions from all your doctors. This flows to your Medications page.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingRx ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : activePrescriptions.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <Pill className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>No active prescriptions</p>
                    <p className="text-sm mt-1">Prescriptions from your doctors will appear here</p>
                  </div>
                ) : (
                  <ScrollArea className="h-[500px]">
                    <div className="space-y-4">
                      {activePrescriptions.map((rx) => {
                        const daysRemaining = rx.expirationDate 
                          ? differenceInDays(new Date(rx.expirationDate), new Date())
                          : null;
                        const isEndingSoon = daysRemaining !== null && daysRemaining <= 7 && daysRemaining > 0;

                        return (
                          <Card key={rx.id} className={`hover-elevate ${rx.hasConflict ? 'border-destructive/50' : ''}`} data-testid={`card-prescription-${rx.id}`}>
                            <CardHeader className="pb-3">
                              <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 flex-wrap mb-1">
                                    <h3 className="font-semibold text-lg">{rx.medicationName}</h3>
                                    {getStatusBadge(rx.status)}
                                    {rx.specialty && (
                                      <Badge variant="outline" className={`text-xs ${getSpecialtyColor(rx.specialty)}`}>
                                        <Stethoscope className="h-3 w-3 mr-1" />
                                        {rx.specialty}
                                      </Badge>
                                    )}
                                    {rx.isContinuous ? (
                                      <Badge variant="secondary" className="text-xs">
                                        <Infinity className="h-3 w-3 mr-1" />
                                        Continuous
                                      </Badge>
                                    ) : rx.durationDays && (
                                      <Badge variant="outline" className="text-xs">
                                        <Timer className="h-3 w-3 mr-1" />
                                        {rx.durationDays} days
                                      </Badge>
                                    )}
                                    {rx.hasConflict && (
                                      <Badge variant="destructive" className="text-xs">
                                        <Ban className="h-3 w-3 mr-1" />
                                        Conflict
                                      </Badge>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-3 text-sm text-muted-foreground">
                                    <span className="font-medium">{rx.dosage}</span>
                                    <span>{rx.frequency}</span>
                                    {rx.doctorName && (
                                      <span className="flex items-center gap-1">
                                        <User className="h-3 w-3" />
                                        Dr. {rx.doctorName}
                                      </span>
                                    )}
                                  </div>
                                </div>
                                <div className="text-right space-y-1">
                                  {rx.startDate && (
                                    <p className="text-xs text-muted-foreground">
                                      Started {format(new Date(rx.startDate), "MMM d, yyyy")}
                                    </p>
                                  )}
                                  {daysRemaining !== null && (
                                    <p className={`text-xs ${isEndingSoon ? 'text-amber-600 font-medium' : 'text-muted-foreground'}`}>
                                      {isEndingSoon && <AlertCircle className="h-3 w-3 inline mr-1" />}
                                      {daysRemaining > 0 ? `${daysRemaining} days remaining` : 'Expired'}
                                    </p>
                                  )}
                                </div>
                              </div>
                            </CardHeader>
                            {(rx.notes || rx.hasConflict) && (
                              <CardContent className="pt-0">
                                {rx.hasConflict && (
                                  <Alert variant="destructive" className="mb-3">
                                    <Shield className="h-4 w-4" />
                                    <AlertDescription className="text-sm">
                                      This prescription is on hold due to a cross-specialty conflict. Your doctors are coordinating a resolution.
                                    </AlertDescription>
                                  </Alert>
                                )}
                                {rx.notes && (
                                  <p className="text-sm text-muted-foreground">{rx.notes}</p>
                                )}
                              </CardContent>
                            )}
                            <CardFooter className="pt-0">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => {
                                  setSelectedPrescriptionForChange(rx);
                                  setShowDosageRequestDialog(true);
                                }}
                                data-testid={`button-request-change-${rx.id}`}
                              >
                                <Edit3 className="h-4 w-4 mr-1" />
                                Request Change
                              </Button>
                            </CardFooter>
                          </Card>
                        );
                      })}
                    </div>
                  </ScrollArea>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="documents" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-1">
                <CardHeader>
                  <CardTitle>Upload Document</CardTitle>
                  <CardDescription>
                    Upload medical records with AWS Textract OCR analysis
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    className="border-2 border-dashed rounded-lg p-6 text-center hover-elevate active-elevate-2 cursor-pointer transition-colors"
                    data-testid="dropzone-upload"
                  >
                    <input
                      type="file"
                      id="file-upload"
                      className="hidden"
                      accept=".jpg,.jpeg,.png,.pdf"
                      onChange={handleFileSelect}
                      data-testid="input-file-upload"
                    />
                    <label htmlFor="file-upload" className="cursor-pointer">
                      <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                      <p className="text-sm font-medium">
                        {selectedFile ? selectedFile.name : "Drop file here or click to browse"}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Supports JPG, PNG, PDF (max 20MB)
                      </p>
                    </label>
                  </div>

                  {selectedFile && (
                    <>
                      <div className="space-y-2">
                        <Label htmlFor="document-type">Document Type</Label>
                        <Select value={documentType} onValueChange={setDocumentType}>
                          <SelectTrigger id="document-type" data-testid="select-document-type">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="lab_report">Lab Report</SelectItem>
                            <SelectItem value="prescription">Prescription</SelectItem>
                            <SelectItem value="imaging">Imaging Report</SelectItem>
                            <SelectItem value="discharge_summary">Discharge Summary</SelectItem>
                            <SelectItem value="other">Other</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="document-date">Document Date (Optional)</Label>
                        <Input
                          id="document-date"
                          type="date"
                          value={documentDate}
                          onChange={(e) => setDocumentDate(e.target.value)}
                          data-testid="input-document-date"
                        />
                      </div>

                      {uploadProgress > 0 && <Progress value={uploadProgress} className="w-full" />}

                      <div className="flex gap-2">
                        <Button
                          onClick={handleUpload}
                          disabled={uploadMutation.isPending}
                          className="flex-1"
                          data-testid="button-upload-document"
                        >
                          {uploadMutation.isPending ? (
                            <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Processing...</>
                          ) : (
                            <><Upload className="mr-2 h-4 w-4" />Upload & Analyze</>
                          )}
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => { setSelectedFile(null); setDocumentType("other"); setDocumentDate(""); }}
                          data-testid="button-cancel-upload"
                        >
                          Cancel
                        </Button>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>

              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Uploaded Documents</CardTitle>
                  <CardDescription>{uploadedDocs.length} document{uploadedDocs.length !== 1 ? "s" : ""} processed</CardDescription>
                </CardHeader>
                <CardContent>
                  {uploadedDocs.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <FileText className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>No uploaded documents yet</p>
                    </div>
                  ) : (
                    <ScrollArea className="h-[400px]">
                      <div className="space-y-3">
                        {uploadedDocs.map((doc) => (
                          <Card key={doc.id} className="hover-elevate" data-testid={`card-document-${doc.id}`}>
                            <CardHeader className="pb-3">
                              <div className="flex items-start justify-between gap-4">
                                <div className="flex items-center gap-2">
                                  {getStatusIcon(doc.processingStatus)}
                                  <div>
                                    <h4 className="text-sm font-medium">{doc.fileName}</h4>
                                    <p className="text-xs text-muted-foreground">
                                      {formatDistanceToNow(new Date(doc.createdAt), { addSuffix: true })}
                                    </p>
                                  </div>
                                </div>
                                <div className="flex items-center gap-2">
                                  {getStatusBadge(doc.processingStatus)}
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => setSelectedDocument(doc)}
                                    data-testid={`button-view-${doc.id}`}
                                  >
                                    <Eye className="h-4 w-4" />
                                  </Button>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => deleteMutation.mutate(doc.id)}
                                    data-testid={`button-delete-${doc.id}`}
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </div>
                              </div>
                            </CardHeader>
                            {doc.extractedData && (
                              <CardContent className="pt-0">
                                <div className="flex flex-wrap gap-2">
                                  {doc.extractedData.medications?.slice(0, 3).map((med, i) => (
                                    <Badge key={i} variant="secondary" className="text-xs">{med}</Badge>
                                  ))}
                                  {doc.extractedData.diagnosis?.slice(0, 2).map((diag, i) => (
                                    <Badge key={i} variant="outline" className="text-xs">{diag}</Badge>
                                  ))}
                                </div>
                              </CardContent>
                            )}
                          </Card>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="ehr" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold">EHR Integrations</h2>
                <p className="text-sm text-muted-foreground">
                  Connect your electronic health records - synced data processed with AWS HealthLake
                </p>
              </div>
              <Button onClick={() => setShowAddEHRDialog(true)} data-testid="button-add-ehr">
                <Plus className="h-4 w-4 mr-2" />
                Add Connection
              </Button>
            </div>

            {isLoadingEHR ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {[1, 2, 3].map((i) => (
                  <Card key={i} className="animate-pulse">
                    <CardHeader><div className="h-4 bg-muted rounded w-1/2" /></CardHeader>
                  </Card>
                ))}
              </div>
            ) : ehrConnections.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {ehrConnections.map((connection) => (
                  <Card key={connection.id} className="hover-elevate" data-testid={`card-ehr-${connection.id}`}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <CardTitle className="flex items-center gap-2 text-lg">
                            {getStatusIcon(connection.connectionStatus)}
                            {connection.ehrSystemName}
                          </CardTitle>
                          <CardDescription className="mt-1">{connection.facilityName}</CardDescription>
                        </div>
                        {getStatusBadge(connection.connectionStatus)}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Patient ID:</span>
                          <span className="font-mono">{connection.patientExternalId}</span>
                        </div>
                        {connection.lastSyncedAt && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Last Synced:</span>
                            <span>{new Date(connection.lastSyncedAt).toLocaleDateString()}</span>
                          </div>
                        )}
                      </div>
                      <div className="flex gap-2 mt-4">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => syncEHRMutation.mutate(connection.id)}
                          disabled={syncEHRMutation.isPending}
                          data-testid={`button-sync-${connection.id}`}
                        >
                          <RefreshCw className={`h-4 w-4 mr-1 ${syncEHRMutation.isPending ? 'animate-spin' : ''}`} />
                          Sync
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteEHRMutation.mutate(connection.id)}
                          data-testid={`button-disconnect-${connection.id}`}
                        >
                          Disconnect
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="py-12 text-center text-muted-foreground">
                  <Database className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No EHR connections yet</p>
                  <p className="text-sm mt-1">Connect your health records from hospitals and clinics</p>
                  <Button onClick={() => setShowAddEHRDialog(true)} className="mt-4" data-testid="button-add-ehr-empty">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Your First Connection
                  </Button>
                </CardContent>
              </Card>
            )}

            {ehrDocs.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>EHR Documents</CardTitle>
                  <CardDescription>{ehrDocs.length} documents synced from EHR systems</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[300px]">
                    <div className="space-y-3">
                      {ehrDocs.map((doc) => (
                        <Card key={doc.id} className="hover-elevate" data-testid={`card-ehr-doc-${doc.id}`}>
                          <CardHeader className="pb-2">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                {getStatusIcon(doc.processingStatus)}
                                <span className="text-sm font-medium">{doc.fileName}</span>
                              </div>
                              <Button variant="ghost" size="sm" onClick={() => setSelectedDocument(doc)}>
                                <Eye className="h-4 w-4" />
                              </Button>
                            </div>
                          </CardHeader>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="imaging" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Image className="h-5 w-5" />
                  Medical Imaging
                </CardTitle>
                <CardDescription>
                  X-rays, MRIs, CT scans, and other medical images
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingDocs ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : imagingDocs.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <Image className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>No imaging files yet</p>
                    <p className="text-sm mt-1">Upload medical images or sync from your EHR</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {imagingDocs.map((doc) => (
                      <Card key={doc.id} className="hover-elevate cursor-pointer" onClick={() => setSelectedDocument(doc)} data-testid={`card-imaging-${doc.id}`}>
                        <CardContent className="p-4">
                          <div className="aspect-square bg-muted rounded-md flex items-center justify-center mb-2">
                            {doc.fileUrl ? (
                              <img src={doc.fileUrl} alt={doc.fileName} className="w-full h-full object-cover rounded-md" />
                            ) : (
                              <Image className="h-8 w-8 text-muted-foreground" />
                            )}
                          </div>
                          <p className="text-xs font-medium truncate">{doc.fileName}</p>
                          <p className="text-xs text-muted-foreground">
                            {formatDistanceToNow(new Date(doc.createdAt), { addSuffix: true })}
                          </p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Archive className="h-5 w-5" />
                  Prescription History
                </CardTitle>
                <CardDescription>
                  Archived, superseded, and completed prescriptions from all specialists
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingHistory ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : historyError ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <AlertCircle className="h-12 w-12 mx-auto mb-3 text-destructive opacity-70" />
                    <p className="font-medium text-destructive">Failed to load prescription history</p>
                    <p className="text-sm mt-1">Please try refreshing the page</p>
                  </div>
                ) : !medicationHistory?.medications || medicationHistory.medications.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <History className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>No prescription history yet</p>
                    <p className="text-sm mt-1">Completed and archived prescriptions will appear here</p>
                  </div>
                ) : (
                  <ScrollArea className="h-[500px]">
                    <div className="space-y-4">
                      {medicationHistory.medications.map((med) => (
                        <Card key={med.id} className="hover-elevate" data-testid={`card-history-${med.id}`}>
                          <CardHeader className="pb-3">
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1 flex-wrap">
                                  <h3 className="font-medium">{med.name}</h3>
                                  {getStatusBadge(med.status)}
                                  {med.specialty && (
                                    <Badge variant="outline" className={`text-xs ${getSpecialtyColor(med.specialty)}`}>
                                      <Stethoscope className="h-3 w-3 mr-1" />
                                      {med.specialty}
                                    </Badge>
                                  )}
                                </div>
                                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                                  <span>{med.dosage}</span>
                                  <span>{med.frequency}</span>
                                  {med.isContinuous ? (
                                    <span className="flex items-center gap-1">
                                      <Infinity className="h-3 w-3" />
                                      Continuous
                                    </span>
                                  ) : med.durationDays && (
                                    <span className="flex items-center gap-1">
                                      <Clock className="h-3 w-3" />
                                      {med.durationDays} days
                                    </span>
                                  )}
                                </div>
                              </div>
                              <div className="text-right text-sm text-muted-foreground">
                                {med.archivedAt ? (
                                  <span className="flex items-center gap-1">
                                    <Archive className="h-3 w-3" />
                                    Archived {formatDistanceToNow(new Date(med.archivedAt), { addSuffix: true })}
                                  </span>
                                ) : med.createdAt && (
                                  <span className="flex items-center gap-1">
                                    <Calendar className="h-3 w-3" />
                                    {formatDistanceToNow(new Date(med.createdAt), { addSuffix: true })}
                                  </span>
                                )}
                              </div>
                            </div>
                          </CardHeader>
                          {(med.supersessionReason || med.archiveReason) && (
                            <CardContent className="pt-0">
                              {med.status === "superseded" && med.supersessionReason && (
                                <div className="flex items-start gap-2 p-3 bg-muted/50 rounded-md border">
                                  <ArrowRightLeft className="h-4 w-4 text-muted-foreground mt-0.5" />
                                  <div>
                                    <p className="text-sm font-medium">Supersession Note</p>
                                    <p className="text-sm text-muted-foreground">{med.supersessionReason}</p>
                                    {med.supersededAt && (
                                      <p className="text-xs text-muted-foreground mt-1">
                                        Superseded on {new Date(med.supersededAt).toLocaleDateString()}
                                      </p>
                                    )}
                                  </div>
                                </div>
                              )}
                              {med.archiveReason && med.status !== "superseded" && (
                                <div className="flex items-start gap-2 p-3 bg-muted/50 rounded-md border">
                                  <Archive className="h-4 w-4 text-muted-foreground mt-0.5" />
                                  <div>
                                    <p className="text-sm text-muted-foreground capitalize">
                                      Archived: {med.archiveReason.replace("_", " ")}
                                    </p>
                                  </div>
                                </div>
                              )}
                            </CardContent>
                          )}
                        </Card>
                      ))}
                    </div>
                    {medicationHistory.total > 0 && (
                      <div className="mt-4 pt-4 border-t text-center text-sm text-muted-foreground">
                        Showing {medicationHistory.medications.length} of {medicationHistory.total} archived prescriptions
                        {medicationHistory.hasMore && (
                          <span className="block text-xs mt-1">More records available</span>
                        )}
                      </div>
                    )}
                  </ScrollArea>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <Dialog open={showAddEHRDialog} onOpenChange={setShowAddEHRDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add EHR Connection</DialogTitle>
              <DialogDescription>
                Connect to your hospital or clinic's electronic health records
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="ehr-system">EHR System</Label>
                <Select value={selectedSystem} onValueChange={setSelectedSystem}>
                  <SelectTrigger id="ehr-system" data-testid="select-ehr-system">
                    <SelectValue placeholder="Select your EHR system..." />
                  </SelectTrigger>
                  <SelectContent>
                    {EHR_SYSTEMS.map((sys) => (
                      <SelectItem key={sys.id} value={sys.id}>
                        {sys.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="facility-name">Facility Name</Label>
                <Input
                  id="facility-name"
                  value={facilityName}
                  onChange={(e) => setFacilityName(e.target.value)}
                  placeholder="e.g., City General Hospital"
                  data-testid="input-facility-name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="patient-id">Your Patient ID (at this facility)</Label>
                <Input
                  id="patient-id"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="e.g., PAT-123456"
                  data-testid="input-patient-id"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowAddEHRDialog(false)}>Cancel</Button>
              <Button onClick={handleAddEHRConnection} disabled={addEHRMutation.isPending} data-testid="button-connect-ehr">
                {addEHRMutation.isPending ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Plus className="h-4 w-4 mr-2" />}
                Connect
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={showDosageRequestDialog} onOpenChange={setShowDosageRequestDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Edit3 className="h-5 w-5" />
                Request Dosage Change
              </DialogTitle>
              <DialogDescription>
                Submit a request to your doctor to change your medication dosage
              </DialogDescription>
            </DialogHeader>
            {selectedPrescriptionForChange && (
              <div className="space-y-4 py-4">
                <Alert>
                  <Pill className="h-4 w-4" />
                  <AlertTitle>Current Prescription</AlertTitle>
                  <AlertDescription>
                    <strong>{selectedPrescriptionForChange.medicationName}</strong>
                    <br />
                    {selectedPrescriptionForChange.dosage} • {selectedPrescriptionForChange.frequency}
                  </AlertDescription>
                </Alert>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="requested-dosage">Requested Dosage</Label>
                    <Input
                      id="requested-dosage"
                      value={dosageChangeForm.requestedDosage}
                      onChange={(e) => setDosageChangeForm({ ...dosageChangeForm, requestedDosage: e.target.value })}
                      placeholder="e.g., 20mg"
                      data-testid="input-requested-dosage"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="requested-frequency">Requested Frequency</Label>
                    <Select
                      value={dosageChangeForm.requestedFrequency}
                      onValueChange={(val) => setDosageChangeForm({ ...dosageChangeForm, requestedFrequency: val })}
                    >
                      <SelectTrigger data-testid="select-requested-frequency">
                        <SelectValue placeholder="Select frequency..." />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="once daily">Once Daily</SelectItem>
                        <SelectItem value="twice daily">Twice Daily</SelectItem>
                        <SelectItem value="three times daily">Three Times Daily</SelectItem>
                        <SelectItem value="as needed">As Needed (PRN)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="request-reason">Reason for Request *</Label>
                  <Textarea
                    id="request-reason"
                    value={dosageChangeForm.reason}
                    onChange={(e) => setDosageChangeForm({ ...dosageChangeForm, reason: e.target.value })}
                    placeholder="Explain why you're requesting this change..."
                    rows={4}
                    data-testid="input-request-reason"
                  />
                </div>
              </div>
            )}
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowDosageRequestDialog(false)}>Cancel</Button>
              <Button
                onClick={() => createDosageRequestMutation.mutate()}
                disabled={!dosageChangeForm.reason.trim() || createDosageRequestMutation.isPending}
                data-testid="button-submit-request"
              >
                {createDosageRequestMutation.isPending ? (
                  <><RefreshCw className="h-4 w-4 mr-2 animate-spin" />Submitting...</>
                ) : (
                  <><Send className="h-4 w-4 mr-2" />Submit Request</>
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {selectedDocument && (
          <Dialog open={!!selectedDocument} onOpenChange={() => setSelectedDocument(null)}>
            <DialogContent className="max-w-3xl max-h-[80vh] overflow-auto">
              <DialogHeader>
                <DialogTitle>Document Details</DialogTitle>
                <DialogDescription>{selectedDocument.fileName}</DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Type:</span>
                    <span className="ml-2 capitalize">{selectedDocument.documentType || "Unknown"}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Status:</span>
                    <span className="ml-2">{getStatusBadge(selectedDocument.processingStatus)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Uploaded:</span>
                    <span className="ml-2">{format(new Date(selectedDocument.createdAt), "PPP")}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Size:</span>
                    <span className="ml-2">{(selectedDocument.fileSize / 1024).toFixed(1)} KB</span>
                  </div>
                </div>

                {selectedDocument.extractedData && (
                  <div className="space-y-3 border-t pt-4">
                    <h4 className="font-medium">AI-Extracted Information</h4>
                    {selectedDocument.extractedData.diagnosis && selectedDocument.extractedData.diagnosis.length > 0 && (
                      <div>
                        <p className="text-sm text-muted-foreground mb-1">Diagnoses:</p>
                        <div className="flex flex-wrap gap-1">
                          {selectedDocument.extractedData.diagnosis.map((d, i) => (
                            <Badge key={i} variant="outline">{d}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    {selectedDocument.extractedData.medications && selectedDocument.extractedData.medications.length > 0 && (
                      <div>
                        <p className="text-sm text-muted-foreground mb-1">Medications:</p>
                        <div className="flex flex-wrap gap-1">
                          {selectedDocument.extractedData.medications.map((m, i) => (
                            <Badge key={i} variant="secondary">{m}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    {selectedDocument.extractedData.icdCodes && selectedDocument.extractedData.icdCodes.length > 0 && (
                      <div>
                        <p className="text-sm text-muted-foreground mb-1">ICD-10 Codes:</p>
                        <div className="flex flex-wrap gap-1">
                          {selectedDocument.extractedData.icdCodes.map((code, i) => (
                            <Badge key={i} variant="default" className="text-xs">
                              {code.code}: {code.description}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {selectedDocument.fileUrl && (
                  <div className="border-t pt-4">
                    <Button asChild>
                      <a href={selectedDocument.fileUrl} target="_blank" rel="noopener noreferrer">
                        <Eye className="h-4 w-4 mr-2" />
                        View Original Document
                      </a>
                    </Button>
                  </div>
                )}
              </div>
            </DialogContent>
          </Dialog>
        )}
      </div>
    </div>
  );
}
