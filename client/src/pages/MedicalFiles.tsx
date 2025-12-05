import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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
import { useToast } from "@/hooks/use-toast";
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
  FlaskConical,
  Lock,
  Calendar,
  Database,
  Pill,
  ArrowRightLeft,
  Clock,
  Ban,
  Infinity,
  Stethoscope,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
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
    medicalEntities?: Array<{
      text: string;
      type: string;
      category: string;
      score: number;
    }>;
    icdCodes?: Array<{ code: string; description: string; score: number }>;
  };
  documentType?: string;
  documentDate?: string;
  processingStatus?: string;
  errorMessage?: string;
  createdAt: string;
  updatedAt: string;
};

export default function MedicalFiles() {
  const { toast } = useToast();
  
  // Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState<string>("other");
  const [documentDate, setDocumentDate] = useState<string>("");
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [selectedDocument, setSelectedDocument] = useState<MedicalDocument | null>(null);

  // EHR state
  const [showAddEHRDialog, setShowAddEHRDialog] = useState(false);
  const [selectedSystem, setSelectedSystem] = useState("");
  const [facilityName, setFacilityName] = useState("");
  const [patientId, setPatientId] = useState("");

  // Queries
  const { data: documents = [], isLoading: isLoadingDocs } = useQuery<MedicalDocument[]>({
    queryKey: ["/api/medical-documents"],
  });

  const { data: ehrConnections, isLoading: isLoadingEHR } = useQuery({
    queryKey: ["/api/ehr/connections"],
  });

  const { data: medicationHistory, isLoading: isLoadingMedHistory, error: medHistoryError } = useQuery<{
    medications: Array<{
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
      createdAt?: string;
    }>;
    total: number;
    hasMore: boolean;
  }>({
    queryKey: ["/api/medications/history"],
  });

  // Mutations
  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      return await apiRequest<{ document: MedicalDocument }>("/api/medical-documents/upload", {
        method: "POST",
        body: formData,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/medical-documents"] });
      setSelectedFile(null);
      setDocumentType("other");
      setDocumentDate("");
      setUploadProgress(0);
      toast({
        title: "Document Uploaded",
        description: "AWS Textract is extracting medical text. Comprehend Medical will analyze entities.",
      });
    },
    onError: (error) => {
      toast({
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Failed to upload document",
        variant: "destructive",
      });
      setUploadProgress(0);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (documentId: string) => {
      return await apiRequest(`/api/medical-documents/${documentId}`, {
        method: "DELETE",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/medical-documents"] });
      toast({
        title: "Document Deleted",
        description: "The document has been removed from your records.",
      });
    },
  });

  const addEHRConnectionMutation = useMutation({
    mutationFn: async (data: any) => {
      return await apiRequest("/api/ehr/connections", {
        method: "POST",
        body: JSON.stringify(data),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ehr/connections"] });
      queryClient.invalidateQueries({ queryKey: ["/api/medical-documents"] });
      setShowAddEHRDialog(false);
      setSelectedSystem("");
      setFacilityName("");
      setPatientId("");
      toast({
        title: "EHR Connected",
        description: "Your health records will sync automatically. AWS HealthLake will organize the data.",
      });
    },
  });

  const syncEHRMutation = useMutation({
    mutationFn: async (id: string) => {
      return await apiRequest(`/api/ehr/connections/${id}/sync`, {
        method: "POST",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ehr/connections"] });
      queryClient.invalidateQueries({ queryKey: ["/api/medical-documents"] });
      toast({
        title: "Sync Complete",
        description: "Health records synchronized and processed with AWS services.",
      });
    },
  });

  const deleteEHRMutation = useMutation({
    mutationFn: async (id: string) => {
      return await apiRequest(`/api/ehr/connections/${id}`, {
        method: "DELETE",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ehr/connections"] });
      toast({
        title: "Connection Removed",
        description: "The EHR connection has been deleted.",
      });
    },
  });

  // Handlers
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 20 * 1024 * 1024) {
        toast({
          title: "File Too Large",
          description: "Maximum file size is 20MB",
          variant: "destructive",
        });
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
    if (documentDate) {
      formData.append("documentDate", documentDate);
    }

    setUploadProgress(50);
    uploadMutation.mutate(formData);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file) {
      if (file.size > 20 * 1024 * 1024) {
        toast({
          title: "File Too Large",
          description: "Maximum file size is 20MB",
          variant: "destructive",
        });
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleAddEHRConnection = () => {
    if (!selectedSystem || !facilityName || !patientId) {
      toast({
        title: "Missing Information",
        description: "Please fill in all fields.",
        variant: "destructive",
      });
      return;
    }

    const system = EHR_SYSTEMS.find((s) => s.id === selectedSystem);
    addEHRConnectionMutation.mutate({
      ehrSystem: selectedSystem,
      ehrSystemName: system?.name || "",
      facilityName,
      patientExternalId: patientId,
      connectionStatus: "pending",
    });
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case "completed":
      case "connected":
        return <CheckCircle2 className="h-4 w-4 text-primary" />;
      case "processing":
      case "pending":
        return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
      case "failed":
      case "error":
        return <XCircle className="h-4 w-4 text-destructive" />;
      default:
        return <FileText className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case "completed":
      case "connected":
        return <Badge variant="default">Processed</Badge>;
      case "processing":
        return <Badge variant="secondary">Processing...</Badge>;
      case "pending":
        return <Badge variant="secondary">Pending</Badge>;
      case "failed":
      case "error":
        return <Badge variant="destructive">Failed</Badge>;
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  // Sort documents by date (newest first)
  const sortedDocuments = [...documents].sort((a, b) => {
    const dateA = new Date(a.documentDate || a.createdAt).getTime();
    const dateB = new Date(b.documentDate || b.createdAt).getTime();
    return dateB - dateA;
  });

  // Group documents by source
  const uploadedDocs = sortedDocuments.filter(d => !d.source || d.source === "upload");
  const ehrDocs = sortedDocuments.filter(d => d.source === "ehr");

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Medical Files</h1>
          <p className="text-muted-foreground">
            Upload documents, connect EHR systems, and view all medical records with AI-powered analysis
          </p>
        </div>

        <div className="flex items-center gap-2 p-3 rounded-md bg-muted/50 border">
          <Lock className="h-4 w-4 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            All files encrypted with AWS S3, OCR via Textract, medical analysis via Comprehend Medical
          </p>
        </div>

        <Tabs defaultValue="all" className="space-y-6">
          <TabsList className="grid w-full max-w-4xl grid-cols-5">
            <TabsTrigger value="all" data-testid="tab-all-files">
              <Database className="h-4 w-4 mr-2" />
              All Files
            </TabsTrigger>
            <TabsTrigger value="upload" data-testid="tab-upload">
              <Upload className="h-4 w-4 mr-2" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="ehr" data-testid="tab-ehr-integration">
              <Activity className="h-4 w-4 mr-2" />
              EHR Integration
            </TabsTrigger>
            <TabsTrigger value="imaging" data-testid="tab-imaging">
              <Image className="h-4 w-4 mr-2" />
              Imaging
            </TabsTrigger>
            <TabsTrigger value="med-history" data-testid="tab-medication-history">
              <Pill className="h-4 w-4 mr-2" />
              Rx History
            </TabsTrigger>
          </TabsList>

          {/* All Files Tab - Chronologically Organized */}
          <TabsContent value="all" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>All Medical Records</CardTitle>
                <CardDescription>
                  Chronologically organized files from uploads and EHR integrations
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingDocs ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : sortedDocuments.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <FileText className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>No medical files yet</p>
                    <p className="text-sm mt-1">Upload documents or connect your EHR system</p>
                  </div>
                ) : (
                  <ScrollArea className="h-[600px]">
                    <div className="space-y-4">
                      {sortedDocuments.map((doc) => (
                        <Card key={doc.id} className="hover-elevate" data-testid={`card-document-${doc.id}`}>
                          <CardHeader>
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex items-start gap-3 flex-1">
                                {getStatusIcon(doc.processingStatus)}
                                <div className="flex-1 min-w-0">
                                  <h3 className="font-medium truncate">{doc.fileName}</h3>
                                  <div className="flex items-center gap-2 mt-1">
                                    <Badge variant="outline" className="text-xs">
                                      {doc.source === "ehr" ? "EHR" : "Upload"}
                                    </Badge>
                                    {doc.documentDate && (
                                      <span className="text-xs text-muted-foreground flex items-center gap-1">
                                        <Calendar className="h-3 w-3" />
                                        {new Date(doc.documentDate).toLocaleDateString()}
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-sm text-muted-foreground mt-1">
                                    {formatDistanceToNow(new Date(doc.createdAt), { addSuffix: true })}
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                {getStatusBadge(doc.processingStatus)}
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  onClick={() => setSelectedDocument(doc)}
                                  data-testid={`button-view-${doc.id}`}
                                >
                                  <Eye className="h-4 w-4" />
                                </Button>
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  onClick={() => {
                                    if (confirm("Are you sure you want to delete this document?")) {
                                      deleteMutation.mutate(doc.id);
                                    }
                                  }}
                                  data-testid={`button-delete-${doc.id}`}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>
                          </CardHeader>
                          {doc.extractedData && (
                            <CardContent>
                              <div className="space-y-2">
                                {doc.extractedData.diagnosis && doc.extractedData.diagnosis.length > 0 && (
                                  <div>
                                    <p className="text-sm font-medium">Diagnosis:</p>
                                    <div className="flex flex-wrap gap-1 mt-1">
                                      {doc.extractedData.diagnosis.slice(0, 3).map((diag, i) => (
                                        <Badge key={i} variant="outline">{diag}</Badge>
                                      ))}
                                    </div>
                                  </div>
                                )}
                                {doc.extractedData.medications && doc.extractedData.medications.length > 0 && (
                                  <div>
                                    <p className="text-sm font-medium">Medications:</p>
                                    <div className="flex flex-wrap gap-1 mt-1">
                                      {doc.extractedData.medications.slice(0, 3).map((med, i) => (
                                        <Badge key={i} variant="secondary">{med}</Badge>
                                      ))}
                                    </div>
                                  </div>
                                )}
                                {doc.extractedData.icdCodes && doc.extractedData.icdCodes.length > 0 && (
                                  <div>
                                    <p className="text-sm font-medium">ICD-10 Codes:</p>
                                    <div className="flex flex-wrap gap-1 mt-1">
                                      {doc.extractedData.icdCodes.slice(0, 2).map((code, i) => (
                                        <Badge key={i} variant="default" className="text-xs">
                                          {code.code}
                                        </Badge>
                                      ))}
                                    </div>
                                  </div>
                                )}
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
          </TabsContent>

          {/* Upload Tab */}
          <TabsContent value="upload" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-1">
                <CardHeader>
                  <CardTitle>Upload Document</CardTitle>
                  <CardDescription>
                    Upload medical records with AWS Textract OCR and Comprehend Medical analysis
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

                      {uploadProgress > 0 && (
                        <Progress value={uploadProgress} className="w-full" />
                      )}

                      <div className="flex gap-2">
                        <Button
                          onClick={handleUpload}
                          disabled={uploadMutation.isPending}
                          className="flex-1"
                          data-testid="button-upload-document"
                        >
                          {uploadMutation.isPending ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              Processing...
                            </>
                          ) : (
                            <>
                              <Upload className="mr-2 h-4 w-4" />
                              Upload & Analyze
                            </>
                          )}
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => {
                            setSelectedFile(null);
                            setDocumentType("other");
                            setDocumentDate("");
                          }}
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
                  <CardDescription>
                    {uploadedDocs.length} document{uploadedDocs.length !== 1 ? 's' : ''} processed
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {uploadedDocs.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <FileText className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>No uploaded documents yet</p>
                    </div>
                  ) : (
                    <ScrollArea className="h-[500px]">
                      <div className="space-y-3">
                        {uploadedDocs.map((doc) => (
                          <Card key={doc.id} className="hover-elevate" data-testid={`card-upload-${doc.id}`}>
                            <CardHeader className="pb-3">
                              <div className="flex items-start justify-between">
                                <div className="flex items-center gap-2">
                                  {getStatusIcon(doc.processingStatus)}
                                  <div>
                                    <h4 className="text-sm font-medium">{doc.fileName}</h4>
                                    <p className="text-xs text-muted-foreground">
                                      {formatDistanceToNow(new Date(doc.createdAt), { addSuffix: true })}
                                    </p>
                                  </div>
                                </div>
                                {getStatusBadge(doc.processingStatus)}
                              </div>
                            </CardHeader>
                          </Card>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* EHR Integration Tab */}
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
                    <CardHeader>
                      <div className="h-4 bg-muted rounded w-1/2" />
                      <div className="h-3 bg-muted rounded w-3/4 mt-2" />
                    </CardHeader>
                  </Card>
                ))}
              </div>
            ) : ehrConnections && ehrConnections.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {ehrConnections.map((connection: any) => (
                  <Card key={connection.id} className="hover-elevate" data-testid={`card-ehr-${connection.id}`}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <CardTitle className="flex items-center gap-2 text-lg">
                            {getStatusIcon(connection.connectionStatus)}
                            {connection.ehrSystemName}
                          </CardTitle>
                          <CardDescription className="mt-1">
                            {connection.facilityName}
                          </CardDescription>
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
                          className="flex-1"
                          onClick={() => syncEHRMutation.mutate(connection.id)}
                          disabled={syncEHRMutation.isPending}
                          data-testid={`button-sync-${connection.id}`}
                        >
                          <RefreshCw className="h-3 w-3 mr-1" />
                          Sync
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteEHRMutation.mutate(connection.id)}
                          data-testid={`button-delete-ehr-${connection.id}`}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <Activity className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No EHR Connections</h3>
                  <p className="text-muted-foreground text-center mb-4">
                    Connect your EHR to automatically sync and analyze medical records
                  </p>
                  <Button onClick={() => setShowAddEHRDialog(true)} data-testid="button-add-ehr-empty">
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
                  <CardDescription>
                    {ehrDocs.length} document{ehrDocs.length !== 1 ? 's' : ''} synced from EHR systems
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[400px]">
                    <div className="space-y-3">
                      {ehrDocs.map((doc) => (
                        <Card key={doc.id} className="hover-elevate">
                          <CardHeader className="pb-3">
                            <div className="flex items-start justify-between">
                              <div className="flex items-center gap-2">
                                {getStatusIcon(doc.processingStatus)}
                                <div>
                                  <h4 className="text-sm font-medium">{doc.fileName}</h4>
                                  <p className="text-xs text-muted-foreground">
                                    {formatDistanceToNow(new Date(doc.createdAt), { addSuffix: true })}
                                  </p>
                                </div>
                              </div>
                              <Badge variant="outline">EHR</Badge>
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

          {/* Medical Imaging Tab */}
          <TabsContent value="imaging">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  Medical Imaging Analysis
                  <Badge variant="secondary">AWS Medical Imaging</Badge>
                </CardTitle>
                <CardDescription>
                  Upload X-rays, CT scans, MRIs for AI-powered analysis with AWS HealthImaging
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12 text-muted-foreground">
                  <Image className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Medical imaging analysis coming soon</p>
                  <p className="text-sm mt-1">AI-powered DICOM analysis with AWS HealthImaging service</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Medication History Tab */}
          <TabsContent value="med-history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Pill className="h-5 w-5" />
                  Medication History
                </CardTitle>
                <CardDescription>
                  Archived, superseded, and discontinued prescriptions from all specialists
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingMedHistory ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : medHistoryError ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <AlertCircle className="h-12 w-12 mx-auto mb-3 text-destructive opacity-70" />
                    <p className="font-medium text-destructive">Failed to load medication history</p>
                    <p className="text-sm mt-1">Please try refreshing the page</p>
                  </div>
                ) : !medicationHistory?.medications || medicationHistory.medications.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <Pill className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>No medication history yet</p>
                    <p className="text-sm mt-1">Archived and discontinued prescriptions will appear here</p>
                  </div>
                ) : (
                  <ScrollArea className="h-[600px]">
                    <div className="space-y-4">
                      {medicationHistory.medications.map((med) => {
                        const getStatusBadgeVariant = (status?: string) => {
                          switch (status) {
                            case 'superseded': return 'secondary';
                            case 'expired': return 'outline';
                            case 'discontinued': return 'destructive';
                            default: return 'outline';
                          }
                        };
                        
                        const getSpecialtyColor = (specialty?: string) => {
                          const colors: Record<string, string> = {
                            cardiology: "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300 border-rose-200 dark:border-rose-800",
                            oncology: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300 border-purple-200 dark:border-purple-800",
                            neurology: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300 border-blue-200 dark:border-blue-800",
                            rheumatology: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300 border-amber-200 dark:border-amber-800",
                            endocrinology: "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-300 border-cyan-200 dark:border-cyan-800",
                            gastroenterology: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300 border-orange-200 dark:border-orange-800",
                            "general medicine": "bg-gray-100 text-gray-700 dark:bg-gray-800/60 dark:text-gray-300 border-gray-200 dark:border-gray-700",
                          };
                          return colors[specialty?.toLowerCase() || ''] || colors['general medicine'];
                        };

                        return (
                          <Card key={med.id} className="hover-elevate" data-testid={`card-archived-med-${med.id}`}>
                            <CardHeader className="pb-3">
                              <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-1">
                                    <h3 className="font-medium">{med.name}</h3>
                                    <Badge variant={getStatusBadgeVariant(med.status)} className="text-xs capitalize">
                                      {med.status || 'archived'}
                                    </Badge>
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
                                  {med.createdAt && (
                                    <span className="flex items-center gap-1">
                                      <Calendar className="h-3 w-3" />
                                      {formatDistanceToNow(new Date(med.createdAt), { addSuffix: true })}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </CardHeader>
                            <CardContent className="pt-0">
                              {med.status === 'superseded' && med.supersessionReason && (
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
                              {med.conflictStatus && med.conflictStatus !== 'none' && (
                                <div className="flex items-start gap-2 p-3 bg-destructive/10 rounded-md border border-destructive/20 mt-2">
                                  <Ban className="h-4 w-4 text-destructive mt-0.5" />
                                  <div>
                                    <p className="text-sm font-medium text-destructive">Had Conflict</p>
                                    <p className="text-sm text-muted-foreground capitalize">{med.conflictStatus}</p>
                                  </div>
                                </div>
                              )}
                              {med.computedEndDate && (
                                <p className="text-xs text-muted-foreground mt-2">
                                  End date: {new Date(med.computedEndDate).toLocaleDateString()}
                                </p>
                              )}
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                    {medicationHistory.total > 0 && (
                      <div className="mt-4 pt-4 border-t text-center text-sm text-muted-foreground">
                        Showing {medicationHistory.medications.length} of {medicationHistory.total} archived medications
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

        {/* Document Details Dialog */}
        {selectedDocument && (
          <Dialog open={!!selectedDocument} onOpenChange={() => setSelectedDocument(null)}>
            <DialogContent className="max-w-3xl max-h-[80vh] overflow-auto">
              <DialogHeader>
                <DialogTitle>Document Details</DialogTitle>
                <DialogDescription>{selectedDocument.fileName}</DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium">Type</p>
                    <p className="text-sm text-muted-foreground capitalize">
                      {selectedDocument.documentType?.replace('_', ' ')}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Source</p>
                    <p className="text-sm text-muted-foreground">
                      {selectedDocument.source === "ehr" ? "EHR Integration" : "Manual Upload"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Size</p>
                    <p className="text-sm text-muted-foreground">
                      {(selectedDocument.fileSize / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Document Date</p>
                    <p className="text-sm text-muted-foreground">
                      {selectedDocument.documentDate
                        ? new Date(selectedDocument.documentDate).toLocaleDateString()
                        : "Not specified"}
                    </p>
                  </div>
                </div>

                <Separator />

                {selectedDocument.extractedText && (
                  <div>
                    <h3 className="font-medium mb-2 flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      Extracted Text (AWS Textract)
                    </h3>
                    <ScrollArea className="h-[200px] border rounded-md p-4">
                      <pre className="text-sm whitespace-pre-wrap">{selectedDocument.extractedText}</pre>
                    </ScrollArea>
                  </div>
                )}

                {selectedDocument.extractedData && (
                  <div className="space-y-4">
                    <h3 className="font-medium flex items-center gap-2">
                      <Activity className="h-4 w-4" />
                      Medical Analysis (AWS Comprehend Medical)
                    </h3>

                    {selectedDocument.extractedData.diagnosis && selectedDocument.extractedData.diagnosis.length > 0 && (
                      <div>
                        <p className="text-sm font-medium mb-2">Diagnoses</p>
                        <div className="flex flex-wrap gap-2">
                          {selectedDocument.extractedData.diagnosis.map((diag, i) => (
                            <Badge key={i}>{diag}</Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {selectedDocument.extractedData.medications && selectedDocument.extractedData.medications.length > 0 && (
                      <div>
                        <p className="text-sm font-medium mb-2">Medications</p>
                        <div className="flex flex-wrap gap-2">
                          {selectedDocument.extractedData.medications.map((med, i) => (
                            <Badge key={i} variant="secondary">{med}</Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {selectedDocument.extractedData.icdCodes && selectedDocument.extractedData.icdCodes.length > 0 && (
                      <div>
                        <p className="text-sm font-medium mb-2">ICD-10 Codes</p>
                        <div className="space-y-2">
                          {selectedDocument.extractedData.icdCodes.map((code, i) => (
                            <div key={i} className="flex items-center gap-2">
                              <Badge variant="default">{code.code}</Badge>
                              <span className="text-sm text-muted-foreground">{code.description}</span>
                              <span className="text-xs text-muted-foreground ml-auto">
                                {(code.score * 100).toFixed(0)}% confidence
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {selectedDocument.extractedData.medicalEntities && selectedDocument.extractedData.medicalEntities.length > 0 && (
                      <div>
                        <p className="text-sm font-medium mb-2">Medical Entities</p>
                        <div className="space-y-1">
                          {selectedDocument.extractedData.medicalEntities.slice(0, 10).map((entity, i) => (
                            <div key={i} className="flex items-center gap-2 text-sm">
                              <Badge variant="outline" className="text-xs">{entity.category}</Badge>
                              <span>{entity.text}</span>
                              <span className="text-xs text-muted-foreground ml-auto">
                                {(entity.score * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={() => setSelectedDocument(null)}
                    data-testid="button-close-details"
                  >
                    Close
                  </Button>
                </DialogFooter>
              </div>
            </DialogContent>
          </Dialog>
        )}

        {/* Add EHR Connection Dialog */}
        <Dialog open={showAddEHRDialog} onOpenChange={setShowAddEHRDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add EHR Connection</DialogTitle>
              <DialogDescription>
                Connect to your healthcare provider's EHR system for automatic record sync
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="ehr-system">EHR System</Label>
                <Select value={selectedSystem} onValueChange={setSelectedSystem}>
                  <SelectTrigger id="ehr-system" data-testid="select-ehr-system">
                    <SelectValue placeholder="Select your EHR system" />
                  </SelectTrigger>
                  <SelectContent>
                    {EHR_SYSTEMS.map((system) => (
                      <SelectItem key={system.id} value={system.id}>
                        <div>
                          <div className="font-medium">{system.name}</div>
                          <div className="text-xs text-muted-foreground">{system.description}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="facility-name">Facility/Hospital Name</Label>
                <Input
                  id="facility-name"
                  placeholder="e.g., Memorial Hospital"
                  value={facilityName}
                  onChange={(e) => setFacilityName(e.target.value)}
                  data-testid="input-facility-name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="patient-id">Your Patient ID</Label>
                <Input
                  id="patient-id"
                  placeholder="e.g., 123456789"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  data-testid="input-patient-id"
                />
                <p className="text-xs text-muted-foreground">
                  Find this on your patient portal or medical documents
                </p>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowAddEHRDialog(false)} data-testid="button-cancel-ehr">
                Cancel
              </Button>
              <Button
                onClick={handleAddEHRConnection}
                disabled={addEHRConnectionMutation.isPending}
                data-testid="button-connect-ehr"
              >
                {addEHRConnectionMutation.isPending ? "Connecting..." : "Connect"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
