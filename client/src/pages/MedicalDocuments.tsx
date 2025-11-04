import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { Upload, FileText, Loader2, CheckCircle2, XCircle, Trash2, Eye } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

type MedicalDocument = {
  id: string;
  userId: string;
  fileName: string;
  fileType: string;
  fileSize: number;
  fileUrl: string;
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
  };
  documentType?: string;
  documentDate?: string;
  processingStatus?: string;
  errorMessage?: string;
  createdAt: string;
  updatedAt: string;
};

export default function MedicalDocuments() {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState<string>("other");
  const [documentDate, setDocumentDate] = useState<string>("");
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [selectedDocument, setSelectedDocument] = useState<MedicalDocument | null>(null);

  const { data: documents = [], isLoading } = useQuery<MedicalDocument[]>({
    queryKey: ["/api/medical-documents"],
  });

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
        description: "Your medical document has been uploaded and is being processed.",
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
    onError: (error) => {
      toast({
        title: "Delete Failed",
        description: error instanceof Error ? error.message : "Failed to delete document",
        variant: "destructive",
      });
    },
  });

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

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-primary" />;
      case "processing":
      case "pending":
        return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-destructive" />;
      default:
        return <FileText className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case "completed":
        return <Badge variant="default">Processed</Badge>;
      case "processing":
        return <Badge variant="secondary">Processing...</Badge>;
      case "pending":
        return <Badge variant="secondary">Pending</Badge>;
      case "failed":
        return <Badge variant="destructive">Failed</Badge>;
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold">Medical Documents</h1>
        <p className="text-muted-foreground">
          Upload and manage your medical documents with automatic OCR text extraction
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Upload Document</CardTitle>
              <CardDescription>
                Upload medical records, lab results, prescriptions, or imaging reports
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
                          Uploading...
                        </>
                      ) : (
                        <>
                          <Upload className="mr-2 h-4 w-4" />
                          Upload
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
        </div>

        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Your Documents</CardTitle>
              <CardDescription>
                View and manage your uploaded medical documents
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : documents.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  <FileText className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No documents uploaded yet</p>
                </div>
              ) : (
                <ScrollArea className="h-[600px]">
                  <div className="space-y-4">
                    {documents.map((doc) => (
                      <Card key={doc.id} className="hover-elevate" data-testid={`card-document-${doc.id}`}>
                        <CardHeader>
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex items-start gap-3 flex-1">
                              {getStatusIcon(doc.processingStatus)}
                              <div className="flex-1 min-w-0">
                                <h3 className="font-medium truncate">{doc.fileName}</h3>
                                <p className="text-sm text-muted-foreground">
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
                                      <Badge key={i} variant="outline">{med}</Badge>
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
        </div>
      </div>

      {selectedDocument && (
        <Card>
          <CardHeader>
            <CardTitle>Document Details</CardTitle>
            <CardDescription>{selectedDocument.fileName}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium">Type</p>
                <p className="text-sm text-muted-foreground capitalize">
                  {selectedDocument.documentType?.replace('_', ' ')}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium">Size</p>
                <p className="text-sm text-muted-foreground">
                  {(selectedDocument.fileSize / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>

            <Separator />

            {selectedDocument.extractedText && (
              <div>
                <h3 className="font-medium mb-2">Extracted Text</h3>
                <ScrollArea className="h-[200px] border rounded-md p-4">
                  <pre className="text-sm whitespace-pre-wrap">{selectedDocument.extractedText}</pre>
                </ScrollArea>
              </div>
            )}

            {selectedDocument.extractedData && (
              <div className="space-y-4">
                <h3 className="font-medium">Extracted Medical Information</h3>
                
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

                {selectedDocument.extractedData.procedures && selectedDocument.extractedData.procedures.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Procedures</p>
                    <div className="flex flex-wrap gap-2">
                      {selectedDocument.extractedData.procedures.map((proc, i) => (
                        <Badge key={i} variant="outline">{proc}</Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            <Button
              variant="outline"
              onClick={() => setSelectedDocument(null)}
              data-testid="button-close-details"
            >
              Close
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
