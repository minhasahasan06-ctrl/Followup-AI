import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { apiRequest, queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { 
  FileText, Plus, Eye, Download, Loader2, CheckCircle, XCircle, 
  Clock, Shield, AlertTriangle, RefreshCw, Sparkles, Lock,
  User, Calendar, Hash, Phone, Mail, MapPin, CreditCard
} from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

interface NLPDocument {
  id: string;
  sourceType: string;
  sourceUri: string;
  patientId?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  originalText?: string;
  redactedUri?: string;
  createdAt: string;
  createdBy: string;
  redactionRuns: RedactionRun[];
}

interface RedactionRun {
  id: string;
  status: string;
  entitiesDetected: number;
  entitiesRedacted: number;
  createdAt: string;
}

interface PHIEntity {
  type: string;
  text: string;
  start: number;
  end: number;
  confidence: number;
}

const STATUS_CONFIG = {
  pending: { label: 'Pending', icon: Clock, color: 'text-blue-500', badge: 'secondary' as const },
  processing: { label: 'Processing', icon: Loader2, color: 'text-amber-500', badge: 'default' as const },
  completed: { label: 'Completed', icon: CheckCircle, color: 'text-emerald-500', badge: 'default' as const },
  failed: { label: 'Failed', icon: XCircle, color: 'text-red-500', badge: 'destructive' as const },
};

const PHI_TYPE_CONFIG: Record<string, { icon: typeof User; color: string; label: string }> = {
  'PERSON_NAME': { icon: User, color: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-100', label: 'Name' },
  'DATE_OF_BIRTH': { icon: Calendar, color: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-100', label: 'DOB' },
  'SSN': { icon: Hash, color: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100', label: 'SSN' },
  'PHONE_NUMBER': { icon: Phone, color: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100', label: 'Phone' },
  'EMAIL': { icon: Mail, color: 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-100', label: 'Email' },
  'ADDRESS': { icon: MapPin, color: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-100', label: 'Address' },
  'MRN': { icon: CreditCard, color: 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-100', label: 'MRN' },
};

const SOURCE_TYPES = [
  { value: 'clinical_note', label: 'Clinical Note' },
  { value: 'lab_report', label: 'Lab Report' },
  { value: 'discharge_summary', label: 'Discharge Summary' },
  { value: 'consultation', label: 'Consultation' },
  { value: 'imaging_report', label: 'Imaging Report' },
  { value: 'other', label: 'Other' },
];

export function NLPRedactionTab() {
  const { toast } = useToast();
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<NLPDocument | null>(null);
  const [newDocument, setNewDocument] = useState({
    sourceType: 'clinical_note',
    sourceUri: '',
    patientId: '',
    text: '',
  });

  const { data: documents = [], isLoading, refetch } = useQuery<NLPDocument[]>({
    queryKey: ['/python-api/v1/research-center/nlp/documents'],
  });

  const createDocumentMutation = useMutation({
    mutationFn: async (data: typeof newDocument) => {
      const response = await apiRequest('POST', '/python-api/v1/research-center/nlp/documents', {
        source_type: data.sourceType,
        source_uri: data.sourceUri || `inline://${Date.now()}`,
        patient_id: data.patientId || undefined,
        text: data.text,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/python-api/v1/research-center/nlp/documents'] });
      setUploadDialogOpen(false);
      resetForm();
      toast({ title: 'Document Created', description: 'Document uploaded successfully. You can now process it for PHI redaction.' });
    },
    onError: (error: Error) => {
      toast({ title: 'Upload Failed', description: error.message, variant: 'destructive' });
    },
  });

  const processDocumentMutation = useMutation({
    mutationFn: async (documentId: string) => {
      const response = await apiRequest('POST', `/python-api/v1/research-center/nlp/documents/${documentId}/process`);
      return response.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/python-api/v1/research-center/nlp/documents'] });
      toast({ 
        title: 'Redaction Complete', 
        description: `Detected ${data.entitiesDetected} PHI entities, redacted ${data.entitiesRedacted}.`
      });
    },
    onError: (error: Error) => {
      toast({ title: 'Processing Failed', description: error.message, variant: 'destructive' });
    },
  });

  const resetForm = () => {
    setNewDocument({
      sourceType: 'clinical_note',
      sourceUri: '',
      patientId: '',
      text: '',
    });
  };

  const handleViewDocument = (doc: NLPDocument) => {
    setSelectedDocument(doc);
    setViewDialogOpen(true);
  };

  const pendingDocs = documents.filter(d => d.status === 'pending');
  const completedDocs = documents.filter(d => d.status === 'completed');
  const totalEntitiesRedacted = documents.reduce((sum, d) => 
    sum + (d.redactionRuns?.[0]?.entitiesRedacted || 0), 0
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold" data-testid="text-nlp-title">PHI Redaction</h2>
          <p className="text-muted-foreground">AI-powered PHI detection and redaction for research documents</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={() => refetch()} data-testid="button-refresh-nlp">
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button onClick={() => setUploadDialogOpen(true)} data-testid="button-upload-document">
            <Plus className="h-4 w-4 mr-2" />
            Upload Document
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card data-testid="card-total-documents">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4 text-primary" />
              Total Documents
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{documents.length}</div>
          </CardContent>
        </Card>
        <Card data-testid="card-pending-redaction">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Clock className="h-4 w-4 text-amber-500" />
              Pending Redaction
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{pendingDocs.length}</div>
          </CardContent>
        </Card>
        <Card data-testid="card-completed-redaction">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-emerald-500" />
              Redacted
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{completedDocs.length}</div>
          </CardContent>
        </Card>
        <Card data-testid="card-entities-redacted">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Shield className="h-4 w-4 text-red-500" />
              Entities Redacted
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalEntitiesRedacted}</div>
          </CardContent>
        </Card>
      </div>

      <Card data-testid="card-documents-list">
        <CardHeader>
          <CardTitle>Document Library</CardTitle>
          <CardDescription>Upload clinical documents for automated PHI redaction</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => <Skeleton key={i} className="h-16 w-full" />)}
            </div>
          ) : documents.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No documents uploaded yet.</p>
              <Button variant="outline" className="mt-4" onClick={() => setUploadDialogOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Upload First Document
              </Button>
            </div>
          ) : (
            <ScrollArea className="h-[400px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Document</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Entities</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {documents.map((doc) => {
                    const StatusIcon = STATUS_CONFIG[doc.status].icon;
                    const latestRun = doc.redactionRuns?.[0];
                    return (
                      <TableRow key={doc.id} data-testid={`document-row-${doc.id}`}>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <FileText className="h-4 w-4 text-muted-foreground" />
                            <span className="font-medium truncate max-w-[200px]">
                              {doc.sourceUri.startsWith('inline://') ? 'Inline Document' : doc.sourceUri.split('/').pop()}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">
                            {SOURCE_TYPES.find(t => t.value === doc.sourceType)?.label || doc.sourceType}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant={STATUS_CONFIG[doc.status].badge} className="gap-1">
                            <StatusIcon className={`h-3 w-3 ${doc.status === 'processing' ? 'animate-spin' : ''}`} />
                            {STATUS_CONFIG[doc.status].label}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {latestRun ? (
                            <div className="flex items-center gap-1">
                              <Shield className="h-3 w-3 text-red-500" />
                              <span>{latestRun.entitiesRedacted}</span>
                              <span className="text-muted-foreground">/ {latestRun.entitiesDetected}</span>
                            </div>
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <div className="text-sm">
                            {formatDistanceToNow(new Date(doc.createdAt), { addSuffix: true })}
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            <Button 
                              variant="ghost" 
                              size="icon"
                              onClick={() => handleViewDocument(doc)}
                              data-testid={`button-view-doc-${doc.id}`}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                            {doc.status === 'pending' && (
                              <Button 
                                variant="ghost" 
                                size="icon"
                                onClick={() => processDocumentMutation.mutate(doc.id)}
                                disabled={processDocumentMutation.isPending}
                                data-testid={`button-process-${doc.id}`}
                              >
                                {processDocumentMutation.isPending ? (
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                  <Sparkles className="h-4 w-4" />
                                )}
                              </Button>
                            )}
                            {doc.status === 'completed' && doc.redactedUri && (
                              <Button 
                                variant="ghost" 
                                size="icon"
                                onClick={() => window.open(doc.redactedUri, '_blank')}
                                data-testid={`button-download-${doc.id}`}
                              >
                                <Download className="h-4 w-4" />
                              </Button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      <Dialog open={uploadDialogOpen} onOpenChange={setUploadDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Upload Document for Redaction</DialogTitle>
            <DialogDescription>
              Upload clinical documents to detect and redact PHI using AI
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Document Type</Label>
                <Select 
                  value={newDocument.sourceType} 
                  onValueChange={(v) => setNewDocument({ ...newDocument, sourceType: v })}
                >
                  <SelectTrigger data-testid="select-source-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SOURCE_TYPES.map((type) => (
                      <SelectItem key={type.value} value={type.value}>{type.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Patient ID (Optional)</Label>
                <Input 
                  placeholder="e.g., PT-12345"
                  value={newDocument.patientId}
                  onChange={(e) => setNewDocument({ ...newDocument, patientId: e.target.value })}
                  data-testid="input-patient-id"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Document Text</Label>
              <Textarea 
                placeholder="Paste clinical document text here for PHI detection..."
                value={newDocument.text}
                onChange={(e) => setNewDocument({ ...newDocument, text: e.target.value })}
                className="min-h-[200px] font-mono text-sm"
                data-testid="input-document-text"
              />
            </div>

            <div className="flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-lg">
              <AlertTriangle className="h-5 w-5 text-amber-600 mt-0.5" />
              <div className="text-sm">
                <div className="font-medium text-amber-800 dark:text-amber-200">HIPAA Notice</div>
                <p className="text-amber-700 dark:text-amber-300">
                  Document processing is logged for compliance. PHI entities will be detected and redacted before research use.
                </p>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setUploadDialogOpen(false)} data-testid="button-cancel-upload">
              Cancel
            </Button>
            <Button 
              onClick={() => createDocumentMutation.mutate(newDocument)}
              disabled={!newDocument.text.trim() || createDocumentMutation.isPending}
              data-testid="button-submit-document"
            >
              {createDocumentMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Plus className="h-4 w-4 mr-2" />
                  Upload Document
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={viewDialogOpen} onOpenChange={setViewDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Document Details
              <Badge variant={selectedDocument ? STATUS_CONFIG[selectedDocument.status].badge : 'secondary'}>
                {selectedDocument && STATUS_CONFIG[selectedDocument.status].label}
              </Badge>
            </DialogTitle>
          </DialogHeader>

          {selectedDocument && (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="p-3 border rounded-lg">
                  <div className="text-sm text-muted-foreground">Type</div>
                  <div className="font-medium">
                    {SOURCE_TYPES.find(t => t.value === selectedDocument.sourceType)?.label}
                  </div>
                </div>
                <div className="p-3 border rounded-lg">
                  <div className="text-sm text-muted-foreground">Created</div>
                  <div className="font-medium text-sm">
                    {format(new Date(selectedDocument.createdAt), 'MMM d, yyyy h:mm a')}
                  </div>
                </div>
                <div className="p-3 border rounded-lg">
                  <div className="text-sm text-muted-foreground">Entities Redacted</div>
                  <div className="font-medium">
                    {selectedDocument.redactionRuns?.[0]?.entitiesRedacted || 0}
                  </div>
                </div>
              </div>

              {selectedDocument.redactionRuns && selectedDocument.redactionRuns.length > 0 && (
                <div className="p-3 border rounded-lg">
                  <div className="text-sm font-medium mb-2">Redaction History</div>
                  <div className="space-y-2">
                    {selectedDocument.redactionRuns.map((run) => (
                      <div key={run.id} className="flex items-center justify-between text-sm p-2 bg-muted rounded">
                        <div className="flex items-center gap-2">
                          <Shield className="h-4 w-4 text-emerald-500" />
                          <span>Detected {run.entitiesDetected}, Redacted {run.entitiesRedacted}</span>
                        </div>
                        <span className="text-muted-foreground">
                          {formatDistanceToNow(new Date(run.createdAt), { addSuffix: true })}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {selectedDocument.status === 'pending' && (
                <Button 
                  className="w-full"
                  onClick={() => {
                    processDocumentMutation.mutate(selectedDocument.id);
                    setViewDialogOpen(false);
                  }}
                  disabled={processDocumentMutation.isPending}
                  data-testid="button-process-from-dialog"
                >
                  <Sparkles className="h-4 w-4 mr-2" />
                  Process for PHI Redaction
                </Button>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
