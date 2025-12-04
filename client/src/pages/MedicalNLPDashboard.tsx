import { useState, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { 
  FileText, 
  Upload, 
  Bot, 
  Brain, 
  MessageSquare,
  Search,
  FileUp,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ShieldCheck,
  Eye,
  EyeOff,
  Download,
  Trash2,
  Clock,
  Stethoscope,
  Pill,
  Activity,
  AlertTriangle,
  RefreshCw,
  Send,
  User,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { format } from "date-fns";

interface MedicalDocument {
  id: string;
  filename: string;
  file_type: string;
  upload_date: string;
  status: 'processing' | 'completed' | 'error';
  extracted_text?: string;
  entities?: MedicalEntity[];
  phi_detected: boolean;
  phi_redacted: boolean;
}

interface MedicalEntity {
  text: string;
  type: 'condition' | 'medication' | 'procedure' | 'anatomy' | 'lab_value' | 'dosage';
  confidence: number;
  start: number;
  end: number;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: string[];
}

const ENTITY_COLORS: Record<string, string> = {
  condition: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
  medication: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
  procedure: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
  anatomy: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
  lab_value: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400',
  dosage: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-400',
};

const ENTITY_ICONS: Record<string, React.ElementType> = {
  condition: Activity,
  medication: Pill,
  procedure: Stethoscope,
  anatomy: User,
  lab_value: FileText,
  dosage: AlertTriangle,
};

export default function MedicalNLPDashboard() {
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedDoc, setSelectedDoc] = useState<MedicalDocument | null>(null);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [showRedacted, setShowRedacted] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const { data: documents, isLoading: docsLoading, refetch: refetchDocs } = useQuery<MedicalDocument[]>({
    queryKey: ['/api/medical-nlp/documents'],
  });

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('/api/medical-nlp/upload', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/medical-nlp/documents'] });
      toast({ title: 'Document Uploaded', description: 'Processing started...' });
    },
    onError: (error: Error) => {
      toast({ title: 'Upload Failed', description: error.message, variant: 'destructive' });
    },
  });

  const chatMutation = useMutation({
    mutationFn: async (message: string) => {
      return apiRequest<{ response: string; sources: string[] }>('/api/medical-nlp/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message,
          document_ids: selectedDoc ? [selectedDoc.id] : documents?.map(d => d.id) || [],
        }),
      });
    },
    onSuccess: (data) => {
      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
        sources: data.sources,
      };
      setChatMessages(prev => [...prev, assistantMessage]);
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
    },
    onError: (error: Error) => {
      toast({ title: 'Chat Error', description: error.message, variant: 'destructive' });
    },
  });

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadMutation.mutate(file);
    }
  };

  const handleSendMessage = () => {
    if (!chatInput.trim()) return;
    
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: chatInput,
      timestamp: new Date().toISOString(),
    };
    
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    chatMutation.mutate(chatInput);
  };

  const processedCount = documents?.filter(d => d.status === 'completed').length || 0;
  const entityCount = documents?.reduce((acc, d) => acc + (d.entities?.length || 0), 0) || 0;

  return (
    <div className="space-y-6" data-testid="page-medical-nlp">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="h-6 w-6 text-purple-500" />
            Medical NLP Dashboard
          </h1>
          <p className="text-muted-foreground">
            AI-powered medical document analysis with entity extraction and Q&A
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => refetchDocs()} data-testid="button-refresh">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button onClick={() => fileInputRef.current?.click()} data-testid="button-upload">
            <Upload className="h-4 w-4 mr-2" />
            Upload Document
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.txt,.doc,.docx"
            onChange={handleFileUpload}
            className="hidden"
            data-testid="input-file-upload"
          />
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card data-testid="card-documents-count">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{documents?.length || 0}</p>
            <p className="text-xs text-muted-foreground">{processedCount} processed</p>
          </CardContent>
        </Card>

        <Card data-testid="card-entities-count">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Entities Extracted</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{entityCount}</p>
            <p className="text-xs text-muted-foreground">Medical concepts identified</p>
          </CardContent>
        </Card>

        <Card data-testid="card-phi-status">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">PHI Protection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-6 w-6 text-green-500" />
              <span className="text-xl font-bold">Active</span>
            </div>
            <p className="text-xs text-muted-foreground">HIPAA-compliant processing</p>
          </CardContent>
        </Card>

        <Card data-testid="card-processing-status">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Processing</CardTitle>
          </CardHeader>
          <CardContent>
            {uploadMutation.isPending ? (
              <div className="flex items-center gap-2">
                <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                <span className="font-medium">In Progress</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                <span className="font-medium">Ready</span>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="documents" className="space-y-6">
        <TabsList className="grid w-full max-w-xl grid-cols-3">
          <TabsTrigger value="documents" className="gap-2" data-testid="tab-documents">
            <FileText className="h-4 w-4" />
            Documents
          </TabsTrigger>
          <TabsTrigger value="entities" className="gap-2" data-testid="tab-entities">
            <Search className="h-4 w-4" />
            Entities
          </TabsTrigger>
          <TabsTrigger value="chat" className="gap-2" data-testid="tab-chat">
            <MessageSquare className="h-4 w-4" />
            Q&A Chat
          </TabsTrigger>
        </TabsList>

        <TabsContent value="documents">
          <div className="grid gap-4 md:grid-cols-3">
            <Card className="md:col-span-1" data-testid="card-document-list">
              <CardHeader>
                <CardTitle className="text-base">Uploaded Documents</CardTitle>
              </CardHeader>
              <CardContent>
                {docsLoading ? (
                  <div className="space-y-2">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="h-16 bg-muted animate-pulse rounded-lg" />
                    ))}
                  </div>
                ) : documents && documents.length > 0 ? (
                  <ScrollArea className="h-[400px]">
                    <div className="space-y-2">
                      {documents.map((doc) => (
                        <div
                          key={doc.id}
                          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                            selectedDoc?.id === doc.id ? 'bg-accent border-primary' : 'hover:bg-muted/50'
                          }`}
                          onClick={() => setSelectedDoc(doc)}
                          data-testid={`doc-item-${doc.id}`}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1 min-w-0">
                              <p className="font-medium text-sm truncate">{doc.filename}</p>
                              <p className="text-xs text-muted-foreground">
                                {format(new Date(doc.upload_date), 'MMM d, yyyy')}
                              </p>
                            </div>
                            <Badge 
                              variant={doc.status === 'completed' ? 'secondary' : doc.status === 'error' ? 'destructive' : 'outline'}
                              className="text-xs"
                            >
                              {doc.status}
                            </Badge>
                          </div>
                          {doc.phi_detected && (
                            <div className="flex items-center gap-1 mt-2 text-xs">
                              <ShieldCheck className="h-3 w-3 text-amber-500" />
                              <span className="text-muted-foreground">PHI detected & redacted</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center py-8">
                    <FileUp className="h-10 w-10 mx-auto text-muted-foreground opacity-50 mb-3" />
                    <p className="text-sm font-medium mb-1">No documents yet</p>
                    <p className="text-xs text-muted-foreground">Upload a medical document to get started</p>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="md:col-span-2" data-testid="card-document-preview">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Document Preview</CardTitle>
                  {selectedDoc && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowRedacted(!showRedacted)}
                      data-testid="button-toggle-redaction"
                    >
                      {showRedacted ? (
                        <><EyeOff className="h-4 w-4 mr-1" /> Show Original</>
                      ) : (
                        <><Eye className="h-4 w-4 mr-1" /> Show Redacted</>
                      )}
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {selectedDoc ? (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                      <FileText className="h-5 w-5 text-muted-foreground" />
                      <span className="font-medium">{selectedDoc.filename}</span>
                      <Badge variant="outline">{selectedDoc.file_type}</Badge>
                    </div>

                    {selectedDoc.status === 'processing' ? (
                      <div className="flex flex-col items-center py-12">
                        <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                        <p className="text-sm font-medium">Processing document...</p>
                        <p className="text-xs text-muted-foreground">Extracting text and entities</p>
                      </div>
                    ) : selectedDoc.status === 'error' ? (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Processing Failed</AlertTitle>
                        <AlertDescription>
                          Unable to process this document. Please try uploading again.
                        </AlertDescription>
                      </Alert>
                    ) : (
                      <ScrollArea className="h-[350px] rounded-lg border p-4 bg-muted/30">
                        <div className="prose prose-sm dark:prose-invert max-w-none">
                          {selectedDoc.extracted_text ? (
                            <pre className="whitespace-pre-wrap font-sans text-sm">
                              {showRedacted && selectedDoc.phi_redacted
                                ? selectedDoc.extracted_text.replace(/\b[A-Z][a-z]+\s[A-Z][a-z]+\b/g, '[REDACTED]')
                                : selectedDoc.extracted_text
                              }
                            </pre>
                          ) : (
                            <p className="text-muted-foreground">No text extracted</p>
                          )}
                        </div>
                      </ScrollArea>
                    )}

                    {selectedDoc.entities && selectedDoc.entities.length > 0 && (
                      <div className="mt-4">
                        <h4 className="text-sm font-medium mb-2">Extracted Entities ({selectedDoc.entities.length})</h4>
                        <div className="flex flex-wrap gap-1">
                          {selectedDoc.entities.slice(0, 20).map((entity, idx) => {
                            const Icon = ENTITY_ICONS[entity.type] || FileText;
                            return (
                              <Badge 
                                key={idx} 
                                className={`gap-1 ${ENTITY_COLORS[entity.type]}`}
                              >
                                <Icon className="h-3 w-3" />
                                {entity.text}
                              </Badge>
                            );
                          })}
                          {selectedDoc.entities.length > 20 && (
                            <Badge variant="outline">+{selectedDoc.entities.length - 20} more</Badge>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center py-16">
                    <FileText className="h-12 w-12 text-muted-foreground opacity-50 mb-4" />
                    <p className="text-sm font-medium mb-1">Select a document</p>
                    <p className="text-xs text-muted-foreground">Click on a document to preview its contents</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="entities">
          <Card data-testid="card-entity-analysis">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5 text-purple-500" />
                Entity Analysis
              </CardTitle>
              <CardDescription>
                Medical entities extracted from all documents using AI-powered NLP
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                {['condition', 'medication', 'procedure', 'anatomy', 'lab_value', 'dosage'].map((type) => {
                  const Icon = ENTITY_ICONS[type];
                  const entities = documents
                    ?.flatMap(d => d.entities || [])
                    .filter(e => e.type === type) || [];
                  
                  return (
                    <Card key={type} className="bg-muted/30" data-testid={`card-entity-type-${type}`}>
                      <CardHeader className="pb-2">
                        <CardTitle className="flex items-center gap-2 text-sm">
                          <Icon className="h-4 w-4" />
                          {type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ')}
                          <Badge variant="secondary" className="ml-auto">{entities.length}</Badge>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        {entities.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {entities.slice(0, 8).map((entity, idx) => (
                              <Badge 
                                key={idx}
                                variant="outline"
                                className="text-xs"
                              >
                                {entity.text}
                              </Badge>
                            ))}
                            {entities.length > 8 && (
                              <Badge variant="outline" className="text-xs">+{entities.length - 8}</Badge>
                            )}
                          </div>
                        ) : (
                          <p className="text-xs text-muted-foreground">No entities found</p>
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="chat">
          <Card className="h-[600px] flex flex-col" data-testid="card-nlp-chat">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bot className="h-5 w-5 text-purple-500" />
                Medical Document Q&A
              </CardTitle>
              <CardDescription>
                Ask questions about your uploaded medical documents
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col min-h-0">
              <ScrollArea className="flex-1 pr-4">
                <div className="space-y-4 pb-4">
                  {chatMessages.length === 0 ? (
                    <div className="text-center py-12">
                      <MessageSquare className="h-10 w-10 mx-auto text-muted-foreground opacity-50 mb-3" />
                      <p className="text-sm font-medium mb-1">Start a conversation</p>
                      <p className="text-xs text-muted-foreground mb-4">
                        Ask questions about your medical documents
                      </p>
                      <div className="flex flex-wrap justify-center gap-2">
                        {[
                          'What medications are mentioned?',
                          'Summarize the diagnoses',
                          'List all lab values',
                          'What procedures are documented?',
                        ].map((suggestion) => (
                          <Button
                            key={suggestion}
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              setChatInput(suggestion);
                              handleSendMessage();
                            }}
                            data-testid={`button-suggestion-${suggestion.slice(0, 10)}`}
                          >
                            {suggestion}
                          </Button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    chatMessages.map((msg) => (
                      <div
                        key={msg.id}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        data-testid={`chat-message-${msg.id}`}
                      >
                        <div
                          className={`max-w-[80%] rounded-lg p-3 ${
                            msg.role === 'user'
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-muted'
                          }`}
                        >
                          <p className="text-sm">{msg.content}</p>
                          {msg.sources && msg.sources.length > 0 && (
                            <div className="mt-2 pt-2 border-t border-border/50">
                              <p className="text-xs opacity-70">Sources:</p>
                              {msg.sources.map((src, idx) => (
                                <span key={idx} className="text-xs opacity-70">{src}</span>
                              ))}
                            </div>
                          )}
                          <p className="text-xs opacity-50 mt-1">
                            {format(new Date(msg.timestamp), 'HH:mm')}
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                  <div ref={chatEndRef} />
                </div>
              </ScrollArea>
              
              <div className="flex gap-2 pt-4 border-t">
                <Input
                  placeholder="Ask about your medical documents..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                  disabled={chatMutation.isPending}
                  data-testid="input-chat"
                />
                <Button 
                  onClick={handleSendMessage}
                  disabled={!chatInput.trim() || chatMutation.isPending}
                  data-testid="button-send-chat"
                >
                  {chatMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
