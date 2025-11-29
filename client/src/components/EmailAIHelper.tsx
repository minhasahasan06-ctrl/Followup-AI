import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { 
  Mail, 
  Send, 
  Sparkles, 
  Loader2, 
  RefreshCw, 
  Copy, 
  CheckCircle,
  AlertCircle,
  Clock,
  FileText,
  Inbox,
  Edit3,
  Wand2,
  ChevronRight,
  MailPlus,
  Link2,
  CheckCircle2,
  Unplug
} from "lucide-react";
import { format } from "date-fns";

interface IntegrationStatus {
  gmail: { connected: boolean; email?: string; lastSync?: string };
  whatsapp: { connected: boolean; number?: string; lastSync?: string };
  twilio: { connected: boolean; number?: string; lastSync?: string };
}

interface EmailThread {
  id: string;
  subject: string;
  fromEmail: string;
  fromName: string;
  patientId?: string;
  category: string;
  priority: string;
  status: string;
  unread: boolean;
  lastMessageAt: string;
  createdAt: string;
}

interface EmailReplyResult {
  suggestedReply: string;
  tone: string;
  keyPoints: string[];
  alternativeResponses?: string[];
}

interface EmailCategorizationResult {
  category: string;
  priority: string;
  suggestedResponse?: string;
  requiresImmediateAttention: boolean;
  confidenceScore: number;
  reasoning: string;
}

const priorityColors: Record<string, string> = {
  urgent: "bg-red-500/10 text-red-600 dark:text-red-400",
  high: "bg-orange-500/10 text-orange-600 dark:text-orange-400",
  normal: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
  low: "bg-gray-500/10 text-gray-600 dark:text-gray-400",
};

const categoryIcons: Record<string, JSX.Element> = {
  urgent: <AlertCircle className="h-3 w-3" />,
  appointment: <Clock className="h-3 w-3" />,
  prescription: <FileText className="h-3 w-3" />,
  routine: <Inbox className="h-3 w-3" />,
  general: <Mail className="h-3 w-3" />,
};

export function EmailAIHelper() {
  const { toast } = useToast();
  const [selectedThread, setSelectedThread] = useState<EmailThread | null>(null);
  const [composeOpen, setComposeOpen] = useState(false);
  const [draftEmail, setDraftEmail] = useState("");
  const [emailSubject, setEmailSubject] = useState("");
  const [recipientEmail, setRecipientEmail] = useState("");
  const [selectedTone, setSelectedTone] = useState<string>("professional");
  const [generatedReply, setGeneratedReply] = useState<EmailReplyResult | null>(null);
  const [copied, setCopied] = useState(false);

  const { data: integrationStatus, isLoading: statusLoading } = useQuery<IntegrationStatus>({
    queryKey: ["/api/v1/integrations/status"],
  });

  const gmailConnected = integrationStatus?.gmail?.connected === true;

  const { data: emailThreads = [], isLoading: threadsLoading } = useQuery<EmailThread[]>({
    queryKey: ["/api/v1/emails/threads"],
    enabled: gmailConnected,
  });

  const getGmailAuthUrl = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/gmail/auth-url");
      return response.json();
    },
    onSuccess: (data: { authUrl: string }) => {
      window.location.href = data.authUrl;
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to start Gmail connection. Please try again.",
        variant: "destructive",
      });
    },
  });

  const syncGmail = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/gmail/sync", { method: "POST" });
      return response.json();
    },
    onSuccess: (data: { emailsFetched: number }) => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/status"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/emails/threads"] });
      toast({
        title: "Sync Complete",
        description: `Successfully synced ${data.emailsFetched} new emails.`,
      });
    },
    onError: () => {
      toast({
        title: "Sync Failed",
        description: "Failed to sync emails. Please try again.",
        variant: "destructive",
      });
    },
  });

  const disconnectGmail = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/gmail/disconnect", { method: "POST" });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/status"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/emails/threads"] });
      toast({
        title: "Disconnected",
        description: "Gmail has been disconnected successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to disconnect Gmail. Please try again.",
        variant: "destructive",
      });
    },
  });

  const generateReplyMutation = useMutation({
    mutationFn: async (threadId: string) => {
      const response = await apiRequest(`/api/v1/emails/threads/${threadId}/generate-reply`, {
        method: "POST",
        json: { tone: selectedTone },
      });
      return response.json();
    },
    onSuccess: (data) => {
      setGeneratedReply(data);
      toast({
        title: "Reply Generated",
        description: "AI has drafted a response for you.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to generate reply. Please try again.",
        variant: "destructive",
      });
    },
  });

  const categorizeMutation = useMutation({
    mutationFn: async ({ subject, content }: { subject: string; content: string }) => {
      const response = await apiRequest("/api/v1/emails/categorize", {
        method: "POST",
        json: { subject, content },
      });
      return response.json();
    },
    onSuccess: (data: EmailCategorizationResult) => {
      toast({
        title: "Email Categorized",
        description: `Category: ${data.category}, Priority: ${data.priority}`,
      });
    },
  });

  const enhanceDraftMutation = useMutation({
    mutationFn: async (content: string) => {
      const response = await apiRequest("/api/chat/send", {
        method: "POST",
        json: {
          content: `Please enhance this email draft to be more professional and clear while maintaining HIPAA compliance. Keep the same meaning but improve the writing:\n\n${content}`,
          agentType: "lysa",
        },
      });
      return response.json();
    },
    onSuccess: (data) => {
      if (data.response) {
        setDraftEmail(data.response);
        toast({
          title: "Draft Enhanced",
          description: "Your email has been professionally refined.",
        });
      }
    },
  });

  const sendEmailMutation = useMutation({
    mutationFn: async ({ to, subject, body }: { to: string; subject: string; body: string }) => {
      const response = await apiRequest("/api/v1/emails/messages", {
        method: "POST",
        json: {
          to,
          subject,
          body,
          isFromDoctor: true,
        },
      });
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Email Sent",
        description: "Your message has been delivered successfully.",
      });
      setComposeOpen(false);
      setDraftEmail("");
      setEmailSubject("");
      setRecipientEmail("");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/emails/threads"] });
    },
  });

  const handleCopyReply = () => {
    if (generatedReply) {
      navigator.clipboard.writeText(generatedReply.suggestedReply);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleUseReply = () => {
    if (generatedReply) {
      setDraftEmail(generatedReply.suggestedReply);
      setComposeOpen(true);
    }
  };

  if (statusLoading) {
    return (
      <Card className="h-full flex flex-col">
        <CardContent className="flex-1 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (!gmailConnected) {
    return (
      <Card className="h-full flex flex-col" data-testid="card-gmail-connect">
        <CardHeader className="flex-shrink-0 pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Mail className="h-5 w-5" />
                Email Assistant
              </CardTitle>
              <CardDescription>Connect Gmail for inbox sync, or compose manually</CardDescription>
            </div>
            <Dialog open={composeOpen} onOpenChange={setComposeOpen}>
              <DialogTrigger asChild>
                <Button size="sm" data-testid="button-compose-email-manual">
                  <MailPlus className="h-4 w-4 mr-1" />
                  Compose
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle className="flex items-center gap-2">
                    <Edit3 className="h-5 w-5" />
                    Compose Email with AI
                  </DialogTitle>
                </DialogHeader>
                <div className="space-y-4 pt-4">
                  <div className="space-y-2">
                    <Label htmlFor="recipient-manual">To</Label>
                    <Input
                      id="recipient-manual"
                      placeholder="patient@example.com"
                      value={recipientEmail}
                      onChange={(e) => setRecipientEmail(e.target.value)}
                      data-testid="input-email-recipient-manual"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="subject-manual">Subject</Label>
                    <Input
                      id="subject-manual"
                      placeholder="Re: Your appointment confirmation"
                      value={emailSubject}
                      onChange={(e) => setEmailSubject(e.target.value)}
                      data-testid="input-email-subject-manual"
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="body-manual">Message</Label>
                      <div className="flex gap-2">
                        <Select value={selectedTone} onValueChange={setSelectedTone}>
                          <SelectTrigger className="w-32 h-8">
                            <SelectValue placeholder="Tone" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="professional">Professional</SelectItem>
                            <SelectItem value="warm">Warm</SelectItem>
                            <SelectItem value="empathetic">Empathetic</SelectItem>
                            <SelectItem value="concise">Concise</SelectItem>
                          </SelectContent>
                        </Select>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => enhanceDraftMutation.mutate(draftEmail)}
                          disabled={!draftEmail.trim() || enhanceDraftMutation.isPending}
                          data-testid="button-enhance-email-manual"
                        >
                          {enhanceDraftMutation.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <>
                              <Wand2 className="h-4 w-4 mr-1" />
                              Enhance
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                    <Textarea
                      id="body-manual"
                      placeholder="Type your message here... AI will help you refine it."
                      value={draftEmail}
                      onChange={(e) => setDraftEmail(e.target.value)}
                      rows={10}
                      className="resize-none"
                      data-testid="input-email-body-manual"
                    />
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button
                      variant="outline"
                      onClick={() => setComposeOpen(false)}
                      data-testid="button-cancel-email-manual"
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={() => {
                        navigator.clipboard.writeText(`To: ${recipientEmail}\nSubject: ${emailSubject}\n\n${draftEmail}`);
                        toast({
                          title: "Copied to Clipboard",
                          description: "Email content copied. Paste into your email client to send.",
                        });
                        setComposeOpen(false);
                      }}
                      disabled={!recipientEmail || !emailSubject || !draftEmail}
                      data-testid="button-copy-email-manual"
                    >
                      <Copy className="h-4 w-4 mr-1" />
                      Copy Email
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col items-center justify-center py-8">
          <div className="h-20 w-20 rounded-full bg-red-500/10 flex items-center justify-center mb-6">
            <Mail className="h-10 w-10 text-red-600" />
          </div>
          <h3 className="text-lg font-semibold mb-2" data-testid="text-gmail-connect-title">
            Connect Your Gmail
          </h3>
          <p className="text-muted-foreground text-center max-w-sm mb-6">
            Link your Gmail account to automatically sync patient emails, categorize messages with AI, and draft professional replies.
          </p>
          <div className="space-y-3 w-full max-w-xs">
            <Button 
              onClick={() => getGmailAuthUrl.mutate()}
              disabled={getGmailAuthUrl.isPending}
              className="w-full"
              data-testid="button-connect-gmail"
            >
              {getGmailAuthUrl.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Link2 className="h-4 w-4 mr-2" />
              )}
              Connect Gmail Account
            </Button>
            <p className="text-xs text-muted-foreground text-center">
              Your credentials are encrypted and stored securely. HIPAA compliant.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex-shrink-0 pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5" />
              Email Assistant
            </CardTitle>
            <CardDescription className="flex items-center gap-2">
              <Badge variant="secondary" className="bg-green-500/10 text-green-600 text-xs">
                <CheckCircle2 className="h-3 w-3 mr-1" />
                {integrationStatus?.gmail?.email || "Connected"}
              </Badge>
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-6 px-2 text-xs"
                onClick={() => syncGmail.mutate()}
                disabled={syncGmail.isPending}
                data-testid="button-sync-gmail"
              >
                {syncGmail.isPending ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <RefreshCw className="h-3 w-3" />
                )}
              </Button>
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-6 px-2 text-xs text-muted-foreground"
                onClick={() => disconnectGmail.mutate()}
                disabled={disconnectGmail.isPending}
                data-testid="button-disconnect-gmail"
              >
                <Unplug className="h-3 w-3" />
              </Button>
            </CardDescription>
          </div>
          <Dialog open={composeOpen} onOpenChange={setComposeOpen}>
            <DialogTrigger asChild>
              <Button size="sm" data-testid="button-compose-email">
                <MailPlus className="h-4 w-4 mr-1" />
                Compose
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Edit3 className="h-5 w-5" />
                  Compose Email with AI
                </DialogTitle>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <div className="space-y-2">
                  <Label htmlFor="recipient">To</Label>
                  <Input
                    id="recipient"
                    placeholder="patient@example.com"
                    value={recipientEmail}
                    onChange={(e) => setRecipientEmail(e.target.value)}
                    data-testid="input-email-recipient"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="subject">Subject</Label>
                  <Input
                    id="subject"
                    placeholder="Re: Your appointment confirmation"
                    value={emailSubject}
                    onChange={(e) => setEmailSubject(e.target.value)}
                    data-testid="input-email-subject"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="body">Message</Label>
                    <div className="flex gap-2">
                      <Select value={selectedTone} onValueChange={setSelectedTone}>
                        <SelectTrigger className="w-32 h-8">
                          <SelectValue placeholder="Tone" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="professional">Professional</SelectItem>
                          <SelectItem value="warm">Warm</SelectItem>
                          <SelectItem value="empathetic">Empathetic</SelectItem>
                          <SelectItem value="concise">Concise</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => enhanceDraftMutation.mutate(draftEmail)}
                        disabled={!draftEmail.trim() || enhanceDraftMutation.isPending}
                        data-testid="button-enhance-email"
                      >
                        {enhanceDraftMutation.isPending ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <>
                            <Wand2 className="h-4 w-4 mr-1" />
                            Enhance
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                  <Textarea
                    id="body"
                    placeholder="Type your message here... AI will help you refine it."
                    value={draftEmail}
                    onChange={(e) => setDraftEmail(e.target.value)}
                    rows={10}
                    className="resize-none"
                    data-testid="input-email-body"
                  />
                </div>
                <div className="flex justify-end gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setComposeOpen(false)}
                    data-testid="button-cancel-email"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={() => sendEmailMutation.mutate({
                      to: recipientEmail,
                      subject: emailSubject,
                      body: draftEmail,
                    })}
                    disabled={!recipientEmail || !emailSubject || !draftEmail || sendEmailMutation.isPending}
                    data-testid="button-send-email"
                  >
                    {sendEmailMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin mr-1" />
                    ) : (
                      <Send className="h-4 w-4 mr-1" />
                    )}
                    Send
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>

      <CardContent className="flex-1 min-h-0 p-0">
        <Tabs defaultValue="inbox" className="h-full flex flex-col">
          <div className="px-4">
            <TabsList className="w-full grid grid-cols-2">
              <TabsTrigger value="inbox" data-testid="tab-email-inbox">
                <Inbox className="h-4 w-4 mr-1" />
                Inbox
              </TabsTrigger>
              <TabsTrigger value="reply" data-testid="tab-email-reply">
                <Sparkles className="h-4 w-4 mr-1" />
                AI Reply
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="inbox" className="flex-1 min-h-0 mt-3 px-4">
            <ScrollArea className="h-[300px]">
              {threadsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : emailThreads.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Inbox className="h-12 w-12 mx-auto mb-3 opacity-30" />
                  <p className="text-sm font-medium">No emails yet</p>
                  <p className="text-xs">Incoming emails will appear here</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {emailThreads.map((thread) => (
                    <div
                      key={thread.id}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedThread?.id === thread.id
                          ? "bg-primary/5 border-primary/30"
                          : "hover-elevate"
                      } ${thread.unread ? "bg-primary/5" : ""}`}
                      onClick={() => setSelectedThread(thread)}
                      data-testid={`email-thread-${thread.id}`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className={`font-medium text-sm truncate ${thread.unread ? "" : "text-muted-foreground"}`}>
                              {thread.fromName || thread.fromEmail}
                            </span>
                            {thread.unread && (
                              <div className="w-2 h-2 rounded-full bg-primary flex-shrink-0" />
                            )}
                          </div>
                          <p className="text-sm truncate">{thread.subject}</p>
                        </div>
                        <div className="flex flex-col items-end gap-1 flex-shrink-0">
                          <Badge variant="secondary" className={`text-xs ${priorityColors[thread.priority] || ""}`}>
                            {categoryIcons[thread.category]}
                            <span className="ml-1">{thread.category}</span>
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {format(new Date(thread.lastMessageAt || thread.createdAt), "MMM d")}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </TabsContent>

          <TabsContent value="reply" className="flex-1 min-h-0 mt-3 px-4">
            {selectedThread ? (
              <div className="space-y-4">
                <div className="p-3 rounded-lg bg-muted/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Replying to:</span>
                    <Badge variant="secondary" className={priorityColors[selectedThread.priority]}>
                      {selectedThread.priority}
                    </Badge>
                  </div>
                  <p className="text-sm">{selectedThread.subject}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    From: {selectedThread.fromName || selectedThread.fromEmail}
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <Select value={selectedTone} onValueChange={setSelectedTone}>
                    <SelectTrigger className="flex-1">
                      <SelectValue placeholder="Select tone" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="professional">Professional</SelectItem>
                      <SelectItem value="warm">Warm & Friendly</SelectItem>
                      <SelectItem value="empathetic">Empathetic</SelectItem>
                      <SelectItem value="concise">Brief & Concise</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button
                    onClick={() => generateReplyMutation.mutate(selectedThread.id)}
                    disabled={generateReplyMutation.isPending}
                    data-testid="button-generate-reply"
                  >
                    {generateReplyMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin mr-1" />
                    ) : (
                      <Sparkles className="h-4 w-4 mr-1" />
                    )}
                    Generate Reply
                  </Button>
                </div>

                {generatedReply && (
                  <div className="space-y-3">
                    <Separator />
                    <div className="p-3 rounded-lg border bg-card">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Sparkles className="h-4 w-4 text-primary" />
                          <span className="text-sm font-medium">AI-Generated Reply</span>
                        </div>
                        <Badge variant="outline">{generatedReply.tone}</Badge>
                      </div>
                      <ScrollArea className="h-[120px]">
                        <p className="text-sm whitespace-pre-wrap">{generatedReply.suggestedReply}</p>
                      </ScrollArea>
                      {generatedReply.keyPoints.length > 0 && (
                        <div className="mt-3 pt-3 border-t">
                          <span className="text-xs font-medium text-muted-foreground">Key Points:</span>
                          <ul className="mt-1 space-y-1">
                            {generatedReply.keyPoints.map((point, i) => (
                              <li key={i} className="text-xs flex items-start gap-1">
                                <ChevronRight className="h-3 w-3 mt-0.5 flex-shrink-0" />
                                <span>{point}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        className="flex-1"
                        onClick={handleCopyReply}
                        data-testid="button-copy-reply"
                      >
                        {copied ? (
                          <>
                            <CheckCircle className="h-4 w-4 mr-1" />
                            Copied!
                          </>
                        ) : (
                          <>
                            <Copy className="h-4 w-4 mr-1" />
                            Copy
                          </>
                        )}
                      </Button>
                      <Button className="flex-1" onClick={handleUseReply} data-testid="button-use-reply">
                        <Edit3 className="h-4 w-4 mr-1" />
                        Edit & Send
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Mail className="h-12 w-12 mx-auto mb-3 opacity-30" />
                <p className="text-sm font-medium">Select an email</p>
                <p className="text-xs">Choose an email from the inbox to generate a reply</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

export function EmailQuickCompose({ patientName, patientEmail, onClose }: {
  patientName?: string;
  patientEmail?: string;
  onClose?: () => void;
}) {
  const { toast } = useToast();
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");

  const sendMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/emails/messages", {
        method: "POST",
        json: {
          to: patientEmail,
          subject,
          body,
          isFromDoctor: true,
        },
      });
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Email Sent",
        description: `Message sent to ${patientName || patientEmail}`,
      });
      onClose?.();
    },
  });

  const enhanceMutation = useMutation({
    mutationFn: async (content: string) => {
      const response = await apiRequest("/api/chat/send", {
        method: "POST",
        json: {
          content: `Enhance this email to be more professional and clear: ${content}`,
          agentType: "lysa",
        },
      });
      return response.json();
    },
    onSuccess: (data) => {
      if (data.response) {
        setBody(data.response);
      }
    },
  });

  return (
    <div className="space-y-3">
      <Input
        placeholder="Subject"
        value={subject}
        onChange={(e) => setSubject(e.target.value)}
        data-testid="input-quick-subject"
      />
      <Textarea
        placeholder="Write your message..."
        value={body}
        onChange={(e) => setBody(e.target.value)}
        rows={5}
        data-testid="input-quick-body"
      />
      <div className="flex justify-between">
        <Button
          variant="outline"
          size="sm"
          onClick={() => enhanceMutation.mutate(body)}
          disabled={!body || enhanceMutation.isPending}
          data-testid="button-quick-enhance"
        >
          {enhanceMutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <>
              <Wand2 className="h-4 w-4 mr-1" />
              AI Enhance
            </>
          )}
        </Button>
        <Button
          size="sm"
          onClick={() => sendMutation.mutate()}
          disabled={!subject || !body || sendMutation.isPending}
          data-testid="button-quick-send"
        >
          {sendMutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <>
              <Send className="h-4 w-4 mr-1" />
              Send
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
