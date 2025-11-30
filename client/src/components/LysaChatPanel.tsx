import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { 
  Send, Bot, Calendar, Mail, Phone, FileText, Stethoscope, Pill, 
  Users, MessageSquare, Clock, Sparkles, ChevronDown, ChevronUp,
  X, Plus, Search, AlertTriangle, CheckCircle, Loader2, ExternalLink, Link2
} from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { format } from "date-fns";
import { BookAppointmentDialog } from "./LysaCalendarBooking";

interface IntegrationStatus {
  gmail: { connected: boolean; email?: string; lastSync?: string };
  whatsapp: { connected: boolean; number?: string; lastSync?: string };
  twilio: { connected: boolean; number?: string; lastSync?: string };
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: Date;
  actionType?: string;
  actionData?: any;
}

interface QuickAction {
  id: string;
  label: string;
  icon: React.ReactNode;
  prompt: string;
  category: "appointment" | "communication" | "clinical" | "records";
}

const quickActions: QuickAction[] = [
  {
    id: "whatsapp",
    label: "WhatsApp",
    icon: <Phone className="h-4 w-4" />,
    prompt: "Show me recent WhatsApp messages from patients.",
    category: "communication"
  },
  {
    id: "email",
    label: "Email",
    icon: <Mail className="h-4 w-4" />,
    prompt: "Help me compose and send a professional email to a patient.",
    category: "communication"
  },
  {
    id: "diagnosis-support",
    label: "Diagnosis Support",
    icon: <Stethoscope className="h-4 w-4" />,
    prompt: "I need help analyzing symptoms and considering differential diagnoses.",
    category: "clinical"
  },
  {
    id: "write-prescription",
    label: "Write Prescription",
    icon: <Pill className="h-4 w-4" />,
    prompt: "Help me write a prescription and check for drug interactions.",
    category: "clinical"
  }
];

interface LysaChatPanelProps {
  onMinimize?: () => void;
  isExpanded?: boolean;
  patientId?: string;
  patientName?: string;
  className?: string;
}

export function LysaChatPanel({ onMinimize, isExpanded = true, patientId, patientName, className = "" }: LysaChatPanelProps) {
  const { toast } = useToast();
  const [message, setMessage] = useState("");
  const [showQuickActions, setShowQuickActions] = useState(true);
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [emailDialog, setEmailDialog] = useState(false);
  const [gmailConnectDialog, setGmailConnectDialog] = useState(false);
  const [whatsappConnectDialog, setWhatsappConnectDialog] = useState(false);
  
  // Email compose state for LysaChatPanel
  const [recipientEmail, setRecipientEmail] = useState("");
  const [emailSubject, setEmailSubject] = useState("");
  const [draftEmail, setDraftEmail] = useState("");
  const [selectedTone, setSelectedTone] = useState<string>("professional");
  const scrollRef = useRef<HTMLDivElement>(null);

  const { data: integrationStatus } = useQuery<IntegrationStatus>({
    queryKey: ["/api/v1/integrations/status"],
  });

  const gmailConnected = integrationStatus?.gmail?.connected === true;
  const whatsappConnected = integrationStatus?.whatsapp?.connected === true;

  const { data: chatMessages = [], isLoading } = useQuery<ChatMessage[]>({
    queryKey: ["/api/chat/messages", { agent: "lysa", patientId }],
    queryFn: async () => {
      const url = patientId 
        ? `/api/chat/messages?agent=lysa&patientId=${patientId}`
        : `/api/chat/messages?agent=lysa`;
      const response = await fetch(url, {
        credentials: "include",
      });
      if (!response.ok) {
        throw new Error(`${response.status}: ${response.statusText}`);
      }
      return response.json();
    },
  });

  const sendMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      const payload: { content: string; agentType: string; patientId?: string; patientName?: string } = { 
        content, 
        agentType: "lysa" 
      };
      if (patientId) {
        payload.patientId = patientId;
        payload.patientName = patientName;
      }
      const response = await apiRequest("/api/chat/send", { 
        method: "POST", 
        json: payload
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/chat/messages", { agent: "lysa", patientId }] });
      setMessage("");
    },
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

  const enhanceDraftMutation = useMutation({
    mutationFn: async (content: string) => {
      const response = await apiRequest("/api/chat/send", {
        method: "POST",
        json: {
          content: `Please enhance this email draft to be more ${selectedTone} and clear while maintaining HIPAA compliance. Keep the same meaning but improve the writing:\n\n${content}`,
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
          description: "Your email has been professionally refined by AI.",
        });
      }
    },
    onError: () => {
      toast({
        title: "Enhancement Failed",
        description: "Could not enhance the draft. Please try again.",
        variant: "destructive",
      });
    },
  });

  const sendEmailMutation = useMutation({
    mutationFn: async ({ to, subject, body }: { to: string; subject: string; body: string }) => {
      const response = await apiRequest("/api/v1/emails/messages", {
        method: "POST",
        json: { to, subject, body, isFromDoctor: true },
      });
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Email Sent",
        description: "Your message has been delivered successfully.",
      });
      setEmailDialog(false);
      setDraftEmail("");
      setEmailSubject("");
      setRecipientEmail("");
    },
    onError: () => {
      toast({
        title: "Send Failed",
        description: "Could not send the email. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleSend = () => {
    if (!message.trim() || sendMessageMutation.isPending) return;
    sendMessageMutation.mutate(message);
  };

  const handleQuickAction = (action: QuickAction) => {
    if (action.id === "email") {
      if (!gmailConnected) {
        setGmailConnectDialog(true);
        return;
      }
      setEmailDialog(true);
      return;
    }
    if (action.id === "whatsapp") {
      if (!whatsappConnected) {
        setWhatsappConnectDialog(true);
        return;
      }
    }
    setMessage(action.prompt);
    setShowQuickActions(false);
  };

  const handleCopyEmail = () => {
    navigator.clipboard.writeText(`To: ${recipientEmail}\nSubject: ${emailSubject}\n\n${draftEmail}`);
    toast({
      title: "Copied to Clipboard",
      description: "Email content copied. Paste into your email client to send.",
    });
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const filteredActions = activeCategory === "all" 
    ? quickActions 
    : quickActions.filter(a => a.category === activeCategory);

  const renderMessage = (msg: ChatMessage) => {
    const isUser = msg.role === "user";
    
    return (
      <div 
        key={msg.id}
        className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}
        data-testid={`lysa-message-${msg.id}`}
      >
        <div className={`max-w-[85%] ${isUser ? "order-2" : "order-1"}`}>
          {!isUser && (
            <div className="flex items-center gap-2 mb-1">
              <div className="h-6 w-6 rounded-full bg-accent flex items-center justify-center">
                <Bot className="h-3 w-3" />
              </div>
              <span className="text-xs font-medium">Assistant Lysa</span>
            </div>
          )}
          <div 
            className={`rounded-lg px-3 py-2 text-sm ${
              isUser 
                ? "bg-primary text-primary-foreground" 
                : "bg-muted"
            }`}
          >
            <p className="whitespace-pre-wrap">{msg.content}</p>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {format(new Date(msg.createdAt), "h:mm a")}
          </p>
        </div>
      </div>
    );
  };

  if (!isExpanded) {
    return (
      <Card className={`${className}`}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-full bg-accent flex items-center justify-center">
                <Bot className="h-4 w-4" />
              </div>
              <div>
                <CardTitle className="text-sm">Assistant Lysa</CardTitle>
                <p className="text-xs text-muted-foreground">AI Medical Assistant</p>
              </div>
            </div>
            <Button 
              variant="ghost" 
              size="icon"
              onClick={onMinimize}
              data-testid="button-expand-lysa"
            >
              <ChevronUp className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className={`flex flex-col ${className}`} data-testid="card-lysa-chat">
      <CardHeader className="pb-3 border-b flex-shrink-0">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-accent flex items-center justify-center">
              <Bot className="h-4 w-4" />
            </div>
            <div>
              <CardTitle className="text-sm" data-testid="text-lysa-title">Assistant Lysa</CardTitle>
              {patientName ? (
                <p className="text-xs text-muted-foreground" data-testid="text-lysa-patient-context">
                  Assisting with: <span className="font-medium text-foreground">{patientName}</span>
                </p>
              ) : (
                <p className="text-xs text-muted-foreground">AI Receptionist & Doctor's Assistant</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1">
            {patientId && (
              <Badge variant="secondary" className="text-xs" data-testid="badge-patient-context">
                <Users className="h-3 w-3 mr-1" />
                Patient
              </Badge>
            )}
            <Badge variant="outline" className="text-xs">
              <Sparkles className="h-3 w-3 mr-1" />
              GPT-4
            </Badge>
            {onMinimize && (
              <Button 
                variant="ghost" 
                size="icon"
                onClick={onMinimize}
                data-testid="button-minimize-lysa"
              >
                <ChevronDown className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-0 min-h-0 overflow-hidden">
        {showQuickActions && chatMessages.length === 0 && (
          <div className="p-4 border-b flex-shrink-0">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Quick Actions</h4>
              <Tabs value={activeCategory} onValueChange={setActiveCategory} className="h-8">
                <TabsList className="h-7">
                  <TabsTrigger value="all" className="text-xs px-2 h-6">All</TabsTrigger>
                  <TabsTrigger value="appointment" className="text-xs px-2 h-6">
                    <Calendar className="h-3 w-3" />
                  </TabsTrigger>
                  <TabsTrigger value="communication" className="text-xs px-2 h-6">
                    <Mail className="h-3 w-3" />
                  </TabsTrigger>
                  <TabsTrigger value="clinical" className="text-xs px-2 h-6">
                    <Stethoscope className="h-3 w-3" />
                  </TabsTrigger>
                  <TabsTrigger value="records" className="text-xs px-2 h-6">
                    <FileText className="h-3 w-3" />
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {filteredActions.map((action) => (
                <Button
                  key={action.id}
                  variant="outline"
                  size="sm"
                  className="h-auto py-2 px-3 flex flex-col items-center gap-1 text-xs"
                  onClick={() => handleQuickAction(action)}
                  data-testid={`button-quick-action-${action.id}`}
                >
                  {action.icon}
                  <span className="text-center leading-tight">{action.label}</span>
                </Button>
              ))}
            </div>
          </div>
        )}

        <ScrollArea className="flex-1 min-h-0" ref={scrollRef}>
          <div className="p-4">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : chatMessages.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Bot className="h-12 w-12 mx-auto mb-3 opacity-30" />
                <p className="text-sm font-medium mb-1" data-testid="text-lysa-greeting">
                  {patientName ? `Ready to assist with ${patientName}` : "Hello! I'm Assistant Lysa"}
                </p>
                <p className="text-xs" data-testid="text-lysa-greeting-subtitle">
                  {patientName 
                    ? "Ask me about their health metrics, prescriptions, appointments, or any clinical questions."
                    : "I can help with appointments, emails, patient records, diagnosis support, and prescriptions."
                  }
                </p>
              </div>
            ) : (
              <>
                {chatMessages.map(renderMessage)}
                {sendMessageMutation.isPending && (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-xs">Lysa is thinking...</span>
                  </div>
                )}
              </>
            )}
          </div>
        </ScrollArea>

        <div className="border-t p-3 flex-shrink-0">
          <div className="flex gap-2">
            <Textarea
              placeholder="Ask Lysa anything... appointments, emails, diagnosis help, prescriptions..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              className="resize-none text-sm min-h-[60px]"
              rows={2}
              data-testid="input-lysa-message"
            />
            <div className="flex flex-col gap-1">
              <Button
                size="icon"
                onClick={handleSend}
                disabled={!message.trim() || sendMessageMutation.isPending}
                data-testid="button-lysa-send"
              >
                <Send className="h-4 w-4" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                onClick={() => setShowQuickActions(!showQuickActions)}
                data-testid="button-toggle-quick-actions"
              >
                <Sparkles className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            AI assistant for clinical workflow. Always verify recommendations.
          </p>
        </div>
      </CardContent>
      
      {/* Email Compose Dialog with AI Features */}
      <Dialog open={emailDialog} onOpenChange={setEmailDialog}>
        <DialogContent className="max-w-2xl" data-testid="dialog-email-compose-panel">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5" />
              Compose Email with AI
            </DialogTitle>
            <DialogDescription>
              Write your email and use AI to enhance it professionally
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 pt-2">
            <div className="space-y-2">
              <Label htmlFor="panel-email-recipient">To</Label>
              <Input
                id="panel-email-recipient"
                placeholder="patient@example.com"
                value={recipientEmail}
                onChange={(e) => setRecipientEmail(e.target.value)}
                data-testid="input-panel-email-recipient"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="panel-email-subject">Subject</Label>
              <Input
                id="panel-email-subject"
                placeholder="Re: Your appointment confirmation"
                value={emailSubject}
                onChange={(e) => setEmailSubject(e.target.value)}
                data-testid="input-panel-email-subject"
              />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="panel-email-body">Message</Label>
                <div className="flex gap-2">
                  <Select value={selectedTone} onValueChange={setSelectedTone}>
                    <SelectTrigger className="w-32 h-8" data-testid="select-panel-email-tone">
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
                    data-testid="button-panel-enhance-email"
                  >
                    {enhanceDraftMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4 mr-1" />
                        Enhance
                      </>
                    )}
                  </Button>
                </div>
              </div>
              <Textarea
                id="panel-email-body"
                placeholder="Type your message here... AI will help you refine it."
                value={draftEmail}
                onChange={(e) => setDraftEmail(e.target.value)}
                rows={10}
                className="resize-none"
                data-testid="input-panel-email-body"
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                onClick={() => setEmailDialog(false)}
                data-testid="button-panel-cancel-email"
              >
                Cancel
              </Button>
              <Button
                variant="outline"
                onClick={handleCopyEmail}
                disabled={!recipientEmail || !emailSubject || !draftEmail}
                data-testid="button-panel-copy-email"
              >
                Copy Email
              </Button>
              <Button
                onClick={() => sendEmailMutation.mutate({
                  to: recipientEmail,
                  subject: emailSubject,
                  body: draftEmail,
                })}
                disabled={!recipientEmail || !emailSubject || !draftEmail || sendEmailMutation.isPending}
                data-testid="button-panel-send-email"
              >
                {sendEmailMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Send className="h-4 w-4 mr-2" />
                )}
                Send Email
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
      
      <Dialog open={gmailConnectDialog} onOpenChange={setGmailConnectDialog}>
        <DialogContent data-testid="dialog-gmail-connect">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5 text-red-600" />
              Connect Gmail
            </DialogTitle>
            <DialogDescription>
              Connect your Gmail account to reply to patient emails.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <div className="flex items-center justify-center mb-4">
              <div className="h-16 w-16 rounded-full bg-red-500/10 flex items-center justify-center">
                <Mail className="h-8 w-8 text-red-600" />
              </div>
            </div>
            <p className="text-sm text-muted-foreground text-center mb-4">
              Link your Gmail to automatically sync patient emails, use AI to categorize them, and draft professional replies.
            </p>
          </div>
          <DialogFooter className="flex-col gap-2 sm:flex-row">
            <Button variant="outline" onClick={() => setGmailConnectDialog(false)} data-testid="button-cancel-gmail-connect">
              Cancel
            </Button>
            <Button 
              onClick={() => getGmailAuthUrl.mutate()}
              disabled={getGmailAuthUrl.isPending}
              data-testid="button-confirm-gmail-connect"
            >
              {getGmailAuthUrl.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Link2 className="h-4 w-4 mr-2" />
              )}
              Connect Gmail
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <WhatsAppConnectDialog open={whatsappConnectDialog} onOpenChange={setWhatsappConnectDialog} />
    </Card>
  );
}

function WhatsAppConnectDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { toast } = useToast();
  const [whatsappForm, setWhatsappForm] = useState({
    businessId: "",
    phoneNumberId: "",
    displayNumber: "",
    accessToken: ""
  });

  const configureWhatsApp = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/whatsapp/configure", {
        method: "POST",
        json: whatsappForm,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/status"] });
      onOpenChange(false);
      setWhatsappForm({ businessId: "", phoneNumberId: "", displayNumber: "", accessToken: "" });
      toast({
        title: "Connected",
        description: "WhatsApp Business has been connected successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Connection Failed",
        description: "Failed to connect WhatsApp Business. Please check your credentials.",
        variant: "destructive",
      });
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent data-testid="dialog-whatsapp-connect">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5 text-green-600" />
            Connect WhatsApp Business
          </DialogTitle>
          <DialogDescription>
            Enter your WhatsApp Business API credentials from Meta Business Suite
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 pt-4">
          <div className="space-y-2">
            <Label htmlFor="businessId">Business ID</Label>
            <Input
              id="businessId"
              placeholder="Your WhatsApp Business ID"
              value={whatsappForm.businessId}
              onChange={(e) => setWhatsappForm(prev => ({ ...prev, businessId: e.target.value }))}
              data-testid="input-whatsapp-businessid"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="phoneNumberId">Phone Number ID</Label>
            <Input
              id="phoneNumberId"
              placeholder="Your Phone Number ID"
              value={whatsappForm.phoneNumberId}
              onChange={(e) => setWhatsappForm(prev => ({ ...prev, phoneNumberId: e.target.value }))}
              data-testid="input-whatsapp-phoneid"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="displayNumber">Display Number</Label>
            <Input
              id="displayNumber"
              placeholder="+1234567890"
              value={whatsappForm.displayNumber}
              onChange={(e) => setWhatsappForm(prev => ({ ...prev, displayNumber: e.target.value }))}
              data-testid="input-whatsapp-number"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="accessToken">Access Token</Label>
            <Input
              id="accessToken"
              type="password"
              placeholder="Your permanent access token"
              value={whatsappForm.accessToken}
              onChange={(e) => setWhatsappForm(prev => ({ ...prev, accessToken: e.target.value }))}
              data-testid="input-whatsapp-token"
            />
          </div>
          <div className="flex justify-between items-center pt-2">
            <a 
              href="https://developers.facebook.com/docs/whatsapp/cloud-api/get-started" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sm text-primary hover:underline flex items-center gap-1"
              data-testid="link-whatsapp-guide"
            >
              <ExternalLink className="h-3 w-3" />
              Setup Guide
            </a>
            <div className="flex gap-2">
              <Button 
                variant="outline"
                onClick={() => onOpenChange(false)}
                data-testid="button-cancel-whatsapp"
              >
                Cancel
              </Button>
              <Button 
                onClick={() => configureWhatsApp.mutate()}
                disabled={configureWhatsApp.isPending || !whatsappForm.businessId.trim() || !whatsappForm.phoneNumberId.trim() || !whatsappForm.displayNumber.trim() || !whatsappForm.accessToken.trim()}
                data-testid="button-submit-whatsapp"
              >
                {configureWhatsApp.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Connect
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function LysaQuickActionsBar() {
  const [emailDialog, setEmailDialog] = useState(false);
  const [whatsappConnectDialog, setWhatsappConnectDialog] = useState(false);
  const [gmailConnectDialog, setGmailConnectDialog] = useState(false);
  const { toast } = useToast();
  
  // Email compose state
  const [recipientEmail, setRecipientEmail] = useState("");
  const [emailSubject, setEmailSubject] = useState("");
  const [draftEmail, setDraftEmail] = useState("");
  const [selectedTone, setSelectedTone] = useState<string>("professional");
  
  const { data: integrationStatus } = useQuery<IntegrationStatus>({
    queryKey: ["/api/v1/integrations/status"],
  });

  const gmailConnected = integrationStatus?.gmail?.connected === true;
  const whatsappConnected = integrationStatus?.whatsapp?.connected === true;
  
  const sendMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      const response = await apiRequest("/api/chat/send", { 
        method: "POST", 
        json: { content, agentType: "lysa" } 
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/chat/messages"] });
    },
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
        title: "Connection Error",
        description: "Failed to get Gmail authorization URL. Please try again.",
        variant: "destructive",
      });
    },
  });

  const enhanceDraftMutation = useMutation({
    mutationFn: async (content: string) => {
      const response = await apiRequest("/api/chat/send", {
        method: "POST",
        json: {
          content: `Please enhance this email draft to be more ${selectedTone} and clear while maintaining HIPAA compliance. Keep the same meaning but improve the writing:\n\n${content}`,
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
          description: "Your email has been professionally refined by AI.",
        });
      }
    },
    onError: () => {
      toast({
        title: "Enhancement Failed",
        description: "Could not enhance the draft. Please try again.",
        variant: "destructive",
      });
    },
  });

  const sendEmailMutation = useMutation({
    mutationFn: async ({ to, subject, body }: { to: string; subject: string; body: string }) => {
      const response = await apiRequest("/api/v1/emails/messages", {
        method: "POST",
        json: { to, subject, body, isFromDoctor: true },
      });
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Email Sent",
        description: "Your message has been delivered successfully.",
      });
      setEmailDialog(false);
      setDraftEmail("");
      setEmailSubject("");
      setRecipientEmail("");
    },
    onError: () => {
      toast({
        title: "Send Failed",
        description: "Could not send the email. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleQuickAction = (action: QuickAction) => {
    if (action.id === "whatsapp") {
      if (!whatsappConnected) {
        setWhatsappConnectDialog(true);
        return;
      }
      sendMessageMutation.mutate(action.prompt);
      return;
    }
    if (action.id === "email") {
      if (!gmailConnected) {
        setGmailConnectDialog(true);
        return;
      }
      setEmailDialog(true);
      return;
    }
    sendMessageMutation.mutate(action.prompt);
  };

  const handleCopyEmail = () => {
    navigator.clipboard.writeText(`To: ${recipientEmail}\nSubject: ${emailSubject}\n\n${draftEmail}`);
    toast({
      title: "Copied to Clipboard",
      description: "Email content copied. Paste into your email client to send.",
    });
  };

  return (
    <>
      <div className="flex items-center gap-2 p-2 bg-muted/50 rounded-lg">
        <Bot className="h-5 w-5 text-muted-foreground flex-shrink-0" />
        <div className="flex gap-1 overflow-x-auto">
          {quickActions.slice(0, 4).map((action) => {
            const isWhatsApp = action.id === "whatsapp";
            const isEmail = action.id === "email";
            const notConnected = (isWhatsApp && !whatsappConnected) || (isEmail && !gmailConnected);
            
            return (
              <Button
                key={action.id}
                variant="ghost"
                size="sm"
                className={`h-7 px-2 text-xs whitespace-nowrap flex-shrink-0 ${notConnected ? 'opacity-70' : ''}`}
                onClick={() => handleQuickAction(action)}
                disabled={sendMessageMutation.isPending}
                data-testid={`button-quick-${action.id}`}
              >
                {notConnected && <Link2 className="h-3 w-3 mr-1 text-muted-foreground" />}
                {action.icon}
                <span className="ml-1">{action.label}</span>
              </Button>
            );
          })}
        </div>
        {sendMessageMutation.isPending && (
          <Loader2 className="h-4 w-4 animate-spin ml-2" />
        )}
      </div>
      
      {/* Email Compose Dialog with AI Features */}
      <Dialog open={emailDialog} onOpenChange={setEmailDialog}>
        <DialogContent className="max-w-2xl" data-testid="dialog-email-compose">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5" />
              Compose Email with AI
            </DialogTitle>
            <DialogDescription>
              Write your email and use AI to enhance it professionally
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 pt-2">
            <div className="space-y-2">
              <Label htmlFor="email-recipient">To</Label>
              <Input
                id="email-recipient"
                placeholder="patient@example.com"
                value={recipientEmail}
                onChange={(e) => setRecipientEmail(e.target.value)}
                data-testid="input-email-recipient"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email-subject">Subject</Label>
              <Input
                id="email-subject"
                placeholder="Re: Your appointment confirmation"
                value={emailSubject}
                onChange={(e) => setEmailSubject(e.target.value)}
                data-testid="input-email-subject"
              />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="email-body">Message</Label>
                <div className="flex gap-2">
                  <Select value={selectedTone} onValueChange={setSelectedTone}>
                    <SelectTrigger className="w-32 h-8" data-testid="select-email-tone">
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
                        <Sparkles className="h-4 w-4 mr-1" />
                        Enhance
                      </>
                    )}
                  </Button>
                </div>
              </div>
              <Textarea
                id="email-body"
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
                onClick={() => setEmailDialog(false)}
                data-testid="button-cancel-email"
              >
                Cancel
              </Button>
              <Button
                variant="outline"
                onClick={handleCopyEmail}
                disabled={!recipientEmail || !emailSubject || !draftEmail}
                data-testid="button-copy-email"
              >
                Copy Email
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
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Send className="h-4 w-4 mr-2" />
                )}
                Send Email
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
      
      {/* Gmail Connect Dialog */}
      <Dialog open={gmailConnectDialog} onOpenChange={setGmailConnectDialog}>
        <DialogContent data-testid="dialog-gmail-connect">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5" />
              Connect Gmail
            </DialogTitle>
            <DialogDescription>
              Connect your Gmail account to enable email features
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                <CheckCircle className="h-5 w-5 text-green-500" />
                <span className="text-sm">Compose and send emails directly</span>
              </div>
              <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                <CheckCircle className="h-5 w-5 text-green-500" />
                <span className="text-sm">AI-powered draft enhancement</span>
              </div>
              <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                <CheckCircle className="h-5 w-5 text-green-500" />
                <span className="text-sm">HIPAA-compliant email management</span>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setGmailConnectDialog(false)} data-testid="button-cancel-gmail">
              Cancel
            </Button>
            <Button 
              onClick={() => getGmailAuthUrl.mutate()}
              disabled={getGmailAuthUrl.isPending}
              data-testid="button-connect-gmail"
            >
              {getGmailAuthUrl.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Connect Gmail
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      <WhatsAppConnectDialog open={whatsappConnectDialog} onOpenChange={setWhatsappConnectDialog} />
    </>
  );
}
