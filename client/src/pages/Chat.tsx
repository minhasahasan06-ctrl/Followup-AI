import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { ChatMessage } from "@/components/ChatMessage";
import { LegalDisclaimer } from "@/components/LegalDisclaimer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Send, Bot, AlertTriangle, Shield, MapPin, X, MessageSquare, Clock, Heart, PanelLeftClose, PanelLeft, Plus, Calendar, ChevronRight } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAuth } from "@/hooks/useAuth";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { ChatMessage as ChatMessageType } from "@shared/schema";
import { format } from "date-fns";

interface RiskAlert {
  id: number;
  alertType: string;
  severity: string;
  message: string;
  status: string;
}

interface ChatSession {
  id: string;
  patientId: string;
  agentType: string;
  sessionTitle: string | null;
  startedAt: Date;
  endedAt: Date | null;
  messageCount: number | null;
  symptomsDiscussed: string[] | null;
  recommendations: string[] | null;
  healthInsights: any;
  aiSummary: string | null;
  createdAt: Date;
  updatedAt: Date;
  doctorNotes: string | null;
}

export default function Chat() {
  const [message, setMessage] = useState("");
  const [dismissedAlerts, setDismissedAlerts] = useState<number[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const { user } = useAuth();
  const isDoctor = user?.role === "doctor";
  const agentType = isDoctor ? "lysa" : "clona";
  const agentName = isDoctor ? "Assistant Lysa" : "Agent Clona";

  // Fetch all sessions
  const { data: sessions, isLoading: sessionsLoading } = useQuery<ChatSession[]>({
    queryKey: ['/api/chat/sessions'],
  });

  // Fetch current chat messages (either selected session or current active session)
  const { data: chatMessages, isLoading } = useQuery<ChatMessageType[]>({
    queryKey: selectedSessionId 
      ? [`/api/chat/sessions/${selectedSessionId}/messages`]
      : ["/api/chat/messages", { agent: agentType }],
  });

  const { data: riskAlerts } = useQuery<RiskAlert[]>({
    queryKey: ["/api/risk/alerts"],
    enabled: !isDoctor,
  });

  const sendMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      return await apiRequest("POST", "/api/chat/send", { content, agentType });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/chat/messages"] });
      queryClient.invalidateQueries({ queryKey: ['/api/chat/sessions'] });
      setMessage("");
    },
  });

  const dismissAlertMutation = useMutation({
    mutationFn: async (alertId: number) => {
      return await apiRequest("POST", `/api/risk/alerts/${alertId}/dismiss`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/risk/alerts"] });
    },
  });

  const handleSend = () => {
    if (!message.trim() || sendMessageMutation.isPending) return;
    sendMessageMutation.mutate(message);
  };

  const handleDismissAlert = (alertId: number, temporary: boolean = false) => {
    if (temporary) {
      setDismissedAlerts([...dismissedAlerts, alertId]);
    } else {
      dismissAlertMutation.mutate(alertId);
    }
  };

  const handleNewChat = () => {
    setSelectedSessionId(null);
    queryClient.invalidateQueries({ queryKey: ["/api/chat/messages"] });
  };

  const handleSessionClick = (sessionId: string) => {
    setSelectedSessionId(sessionId);
  };

  const activeAlerts = riskAlerts?.filter(
    (alert) => alert.status === "active" && !dismissedAlerts.includes(alert.id)
  ) || [];

  const getAlertIcon = (type: string) => {
    if (type.includes("immune")) return Shield;
    if (type.includes("environmental")) return MapPin;
    return AlertTriangle;
  };

  const getAlertLink = (type: string) => {
    if (type.includes("immune")) return "/immune-monitoring";
    if (type.includes("environmental")) return "/environmental-risk";
    return "#";
  };

  const getAgentIcon = (agentType: string) => {
    return agentType === 'clona' ? <Heart className="h-4 w-4" /> : <Bot className="h-4 w-4" />;
  };

  const formatDate = (date: Date | string) => {
    const d = new Date(date);
    return format(d, 'MMM d, h:mm a');
  };

  const selectedSession = sessions?.find(s => s.id === selectedSessionId);

  return (
    <div className="h-full flex gap-6">
      {/* Sessions Sidebar */}
      {sidebarOpen && (
        <div className="w-80 flex-shrink-0 flex flex-col gap-4">
          {/* Sidebar Header */}
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <MessageSquare className="h-5 w-5" />
              Chat History
            </h2>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(false)}
              data-testid="button-close-sidebar"
            >
              <PanelLeftClose className="h-4 w-4" />
            </Button>
          </div>

          {/* New Chat Button */}
          <Button
            onClick={handleNewChat}
            className="w-full gap-2"
            variant={selectedSessionId ? "outline" : "default"}
            data-testid="button-new-chat"
          >
            <Plus className="h-4 w-4" />
            New Chat
          </Button>

          {/* Sessions List */}
          <Card className="flex-1 flex flex-col min-h-0">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Calendar className="h-4 w-4" />
                Previous Sessions
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 flex-1 min-h-0">
              <ScrollArea className="h-full">
                <div className="space-y-2 p-4">
                  {sessionsLoading ? (
                    <div className="space-y-2">
                      {[1, 2, 3].map(i => (
                        <div key={i} className="h-20 bg-muted rounded animate-pulse" />
                      ))}
                    </div>
                  ) : sessions && sessions.length > 0 ? (
                    sessions.map((session) => (
                      <button
                        key={session.id}
                        onClick={() => handleSessionClick(session.id)}
                        className={`w-full text-left p-3 rounded-md border transition-colors hover-elevate ${
                          selectedSessionId === session.id
                            ? 'bg-accent border-accent-foreground/20'
                            : 'bg-card border-border'
                        }`}
                        data-testid={`button-session-${session.id}`}
                      >
                        <div className="flex items-start justify-between gap-2 mb-1">
                          <div className="flex items-center gap-2">
                            {getAgentIcon(session.agentType)}
                            <span className="text-xs font-medium">
                              {session.agentType === 'clona' ? 'Agent Clona' : 'Assistant Lysa'}
                            </span>
                          </div>
                          <ChevronRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                        </div>
                        <p className="text-xs text-muted-foreground mb-1 line-clamp-2">
                          {session.sessionTitle || 'No title'}
                        </p>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          {formatDate(session.startedAt)}
                        </div>
                        {session.messageCount && (
                          <Badge variant="secondary" className="text-xs mt-1">
                            {session.messageCount} msgs
                          </Badge>
                        )}
                      </button>
                    ))
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <MessageSquare className="h-12 w-12 mx-auto mb-2 opacity-30" />
                      <p className="text-xs">No previous sessions</p>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header */}
        <div className="mb-4">
          <div className="flex items-center gap-3 mb-2">
            {!sidebarOpen && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSidebarOpen(true)}
                data-testid="button-open-sidebar"
              >
                <PanelLeft className="h-4 w-4" />
              </Button>
            )}
            <div className={`flex h-10 w-10 items-center justify-center rounded-full ${
              isDoctor ? "bg-accent" : "bg-primary"
            } text-primary-foreground`} data-testid="avatar-ai-agent">
              {isDoctor ? <Bot className="h-5 w-5" /> : <Heart className="h-5 w-5" />}
            </div>
            <div>
              <h1 className="text-xl font-semibold" data-testid="text-agent-name">{agentName}</h1>
              <p className="text-xs text-muted-foreground" data-testid="text-agent-description">
                Powered by GPT-4 {isDoctor ? "- Clinical Insights" : "- Your Health Companion"}
              </p>
            </div>
          </div>
          {!isDoctor && <LegalDisclaimer variant="wellness" className="mt-3" />}
        </div>

      {!isDoctor && activeAlerts.length > 0 && (
        <div className="mb-4 space-y-2" data-testid="section-active-alerts">
          {activeAlerts.map((alert) => {
            const AlertIcon = getAlertIcon(alert.alertType);
            const alertLink = getAlertLink(alert.alertType);
            
            return (
              <Card 
                key={alert.id}
                className="border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/20"
                data-testid={`alert-banner-${alert.id}`}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <AlertIcon className="h-5 w-5 text-red-600 mt-0.5 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant="destructive" data-testid={`badge-alert-severity-${alert.severity}`}>
                          {alert.severity}
                        </Badge>
                        <Badge variant="outline" data-testid={`badge-alert-type-${alert.alertType}`}>
                          {alert.alertType}
                        </Badge>
                      </div>
                      <p className="text-sm font-medium text-red-900 dark:text-red-100 mb-2" data-testid={`text-alert-message-${alert.id}`}>
                        {alert.message}
                      </p>
                      <div className="flex gap-2">
                        <Button 
                          size="sm" 
                          variant="outline"
                          onClick={() => window.location.href = alertLink}
                          data-testid={`button-view-details-${alert.id}`}
                        >
                          View Details
                        </Button>
                        <Button 
                          size="sm" 
                          variant="ghost"
                          onClick={() => handleDismissAlert(alert.id, true)}
                          data-testid={`button-dismiss-temp-${alert.id}`}
                        >
                          <X className="h-4 w-4 mr-1" />
                          Hide
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      <Card className="flex-1 flex flex-col min-h-0" data-testid="card-chat">
        <CardHeader className="border-b">
          <CardTitle className="text-base" data-testid="text-chat-title">Chat with {agentName}</CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-0 min-h-0">
          <ScrollArea className="flex-1 p-4" data-testid="scroll-messages">
            {isLoading ? (
              <div className="flex items-center justify-center py-8" data-testid="loading-messages">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
              </div>
            ) : (
              <div className="space-y-4">
                {chatMessages && chatMessages.length > 0 ? (
                  chatMessages.map((msg) => (
                    <ChatMessage
                      key={msg.id}
                      role={msg.role as "user" | "assistant"}
                      content={msg.content}
                      timestamp={msg.createdAt ? new Date(msg.createdAt).toLocaleTimeString() : ""}
                      isGP={msg.role === "assistant"}
                      entities={msg.medicalEntities || undefined}
                      testId={`message-${msg.id}`}
                    />
                  ))
                ) : (
                  <div className="text-center py-8 text-muted-foreground" data-testid="text-no-messages">
                    <p>Start a conversation with {agentName}</p>
                  </div>
                )}
                {sendMessageMutation.isPending && (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground" data-testid="indicator-typing">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    <span data-testid="text-typing">{agentName} is typing...</span>
                  </div>
                )}
              </div>
            )}
          </ScrollArea>
          
          <div className="border-t p-4" data-testid="section-message-input">
            <div className="flex gap-2">
              <Textarea
                placeholder="Describe your symptoms or ask a question..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                className="resize-none"
                rows={3}
                data-testid="input-chat-message"
              />
              <Button
                size="icon"
                onClick={handleSend}
                disabled={!message.trim()}
                data-testid="button-send-message"
                className="h-auto"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-2" data-testid="text-disclaimer">
              {isDoctor 
                ? "AI-powered clinical insights. Always use professional judgment."
                : "This AI assistant provides general health information. Always consult your doctor for medical decisions."}
            </p>
          </div>
        </CardContent>
      </Card>
      </div>
    </div>
  );
}
