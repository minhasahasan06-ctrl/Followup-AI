import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { 
  Send, 
  Bot, 
  User, 
  Headphones, 
  MessageSquare, 
  Clock, 
  CheckCheck, 
  Check,
  Circle,
  Loader2,
  Phone,
  Video,
  MoreVertical,
  Search,
  Plus,
  Settings,
  Shield,
  AlertTriangle,
  Stethoscope,
  Heart,
  Wrench,
  ThumbsUp,
  ThumbsDown,
  AlertCircle,
  CheckCircle,
  XCircle,
  Activity
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { format, formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";

interface AgentMessage {
  id: string;
  msgId: string;
  fromType: "agent" | "user" | "system";
  fromId: string;
  toType: "agent" | "user";
  toId: string;
  messageType: "chat" | "command" | "event" | "tool_call" | "ack";
  content: string;
  payload?: Record<string, unknown>;
  delivered: boolean;
  deliveredAt?: string;
  readAt?: string;
  createdAt: string;
}

interface Conversation {
  id: string;
  participantType: "agent" | "user";
  participantId: string;
  participantName: string;
  lastMessage?: string;
  lastMessageAt?: string;
  unreadCount: number;
  isOnline: boolean;
}

interface PresenceStatus {
  isOnline: boolean;
  lastSeen?: string;
  isTyping?: boolean;
}

interface ToolCallApproval {
  id: string;
  toolName: string;
  reason: string;
  parameters?: Record<string, unknown>;
  timestamp: string;
  status: "pending" | "approved" | "rejected";
}

interface ToolCallStatus {
  id: string;
  toolName: string;
  status: "pending" | "running" | "completed" | "failed";
  result?: string;
  error?: string;
}

export default function AgentHub() {
  const { user } = useAuth();
  const isDoctor = user?.role === "doctor";
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [agentPresence, setAgentPresence] = useState<PresenceStatus>({ isOnline: true });
  const [pendingApprovals, setPendingApprovals] = useState<ToolCallApproval[]>([]);
  const [toolCallStatuses, setToolCallStatuses] = useState<ToolCallStatus[]>([]);
  const [approvalDialogOpen, setApprovalDialogOpen] = useState(false);
  const [selectedApproval, setSelectedApproval] = useState<ToolCallApproval | null>(null);

  const agentName = isDoctor ? "Assistant Lysa" : "Agent Clona";
  const agentDescription = isDoctor 
    ? "Your AI-powered clinical assistant for patient management, scheduling, and medical insights"
    : "Your personal AI health companion for daily check-ins, symptom tracking, and wellness support";
  const agentIcon = isDoctor ? Headphones : Bot;
  const AgentIcon = agentIcon;

  // Fetch conversations list
  const { data: conversations = [], isLoading: conversationsLoading } = useQuery<Conversation[]>({
    queryKey: ["/api/agent/conversations"],
    refetchInterval: 30000,
  });

  // Fetch messages for selected conversation
  const { data: messages = [], isLoading: messagesLoading, refetch: refetchMessages } = useQuery<AgentMessage[]>({
    queryKey: ["/api/agent/messages", selectedConversation],
    enabled: !!selectedConversation,
  });

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      const messageEnvelope = {
        msgId: crypto.randomUUID(),
        from: { type: "user", id: user?.id },
        to: [{ type: "agent", id: isDoctor ? "lysa" : "clona" }],
        type: "chat",
        timestamp: new Date().toISOString(),
        payload: { content },
      };
      return apiRequest("/api/agent/messages", {
        method: "POST",
        body: JSON.stringify(messageEnvelope),
      });
    },
    onSuccess: () => {
      setMessage("");
      queryClient.invalidateQueries({ queryKey: ["/api/agent/messages", selectedConversation] });
      queryClient.invalidateQueries({ queryKey: ["/api/agent/conversations"] });
    },
  });

  // WebSocket connection for real-time updates
  // Connect to Python FastAPI backend for agent communication
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    // Use the same host for WebSocket - Vite proxy will forward /ws/agent to Python backend
    const wsUrl = `${protocol}//${window.location.host}/ws/agent`;
    
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log("Agent WebSocket connected");
          setWsConnected(true);
          reconnectAttempts = 0;
          
          // Send authentication
          ws.send(JSON.stringify({
            type: "auth",
            payload: { 
              userId: user?.id, 
              role: user?.role,
              token: document.cookie.includes('session') ? 'session' : undefined
            }
          }));
        };
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
          } catch (e) {
            console.error("Failed to parse WebSocket message:", e);
          }
        };
        
        ws.onclose = (event) => {
          console.log("Agent WebSocket disconnected:", event.code, event.reason);
          setWsConnected(false);
          
          // Reconnect with exponential backoff
          if (reconnectAttempts < maxReconnectAttempts) {
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            reconnectAttempts++;
            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
            reconnectTimeout = setTimeout(connectWebSocket, delay);
          } else {
            console.log("Max reconnection attempts reached");
          }
        };
        
        ws.onerror = (error) => {
          console.error("Agent WebSocket error:", error);
        };
        
        wsRef.current = ws;
      } catch (error) {
        console.error("Failed to connect WebSocket:", error);
        // Retry on connection failure
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          reconnectTimeout = setTimeout(connectWebSocket, 3000);
        }
      }
    };
    
    if (user?.id) {
      connectWebSocket();
    }
    
    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [user?.id, user?.role]);

  const handleWebSocketMessage = useCallback((data: any) => {
    switch (data.type) {
      case "message":
        refetchMessages();
        break;
      case "presence":
        setAgentPresence(data.payload);
        break;
      case "typing":
        setIsTyping(data.payload.isTyping);
        break;
      case "ack":
        // Handle delivery acknowledgment
        queryClient.invalidateQueries({ queryKey: ["/api/agent/messages"] });
        break;
      case "approval_required":
        // Add new approval request
        const approval: ToolCallApproval = {
          id: data.payload.messageId || crypto.randomUUID(),
          toolName: data.payload.toolName,
          reason: data.payload.reason,
          parameters: data.payload.parameters,
          timestamp: new Date().toISOString(),
          status: "pending"
        };
        setPendingApprovals(prev => [...prev, approval]);
        setSelectedApproval(approval);
        setApprovalDialogOpen(true);
        break;
      case "tool_call":
        // Update tool call status
        const toolStatus: ToolCallStatus = {
          id: data.payload.toolCallId || crypto.randomUUID(),
          toolName: data.payload.toolName,
          status: data.payload.status,
          result: data.payload.result,
          error: data.payload.error
        };
        setToolCallStatuses(prev => {
          const existing = prev.find(t => t.id === toolStatus.id);
          if (existing) {
            return prev.map(t => t.id === toolStatus.id ? toolStatus : t);
          }
          return [...prev, toolStatus];
        });
        // Remove from pending if completed or failed
        if (toolStatus.status === "completed" || toolStatus.status === "failed") {
          setTimeout(() => {
            setToolCallStatuses(prev => prev.filter(t => t.id !== toolStatus.id));
          }, 5000);
        }
        break;
    }
  }, [refetchMessages]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-select first conversation or create default agent conversation
  useEffect(() => {
    if (!selectedConversation && conversations.length > 0) {
      setSelectedConversation(conversations[0].id);
    } else if (!selectedConversation && conversations.length === 0) {
      // Create default agent conversation
      setSelectedConversation(isDoctor ? "lysa" : "clona");
    }
  }, [conversations, selectedConversation, isDoctor]);

  const handleSendMessage = () => {
    if (message.trim() && !sendMessageMutation.isPending) {
      sendMessageMutation.mutate(message.trim());
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getDeliveryIcon = (msg: AgentMessage) => {
    if (msg.readAt) {
      return <CheckCheck className="h-3 w-3 text-primary" />;
    } else if (msg.deliveredAt || msg.delivered) {
      return <CheckCheck className="h-3 w-3 text-muted-foreground" />;
    }
    return <Check className="h-3 w-3 text-muted-foreground" />;
  };

  // Tool approval mutation
  const approvalMutation = useMutation({
    mutationFn: async ({ approvalId, approved }: { approvalId: string; approved: boolean }) => {
      return apiRequest("/api/agent/tool-approval", {
        method: "POST",
        body: JSON.stringify({
          approvalId,
          approved,
          userId: user?.id
        }),
      });
    },
    onSuccess: (_, variables) => {
      setPendingApprovals(prev => 
        prev.map(a => a.id === variables.approvalId 
          ? { ...a, status: variables.approved ? "approved" : "rejected" } 
          : a
        )
      );
      setApprovalDialogOpen(false);
      setSelectedApproval(null);
      // Remove from pending after a short delay
      setTimeout(() => {
        setPendingApprovals(prev => prev.filter(a => a.id !== variables.approvalId));
      }, 3000);
    },
  });

  const handleApproval = (approved: boolean) => {
    if (selectedApproval) {
      approvalMutation.mutate({ approvalId: selectedApproval.id, approved });
    }
  };

  const getToolIcon = (toolName: string) => {
    const toolIcons: Record<string, typeof Wrench> = {
      calendar: Clock,
      prescription: Stethoscope,
      message: MessageSquare,
      alert: AlertCircle,
      lab: Activity,
    };
    const Icon = toolIcons[toolName.toLowerCase()] || Wrench;
    return <Icon className="h-4 w-4" />;
  };

  const getToolStatusColor = (status: ToolCallStatus["status"]) => {
    switch (status) {
      case "pending": return "text-yellow-500";
      case "running": return "text-blue-500";
      case "completed": return "text-green-500";
      case "failed": return "text-red-500";
      default: return "text-muted-foreground";
    }
  };

  const getToolStatusIcon = (status: ToolCallStatus["status"]) => {
    switch (status) {
      case "pending": return <Clock className="h-4 w-4 animate-pulse" />;
      case "running": return <Loader2 className="h-4 w-4 animate-spin" />;
      case "completed": return <CheckCircle className="h-4 w-4" />;
      case "failed": return <XCircle className="h-4 w-4" />;
      default: return <Circle className="h-4 w-4" />;
    }
  };

  return (
    <>
    {/* Tool Approval Dialog */}
    <AlertDialog open={approvalDialogOpen} onOpenChange={setApprovalDialogOpen}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            Action Approval Required
          </AlertDialogTitle>
          <AlertDialogDescription className="space-y-3">
            <p>{agentName} is requesting permission to perform the following action:</p>
            {selectedApproval && (
              <div className="bg-muted rounded-lg p-4 space-y-2">
                <div className="flex items-center gap-2">
                  {getToolIcon(selectedApproval.toolName)}
                  <span className="font-medium">{selectedApproval.toolName}</span>
                </div>
                <p className="text-sm">{selectedApproval.reason}</p>
                {selectedApproval.parameters && (
                  <details className="text-xs">
                    <summary className="cursor-pointer text-muted-foreground">View details</summary>
                    <pre className="mt-2 p-2 bg-background rounded text-xs overflow-auto max-h-32">
                      {JSON.stringify(selectedApproval.parameters, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}
            <p className="text-sm text-muted-foreground">
              This action requires your explicit approval before proceeding.
            </p>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel 
            onClick={() => handleApproval(false)}
            disabled={approvalMutation.isPending}
            data-testid="button-reject-approval"
          >
            <ThumbsDown className="h-4 w-4 mr-2" />
            Reject
          </AlertDialogCancel>
          <AlertDialogAction 
            onClick={() => handleApproval(true)}
            disabled={approvalMutation.isPending}
            data-testid="button-approve-action"
          >
            {approvalMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <ThumbsUp className="h-4 w-4 mr-2" />
            )}
            Approve
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
    
    <div className="flex h-[calc(100vh-8rem)] gap-4" data-testid="container-agent-hub">
      {/* Conversations Sidebar */}
      <Card className="w-80 flex flex-col">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg" data-testid="heading-conversations">Messages</CardTitle>
            <Button variant="ghost" size="icon" data-testid="button-new-conversation">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search conversations..."
              className="w-full pl-8 pr-4 py-2 text-sm border rounded-md bg-background"
              data-testid="input-search-conversations"
            />
          </div>
        </CardHeader>
        <ScrollArea className="flex-1">
          <div className="px-2 pb-2 space-y-1">
            {/* Primary Agent Conversation */}
            <div
              className={cn(
                "flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors",
                selectedConversation === (isDoctor ? "lysa" : "clona")
                  ? "bg-primary/10"
                  : "hover-elevate"
              )}
              onClick={() => setSelectedConversation(isDoctor ? "lysa" : "clona")}
              data-testid="conversation-primary-agent"
            >
              <Avatar className="h-10 w-10">
                <AvatarFallback className="bg-primary text-primary-foreground">
                  <AgentIcon className="h-5 w-5" />
                </AvatarFallback>
              </Avatar>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">{agentName}</span>
                  <div className="flex items-center gap-1">
                    <Circle className={cn(
                      "h-2 w-2 fill-current",
                      agentPresence.isOnline ? "text-green-500" : "text-muted-foreground"
                    )} />
                  </div>
                </div>
                <p className="text-xs text-muted-foreground truncate">
                  {agentPresence.isOnline ? "Online" : "Available"}
                </p>
              </div>
              <Badge variant="default" className="text-xs">AI</Badge>
            </div>

            <Separator className="my-2" />

            {/* Other Conversations */}
            {conversationsLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : conversations.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                {isDoctor ? "Patient messages will appear here" : "Doctor messages will appear here"}
              </p>
            ) : (
              conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={cn(
                    "flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors",
                    selectedConversation === conv.id
                      ? "bg-primary/10"
                      : "hover-elevate"
                  )}
                  onClick={() => setSelectedConversation(conv.id)}
                  data-testid={`conversation-${conv.id}`}
                >
                  <Avatar className="h-10 w-10">
                    <AvatarFallback>
                      {conv.participantType === "agent" ? (
                        <Bot className="h-5 w-5" />
                      ) : (
                        <User className="h-5 w-5" />
                      )}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-sm">{conv.participantName}</span>
                      {conv.lastMessageAt && (
                        <span className="text-xs text-muted-foreground">
                          {formatDistanceToNow(new Date(conv.lastMessageAt), { addSuffix: true })}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground truncate">
                      {conv.lastMessage || "No messages yet"}
                    </p>
                  </div>
                  {conv.unreadCount > 0 && (
                    <Badge variant="default" className="text-xs">
                      {conv.unreadCount}
                    </Badge>
                  )}
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </Card>

      {/* Main Chat Area */}
      <Card className="flex-1 flex flex-col">
        {/* Chat Header */}
        <CardHeader className="pb-3 border-b">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Avatar className="h-10 w-10">
                <AvatarFallback className="bg-primary text-primary-foreground">
                  <AgentIcon className="h-5 w-5" />
                </AvatarFallback>
              </Avatar>
              <div>
                <CardTitle className="text-lg" data-testid="heading-agent-name">{agentName}</CardTitle>
                <CardDescription className="flex items-center gap-2">
                  <Circle className={cn(
                    "h-2 w-2 fill-current",
                    agentPresence.isOnline ? "text-green-500" : "text-muted-foreground"
                  )} />
                  <span>{agentPresence.isOnline ? "Online" : "Available"}</span>
                  {isTyping && <span className="text-primary">typing...</span>}
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="gap-1">
                <Shield className="h-3 w-3" />
                HIPAA Compliant
              </Badge>
              <Button variant="ghost" size="icon" data-testid="button-voice-call">
                <Phone className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="icon" data-testid="button-video-call">
                <Video className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="icon" data-testid="button-chat-settings">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        {/* Messages Area */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {/* Welcome Message */}
            {messages.length === 0 && !messagesLoading && (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                  <AgentIcon className="h-8 w-8 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2" data-testid="text-welcome-title">
                  Welcome to {agentName}
                </h3>
                <p className="text-muted-foreground max-w-md mb-6">
                  {agentDescription}
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {isDoctor ? (
                    <>
                      <Button variant="outline" size="sm" className="gap-2" data-testid="button-quick-action-schedule">
                        <Clock className="h-4 w-4" />
                        View Today's Schedule
                      </Button>
                      <Button variant="outline" size="sm" className="gap-2" data-testid="button-quick-action-patients">
                        <Stethoscope className="h-4 w-4" />
                        Patient Summaries
                      </Button>
                      <Button variant="outline" size="sm" className="gap-2" data-testid="button-quick-action-alerts">
                        <AlertTriangle className="h-4 w-4" />
                        Health Alerts
                      </Button>
                    </>
                  ) : (
                    <>
                      <Button variant="outline" size="sm" className="gap-2" data-testid="button-quick-action-checkin">
                        <Heart className="h-4 w-4" />
                        Daily Check-in
                      </Button>
                      <Button variant="outline" size="sm" className="gap-2" data-testid="button-quick-action-symptoms">
                        <MessageSquare className="h-4 w-4" />
                        Log Symptoms
                      </Button>
                      <Button variant="outline" size="sm" className="gap-2" data-testid="button-quick-action-medications">
                        <Clock className="h-4 w-4" />
                        Medication Reminder
                      </Button>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Loading State */}
            {messagesLoading && (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            )}

            {/* Messages */}
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={cn(
                  "flex gap-3",
                  msg.fromType === "user" ? "flex-row-reverse" : "flex-row"
                )}
                data-testid={`message-${msg.id}`}
              >
                <Avatar className="h-8 w-8 flex-shrink-0">
                  <AvatarFallback className={cn(
                    msg.fromType === "user" 
                      ? "bg-secondary text-secondary-foreground"
                      : "bg-primary text-primary-foreground"
                  )}>
                    {msg.fromType === "user" ? (
                      <User className="h-4 w-4" />
                    ) : (
                      <AgentIcon className="h-4 w-4" />
                    )}
                  </AvatarFallback>
                </Avatar>
                <div className={cn(
                  "flex flex-col max-w-[70%]",
                  msg.fromType === "user" ? "items-end" : "items-start"
                )}>
                  <div className={cn(
                    "rounded-lg px-4 py-2",
                    msg.fromType === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  )}>
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  </div>
                  <div className="flex items-center gap-1 mt-1">
                    <span className="text-xs text-muted-foreground">
                      {format(new Date(msg.createdAt), "HH:mm")}
                    </span>
                    {msg.fromType === "user" && getDeliveryIcon(msg)}
                  </div>
                </div>
              </div>
            ))}

            {/* Tool Call Status Indicators */}
            {toolCallStatuses.length > 0 && (
              <div className="space-y-2" data-testid="container-tool-statuses">
                {toolCallStatuses.map((tool) => (
                  <div 
                    key={tool.id} 
                    className="flex gap-3"
                    data-testid={`tool-status-${tool.id}`}
                  >
                    <Avatar className="h-8 w-8">
                      <AvatarFallback className="bg-primary text-primary-foreground">
                        {getToolIcon(tool.toolName)}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <div className={cn(
                        "rounded-lg px-4 py-2 border",
                        tool.status === "completed" && "border-green-500/30 bg-green-500/10",
                        tool.status === "failed" && "border-red-500/30 bg-red-500/10",
                        tool.status === "running" && "border-blue-500/30 bg-blue-500/10",
                        tool.status === "pending" && "border-yellow-500/30 bg-yellow-500/10"
                      )}>
                        <div className="flex items-center gap-2">
                          <span className={getToolStatusColor(tool.status)}>
                            {getToolStatusIcon(tool.status)}
                          </span>
                          <span className="text-sm font-medium">{tool.toolName}</span>
                          <Badge variant="outline" className="text-xs">
                            {tool.status}
                          </Badge>
                        </div>
                        {tool.result && (
                          <p className="text-xs text-muted-foreground mt-1">{tool.result}</p>
                        )}
                        {tool.error && (
                          <p className="text-xs text-red-500 mt-1">{tool.error}</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Pending Approval Indicators */}
            {pendingApprovals.filter(a => a.status === "pending").length > 0 && (
              <div className="space-y-2" data-testid="container-pending-approvals">
                {pendingApprovals.filter(a => a.status === "pending").map((approval) => (
                  <div 
                    key={approval.id}
                    className="flex gap-3 cursor-pointer"
                    onClick={() => {
                      setSelectedApproval(approval);
                      setApprovalDialogOpen(true);
                    }}
                    data-testid={`pending-approval-${approval.id}`}
                  >
                    <Avatar className="h-8 w-8">
                      <AvatarFallback className="bg-yellow-500 text-yellow-900">
                        <AlertTriangle className="h-4 w-4" />
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <div className="rounded-lg px-4 py-2 border border-yellow-500/30 bg-yellow-500/10">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">Approval Required</span>
                            <Badge variant="outline" className="text-xs border-yellow-500">
                              {approval.toolName}
                            </Badge>
                          </div>
                          <Button size="sm" variant="outline" className="h-7">
                            Review
                          </Button>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">{approval.reason}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Typing Indicator */}
            {isTyping && (
              <div className="flex gap-3">
                <Avatar className="h-8 w-8">
                  <AvatarFallback className="bg-primary text-primary-foreground">
                    <AgentIcon className="h-4 w-4" />
                  </AvatarFallback>
                </Avatar>
                <div className="bg-muted rounded-lg px-4 py-2">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Message Input */}
        <div className="p-4 border-t">
          <div className="flex gap-2">
            <Textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={`Message ${agentName}...`}
              className="min-h-[44px] max-h-32 resize-none"
              data-testid="input-message"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!message.trim() || sendMessageMutation.isPending}
              size="icon"
              className="h-[44px] w-[44px]"
              data-testid="button-send-message"
            >
              {sendMessageMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <Circle className={cn(
                "h-2 w-2 fill-current",
                wsConnected ? "text-green-500" : "text-yellow-500"
              )} />
              <span>{wsConnected ? "Connected" : "Connecting..."}</span>
            </div>
            <div className="flex items-center gap-1">
              <Shield className="h-3 w-3" />
              <span>End-to-end encrypted</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Info Panel (optional, can be toggled) */}
      <Card className="w-72 hidden xl:flex flex-col">
        <CardHeader>
          <CardTitle className="text-lg">About {agentName}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col items-center text-center">
            <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center mb-3">
              <AgentIcon className="h-10 w-10 text-primary" />
            </div>
            <h3 className="font-semibold">{agentName}</h3>
            <p className="text-sm text-muted-foreground mt-1">
              {agentDescription}
            </p>
          </div>

          <Separator />

          <div className="space-y-3">
            <h4 className="font-medium text-sm">Capabilities</h4>
            <div className="space-y-2">
              {isDoctor ? (
                <>
                  <div className="flex items-center gap-2 text-sm">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span>Schedule Management</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Stethoscope className="h-4 w-4 text-muted-foreground" />
                    <span>Patient Summaries</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                    <span>Health Alert Triage</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <MessageSquare className="h-4 w-4 text-muted-foreground" />
                    <span>Email Categorization</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-2 text-sm">
                    <Heart className="h-4 w-4 text-muted-foreground" />
                    <span>Health Monitoring</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <MessageSquare className="h-4 w-4 text-muted-foreground" />
                    <span>Symptom Logging</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span>Medication Reminders</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Bot className="h-4 w-4 text-muted-foreground" />
                    <span>Wellness Support</span>
                  </div>
                </>
              )}
            </div>
          </div>

          <Separator />

          <div className="space-y-2">
            <h4 className="font-medium text-sm">Security</h4>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Shield className="h-4 w-4" />
              <span>HIPAA Compliant</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Shield className="h-4 w-4" />
              <span>End-to-end Encryption</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Shield className="h-4 w-4" />
              <span>Audit Logging</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
    </>
  );
}
