import { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Skeleton } from '@/components/ui/skeleton';
import { apiRequest, queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { 
  MessageSquare, Plus, Send, Loader2, Bot, User, 
  Sparkles, RefreshCw, Trash2, Clock, Database, ChevronRight
} from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

interface QASession {
  id: string;
  title: string;
  cohortId?: string;
  studyId?: string;
  datasetId?: string;
  messageCount: number;
  createdAt: string;
  updatedAt: string;
}

interface QAMessage {
  id: string;
  sessionId: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  metadata?: {
    model?: string;
    tokensUsed?: number;
    dataContext?: string;
  };
}

export function ResearchQATab() {
  const { toast } = useToast();
  const [selectedSession, setSelectedSession] = useState<QASession | null>(null);
  const [newMessage, setNewMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { data: sessions = [], isLoading: sessionsLoading, refetch: refetchSessions } = useQuery<QASession[]>({
    queryKey: ['/python-api/v1/research-center/ai/qa/sessions'],
  });

  const { data: sessionDetails, isLoading: messagesLoading } = useQuery<{
    session: QASession;
    messages: QAMessage[];
  }>({
    queryKey: ['/python-api/v1/research-center/ai/qa/sessions', selectedSession?.id],
    queryFn: async () => {
      if (!selectedSession?.id) return { session: null, messages: [] };
      const res = await fetch(`/python-api/v1/research-center/ai/qa/sessions/${selectedSession.id}`);
      if (!res.ok) throw new Error('Failed to fetch session');
      return res.json();
    },
    enabled: !!selectedSession,
  });

  const messages = sessionDetails?.messages || [];

  const createSessionMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('/python-api/v1/research-center/ai/qa/sessions', {
        method: 'POST',
        json: { title: `Research Session ${format(new Date(), 'MMM d, h:mm a')}` }
      });
      return response.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/python-api/v1/research-center/ai/qa/sessions'] });
      setSelectedSession(data);
      toast({ title: 'Session Created', description: 'New Q&A session started.' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const sendMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      if (!selectedSession) throw new Error('No session selected');
      const response = await apiRequest(`/python-api/v1/research-center/ai/qa/sessions/${selectedSession.id}/messages`, {
        method: 'POST',
        json: { content }
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/python-api/v1/research-center/ai/qa/sessions', selectedSession?.id] });
      queryClient.invalidateQueries({ queryKey: ['/python-api/v1/research-center/ai/qa/sessions'] });
      setNewMessage('');
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSendMessage = () => {
    if (!newMessage.trim()) return;
    sendMessageMutation.mutate(newMessage.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="h-[calc(100vh-300px)] min-h-[600px] flex gap-4">
      <Card className="w-80 flex flex-col" data-testid="card-sessions-list">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Q&A Sessions</CardTitle>
            <div className="flex gap-1">
              <Button variant="ghost" size="icon" onClick={() => refetchSessions()} data-testid="button-refresh-sessions">
                <RefreshCw className="h-4 w-4" />
              </Button>
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={() => createSessionMutation.mutate()}
                disabled={createSessionMutation.isPending}
                data-testid="button-new-session"
              >
                {createSessionMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Plus className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
          <CardDescription>AI-powered research data analysis</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 p-0">
          <ScrollArea className="h-full">
            {sessionsLoading ? (
              <div className="space-y-2 p-4">
                {[1, 2, 3].map((i) => <Skeleton key={i} className="h-16 w-full" />)}
              </div>
            ) : sessions.length === 0 ? (
              <div className="text-center py-8 px-4 text-muted-foreground">
                <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No sessions yet</p>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="mt-3"
                  onClick={() => createSessionMutation.mutate()}
                >
                  Start First Session
                </Button>
              </div>
            ) : (
              <div className="p-2 space-y-1">
                {sessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => setSelectedSession(session)}
                    className={`w-full p-3 rounded-lg text-left transition-colors hover-elevate ${
                      selectedSession?.id === session.id 
                        ? 'bg-primary/10 border border-primary/30' 
                        : 'hover:bg-muted'
                    }`}
                    data-testid={`session-${session.id}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">{session.title}</div>
                        <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                          <MessageSquare className="h-3 w-3" />
                          <span>{session.messageCount} messages</span>
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {formatDistanceToNow(new Date(session.updatedAt), { addSuffix: true })}
                        </div>
                      </div>
                      <ChevronRight className={`h-4 w-4 text-muted-foreground transition-transform ${
                        selectedSession?.id === session.id ? 'rotate-90' : ''
                      }`} />
                    </div>
                  </button>
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>

      <Card className="flex-1 flex flex-col" data-testid="card-chat">
        {!selectedSession ? (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Sparkles className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <h3 className="font-medium mb-2">Research Q&A Assistant</h3>
              <p className="text-sm max-w-md">
                Ask questions about your research data. The AI can analyze cohorts, 
                datasets, and provide insights while maintaining HIPAA compliance.
              </p>
              <Button 
                className="mt-4"
                onClick={() => createSessionMutation.mutate()}
                disabled={createSessionMutation.isPending}
                data-testid="button-start-session"
              >
                <Plus className="h-4 w-4 mr-2" />
                Start New Session
              </Button>
            </div>
          </div>
        ) : (
          <>
            <CardHeader className="pb-3 border-b">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-primary" />
                    {selectedSession.title}
                  </CardTitle>
                  <CardDescription className="flex items-center gap-3 mt-1">
                    <span className="flex items-center gap-1">
                      <MessageSquare className="h-3 w-3" />
                      {messages.length} messages
                    </span>
                    {selectedSession.datasetId && (
                      <Badge variant="outline" className="gap-1">
                        <Database className="h-3 w-3" />
                        Dataset linked
                      </Badge>
                    )}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent className="flex-1 p-0 overflow-hidden">
              <ScrollArea className="h-full p-4">
                {messagesLoading ? (
                  <div className="space-y-4">
                    {[1, 2].map((i) => (
                      <div key={i} className="flex gap-3">
                        <Skeleton className="h-8 w-8 rounded-full" />
                        <Skeleton className="h-20 flex-1 rounded-lg" />
                      </div>
                    ))}
                  </div>
                ) : messages.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Bot className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Start the conversation by asking a question</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <div 
                        key={message.id} 
                        className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : ''}`}
                        data-testid={`message-${message.id}`}
                      >
                        {message.role === 'assistant' && (
                          <Avatar className="h-8 w-8">
                            <AvatarFallback className="bg-primary/10 text-primary">
                              <Bot className="h-4 w-4" />
                            </AvatarFallback>
                          </Avatar>
                        )}
                        <div className={`max-w-[70%] ${
                          message.role === 'user' 
                            ? 'bg-primary text-primary-foreground rounded-2xl rounded-tr-sm px-4 py-2' 
                            : 'bg-muted rounded-2xl rounded-tl-sm px-4 py-3'
                        }`}>
                          <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                          <div className={`text-xs mt-1 ${
                            message.role === 'user' ? 'text-primary-foreground/70' : 'text-muted-foreground'
                          }`}>
                            {format(new Date(message.createdAt), 'h:mm a')}
                          </div>
                        </div>
                        {message.role === 'user' && (
                          <Avatar className="h-8 w-8">
                            <AvatarFallback className="bg-primary">
                              <User className="h-4 w-4 text-primary-foreground" />
                            </AvatarFallback>
                          </Avatar>
                        )}
                      </div>
                    ))}
                    {sendMessageMutation.isPending && (
                      <div className="flex gap-3">
                        <Avatar className="h-8 w-8">
                          <AvatarFallback className="bg-primary/10 text-primary">
                            <Bot className="h-4 w-4" />
                          </AvatarFallback>
                        </Avatar>
                        <div className="bg-muted rounded-2xl rounded-tl-sm px-4 py-3">
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Analyzing...
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </ScrollArea>
            </CardContent>

            <div className="p-4 border-t">
              <div className="flex gap-2">
                <Input
                  placeholder="Ask about your research data..."
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={sendMessageMutation.isPending}
                  className="flex-1"
                  data-testid="input-message"
                />
                <Button 
                  onClick={handleSendMessage}
                  disabled={!newMessage.trim() || sendMessageMutation.isPending}
                  data-testid="button-send-message"
                >
                  {sendMessageMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground mt-2 text-center">
                AI responses are based on de-identified research data. All queries are logged for HIPAA compliance.
              </p>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}
