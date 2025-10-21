import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Calendar, MessageSquare, Clock, FileText, ChevronRight, Heart, Bot } from "lucide-react";
import { Link } from "wouter";
import { useState } from "react";

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

interface ChatMessage {
  id: string;
  sessionId: string;
  userId: string;
  role: string;
  content: string;
  agentType: string | null;
  medicalEntities: any;
  createdAt: Date;
  patientContextId: string | null;
}

export default function PreviousSessions() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  const { data: sessions, isLoading: sessionsLoading } = useQuery<ChatSession[]>({
    queryKey: ['/api/chat/sessions'],
  });

  const { data: messages } = useQuery<ChatMessage[]>({
    queryKey: ['/api/chat/sessions', selectedSessionId],
    enabled: !!selectedSessionId,
  });

  const formatDate = (date: Date | string) => {
    const d = new Date(date);
    return d.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getAgentIcon = (agentType: string) => {
    return agentType === 'clona' ? <Heart className="h-4 w-4" /> : <Bot className="h-4 w-4" />;
  };

  const getAgentName = (agentType: string) => {
    return agentType === 'clona' ? 'Agent Clona' : 'Assistant Lysa';
  };

  if (sessionsLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-muted rounded w-1/3"></div>
          <div className="h-32 bg-muted rounded"></div>
          <div className="h-32 bg-muted rounded"></div>
        </div>
      </div>
    );
  }

  const selectedSession = sessions?.find(s => s.id === selectedSessionId);

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2" data-testid="text-page-title">Previous Sessions</h1>
        <p className="text-muted-foreground">
          View your complete medical history from all conversations with Agent Clona and Assistant Lysa
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Sessions List */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="h-5 w-5" />
                All Sessions
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <ScrollArea className="h-[600px]">
                <div className="space-y-2 p-4">
                  {sessions && sessions.length > 0 ? (
                    sessions.map((session) => (
                      <button
                        key={session.id}
                        onClick={() => setSelectedSessionId(session.id)}
                        className={`w-full text-left p-4 rounded-md border transition-colors hover-elevate ${
                          selectedSessionId === session.id
                            ? 'bg-accent border-accent-foreground/20'
                            : 'bg-card border-border'
                        }`}
                        data-testid={`button-session-${session.id}`}
                      >
                        <div className="flex items-start justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2">
                            {getAgentIcon(session.agentType)}
                            <span className="text-sm font-medium">
                              {getAgentName(session.agentType)}
                            </span>
                          </div>
                          <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                        </div>
                        <p className="text-xs text-muted-foreground mb-2 line-clamp-2">
                          {session.sessionTitle || 'No title'}
                        </p>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          {formatDate(session.startedAt)}
                        </div>
                        {session.messageCount && (
                          <div className="mt-2">
                            <Badge variant="secondary" className="text-xs">
                              {session.messageCount} messages
                            </Badge>
                          </div>
                        )}
                      </button>
                    ))
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <MessageSquare className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>No previous sessions</p>
                      <Link href="/chat">
                        <Button variant="link" size="sm" className="mt-2" data-testid="button-start-chat">
                          Start your first chat
                        </Button>
                      </Link>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Session Details */}
        <div className="lg:col-span-2">
          {selectedSession ? (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <CardTitle className="flex items-center gap-2 mb-2">
                        {getAgentIcon(selectedSession.agentType)}
                        {getAgentName(selectedSession.agentType)}
                      </CardTitle>
                      <p className="text-sm text-muted-foreground">
                        {formatDate(selectedSession.startedAt)}
                        {selectedSession.endedAt && (
                          <> • Ended {formatDate(selectedSession.endedAt)}</>
                        )}
                      </p>
                    </div>
                    {!selectedSession.endedAt && (
                      <Badge variant="default">Active</Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {selectedSession.sessionTitle && (
                    <div>
                      <h3 className="text-sm font-medium mb-1">Topic</h3>
                      <p className="text-sm text-muted-foreground">{selectedSession.sessionTitle}</p>
                    </div>
                  )}

                  {selectedSession.symptomsDiscussed && selectedSession.symptomsDiscussed.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium mb-2">Symptoms Discussed</h3>
                      <div className="flex flex-wrap gap-2">
                        {selectedSession.symptomsDiscussed.map((symptom, idx) => (
                          <Badge key={idx} variant="outline" data-testid={`badge-symptom-${idx}`}>
                            {symptom}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedSession.aiSummary && (
                    <>
                      <Separator />
                      <div>
                        <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
                          <FileText className="h-4 w-4" />
                          AI Summary
                        </h3>
                        <Card className="bg-muted/50">
                          <CardContent className="pt-4">
                            <p className="text-sm whitespace-pre-wrap" data-testid="text-ai-summary">
                              {selectedSession.aiSummary}
                            </p>
                          </CardContent>
                        </Card>
                      </div>
                    </>
                  )}

                  {selectedSession.doctorNotes && (
                    <>
                      <Separator />
                      <div>
                        <h3 className="text-sm font-medium mb-2">Doctor's Notes</h3>
                        <p className="text-sm text-muted-foreground">{selectedSession.doctorNotes}</p>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>

              {/* Messages */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <MessageSquare className="h-5 w-5" />
                    Conversation
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[400px]">
                    <div className="space-y-4 pr-4">
                      {messages && messages.length > 0 ? (
                        messages.map((message, idx) => (
                          <div
                            key={message.id}
                            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            data-testid={`message-${idx}`}
                          >
                            <div
                              className={`max-w-[80%] p-3 rounded-lg ${
                                message.role === 'user'
                                  ? 'bg-primary text-primary-foreground'
                                  : 'bg-muted'
                              }`}
                            >
                              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                              <p className="text-xs opacity-70 mt-1">
                                {new Date(message.createdAt).toLocaleTimeString('en-US', {
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </p>
                            </div>
                          </div>
                        ))
                      ) : (
                        <p className="text-sm text-muted-foreground text-center py-8">
                          No messages in this session
                        </p>
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card className="h-[600px] flex items-center justify-center">
              <CardContent className="text-center">
                <Calendar className="h-16 w-16 mx-auto mb-4 text-muted-foreground opacity-50" />
                <h3 className="text-lg font-medium mb-2">Select a session</h3>
                <p className="text-sm text-muted-foreground">
                  Choose a session from the list to view details and messages
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
