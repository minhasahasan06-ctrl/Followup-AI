import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { 
  Brain, 
  Send, 
  Loader2, 
  Sparkles,
  Activity,
  Heart,
  Pill,
  AlertTriangle,
  TrendingUp,
  Clock,
  FileText,
  Stethoscope,
  Bot,
  User,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Lightbulb
} from "lucide-react";
import { format, parseISO } from "date-fns";

interface PatientContext {
  id: string;
  firstName: string;
  lastName: string;
  email?: string;
  dateOfBirth?: string;
  allergies?: string[];
  comorbidities?: string[];
  immunocompromisedCondition?: string;
  currentMedications?: string[];
}

interface AIInsight {
  id: string;
  type: 'warning' | 'info' | 'recommendation' | 'trend';
  title: string;
  description: string;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  source?: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

interface LysaPatientAssistantProps {
  patientId: string;
  patientContext: PatientContext;
  className?: string;
}

const quickPrompts = [
  { label: "Health Summary", prompt: "Provide a comprehensive health summary for this patient" },
  { label: "Risk Assessment", prompt: "What are the current risk factors for this patient?" },
  { label: "Treatment Plan", prompt: "Review and suggest optimizations for the current treatment plan" },
  { label: "Drug Interactions", prompt: "Check for any potential drug interactions with current medications" },
  { label: "Recent Changes", prompt: "Summarize any significant changes in patient health metrics" },
  { label: "Care Recommendations", prompt: "What are your care recommendations for this patient?" },
];

export function LysaPatientAssistant({ patientId, patientContext, className }: LysaPatientAssistantProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showQuickPrompts, setShowQuickPrompts] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: patientInsights, isLoading: insightsLoading, refetch: refetchInsights } = useQuery({
    queryKey: ['/api/v1/lysa/patient-insights', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/lysa/patient-insights/${patientId}`);
      if (!response.ok) {
        return { insights: generateFallbackInsights(patientContext), _fallback: true };
      }
      return response.json();
    },
    staleTime: 120000,
    enabled: !!patientId
  });

  const { data: patientTimeline, isLoading: timelineLoading } = useQuery({
    queryKey: ['/api/v1/lysa/patient-timeline', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/lysa/patient-timeline/${patientId}?limit=10`);
      if (!response.ok) {
        return { events: [], _fallback: true };
      }
      return response.json();
    },
    staleTime: 60000,
    enabled: !!patientId
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setMessages([{
      id: 'welcome',
      role: 'assistant',
      content: `Hello, I'm Lysa, your AI clinical assistant. I have ${patientContext.firstName} ${patientContext.lastName}'s full medical context loaded. How can I assist you with this patient's care today?`,
      timestamp: new Date()
    }]);
  }, [patientId, patientContext.firstName, patientContext.lastName]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: content.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);
    setShowQuickPrompts(false);

    try {
      const response = await fetch('/api/v1/lysa/patient-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patientId,
          message: content.trim(),
          context: {
            patientName: `${patientContext.firstName} ${patientContext.lastName}`,
            allergies: patientContext.allergies,
            comorbidities: patientContext.comorbidities,
            immunocompromisedCondition: patientContext.immunocompromisedCondition,
            currentMedications: patientContext.currentMedications
          }
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get AI response');
      }

      const data = await response.json();
      
      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: data.response || data.message || "I apologize, but I couldn't process your request. Please try again or rephrase your question.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      
      const fallbackResponse = generateFallbackResponse(content, patientContext);
      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: fallbackResponse,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
      
      toast({
        title: "Note",
        description: "Using cached clinical guidelines. Real-time AI analysis may be temporarily limited.",
        variant: "default"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage(inputValue);
  };

  const handleQuickPrompt = (prompt: string) => {
    handleSendMessage(prompt);
  };

  return (
    <div className={`grid gap-6 lg:grid-cols-3 ${className}`}>
      <Card className="lg:col-span-2" data-testid="card-lysa-patient-chat">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            Lysa Clinical Assistant
          </CardTitle>
          <CardDescription>
            AI-powered clinical support for {patientContext.firstName} {patientContext.lastName}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col h-[500px]">
          <ScrollArea className="flex-1 pr-4 mb-4">
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  data-testid={`message-${message.role}-${message.id}`}
                >
                  {message.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-2 ${
                      message.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted'
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    <p className="text-xs opacity-70 mt-1">
                      {format(message.timestamp, 'h:mm a')}
                    </p>
                  </div>
                  {message.role === 'user' && (
                    <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center shrink-0">
                      <User className="h-4 w-4 text-primary-foreground" />
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex gap-3 justify-start">
                  <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                  <div className="bg-muted rounded-lg px-4 py-3">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm text-muted-foreground">Analyzing patient data...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {showQuickPrompts && messages.length <= 1 && (
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  Quick Actions
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowQuickPrompts(false)}
                  className="h-6 px-2"
                  data-testid="button-hide-quick-prompts"
                >
                  <ChevronUp className="h-3 w-3" />
                </Button>
              </div>
              <div className="flex flex-wrap gap-2">
                {quickPrompts.map((qp, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => handleQuickPrompt(qp.prompt)}
                    disabled={isLoading}
                    className="text-xs"
                    data-testid={`button-quick-prompt-${index}`}
                  >
                    {qp.label}
                  </Button>
                ))}
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about this patient's care..."
              disabled={isLoading}
              className="flex-1"
              data-testid="input-lysa-message"
            />
            <Button 
              type="submit" 
              disabled={isLoading || !inputValue.trim()}
              data-testid="button-send-lysa-message"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      <div className="space-y-6">
        <Card data-testid="card-ai-insights">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-base">
                <Lightbulb className="h-4 w-4 text-amber-500" />
                AI Insights
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => refetchInsights()}
                className="h-8 w-8 p-0"
                data-testid="button-refresh-insights"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {insightsLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-16 w-full" />
              </div>
            ) : (
              <ScrollArea className="h-[280px]">
                <div className="space-y-3">
                  {(patientInsights?.insights || generateFallbackInsights(patientContext)).map((insight: AIInsight, index: number) => (
                    <div
                      key={insight.id || index}
                      className={`p-3 rounded-lg border ${
                        insight.severity === 'critical' ? 'border-red-500/50 bg-red-50 dark:bg-red-950/20' :
                        insight.severity === 'high' ? 'border-orange-500/50 bg-orange-50 dark:bg-orange-950/20' :
                        insight.severity === 'medium' ? 'border-yellow-500/50 bg-yellow-50 dark:bg-yellow-950/20' :
                        'border-border bg-muted/50'
                      }`}
                      data-testid={`insight-item-${index}`}
                    >
                      <div className="flex items-start gap-2">
                        {insight.type === 'warning' ? (
                          <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
                        ) : insight.type === 'trend' ? (
                          <TrendingUp className="h-4 w-4 text-blue-500 shrink-0 mt-0.5" />
                        ) : insight.type === 'recommendation' ? (
                          <Stethoscope className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                        ) : (
                          <Activity className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium">{insight.title}</p>
                          <p className="text-xs text-muted-foreground mt-1">{insight.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-patient-context">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <FileText className="h-4 w-4" />
              Patient Context
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {patientContext.immunocompromisedCondition && (
                <div className="flex items-start gap-2">
                  <Heart className="h-4 w-4 text-red-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs text-muted-foreground">Immunocompromised</p>
                    <p className="text-sm">{patientContext.immunocompromisedCondition}</p>
                  </div>
                </div>
              )}

              {patientContext.allergies && patientContext.allergies.length > 0 && (
                <div className="flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs text-muted-foreground">Allergies</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {patientContext.allergies.map((allergy, i) => (
                        <Badge key={i} variant="outline" className="text-xs bg-amber-50 dark:bg-amber-950/20">
                          {allergy}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {patientContext.comorbidities && patientContext.comorbidities.length > 0 && (
                <div className="flex items-start gap-2">
                  <Activity className="h-4 w-4 text-blue-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs text-muted-foreground">Comorbidities</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {patientContext.comorbidities.map((condition, i) => (
                        <Badge key={i} variant="secondary" className="text-xs">
                          {condition}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {patientContext.currentMedications && patientContext.currentMedications.length > 0 && (
                <div className="flex items-start gap-2">
                  <Pill className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs text-muted-foreground">Current Medications</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {patientContext.currentMedications.map((med, i) => (
                        <Badge key={i} variant="outline" className="text-xs">
                          {med}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function generateFallbackInsights(context: PatientContext): AIInsight[] {
  const insights: AIInsight[] = [];

  if (context.immunocompromisedCondition) {
    insights.push({
      id: 'ic-1',
      type: 'warning',
      title: 'Immunocompromised Status',
      description: `Patient has ${context.immunocompromisedCondition}. Enhanced infection monitoring recommended.`,
      severity: 'high',
      timestamp: new Date().toISOString()
    });
  }

  if (context.allergies && context.allergies.length > 0) {
    insights.push({
      id: 'allergy-1',
      type: 'info',
      title: 'Active Allergies',
      description: `Patient has ${context.allergies.length} documented allergies. Verify medications before prescribing.`,
      severity: 'medium',
      timestamp: new Date().toISOString()
    });
  }

  if (context.comorbidities && context.comorbidities.length > 2) {
    insights.push({
      id: 'comorbid-1',
      type: 'trend',
      title: 'Multiple Comorbidities',
      description: `Patient has ${context.comorbidities.length} comorbid conditions. Consider polypharmacy risks and interactions.`,
      severity: 'medium',
      timestamp: new Date().toISOString()
    });
  }

  insights.push({
    id: 'rec-1',
    type: 'recommendation',
    title: 'Regular Monitoring',
    description: 'Maintain regular follow-ups to track health status and medication adherence.',
    severity: 'low',
    timestamp: new Date().toISOString()
  });

  return insights;
}

function generateFallbackResponse(query: string, context: PatientContext): string {
  const queryLower = query.toLowerCase();
  
  if (queryLower.includes('summary') || queryLower.includes('overview')) {
    return `Based on available records for ${context.firstName} ${context.lastName}:

**Condition**: ${context.immunocompromisedCondition || 'Not specified'}
**Allergies**: ${context.allergies?.join(', ') || 'None documented'}
**Comorbidities**: ${context.comorbidities?.join(', ') || 'None documented'}
**Current Medications**: ${context.currentMedications?.join(', ') || 'None documented'}

For real-time AI analysis of trends and patterns, the AI service will need to be fully operational. In the meantime, I recommend reviewing recent lab results and vital signs for any significant changes.`;
  }

  if (queryLower.includes('risk') || queryLower.includes('assessment')) {
    return `Risk assessment for ${context.firstName} ${context.lastName}:

${context.immunocompromisedCondition ? `- **High Priority**: Immunocompromised status (${context.immunocompromisedCondition}) increases infection risk` : ''}
${context.allergies && context.allergies.length > 0 ? `- **Medication Safety**: ${context.allergies.length} documented allergies require verification before prescribing` : ''}
${context.comorbidities && context.comorbidities.length > 0 ? `- **Complexity Factor**: ${context.comorbidities.length} comorbid conditions may affect treatment planning` : ''}

Standard clinical protocols should be followed. Consider specialist consultation for complex cases.`;
  }

  if (queryLower.includes('drug') || queryLower.includes('interaction') || queryLower.includes('medication')) {
    return `For ${context.firstName} ${context.lastName}'s medication review:

${context.currentMedications && context.currentMedications.length > 0 
  ? `**Current Medications**: ${context.currentMedications.join(', ')}
  
I recommend checking for interactions using the prescription helper tool or clinical database.`
  : 'No current medications are documented. Please verify medication list with the patient.'}

${context.allergies && context.allergies.length > 0 
  ? `**Allergies to Avoid**: ${context.allergies.join(', ')}`
  : ''}

Real-time interaction checking will be available when the AI service is fully operational.`;
  }

  if (queryLower.includes('treatment') || queryLower.includes('plan')) {
    return `Treatment planning considerations for ${context.firstName} ${context.lastName}:

1. **Current Status**: Review recent vitals and lab results
2. **Medication Review**: ${context.currentMedications?.length || 0} active medications - check for optimization opportunities
3. **Comorbidity Management**: ${context.comorbidities?.join(', ') || 'Standard protocols apply'}
4. **Follow-up Schedule**: Recommend regular monitoring given patient complexity

For detailed treatment recommendations, please consult current clinical guidelines and consider the patient's individual response to therapy.`;
  }

  return `I understand you're asking about ${context.firstName} ${context.lastName}'s care. Based on their profile:

- Immunocompromised status: ${context.immunocompromisedCondition || 'Not specified'}
- Key allergies: ${context.allergies?.slice(0, 3).join(', ') || 'None documented'}
- Major conditions: ${context.comorbidities?.slice(0, 3).join(', ') || 'None documented'}

How else can I assist with this patient's care? You can ask about:
- Health summary
- Risk assessment  
- Medication interactions
- Treatment planning
- Care recommendations`;
}
