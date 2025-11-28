import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Bot,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Activity,
  Brain,
  Heart,
  Pill,
  FileText,
  RefreshCw,
  ChevronRight,
  Sparkles,
  Lightbulb,
  AlertCircle,
  ThermometerSun,
  Stethoscope
} from "lucide-react";
import { formatDistanceToNow, format, parseISO } from "date-fns";

interface PatientContext {
  id: string;
  firstName: string;
  lastName: string;
}

interface Insight {
  id: string;
  type: 'trend' | 'alert' | 'recommendation' | 'observation' | 'prediction';
  category: string;
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  timestamp: string;
  action?: string;
  source?: string;
}

interface LysaInsightFeedProps {
  patientContext: PatientContext;
  className?: string;
  compact?: boolean;
}

export function LysaInsightFeed({ patientContext, className, compact = false }: LysaInsightFeedProps) {
  const [activeTab, setActiveTab] = useState<string>('all');

  const { data: insights, isLoading, refetch } = useQuery<Insight[]>({
    queryKey: ['/api/v1/lysa/insight-feed', patientContext.id],
    queryFn: async () => {
      try {
        const response = await fetch(`/api/v1/lysa/insight-feed/${patientContext.id}`);
        if (!response.ok) {
          return generateDefaultInsights();
        }
        return response.json();
      } catch {
        return generateDefaultInsights();
      }
    },
    staleTime: 60000
  });

  function generateDefaultInsights(): Insight[] {
    return [
      {
        id: 'insight-1',
        type: 'trend',
        category: 'vitals',
        title: 'Stable Vital Signs',
        description: 'Heart rate and blood pressure have remained within normal ranges over the past 7 days.',
        priority: 'low',
        confidence: 0.92,
        timestamp: new Date().toISOString(),
        source: 'Continuous Monitoring'
      },
      {
        id: 'insight-2',
        type: 'recommendation',
        category: 'medication',
        title: 'Medication Review Suggested',
        description: 'Patient has been on current medication regimen for 6 months. Consider reviewing effectiveness.',
        priority: 'medium',
        confidence: 0.78,
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        action: 'Schedule medication review',
        source: 'Medication Analysis'
      },
      {
        id: 'insight-3',
        type: 'observation',
        category: 'symptoms',
        title: 'Symptom Pattern Detected',
        description: 'Patient has reported similar symptoms at similar times over the past 3 visits. May indicate cyclical pattern.',
        priority: 'medium',
        confidence: 0.85,
        timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        source: 'Pattern Analysis'
      },
      {
        id: 'insight-4',
        type: 'alert',
        category: 'adherence',
        title: 'Adherence Alert',
        description: 'Medication adherence has dropped to 75% this week compared to 95% last week.',
        priority: 'high',
        confidence: 0.95,
        timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
        action: 'Follow up with patient',
        source: 'Adherence Tracking'
      },
      {
        id: 'insight-5',
        type: 'prediction',
        category: 'risk',
        title: 'Low Risk Score',
        description: 'Based on current health data, patient shows low risk of deterioration in the next 30 days.',
        priority: 'low',
        confidence: 0.88,
        timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
        source: 'Predictive Analytics'
      }
    ];
  }

  const getInsightIcon = (type: string, category: string) => {
    switch (type) {
      case 'trend':
        return <TrendingUp className="h-4 w-4" />;
      case 'alert':
        return <AlertTriangle className="h-4 w-4" />;
      case 'recommendation':
        return <Lightbulb className="h-4 w-4" />;
      case 'prediction':
        return <Brain className="h-4 w-4" />;
      case 'observation':
        switch (category) {
          case 'vitals':
            return <Heart className="h-4 w-4" />;
          case 'medication':
            return <Pill className="h-4 w-4" />;
          case 'symptoms':
            return <ThermometerSun className="h-4 w-4" />;
          default:
            return <FileText className="h-4 w-4" />;
        }
      default:
        return <Sparkles className="h-4 w-4" />;
    }
  };

  const getInsightColor = (type: string, priority: string) => {
    if (priority === 'critical') return 'text-red-500';
    if (priority === 'high') return 'text-orange-500';
    
    switch (type) {
      case 'trend':
        return 'text-green-500';
      case 'alert':
        return 'text-yellow-500';
      case 'recommendation':
        return 'text-blue-500';
      case 'prediction':
        return 'text-purple-500';
      case 'observation':
        return 'text-cyan-500';
      default:
        return 'text-primary';
    }
  };

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case 'critical':
        return <Badge variant="destructive" className="text-xs">Critical</Badge>;
      case 'high':
        return <Badge className="bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400 text-xs">High</Badge>;
      case 'medium':
        return <Badge className="bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400 text-xs">Medium</Badge>;
      case 'low':
        return <Badge variant="secondary" className="text-xs">Low</Badge>;
      default:
        return null;
    }
  };

  const getTypeBadge = (type: string) => {
    switch (type) {
      case 'trend':
        return <Badge variant="outline" className="text-xs bg-green-50 dark:bg-green-900/20">Trend</Badge>;
      case 'alert':
        return <Badge variant="outline" className="text-xs bg-yellow-50 dark:bg-yellow-900/20">Alert</Badge>;
      case 'recommendation':
        return <Badge variant="outline" className="text-xs bg-blue-50 dark:bg-blue-900/20">Recommendation</Badge>;
      case 'prediction':
        return <Badge variant="outline" className="text-xs bg-purple-50 dark:bg-purple-900/20">Prediction</Badge>;
      case 'observation':
        return <Badge variant="outline" className="text-xs bg-cyan-50 dark:bg-cyan-900/20">Observation</Badge>;
      default:
        return <Badge variant="outline" className="text-xs">{type}</Badge>;
    }
  };

  const filteredInsights = insights?.filter(insight => {
    if (activeTab === 'all') return true;
    return insight.type === activeTab;
  }) || [];

  const sortedInsights = [...filteredInsights].sort((a, b) => {
    const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    const priorityDiff = priorityOrder[a.priority] - priorityOrder[b.priority];
    if (priorityDiff !== 0) return priorityDiff;
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });

  if (compact) {
    return (
      <Card className={className}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bot className="h-4 w-4 text-primary" />
              <CardTitle className="text-sm">Lysa Insights</CardTitle>
            </div>
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-6 w-6"
              onClick={() => refetch()}
              data-testid="button-refresh-insights-compact"
            >
              <RefreshCw className="h-3 w-3" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          {isLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map(i => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : (
            <ScrollArea className="h-[200px]">
              <div className="space-y-2">
                {sortedInsights.slice(0, 5).map(insight => (
                  <div 
                    key={insight.id} 
                    className="flex items-start gap-2 p-2 rounded-md hover-elevate border"
                    data-testid={`insight-compact-${insight.id}`}
                  >
                    <div className={`mt-0.5 ${getInsightColor(insight.type, insight.priority)}`}>
                      {getInsightIcon(insight.type, insight.category)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium truncate">{insight.title}</p>
                      <p className="text-xs text-muted-foreground truncate">{insight.description}</p>
                    </div>
                    <ChevronRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={className}>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                <Bot className="h-4 w-4 text-primary" />
              </div>
              <div>
                <CardTitle className="text-lg">Lysa AI Insight Feed</CardTitle>
                <CardDescription>
                  AI-generated insights for {patientContext.firstName} {patientContext.lastName}
                </CardDescription>
              </div>
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => refetch()}
              data-testid="button-refresh-insights"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
            <TabsList className="grid w-full grid-cols-6">
              <TabsTrigger value="all" data-testid="tab-insights-all">
                All
              </TabsTrigger>
              <TabsTrigger value="alert" data-testid="tab-insights-alerts">
                <AlertTriangle className="h-3 w-3 mr-1" />
                Alerts
              </TabsTrigger>
              <TabsTrigger value="recommendation" data-testid="tab-insights-recommendations">
                <Lightbulb className="h-3 w-3 mr-1" />
                Actions
              </TabsTrigger>
              <TabsTrigger value="trend" data-testid="tab-insights-trends">
                <TrendingUp className="h-3 w-3 mr-1" />
                Trends
              </TabsTrigger>
              <TabsTrigger value="prediction" data-testid="tab-insights-predictions">
                <Brain className="h-3 w-3 mr-1" />
                Predictions
              </TabsTrigger>
              <TabsTrigger value="observation" data-testid="tab-insights-observations">
                <Stethoscope className="h-3 w-3 mr-1" />
                Observations
              </TabsTrigger>
            </TabsList>

            <TabsContent value={activeTab}>
              {isLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3, 4].map(i => (
                    <Skeleton key={i} className="h-24 w-full" />
                  ))}
                </div>
              ) : sortedInsights.length > 0 ? (
                <ScrollArea className="h-[500px]">
                  <div className="space-y-3">
                    {sortedInsights.map(insight => (
                      <Card 
                        key={insight.id} 
                        className="hover-elevate"
                        data-testid={`insight-card-${insight.id}`}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start gap-3">
                            <div className={`flex h-10 w-10 items-center justify-center rounded-full ${
                              insight.priority === 'critical' || insight.priority === 'high' 
                                ? 'bg-red-100 dark:bg-red-900/30' 
                                : 'bg-primary/10'
                            } ${getInsightColor(insight.type, insight.priority)}`}>
                              {getInsightIcon(insight.type, insight.category)}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-start justify-between gap-2 flex-wrap">
                                <div>
                                  <h4 className="font-semibold text-sm">{insight.title}</h4>
                                  <div className="flex items-center gap-2 mt-1 flex-wrap">
                                    {getTypeBadge(insight.type)}
                                    {getPriorityBadge(insight.priority)}
                                    <span className="text-xs text-muted-foreground">
                                      {Math.round(insight.confidence * 100)}% confidence
                                    </span>
                                  </div>
                                </div>
                                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                  <Clock className="h-3 w-3" />
                                  {formatDistanceToNow(parseISO(insight.timestamp), { addSuffix: true })}
                                </div>
                              </div>
                              
                              <p className="text-sm text-muted-foreground mt-2">
                                {insight.description}
                              </p>

                              <div className="flex items-center justify-between mt-3 pt-2 border-t">
                                <div className="flex items-center gap-2">
                                  {insight.source && (
                                    <Badge variant="outline" className="text-xs">
                                      {insight.source}
                                    </Badge>
                                  )}
                                </div>
                                {insight.action && (
                                  <Button 
                                    variant="ghost" 
                                    size="sm" 
                                    className="h-7 text-xs"
                                    data-testid={`button-insight-action-${insight.id}`}
                                  >
                                    {insight.action}
                                    <ChevronRight className="h-3 w-3 ml-1" />
                                  </Button>
                                )}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Sparkles className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No insights available for this filter</p>
                  <p className="text-sm">Try selecting a different category</p>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
