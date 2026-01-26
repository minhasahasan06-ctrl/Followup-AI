import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { 
  TrendingUp, 
  TrendingDown, 
  Network,
  AlertCircle,
  Sparkles,
  RefreshCw,
  Activity,
  Brain,
  ArrowRight
} from "lucide-react";
import { format } from "date-fns";

interface CorrelationPattern {
  id: string;
  userId: string;
  patternName: string;
  patternType: 'positive' | 'negative' | 'neutral';
  factors: Array<{ type: string; name: string; value: any }>;
  correlationStrength: string;
  confidence: string;
  sampleSize: number;
  timeWindow: string;
  firstObserved: Date;
  lastObserved: Date;
  frequency: number;
  insight: string;
  recommendation: string;
  severity: 'low' | 'moderate' | 'high' | 'critical';
  createdAt: Date | null;
}

interface CorrelationReport {
  patterns: CorrelationPattern[];
  summary: string;
  recommendations: string[];
}

export default function CorrelationInsights() {
  const { toast } = useToast();

  const { data: patterns, isLoading: patternsLoading } = useQuery<CorrelationPattern[]>({
    queryKey: ["/api/correlations"],
  });

  const { data: report, isLoading: reportLoading } = useQuery<CorrelationReport>({
    queryKey: ["/api/correlations/report"],
  });

  const analyzeMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("/api/correlations/analyze", { method: "POST", json: {} });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/correlations"] });
      queryClient.invalidateQueries({ queryKey: ["/api/correlations/report"] });
      toast({
        title: "Analysis Complete",
        description: "Your health correlation patterns have been updated.",
      });
    },
    onError: () => {
      toast({
        title: "Analysis Failed",
        description: "Unable to analyze correlations. Please try again.",
        variant: "destructive",
      });
    },
  });

  const getSeverityBadge = (severity: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      low: "outline",
      moderate: "secondary",
      high: "default",
      critical: "destructive",
    };

    const colors: Record<string, string> = {
      low: "text-blue-600 dark:text-blue-400",
      moderate: "text-yellow-600 dark:text-yellow-400",
      high: "text-orange-600 dark:text-orange-400",
      critical: "text-red-600 dark:text-red-400",
    };

    return (
      <Badge 
        variant={variants[severity.toLowerCase()] || "outline"}
        className={colors[severity.toLowerCase()] || ""}
        data-testid={`badge-severity-${severity}`}
      >
        {severity.toUpperCase()}
      </Badge>
    );
  };

  const getPatternTypeIcon = (type: string) => {
    switch (type) {
      case 'positive':
        return <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />;
      case 'negative':
        return <TrendingDown className="w-5 h-5 text-red-600 dark:text-red-400" />;
      default:
        return <Network className="w-5 h-5 text-blue-600 dark:text-blue-400" />;
    }
  };

  const getCorrelationStrengthColor = (strength: number) => {
    const absStrength = Math.abs(strength);
    if (absStrength >= 0.7) return "text-green-600 dark:text-green-400";
    if (absStrength >= 0.4) return "text-yellow-600 dark:text-yellow-400";
    return "text-muted-foreground";
  };

  const highSeverityPatterns = patterns?.filter(p => 
    p.severity === 'critical' || p.severity === 'high'
  ) || [];

  const criticalPatterns = patterns?.filter(p => p.severity === 'critical') || [];

  return (
    <div className="container mx-auto p-6 space-y-6" data-testid="page-correlation-insights">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="w-8 h-8 text-primary" />
            Multi-Condition Correlation Engine
          </h1>
          <p className="text-muted-foreground mt-2">
            AI-powered detection of hidden patterns across your health data
          </p>
        </div>
        <Button 
          onClick={() => analyzeMutation.mutate()}
          disabled={analyzeMutation.isPending}
          data-testid="button-analyze-correlations"
        >
          {analyzeMutation.isPending ? (
            <>
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 mr-2" />
              Analyze Patterns
            </>
          )}
        </Button>
      </div>

      {criticalPatterns.length > 0 && (
        <Card className="border-destructive bg-destructive/5" data-testid="card-critical-alerts">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-destructive">
              <AlertCircle className="w-5 h-5" />
              Critical Patterns Detected
            </CardTitle>
            <CardDescription>
              These patterns require immediate attention
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {criticalPatterns.map((pattern) => (
              <div 
                key={pattern.id} 
                className="p-4 bg-background rounded-lg border border-destructive/20"
                data-testid={`pattern-critical-${pattern.id}`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      {getPatternTypeIcon(pattern.patternType)}
                      <h3 className="font-semibold">{pattern.patternName}</h3>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">{pattern.insight}</p>
                    <p className="text-sm font-medium text-primary">{pattern.recommendation}</p>
                  </div>
                  {getSeverityBadge(pattern.severity)}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card data-testid="card-total-patterns">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Patterns</CardTitle>
            <Network className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{patterns?.length || 0}</div>
            <p className="text-xs text-muted-foreground">
              Detected correlations
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-high-severity">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">High Priority</CardTitle>
            <AlertCircle className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
              {highSeverityPatterns.length}
            </div>
            <p className="text-xs text-muted-foreground">
              Require attention
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-average-confidence">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
            <Activity className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {patterns && patterns.length > 0
                ? Math.round(
                    (patterns.reduce((sum, p) => sum + parseFloat(p.confidence), 0) / patterns.length) * 100
                  )
                : 0}%
            </div>
            <p className="text-xs text-muted-foreground">
              Pattern reliability
            </p>
          </CardContent>
        </Card>
      </div>

      {report && !reportLoading && (
        <Card data-testid="card-ai-summary">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary" />
              AI-Generated Summary
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">{report.summary}</p>
            
            {report.recommendations.length > 0 && (
              <div>
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <ArrowRight className="w-4 h-4" />
                  Actionable Recommendations
                </h4>
                <ul className="space-y-2">
                  {report.recommendations.map((rec, idx) => (
                    <li 
                      key={idx} 
                      className="flex items-start gap-2"
                      data-testid={`recommendation-${idx}`}
                    >
                      <span className="text-primary">•</span>
                      <span className="text-sm">{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <Card data-testid="card-all-patterns">
        <CardHeader>
          <CardTitle>Detected Correlation Patterns</CardTitle>
          <CardDescription>
            Patterns identified across medications, biomarkers, environment, mood, and sleep
          </CardDescription>
        </CardHeader>
        <CardContent>
          {patternsLoading ? (
            <div className="text-center py-8 text-muted-foreground">
              <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
              Loading patterns...
            </div>
          ) : !patterns || patterns.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Network className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="mb-2">No correlation patterns detected yet</p>
              <p className="text-sm">Click "Analyze Patterns" to start discovering correlations in your health data</p>
            </div>
          ) : (
            <div className="space-y-4">
              {patterns.map((pattern) => (
                <div 
                  key={pattern.id}
                  className="p-4 border rounded-lg space-y-3 hover-elevate"
                  data-testid={`pattern-${pattern.id}`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-3 flex-1">
                      {getPatternTypeIcon(pattern.patternType)}
                      <div className="flex-1">
                        <h3 className="font-semibold">{pattern.patternName}</h3>
                        <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                          <span>Strength: <span className={`font-medium ${getCorrelationStrengthColor(parseFloat(pattern.correlationStrength))}`}>
                            {(parseFloat(pattern.correlationStrength) * 100).toFixed(0)}%
                          </span></span>
                          <span>Confidence: {(parseFloat(pattern.confidence) * 100).toFixed(0)}%</span>
                          <span>Window: {pattern.timeWindow}</span>
                        </div>
                      </div>
                    </div>
                    {getSeverityBadge(pattern.severity)}
                  </div>

                  <div className="space-y-2">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Insight:</p>
                      <p className="text-sm">{pattern.insight}</p>
                    </div>
                    
                    {pattern.recommendation && (
                      <div>
                        <p className="text-sm text-muted-foreground mb-1">Recommendation:</p>
                        <p className="text-sm font-medium text-primary">{pattern.recommendation}</p>
                      </div>
                    )}

                    {pattern.factors && pattern.factors.length > 0 && (
                      <div>
                        <p className="text-sm text-muted-foreground mb-2">Contributing Factors:</p>
                        <div className="flex flex-wrap gap-2">
                          {pattern.factors.map((factor, idx) => (
                            <Badge 
                              key={idx} 
                              variant="outline"
                              data-testid={`factor-${pattern.id}-${idx}`}
                            >
                              {factor.name}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
                      <span>First observed: {format(new Date(pattern.firstObserved), 'MMM d, yyyy')}</span>
                      <span>Sample size: {pattern.sampleSize} data points</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
