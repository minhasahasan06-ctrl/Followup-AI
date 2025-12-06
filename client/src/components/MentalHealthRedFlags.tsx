import { useQuery } from "@tanstack/react-query";
import { format } from "date-fns";
import { Calendar, Loader2, AlertTriangle, Info, MessageCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface MentalHealthRedFlag {
  id: string;
  timestamp: string;
  sessionId: string;
  messageId?: string;
  rawText: string;
  redFlagTypes: string[];
  severityLevel: 'low' | 'moderate' | 'high' | 'critical';
  specificConcerns: string[];
  emotionalTone: string;
  recommendedAction: string;
  crisisIndicators: boolean;
  confidence: number;
  severityScore: number;
  requiresImmediateAttention: boolean;
  clinicianNotified: boolean;
  dataSource: string;
  observationalLabel: string;
}

export function MentalHealthRedFlags() {
  const { data: redFlags, isLoading, error } = useQuery<MentalHealthRedFlag[]>({
    queryKey: ['/api/mental-health/red-flag-symptoms'],
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const getSeverityColor = (level: string) => {
    switch (level) {
      case 'critical':
        return 'bg-red-100 text-red-900 dark:bg-red-900 dark:text-red-100 border-red-300';
      case 'high':
        return 'bg-orange-100 text-orange-900 dark:bg-orange-900 dark:text-orange-100 border-orange-300';
      case 'moderate':
        return 'bg-yellow-100 text-yellow-900 dark:bg-yellow-900 dark:text-yellow-100 border-yellow-300';
      case 'low':
        return 'bg-blue-100 text-blue-900 dark:bg-blue-900 dark:text-blue-100 border-blue-300';
      default:
        return 'bg-muted';
    }
  };

  const formatRedFlagType = (type: string) => {
    return type
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-6">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">Loading mental health indicators...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="text-xs">
        <AlertTriangle className="h-3 w-3" />
        <AlertDescription>
          Failed to load mental health symptom indicators. Please try again later.
        </AlertDescription>
      </Alert>
    );
  }

  if (!redFlags || redFlags.length === 0) {
    return (
      <div className="text-center py-6 border rounded-md bg-muted/20">
        <MessageCircle className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
        <p className="text-sm text-muted-foreground">
          No mental health symptom indicators detected in the last 30 days.
        </p>
        <p className="text-xs text-muted-foreground mt-1">
          AI monitors your conversations with Agent Clona for potential concerns.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* HIPAA Compliance Disclaimer */}
      <Alert className="bg-orange-50/50 dark:bg-orange-950/20 border-orange-200 dark:border-orange-800" data-testid="alert-mh-disclaimer">
        <Info className="h-3 w-3 text-orange-600" />
        <AlertDescription className="text-xs">
          <strong>Observational Indicators Only:</strong> These are AI-observed symptom patterns from your chats with Agent Clona. They are NOT diagnoses and require professional clinical interpretation. If you're experiencing crisis symptoms, please contact emergency services or crisis hotline immediately.
        </AlertDescription>
      </Alert>

      {/* Red Flag Symptoms Timeline */}
      <div className="space-y-2">
        {redFlags.slice(0, 10).map((flag) => {
          const timestamp = flag.timestamp ? new Date(flag.timestamp) : new Date();
          const isValidDate = timestamp instanceof Date && !isNaN(timestamp.getTime());

          return (
            <div
              key={flag.id}
              className="rounded-md border p-3 space-y-2 bg-muted/20"
              data-testid={`red-flag-${flag.id}`}
            >
              {/* Header with timestamp and severity */}
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 flex-1">
                  <Calendar className="h-3 w-3 text-muted-foreground" />
                  <span className="text-xs font-medium">
                    {isValidDate ? format(timestamp, "MMM d, h:mm a") : "Invalid date"}
                  </span>
                </div>
                <div className="flex gap-1 flex-wrap justify-end">
                  {/* AI-Observed Badge - HIPAA Requirement */}
                  <Badge variant="secondary" className="text-xs" data-testid={`badge-ai-observed-${flag.id}`}>
                    {flag.observationalLabel}
                  </Badge>
                  {/* Severity Badge */}
                  <Badge className={`text-xs ${getSeverityColor(flag.severityLevel)}`}>
                    {flag.severityLevel.toUpperCase()}
                  </Badge>
                  {/* Confidence Score */}
                  {flag.confidence > 0 && (
                    <Badge variant="outline" className="text-xs">
                      {Math.round(flag.confidence * 100)}% confidence
                    </Badge>
                  )}
                  {/* Crisis Flag */}
                  {flag.crisisIndicators && (
                    <Badge variant="destructive" className="text-xs">
                      <AlertTriangle className="h-2 w-2 mr-1" />
                      Crisis Indicator
                    </Badge>
                  )}
                </div>
              </div>

              {/* Red Flag Types */}
              {flag.redFlagTypes && flag.redFlagTypes.length > 0 && (
                <div className="space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">Symptom Indicators:</div>
                  <div className="flex gap-1 flex-wrap">
                    {flag.redFlagTypes.map((type: string) => (
                      <Badge
                        key={type}
                        variant="outline"
                        className="text-xs bg-red-50 dark:bg-red-950/30 text-red-900 dark:text-red-100 border-red-200 dark:border-red-800"
                      >
                        {formatRedFlagType(type)}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Specific Concerns */}
              {flag.specificConcerns && flag.specificConcerns.length > 0 && (
                <div className="text-xs space-y-1">
                  <div className="font-medium text-muted-foreground">Specific Concerns:</div>
                  <ul className="list-disc list-inside space-y-0.5 text-muted-foreground">
                    {flag.specificConcerns.map((concern: string, idx: number) => (
                      <li key={idx} className="text-xs">"{concern}"</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Emotional Tone */}
              {flag.emotionalTone && (
                <div className="text-xs">
                  <span className="text-muted-foreground">Emotional Tone: </span>
                  <span className="font-medium">{flag.emotionalTone}</span>
                </div>
              )}

              {/* Recommended Action */}
              {flag.recommendedAction && (
                <div className="text-xs p-2 rounded bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800">
                  <span className="font-medium text-blue-900 dark:text-blue-100">Recommended: </span>
                  <span className="text-blue-800 dark:text-blue-200">{flag.recommendedAction}</span>
                </div>
              )}

              {/* Immediate Attention Flag */}
              {flag.requiresImmediateAttention && (
                <Alert variant="destructive" className="text-xs py-2">
                  <AlertTriangle className="h-3 w-3" />
                  <AlertDescription>
                    This indicator suggests immediate clinical attention may be warranted. Please contact a mental health professional or crisis hotline if you're experiencing these symptoms.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          );
        })}
      </div>

      {/* Crisis Resources */}
      {redFlags.some(f => f.crisisIndicators || f.requiresImmediateAttention) && (
        <Alert className="bg-red-50 dark:bg-red-950/20 border-red-300 dark:border-red-700">
          <AlertTriangle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-xs space-y-2">
            <p className="font-semibold text-red-900 dark:text-red-100">
              If you're in crisis or experiencing thoughts of self-harm:
            </p>
            <ul className="list-disc list-inside space-y-1 text-red-800 dark:text-red-200">
              <li><strong>988 Suicide & Crisis Lifeline:</strong> Call or text 988</li>
              <li><strong>Crisis Text Line:</strong> Text "HELLO" to 741741</li>
              <li><strong>Emergency:</strong> Call 911 or go to nearest emergency room</li>
            </ul>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
