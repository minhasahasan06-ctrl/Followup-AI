import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertTriangle,
  AlertCircle,
  Info,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Pill,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface DrugInteractionAlertProps {
  drug1: string;
  drug2: string;
  severityLevel: 'severe' | 'moderate' | 'minor';
  interactionType: string;
  clinicalEffects: string;
  mechanismDescription: string;
  managementRecommendations: string;
  alternativeSuggestions?: string[];
  riskForImmunocompromised?: 'high' | 'medium' | 'low';
  requiresMonitoring?: boolean;
  monitoringParameters?: string[];
  evidenceLevel?: string;
  aiAnalysisConfidence?: number;
  onAcknowledge?: () => void;
  status?: 'active' | 'acknowledged' | 'overridden';
  testId?: string;
}

export function DrugInteractionAlert({
  drug1,
  drug2,
  severityLevel,
  interactionType,
  clinicalEffects,
  mechanismDescription,
  managementRecommendations,
  alternativeSuggestions = [],
  riskForImmunocompromised = 'medium',
  requiresMonitoring = false,
  monitoringParameters = [],
  evidenceLevel = 'proven',
  aiAnalysisConfidence = 95,
  onAcknowledge,
  status = 'active',
  testId,
}: DrugInteractionAlertProps) {
  const [expanded, setExpanded] = useState(false);

  const severityConfig = {
    severe: {
      icon: AlertTriangle,
      color: "text-destructive",
      bg: "bg-destructive/10",
      border: "border-destructive/20",
      label: "Severe Interaction",
      description: "Do not combine without doctor approval",
    },
    moderate: {
      icon: AlertCircle,
      color: "text-chart-3",
      bg: "bg-chart-3/10",
      border: "border-chart-3/20",
      label: "Moderate Interaction",
      description: "Use with caution, monitoring recommended",
    },
    minor: {
      icon: Info,
      color: "text-chart-1",
      bg: "bg-chart-1/10",
      border: "border-chart-1/20",
      label: "Minor Interaction",
      description: "Usually manageable, be aware",
    },
  };

  const riskConfig = {
    high: { label: "High Risk", color: "text-destructive" },
    medium: { label: "Medium Risk", color: "text-chart-3" },
    low: { label: "Low Risk", color: "text-chart-2" },
  };

  const SeverityIcon = severityConfig[severityLevel].icon;
  const config = severityConfig[severityLevel];

  return (
    <Card 
      className={cn(
        "border-2",
        config.border,
        status === 'acknowledged' && "opacity-60",
        status === 'overridden' && "opacity-40"
      )}
      data-testid={testId || `alert-${drug1}-${drug2}`.toLowerCase().replace(/\s+/g, "-")}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 flex-1">
            <div className={cn("rounded-md p-2", config.bg)}>
              <SeverityIcon className={cn("h-5 w-5", config.color)} />
            </div>
            <div className="flex-1 space-y-1">
              <div className="flex items-center gap-2 flex-wrap">
                <CardTitle className="text-base" data-testid={testId ? `${testId}-title` : undefined}>
                  {config.label}
                </CardTitle>
                <Badge 
                  variant="secondary" 
                  className="text-xs"
                  data-testid={testId ? `${testId}-evidence-badge` : undefined}
                >
                  {evidenceLevel === 'proven' ? 'Clinically Proven' : evidenceLevel}
                </Badge>
                {aiAnalysisConfidence && aiAnalysisConfidence >= 90 && (
                  <Badge 
                    variant="outline" 
                    className="text-xs"
                    data-testid={testId ? `${testId}-confidence-badge` : undefined}
                  >
                    {aiAnalysisConfidence}% AI Confidence
                  </Badge>
                )}
              </div>
              <CardDescription className="text-sm" data-testid={testId ? `${testId}-description` : undefined}>
                {config.description}
              </CardDescription>
            </div>
          </div>
          {status === 'acknowledged' && (
            <Badge variant="outline" className="gap-1" data-testid={testId ? `${testId}-status-badge` : undefined}>
              <CheckCircle2 className="h-3 w-3" />
              Acknowledged
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        <Alert className={cn(config.bg, "border-0")}>
          <Pill className="h-4 w-4" />
          <AlertTitle className="text-sm font-semibold" data-testid={testId ? `${testId}-drugs` : undefined}>
            Interaction between {drug1} and {drug2}
          </AlertTitle>
          <AlertDescription className="text-sm mt-1">
            <span className="font-medium">Type:</span> {interactionType}
            {riskForImmunocompromised && (
              <>
                {" • "}
                <span className={cn("font-medium", riskConfig[riskForImmunocompromised].color)}>
                  {riskConfig[riskForImmunocompromised].label} for Immunocompromised
                </span>
              </>
            )}
          </AlertDescription>
        </Alert>

        <div className="space-y-2">
          <div>
            <h4 className="text-sm font-semibold mb-1">Clinical Effects:</h4>
            <p className="text-sm text-muted-foreground" data-testid={testId ? `${testId}-effects` : undefined}>
              {clinicalEffects}
            </p>
          </div>

          <div>
            <h4 className="text-sm font-semibold mb-1">What You Should Do:</h4>
            <p className="text-sm text-muted-foreground" data-testid={testId ? `${testId}-recommendations` : undefined}>
              {managementRecommendations}
            </p>
          </div>

          {requiresMonitoring && monitoringParameters.length > 0 && (
            <div className="rounded-md bg-chart-2/10 p-3 border border-chart-2/20">
              <h4 className="text-sm font-semibold mb-1 flex items-center gap-1">
                <AlertCircle className="h-3 w-3" />
                Monitoring Required:
              </h4>
              <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                {monitoringParameters.map((param, index) => (
                  <li key={index}>{param}</li>
                ))}
              </ul>
            </div>
          )}

          {alternativeSuggestions.length > 0 && (
            <div className="rounded-md bg-primary/5 p-3 border border-primary/10">
              <h4 className="text-sm font-semibold mb-1">Alternative Medications:</h4>
              <p className="text-sm text-muted-foreground">
                {alternativeSuggestions.join(", ")}
              </p>
            </div>
          )}
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={() => setExpanded(!expanded)}
          className="w-full"
          data-testid={testId ? `${testId}-toggle-details` : "button-toggle-details"}
        >
          {expanded ? (
            <>
              <ChevronUp className="h-4 w-4 mr-1" />
              Hide Details
            </>
          ) : (
            <>
              <ChevronDown className="h-4 w-4 mr-1" />
              Show Details
            </>
          )}
        </Button>

        {expanded && (
          <div className="space-y-2 pt-2 border-t">
            <div>
              <h4 className="text-sm font-semibold mb-1">How This Interaction Works:</h4>
              <p className="text-sm text-muted-foreground" data-testid={testId ? `${testId}-mechanism` : undefined}>
                {mechanismDescription}
              </p>
            </div>

            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Evidence Level: {evidenceLevel}</span>
              {aiAnalysisConfidence && (
                <span>AI Confidence: {aiAnalysisConfidence}%</span>
              )}
            </div>
          </div>
        )}

        {onAcknowledge && status === 'active' && (
          <div className="flex gap-2 pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onAcknowledge}
              className="flex-1"
              data-testid={testId ? `${testId}-acknowledge` : "button-acknowledge"}
            >
              <CheckCircle2 className="h-4 w-4 mr-1" />
              I Understand
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
