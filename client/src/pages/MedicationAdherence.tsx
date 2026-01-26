import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Pill,
  Calendar,
  CheckCircle2,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { Link } from "wouter";

interface AdherenceTrendPoint {
  date: string;
  adherenceRate: number;
}

interface RegimenRisk {
  level: "low" | "moderate" | "high" | "unknown";
  rationale: string;
}

interface MissedDoseEscalation {
  count: number;
  severity: "none" | "warning" | "critical";
}

interface MedicationAdherenceData {
  currentAdherenceRate: number | null;
  sevenDayTrend: AdherenceTrendPoint[];
  regimenRisk: RegimenRisk;
  missedDoseEscalation: MissedDoseEscalation;
}

export default function MedicationAdherence() {
  const { user } = useAuth();

  // Fetch medication adherence data from Python backend
  const { data: adherenceData, isLoading } = useQuery<MedicationAdherenceData>({
    queryKey: [`/python/api/v1/behavior-ai/medication-adherence/${user?.id}`],
    enabled: !!user?.id,
  });

  if (!user) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive" data-testid="alert-unauthorized">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Authentication Required</AlertTitle>
          <AlertDescription>
            Please <Link href="/login" className="underline">log in</Link> to view your medication adherence.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case "high":
        return "destructive";
      case "moderate":
        return "default";
      case "low":
        return "secondary";
      default:
        return "outline";
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "destructive";
      case "warning":
        return "default";
      default:
        return "secondary";
    }
  };

  const formatAdherenceRate = (rate: number | null) => {
    if (rate === null) return "No data";
    return `${(rate * 100).toFixed(1)}%`;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold" data-testid="heading-medication-adherence">
          Medication Adherence
        </h1>
        <p className="text-muted-foreground">
          Track your medication adherence, identify interaction risks, and monitor compliance trends.
        </p>
      </div>

      {/* Current Adherence Rate */}
      <Card data-testid="card-current-adherence">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Pill className="h-5 w-5" />
            Current Adherence Rate
          </CardTitle>
          <CardDescription>Your medication compliance over the last 7 days</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-24 w-full" />
          ) : (
            <div className="space-y-4">
              <div className="text-5xl font-bold" data-testid="text-adherence-rate">
                {formatAdherenceRate(adherenceData?.currentAdherenceRate ?? null)}
              </div>
              {adherenceData?.currentAdherenceRate !== null && (
                <Progress
                  value={(adherenceData.currentAdherenceRate ?? 0) * 100}
                  className="h-3"
                  data-testid="progress-adherence"
                />
              )}
              {(adherenceData?.currentAdherenceRate ?? 0) < 0.8 && (
                <Alert variant="destructive" data-testid="alert-low-adherence">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Low Adherence Detected</AlertTitle>
                  <AlertDescription>
                    Your adherence rate is below 80%. Missing medications can impact your health outcomes.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 7-Day Trend */}
      <Card data-testid="card-seven-day-trend">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            7-Day Adherence Trend
          </CardTitle>
          <CardDescription>Daily medication compliance for the past week</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-64 w-full" />
          ) : adherenceData?.sevenDayTrend.length ? (
            <div className="space-y-4">
              {adherenceData.sevenDayTrend.map((point, index) => (
                <div
                  key={point.date}
                  className="flex items-center justify-between"
                  data-testid={`trend-point-${index}`}
                >
                  <div className="flex items-center gap-3">
                    {point.adherenceRate >= 0.8 ? (
                      <CheckCircle2 className="h-5 w-5 text-green-600" />
                    ) : point.adherenceRate >= 0.5 ? (
                      <AlertCircle className="h-5 w-5 text-yellow-600" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-600" />
                    )}
                    <span className="text-sm font-medium">
                      {new Date(point.date).toLocaleDateString('en-US', {
                        weekday: 'short',
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Progress
                      value={point.adherenceRate * 100}
                      className="h-2 w-32"
                    />
                    <span className="text-sm font-medium w-12 text-right">
                      {formatAdherenceRate(point.adherenceRate)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-center py-8">
              No adherence data available for the past 7 days.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Regimen Risk Analysis */}
      <Card data-testid="card-regimen-risk">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            Regimen Risk Analysis
          </CardTitle>
          <CardDescription>Drug interaction and complexity assessment</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-32 w-full" />
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-lg font-semibold">Risk Level:</span>
                <Badge
                  variant={getRiskColor(adherenceData?.regimenRisk.level ?? "unknown")}
                  className="text-lg px-4 py-1"
                  data-testid="badge-risk-level"
                >
                  {adherenceData?.regimenRisk.level.toUpperCase() ?? "UNKNOWN"}
                </Badge>
              </div>
              {adherenceData?.regimenRisk.rationale && (
                <Alert data-testid="alert-risk-rationale">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Risk Factors</AlertTitle>
                  <AlertDescription>{adherenceData.regimenRisk.rationale}</AlertDescription>
                </Alert>
              )}
              {(adherenceData?.regimenRisk.rationale?.includes("drug interactions") ||
                adherenceData?.regimenRisk.rationale?.includes("interaction")) && (
                <div className="pt-4">
                  <Link href="/medications">
                    <a className="text-primary hover:underline flex items-center gap-2">
                      View detailed drug interaction analysis →
                    </a>
                  </Link>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Missed Dose Escalation */}
      <Card data-testid="card-missed-doses">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <XCircle className="h-5 w-5" />
            Missed Dose Escalation
          </CardTitle>
          <CardDescription>Tracking missed medications in the last 30 days</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-24 w-full" />
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-lg font-semibold">Missed Doses:</span>
                <div className="flex items-center gap-3">
                  <span className="text-4xl font-bold" data-testid="text-missed-count">
                    {adherenceData?.missedDoseEscalation.count ?? 0}
                  </span>
                  <Badge
                    variant={getSeverityColor(
                      adherenceData?.missedDoseEscalation.severity ?? "none"
                    )}
                    data-testid="badge-missed-severity"
                  >
                    {adherenceData?.missedDoseEscalation.severity.toUpperCase() ?? "NONE"}
                  </Badge>
                </div>
              </div>
              {(adherenceData?.missedDoseEscalation.count ?? 0) > 5 && (
                <Alert
                  variant={
                    (adherenceData?.missedDoseEscalation.count ?? 0) > 10
                      ? "destructive"
                      : "default"
                  }
                  data-testid="alert-missed-doses"
                >
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>
                    {(adherenceData?.missedDoseEscalation.count ?? 0) > 10
                      ? "Critical: Multiple Missed Doses"
                      : "Warning: Missed Doses Detected"}
                  </AlertTitle>
                  <AlertDescription>
                    You've missed {adherenceData?.missedDoseEscalation.count} doses in the last 30
                    days. Contact your healthcare provider if you're having trouble with your
                    medication schedule.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
