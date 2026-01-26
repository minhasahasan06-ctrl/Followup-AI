import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import {
  MapPin,
  Wind,
  Thermometer,
  Droplets,
  Leaf,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Info,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface PatientEnvironmentRiskCardProps {
  patientId: string;
  compact?: boolean;
}

interface EnvironmentSummary {
  success: boolean;
  hasProfile: boolean;
  summary: {
    location: {
      zipCode: string;
      city: string;
      state: string;
    };
    conditions: string[];
    currentRisk: {
      score: number | null;
      level: string | null;
      trend: string;
      topFactors: Array<{
        factor: string;
        severity: string;
        recommendation: string;
      }>;
    };
    currentConditions: {
      aqi: number | null;
      aqiCategory: string | null;
      temperature: number | null;
      humidity: number | null;
      pollenLevel: string | null;
    };
    forecast: {
      "24hr": string | null;
      "48hr": string | null;
    };
    activeAlerts: number;
    alertsSummary: Array<{ severity: string; title: string }>;
    significantCorrelations: Array<{
      symptom: string;
      factor: string;
      strength: string;
    }>;
    weeklyTrend: {
      averageScore: number | null;
      highestScore: number | null;
      lowestScore: number | null;
      dataPoints: number;
    };
  } | null;
}

const CONDITION_LABELS: Record<string, string> = {
  asthma: "Asthma",
  copd: "COPD",
  heart_failure: "Heart Failure",
  arthritis: "Arthritis",
  migraines: "Migraines",
  eczema: "Eczema",
};

export function PatientEnvironmentRiskCard({ patientId, compact = false }: PatientEnvironmentRiskCardProps) {
  const { data, isLoading, error } = useQuery<EnvironmentSummary>({
    queryKey: ["/api/v1/environment/patient", patientId, "summary"],
    queryFn: async () => {
      const res = await fetch(`/api/v1/environment/patient/${patientId}/summary`);
      if (!res.ok) throw new Error("Failed to fetch environmental summary");
      return res.json();
    },
    refetchInterval: 5 * 60 * 1000,
  });

  const getRiskColor = (level: string | null) => {
    switch (level?.toLowerCase()) {
      case "low": return "text-green-600 dark:text-green-400";
      case "moderate": return "text-yellow-600 dark:text-yellow-400";
      case "high": return "text-orange-600 dark:text-orange-400";
      case "critical": return "text-red-600 dark:text-red-400";
      default: return "text-muted-foreground";
    }
  };

  const getRiskBadgeVariant = (level: string | null): "default" | "secondary" | "destructive" | "outline" => {
    switch (level?.toLowerCase()) {
      case "low": return "default";
      case "moderate": return "secondary";
      case "high": return "destructive";
      case "critical": return "destructive";
      default: return "outline";
    }
  };

  const getTrendIcon = (trend: string) => {
    if (trend === "worsening") return <TrendingUp className="h-4 w-4 text-red-500" />;
    if (trend === "improving") return <TrendingDown className="h-4 w-4 text-green-500" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };

  const getAQIColor = (aqi: number | null) => {
    if (aqi === null) return "text-muted-foreground";
    if (aqi <= 50) return "text-green-600";
    if (aqi <= 100) return "text-yellow-600";
    if (aqi <= 150) return "text-orange-600";
    return "text-red-600";
  };

  if (isLoading) {
    return (
      <Card data-testid="card-env-risk-loading">
        <CardHeader className="pb-2">
          <Skeleton className="h-5 w-40" />
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-8 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error || !data?.success) {
    return (
      <Card data-testid="card-env-risk-error">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <MapPin className="h-4 w-4" />
            Environmental Risk
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Unable to load environmental data</p>
        </CardContent>
      </Card>
    );
  }

  if (!data.hasProfile || !data.summary) {
    return (
      <Card data-testid="card-env-risk-no-profile">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <MapPin className="h-4 w-4" />
            Environmental Risk
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Patient has not set up an environmental profile
          </p>
        </CardContent>
      </Card>
    );
  }

  const { summary } = data;

  if (compact) {
    return (
      <Card data-testid="card-env-risk-compact">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm flex items-center gap-2">
              <MapPin className="h-4 w-4" />
              Environmental Risk
            </CardTitle>
            {summary.currentRisk.level && (
              <Badge variant={getRiskBadgeVariant(summary.currentRisk.level)} data-testid="badge-risk-compact">
                {summary.currentRisk.level.toUpperCase()}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className={`text-2xl font-bold ${getRiskColor(summary.currentRisk.level)}`}>
                {summary.currentRisk.score?.toFixed(0) ?? "--"}
              </span>
              <span className="text-muted-foreground">/100</span>
              {getTrendIcon(summary.currentRisk.trend)}
            </div>
            <div className="text-right text-sm">
              <p className="text-muted-foreground">
                {summary.location.city}, {summary.location.state}
              </p>
              {summary.activeAlerts > 0 && (
                <Badge variant="destructive" className="mt-1">
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  {summary.activeAlerts} Alert{summary.activeAlerts > 1 ? "s" : ""}
                </Badge>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="card-env-risk-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <MapPin className="h-5 w-5" />
            Environmental Risk Assessment
          </CardTitle>
          {summary.currentRisk.level && (
            <Badge variant={getRiskBadgeVariant(summary.currentRisk.level)} data-testid="badge-risk-full">
              {summary.currentRisk.level.toUpperCase()}
            </Badge>
          )}
        </div>
        <CardDescription>
          {summary.location.city}, {summary.location.state} ({summary.location.zipCode})
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center gap-6">
          <div>
            <div className="flex items-baseline gap-2">
              <span className={`text-4xl font-bold ${getRiskColor(summary.currentRisk.level)}`}>
                {summary.currentRisk.score?.toFixed(0) ?? "--"}
              </span>
              <span className="text-lg text-muted-foreground">/100</span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              {getTrendIcon(summary.currentRisk.trend)}
              <span className="text-sm text-muted-foreground capitalize">{summary.currentRisk.trend}</span>
            </div>
          </div>
          <Progress value={summary.currentRisk.score ?? 0} className="flex-1 h-3" />
        </div>

        {summary.activeAlerts > 0 && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="h-4 w-4 text-destructive" />
              <span className="font-medium text-destructive">
                {summary.activeAlerts} Active Alert{summary.activeAlerts > 1 ? "s" : ""}
              </span>
            </div>
            <div className="space-y-1">
              {summary.alertsSummary.map((alert, idx) => (
                <div key={idx} className="flex items-center gap-2 text-sm">
                  <Badge variant="destructive" className="text-xs">{alert.severity}</Badge>
                  <span>{alert.title}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Wind className="h-4 w-4" />
              Air Quality
            </div>
            <div className={`font-medium ${getAQIColor(summary.currentConditions.aqi)}`}>
              AQI: {summary.currentConditions.aqi ?? "N/A"}
              <span className="text-xs text-muted-foreground ml-1 capitalize">
                ({summary.currentConditions.aqiCategory?.replace(/_/g, " ") ?? "Unknown"})
              </span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Thermometer className="h-4 w-4" />
              Temperature
            </div>
            <div className="font-medium">
              {summary.currentConditions.temperature !== null
                ? `${summary.currentConditions.temperature.toFixed(1)}°C`
                : "N/A"}
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Droplets className="h-4 w-4" />
              Humidity
            </div>
            <div className="font-medium">
              {summary.currentConditions.humidity !== null
                ? `${summary.currentConditions.humidity.toFixed(0)}%`
                : "N/A"}
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Leaf className="h-4 w-4" />
              Pollen
            </div>
            <div className="font-medium capitalize">
              {summary.currentConditions.pollenLevel?.replace(/_/g, " ") ?? "N/A"}
            </div>
          </div>
        </div>

        {summary.conditions.length > 0 && (
          <div>
            <h4 className="text-sm font-medium mb-2">Tracked Conditions</h4>
            <div className="flex flex-wrap gap-2">
              {summary.conditions.map((condition) => (
                <Badge key={condition} variant="outline">
                  {CONDITION_LABELS[condition] || condition}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {summary.currentRisk.topFactors.length > 0 && (
          <div>
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Info className="h-4 w-4" />
              Top Risk Factors
            </h4>
            <div className="space-y-2">
              {summary.currentRisk.topFactors.slice(0, 3).map((factor, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <Badge variant={getRiskBadgeVariant(factor.severity)} className="shrink-0">
                    {factor.factor}
                  </Badge>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <p className="text-sm text-muted-foreground truncate cursor-help">
                        {factor.recommendation}
                      </p>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <p>{factor.recommendation}</p>
                    </TooltipContent>
                  </Tooltip>
                </div>
              ))}
            </div>
          </div>
        )}

        {summary.significantCorrelations.length > 0 && (
          <div>
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Detected Symptom Triggers
            </h4>
            <div className="space-y-1">
              {summary.significantCorrelations.map((corr, idx) => (
                <div key={idx} className="flex items-center gap-2 text-sm">
                  <Badge variant="outline">{corr.symptom}</Badge>
                  <span className="text-muted-foreground">correlates with</span>
                  <Badge variant="secondary">{corr.factor}</Badge>
                  <span className={`text-xs ${
                    corr.strength === "strong" ? "text-red-600" : "text-orange-600"
                  }`}>
                    ({corr.strength})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {(summary.forecast["24hr"] || summary.forecast["48hr"]) && (
          <div className="pt-4 border-t">
            <h4 className="text-sm font-medium mb-2">Forecast</h4>
            <div className="flex gap-4">
              {summary.forecast["24hr"] && (
                <div className="text-center">
                  <p className="text-xs text-muted-foreground">24hr</p>
                  <Badge variant={getRiskBadgeVariant(summary.forecast["24hr"])}>
                    {summary.forecast["24hr"]}
                  </Badge>
                </div>
              )}
              {summary.forecast["48hr"] && (
                <div className="text-center">
                  <p className="text-xs text-muted-foreground">48hr</p>
                  <Badge variant={getRiskBadgeVariant(summary.forecast["48hr"])}>
                    {summary.forecast["48hr"]}
                  </Badge>
                </div>
              )}
            </div>
          </div>
        )}

        {summary.weeklyTrend.dataPoints > 0 && (
          <div className="pt-4 border-t">
            <h4 className="text-sm font-medium mb-2">7-Day Statistics</h4>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-xs text-muted-foreground">Average</p>
                <p className="font-medium">{summary.weeklyTrend.averageScore?.toFixed(0) ?? "--"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Highest</p>
                <p className="font-medium text-red-600">{summary.weeklyTrend.highestScore?.toFixed(0) ?? "--"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Lowest</p>
                <p className="font-medium text-green-600">{summary.weeklyTrend.lowestScore?.toFixed(0) ?? "--"}</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default PatientEnvironmentRiskCard;
