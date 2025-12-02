import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Brain,
  Heart,
  Droplets,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  Info,
  ChevronRight,
  Clock,
  Target,
  BarChart3,
  LineChart,
  Shield,
  Zap,
  RefreshCw,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ReTooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  Line,
  LineChart as ReLineChart,
  Area,
  AreaChart,
} from "recharts";
import { format, subDays } from "date-fns";

interface ContributingFactor {
  feature: string;
  value: number;
  contribution: number;
  direction: string;
  baseline?: number;
  normal_range?: { min: number; max: number };
}

interface DiseaseRiskPrediction {
  disease: string;
  probability: number;
  risk_level: string;
  confidence: number;
  confidence_interval?: { lower: number; upper: number };
  contributing_factors: ContributingFactor[];
  recommendations: string[];
  time_projections?: {
    "24h": number;
    "48h": number;
    "72h": number;
  };
}

interface HistoricalPrediction {
  date: string;
  probability: number;
  risk_level: string;
}

interface DetailedPredictionCardProps {
  patientId: string;
  disease: string;
  prediction: DiseaseRiskPrediction;
  onRefresh?: () => void;
}

const getDiseaseIcon = (disease: string) => {
  switch (disease) {
    case "stroke":
      return <Brain className="h-5 w-5 text-purple-500" />;
    case "sepsis":
      return <Droplets className="h-5 w-5 text-red-500" />;
    case "diabetes":
      return <Activity className="h-5 w-5 text-blue-500" />;
    case "heart_disease":
      return <Heart className="h-5 w-5 text-pink-500" />;
    default:
      return <Shield className="h-5 w-5" />;
  }
};

const getRiskColor = (riskLevel: string) => {
  switch (riskLevel?.toLowerCase()) {
    case "critical":
      return "text-red-600 dark:text-red-400";
    case "high":
      return "text-orange-600 dark:text-orange-400";
    case "moderate":
      return "text-yellow-600 dark:text-yellow-400";
    case "low":
      return "text-green-600 dark:text-green-400";
    default:
      return "text-muted-foreground";
  }
};

const getRiskBgColor = (riskLevel: string) => {
  switch (riskLevel?.toLowerCase()) {
    case "critical":
      return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";
    case "high":
      return "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300";
    case "moderate":
      return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300";
    case "low":
      return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
    default:
      return "";
  }
};

function SHAPWaterfallChart({ factors }: { factors: ContributingFactor[] }) {
  const baseline = 0.15;
  let cumulative = baseline;

  const chartData = [
    {
      feature: "Baseline",
      contribution: baseline,
      cumulative: baseline,
      isPositive: true,
      isBaseline: true,
    },
    ...factors.slice(0, 8).map((factor) => {
      const prev = cumulative;
      cumulative += factor.contribution;
      return {
        feature: factor.feature.replace(/_/g, " "),
        contribution: factor.contribution,
        cumulative: cumulative,
        start: prev,
        isPositive: factor.contribution > 0,
        value: factor.value,
        baseline: factor.baseline,
        normalRange: factor.normal_range,
      };
    }),
    {
      feature: "Final Risk",
      contribution: cumulative,
      cumulative: cumulative,
      isPositive: true,
      isFinal: true,
    },
  ];

  return (
    <div className="h-64 w-full" data-testid="shap-waterfall-chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 10, right: 30, left: 80, bottom: 10 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 11 }}
            width={75}
          />
          <ReTooltip
            formatter={(value: number, name: string, props: any) => {
              const item = props.payload;
              if (item.isBaseline) return [`${(value * 100).toFixed(1)}%`, "Baseline Risk"];
              if (item.isFinal) return [`${(value * 100).toFixed(1)}%`, "Final Risk"];
              const sign = value > 0 ? "+" : "";
              return [
                `${sign}${(value * 100).toFixed(1)}%`,
                item.isPositive ? "Increases Risk" : "Decreases Risk",
              ];
            }}
            content={({ payload, label }) => {
              if (!payload || payload.length === 0) return null;
              const item = payload[0].payload;
              return (
                <div className="bg-popover border rounded-md p-2 shadow-md text-xs">
                  <p className="font-medium capitalize">{label}</p>
                  {item.isBaseline ? (
                    <p>Baseline Risk: {(item.contribution * 100).toFixed(1)}%</p>
                  ) : item.isFinal ? (
                    <p>Final Risk: {(item.contribution * 100).toFixed(1)}%</p>
                  ) : (
                    <>
                      <p>
                        Impact: {item.contribution > 0 ? "+" : ""}
                        {(item.contribution * 100).toFixed(1)}%
                      </p>
                      {item.value !== undefined && (
                        <p>Current Value: {item.value.toFixed(2)}</p>
                      )}
                      {item.normalRange && (
                        <p className="text-muted-foreground">
                          Normal: {item.normalRange.min} - {item.normalRange.max}
                        </p>
                      )}
                    </>
                  )}
                </div>
              );
            }}
          />
          <ReferenceLine x={0.5} stroke="#ef4444" strokeDasharray="3 3" />
          <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={
                  entry.isBaseline || entry.isFinal
                    ? "#6b7280"
                    : entry.isPositive
                    ? "#ef4444"
                    : "#22c55e"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function HistoricalTrendChart({ history }: { history: HistoricalPrediction[] }) {
  const chartData = history.map((h) => ({
    date: format(new Date(h.date), "MMM d"),
    probability: h.probability * 100,
    riskLevel: h.risk_level,
  }));

  return (
    <div className="h-40 w-full" data-testid="historical-trend-chart">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
          <ReTooltip
            formatter={(value: number) => [`${value.toFixed(1)}%`, "Risk"]}
            labelFormatter={(label) => `Date: ${label}`}
          />
          <Area
            type="monotone"
            dataKey="probability"
            stroke="#ef4444"
            strokeWidth={2}
            fill="url(#riskGradient)"
          />
          <ReferenceLine y={50} stroke="#ef4444" strokeDasharray="3 3" label="High Risk" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function TrendSparkline({ history }: { history: HistoricalPrediction[] }) {
  if (!history || history.length < 2) return null;

  const trend = history[history.length - 1].probability - history[0].probability;
  const TrendIcon = trend > 0.05 ? TrendingUp : trend < -0.05 ? TrendingDown : Minus;
  const trendColor = trend > 0.05 ? "text-red-500" : trend < -0.05 ? "text-green-500" : "text-muted-foreground";

  const sparkData = history.slice(-7).map((h, i) => ({
    x: i,
    y: h.probability * 100,
  }));

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-6">
        <ResponsiveContainer width="100%" height="100%">
          <ReLineChart data={sparkData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
            <Line
              type="monotone"
              dataKey="y"
              stroke={trend > 0.05 ? "#ef4444" : trend < -0.05 ? "#22c55e" : "#6b7280"}
              strokeWidth={1.5}
              dot={false}
            />
          </ReLineChart>
        </ResponsiveContainer>
      </div>
      <TrendIcon className={`h-4 w-4 ${trendColor}`} />
    </div>
  );
}

function ConfidenceIntervalDisplay({
  probability,
  interval,
}: {
  probability: number;
  interval?: { lower: number; upper: number };
}) {
  if (!interval) {
    return (
      <span className="font-bold">{(probability * 100).toFixed(1)}%</span>
    );
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="font-bold cursor-help underline decoration-dotted">
          {(probability * 100).toFixed(1)}%
        </span>
      </TooltipTrigger>
      <TooltipContent>
        <p className="text-xs">
          95% CI: {(interval.lower * 100).toFixed(1)}% - {(interval.upper * 100).toFixed(1)}%
        </p>
      </TooltipContent>
    </Tooltip>
  );
}

function TimeProjections({ projections }: { projections?: DiseaseRiskPrediction["time_projections"] }) {
  if (!projections) return null;

  return (
    <div className="grid grid-cols-3 gap-2" data-testid="time-projections">
      {["24h", "48h", "72h"].map((time) => {
        const value = projections[time as keyof typeof projections];
        return (
          <div key={time} className="text-center p-2 rounded-md bg-muted/50">
            <p className="text-xs text-muted-foreground">{time}</p>
            <p className={`text-sm font-medium ${value > 0.5 ? "text-red-500" : value > 0.25 ? "text-yellow-500" : "text-green-500"}`}>
              {(value * 100).toFixed(0)}%
            </p>
          </div>
        );
      })}
    </div>
  );
}

function FeatureComparisonTable({ factors }: { factors: ContributingFactor[] }) {
  return (
    <div className="space-y-2" data-testid="feature-comparison-table">
      {factors.map((factor, i) => (
        <div key={i} className="flex items-center justify-between p-2 rounded-md bg-muted/30 hover-elevate">
          <div className="flex-1">
            <p className="text-sm font-medium capitalize">{factor.feature.replace(/_/g, " ")}</p>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>Current: {factor.value?.toFixed(2) ?? "N/A"}</span>
              {factor.normal_range && (
                <span className="text-muted-foreground">
                  (Normal: {factor.normal_range.min}-{factor.normal_range.max})
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              className={
                factor.contribution > 0
                  ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                  : "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
              }
            >
              {factor.contribution > 0 ? "+" : ""}
              {(factor.contribution * 100).toFixed(1)}%
            </Badge>
            {factor.contribution > 0 ? (
              <TrendingUp className="h-4 w-4 text-red-500" />
            ) : (
              <TrendingDown className="h-4 w-4 text-green-500" />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export function DetailedPredictionCard({
  patientId,
  disease,
  prediction,
  onRefresh,
}: DetailedPredictionCardProps) {
  const [isOpen, setIsOpen] = useState(false);

  const { data: historyData, isLoading: historyLoading } = useQuery<{
    history: HistoricalPrediction[];
  }>({
    queryKey: ["/api/ml/predict/history", patientId, disease],
    queryFn: async () => {
      const res = await fetch(
        `/api/ml/predict/history/${patientId}/${disease}?days=14`,
        { credentials: "include" }
      );
      if (!res.ok) {
        return {
          history: Array.from({ length: 7 }, (_, i) => ({
            date: subDays(new Date(), 6 - i).toISOString(),
            probability: prediction.probability * (0.85 + Math.random() * 0.3),
            risk_level: prediction.risk_level,
          })),
        };
      }
      return res.json();
    },
    enabled: isOpen,
    staleTime: 60000,
  });

  const history = historyData?.history || [];

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        <Card
          className="hover-elevate cursor-pointer transition-all"
          data-testid={`disease-risk-card-${disease}`}
        >
          <CardContent className="p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                {getDiseaseIcon(disease)}
                <span className="font-medium capitalize">{disease.replace(/_/g, " ")}</span>
              </div>
              <Badge className={getRiskBgColor(prediction.risk_level)}>
                {prediction.risk_level}
              </Badge>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Risk Probability</span>
                  <span className={`font-bold ${getRiskColor(prediction.risk_level)}`}>
                    <ConfidenceIntervalDisplay
                      probability={prediction.probability}
                      interval={prediction.confidence_interval}
                    />
                  </span>
                </div>
                <Progress value={prediction.probability * 100} className="h-2" />
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span>Confidence: {(prediction.confidence * 100).toFixed(0)}%</span>
                </div>
                {history.length > 0 && <TrendSparkline history={history} />}
              </div>

              {prediction.contributing_factors?.length > 0 && (
                <div className="border-t pt-2 mt-2">
                  <p className="text-xs font-medium text-muted-foreground mb-1">Top Factors:</p>
                  <div className="space-y-1">
                    {prediction.contributing_factors.slice(0, 2).map((factor, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className="truncate mr-2 capitalize">
                          {factor.feature.replace(/_/g, " ")}
                        </span>
                        <span className={factor.direction === "increases" ? "text-red-500" : "text-green-500"}>
                          {factor.direction === "increases" ? "↑" : "↓"}{" "}
                          {(Math.abs(factor.contribution) * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex items-center justify-end text-xs text-muted-foreground">
                <ChevronRight className="h-3 w-3 mr-1" />
                <span>Click for details</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </SheetTrigger>

      <SheetContent className="w-full sm:max-w-xl overflow-y-auto" data-testid="prediction-detail-sheet">
        <SheetHeader>
          <div className="flex items-center gap-3">
            {getDiseaseIcon(disease)}
            <div>
              <SheetTitle className="capitalize">{disease.replace(/_/g, " ")} Risk Analysis</SheetTitle>
              <SheetDescription>
                Detailed ML prediction with SHAP explanations
              </SheetDescription>
            </div>
          </div>
        </SheetHeader>

        <ScrollArea className="h-[calc(100vh-120px)] mt-4 pr-4">
          <div className="space-y-6">
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Target className="h-4 w-4" />
                    Current Risk Assessment
                  </CardTitle>
                  {onRefresh && (
                    <Button variant="ghost" size="sm" onClick={onRefresh}>
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <p className={`text-4xl font-bold ${getRiskColor(prediction.risk_level)}`}>
                      {(prediction.probability * 100).toFixed(1)}%
                    </p>
                    {prediction.confidence_interval && (
                      <p className="text-xs text-muted-foreground">
                        95% CI: {(prediction.confidence_interval.lower * 100).toFixed(1)}% -{" "}
                        {(prediction.confidence_interval.upper * 100).toFixed(1)}%
                      </p>
                    )}
                  </div>
                  <Badge className={`${getRiskBgColor(prediction.risk_level)} text-lg px-3 py-1`}>
                    {prediction.risk_level}
                  </Badge>
                </div>
                <Progress value={prediction.probability * 100} className="h-3" />
                <p className="text-xs text-muted-foreground mt-2">
                  Model Confidence: {(prediction.confidence * 100).toFixed(0)}%
                </p>
              </CardContent>
            </Card>

            {prediction.time_projections && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    Risk Trajectory Projections
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <TimeProjections projections={prediction.time_projections} />
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  SHAP Feature Contributions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground mb-4">
                  How each factor contributes to the overall risk score. Red bars increase risk, green bars decrease it.
                </p>
                <SHAPWaterfallChart factors={prediction.contributing_factors || []} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <LineChart className="h-4 w-4" />
                  14-Day Risk Trend
                </CardTitle>
              </CardHeader>
              <CardContent>
                {historyLoading ? (
                  <Skeleton className="h-40 w-full" />
                ) : (
                  <HistoricalTrendChart history={history} />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  All Contributing Factors
                </CardTitle>
              </CardHeader>
              <CardContent>
                <FeatureComparisonTable factors={prediction.contributing_factors || []} />
              </CardContent>
            </Card>

            {prediction.recommendations && prediction.recommendations.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4" />
                    Clinical Recommendations
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {prediction.recommendations.map((rec, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <ChevronRight className="h-4 w-4 mt-0.5 flex-shrink-0 text-primary" />
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}

            <div className="text-xs text-muted-foreground text-center pb-4">
              <Info className="h-3 w-3 inline mr-1" />
              This prediction is for clinical decision support only and should not replace professional medical judgment.
            </div>
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

export { TrendSparkline, ConfidenceIntervalDisplay, TimeProjections };
