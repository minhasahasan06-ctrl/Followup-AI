import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { 
  MapPin,
  Wind,
  AlertTriangle,
  CheckCircle2,
  Droplets,
  Sparkles,
  RefreshCw,
  Thermometer,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Bell,
  BellOff,
  Clock,
  BarChart3,
  Gauge,
  CloudRain,
  Leaf,
  AlertCircle,
  Info,
  ChevronRight,
  Shield,
  Heart,
  Brain,
  Stethoscope,
  Bone,
} from "lucide-react";
import { format, formatDistanceToNow } from "date-fns";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface EnvironmentalProfile {
  id: string;
  patientId: string;
  zipCode: string;
  city: string;
  state: string;
  conditions: string[];
  allergies: string[];
  alertsEnabled: boolean;
  correlationConsent: boolean;
}

interface WeatherData {
  temperature: number | null;
  feelsLike: number | null;
  humidity: number | null;
  pressure: number | null;
  uvIndex: number | null;
}

interface AirQualityData {
  aqi: number | null;
  category: string | null;
  pm25: number | null;
  pm10: number | null;
  ozone: number | null;
}

interface AllergenData {
  pollenOverall: number | null;
  pollenCategory: string | null;
  moldCount: number | null;
}

interface RiskScore {
  composite: number | null;
  level: string | null;
  computedAt: string | null;
  volatility?: number;
  components: {
    weather: number | null;
    airQuality: number | null;
    allergens: number | null;
    hazards: number | null;
  };
  trends: {
    "24hr": number | null;
    "48hr": number | null;
    "72hr": number | null;
  };
  topFactors: Array<{
    factor: string;
    severity: string;
    recommendation: string;
  }>;
}

interface Forecast {
  horizon: string;
  targetTime: string;
  predictedScore: number;
  predictedLevel: string;
  confidence: {
    lower: number;
    upper: number;
    confidence: number;
  };
}

interface Alert {
  id: string;
  type: string;
  severity: string;
  title: string;
  message: string;
  recommendations: Array<{
    action: string;
    urgency: string;
    category: string;
  }>;
  createdAt: string;
}

interface Correlation {
  symptom: string;
  factor: string;
  correlation: number;
  strength: string;
  direction: string;
  lagHours: number;
  interpretation: string;
  confidence: number | null;
}

interface EnvironmentalData {
  success: boolean;
  profile: EnvironmentalProfile | null;
  currentData: {
    measuredAt: string;
    weather: WeatherData;
    airQuality: AirQualityData;
    allergens: AllergenData;
    hazards: any[];
  } | null;
  riskScore: RiskScore | null;
  forecasts: Forecast[];
  activeAlerts: Alert[];
}

const CONDITION_ICONS: Record<string, any> = {
  asthma: Stethoscope,
  copd: Stethoscope,
  heart_failure: Heart,
  arthritis: Bone,
  migraines: Brain,
  eczema: Shield,
};

const CONDITION_LABELS: Record<string, string> = {
  asthma: "Asthma",
  copd: "COPD",
  heart_failure: "Heart Failure",
  arthritis: "Arthritis",
  migraines: "Migraines",
  eczema: "Eczema",
};

interface EnvironmentalRiskSectionProps {
  patientId: string;
  showHeader?: boolean;
}

export function EnvironmentalRiskSection({ patientId, showHeader = true }: EnvironmentalRiskSectionProps) {
  const { toast } = useToast();
  const [zipCode, setZipCode] = useState("");
  const [selectedConditions, setSelectedConditions] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState("overview");

  const { data: envData, isLoading, refetch } = useQuery<EnvironmentalData>({
    queryKey: ["/api/v1/environment/current", patientId],
    queryFn: async () => {
      const res = await fetch(`/api/v1/environment/current?patient_id=${patientId}`, {
        credentials: 'include',
      });
      if (!res.ok) throw new Error("Failed to fetch environmental data");
      return res.json();
    },
    refetchInterval: 5 * 60 * 1000,
    enabled: !!patientId,
  });

  const { data: historyData } = useQuery({
    queryKey: ["/api/v1/environment/history", patientId],
    queryFn: async () => {
      const res = await fetch(`/api/v1/environment/history?patient_id=${patientId}&days=7`, {
        credentials: 'include',
      });
      if (!res.ok) return { history: [] };
      return res.json();
    },
    enabled: !!envData?.profile && !!patientId,
  });

  const { data: correlationData } = useQuery({
    queryKey: ["/api/v1/environment/correlations", patientId],
    queryFn: async () => {
      const res = await fetch(`/api/v1/environment/correlations?patient_id=${patientId}`, {
        credentials: 'include',
      });
      if (!res.ok) return { correlations: [] };
      return res.json();
    },
    enabled: !!envData?.profile?.correlationConsent && !!patientId,
  });

  const refreshMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/v1/environment/refresh?patient_id=${patientId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: 'include',
        body: JSON.stringify({ zipCode: zipCode || envData?.profile?.zipCode }),
      });
      if (!res.ok) throw new Error("Failed to refresh data");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/environment/current"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/environment/history"] });
      toast({
        title: "Data Refreshed",
        description: "Environmental data has been updated.",
      });
    },
    onError: () => {
      toast({
        title: "Refresh Failed",
        description: "Unable to refresh environmental data.",
        variant: "destructive",
      });
    },
  });

  const createProfileMutation = useMutation({
    mutationFn: async (data: { zipCode: string; conditions: string[] }) => {
      const res = await fetch(`/api/v1/environment/profile?patient_id=${patientId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: 'include',
        body: JSON.stringify({
          zipCode: data.zipCode,
          conditions: data.conditions,
        }),
      });
      if (!res.ok) throw new Error("Failed to create profile");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/environment/current"] });
      toast({
        title: "Profile Created",
        description: "Your environmental profile has been set up.",
      });
      setZipCode("");
    },
    onError: () => {
      toast({
        title: "Setup Failed",
        description: "Unable to create environmental profile.",
        variant: "destructive",
      });
    },
  });

  const acknowledgeAlertMutation = useMutation({
    mutationFn: async (alertId: string) => {
      const res = await fetch(`/api/v1/environment/alerts/acknowledge?patient_id=${patientId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: 'include',
        body: JSON.stringify({ alertId }),
      });
      if (!res.ok) throw new Error("Failed to acknowledge alert");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/environment/current"] });
      toast({ title: "Alert Acknowledged" });
    },
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

  const getAQIColor = (aqi: number | null) => {
    if (aqi === null) return "text-muted-foreground";
    if (aqi <= 50) return "text-green-600";
    if (aqi <= 100) return "text-yellow-600";
    if (aqi <= 150) return "text-orange-600";
    if (aqi <= 200) return "text-red-600";
    return "text-purple-600";
  };

  const getPollenColor = (level: string | null) => {
    switch (level?.toLowerCase()) {
      case "low": return "text-green-600";
      case "moderate": return "text-yellow-600";
      case "high": return "text-orange-600";
      case "very_high": return "text-red-600";
      default: return "text-muted-foreground";
    }
  };

  const getTrendIcon = (trend: number | null) => {
    if (trend === null) return <Minus className="h-4 w-4" />;
    if (trend > 0.1) return <TrendingUp className="h-4 w-4 text-red-500" />;
    if (trend < -0.1) return <TrendingDown className="h-4 w-4 text-green-500" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };

  const formatTemperature = (temp: number | null) => {
    if (temp === null) return "N/A";
    const fahrenheit = (temp * 9/5) + 32;
    return `${temp.toFixed(1)}°C (${fahrenheit.toFixed(0)}°F)`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto mb-4" />
          <p className="text-muted-foreground">Loading environmental risk data...</p>
        </div>
      </div>
    );
  }

  if (!envData?.profile) {
    return (
      <div className="space-y-6">
        {showHeader && (
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2" data-testid="heading-environmental-risk">
              <MapPin className="h-6 w-6 text-primary" />
              Environmental Risk Map
            </h2>
            <p className="text-muted-foreground mt-2">
              Get personalized environmental health insights based on your location and conditions
            </p>
          </div>
        )}

        <Card data-testid="card-setup-profile">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MapPin className="h-5 w-5" />
              Set Up Your Environmental Profile
            </CardTitle>
            <CardDescription>
              Enter your ZIP code and select your health conditions to receive personalized environmental risk assessments
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="zipCode">ZIP Code</Label>
              <Input
                id="zipCode"
                type="text"
                placeholder="Enter 5-digit ZIP code"
                value={zipCode}
                onChange={(e) => setZipCode(e.target.value.replace(/\D/g, "").slice(0, 5))}
                maxLength={5}
                className="max-w-xs"
                data-testid="input-zipcode"
              />
            </div>

            <div className="space-y-3">
              <Label>Health Conditions (select all that apply)</Label>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(CONDITION_LABELS).map(([code, label]) => {
                  const Icon = CONDITION_ICONS[code] || Heart;
                  const isSelected = selectedConditions.includes(code);
                  return (
                    <Button
                      key={code}
                      variant={isSelected ? "default" : "outline"}
                      className="justify-start h-auto py-3"
                      onClick={() => {
                        setSelectedConditions(prev =>
                          isSelected
                            ? prev.filter(c => c !== code)
                            : [...prev, code]
                        );
                      }}
                      data-testid={`button-condition-${code}`}
                    >
                      <Icon className="h-4 w-4 mr-2" />
                      {label}
                    </Button>
                  );
                })}
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button
              onClick={() => {
                if (zipCode.length !== 5) {
                  toast({
                    title: "Invalid ZIP Code",
                    description: "Please enter a valid 5-digit ZIP code.",
                    variant: "destructive",
                  });
                  return;
                }
                createProfileMutation.mutate({ zipCode, conditions: selectedConditions });
              }}
              disabled={createProfileMutation.isPending || zipCode.length !== 5}
              data-testid="button-create-profile"
            >
              {createProfileMutation.isPending ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <CheckCircle2 className="h-4 w-4 mr-2" />
              )}
              Create Profile
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  const { profile, currentData, riskScore, forecasts, activeAlerts } = envData;
  const history = historyData?.history || [];
  const correlations: Correlation[] = correlationData?.correlations || [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        {showHeader && (
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2" data-testid="heading-environmental-risk">
              <MapPin className="h-6 w-6 text-primary" />
              Environmental Risk Map
            </h2>
            <p className="text-muted-foreground flex items-center gap-2 mt-1">
              <MapPin className="h-4 w-4" />
              {profile.city && profile.state ? `${profile.city}, ${profile.state}` : profile.zipCode}
              {currentData?.measuredAt && (
                <span className="text-xs">
                  (Updated {formatDistanceToNow(new Date(currentData.measuredAt), { addSuffix: true })})
                </span>
              )}
            </p>
          </div>
        )}
        {!showHeader && (
          <div className="flex items-center gap-2">
            <MapPin className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              {profile.city && profile.state ? `${profile.city}, ${profile.state}` : profile.zipCode}
            </span>
            {currentData?.measuredAt && (
              <span className="text-xs text-muted-foreground">
                (Updated {formatDistanceToNow(new Date(currentData.measuredAt), { addSuffix: true })})
              </span>
            )}
          </div>
        )}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => refreshMutation.mutate()}
            disabled={refreshMutation.isPending}
            data-testid="button-refresh"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshMutation.isPending ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {activeAlerts.length > 0 && (
        <Card className="border-destructive" data-testid="card-active-alerts">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-destructive">
              <AlertTriangle className="h-5 w-5" />
              Active Alerts ({activeAlerts.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {activeAlerts.slice(0, 3).map((alert) => (
                <div
                  key={alert.id}
                  className="flex items-start justify-between gap-4 p-3 rounded-lg bg-destructive/10"
                  data-testid={`alert-${alert.id}`}
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="destructive">{alert.severity}</Badge>
                      <span className="font-medium">{alert.title}</span>
                    </div>
                    <p className="text-sm text-muted-foreground">{alert.message}</p>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => acknowledgeAlertMutation.mutate(alert.id)}
                    data-testid={`button-acknowledge-${alert.id}`}
                  >
                    Dismiss
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview" data-testid="tab-overview">
            <Gauge className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="forecast" data-testid="tab-forecast">
            <BarChart3 className="h-4 w-4 mr-2" />
            Forecast
          </TabsTrigger>
          <TabsTrigger value="correlations" data-testid="tab-correlations">
            <Activity className="h-4 w-4 mr-2" />
            Insights
          </TabsTrigger>
          <TabsTrigger value="settings" data-testid="tab-settings">
            <Bell className="h-4 w-4 mr-2" />
            Alerts
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <Card data-testid="card-risk-score">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Overall Environmental Risk</CardTitle>
                {riskScore?.level && (
                  <Badge variant={getRiskBadgeVariant(riskScore.level)} data-testid="badge-risk-level">
                    {riskScore.level.toUpperCase()}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="flex items-baseline gap-3">
                    <span className={`text-5xl font-bold ${getRiskColor(riskScore?.level)}`} data-testid="text-risk-score">
                      {riskScore?.composite?.toFixed(0) ?? "--"}
                    </span>
                    <span className="text-xl text-muted-foreground">/100</span>
                  </div>
                  <Progress
                    value={riskScore?.composite ?? 0}
                    className="h-3"
                  />
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-muted-foreground">24hr Trend:</span>
                      {getTrendIcon(riskScore?.trends?.["24hr"] ?? null)}
                    </div>
                    {riskScore?.volatility !== undefined && (
                      <div className="flex items-center gap-1">
                        <span className="text-muted-foreground">Volatility:</span>
                        <span>{(riskScore.volatility as any)?.toFixed?.(0) ?? "--"}%</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="font-medium text-sm text-muted-foreground">Risk Components</h4>
                  <div className="space-y-2">
                    {[
                      { label: "Weather", value: riskScore?.components?.weather, icon: CloudRain },
                      { label: "Air Quality", value: riskScore?.components?.airQuality, icon: Wind },
                      { label: "Allergens", value: riskScore?.components?.allergens, icon: Leaf },
                      { label: "Hazards", value: riskScore?.components?.hazards, icon: AlertTriangle },
                    ].map(({ label, value, icon: Icon }) => (
                      <div key={label} className="flex items-center gap-3">
                        <Icon className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm flex-1">{label}</span>
                        <Progress value={value ?? 0} className="w-24 h-2" />
                        <span className="text-sm w-8 text-right">{value?.toFixed(0) ?? "--"}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card data-testid="card-temperature">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Temperature</CardTitle>
                  <Thermometer className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold" data-testid="text-temperature">
                  {formatTemperature(currentData?.weather?.temperature ?? null)}
                </div>
                {currentData?.weather?.feelsLike !== null && (
                  <p className="text-xs text-muted-foreground">
                    Feels like {formatTemperature(currentData?.weather?.feelsLike ?? null)}
                  </p>
                )}
              </CardContent>
            </Card>

            <Card data-testid="card-aqi">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Air Quality</CardTitle>
                  <Wind className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${getAQIColor(currentData?.airQuality?.aqi ?? null)}`} data-testid="text-aqi">
                  {currentData?.airQuality?.aqi ?? "N/A"}
                </div>
                <p className="text-xs text-muted-foreground capitalize">
                  {currentData?.airQuality?.category?.replace(/_/g, " ") ?? "Unknown"}
                </p>
                {currentData?.airQuality?.pm25 !== null && (
                  <p className="text-xs text-muted-foreground mt-1">
                    PM2.5: {currentData?.airQuality?.pm25?.toFixed(1)} µg/m³
                  </p>
                )}
              </CardContent>
            </Card>

            <Card data-testid="card-pollen">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Pollen Level</CardTitle>
                  <Sparkles className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${getPollenColor(currentData?.allergens?.pollenCategory ?? null)}`} data-testid="text-pollen">
                  {currentData?.allergens?.pollenOverall ?? "N/A"}
                  <span className="text-sm text-muted-foreground font-normal">/12</span>
                </div>
                <p className="text-xs text-muted-foreground capitalize">
                  {currentData?.allergens?.pollenCategory?.replace(/_/g, " ") ?? "Unknown"}
                </p>
              </CardContent>
            </Card>

            <Card data-testid="card-humidity">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Humidity</CardTitle>
                  <Droplets className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold" data-testid="text-humidity">
                  {currentData?.weather?.humidity?.toFixed(0) ?? "N/A"}%
                </div>
                <p className="text-xs text-muted-foreground">
                  Pressure: {currentData?.weather?.pressure?.toFixed(0) ?? "N/A"} hPa
                </p>
              </CardContent>
            </Card>
          </div>

          {riskScore?.topFactors && riskScore.topFactors.length > 0 && (
            <Card data-testid="card-recommendations">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="h-5 w-5" />
                  Personalized Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {riskScore.topFactors.map((factor, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-3 rounded-lg bg-muted/50"
                      data-testid={`recommendation-${idx}`}
                    >
                      <Badge variant={getRiskBadgeVariant(factor.severity)}>
                        {factor.factor}
                      </Badge>
                      <p className="text-sm flex-1">{factor.recommendation}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {history.length > 0 && (
            <Card data-testid="card-history">
              <CardHeader>
                <CardTitle>7-Day Risk History</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="computedAt"
                        tickFormatter={(val) => format(new Date(val), "MMM d")}
                        className="text-xs"
                      />
                      <YAxis domain={[0, 100]} className="text-xs" />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-popover border rounded-lg p-3 shadow-lg">
                                <p className="font-medium">{format(new Date(data.computedAt), "MMM d, h:mm a")}</p>
                                <p className="text-sm">Risk Score: <span className="font-bold">{data.compositeScore?.toFixed(0)}</span></p>
                                <p className="text-xs text-muted-foreground capitalize">{data.riskLevel}</p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="compositeScore"
                        stroke="hsl(var(--primary))"
                        fill="hsl(var(--primary) / 0.2)"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="forecast" className="space-y-6">
          <Card data-testid="card-forecasts">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Risk Forecast
              </CardTitle>
              <CardDescription>
                Predicted environmental risk levels based on weather patterns and historical data
              </CardDescription>
            </CardHeader>
            <CardContent>
              {forecasts.length > 0 ? (
                <div className="grid gap-4 md:grid-cols-3">
                  {forecasts.map((forecast) => (
                    <Card key={forecast.horizon} className="bg-muted/30" data-testid={`forecast-${forecast.horizon}`}>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium">
                          {forecast.horizon} Forecast
                        </CardTitle>
                        <CardDescription className="text-xs">
                          {format(new Date(forecast.targetTime), "MMM d, h:mm a")}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-baseline gap-2">
                          <span className={`text-3xl font-bold ${getRiskColor(forecast.predictedLevel)}`}>
                            {forecast.predictedScore.toFixed(0)}
                          </span>
                          <Badge variant={getRiskBadgeVariant(forecast.predictedLevel)}>
                            {forecast.predictedLevel}
                          </Badge>
                        </div>
                        {forecast.confidence && (
                          <p className="text-xs text-muted-foreground mt-2">
                            95% CI: {forecast.confidence.lower.toFixed(0)} - {forecast.confidence.upper.toFixed(0)}
                          </p>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No forecasts available yet</p>
                  <p className="text-sm">Forecasts will be generated as more data is collected</p>
                </div>
              )}
            </CardContent>
            <CardFooter className="text-xs text-muted-foreground border-t pt-4">
              <AlertCircle className="h-3 w-3 mr-1" />
              Environmental Trend Forecast — Not Medical Advice. Consult your healthcare provider for medical decisions.
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="correlations" className="space-y-6">
          <Card data-testid="card-correlations">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Symptom-Environment Correlations
              </CardTitle>
              <CardDescription>
                Learned patterns between your symptoms and environmental factors
              </CardDescription>
            </CardHeader>
            <CardContent>
              {correlations.length > 0 ? (
                <div className="space-y-4">
                  {correlations.map((corr, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-4 rounded-lg border"
                      data-testid={`correlation-${idx}`}
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="outline">{corr.symptom}</Badge>
                          <ChevronRight className="h-4 w-4 text-muted-foreground" />
                          <Badge variant="secondary">{corr.factor}</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{corr.interpretation}</p>
                        {corr.lagHours > 0 && (
                          <p className="text-xs text-muted-foreground mt-1">
                            <Clock className="h-3 w-3 inline mr-1" />
                            Effect seen ~{corr.lagHours} hours after exposure
                          </p>
                        )}
                      </div>
                      <div className="text-right">
                        <div className={`text-lg font-bold ${
                          corr.strength === "strong" ? "text-red-600" :
                          corr.strength === "moderate" ? "text-orange-600" :
                          "text-muted-foreground"
                        }`}>
                          {(corr.correlation * 100).toFixed(0)}%
                        </div>
                        <p className="text-xs text-muted-foreground capitalize">{corr.strength}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No correlations found yet</p>
                  <p className="text-sm">Log symptoms and enable correlation consent to discover patterns</p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card data-testid="card-conditions">
            <CardHeader>
              <CardTitle>Your Tracked Conditions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {(profile.conditions || []).map((condition) => {
                  const Icon = CONDITION_ICONS[condition] || Heart;
                  return (
                    <Badge key={condition} variant="secondary" className="py-2 px-3">
                      <Icon className="h-4 w-4 mr-2" />
                      {CONDITION_LABELS[condition] || condition}
                    </Badge>
                  );
                })}
                {(!profile.conditions || profile.conditions.length === 0) && (
                  <p className="text-muted-foreground text-sm">No conditions selected</p>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-6">
          <Card data-testid="card-alert-settings">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5" />
                Alert Preferences
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="font-medium">Environmental Alerts</Label>
                  <p className="text-sm text-muted-foreground">
                    Receive notifications when risk levels change
                  </p>
                </div>
                <Switch checked={profile.alertsEnabled} data-testid="switch-alerts" />
              </div>

              <Separator />

              <div className="flex items-center justify-between">
                <div>
                  <Label className="font-medium">Correlation Analysis Consent</Label>
                  <p className="text-sm text-muted-foreground">
                    Allow AI to analyze patterns between your symptoms and environment
                  </p>
                </div>
                <Switch checked={profile.correlationConsent} data-testid="switch-correlation-consent" />
              </div>
            </CardContent>
          </Card>

          <Card data-testid="card-alert-history">
            <CardHeader>
              <CardTitle>Alert History</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-64">
                {activeAlerts.length > 0 ? (
                  <div className="space-y-3">
                    {activeAlerts.map((alert) => (
                      <div
                        key={alert.id}
                        className="p-3 rounded-lg border"
                        data-testid={`alert-history-${alert.id}`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-2">
                            <Badge variant={alert.severity === "critical" ? "destructive" : "secondary"}>
                              {alert.severity}
                            </Badge>
                            <span className="font-medium text-sm">{alert.title}</span>
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {formatDistanceToNow(new Date(alert.createdAt), { addSuffix: true })}
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground">{alert.message}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <BellOff className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No alerts</p>
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default EnvironmentalRiskSection;
