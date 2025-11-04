import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { 
  MapPin,
  Wind,
  AlertTriangle,
  CheckCircle2,
  Droplets,
  Sun,
  Sparkles,
  RefreshCw,
  Search,
  Thermometer,
  Activity,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { format } from "date-fns";

interface EnvironmentalRiskData {
  id: number;
  zipCode?: string;
  city?: string;
  state?: string;
  latitude: string;
  longitude: string;
  locationName?: string;
  measuredAt: Date;
  aqi?: number;
  aqiCategory?: string;
  pm25?: number;
  pm10?: number;
  ozone?: number;
  immunocompromisedRisk: string;
  overallRiskScore: number;
  pollutants?: any;
  localOutbreaks?: any[];
  detectedPathogens?: any[];
  pollenCount?: number;
  uvIndex?: number;
}

interface PathogenRiskMap {
  location: {
    zipCode?: string;
    city?: string;
    state?: string;
    latitude: string;
    longitude: string;
  };
  measuredAt: Date;
  overallRisk: {
    score: number;
    level: string;
    description: string;
  };
  airQuality: {
    aqi?: number;
    aqiCategory?: string;
    pm25?: number;
    pollutants?: any;
  };
  outbreaks: any[];
  pathogens: any[];
  pollenCount?: number;
  uvIndex?: number;
  recommendations: any[];
}

export default function EnvironmentalRiskMap() {
  const { toast } = useToast();
  const [zipCode, setZipCode] = useState("");

  const { data: currentRisk, isLoading: riskLoading } = useQuery<EnvironmentalRiskData>({
    queryKey: ["/api/environmental/risk"],
  });

  const { data: pathogenMap, isLoading: mapLoading } = useQuery<PathogenRiskMap>({
    queryKey: ["/api/environmental/pathogen-map"],
  });

  const updateRiskMutation = useMutation({
    mutationFn: async (zip: string) => {
      return await apiRequest("POST", "/api/environmental/update", { zipCode: zip });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/environmental/risk"] });
      queryClient.invalidateQueries({ queryKey: ["/api/environmental/pathogen-map"] });
      toast({
        title: "Environmental Data Updated",
        description: "Latest risk assessment has been retrieved.",
      });
      setZipCode("");
    },
    onError: () => {
      toast({
        title: "Update Failed",
        description: "Unable to fetch environmental data. Please check the zip code and try again.",
        variant: "destructive",
      });
    },
  });

  const handleUpdate = () => {
    if (!zipCode || zipCode.length !== 5) {
      toast({
        title: "Invalid Zip Code",
        description: "Please enter a valid 5-digit US zip code.",
        variant: "destructive",
      });
      return;
    }
    updateRiskMutation.mutate(zipCode);
  };

  const getAQICategoryColor = (category?: string) => {
    switch (category?.toLowerCase()) {
      case "good":
        return "text-green-600 dark:text-green-400";
      case "moderate":
        return "text-yellow-600 dark:text-yellow-400";
      case "unhealthy for sensitive groups":
        return "text-orange-600 dark:text-orange-400";
      case "unhealthy":
        return "text-red-600 dark:text-red-400";
      case "very unhealthy":
      case "hazardous":
        return "text-purple-600 dark:text-purple-400";
      default:
        return "text-muted-foreground";
    }
  };

  const getRiskLevelBadge = (level?: string) => {
    if (!level) return null;
    
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      low: "default",
      moderate: "secondary",
      high: "destructive",
      critical: "destructive",
    };

    return (
      <Badge variant={variants[level] || "outline"} data-testid={`badge-risk-${level}`}>
        {level.toUpperCase()}
      </Badge>
    );
  };

  const getUVIndexColor = (uv?: number) => {
    if (!uv) return "text-muted-foreground";
    if (uv <= 2) return "text-green-600";
    if (uv <= 5) return "text-yellow-600";
    if (uv <= 7) return "text-orange-600";
    if (uv <= 10) return "text-red-600";
    return "text-purple-600";
  };

  if (riskLoading || mapLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto mb-4" />
          <p className="text-muted-foreground">Loading environmental risk data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight" data-testid="heading-environmental-risk">
            Environmental Risk Map
          </h1>
          <p className="text-muted-foreground">
            Real-time pathogen tracking and air quality monitoring for immunocompromised safety
          </p>
        </div>
      </div>

      <Card data-testid="card-location-update">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MapPin className="h-5 w-5" />
            Update Location
          </CardTitle>
          <CardDescription>
            Enter your zip code to get the latest environmental risk assessment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            <div className="flex-1 max-w-xs">
              <Label htmlFor="zipCode" className="sr-only">Zip Code</Label>
              <Input
                id="zipCode"
                type="text"
                placeholder="Enter 5-digit zip code"
                value={zipCode}
                onChange={(e) => setZipCode(e.target.value.replace(/\D/g, "").slice(0, 5))}
                maxLength={5}
                data-testid="input-zipcode"
              />
            </div>
            <Button
              onClick={handleUpdate}
              disabled={updateRiskMutation.isPending}
              data-testid="button-update-location"
            >
              <Search className={`h-4 w-4 mr-2 ${updateRiskMutation.isPending ? "animate-spin" : ""}`} />
              Update
            </Button>
          </div>
        </CardContent>
      </Card>

      {pathogenMap && (
        <>
          <Card data-testid="card-overall-risk">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Overall Environmental Risk</CardTitle>
                {getRiskLevelBadge(pathogenMap.overallRisk.level)}
              </div>
              <CardDescription>
                {pathogenMap.location.city && pathogenMap.location.state 
                  ? `${pathogenMap.location.city}, ${pathogenMap.location.state}` 
                  : pathogenMap.location.zipCode || "Current Location"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex items-baseline gap-2 mb-2">
                  <span className="text-4xl font-bold" data-testid="text-risk-score">
                    {pathogenMap.overallRisk.score}
                  </span>
                  <span className="text-lg text-muted-foreground">/100</span>
                </div>
                <p className="text-sm text-muted-foreground" data-testid="text-risk-description">
                  {pathogenMap.overallRisk.description}
                </p>
              </div>
              <p className="text-xs text-muted-foreground" data-testid="text-measured-at">
                Last updated: {format(new Date(pathogenMap.measuredAt), "MMM d, yyyy 'at' h:mm a")}
              </p>
            </CardContent>
          </Card>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <Card data-testid="card-air-quality">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Air Quality Index</CardTitle>
                  <Wind className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                {pathogenMap.airQuality.aqi !== null && pathogenMap.airQuality.aqi !== undefined ? (
                  <>
                    <div className={`text-3xl font-bold ${getAQICategoryColor(pathogenMap.airQuality.aqiCategory)}`} data-testid="text-aqi">
                      {pathogenMap.airQuality.aqi}
                    </div>
                    <p className="text-xs text-muted-foreground capitalize">
                      {pathogenMap.airQuality.aqiCategory || "N/A"}
                    </p>
                    {pathogenMap.airQuality.pm25 !== null && pathogenMap.airQuality.pm25 !== undefined && (
                      <p className="text-xs text-muted-foreground mt-1">
                        PM2.5: {pathogenMap.airQuality.pm25} µg/m³
                      </p>
                    )}
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">No data</p>
                )}
              </CardContent>
            </Card>

            <Card data-testid="card-outbreaks">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Local Outbreaks</CardTitle>
                  <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold" data-testid="text-outbreak-count">
                  {pathogenMap.outbreaks.length}
                </div>
                <p className="text-xs text-muted-foreground">
                  {pathogenMap.outbreaks.length === 0 ? "No active outbreaks" : "Active in area"}
                </p>
              </CardContent>
            </Card>

            <Card data-testid="card-pollen">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Pollen Count</CardTitle>
                  <Sparkles className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                {pathogenMap.pollenCount !== null && pathogenMap.pollenCount !== undefined ? (
                  <>
                    <div className="text-3xl font-bold" data-testid="text-pollen-count">
                      {pathogenMap.pollenCount}
                    </div>
                    <p className="text-xs text-muted-foreground">grains/m³</p>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">No data</p>
                )}
              </CardContent>
            </Card>

            <Card data-testid="card-uv-index">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">UV Index</CardTitle>
                  <Sun className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                {pathogenMap.uvIndex !== null && pathogenMap.uvIndex !== undefined ? (
                  <>
                    <div className={`text-3xl font-bold ${getUVIndexColor(pathogenMap.uvIndex)}`} data-testid="text-uv-index">
                      {pathogenMap.uvIndex}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {pathogenMap.uvIndex <= 2 ? "Low" : 
                       pathogenMap.uvIndex <= 5 ? "Moderate" :
                       pathogenMap.uvIndex <= 7 ? "High" :
                       pathogenMap.uvIndex <= 10 ? "Very High" : "Extreme"}
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">No data</p>
                )}
              </CardContent>
            </Card>
          </div>

          {pathogenMap.outbreaks.length > 0 && (
            <Card data-testid="card-outbreak-details">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-orange-600" />
                  Active Outbreaks
                </CardTitle>
                <CardDescription>Disease outbreaks detected in your area</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {pathogenMap.outbreaks.map((outbreak: any, idx: number) => (
                    <div 
                      key={idx} 
                      className="p-4 border rounded-lg"
                      data-testid={`outbreak-${idx}`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h4 className="font-medium" data-testid={`text-outbreak-disease-${idx}`}>
                            {outbreak.disease || "Unknown Disease"}
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            {outbreak.location || "Local area"}
                          </p>
                        </div>
                        <Badge variant={outbreak.severity === "high" ? "destructive" : "secondary"}>
                          {outbreak.severity || "moderate"}
                        </Badge>
                      </div>
                      {outbreak.caseCount && (
                        <p className="text-xs text-muted-foreground">
                          Reported cases: {outbreak.caseCount}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {pathogenMap.pathogens.length > 0 && (
            <Card data-testid="card-pathogen-details">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Droplets className="h-5 w-5 text-blue-600" />
                  Detected Pathogens
                </CardTitle>
                <CardDescription>Pathogens found in wastewater surveillance</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {pathogenMap.pathogens.map((pathogen: any, idx: number) => (
                    <div 
                      key={idx} 
                      className="p-4 border rounded-lg"
                      data-testid={`pathogen-${idx}`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h4 className="font-medium" data-testid={`text-pathogen-name-${idx}`}>
                            {pathogen.pathogen || "Unknown Pathogen"}
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            {pathogen.source || "Wastewater"}
                          </p>
                        </div>
                        <Badge variant="outline">
                          {pathogen.concentration || "detected"}
                        </Badge>
                      </div>
                      {pathogen.trend && (
                        <div className="flex items-center gap-1 text-xs">
                          {pathogen.trend === "increasing" ? (
                            <TrendingUp className="h-3 w-3 text-red-600" />
                          ) : (
                            <TrendingDown className="h-3 w-3 text-green-600" />
                          )}
                          <span className="text-muted-foreground capitalize">{pathogen.trend}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {pathogenMap.recommendations && pathogenMap.recommendations.length > 0 && (
            <Card data-testid="card-recommendations">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  Safety Recommendations
                </CardTitle>
                <CardDescription>
                  Personalized guidance for immunocompromised individuals
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {pathogenMap.recommendations.map((rec: any, idx: number) => (
                    <div 
                      key={idx} 
                      className="p-4 bg-muted/50 rounded-lg"
                      data-testid={`recommendation-${idx}`}
                    >
                      <div className="flex items-start gap-3">
                        <div className="mt-0.5">
                          {rec.category === "urgent" ? (
                            <AlertTriangle className="h-5 w-5 text-red-600" />
                          ) : rec.category === "medical" ? (
                            <Activity className="h-5 w-5 text-blue-600" />
                          ) : (
                            <CheckCircle2 className="h-5 w-5 text-green-600" />
                          )}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-medium mb-1" data-testid={`text-recommendation-action-${idx}`}>
                            {rec.action}
                          </p>
                          {rec.urgency && (
                            <Badge variant="outline" className="text-xs">
                              {rec.urgency}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {!pathogenMap && !mapLoading && (
        <Card className="text-center py-12" data-testid="card-no-data">
          <CardContent>
            <MapPin className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No Environmental Data Available</h3>
            <p className="text-muted-foreground mb-4">
              Enter your zip code above to view environmental risk information for your area
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
