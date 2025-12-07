import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Progress } from "@/components/ui/progress";
import { 
  Pill, 
  Activity, 
  Syringe, 
  Search, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle,
  Shield,
  MapPin,
  RefreshCw,
  Loader2,
  ChevronRight,
  Info,
  BarChart3,
  LineChart,
  Users
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { LineChart as ReLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, ResponsiveContainer, BarChart, Bar, Legend } from "recharts";

const PYTHON_BACKEND_URL = import.meta.env.VITE_PYTHON_BACKEND_URL || "http://localhost:8000";

interface DrugSignal {
  id: number;
  drug_code: string;
  drug_name: string;
  outcome_code: string;
  outcome_name: string;
  patient_location_id?: string;
  estimate: number;
  ci_lower: number;
  ci_upper: number;
  p_value: number;
  signal_strength: number;
  n_patients: number;
  n_events: number;
  flagged: boolean;
  suppressed?: boolean;
}

interface SignalResponse {
  signals: DrugSignal[];
  total_count: number;
  suppressed_count: number;
  privacy_note: string;
}

interface EpicurvePoint {
  date: string;
  case_count: number;
  cumulative_cases: number;
  death_count: number;
}

interface EpicurveResponse {
  pathogen_code: string;
  pathogen_name: string;
  data: EpicurvePoint[];
  total_cases: number;
  total_deaths: number;
}

interface R0Response {
  pathogen_code: string;
  r_value: number;
  r_lower: number;
  r_upper: number;
  interpretation: string;
  calculation_date: string;
}

interface VaccineCoverage {
  vaccine_code: string;
  vaccine_name: string;
  coverage_rate: number;
  vaccinated_count: number;
  total_population: number;
  by_dose: Record<number, number>;
}

interface VaccineEffectiveness {
  vaccine_code: string;
  vaccine_name: string;
  outcome_code: string;
  effectiveness: number;
  ci_lower: number;
  ci_upper: number;
  n_vaccinated: number;
  n_unvaccinated: number;
}

interface Location {
  id: string;
  name: string;
  city?: string;
  state?: string;
  signal_count?: number;
}

function DrugSafetyPanel() {
  const { toast } = useToast();
  const [drugQuery, setDrugQuery] = useState("");
  const [outcomeQuery, setOutcomeQuery] = useState("");
  const [selectedLocation, setSelectedLocation] = useState<string>("");
  const [scope, setScope] = useState<string>("all");

  const { data: signalData, isLoading: loadingSignals, refetch } = useQuery<SignalResponse>({
    queryKey: [PYTHON_BACKEND_URL, '/api/v1/pharmaco/signals', drugQuery, outcomeQuery, selectedLocation, scope],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (drugQuery) params.append('drug_query', drugQuery);
      if (outcomeQuery) params.append('outcome_query', outcomeQuery);
      if (selectedLocation) params.append('patient_location_id', selectedLocation);
      params.append('scope', scope);
      params.append('limit', '50');
      
      const response = await fetch(`${PYTHON_BACKEND_URL}/api/v1/pharmaco/signals?${params}`, {
        headers: { 'Authorization': 'Bearer token' }
      });
      if (!response.ok) throw new Error('Failed to fetch signals');
      return response.json();
    },
    enabled: true
  });

  const { data: locations } = useQuery<{ locations: Location[] }>({
    queryKey: [PYTHON_BACKEND_URL, '/api/v1/pharmaco/locations'],
    queryFn: async () => {
      const response = await fetch(`${PYTHON_BACKEND_URL}/api/v1/pharmaco/locations`, {
        headers: { 'Authorization': 'Bearer token' }
      });
      if (!response.ok) throw new Error('Failed to fetch locations');
      return response.json();
    }
  });

  const triggerScanMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(`${PYTHON_BACKEND_URL}/api/v1/pharmaco/run-scan`, {
        method: 'POST',
        headers: { 
          'Authorization': 'Bearer token',
          'Content-Type': 'application/json'
        }
      });
      if (!response.ok) throw new Error('Failed to trigger scan');
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Scan Started", description: "Drug safety scan has been triggered" });
      refetch();
    }
  });

  const getSignalColor = (signal: DrugSignal) => {
    if (signal.suppressed) return 'secondary';
    if (signal.flagged) return 'destructive';
    if (signal.estimate > 2) return 'default';
    return 'outline';
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-end">
        <div className="flex-1">
          <label className="text-sm font-medium mb-1 block">Drug Name</label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input 
              placeholder="Search drugs..." 
              value={drugQuery}
              onChange={(e) => setDrugQuery(e.target.value)}
              className="pl-9"
              data-testid="input-drug-search"
            />
          </div>
        </div>
        <div className="flex-1">
          <label className="text-sm font-medium mb-1 block">Outcome/Adverse Event</label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input 
              placeholder="Search outcomes..." 
              value={outcomeQuery}
              onChange={(e) => setOutcomeQuery(e.target.value)}
              className="pl-9"
              data-testid="input-outcome-search"
            />
          </div>
        </div>
        <div className="w-48">
          <label className="text-sm font-medium mb-1 block">Location</label>
          <Select value={selectedLocation} onValueChange={setSelectedLocation}>
            <SelectTrigger data-testid="select-location">
              <SelectValue placeholder="All locations" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">All locations</SelectItem>
              {locations?.locations?.map((loc) => (
                <SelectItem key={loc.id} value={loc.id}>
                  {loc.name} ({loc.signal_count || 0})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="w-40">
          <label className="text-sm font-medium mb-1 block">Scope</label>
          <Select value={scope} onValueChange={setScope}>
            <SelectTrigger data-testid="select-scope">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Patients</SelectItem>
              <SelectItem value="my_patients">My Patients</SelectItem>
              <SelectItem value="research_cohort">Research Cohort</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Button 
          variant="outline" 
          onClick={() => refetch()}
          data-testid="button-refresh-signals"
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {signalData?.privacy_note && (
        <Alert>
          <Shield className="h-4 w-4" />
          <AlertTitle>Privacy Protection</AlertTitle>
          <AlertDescription>{signalData.privacy_note}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2">
          <div>
            <CardTitle className="text-lg">Drug Safety Signals</CardTitle>
            <CardDescription>
              {signalData?.total_count || 0} signals found
              {signalData?.suppressed_count ? ` (${signalData.suppressed_count} suppressed for privacy)` : ''}
            </CardDescription>
          </div>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => triggerScanMutation.mutate()}
            disabled={triggerScanMutation.isPending}
            data-testid="button-run-scan"
          >
            {triggerScanMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Activity className="h-4 w-4 mr-2" />
            )}
            Run Scan
          </Button>
        </CardHeader>
        <CardContent>
          {loadingSignals ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <ScrollArea className="h-[400px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Drug</TableHead>
                    <TableHead>Outcome</TableHead>
                    <TableHead>Effect (OR)</TableHead>
                    <TableHead>95% CI</TableHead>
                    <TableHead>P-value</TableHead>
                    <TableHead>Signal</TableHead>
                    <TableHead>N</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {signalData?.signals?.map((signal) => (
                    <TableRow key={signal.id} className={signal.flagged ? 'bg-destructive/5' : ''}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Pill className="h-4 w-4 text-primary" />
                          <div>
                            <div className="font-medium">{signal.drug_name}</div>
                            <div className="text-xs text-muted-foreground">{signal.drug_code}</div>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div>
                          <div className="font-medium">{signal.outcome_name}</div>
                          <div className="text-xs text-muted-foreground">{signal.outcome_code}</div>
                        </div>
                      </TableCell>
                      <TableCell>
                        {signal.suppressed ? (
                          <span className="text-muted-foreground">Suppressed</span>
                        ) : (
                          <div className="flex items-center gap-1">
                            {signal.estimate > 1 ? (
                              <TrendingUp className="h-4 w-4 text-destructive" />
                            ) : (
                              <TrendingDown className="h-4 w-4 text-green-500" />
                            )}
                            <span className="font-mono">{signal.estimate?.toFixed(2)}</span>
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        {!signal.suppressed && (
                          <span className="font-mono text-sm">
                            [{signal.ci_lower?.toFixed(2)}, {signal.ci_upper?.toFixed(2)}]
                          </span>
                        )}
                      </TableCell>
                      <TableCell>
                        {!signal.suppressed && (
                          <span className={`font-mono text-sm ${signal.p_value < 0.05 ? 'text-destructive font-medium' : ''}`}>
                            {signal.p_value < 0.001 ? '<0.001' : signal.p_value?.toFixed(3)}
                          </span>
                        )}
                      </TableCell>
                      <TableCell>
                        {!signal.suppressed && (
                          <div className="w-24">
                            <Progress value={signal.signal_strength} className="h-2" />
                            <span className="text-xs text-muted-foreground">{signal.signal_strength?.toFixed(0)}</span>
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger>
                              <span className="font-mono">{signal.n_patients}</span>
                            </TooltipTrigger>
                            <TooltipContent>
                              {signal.n_events} events in {signal.n_patients} patients
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </TableCell>
                      <TableCell>
                        <Badge variant={getSignalColor(signal)}>
                          {signal.suppressed ? 'Suppressed' : signal.flagged ? 'Flagged' : 'Active'}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                  {(!signalData?.signals || signalData.signals.length === 0) && (
                    <TableRow>
                      <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">
                        No signals found. Try adjusting your search criteria.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function InfectiousDiseasePanel() {
  const [pathogenCode, setPathogenCode] = useState("COVID-19");
  const [selectedLocation, setSelectedLocation] = useState("");

  const { data: epicurveData, isLoading: loadingEpicurve } = useQuery<EpicurveResponse>({
    queryKey: [PYTHON_BACKEND_URL, '/api/v1/infectious/epicurve', pathogenCode, selectedLocation],
    queryFn: async () => {
      const params = new URLSearchParams({ pathogen_code: pathogenCode });
      if (selectedLocation) params.append('location_id', selectedLocation);
      
      const response = await fetch(`${PYTHON_BACKEND_URL}/api/v1/infectious/epicurve?${params}`, {
        headers: { 'Authorization': 'Bearer token' }
      });
      if (!response.ok) throw new Error('Failed to fetch epicurve');
      return response.json();
    },
    enabled: !!pathogenCode
  });

  const { data: r0Data, isLoading: loadingR0 } = useQuery<R0Response>({
    queryKey: [PYTHON_BACKEND_URL, '/api/v1/infectious/r0', pathogenCode, selectedLocation],
    queryFn: async () => {
      const params = new URLSearchParams({ pathogen_code: pathogenCode });
      if (selectedLocation) params.append('location_id', selectedLocation);
      
      const response = await fetch(`${PYTHON_BACKEND_URL}/api/v1/infectious/r0?${params}`, {
        headers: { 'Authorization': 'Bearer token' }
      });
      if (!response.ok) throw new Error('Failed to fetch R0');
      return response.json();
    },
    enabled: !!pathogenCode
  });

  const { data: pathogens } = useQuery<{ pathogens: Array<{ pathogen_code: string; pathogen_name: string; case_count: number }> }>({
    queryKey: [PYTHON_BACKEND_URL, '/api/v1/infectious/pathogens'],
    queryFn: async () => {
      const response = await fetch(`${PYTHON_BACKEND_URL}/api/v1/infectious/pathogens`, {
        headers: { 'Authorization': 'Bearer token' }
      });
      if (!response.ok) throw new Error('Failed to fetch pathogens');
      return response.json();
    }
  });

  const getR0Color = (r: number) => {
    if (r < 1) return 'text-green-500';
    if (r < 1.5) return 'text-yellow-500';
    return 'text-destructive';
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-end">
        <div className="w-64">
          <label className="text-sm font-medium mb-1 block">Pathogen</label>
          <Select value={pathogenCode} onValueChange={setPathogenCode}>
            <SelectTrigger data-testid="select-pathogen">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {pathogens?.pathogens?.map((p) => (
                <SelectItem key={p.pathogen_code} value={p.pathogen_code}>
                  {p.pathogen_name} ({p.case_count} cases)
                </SelectItem>
              )) || (
                <>
                  <SelectItem value="COVID-19">COVID-19</SelectItem>
                  <SelectItem value="INFLUENZA-A">Influenza A</SelectItem>
                  <SelectItem value="RSV">RSV</SelectItem>
                </>
              )}
            </SelectContent>
          </Select>
        </div>
        <div className="w-48">
          <label className="text-sm font-medium mb-1 block">Location</label>
          <Select value={selectedLocation} onValueChange={setSelectedLocation}>
            <SelectTrigger data-testid="select-infectious-location">
              <SelectValue placeholder="All locations" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">All locations</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Cases</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold" data-testid="text-total-cases">
              {epicurveData?.total_cases?.toLocaleString() || 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Deaths</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-destructive" data-testid="text-total-deaths">
              {epicurveData?.total_deaths?.toLocaleString() || 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">R₀ / Rt</CardTitle>
          </CardHeader>
          <CardContent>
            <p className={`text-3xl font-bold ${getR0Color(r0Data?.r_value || 1)}`} data-testid="text-r0">
              {r0Data?.r_value?.toFixed(2) || '-'}
            </p>
            <p className="text-xs text-muted-foreground">
              CI: [{r0Data?.r_lower?.toFixed(2)}, {r0Data?.r_upper?.toFixed(2)}]
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Status</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant={r0Data?.r_value && r0Data.r_value < 1 ? 'secondary' : 'destructive'}>
              {r0Data?.interpretation || 'Loading...'}
            </Badge>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <LineChart className="h-5 w-5" />
            Epidemic Curve
          </CardTitle>
          <CardDescription>
            {epicurveData?.pathogen_name} case counts over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loadingEpicurve ? (
            <div className="flex items-center justify-center h-80">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : epicurveData?.data && epicurveData.data.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <ReLineChart data={epicurveData.data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tickFormatter={(d) => new Date(d).toLocaleDateString()} />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <ReTooltip />
                <Legend />
                <Line 
                  yAxisId="left" 
                  type="monotone" 
                  dataKey="case_count" 
                  stroke="#8884d8" 
                  name="Daily Cases" 
                />
                <Line 
                  yAxisId="right" 
                  type="monotone" 
                  dataKey="cumulative_cases" 
                  stroke="#82ca9d" 
                  name="Cumulative Cases" 
                />
              </ReLineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-80 text-muted-foreground">
              No epidemic data available for selected criteria
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function VaccineAnalyticsPanel() {
  const [vaccineCode, setVaccineCode] = useState("COVID-19-MRNA");
  const [selectedLocation, setSelectedLocation] = useState("");
  const [outcomeCode, setOutcomeCode] = useState("COVID-19");

  const { data: coverageData, isLoading: loadingCoverage } = useQuery<VaccineCoverage>({
    queryKey: [PYTHON_BACKEND_URL, '/api/v1/vaccine/coverage', vaccineCode, selectedLocation],
    queryFn: async () => {
      const params = new URLSearchParams({ vaccine_code: vaccineCode });
      if (selectedLocation) params.append('location_id', selectedLocation);
      
      const response = await fetch(`${PYTHON_BACKEND_URL}/api/v1/vaccine/coverage?${params}`, {
        headers: { 'Authorization': 'Bearer token' }
      });
      if (!response.ok) throw new Error('Failed to fetch coverage');
      return response.json();
    },
    enabled: !!vaccineCode
  });

  const { data: effectivenessData, isLoading: loadingEffectiveness } = useQuery<VaccineEffectiveness>({
    queryKey: [PYTHON_BACKEND_URL, '/api/v1/vaccine/effectiveness', vaccineCode, outcomeCode, selectedLocation],
    queryFn: async () => {
      const params = new URLSearchParams({ 
        vaccine_code: vaccineCode,
        outcome_code: outcomeCode
      });
      if (selectedLocation) params.append('location_id', selectedLocation);
      
      const response = await fetch(`${PYTHON_BACKEND_URL}/api/v1/vaccine/effectiveness?${params}`, {
        headers: { 'Authorization': 'Bearer token' }
      });
      if (!response.ok) throw new Error('Failed to fetch effectiveness');
      return response.json();
    },
    enabled: !!vaccineCode && !!outcomeCode
  });

  const { data: vaccines } = useQuery<Array<{ vaccine_code: string; vaccine_name: string; total_doses: number }>>({
    queryKey: [PYTHON_BACKEND_URL, '/api/v1/vaccine/vaccines'],
    queryFn: async () => {
      const response = await fetch(`${PYTHON_BACKEND_URL}/api/v1/vaccine/vaccines`, {
        headers: { 'Authorization': 'Bearer token' }
      });
      if (!response.ok) throw new Error('Failed to fetch vaccines');
      return response.json();
    }
  });

  const getEffectivenessColor = (ve: number) => {
    if (ve >= 80) return 'text-green-500';
    if (ve >= 50) return 'text-yellow-500';
    return 'text-destructive';
  };

  const doseData = coverageData?.by_dose 
    ? Object.entries(coverageData.by_dose).map(([dose, count]) => ({
        dose: `Dose ${dose}`,
        count: count
      }))
    : [];

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-end">
        <div className="w-64">
          <label className="text-sm font-medium mb-1 block">Vaccine</label>
          <Select value={vaccineCode} onValueChange={setVaccineCode}>
            <SelectTrigger data-testid="select-vaccine">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {vaccines && Array.isArray(vaccines) ? vaccines.map((v) => (
                <SelectItem key={v.vaccine_code} value={v.vaccine_code}>
                  {v.vaccine_name} ({v.total_doses?.toLocaleString()} doses)
                </SelectItem>
              )) : (
                <>
                  <SelectItem value="COVID-19-MRNA">COVID-19 mRNA</SelectItem>
                  <SelectItem value="INFLUENZA-2024">Influenza 2024</SelectItem>
                  <SelectItem value="RSV-ADULT">RSV Adult</SelectItem>
                </>
              )}
            </SelectContent>
          </Select>
        </div>
        <div className="w-48">
          <label className="text-sm font-medium mb-1 block">Outcome (for VE)</label>
          <Select value={outcomeCode} onValueChange={setOutcomeCode}>
            <SelectTrigger data-testid="select-outcome">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="COVID-19">COVID-19 Infection</SelectItem>
              <SelectItem value="INFLUENZA-A">Influenza A</SelectItem>
              <SelectItem value="HOSPITALIZATION">Hospitalization</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="w-48">
          <label className="text-sm font-medium mb-1 block">Location</label>
          <Select value={selectedLocation} onValueChange={setSelectedLocation}>
            <SelectTrigger data-testid="select-vaccine-location">
              <SelectValue placeholder="All locations" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">All locations</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Coverage Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold" data-testid="text-coverage-rate">
              {coverageData?.coverage_rate?.toFixed(1) || 0}%
            </p>
            <Progress value={coverageData?.coverage_rate || 0} className="mt-2 h-2" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Vaccinated</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold" data-testid="text-vaccinated">
              {coverageData?.vaccinated_count?.toLocaleString() || 0}
            </p>
            <p className="text-xs text-muted-foreground">
              of {coverageData?.total_population?.toLocaleString() || 0} total
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Vaccine Effectiveness</CardTitle>
          </CardHeader>
          <CardContent>
            <p className={`text-3xl font-bold ${getEffectivenessColor(effectivenessData?.effectiveness || 0)}`} data-testid="text-effectiveness">
              {effectivenessData?.effectiveness?.toFixed(1) || '-'}%
            </p>
            <p className="text-xs text-muted-foreground">
              CI: [{effectivenessData?.ci_lower?.toFixed(1)}, {effectivenessData?.ci_upper?.toFixed(1)}]
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <div>
                <p className="text-sm font-medium text-green-600">
                  {effectivenessData?.n_vaccinated?.toLocaleString() || 0}
                </p>
                <p className="text-xs text-muted-foreground">Vaccinated</p>
              </div>
              <div>
                <p className="text-sm font-medium text-orange-600">
                  {effectivenessData?.n_unvaccinated?.toLocaleString() || 0}
                </p>
                <p className="text-xs text-muted-foreground">Unvaccinated</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {doseData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Doses Administered
            </CardTitle>
            <CardDescription>
              Distribution of vaccine doses
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={doseData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="dose" />
                <YAxis />
                <ReTooltip />
                <Bar dataKey="count" fill="#8884d8" name="Recipients" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export function EpidemiologyTab() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Epidemiology Research</h2>
          <p className="text-muted-foreground">
            Drug safety signals, infectious disease surveillance, and vaccine analytics
          </p>
        </div>
        <Badge variant="outline" className="gap-1">
          <Shield className="h-3 w-3" />
          Privacy-Protected
        </Badge>
      </div>

      <Tabs defaultValue="drug-safety" className="space-y-6">
        <TabsList className="flex flex-wrap gap-1 w-full max-w-xl h-auto p-1">
          <TabsTrigger value="drug-safety" className="gap-2" data-testid="tab-drug-safety">
            <Pill className="h-4 w-4" />
            Drug Safety
          </TabsTrigger>
          <TabsTrigger value="infectious" className="gap-2" data-testid="tab-infectious">
            <Activity className="h-4 w-4" />
            Infectious Disease
          </TabsTrigger>
          <TabsTrigger value="vaccine" className="gap-2" data-testid="tab-vaccine">
            <Syringe className="h-4 w-4" />
            Vaccine Analytics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="drug-safety">
          <DrugSafetyPanel />
        </TabsContent>

        <TabsContent value="infectious">
          <InfectiousDiseasePanel />
        </TabsContent>

        <TabsContent value="vaccine">
          <VaccineAnalyticsPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
