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
  Users,
  HardHat,
  Dna,
  FlaskConical
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { LineChart as ReLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, ResponsiveContainer, BarChart, Bar, Legend } from "recharts";

// Note: Python backend routing is now handled by queryClient.ts getPythonBackendUrl()
// No need to use PYTHON_BACKEND_URL directly - use relative paths and the shared queryFn

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

interface OccupationalSignal {
  id: number;
  industry_code: string;
  industry_name: string;
  hazard_code: string;
  hazard_name: string;
  outcome_code: string;
  outcome_name: string;
  estimate: number;
  ci_lower: number;
  ci_upper: number;
  p_value: number;
  signal_strength: number;
  n_workers: number;
  n_events: number;
  mean_exposure_years: number;
  flagged: boolean;
}

interface OccupationalResponse {
  signals: OccupationalSignal[];
  total_count: number;
  suppressed_count: number;
  privacy_note: string;
}

interface GeneticAssociation {
  id: number;
  rsid: string;
  gene_symbol: string;
  gene_name?: string;
  outcome_code: string;
  outcome_name: string;
  estimate: number;
  ci_lower: number;
  ci_upper: number;
  p_value: number;
  signal_strength: number;
  n_carriers: number;
  n_non_carriers: number;
  flagged: boolean;
}

interface GeneticResponse {
  associations: GeneticAssociation[];
  total_count: number;
  suppressed_count: number;
  privacy_note: string;
}

interface PharmacogenomicInteraction {
  id: number;
  rsid: string;
  gene_symbol: string;
  gene_name?: string;
  drug_code: string;
  drug_name: string;
  interaction_type: string;
  phenotype: string;
  recommendation: string;
  evidence_level: string;
  clinical_impact: string;
  n_patients: number;
}

function DrugSafetyPanel() {
  const { toast } = useToast();
  const [drugQuery, setDrugQuery] = useState("");
  const [outcomeQuery, setOutcomeQuery] = useState("");
  const [selectedLocation, setSelectedLocation] = useState<string>("all");
  const [scope, setScope] = useState<string>("all");

  const { data: signalData, isLoading: loadingSignals, refetch } = useQuery<SignalResponse>({
    queryKey: ['/api/v1/pharmaco/signals', {
      drug_query: drugQuery || undefined,
      outcome_query: outcomeQuery || undefined,
      patient_location_id: selectedLocation !== "all" ? selectedLocation : undefined,
      scope: scope,
      limit: '50'
    }],
    staleTime: 30000,
  });

  const { data: locations } = useQuery<{ locations: Location[] }>({
    queryKey: ['/api/v1/pharmaco/locations'],
    staleTime: 60000,
  });

  const triggerScanMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('/api/v1/pharmaco/run-scan', { method: 'POST' });
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Scan Started", description: "Drug safety scan has been triggered" });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/pharmaco/signals'] });
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
              <SelectItem value="all">All locations</SelectItem>
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
  const [selectedLocation, setSelectedLocation] = useState("all");

  const { data: epicurveData, isLoading: loadingEpicurve } = useQuery<EpicurveResponse>({
    queryKey: ['/api/v1/infectious/epicurve', {
      pathogen_code: pathogenCode,
      location_id: selectedLocation !== "all" ? selectedLocation : undefined
    }],
    staleTime: 30000,
    enabled: !!pathogenCode
  });

  const { data: r0Data, isLoading: loadingR0 } = useQuery<R0Response>({
    queryKey: ['/api/v1/infectious/r0', {
      pathogen_code: pathogenCode,
      location_id: selectedLocation !== "all" ? selectedLocation : undefined
    }],
    staleTime: 30000,
    enabled: !!pathogenCode
  });

  const { data: pathogens } = useQuery<{ pathogens: Array<{ pathogen_code: string; pathogen_name: string; case_count: number }> }>({
    queryKey: ['/api/v1/infectious/pathogens'],
    staleTime: 60000,
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
              <SelectItem value="all">All locations</SelectItem>
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
  const [selectedLocation, setSelectedLocation] = useState("all");
  const [outcomeCode, setOutcomeCode] = useState("COVID-19");

  const { data: coverageData, isLoading: loadingCoverage } = useQuery<VaccineCoverage>({
    queryKey: ['/api/v1/vaccine/coverage', {
      vaccine_code: vaccineCode,
      location_id: selectedLocation !== "all" ? selectedLocation : undefined
    }],
    staleTime: 30000,
    enabled: !!vaccineCode
  });

  const { data: effectivenessData, isLoading: loadingEffectiveness } = useQuery<VaccineEffectiveness>({
    queryKey: ['/api/v1/vaccine/effectiveness', {
      vaccine_code: vaccineCode,
      outcome_code: outcomeCode,
      location_id: selectedLocation !== "all" ? selectedLocation : undefined
    }],
    staleTime: 30000,
    enabled: !!vaccineCode && !!outcomeCode
  });

  const { data: vaccines } = useQuery<Array<{ vaccine_code: string; vaccine_name: string; total_doses: number }>>({
    queryKey: ['/api/v1/vaccine/vaccines'],
    staleTime: 60000,
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
              <SelectItem value="all">All locations</SelectItem>
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

function OccupationalEpidemiologyPanel() {
  const { toast } = useToast();
  const [industryQuery, setIndustryQuery] = useState("");
  const [hazardQuery, setHazardQuery] = useState("");
  const [flaggedOnly, setFlaggedOnly] = useState(false);

  const { data: signalData, isLoading, refetch } = useQuery<OccupationalResponse>({
    queryKey: ['/api/v1/occupational/signals', { 
      industry_query: industryQuery || undefined, 
      hazard_query: hazardQuery || undefined, 
      flagged_only: flaggedOnly ? 'true' : undefined,
      limit: '50' 
    }],
    staleTime: 30000,
  });

  const signals = signalData?.signals || [];

  return (
    <div className="space-y-6">
      <Alert>
        <HardHat className="h-4 w-4" />
        <AlertTitle>Occupational Epidemiology</AlertTitle>
        <AlertDescription>
          Workplace hazard analysis and industry-specific health risk signals. Data aggregated with privacy protection.
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HardHat className="h-5 w-5" />
            Workplace Hazard Signals
          </CardTitle>
          <CardDescription>
            Industry-specific risk signals for occupational exposures
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-3">
            <div className="flex-1 min-w-[200px]">
              <Input
                placeholder="Filter by industry code..."
                value={industryQuery}
                onChange={(e) => setIndustryQuery(e.target.value)}
                className="w-full"
                data-testid="input-industry-filter"
              />
            </div>
            <div className="flex-1 min-w-[200px]">
              <Input
                placeholder="Filter by hazard code..."
                value={hazardQuery}
                onChange={(e) => setHazardQuery(e.target.value)}
                className="w-full"
                data-testid="input-hazard-filter"
              />
            </div>
            <Button
              variant={flaggedOnly ? "default" : "outline"}
              onClick={() => setFlaggedOnly(!flaggedOnly)}
              className="gap-2"
              data-testid="button-flagged-only"
            >
              <AlertTriangle className="h-4 w-4" />
              Flagged Only
            </Button>
            <Button
              variant="outline"
              onClick={() => refetch()}
              className="gap-2"
              data-testid="button-refresh-occupational"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : (
            <ScrollArea className="h-[400px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Industry</TableHead>
                    <TableHead>Hazard</TableHead>
                    <TableHead>Outcome</TableHead>
                    <TableHead className="text-right">HR</TableHead>
                    <TableHead className="text-right">95% CI</TableHead>
                    <TableHead className="text-right">Workers</TableHead>
                    <TableHead className="text-right">Signal</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {signals.map((signal) => (
                    <TableRow key={signal.id} data-testid={`row-occupational-signal-${signal.id}`}>
                      <TableCell>
                        <div className="font-medium">{signal.industry_name}</div>
                        <div className="text-xs text-muted-foreground">{signal.industry_code}</div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="gap-1">
                          <AlertTriangle className="h-3 w-3" />
                          {signal.hazard_name}
                        </Badge>
                      </TableCell>
                      <TableCell>{signal.outcome_name}</TableCell>
                      <TableCell className="text-right font-mono">
                        {signal.estimate.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm text-muted-foreground">
                        ({signal.ci_lower.toFixed(2)} - {signal.ci_upper.toFixed(2)})
                      </TableCell>
                      <TableCell className="text-right">
                        {signal.n_workers.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right">
                        {signal.flagged ? (
                          <Badge variant="destructive" className="gap-1">
                            <AlertTriangle className="h-3 w-3" />
                            {signal.signal_strength?.toFixed(0)}%
                          </Badge>
                        ) : (
                          <Badge variant="secondary">{signal.signal_strength?.toFixed(0)}%</Badge>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                  {signals.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                        No occupational signals found
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </ScrollArea>
          )}

          {signalData?.privacy_note && (
            <p className="text-xs text-muted-foreground">{signalData.privacy_note}</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function GeneticEpidemiologyPanel() {
  const { toast } = useToast();
  const [geneQuery, setGeneQuery] = useState("");
  const [variantQuery, setVariantQuery] = useState("");
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [activeSubTab, setActiveSubTab] = useState("associations");

  const { data: associationData, isLoading: loadingAssoc, refetch: refetchAssoc } = useQuery<GeneticResponse>({
    queryKey: ['/api/v1/genetic/variant-associations', {
      gene_query: geneQuery || undefined,
      rsid_query: variantQuery || undefined,
      flagged_only: flaggedOnly ? 'true' : undefined,
      limit: '50'
    }],
    staleTime: 30000,
  });

  const { data: pgxData, isLoading: loadingPgx } = useQuery<{ interactions: PharmacogenomicInteraction[]; total_count: number }>({
    queryKey: ['/api/v1/genetic/pharmacogenomics', {
      gene_query: geneQuery || undefined,
      limit: '50'
    }],
    staleTime: 30000,
    enabled: activeSubTab === "pharmacogenomics",
  });

  const associations = associationData?.associations || [];
  const pgxInteractions = pgxData?.interactions || [];

  return (
    <div className="space-y-6">
      <Alert>
        <Dna className="h-4 w-4" />
        <AlertTitle>Genetic/Molecular Epidemiology</AlertTitle>
        <AlertDescription>
          Variant-outcome associations, GWAS results, and pharmacogenomic interactions. Privacy-protected aggregated data.
        </AlertDescription>
      </Alert>

      <Tabs value={activeSubTab} onValueChange={setActiveSubTab}>
        <TabsList className="flex flex-wrap gap-1 h-auto p-1">
          <TabsTrigger value="associations" className="gap-2" data-testid="subtab-associations">
            <Dna className="h-4 w-4" />
            Variant Associations
          </TabsTrigger>
          <TabsTrigger value="pharmacogenomics" className="gap-2" data-testid="subtab-pharmacogenomics">
            <FlaskConical className="h-4 w-4" />
            Pharmacogenomics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="associations" className="space-y-4 mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Dna className="h-5 w-5" />
                Variant-Outcome Associations
              </CardTitle>
              <CardDescription>
                Genetic variant associations with health outcomes
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-3">
                <div className="flex-1 min-w-[200px]">
                  <Input
                    placeholder="Filter by gene (e.g., CYP2D6)..."
                    value={geneQuery}
                    onChange={(e) => setGeneQuery(e.target.value)}
                    className="w-full"
                    data-testid="input-gene-filter"
                  />
                </div>
                <div className="flex-1 min-w-[200px]">
                  <Input
                    placeholder="Filter by variant (e.g., rs1065852)..."
                    value={variantQuery}
                    onChange={(e) => setVariantQuery(e.target.value)}
                    className="w-full"
                    data-testid="input-variant-filter"
                  />
                </div>
                <Button
                  variant={flaggedOnly ? "default" : "outline"}
                  onClick={() => setFlaggedOnly(!flaggedOnly)}
                  className="gap-2"
                  data-testid="button-genetic-flagged"
                >
                  <AlertTriangle className="h-4 w-4" />
                  Flagged Only
                </Button>
                <Button
                  variant="outline"
                  onClick={() => refetchAssoc()}
                  className="gap-2"
                  data-testid="button-refresh-genetic"
                >
                  <RefreshCw className="h-4 w-4" />
                  Refresh
                </Button>
              </div>

              {loadingAssoc ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : (
                <ScrollArea className="h-[400px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Variant</TableHead>
                        <TableHead>Gene</TableHead>
                        <TableHead>Outcome</TableHead>
                        <TableHead className="text-right">OR</TableHead>
                        <TableHead className="text-right">95% CI</TableHead>
                        <TableHead className="text-right">P-value</TableHead>
                        <TableHead className="text-right">Signal</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {associations.map((assoc) => (
                        <TableRow key={assoc.id} data-testid={`row-genetic-assoc-${assoc.id}`}>
                          <TableCell className="font-mono">{assoc.rsid}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{assoc.gene_symbol}</Badge>
                          </TableCell>
                          <TableCell>{assoc.outcome_name}</TableCell>
                          <TableCell className="text-right font-mono">
                            {assoc.estimate.toFixed(2)}
                          </TableCell>
                          <TableCell className="text-right font-mono text-sm text-muted-foreground">
                            ({assoc.ci_lower.toFixed(2)} - {assoc.ci_upper.toFixed(2)})
                          </TableCell>
                          <TableCell className="text-right font-mono text-sm">
                            {assoc.p_value < 0.001 ? assoc.p_value.toExponential(2) : assoc.p_value.toFixed(4)}
                          </TableCell>
                          <TableCell className="text-right">
                            {assoc.flagged ? (
                              <Badge variant="destructive" className="gap-1">
                                <AlertTriangle className="h-3 w-3" />
                                {assoc.signal_strength?.toFixed(0)}%
                              </Badge>
                            ) : (
                              <Badge variant="secondary">{assoc.signal_strength?.toFixed(0)}%</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                      {associations.length === 0 && (
                        <TableRow>
                          <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                            No genetic associations found
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </ScrollArea>
              )}

              {associationData?.privacy_note && (
                <p className="text-xs text-muted-foreground">{associationData.privacy_note}</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="pharmacogenomics" className="space-y-4 mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FlaskConical className="h-5 w-5" />
                Pharmacogenomic Interactions
              </CardTitle>
              <CardDescription>
                Gene-drug interactions affecting drug response and toxicity
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-3">
                <div className="flex-1 min-w-[200px]">
                  <Input
                    placeholder="Filter by gene (e.g., CYP2D6)..."
                    value={geneQuery}
                    onChange={(e) => setGeneQuery(e.target.value)}
                    className="w-full"
                    data-testid="input-pgx-gene-filter"
                  />
                </div>
              </div>

              {loadingPgx ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : (
                <ScrollArea className="h-[400px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Variant</TableHead>
                        <TableHead>Gene</TableHead>
                        <TableHead>Drug</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Phenotype</TableHead>
                        <TableHead>Recommendation</TableHead>
                        <TableHead className="text-center">Evidence</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pgxInteractions.map((pgx) => (
                        <TableRow key={pgx.id} data-testid={`row-pgx-${pgx.id}`}>
                          <TableCell className="font-mono">{pgx.rsid}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{pgx.gene_symbol}</Badge>
                          </TableCell>
                          <TableCell>
                            <Badge variant="secondary" className="gap-1">
                              <Pill className="h-3 w-3" />
                              {pgx.drug_name}
                            </Badge>
                          </TableCell>
                          <TableCell>{pgx.interaction_type}</TableCell>
                          <TableCell className="text-sm">{pgx.phenotype}</TableCell>
                          <TableCell className="text-sm max-w-[200px] truncate">
                            {pgx.recommendation}
                          </TableCell>
                          <TableCell className="text-center">
                            <Badge 
                              variant={pgx.evidence_level.startsWith('1') ? 'default' : 'secondary'}
                            >
                              {pgx.evidence_level}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                      {pgxInteractions.length === 0 && (
                        <TableRow>
                          <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                            No pharmacogenomic interactions found
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export function AdvancedAnalyticsTab() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Advanced Analytics</h2>
          <p className="text-muted-foreground">
            Drug safety signals, infectious disease surveillance, vaccine analytics, occupational and genetic epidemiology
          </p>
        </div>
        <Badge variant="outline" className="gap-1">
          <Shield className="h-3 w-3" />
          Privacy-Protected
        </Badge>
      </div>

      <Tabs defaultValue="drug-safety" className="space-y-6">
        <TabsList className="flex flex-wrap gap-1 w-full max-w-3xl h-auto p-1">
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
          <TabsTrigger value="occupational" className="gap-2" data-testid="tab-occupational">
            <HardHat className="h-4 w-4" />
            Occupational
          </TabsTrigger>
          <TabsTrigger value="genetic" className="gap-2" data-testid="tab-genetic">
            <Dna className="h-4 w-4" />
            Genetic
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

        <TabsContent value="occupational">
          <OccupationalEpidemiologyPanel />
        </TabsContent>

        <TabsContent value="genetic">
          <GeneticEpidemiologyPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Backward compatible export
export const EpidemiologyTab = AdvancedAnalyticsTab;
