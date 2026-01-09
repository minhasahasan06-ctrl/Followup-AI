import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { 
  Users, 
  Filter, 
  ChevronDown, 
  ChevronUp, 
  Save, 
  RefreshCw, 
  Loader2,
  Search,
  AlertTriangle,
  Shield,
  Heart,
  Activity,
  Thermometer,
  Brain,
  Droplet,
  Pill,
  Calendar,
  TrendingUp,
  BarChart,
  PieChart,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { PieChart as RePieChart, Pie, Cell, ResponsiveContainer, BarChart as ReBarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

const CHART_COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00C49F', '#FFBB28', '#FF8042'];

interface CohortFilter {
  ageMin: number | null;
  ageMax: number | null;
  sex: string | null;
  conditions: string[];
  excludeConditions: string[];
  riskScoreMin: number | null;
  riskScoreMax: number | null;
  immuneMarkers: { marker: string; minValue: number | null; maxValue: number | null }[];
  environmentalExposure: { exposureType: string; minValue: number | null; maxValue: number | null }[];
  followupPatterns: {
    minFollowupDays: number | null;
    minFollowupCount: number | null;
    lastFollowupWithinDays: number | null;
  };
  consentDataTypes: string[];
}

interface CohortPreview {
  totalPatients: number;
  matchingPatients: number;
  ageDistribution: { range: string; count: number }[];
  genderDistribution: { name: string; value: number }[];
  conditionDistribution: { name: string; value: number }[];
  riskScoreDistribution: { range: string; count: number }[];
  averageAge: number;
  averageRiskScore: number;
  dataCompleteness: { dataType: string; percentage: number }[];
}

interface SavedCohort {
  id: string;
  name: string;
  description: string;
  filters: CohortFilter;
  patientCount: number;
  createdAt: string;
  updatedAt: string;
}

const COMMON_CONDITIONS = [
  'Post-Transplant',
  'Autoimmune Disease', 
  'Cancer Treatment',
  'Primary Immunodeficiency',
  'HIV/AIDS',
  'Chronic Kidney Disease',
  'Diabetes Type 2',
  'Cardiovascular Disease',
  'COPD',
  'Asthma',
];

const IMMUNE_MARKERS = [
  'CD4 Count',
  'CD8 Count',
  'CD4/CD8 Ratio',
  'IgG Level',
  'IgA Level',
  'IgM Level',
  'WBC Count',
  'Lymphocyte Count',
  'Neutrophil Count',
];

const ENVIRONMENTAL_EXPOSURES = [
  'Air Quality Index (AQI)',
  'PM2.5 Exposure',
  'Pollen Count',
  'Temperature Extreme',
  'UV Index',
  'Humidity Level',
];

const DATA_TYPES_FOR_CONSENT = [
  { key: 'dailyFollowups', label: 'Daily Follow-ups' },
  { key: 'healthAlerts', label: 'Health Alerts' },
  { key: 'medications', label: 'Medications' },
  { key: 'vitals', label: 'Vital Signs' },
  { key: 'immuneMarkers', label: 'Immune Markers' },
  { key: 'environmentalRisk', label: 'Environmental Risk' },
  { key: 'wearableData', label: 'Wearable Data' },
  { key: 'mentalHealth', label: 'Mental Health' },
  { key: 'labResults', label: 'Lab Results' },
];

const defaultFilter: CohortFilter = {
  ageMin: null,
  ageMax: null,
  sex: null,
  conditions: [],
  excludeConditions: [],
  riskScoreMin: null,
  riskScoreMax: null,
  immuneMarkers: [],
  environmentalExposure: [],
  followupPatterns: {
    minFollowupDays: null,
    minFollowupCount: null,
    lastFollowupWithinDays: null,
  },
  consentDataTypes: [],
};

export default function CohortBuilderTab() {
  const { toast } = useToast();
  const [filters, setFilters] = useState<CohortFilter>(defaultFilter);
  const [cohortName, setCohortName] = useState("");
  const [cohortDescription, setCohortDescription] = useState("");
  const [saveCohortOpen, setSaveCohortOpen] = useState(false);
  const [expandedSections, setExpandedSections] = useState<string[]>(['demographics', 'conditions']);

  const { data: savedCohorts } = useQuery<SavedCohort[]>({
    queryKey: ['/api/v1/research-center/cohorts'],
  });

  const { data: preview, isLoading: previewLoading, refetch: refetchPreview } = useQuery<CohortPreview>({
    queryKey: ['/api/v1/research-center/cohort/preview', filters],
    enabled: false,
  });

  const previewMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('POST', '/api/v1/research-center/cohorts/preview', { filters });
      return response.json();
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['/api/v1/research-center/cohort/preview', filters], data);
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const saveCohortMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('POST', '/api/v1/research-center/cohorts', {
        name: cohortName,
        description: cohortDescription,
        filters,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/cohorts'] });
      toast({ title: 'Cohort Saved', description: 'Your cohort definition has been saved' });
      setSaveCohortOpen(false);
      setCohortName("");
      setCohortDescription("");
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const toggleSection = (section: string) => {
    setExpandedSections(prev => 
      prev.includes(section) ? prev.filter(s => s !== section) : [...prev, section]
    );
  };

  const updateFilter = <K extends keyof CohortFilter>(key: K, value: CohortFilter[K]) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const toggleCondition = (condition: string, exclude: boolean = false) => {
    const key = exclude ? 'excludeConditions' : 'conditions';
    const oppositeKey = exclude ? 'conditions' : 'excludeConditions';
    
    setFilters(prev => ({
      ...prev,
      [key]: prev[key].includes(condition) 
        ? prev[key].filter(c => c !== condition)
        : [...prev[key], condition],
      [oppositeKey]: prev[oppositeKey].filter(c => c !== condition),
    }));
  };

  const addImmuneMarkerFilter = (marker: string) => {
    if (filters.immuneMarkers.some(m => m.marker === marker)) return;
    updateFilter('immuneMarkers', [...filters.immuneMarkers, { marker, minValue: null, maxValue: null }]);
  };

  const updateImmuneMarkerFilter = (marker: string, field: 'minValue' | 'maxValue', value: number | null) => {
    updateFilter('immuneMarkers', filters.immuneMarkers.map(m => 
      m.marker === marker ? { ...m, [field]: value } : m
    ));
  };

  const removeImmuneMarkerFilter = (marker: string) => {
    updateFilter('immuneMarkers', filters.immuneMarkers.filter(m => m.marker !== marker));
  };

  const toggleConsentDataType = (dataType: string) => {
    setFilters(prev => ({
      ...prev,
      consentDataTypes: prev.consentDataTypes.includes(dataType)
        ? prev.consentDataTypes.filter(d => d !== dataType)
        : [...prev.consentDataTypes, dataType],
    }));
  };

  const resetFilters = () => {
    setFilters(defaultFilter);
  };

  const loadCohort = (cohort: SavedCohort) => {
    setFilters(cohort.filters);
    toast({ title: 'Cohort Loaded', description: `Loaded "${cohort.name}"` });
  };

  const activeFilterCount = [
    filters.ageMin !== null || filters.ageMax !== null,
    filters.sex !== null,
    filters.conditions.length > 0,
    filters.excludeConditions.length > 0,
    filters.riskScoreMin !== null || filters.riskScoreMax !== null,
    filters.immuneMarkers.length > 0,
    filters.environmentalExposure.length > 0,
    filters.followupPatterns.minFollowupDays !== null ||
    filters.followupPatterns.minFollowupCount !== null ||
    filters.followupPatterns.lastFollowupWithinDays !== null,
    filters.consentDataTypes.length > 0,
  ].filter(Boolean).length;

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2 space-y-4">
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Filter className="h-5 w-5 text-primary" />
                <CardTitle>Consent-Aware Cohort Builder</CardTitle>
              </div>
              <div className="flex items-center gap-2">
                {activeFilterCount > 0 && (
                  <Badge variant="secondary">{activeFilterCount} active filters</Badge>
                )}
                <Button variant="ghost" size="sm" onClick={resetFilters} data-testid="button-reset-filters">
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Reset
                </Button>
              </div>
            </div>
            <CardDescription>
              Define cohort criteria with consent verification. Only patients who have consented to share selected data types will be included.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[600px] pr-4">
              <div className="space-y-4">
                <Collapsible open={expandedSections.includes('demographics')} onOpenChange={() => toggleSection('demographics')}>
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                      <div className="flex items-center gap-2">
                        <Users className="h-4 w-4 text-blue-500" />
                        <span className="font-medium">Demographics</span>
                      </div>
                      {expandedSections.includes('demographics') ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="pt-4 space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Age Range</Label>
                        <div className="flex gap-2 items-center">
                          <Input
                            type="number"
                            placeholder="Min"
                            value={filters.ageMin ?? ''}
                            onChange={(e) => updateFilter('ageMin', e.target.value ? parseInt(e.target.value) : null)}
                            className="w-20"
                            data-testid="input-age-min"
                          />
                          <span className="text-muted-foreground">to</span>
                          <Input
                            type="number"
                            placeholder="Max"
                            value={filters.ageMax ?? ''}
                            onChange={(e) => updateFilter('ageMax', e.target.value ? parseInt(e.target.value) : null)}
                            className="w-20"
                            data-testid="input-age-max"
                          />
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>Sex</Label>
                        <Select value={filters.sex || ''} onValueChange={(v) => updateFilter('sex', v || null)}>
                          <SelectTrigger data-testid="select-sex">
                            <SelectValue placeholder="Any" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="any">Any</SelectItem>
                            <SelectItem value="male">Male</SelectItem>
                            <SelectItem value="female">Female</SelectItem>
                            <SelectItem value="other">Other</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </CollapsibleContent>
                </Collapsible>

                <Separator />

                <Collapsible open={expandedSections.includes('conditions')} onOpenChange={() => toggleSection('conditions')}>
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                      <div className="flex items-center gap-2">
                        <Heart className="h-4 w-4 text-red-500" />
                        <span className="font-medium">Medical Conditions</span>
                        {(filters.conditions.length > 0 || filters.excludeConditions.length > 0) && (
                          <Badge variant="secondary" className="ml-2">
                            {filters.conditions.length + filters.excludeConditions.length}
                          </Badge>
                        )}
                      </div>
                      {expandedSections.includes('conditions') ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="pt-4 space-y-4">
                    <div className="space-y-3">
                      <Label className="text-sm text-muted-foreground">Include patients with these conditions:</Label>
                      <div className="flex flex-wrap gap-2">
                        {COMMON_CONDITIONS.map((condition) => (
                          <Badge
                            key={condition}
                            variant={filters.conditions.includes(condition) ? "default" : "outline"}
                            className={`cursor-pointer ${filters.excludeConditions.includes(condition) ? 'opacity-50' : ''}`}
                            onClick={() => toggleCondition(condition, false)}
                            data-testid={`badge-include-${condition.toLowerCase().replace(/\s+/g, '-')}`}
                          >
                            {condition}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <Label className="text-sm text-muted-foreground">Exclude patients with these conditions:</Label>
                      <div className="flex flex-wrap gap-2">
                        {COMMON_CONDITIONS.map((condition) => (
                          <Badge
                            key={condition}
                            variant={filters.excludeConditions.includes(condition) ? "destructive" : "outline"}
                            className={`cursor-pointer ${filters.conditions.includes(condition) ? 'opacity-50' : ''}`}
                            onClick={() => toggleCondition(condition, true)}
                            data-testid={`badge-exclude-${condition.toLowerCase().replace(/\s+/g, '-')}`}
                          >
                            {condition}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CollapsibleContent>
                </Collapsible>

                <Separator />

                <Collapsible open={expandedSections.includes('risk')} onOpenChange={() => toggleSection('risk')}>
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-amber-500" />
                        <span className="font-medium">Risk Score</span>
                      </div>
                      {expandedSections.includes('risk') ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="pt-4 space-y-4">
                    <div className="space-y-2">
                      <Label>Composite Risk Score Range (0-15)</Label>
                      <div className="flex gap-2 items-center">
                        <Input
                          type="number"
                          placeholder="Min"
                          min={0}
                          max={15}
                          value={filters.riskScoreMin ?? ''}
                          onChange={(e) => updateFilter('riskScoreMin', e.target.value ? parseFloat(e.target.value) : null)}
                          className="w-24"
                          data-testid="input-risk-min"
                        />
                        <span className="text-muted-foreground">to</span>
                        <Input
                          type="number"
                          placeholder="Max"
                          min={0}
                          max={15}
                          value={filters.riskScoreMax ?? ''}
                          onChange={(e) => updateFilter('riskScoreMax', e.target.value ? parseFloat(e.target.value) : null)}
                          className="w-24"
                          data-testid="input-risk-max"
                        />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Filter by deterioration risk: Low (0-5), Moderate (5-10), High (10-15)
                      </p>
                    </div>
                  </CollapsibleContent>
                </Collapsible>

                <Separator />

                <Collapsible open={expandedSections.includes('immune')} onOpenChange={() => toggleSection('immune')}>
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                      <div className="flex items-center gap-2">
                        <Droplet className="h-4 w-4 text-purple-500" />
                        <span className="font-medium">Immune Markers</span>
                        {filters.immuneMarkers.length > 0 && (
                          <Badge variant="secondary" className="ml-2">{filters.immuneMarkers.length}</Badge>
                        )}
                      </div>
                      {expandedSections.includes('immune') ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="pt-4 space-y-4">
                    <div className="space-y-2">
                      <Label>Add Immune Marker Filter</Label>
                      <Select onValueChange={addImmuneMarkerFilter}>
                        <SelectTrigger data-testid="select-add-immune-marker">
                          <SelectValue placeholder="Select marker" />
                        </SelectTrigger>
                        <SelectContent>
                          {IMMUNE_MARKERS.filter(m => !filters.immuneMarkers.some(f => f.marker === m)).map((marker) => (
                            <SelectItem key={marker} value={marker}>{marker}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    {filters.immuneMarkers.map((m) => (
                      <Card key={m.marker} className="bg-muted/50">
                        <CardContent className="p-3">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium">{m.marker}</span>
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              className="h-6 w-6 p-0"
                              onClick={() => removeImmuneMarkerFilter(m.marker)}
                            >
                              ×
                            </Button>
                          </div>
                          <div className="flex gap-2 items-center">
                            <Input
                              type="number"
                              placeholder="Min"
                              value={m.minValue ?? ''}
                              onChange={(e) => updateImmuneMarkerFilter(m.marker, 'minValue', e.target.value ? parseFloat(e.target.value) : null)}
                              className="w-24 h-8"
                            />
                            <span className="text-muted-foreground text-sm">to</span>
                            <Input
                              type="number"
                              placeholder="Max"
                              value={m.maxValue ?? ''}
                              onChange={(e) => updateImmuneMarkerFilter(m.marker, 'maxValue', e.target.value ? parseFloat(e.target.value) : null)}
                              className="w-24 h-8"
                            />
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </CollapsibleContent>
                </Collapsible>

                <Separator />

                <Collapsible open={expandedSections.includes('followup')} onOpenChange={() => toggleSection('followup')}>
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-green-500" />
                        <span className="font-medium">Follow-up Patterns</span>
                      </div>
                      {expandedSections.includes('followup') ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="pt-4 space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <div className="space-y-2">
                        <Label className="text-xs">Min Follow-up Days</Label>
                        <Input
                          type="number"
                          placeholder="e.g., 30"
                          value={filters.followupPatterns.minFollowupDays ?? ''}
                          onChange={(e) => updateFilter('followupPatterns', {
                            ...filters.followupPatterns,
                            minFollowupDays: e.target.value ? parseInt(e.target.value) : null,
                          })}
                          data-testid="input-min-followup-days"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">Min Entry Count</Label>
                        <Input
                          type="number"
                          placeholder="e.g., 10"
                          value={filters.followupPatterns.minFollowupCount ?? ''}
                          onChange={(e) => updateFilter('followupPatterns', {
                            ...filters.followupPatterns,
                            minFollowupCount: e.target.value ? parseInt(e.target.value) : null,
                          })}
                          data-testid="input-min-followup-count"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">Active Within (days)</Label>
                        <Input
                          type="number"
                          placeholder="e.g., 7"
                          value={filters.followupPatterns.lastFollowupWithinDays ?? ''}
                          onChange={(e) => updateFilter('followupPatterns', {
                            ...filters.followupPatterns,
                            lastFollowupWithinDays: e.target.value ? parseInt(e.target.value) : null,
                          })}
                          data-testid="input-last-followup-days"
                        />
                      </div>
                    </div>
                  </CollapsibleContent>
                </Collapsible>

                <Separator />

                <Collapsible open={expandedSections.includes('consent')} onOpenChange={() => toggleSection('consent')}>
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                      <div className="flex items-center gap-2">
                        <Shield className="h-4 w-4 text-cyan-500" />
                        <span className="font-medium">Data Consent Requirements</span>
                        {filters.consentDataTypes.length > 0 && (
                          <Badge variant="secondary" className="ml-2">{filters.consentDataTypes.length}</Badge>
                        )}
                      </div>
                      {expandedSections.includes('consent') ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="pt-4 space-y-4">
                    <p className="text-sm text-muted-foreground">
                      Only include patients who have consented to share these data types for research:
                    </p>
                    <div className="grid grid-cols-2 gap-3">
                      {DATA_TYPES_FOR_CONSENT.map((dt) => (
                        <div key={dt.key} className="flex items-center space-x-2">
                          <Checkbox
                            id={`consent-${dt.key}`}
                            checked={filters.consentDataTypes.includes(dt.key)}
                            onCheckedChange={() => toggleConsentDataType(dt.key)}
                            data-testid={`checkbox-consent-${dt.key}`}
                          />
                          <label
                            htmlFor={`consent-${dt.key}`}
                            className="text-sm cursor-pointer"
                          >
                            {dt.label}
                          </label>
                        </div>
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
            </ScrollArea>
          </CardContent>
          <CardFooter className="flex justify-between gap-2 pt-4 border-t">
            <Button
              variant="outline"
              onClick={() => previewMutation.mutate()}
              disabled={previewMutation.isPending}
              data-testid="button-preview-cohort"
            >
              {previewMutation.isPending ? (
                <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Calculating...</>
              ) : (
                <><Search className="h-4 w-4 mr-2" /> Preview Cohort</>
              )}
            </Button>
            <Dialog open={saveCohortOpen} onOpenChange={setSaveCohortOpen}>
              <DialogTrigger asChild>
                <Button data-testid="button-save-cohort">
                  <Save className="h-4 w-4 mr-2" />
                  Save Cohort
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Save Cohort Definition</DialogTitle>
                  <DialogDescription>
                    Save this cohort for use in studies and analyses
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="cohort-name">Cohort Name</Label>
                    <Input
                      id="cohort-name"
                      value={cohortName}
                      onChange={(e) => setCohortName(e.target.value)}
                      placeholder="e.g., High-risk chronic care adults"
                      data-testid="input-cohort-name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="cohort-desc">Description</Label>
                    <Input
                      id="cohort-desc"
                      value={cohortDescription}
                      onChange={(e) => setCohortDescription(e.target.value)}
                      placeholder="Brief description of this cohort"
                      data-testid="input-cohort-description"
                    />
                  </div>
                  <div className="bg-muted/50 p-3 rounded-lg">
                    <p className="text-sm font-medium mb-2">Active Filters:</p>
                    <p className="text-sm text-muted-foreground">{activeFilterCount} criteria defined</p>
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setSaveCohortOpen(false)}>Cancel</Button>
                  <Button
                    onClick={() => saveCohortMutation.mutate()}
                    disabled={!cohortName || saveCohortMutation.isPending}
                    data-testid="button-confirm-save-cohort"
                  >
                    {saveCohortMutation.isPending ? (
                      <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Saving...</>
                    ) : 'Save Cohort'}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </CardFooter>
        </Card>
      </div>

      <div className="space-y-4">
        {preview ? (
          <>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  Cohort Preview
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-4">
                  <p className="text-4xl font-bold text-primary" data-testid="text-matching-patients">
                    {preview.matchingPatients}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    of {preview.totalPatients} patients match criteria
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm mt-4">
                  <div>
                    <p className="text-muted-foreground">Avg Age</p>
                    <p className="font-medium">{preview.averageAge?.toFixed(1)} years</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Avg Risk</p>
                    <p className="font-medium">{preview.averageRiskScore?.toFixed(2)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <PieChart className="h-4 w-4" />
                  Gender Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={150}>
                  <RePieChart>
                    <Pie
                      data={preview.genderDistribution}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={30}
                      outerRadius={60}
                    >
                      {preview.genderDistribution.map((entry, index) => (
                        <Cell key={entry.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </RePieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <BarChart className="h-4 w-4" />
                  Age Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={150}>
                  <ReBarChart data={preview.ageDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8884d8" radius={[4, 4, 0, 0]} />
                  </ReBarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </>
        ) : (
          <Card>
            <CardContent className="p-6 text-center">
              <Users className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
              <h3 className="text-sm font-medium mb-2">No Preview Yet</h3>
              <p className="text-xs text-muted-foreground">
                Define your cohort criteria and click "Preview Cohort" to see matching patients
              </p>
            </CardContent>
          </Card>
        )}

        {savedCohorts && savedCohorts.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Saved Cohorts</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-48">
                <div className="space-y-2">
                  {savedCohorts.map((cohort) => (
                    <div
                      key={cohort.id}
                      className="p-3 rounded-lg border hover-elevate cursor-pointer"
                      onClick={() => loadCohort(cohort)}
                      data-testid={`card-saved-cohort-${cohort.id}`}
                    >
                      <p className="font-medium text-sm">{cohort.name}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="secondary" className="text-xs">{cohort.patientCount} patients</Badge>
                        <span className="text-xs text-muted-foreground">
                          {new Date(cohort.createdAt).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
