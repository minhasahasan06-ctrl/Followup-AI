import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { useState, useEffect } from "react";
import {
  Shield,
  Microscope,
  Heart,
  Brain,
  Pill,
  Activity,
  AlertTriangle,
  Clock,
  MapPin,
  Watch,
  FileText,
  MessageSquare,
  BarChart3,
  Lock,
  Info,
  CheckCircle2,
  XCircle,
  Loader2,
  HelpCircle,
  Users,
  Clipboard,
} from "lucide-react";
import type { ResearchDataConsent } from "@shared/schema";

interface DataTypePermissions {
  dailyFollowups: boolean;
  healthAlerts: boolean;
  deteriorationIndex: boolean;
  mlPredictions: boolean;
  environmentalRisk: boolean;
  medications: boolean;
  vitals: boolean;
  immuneMarkers: boolean;
  behavioralData: boolean;
  mentalHealth: boolean;
  wearableData: boolean;
  labResults: boolean;
  conditions: boolean;
  demographics: boolean;
  painTracking: boolean;
  symptomJournal: boolean;
}

const defaultPermissions: DataTypePermissions = {
  dailyFollowups: false,
  healthAlerts: false,
  deteriorationIndex: false,
  mlPredictions: false,
  environmentalRisk: false,
  medications: false,
  vitals: false,
  immuneMarkers: false,
  behavioralData: false,
  mentalHealth: false,
  wearableData: false,
  labResults: false,
  conditions: false,
  demographics: false,
  painTracking: false,
  symptomJournal: false,
};

interface DataTypeInfo {
  key: keyof DataTypePermissions;
  label: string;
  description: string;
  icon: typeof Heart;
  category: "health" | "behavioral" | "clinical" | "environmental" | "personal";
}

const dataTypes: DataTypeInfo[] = [
  {
    key: "dailyFollowups",
    label: "Daily Follow-ups",
    description: "Your daily health check-in responses, including symptom reports and wellness ratings",
    icon: Clock,
    category: "health",
  },
  {
    key: "healthAlerts",
    label: "Health Alerts",
    description: "System-generated health alerts and notifications based on your data patterns",
    icon: AlertTriangle,
    category: "health",
  },
  {
    key: "deteriorationIndex",
    label: "Deterioration Index",
    description: "Calculated risk scores tracking potential health deterioration over time",
    icon: BarChart3,
    category: "clinical",
  },
  {
    key: "mlPredictions",
    label: "ML Predictions",
    description: "AI/ML model predictions about your health outcomes and risks",
    icon: Brain,
    category: "clinical",
  },
  {
    key: "environmentalRisk",
    label: "Environmental Risk Data",
    description: "Environmental exposure data including air quality, pollen, and location-based health risks",
    icon: MapPin,
    category: "environmental",
  },
  {
    key: "medications",
    label: "Medication Records",
    description: "Your medication list, dosages, adherence patterns, and prescription history",
    icon: Pill,
    category: "clinical",
  },
  {
    key: "vitals",
    label: "Vital Signs",
    description: "Heart rate, blood pressure, temperature, oxygen saturation, and other vital measurements",
    icon: Heart,
    category: "health",
  },
  {
    key: "immuneMarkers",
    label: "Immune Markers",
    description: "Immune system data including white blood cell counts, antibody levels, and inflammatory markers",
    icon: Shield,
    category: "clinical",
  },
  {
    key: "behavioralData",
    label: "Behavioral Insights",
    description: "Patterns in app usage, sleep habits, activity levels, and lifestyle behaviors",
    icon: Activity,
    category: "behavioral",
  },
  {
    key: "mentalHealth",
    label: "Mental Health",
    description: "Mental wellness questionnaire responses (PHQ-9, GAD-7, PSS-10) and mood tracking",
    icon: Brain,
    category: "behavioral",
  },
  {
    key: "wearableData",
    label: "Wearable Device Data",
    description: "Data from connected fitness trackers and smartwatches (steps, sleep, heart rate variability)",
    icon: Watch,
    category: "health",
  },
  {
    key: "labResults",
    label: "Lab Results",
    description: "Blood tests, imaging results, and other laboratory data imported from healthcare providers",
    icon: FileText,
    category: "clinical",
  },
  {
    key: "conditions",
    label: "Medical Conditions",
    description: "Your diagnosed conditions, comorbidities, and chronic care status",
    icon: Clipboard,
    category: "personal",
  },
  {
    key: "demographics",
    label: "Demographics",
    description: "Age, gender, location (anonymized), and other demographic information",
    icon: Users,
    category: "personal",
  },
  {
    key: "painTracking",
    label: "Pain Tracking",
    description: "Pain level assessments, body locations, and pain pattern history",
    icon: AlertTriangle,
    category: "health",
  },
  {
    key: "symptomJournal",
    label: "Symptom Journal",
    description: "Detailed symptom entries and AI-extracted symptom mentions from conversations",
    icon: MessageSquare,
    category: "health",
  },
];

const categoryLabels: Record<string, string> = {
  health: "Health Monitoring",
  behavioral: "Behavioral & Mental Health",
  clinical: "Clinical & Predictions",
  environmental: "Environmental",
  personal: "Personal Information",
};

const categoryDescriptions: Record<string, string> = {
  health: "Daily health tracking and vital measurements",
  behavioral: "Behavioral patterns and mental wellness data",
  clinical: "Medical records, predictions, and clinical assessments",
  environmental: "Location-based environmental health factors",
  personal: "Demographics and conditions for cohort matching",
};

export default function ResearchConsentSettings() {
  const { toast } = useToast();
  const [permissions, setPermissions] = useState<DataTypePermissions>(defaultPermissions);
  const [masterConsent, setMasterConsent] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [initialLoaded, setInitialLoaded] = useState(false);

  const { data: consent, isLoading, error } = useQuery<ResearchDataConsent | null>({
    queryKey: ["/api/v1/research-center/consent"],
  });

  useEffect(() => {
    if (consent && !initialLoaded) {
      setMasterConsent(consent.consentEnabled);
      if (consent.dataTypePermissions) {
        setPermissions(consent.dataTypePermissions as DataTypePermissions);
      }
      setInitialLoaded(true);
    }
  }, [consent, initialLoaded]);

  const updateMutation = useMutation({
    mutationFn: async (data: { consentEnabled: boolean; dataTypePermissions: DataTypePermissions }) => {
      const res = await apiRequest("/api/v1/research-center/consent", { method: "POST", json: data });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/research-center/consent"] });
      toast({
        title: "Consent settings saved",
        description: "Your research participation preferences have been updated.",
      });
      setHasChanges(false);
    },
    onError: () => {
      toast({
        title: "Error saving settings",
        description: "Failed to update your consent preferences. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleMasterConsentChange = (enabled: boolean) => {
    setMasterConsent(enabled);
    if (!enabled) {
      setPermissions(defaultPermissions);
    }
    setHasChanges(true);
  };

  const handlePermissionChange = (key: keyof DataTypePermissions, value: boolean) => {
    setPermissions((prev) => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleEnableAll = () => {
    const allEnabled = Object.fromEntries(
      Object.keys(defaultPermissions).map((k) => [k, true])
    ) as DataTypePermissions;
    setPermissions(allEnabled);
    setHasChanges(true);
  };

  const handleDisableAll = () => {
    setPermissions(defaultPermissions);
    setHasChanges(true);
  };

  const handleSave = () => {
    updateMutation.mutate({
      consentEnabled: masterConsent,
      dataTypePermissions: permissions,
    });
  };

  const enabledCount = Object.values(permissions).filter(Boolean).length;
  const totalCount = Object.keys(permissions).length;

  const groupedDataTypes = dataTypes.reduce((acc, dt) => {
    if (!acc[dt.category]) acc[dt.category] = [];
    acc[dt.category].push(dt);
    return acc;
  }, {} as Record<string, DataTypeInfo[]>);

  if (isLoading) {
    return (
      <div className="container mx-auto p-6 max-w-4xl space-y-6">
        <Skeleton className="h-12 w-64" />
        <Skeleton className="h-6 w-96" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6 max-w-4xl">
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error loading consent settings</AlertTitle>
          <AlertDescription>
            Unable to load your research consent preferences. Please refresh the page or contact support.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-4xl space-y-6">
      <div className="flex items-center gap-3">
        <Microscope className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold" data-testid="text-page-title">Research Participation Settings</h1>
          <p className="text-muted-foreground mt-1">
            Control how your health data contributes to epidemiological research
          </p>
        </div>
      </div>

      <Alert className="border-primary/20 bg-primary/5">
        <Info className="h-4 w-4" />
        <AlertTitle>About Research Participation</AlertTitle>
        <AlertDescription className="space-y-2">
          <p>
            By participating in research, your anonymized health data helps improve care for 
            chronic care patients worldwide. Your identity is never shared with researchers.
          </p>
          <ul className="list-disc list-inside text-sm space-y-1 mt-2">
            <li>All data is de-identified before being used in any study</li>
            <li>You can change your preferences or withdraw consent at any time</li>
            <li>Your care is never affected by your research participation choices</li>
            <li>Only approved institutional studies with IRB oversight use this data</li>
          </ul>
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="h-6 w-6 text-primary" />
              <div>
                <CardTitle data-testid="text-master-consent-title">Research Participation</CardTitle>
                <CardDescription>Enable or disable all research data sharing</CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {masterConsent ? (
                <Badge variant="default" className="bg-green-600">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Participating
                </Badge>
              ) : (
                <Badge variant="secondary">
                  <XCircle className="h-3 w-3 mr-1" />
                  Not Participating
                </Badge>
              )}
              <Switch
                checked={masterConsent}
                onCheckedChange={handleMasterConsentChange}
                data-testid="switch-master-consent"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">
              {masterConsent
                ? `${enabledCount} of ${totalCount} data types shared`
                : "No data shared with research studies"}
            </span>
            {masterConsent && (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleEnableAll}
                  data-testid="button-enable-all"
                >
                  Enable All
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDisableAll}
                  data-testid="button-disable-all"
                >
                  Disable All
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {masterConsent && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lock className="h-5 w-5" />
              Data Type Permissions
            </CardTitle>
            <CardDescription>
              Select which types of health data you want to share with research studies
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="max-h-[600px] pr-4">
              <Accordion type="multiple" defaultValue={Object.keys(groupedDataTypes)} className="space-y-2">
                {Object.entries(groupedDataTypes).map(([category, items]) => {
                  const categoryEnabled = items.filter((item) => permissions[item.key]).length;
                  return (
                    <AccordionItem key={category} value={category} className="border rounded-lg px-4">
                      <AccordionTrigger className="hover:no-underline">
                        <div className="flex items-center justify-between w-full pr-4">
                          <div className="flex items-center gap-3">
                            <span className="font-medium">{categoryLabels[category]}</span>
                            <Badge variant="outline" className="text-xs">
                              {categoryEnabled}/{items.length}
                            </Badge>
                          </div>
                          <span className="text-sm text-muted-foreground hidden sm:block">
                            {categoryDescriptions[category]}
                          </span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-4 pt-2">
                        {items.map((dataType) => {
                          const Icon = dataType.icon;
                          return (
                            <div
                              key={dataType.key}
                              className="flex items-start justify-between p-3 rounded-lg border bg-card hover-elevate"
                            >
                              <div className="flex items-start gap-3 flex-1">
                                <Icon className="h-5 w-5 text-muted-foreground mt-0.5" />
                                <div className="space-y-1">
                                  <Label
                                    htmlFor={`permission-${dataType.key}`}
                                    className="font-medium cursor-pointer"
                                  >
                                    {dataType.label}
                                  </Label>
                                  <p className="text-sm text-muted-foreground">
                                    {dataType.description}
                                  </p>
                                </div>
                              </div>
                              <Switch
                                id={`permission-${dataType.key}`}
                                checked={permissions[dataType.key]}
                                onCheckedChange={(checked) =>
                                  handlePermissionChange(dataType.key, checked)
                                }
                                data-testid={`switch-permission-${dataType.key}`}
                              />
                            </div>
                          );
                        })}
                      </AccordionContent>
                    </AccordionItem>
                  );
                })}
              </Accordion>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      <Card className="border-muted">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <HelpCircle className="h-5 w-5" />
            Frequently Asked Questions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="space-y-2">
            <AccordionItem value="what-happens" className="border-none">
              <AccordionTrigger className="text-sm hover:no-underline py-2">
                What happens to my data when I consent to research?
              </AccordionTrigger>
              <AccordionContent className="text-sm text-muted-foreground">
                Your data is anonymized before being included in any research study. Personal 
                identifiers like your name, contact information, and exact location are removed. 
                Only aggregated patterns and anonymized records are used for analysis.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="who-sees" className="border-none">
              <AccordionTrigger className="text-sm hover:no-underline py-2">
                Who can see my research data?
              </AccordionTrigger>
              <AccordionContent className="text-sm text-muted-foreground">
                Only authorized researchers conducting IRB-approved studies can access 
                de-identified research data. Your personal healthcare team sees your full 
                medical records, but research data is separate and anonymized.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="can-withdraw" className="border-none">
              <AccordionTrigger className="text-sm hover:no-underline py-2">
                Can I withdraw my consent later?
              </AccordionTrigger>
              <AccordionContent className="text-sm text-muted-foreground">
                Yes, you can withdraw consent at any time by disabling the master switch above. 
                Previously collected anonymized data may remain in completed studies, but no new 
                data will be collected after withdrawal.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="affect-care" className="border-none">
              <AccordionTrigger className="text-sm hover:no-underline py-2">
                Will my healthcare be affected by my choice?
              </AccordionTrigger>
              <AccordionContent className="text-sm text-muted-foreground">
                No. Your healthcare is completely independent of your research participation 
                choices. You will receive the same quality of care regardless of whether you 
                participate in research.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="emergency-data" className="border-none">
              <AccordionTrigger className="text-sm hover:no-underline py-2">
                Can my data be released in emergency situations?
              </AccordionTrigger>
              <AccordionContent className="text-sm text-muted-foreground">
                Research data is separate from your clinical care records and is not used for 
                emergency medical decisions. In a medical emergency, healthcare providers access 
                your clinical records through standard healthcare channels, not research databases. 
                Your research consent settings do not affect emergency care access to your 
                medical information.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="data-security" className="border-none">
              <AccordionTrigger className="text-sm hover:no-underline py-2">
                How is my research data protected?
              </AccordionTrigger>
              <AccordionContent className="text-sm text-muted-foreground">
                All research data is protected with HIPAA-compliant security measures including 
                encryption at rest and in transit, access controls, and comprehensive audit 
                logging. De-identification techniques ensure your identity cannot be linked 
                back to the research data. Regular security audits verify our protective measures.
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between sticky bottom-0 bg-background py-4 border-t -mx-6 px-6">
        <div className="text-sm text-muted-foreground">
          {hasChanges ? (
            <span className="text-yellow-600 flex items-center gap-1">
              <AlertTriangle className="h-4 w-4" />
              You have unsaved changes
            </span>
          ) : (
            consent?.updatedAt && (
              <span>
                Last updated: {new Date(consent.updatedAt).toLocaleDateString()}
              </span>
            )
          )}
        </div>
        <Button
          onClick={handleSave}
          disabled={!hasChanges || updateMutation.isPending}
          data-testid="button-save-consent"
        >
          {updateMutation.isPending ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Saving...
            </>
          ) : (
            "Save Preferences"
          )}
        </Button>
      </div>
    </div>
  );
}
