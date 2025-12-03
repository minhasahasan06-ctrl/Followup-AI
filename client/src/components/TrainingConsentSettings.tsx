import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { 
  Database,
  Shield,
  Lock,
  Eye,
  CheckCircle2,
  XCircle,
  Info,
  HeartPulse,
  Brain,
  Pill,
  Activity,
  History,
  AlertTriangle,
  Calendar,
  CheckSquare,
  Sparkles,
  Smartphone,
  Watch,
  Waves,
  CloudSun,
  FileText,
  Stethoscope,
  Footprints,
  Moon,
  Zap,
  Thermometer
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface TrainingConsentData {
  id?: string;
  patientId: string;
  isActive: boolean;
  consentedDataTypes: string[];
  anonymizationLevel: string;
  researchUseOnly: boolean;
  optInDate: string | null;
  withdrawalDate: string | null;
}

interface ContributionRecord {
  id: string;
  modelName: string;
  dataTypesContributed: string[];
  recordCount: number;
  contributedAt: string;
  status: string;
}

const DATA_TYPE_CATEGORIES = [
  {
    category: "Daily Health Tracking",
    options: [
      { 
        id: "daily_followup", 
        label: "Daily Followup", 
        description: "Daily check-ins, wellness scores, activity summaries, and health status updates",
        icon: Calendar
      },
      { 
        id: "vitals", 
        label: "Vital Signs", 
        description: "Heart rate, blood pressure, temperature, SpO2, respiratory rate",
        icon: HeartPulse
      },
      { 
        id: "symptoms", 
        label: "Symptom Reports", 
        description: "Daily symptom check-ins, severity ratings, and symptom patterns",
        icon: Activity
      },
    ]
  },
  {
    category: "Lifestyle & Wellness",
    options: [
      { 
        id: "habits", 
        label: "Habit Tracker", 
        description: "Routines, streaks, habit completion patterns, and behavioral trends",
        icon: CheckSquare
      },
      { 
        id: "wellness", 
        label: "Wellness Activities", 
        description: "Wellness events, lifestyle activities, exercise, and self-care routines",
        icon: Sparkles
      },
      { 
        id: "mental_health", 
        label: "Mental Health Scores", 
        description: "PHQ-9, GAD-7, PSS-10 assessment results (anonymized)",
        icon: Brain
      },
    ]
  },
  {
    category: "Wearable Devices",
    options: [
      { 
        id: "wearable_devices", 
        label: "Wearable Devices", 
        description: "Smartwatch, fitness tracker, CGM, pulse oximeter, and other connected devices",
        icon: Watch
      },
      { 
        id: "wearable_heart", 
        label: "Heart & Cardiovascular", 
        description: "Heart rate, HRV, ECG readings, and cardiovascular metrics from wearables",
        icon: HeartPulse
      },
      { 
        id: "wearable_activity", 
        label: "Activity & Movement", 
        description: "Steps, distance, calories burned, activity zones, and exercise data",
        icon: Footprints
      },
      { 
        id: "wearable_sleep", 
        label: "Sleep Data", 
        description: "Sleep duration, sleep stages, sleep quality scores, and sleep patterns",
        icon: Moon
      },
      { 
        id: "wearable_oxygen", 
        label: "Blood Oxygen & Respiratory", 
        description: "SpO2 levels, breathing rate, and respiratory health metrics",
        icon: Waves
      },
      { 
        id: "wearable_stress", 
        label: "Stress & Recovery", 
        description: "Stress scores, recovery metrics, body battery, and readiness data",
        icon: Zap
      },
    ]
  },
  {
    category: "Connected Apps & External Data",
    options: [
      { 
        id: "connected_apps", 
        label: "Connected Apps", 
        description: "Data synced from external health apps (Apple Health, Google Fit, etc.)",
        icon: Smartphone
      },
      { 
        id: "environmental_risk", 
        label: "Environmental Risk", 
        description: "Location-based health risks, air quality, allergens, and weather impacts",
        icon: CloudSun
      },
    ]
  },
  {
    category: "Medical Records",
    options: [
      { 
        id: "medications", 
        label: "Medication Adherence", 
        description: "Medication schedules, adherence patterns, and drug interactions",
        icon: Pill
      },
      { 
        id: "lab_results", 
        label: "Lab Results (Anonymized)", 
        description: "Blood work and test results with all identifiers removed",
        icon: Database
      },
      { 
        id: "medical_history", 
        label: "Medical History", 
        description: "Previous conditions, surgeries, hospitalizations, and past treatments",
        icon: FileText
      },
      { 
        id: "current_conditions", 
        label: "Current Conditions", 
        description: "Active diagnoses, ongoing treatments, and current health status",
        icon: Stethoscope
      },
    ]
  },
];

const DATA_TYPE_OPTIONS = DATA_TYPE_CATEGORIES.flatMap(cat => cat.options);

export default function TrainingConsentSettings() {
  const { toast } = useToast();
  const [pendingChanges, setPendingChanges] = useState<Partial<TrainingConsentData> | null>(null);

  const { data: consentData, isLoading: consentLoading } = useQuery<TrainingConsentData>({
    queryKey: ["/api/ml/training/consent"],
  });

  const { data: contributions } = useQuery<ContributionRecord[]>({
    queryKey: ["/api/ml/training/contributions"],
  });

  const updateConsentMutation = useMutation({
    mutationFn: async (data: Partial<TrainingConsentData>) => {
      return await apiRequest("POST", "/api/ml/training/consent", data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ml/training/consent"] });
      queryClient.invalidateQueries({ queryKey: ["/api/ml/training/contributions"] });
      setPendingChanges(null);
      toast({
        title: "Consent Updated",
        description: "Your ML training preferences have been saved.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to update consent settings",
        variant: "destructive",
      });
    },
  });

  const withdrawConsentMutation = useMutation({
    mutationFn: async () => {
      return await apiRequest("POST", "/api/ml/training/consent/withdraw", {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ml/training/consent"] });
      queryClient.invalidateQueries({ queryKey: ["/api/ml/training/contributions"] });
      toast({
        title: "Consent Withdrawn",
        description: "Your data will no longer be used for ML training.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to withdraw consent",
        variant: "destructive",
      });
    },
  });

  const currentData = pendingChanges ? { ...consentData, ...pendingChanges } : consentData;
  const isActive = currentData?.isActive ?? false;
  const consentedTypes = currentData?.consentedDataTypes ?? [];

  const handleToggleConsent = (checked: boolean) => {
    const newData = { 
      ...currentData, 
      isActive: checked,
      consentedDataTypes: checked ? (consentedTypes.length > 0 ? consentedTypes : ["vitals", "symptoms"]) : []
    };
    setPendingChanges(newData);
  };

  const handleDataTypeToggle = (typeId: string, checked: boolean) => {
    const newTypes = checked 
      ? [...consentedTypes, typeId]
      : consentedTypes.filter(t => t !== typeId);
    
    setPendingChanges({ 
      ...currentData, 
      consentedDataTypes: newTypes,
      isActive: newTypes.length > 0 ? true : false
    });
  };

  const handleSaveChanges = () => {
    if (pendingChanges) {
      updateConsentMutation.mutate(pendingChanges);
    }
  };

  const hasChanges = pendingChanges !== null;

  if (consentLoading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center py-8">
            <div className="animate-pulse text-muted-foreground">Loading consent settings...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                ML Training Data Contribution
              </CardTitle>
              <CardDescription className="mt-2">
                Help improve health predictions for immunocompromised patients by contributing your anonymized health data to train AI models
              </CardDescription>
            </div>
            <Badge 
              variant={isActive ? "default" : "secondary"}
              data-testid="badge-consent-status"
            >
              {isActive ? (
                <>
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Contributing
                </>
              ) : (
                <>
                  <XCircle className="h-3 w-3 mr-1" />
                  Not Contributing
                </>
              )}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
            <div className="flex items-start gap-3">
              <Shield className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <p className="font-medium">Contribute to ML Research</p>
                <p className="text-sm text-muted-foreground">
                  Your data is fully anonymized and used only to train models that help all patients
                </p>
              </div>
            </div>
            <Switch
              checked={isActive}
              onCheckedChange={handleToggleConsent}
              data-testid="switch-ml-consent"
            />
          </div>

          {isActive && (
            <>
              <Separator />
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Eye className="h-4 w-4 text-muted-foreground" />
                    <h4 className="font-medium">Data Types to Contribute</h4>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const allIds = DATA_TYPE_OPTIONS.map(o => o.id);
                        setPendingChanges({ ...currentData, consentedDataTypes: allIds, isActive: true });
                      }}
                      data-testid="button-select-all"
                    >
                      Select All
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setPendingChanges({ ...currentData, consentedDataTypes: [], isActive: false });
                      }}
                      data-testid="button-clear-all"
                    >
                      Clear All
                    </Button>
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">
                  Select which types of your health data can be used for training. All data is anonymized before use.
                </p>
                
                <div className="space-y-6">
                  {DATA_TYPE_CATEGORIES.map((category) => {
                    const categoryOptions = category.options;
                    const selectedCount = categoryOptions.filter(o => consentedTypes.includes(o.id)).length;
                    
                    return (
                      <div key={category.category} className="space-y-3">
                        <div className="flex items-center justify-between">
                          <h5 className="text-sm font-medium text-muted-foreground">{category.category}</h5>
                          <Badge variant="secondary" className="text-xs" data-testid={`badge-category-${category.category.toLowerCase().replace(/\s+/g, '-')}`}>
                            {selectedCount}/{categoryOptions.length} selected
                          </Badge>
                        </div>
                        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                          {categoryOptions.map((option) => {
                            const Icon = option.icon;
                            const isChecked = consentedTypes.includes(option.id);
                            
                            return (
                              <div 
                                key={option.id}
                                className={`flex items-start gap-3 p-3 rounded-lg border transition-colors ${
                                  isChecked ? "border-primary bg-primary/5" : "border-border"
                                }`}
                                data-testid={`card-data-type-${option.id}`}
                              >
                                <Checkbox
                                  id={`data-type-${option.id}`}
                                  checked={isChecked}
                                  onCheckedChange={(checked) => handleDataTypeToggle(option.id, checked as boolean)}
                                  data-testid={`checkbox-data-type-${option.id}`}
                                />
                                <div className="flex-1 min-w-0">
                                  <Label 
                                    htmlFor={`data-type-${option.id}`}
                                    className="flex items-center gap-2 cursor-pointer"
                                  >
                                    <Icon className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                                    <span className="truncate">{option.label}</span>
                                  </Label>
                                  <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                    {option.description}
                                  </p>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <Separator />

              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="privacy">
                  <AccordionTrigger className="text-sm">
                    <div className="flex items-center gap-2">
                      <Lock className="h-4 w-4" />
                      Privacy & Security Information
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="space-y-3 text-sm text-muted-foreground">
                    <div className="flex gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                      <p><strong>Full Anonymization:</strong> All personal identifiers (name, email, location, dates of birth) are removed before any data is used.</p>
                    </div>
                    <div className="flex gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                      <p><strong>K-Anonymity:</strong> Data is only included if at least 10 other patients have similar profiles, preventing re-identification.</p>
                    </div>
                    <div className="flex gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                      <p><strong>Differential Privacy:</strong> Mathematical noise is added to aggregate statistics to prevent individual inference.</p>
                    </div>
                    <div className="flex gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                      <p><strong>Research Use Only:</strong> Your data will never be sold or used for marketing. It is exclusively used to improve health prediction models.</p>
                    </div>
                    <div className="flex gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                      <p><strong>Right to Withdraw:</strong> You can stop contributing at any time. Previously contributed data will be excluded from future training.</p>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </>
          )}

          {hasChanges && (
            <div className="flex justify-end gap-2 pt-4 border-t">
              <Button 
                variant="outline" 
                onClick={() => setPendingChanges(null)}
                data-testid="button-cancel-consent"
              >
                Cancel
              </Button>
              <Button 
                onClick={handleSaveChanges}
                disabled={updateConsentMutation.isPending}
                data-testid="button-save-consent"
              >
                {updateConsentMutation.isPending ? "Saving..." : "Save Changes"}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {isActive && contributions && contributions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <History className="h-4 w-4" />
              Contribution History
            </CardTitle>
            <CardDescription>
              Records of when your anonymized data was used for model training
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3" data-testid="list-contributions">
              {contributions.slice(0, 5).map((contrib, index) => (
                <div 
                  key={contrib.id}
                  className="flex items-center justify-between p-3 rounded-lg border"
                  data-testid={`card-contribution-${index}`}
                >
                  <div>
                    <p className="font-medium text-sm" data-testid={`text-model-name-${index}`}>{contrib.modelName}</p>
                    <p className="text-xs text-muted-foreground" data-testid={`text-record-count-${index}`}>
                      {contrib.recordCount} records · {contrib.dataTypesContributed.join(", ")}
                    </p>
                  </div>
                  <div className="text-right">
                    <Badge variant="secondary" className="text-xs" data-testid={`badge-status-${index}`}>
                      {contrib.status}
                    </Badge>
                    <p className="text-xs text-muted-foreground mt-1" data-testid={`text-date-${index}`}>
                      {new Date(contrib.contributedAt).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {isActive && (
        <Card className="border-destructive/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base text-destructive">
              <AlertTriangle className="h-4 w-4" />
              Withdraw Consent
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Withdrawing consent will stop your data from being used in future training. 
              Data already used in trained models cannot be removed, but will be excluded from all future training runs.
            </p>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button 
                  variant="destructive" 
                  size="sm"
                  data-testid="button-withdraw-consent"
                >
                  Withdraw All Consent
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Withdraw ML Training Consent?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will immediately stop your data from being used for ML training. 
                    Your anonymized data that was already included in trained models cannot be retroactively removed, 
                    but no future training will use your data.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel data-testid="button-cancel-withdraw">Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => withdrawConsentMutation.mutate()}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                    data-testid="button-confirm-withdraw"
                  >
                    Withdraw Consent
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </CardContent>
        </Card>
      )}

      <Card className="bg-muted/30">
        <CardContent className="p-4">
          <div className="flex gap-3">
            <Info className="h-5 w-5 text-muted-foreground flex-shrink-0" />
            <div className="text-sm text-muted-foreground">
              <p className="font-medium text-foreground">About ML Training</p>
              <p className="mt-1">
                Our AI models help predict health deterioration patterns for immunocompromised patients. 
                By contributing your anonymized data, you help improve predictions for yourself and others 
                with similar conditions. All data handling complies with HIPAA regulations.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
