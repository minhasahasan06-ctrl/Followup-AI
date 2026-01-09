import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { 
  Shield, 
  Lock, 
  Database, 
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Brain,
  FlaskConical,
  Eye,
  EyeOff,
  Server,
  Cloud,
  ExternalLink,
  Info,
  History,
  Download,
  FileText,
  ShieldCheck,
  ShieldOff,
  Activity,
  HeartPulse,
  Pill,
  Calendar,
  Sparkles,
} from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { format } from "date-fns";

interface ResearchConsent {
  id: string;
  category: string;
  enabled: boolean;
  lastUpdated: string;
}

interface ExternalProcessor {
  name: string;
  purpose: string;
  baaStatus: 'active' | 'pending' | 'none';
  enabled: boolean;
}

interface PrivacySettings {
  globalResearchConsent: boolean;
  deIdentificationEnabled: boolean;
  tinkerEnabled: boolean;
  tinkerBAAMode: boolean;
  externalProcessors: ExternalProcessor[];
  dataCategories: {
    id: string;
    label: string;
    description: string;
    enabled: boolean;
    icon: string;
  }[];
  auditLogCount: number;
  lastAuditDate: string | null;
}

const DATA_CATEGORIES = [
  { id: 'vitals', label: 'Vital Signs', description: 'Heart rate, blood pressure, temperature', icon: 'HeartPulse' },
  { id: 'symptoms', label: 'Symptom Reports', description: 'Daily symptom check-ins and patterns', icon: 'Activity' },
  { id: 'medications', label: 'Medication History', description: 'Prescription data and adherence', icon: 'Pill' },
  { id: 'mental_health', label: 'Mental Health Scores', description: 'PHQ-9, GAD-7 assessments (anonymized)', icon: 'Brain' },
  { id: 'habits', label: 'Habit & Wellness Data', description: 'Daily routines and lifestyle patterns', icon: 'Sparkles' },
  { id: 'followups', label: 'Daily Follow-ups', description: 'Check-in responses and wellness scores', icon: 'Calendar' },
];

const IconMap: Record<string, any> = {
  HeartPulse,
  Activity,
  Pill,
  Brain,
  Sparkles,
  Calendar,
};

export default function PatientPrivacyResearch() {
  const { toast } = useToast();
  
  const { data: settings, isLoading, error, refetch } = useQuery<PrivacySettings>({
    queryKey: ['/api/v1/patient/privacy-settings'],
  });
  
  const updateConsentMutation = useMutation({
    mutationFn: async (data: { category?: string; enabled: boolean; global?: boolean }) => {
      return apiRequest('/api/v1/ml-training/consent/update', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/patient/privacy-settings'] });
      toast({
        title: 'Preferences updated',
        description: 'Your research consent settings have been saved.',
      });
    },
    onError: () => {
      toast({
        title: 'Update failed',
        description: 'Unable to save your preferences. Please try again.',
        variant: 'destructive',
      });
    },
  });
  
  const handleCategoryToggle = (categoryId: string, enabled: boolean) => {
    updateConsentMutation.mutate({ category: categoryId, enabled });
  };
  
  const handleGlobalToggle = (enabled: boolean) => {
    updateConsentMutation.mutate({ global: true, enabled });
  };
  
  if (isLoading) {
    return (
      <div className="container mx-auto py-8 px-4 space-y-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }
  
  const tinkerNonBAAMode = settings?.tinkerEnabled && !settings?.tinkerBAAMode;
  
  return (
    <div className="container mx-auto py-8 px-4 space-y-6" data-testid="page-patient-privacy-research">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Shield className="h-6 w-6 text-primary" />
            Privacy & Research Settings
          </h1>
          <p className="text-muted-foreground">
            Control how your health data is used for research and AI improvement
          </p>
        </div>
        <Badge variant="outline" className="gap-1">
          <Lock className="h-3 w-3" />
          HIPAA Protected
        </Badge>
      </div>
      
      {tinkerNonBAAMode && (
        <Alert variant="destructive" data-testid="alert-non-baa">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle className="font-semibold">External AI Processing (NON-BAA Mode)</AlertTitle>
          <AlertDescription className="mt-2">
            <p className="mb-3">
              Tinker Thinking Machine is operating in <strong>NON-BAA mode</strong>. This means:
            </p>
            <ul className="list-disc list-inside space-y-1 text-sm mb-3">
              <li>Your data is <strong>heavily de-identified</strong> before any external processing</li>
              <li>Only <strong>SHA256 hashed identifiers</strong> are used (no names, MRNs, or direct identifiers)</li>
              <li>All values are <strong>bucketed</strong> (e.g., age ranges instead of exact age)</li>
              <li><strong>K-anonymity (k≥25)</strong> is enforced - your data is grouped with 24+ similar patients</li>
              <li>No Protected Health Information (PHI) is ever sent to external systems</li>
            </ul>
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4" />
              <span className="text-sm font-medium">Privacy Firewall: Active</span>
            </div>
          </AlertDescription>
        </Alert>
      )}
      
      <Card data-testid="card-global-consent">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <FlaskConical className="h-5 w-5 text-primary" />
                Research Participation
              </CardTitle>
              <CardDescription>
                Allow your de-identified data to be used for medical research
              </CardDescription>
            </div>
            <Switch
              checked={settings?.globalResearchConsent ?? false}
              onCheckedChange={handleGlobalToggle}
              disabled={updateConsentMutation.isPending}
              data-testid="switch-global-consent"
            />
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-start gap-3 p-4 rounded-lg bg-muted/50">
            <Info className="h-5 w-5 text-primary mt-0.5" />
            <div className="text-sm">
              <p className="font-medium mb-1">How your data helps research</p>
              <p className="text-muted-foreground">
                Your anonymized health patterns help train AI models that can detect early signs of 
                deterioration in chronic care patients, potentially saving lives. All data is 
                stripped of identifying information before use.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card data-testid="card-data-categories">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            Data Categories
          </CardTitle>
          <CardDescription>
            Choose which types of data can be used for research
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {DATA_CATEGORIES.map(category => {
              const Icon = IconMap[category.icon] || Database;
              const isEnabled = settings?.dataCategories?.find(c => c.id === category.id)?.enabled ?? false;
              
              return (
                <div 
                  key={category.id}
                  className="flex items-center justify-between p-4 rounded-lg border hover-elevate"
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <Icon className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium">{category.label}</p>
                      <p className="text-sm text-muted-foreground">{category.description}</p>
                    </div>
                  </div>
                  <Switch
                    checked={isEnabled}
                    onCheckedChange={(enabled) => handleCategoryToggle(category.id, enabled)}
                    disabled={updateConsentMutation.isPending || !settings?.globalResearchConsent}
                    data-testid={`switch-category-${category.id}`}
                  />
                </div>
              );
            })}
          </div>
          
          {!settings?.globalResearchConsent && (
            <p className="text-sm text-muted-foreground mt-4 text-center">
              Enable global research participation to control individual data categories
            </p>
          )}
        </CardContent>
      </Card>
      
      <Card data-testid="card-privacy-protections">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Lock className="h-5 w-5 text-primary" />
            Privacy Protections
          </CardTitle>
          <CardDescription>
            Security measures protecting your data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg border" data-testid="protection-deidentification">
              <div className="flex items-center gap-2 mb-2">
                <EyeOff className="h-4 w-4 text-green-500" />
                <span className="font-medium">De-identification</span>
                <Badge className="bg-green-500" data-testid="status-deidentification">Active</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                All identifying information (name, MRN, DOB) is removed before research use
              </p>
            </div>
            
            <div className="p-4 rounded-lg border" data-testid="protection-kanonymity">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="h-4 w-4 text-green-500" />
                <span className="font-medium">K-Anonymity</span>
                <Badge className="bg-green-500" data-testid="status-kanonymity">k≥25</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                Your data is always grouped with at least 24 similar patients
              </p>
            </div>
            
            <div className="p-4 rounded-lg border" data-testid="protection-encryption">
              <div className="flex items-center gap-2 mb-2">
                <Lock className="h-4 w-4 text-green-500" />
                <span className="font-medium">Encryption</span>
                <Badge className="bg-green-500" data-testid="status-encryption">AES-256</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                All data is encrypted at rest and in transit
              </p>
            </div>
            
            <div className="p-4 rounded-lg border" data-testid="protection-audit">
              <div className="flex items-center gap-2 mb-2">
                <History className="h-4 w-4 text-blue-500" />
                <span className="font-medium">Audit Logging</span>
                <Badge variant="secondary" data-testid="text-audit-count">{settings?.auditLogCount ?? 0} events</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                All data access is logged for HIPAA compliance
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card data-testid="card-external-processors">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cloud className="h-5 w-5 text-primary" />
            External AI Processors
          </CardTitle>
          <CardDescription>
            Third-party AI systems that may analyze your de-identified data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 rounded-lg border" data-testid="processor-tinker">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-500/10">
                  <Brain className="h-5 w-5 text-purple-500" />
                </div>
                <div>
                  <p className="font-medium">Tinker Thinking Machine</p>
                  <p className="text-sm text-muted-foreground">
                    Advanced AI analysis for cohort studies
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {settings?.tinkerBAAMode ? (
                  <Badge className="bg-green-500 gap-1" data-testid="status-tinker-baa">
                    <ShieldCheck className="h-3 w-3" />
                    BAA Active
                  </Badge>
                ) : (
                  <Badge variant="destructive" className="gap-1" data-testid="status-tinker-nonbaa">
                    <ShieldOff className="h-3 w-3" />
                    NON-BAA
                  </Badge>
                )}
                <Badge variant={settings?.tinkerEnabled ? "default" : "secondary"} data-testid="status-tinker-enabled">
                  {settings?.tinkerEnabled ? "Enabled" : "Disabled"}
                </Badge>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-4 rounded-lg border" data-testid="processor-openai">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-500/10">
                  <Server className="h-5 w-5 text-blue-500" />
                </div>
                <div>
                  <p className="font-medium">OpenAI GPT-4o</p>
                  <p className="text-sm text-muted-foreground">
                    Clinical reasoning and PHI detection
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge className="bg-green-500 gap-1" data-testid="status-openai-baa">
                  <ShieldCheck className="h-3 w-3" />
                  BAA Active
                </Badge>
                <Badge data-testid="status-openai-enabled">Enabled</Badge>
              </div>
            </div>
          </div>
          
          <Alert className="mt-4">
            <Info className="h-4 w-4" />
            <AlertDescription className="text-sm">
              External processors in BAA mode have signed Business Associate Agreements 
              ensuring HIPAA-compliant handling of your data. NON-BAA processors never 
              receive any PHI - only fully de-identified, k-anonymized aggregates.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
      
      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="rights">
          <AccordionTrigger className="text-lg font-semibold">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Your Rights
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4 p-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium">Right to Withdraw</p>
                  <p className="text-sm text-muted-foreground">
                    You can withdraw your research consent at any time. Future data will no 
                    longer be used, though data already incorporated into aggregate models 
                    cannot be extracted.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium">Right to Access</p>
                  <p className="text-sm text-muted-foreground">
                    You can request a copy of all data collected about you through your 
                    patient dashboard.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium">Right to Be Forgotten</p>
                  <p className="text-sm text-muted-foreground">
                    You can request deletion of your personal data, subject to legal 
                    retention requirements.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium">Right to Know</p>
                  <p className="text-sm text-muted-foreground">
                    You can view which studies have used your de-identified data and 
                    review the audit log of access events.
                  </p>
                </div>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
        
        <AccordionItem value="faq">
          <AccordionTrigger className="text-lg font-semibold">
            <div className="flex items-center gap-2">
              <Info className="h-5 w-5" />
              Frequently Asked Questions
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4 p-4">
              <div>
                <p className="font-medium">Will my doctor see my research preferences?</p>
                <p className="text-sm text-muted-foreground mt-1">
                  No, your research consent choices are private. Your doctor can only see 
                  whether you have consented to their specific care, not your research preferences.
                </p>
              </div>
              
              <div>
                <p className="font-medium">Can my data be sold?</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Never. Your data is only used for academic and clinical research to 
                  improve patient outcomes. It is never sold to third parties.
                </p>
              </div>
              
              <div>
                <p className="font-medium">What is k-anonymity?</p>
                <p className="text-sm text-muted-foreground mt-1">
                  K-anonymity means your data is only released when it's indistinguishable 
                  from at least k-1 other patients. With k=25, your record looks identical 
                  to at least 24 others, making re-identification virtually impossible.
                </p>
              </div>
              
              <div>
                <p className="font-medium">What does NON-BAA mode mean?</p>
                <p className="text-sm text-muted-foreground mt-1">
                  In NON-BAA mode, external AI systems have not signed a Business Associate 
                  Agreement. However, these systems never receive any PHI - only fully 
                  de-identified, hashed, and aggregated data that cannot be traced back to you.
                </p>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
      
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              Last updated: {settings?.lastAuditDate ? format(new Date(settings.lastAuditDate), 'PPP') : 'Never'}
            </div>
            <Button variant="outline" size="sm" className="gap-2">
              <Download className="h-4 w-4" />
              Download Privacy Report
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
