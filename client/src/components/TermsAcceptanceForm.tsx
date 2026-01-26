import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { Check, Shield, FileText, FlaskConical } from "lucide-react";
import api from "@/lib/api";

interface TermsAcceptanceFormProps {
  onAccepted?: () => void;
  showResearchConsent?: boolean;
}

export function TermsAcceptanceForm({ 
  onAccepted,
  showResearchConsent = true 
}: TermsAcceptanceFormProps) {
  const { toast } = useToast();
  const [termsRead, setTermsRead] = useState(false);
  const [termsAccepted, setTermsAccepted] = useState(false);
  const [researchConsent, setResearchConsent] = useState(false);

  const acceptTermsMutation = useMutation({
    mutationFn: async () => {
      const response = await api.post("/api/terms/accept", {
        terms_version: "v2025-01",
        research_consent: researchConsent,
      });
      return response.data;
    },
    onSuccess: () => {
      toast({
        title: "Terms Accepted",
        description: "Thank you for accepting the Terms and Conditions.",
      });
      onAccepted?.();
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to record terms acceptance. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleScrollEnd = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.target as HTMLDivElement;
    const isAtBottom = target.scrollHeight - target.scrollTop <= target.clientHeight + 50;
    if (isAtBottom) {
      setTermsRead(true);
    }
  };

  const handleSubmit = () => {
    if (!termsAccepted) {
      toast({
        title: "Terms Required",
        description: "You must accept the Terms and Conditions to continue.",
        variant: "destructive",
      });
      return;
    }
    acceptTermsMutation.mutate();
  };

  return (
    <Card className="max-w-2xl mx-auto" data-testid="terms-acceptance-card">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
            <FileText className="h-5 w-5 text-primary" />
          </div>
          <div>
            <CardTitle>Terms and Conditions</CardTitle>
            <CardDescription>Please review and accept to continue</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <ScrollArea 
          className="h-64 w-full rounded-md border p-4"
          onScroll={handleScrollEnd}
          data-testid="terms-scroll-area"
        >
          <div className="space-y-4 text-sm text-muted-foreground">
            <h3 className="font-semibold text-foreground">1. Acceptance of Terms</h3>
            <p>
              By accessing and using Followup AI, you accept and agree to be bound by these terms.
              If you do not agree, please do not use our platform.
            </p>

            <h3 className="font-semibold text-foreground">2. Medical Disclaimer</h3>
            <p>
              <strong>NOT A MEDICAL DEVICE:</strong> Followup AI is a wellness monitoring platform 
              designed for personal health awareness. This platform is NOT intended to diagnose, 
              treat, cure, or prevent any disease. Always consult with your healthcare provider.
            </p>

            <h3 className="font-semibold text-foreground">3. Data Privacy</h3>
            <p>
              We are committed to protecting your health information in accordance with HIPAA 
              regulations. Your data is encrypted, securely stored, and never shared without 
              your explicit consent.
            </p>

            <h3 className="font-semibold text-foreground">4. AI-Generated Insights</h3>
            <p>
              Agent Clona and Assistant Lysa use AI to provide wellness insights. These represent 
              pattern analysis for personal awareness only—NOT medical advice. Discuss all insights 
              with your healthcare provider.
            </p>

            <h3 className="font-semibold text-foreground">5. User Responsibilities</h3>
            <p>
              You agree not to use the platform for unlawful purposes, share credentials, 
              or upload false health information. You are responsible for all medical decisions 
              regarding your health.
            </p>

            <h3 className="font-semibold text-foreground">6. Emergency Notice</h3>
            <p className="font-semibold text-destructive">
              If you are experiencing a medical emergency, call 911 or seek immediate medical 
              attention. Do not rely on this platform for emergency situations.
            </p>

            <div className="pt-4 text-center text-xs">
              <p>Scroll down to continue reading...</p>
            </div>
          </div>
        </ScrollArea>

        {!termsRead && (
          <p className="text-sm text-muted-foreground text-center">
            Please scroll to read the full terms before accepting
          </p>
        )}

        <div className="space-y-4">
          <div className="flex items-start space-x-3 p-4 border rounded-md">
            <Checkbox
              id="accept-terms"
              checked={termsAccepted}
              onCheckedChange={(checked) => setTermsAccepted(checked as boolean)}
              disabled={!termsRead}
              data-testid="checkbox-accept-terms"
            />
            <div className="space-y-1">
              <Label htmlFor="accept-terms" className="font-medium cursor-pointer">
                I have read and agree to the Terms and Conditions
              </Label>
              <p className="text-xs text-muted-foreground">
                Required to use Followup AI
              </p>
            </div>
          </div>

          {showResearchConsent && (
            <div className="flex items-start space-x-3 p-4 border rounded-md bg-muted/30">
              <Checkbox
                id="research-consent"
                checked={researchConsent}
                onCheckedChange={(checked) => setResearchConsent(checked as boolean)}
                data-testid="checkbox-research-consent"
              />
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <Label htmlFor="research-consent" className="font-medium cursor-pointer">
                    Contribute to medical research
                  </Label>
                  <Badge variant="secondary">
                    <FlaskConical className="h-3 w-3 mr-1" />
                    Optional
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  Allow de-identified data to be used for improving healthcare outcomes. 
                  Your privacy is protected through k-anonymity safeguards.
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-4 pt-4 border-t">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Shield className="h-4 w-4" />
            <span>HIPAA Compliant</span>
          </div>
          <Button
            onClick={handleSubmit}
            disabled={!termsAccepted || acceptTermsMutation.isPending}
            data-testid="button-accept-terms"
          >
            {acceptTermsMutation.isPending ? (
              "Processing..."
            ) : (
              <>
                <Check className="h-4 w-4 mr-2" />
                Accept Terms
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default TermsAcceptanceForm;
