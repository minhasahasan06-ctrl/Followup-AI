import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info } from "lucide-react";

interface LegalDisclaimerProps {
  className?: string;
  variant?: "default" | "wellness" | "monitoring";
}

export function LegalDisclaimer({ className, variant = "default" }: LegalDisclaimerProps) {
  const messages = {
    default: "This platform is not a medical device and is not intended to diagnose, treat, cure, or prevent any disease. All information provided is for wellness monitoring and personal tracking purposes only. Please discuss any changes or concerns with your healthcare provider.",
    wellness: "Wellness monitoring tool - not a medical device. For informational purposes only. Consult your healthcare provider for medical advice.",
    monitoring: "Change detection and monitoring tool. Not intended for diagnosis. All detected changes should be discussed with your healthcare provider."
  };

  return (
    <Alert className={className} data-testid="alert-legal-disclaimer">
      <Info className="h-4 w-4" />
      <AlertDescription className="text-xs" data-testid="text-disclaimer">
        {messages[variant]}
      </AlertDescription>
    </Alert>
  );
}
