import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Phone, X } from "lucide-react";
import { useState } from "react";

interface EmergencyAlertProps {
  symptoms: string[];
  onDismiss?: () => void;
}

export function EmergencyAlert({ symptoms, onDismiss }: EmergencyAlertProps) {
  const [dismissed, setDismissed] = useState(false);

  const handleCall911 = () => {
    console.log("Calling 911 with patient information");
    window.location.href = "tel:911";
  };

  if (dismissed) return null;

  return (
    <Alert
      variant="destructive"
      className="border-2 border-destructive mb-4"
      data-testid="alert-emergency"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3 flex-1">
          <AlertTriangle className="h-5 w-5 mt-0.5" />
          <div className="space-y-2 flex-1">
            <AlertDescription className="font-semibold text-base">
              Emergency Symptoms Detected
            </AlertDescription>
            <AlertDescription>
              The following symptoms may indicate a life-threatening condition:
              <ul className="list-disc list-inside mt-2 space-y-1">
                {symptoms.map((symptom, idx) => (
                  <li key={idx}>{symptom}</li>
                ))}
              </ul>
            </AlertDescription>
            <div className="flex gap-2 pt-2">
              <Button
                variant="default"
                className="bg-white text-destructive hover:bg-white/90"
                onClick={handleCall911}
                data-testid="button-call-911"
              >
                <Phone className="h-4 w-4 mr-2" />
                Call 911 Now
              </Button>
              <Button
                variant="outline"
                className="border-white text-white hover:bg-white/10"
                onClick={() => console.log("Notifying emergency contacts")}
                data-testid="button-notify-contacts"
              >
                Notify Emergency Contacts
              </Button>
            </div>
          </div>
        </div>
        {onDismiss && (
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-white hover:bg-white/10"
            onClick={() => {
              setDismissed(true);
              onDismiss();
            }}
            data-testid="button-dismiss-alert"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>
    </Alert>
  );
}
