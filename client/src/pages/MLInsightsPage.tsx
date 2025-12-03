import { useParams, useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { MLInsightsPanel } from "@/components/MLInsightsPanel";
import { useAuth } from "@/hooks/useAuth";
import { ArrowLeft, Shield } from "lucide-react";
import type { User } from "@shared/schema";

export default function MLInsightsPage() {
  const params = useParams<{ patientId?: string }>();
  const [, navigate] = useLocation();
  const { user } = useAuth();

  const patientId = params.patientId || user?.id;
  const isDoctor = user?.role === "doctor";

  const { data: patient, isLoading: patientLoading } = useQuery<User>({
    queryKey: [`/api/doctor/patients/${patientId}`],
    enabled: isDoctor && !!patientId && patientId !== user?.id,
  });

  const patientName = isDoctor && patient 
    ? `${patient.firstName || ''} ${patient.lastName || ''}`.trim() || patient.email
    : undefined;

  if (!patientId) {
    return (
      <div className="container mx-auto px-4 py-8 text-center">
        <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
        <h1 className="text-2xl font-bold mb-2">ML Insights</h1>
        <p className="text-muted-foreground">Unable to determine patient context.</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6 space-y-6">
      <div className="flex items-center gap-4">
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={() => navigate(isDoctor ? `/doctor/patient/${patientId}` : "/ai-alerts")}
          data-testid="button-back-ml-insights"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
        
        {patientLoading ? (
          <Skeleton className="h-6 w-48" />
        ) : patientName && (
          <div className="text-sm text-muted-foreground">
            Viewing insights for <span className="font-medium text-foreground">{patientName}</span>
          </div>
        )}
      </div>

      <MLInsightsPanel
        patientId={patientId}
        patientName={patientName}
        isDoctor={isDoctor}
      />
    </div>
  );
}
