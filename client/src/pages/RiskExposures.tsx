import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/useAuth";
import { 
  Shield, 
  Syringe, 
  Briefcase, 
  Dna, 
  Activity
} from "lucide-react";
import {
  RiskExposuresPanel,
} from "@/components/RiskExposuresCards";

interface RiskSummary {
  infectionsCount: number;
  immunizationsCount: number;
  occupationsCount: number;
  geneticFlagsCount: number;
}

export default function RiskExposures() {
  const { user } = useAuth();
  const patientId = user?.id || '';
  const isDoctor = user?.role === 'doctor';

  const { data: summary, isLoading: summaryLoading } = useQuery<RiskSummary>({
    queryKey: ['/api/patients', patientId, 'risk', 'summary'],
    queryFn: () => fetch(`/api/patients/${patientId}/risk/summary`, { credentials: 'include' }).then(r => r.json()),
    enabled: !!patientId,
  });

  if (!user) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Please log in to view your risk profile</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-6 px-4 max-w-7xl" data-testid="page-risk-exposures">
      <div className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-2" data-testid="title-risk-exposures">
          <Activity className="h-6 w-6 text-primary" />
          Risk & Exposures
        </h1>
        <p className="text-muted-foreground mt-1">
          Auto-generated risk profile based on infections, vaccinations, work, and genetic/family risk.
        </p>
      </div>

      {summaryLoading ? (
        <div className="grid grid-cols-4 gap-4 mb-6">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      ) : summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" data-testid="summary-cards">
          <Card className="p-4">
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-red-500" />
              <div>
                <p className="text-2xl font-bold">{summary.infectionsCount}</p>
                <p className="text-xs text-muted-foreground">Infections</p>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-2">
              <Syringe className="h-5 w-5 text-blue-500" />
              <div>
                <p className="text-2xl font-bold">{summary.immunizationsCount}</p>
                <p className="text-xs text-muted-foreground">Vaccinations</p>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-2">
              <Briefcase className="h-5 w-5 text-amber-500" />
              <div>
                <p className="text-2xl font-bold">{summary.occupationsCount}</p>
                <p className="text-xs text-muted-foreground">Occupations</p>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-2">
              <Dna className="h-5 w-5 text-purple-500" />
              <div>
                <p className="text-2xl font-bold">{summary.geneticFlagsCount}</p>
                <p className="text-xs text-muted-foreground">Genetic Flags</p>
              </div>
            </div>
          </Card>
        </div>
      )}

      <RiskExposuresPanel 
        patientId={patientId} 
        isDoctor={isDoctor}
        compact={false}
      />
    </div>
  );
}
