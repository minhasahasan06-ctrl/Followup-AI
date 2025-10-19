import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Pill, Clock, AlertCircle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface MedicationCardProps {
  name: string;
  dosage: string;
  frequency: string;
  nextDose: string;
  status: "taken" | "pending" | "missed";
  aiSuggestion?: string;
  isOTC?: boolean;
}

export function MedicationCard({
  name,
  dosage,
  frequency,
  nextDose,
  status,
  aiSuggestion,
  isOTC,
}: MedicationCardProps) {
  const statusConfig = {
    taken: {
      icon: CheckCircle2,
      color: "text-chart-2",
      bg: "bg-chart-2/10",
    },
    pending: {
      icon: Clock,
      color: "text-chart-3",
      bg: "bg-chart-3/10",
    },
    missed: {
      icon: AlertCircle,
      color: "text-destructive",
      bg: "bg-destructive/10",
    },
  };

  const StatusIcon = statusConfig[status].icon;

  return (
    <Card className="hover-elevate">
      <CardHeader className="flex flex-row items-start justify-between gap-2 space-y-0 pb-3">
        <div className="flex items-start gap-3">
          <div className={cn("rounded-md p-2", statusConfig[status].bg)}>
            <Pill className={cn("h-4 w-4", statusConfig[status].color)} />
          </div>
          <div>
            <h3 className="font-semibold" data-testid={`medication-${name.toLowerCase().replace(/\s+/g, "-")}`}>
              {name}
            </h3>
            <p className="text-sm text-muted-foreground">{dosage}</p>
          </div>
        </div>
        {isOTC && (
          <Badge variant="secondary" className="text-xs">
            OTC
          </Badge>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Frequency</span>
          <span className="font-medium">{frequency}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Next Dose</span>
          <div className="flex items-center gap-1">
            <StatusIcon className={cn("h-3 w-3", statusConfig[status].color)} />
            <span className="font-medium">{nextDose}</span>
          </div>
        </div>
        {aiSuggestion && (
          <div className="rounded-md bg-primary/5 p-2 text-xs">
            <p className="flex items-start gap-1">
              <AlertCircle className="h-3 w-3 mt-0.5 flex-shrink-0 text-primary" />
              <span className="text-muted-foreground">
                <span className="font-medium text-foreground">AI Suggestion:</span>{" "}
                {aiSuggestion}
              </span>
            </p>
          </div>
        )}
        {status === "pending" && (
          <Button
            size="sm"
            className="w-full"
            data-testid="button-mark-taken"
            onClick={() => console.log("Marked as taken")}
          >
            Mark as Taken
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
