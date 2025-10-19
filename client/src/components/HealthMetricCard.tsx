import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { LucideIcon, TrendingUp, TrendingDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface HealthMetricCardProps {
  title: string;
  value: string;
  unit?: string;
  trend?: "up" | "down" | "stable";
  trendValue?: string;
  icon: LucideIcon;
  status?: "normal" | "warning" | "critical";
  lastUpdated?: string;
}

export function HealthMetricCard({
  title,
  value,
  unit,
  trend,
  trendValue,
  icon: Icon,
  status = "normal",
  lastUpdated,
}: HealthMetricCardProps) {
  const statusColors = {
    normal: "border-l-chart-2",
    warning: "border-l-chart-3",
    critical: "border-l-destructive",
  };

  return (
    <Card className={cn("border-l-4", statusColors[status])}>
      <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
        <p className="text-sm font-medium text-muted-foreground">{title}</p>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline gap-2">
          <p className="text-3xl font-bold" data-testid={`metric-${title.toLowerCase().replace(/\s+/g, "-")}`}>
            {value}
          </p>
          {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
        </div>
        {trend && trendValue && (
          <div className="mt-2 flex items-center gap-1 text-xs">
            {trend === "up" && <TrendingUp className="h-3 w-3 text-chart-2" />}
            {trend === "down" && <TrendingDown className="h-3 w-3 text-destructive" />}
            <span className={cn(
              trend === "up" && "text-chart-2",
              trend === "down" && "text-destructive"
            )}>
              {trendValue}
            </span>
            <span className="text-muted-foreground">vs last week</span>
          </div>
        )}
        {lastUpdated && (
          <p className="mt-1 text-xs text-muted-foreground">
            Updated {lastUpdated}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
