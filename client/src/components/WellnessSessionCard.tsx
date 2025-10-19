import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, Clock, Heart } from "lucide-react";
import { cn } from "@/lib/utils";

interface WellnessSessionCardProps {
  title: string;
  duration: string;
  difficulty: "Easy" | "Moderate" | "Intense";
  description: string;
  type: "meditation" | "exercise";
  recommended?: boolean;
  onStart?: () => void;
}

export function WellnessSessionCard({
  title,
  duration,
  difficulty,
  description,
  type,
  recommended,
  onStart,
}: WellnessSessionCardProps) {
  const difficultyColors = {
    Easy: "bg-chart-2/20 text-chart-2",
    Moderate: "bg-chart-3/20 text-chart-3",
    Intense: "bg-destructive/20 text-destructive",
  };

  const gradientBg = type === "meditation"
    ? "bg-gradient-to-br from-chart-4/5 to-chart-4/10"
    : "bg-gradient-to-br from-destructive/5 to-destructive/10";

  return (
    <Card className={cn("font-wellness hover-elevate", gradientBg)}>
      <CardHeader className="space-y-2">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-lg">{title}</CardTitle>
          {recommended && (
            <Badge variant="default" className="text-xs">
              Recommended
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className={cn("text-xs", difficultyColors[difficulty])}>
            {difficulty}
          </Badge>
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            {duration}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{description}</p>
        <Button
          className="w-full"
          onClick={() => {
            console.log(`Starting ${title}`);
            onStart?.();
          }}
          data-testid={`button-start-${title.toLowerCase().replace(/\s+/g, "-")}`}
        >
          <Play className="h-4 w-4 mr-2" />
          Start Session
        </Button>
      </CardContent>
    </Card>
  );
}
