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
  testId?: string;
}

export function WellnessSessionCard({
  title,
  duration,
  difficulty,
  description,
  type,
  recommended,
  onStart,
  testId,
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
    <Card className={cn("font-wellness hover-elevate", gradientBg)} data-testid={testId}>
      <CardHeader className="space-y-2">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-lg" data-testid={testId ? `${testId}-title` : undefined}>{title}</CardTitle>
          {recommended && (
            <Badge variant="default" className="text-xs" data-testid={testId ? `${testId}-badge-recommended` : undefined}>
              Recommended
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className={cn("text-xs", difficultyColors[difficulty])} data-testid={testId ? `${testId}-difficulty` : undefined}>
            {difficulty}
          </Badge>
          <div className="flex items-center gap-1 text-xs text-muted-foreground" data-testid={testId ? `${testId}-duration` : undefined}>
            <Clock className="h-3 w-3" />
            {duration}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground" data-testid={testId ? `${testId}-description` : undefined}>{description}</p>
        <Button
          className="w-full"
          onClick={() => {
            console.log(`Starting ${title}`);
            onStart?.();
          }}
          data-testid={testId ? `${testId}-button-start` : `button-start-${title.toLowerCase().replace(/\s+/g, "-")}`}
        >
          <Play className="h-4 w-4 mr-2" />
          Start Session
        </Button>
      </CardContent>
    </Card>
  );
}
