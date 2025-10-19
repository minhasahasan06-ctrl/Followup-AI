import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Camera, CheckCircle2, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

interface AssessmentResult {
  condition: string;
  detected: boolean;
  confidence?: number;
}

interface FollowUpCardProps {
  date: string;
  time: string;
  type: string;
  completed?: boolean;
  results?: AssessmentResult[];
  onStartAssessment?: () => void;
}

export function FollowUpCard({
  date,
  time,
  type,
  completed,
  results,
  onStartAssessment,
}: FollowUpCardProps) {
  return (
    <Card className={cn(
      "border-l-4",
      completed ? "border-l-chart-2" : "border-l-chart-3"
    )}>
      <CardHeader className="flex flex-row items-start justify-between gap-2 space-y-0 pb-3">
        <div>
          <h3 className="font-semibold">{type}</h3>
          <p className="text-sm text-muted-foreground">{date} at {time}</p>
        </div>
        <Badge variant={completed ? "default" : "secondary"} className="bg-chart-2 text-white">
          {completed ? "Complete" : "Pending"}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        {results && results.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-medium">Assessment Results:</p>
            {results.map((result, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between rounded-md bg-muted/50 p-2 text-sm"
              >
                <span className="text-muted-foreground">{result.condition}</span>
                <div className="flex items-center gap-2">
                  {result.detected && result.confidence && (
                    <span className="text-xs text-muted-foreground">
                      {result.confidence}% confidence
                    </span>
                  )}
                  <Badge
                    variant={result.detected ? "destructive" : "default"}
                    className={cn(
                      "text-xs",
                      !result.detected && "bg-chart-2"
                    )}
                  >
                    {result.detected ? "Detected" : "Clear"}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        )}
        
        {!completed && (
          <Button
            className="w-full"
            onClick={() => {
              console.log("Starting camera assessment");
              onStartAssessment?.();
            }}
            data-testid="button-start-assessment"
          >
            <Camera className="h-4 w-4 mr-2" />
            Start Camera Assessment
          </Button>
        )}
        
        {completed && (
          <div className="flex items-center gap-2 text-sm text-chart-2">
            <CheckCircle2 className="h-4 w-4" />
            <span>Assessment completed successfully</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
