import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Pause, RotateCcw, Wind } from "lucide-react";
import { cn } from "@/lib/utils";

interface BreathingExerciseProps {
  name: string;
  description: string;
  pattern: number[]; // [inhale, hold, exhale, hold] in seconds
}

export function BreathingExercise({
  name,
  description,
  pattern,
}: BreathingExerciseProps) {
  const [isActive, setIsActive] = useState(false);
  const [currentPhase, setCurrentPhase] = useState(0);
  const [timeLeft, setTimeLeft] = useState(pattern[0]);
  const phases = ["Inhale", "Hold", "Exhale", "Hold"];
  
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isActive) {
      interval = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            const nextPhase = (currentPhase + 1) % 4;
            setCurrentPhase(nextPhase);
            return pattern[nextPhase];
          }
          return prev - 1;
        });
      }, 1000);
    }
    
    return () => clearInterval(interval);
  }, [isActive, currentPhase, pattern]);

  const handleReset = () => {
    setIsActive(false);
    setCurrentPhase(0);
    setTimeLeft(pattern[0]);
  };

  const progress = ((pattern[currentPhase] - timeLeft) / pattern[currentPhase]) * 100;

  return (
    <Card className="font-wellness">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Wind className="h-5 w-5 text-chart-4" />
          <CardTitle className="text-lg">{name}</CardTitle>
        </div>
        <p className="text-sm text-muted-foreground">{description}</p>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="relative h-40 w-40">
            <svg className="transform -rotate-90" width="160" height="160">
              <circle
                cx="80"
                cy="80"
                r="70"
                stroke="hsl(var(--muted))"
                strokeWidth="8"
                fill="none"
              />
              <circle
                cx="80"
                cy="80"
                r="70"
                stroke="hsl(var(--chart-4))"
                strokeWidth="8"
                fill="none"
                strokeDasharray={`${2 * Math.PI * 70}`}
                strokeDashoffset={`${2 * Math.PI * 70 * (1 - progress / 100)}`}
                className="transition-all duration-1000 ease-linear"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <p className="text-3xl font-bold">{timeLeft}</p>
              <p className="text-sm text-muted-foreground">{phases[currentPhase]}</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant={isActive ? "secondary" : "default"}
              size="icon"
              onClick={() => setIsActive(!isActive)}
              data-testid="button-breathing-toggle"
            >
              {isActive ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={handleReset}
              data-testid="button-breathing-reset"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        <div className="grid grid-cols-4 gap-2 text-center text-xs">
          {phases.map((phase, idx) => (
            <div
              key={phase}
              className={cn(
                "rounded-md p-2 transition-colors",
                idx === currentPhase && isActive
                  ? "bg-chart-4/20 text-chart-4 font-medium"
                  : "bg-muted text-muted-foreground"
              )}
            >
              <div>{phase}</div>
              <div className="font-mono">{pattern[idx]}s</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
