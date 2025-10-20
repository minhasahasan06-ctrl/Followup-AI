import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bell, Droplet, Dumbbell, Pill, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";

interface ReminderCardProps {
  type: "water" | "exercise" | "meditation" | "medication";
  title: string;
  time: string;
  description: string;
  testId?: string;
}

const reminderConfig = {
  water: {
    icon: Droplet,
    color: "text-blue-500",
    bg: "bg-blue-500/10",
  },
  exercise: {
    icon: Dumbbell,
    color: "text-destructive",
    bg: "bg-destructive/10",
  },
  meditation: {
    icon: Bell,
    color: "text-chart-4",
    bg: "bg-chart-4/10",
  },
  medication: {
    icon: Pill,
    color: "text-chart-2",
    bg: "bg-chart-2/10",
  },
};

export function ReminderCard({ type, title, time, description, testId }: ReminderCardProps) {
  const [completed, setCompleted] = useState(false);
  const config = reminderConfig[type];
  const Icon = config.icon;

  if (completed) {
    return (
      <Card className="border-l-4 border-l-chart-2" data-testid={testId ? `${testId}-completed` : undefined}>
        <CardContent className="flex items-center gap-3 p-4">
          <CheckCircle2 className="h-5 w-5 text-chart-2" />
          <div className="flex-1">
            <p className="text-sm font-medium line-through text-muted-foreground" data-testid={testId ? `${testId}-title` : undefined}>
              {title}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="hover-elevate" data-testid={testId}>
      <CardContent className="flex items-start gap-3 p-4">
        <div className={cn("rounded-md p-2", config.bg)}>
          <Icon className={cn("h-5 w-5", config.color)} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-1">
            <h3 className="font-medium text-sm" data-testid={testId ? `${testId}-title` : undefined}>{title}</h3>
            <span className="text-xs text-muted-foreground whitespace-nowrap" data-testid={testId ? `${testId}-time` : undefined}>{time}</span>
          </div>
          <p className="text-sm text-muted-foreground mb-3" data-testid={testId ? `${testId}-description` : undefined}>{description}</p>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              setCompleted(true);
              console.log("Reminder completed:", title);
            }}
            data-testid={testId ? `${testId}-button-complete` : `button-complete-${type}`}
          >
            Mark Complete
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
