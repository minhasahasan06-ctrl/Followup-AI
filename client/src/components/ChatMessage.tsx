import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Bot, User } from "lucide-react";
import { cn } from "@/lib/utils";

interface MedicalEntity {
  text: string;
  type: "medication" | "symptom" | "finding" | "dosage";
}

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  entities?: MedicalEntity[];
  isGP?: boolean;
  testId?: string;
}

export function ChatMessage({
  role,
  content,
  timestamp,
  entities,
  isGP,
  testId,
}: ChatMessageProps) {
  const entityColors = {
    medication: "bg-chart-2/20 text-chart-2 border-chart-2/30",
    symptom: "bg-chart-3/20 text-chart-3 border-chart-3/30",
    finding: "bg-primary/20 text-primary border-primary/30",
    dosage: "bg-chart-4/20 text-chart-4 border-chart-4/30",
  };

  return (
    <div
      className={cn(
        "flex gap-3 mb-4",
        role === "user" && "flex-row-reverse"
      )}
      data-testid={testId || `message-${role}`}
    >
      <Avatar className="h-8 w-8 flex-shrink-0">
        <AvatarFallback className={cn(
          role === "assistant" && "bg-primary text-primary-foreground"
        )}>
          {role === "assistant" ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
        </AvatarFallback>
      </Avatar>
      <div className={cn(
        "flex-1 max-w-2xl space-y-2",
        role === "user" && "flex flex-col items-end"
      )}>
        <div className={cn(
          "rounded-lg p-3",
          role === "user" && "bg-primary/10 border border-primary/20",
          role === "assistant" && "bg-card border border-card-border"
        )}>
          {isGP && (
            <Badge variant="default" className="mb-2 text-xs" data-testid={testId ? `${testId}-badge-ai` : undefined}>
              AI GP Agent
            </Badge>
          )}
          <p className="text-sm leading-relaxed whitespace-pre-wrap" data-testid={testId ? `${testId}-content` : undefined}>
            {content}
          </p>
          {entities && entities.length > 0 && (
            <div className="mt-3 pt-3 border-t flex flex-wrap gap-1.5" data-testid={testId ? `${testId}-entities` : undefined}>
              {entities.map((entity, idx) => (
                <Badge
                  key={idx}
                  variant="outline"
                  className={cn("text-xs font-normal", entityColors[entity.type])}
                  data-testid={testId ? `${testId}-entity-${idx}` : undefined}
                >
                  {entity.text}
                </Badge>
              ))}
            </div>
          )}
        </div>
        <p className="text-xs text-muted-foreground px-1" data-testid={testId ? `${testId}-timestamp` : undefined}>{timestamp}</p>
      </div>
    </div>
  );
}
