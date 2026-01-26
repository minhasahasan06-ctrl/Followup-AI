import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Sparkles, 
  CheckCircle2, 
  Clock, 
  Target, 
  Calendar,
  ChevronRight,
  Zap,
  Heart,
  Pill,
  Activity,
} from 'lucide-react';
import type { AutopilotTemplate } from '@/hooks/usePatientAI';
import { AIFeedbackButtons } from './AIFeedbackButtons';

interface AutopilotPlanCardProps {
  templates: AutopilotTemplate[];
  experienceId: string;
  patientId: string;
  isLoading?: boolean;
  onSelectTemplate?: (template: AutopilotTemplate) => void;
}

const getCategoryIcon = (category: string) => {
  switch (category.toLowerCase()) {
    case 'medication':
      return <Pill className="h-4 w-4" />;
    case 'exercise':
    case 'activity':
      return <Activity className="h-4 w-4" />;
    case 'wellness':
    case 'health':
      return <Heart className="h-4 w-4" />;
    default:
      return <Target className="h-4 w-4" />;
  }
};

const getPriorityColor = (priority?: string) => {
  switch (priority) {
    case 'high':
      return 'text-red-500 bg-red-500/10';
    case 'medium':
      return 'text-yellow-500 bg-yellow-500/10';
    case 'low':
      return 'text-green-500 bg-green-500/10';
    default:
      return 'text-muted-foreground bg-muted';
  }
};

export function AutopilotPlanCard({
  templates,
  experienceId,
  patientId,
  isLoading,
  onSelectTemplate,
}: AutopilotPlanCardProps) {
  if (isLoading) {
    return (
      <Card data-testid="card-autopilot-loading">
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-64 mt-2" />
        </CardHeader>
        <CardContent className="space-y-4">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </CardContent>
      </Card>
    );
  }

  if (!templates || templates.length === 0) {
    return null;
  }

  return (
    <Card data-testid="card-autopilot-plan">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            Your Personalized Plan
          </CardTitle>
          <Badge variant="secondary" className="gap-1">
            <Sparkles className="h-3 w-3" />
            AI-Generated
          </Badge>
        </div>
        <CardDescription>
          Based on your answers, here are your recommended activities
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {templates.map((template, index) => (
          <div
            key={template.id}
            className="p-4 rounded-lg border bg-card hover-elevate cursor-pointer transition-all"
            onClick={() => onSelectTemplate?.(template)}
            data-testid={`card-template-${template.id}`}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-start gap-3 flex-1">
                <div className={`p-2 rounded-lg ${getPriorityColor(template.priority)}`}>
                  {getCategoryIcon(template.category)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-medium">{template.name}</h4>
                    {template.priority && (
                      <Badge variant="outline" className="text-xs capitalize">
                        {template.priority}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground line-clamp-2">
                    {template.description}
                  </p>
                  <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                    {template.frequency && (
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {template.frequency}
                      </span>
                    )}
                    {template.duration && (
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {template.duration}
                      </span>
                    )}
                  </div>
                  {template.actions && template.actions.length > 0 && (
                    <div className="mt-3 space-y-1">
                      {template.actions.slice(0, 3).map((action, i) => (
                        <div key={i} className="flex items-center gap-2 text-sm">
                          <CheckCircle2 className="h-3 w-3 text-green-500" />
                          <span>{action}</span>
                        </div>
                      ))}
                      {template.actions.length > 3 && (
                        <span className="text-xs text-muted-foreground">
                          +{template.actions.length - 3} more steps
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
              <ChevronRight className="h-5 w-5 text-muted-foreground shrink-0" />
            </div>
          </div>
        ))}
      </CardContent>
      <CardFooter className="flex items-center justify-between border-t pt-4">
        <AIFeedbackButtons
          experienceId={experienceId}
          patientId={patientId}
        />
        <Button variant="outline" size="sm" data-testid="button-view-all-templates">
          View All Templates
        </Button>
      </CardFooter>
    </Card>
  );
}
