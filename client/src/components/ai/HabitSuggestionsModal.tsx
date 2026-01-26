import { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Sparkles, 
  Target, 
  Clock, 
  Flame,
  Heart,
  Activity,
  Moon,
  Salad,
  Pill,
  CheckCircle2,
  Plus,
  Loader2,
  Lightbulb,
  TrendingUp,
} from 'lucide-react';
import type { HabitSuggestion } from '@/hooks/usePatientAI';
import { AIFeedbackButtons } from './AIFeedbackButtons';

interface HabitSuggestionsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  suggestions: HabitSuggestion[];
  experienceId: string;
  patientId: string;
  isLoading?: boolean;
  onAddHabit?: (habit: HabitSuggestion) => void;
}

const getCategoryIcon = (category: string) => {
  switch (category.toLowerCase()) {
    case 'health':
      return <Heart className="h-4 w-4" />;
    case 'exercise':
      return <Activity className="h-4 w-4" />;
    case 'sleep':
      return <Moon className="h-4 w-4" />;
    case 'nutrition':
      return <Salad className="h-4 w-4" />;
    case 'medication':
      return <Pill className="h-4 w-4" />;
    case 'wellness':
      return <Flame className="h-4 w-4" />;
    default:
      return <Target className="h-4 w-4" />;
  }
};

const getDifficultyColor = (difficulty: string) => {
  switch (difficulty) {
    case 'easy':
      return 'text-green-500 bg-green-500/10';
    case 'medium':
      return 'text-yellow-500 bg-yellow-500/10';
    case 'hard':
      return 'text-red-500 bg-red-500/10';
    default:
      return 'text-muted-foreground bg-muted';
  }
};

export function HabitSuggestionsModal({
  open,
  onOpenChange,
  suggestions,
  experienceId,
  patientId,
  isLoading,
  onAddHabit,
}: HabitSuggestionsModalProps) {
  const [addedHabits, setAddedHabits] = useState<Set<string>>(new Set());
  const [addingHabit, setAddingHabit] = useState<string | null>(null);

  const handleAddHabit = async (habit: HabitSuggestion) => {
    setAddingHabit(habit.id);
    try {
      await onAddHabit?.(habit);
      setAddedHabits((prev) => new Set([...prev, habit.id]));
    } finally {
      setAddingHabit(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            AI-Optimized Habit Suggestions
          </DialogTitle>
          <DialogDescription>
            Personalized habits based on your health profile and goals
          </DialogDescription>
        </DialogHeader>
        
        <div className="flex-1 overflow-y-auto py-4 space-y-4" data-testid="habit-suggestions-list">
          {isLoading ? (
            <>
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-32 w-full" />
              ))}
            </>
          ) : suggestions.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Lightbulb className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No suggestions available at this time.</p>
              <p className="text-sm">Check back later for personalized recommendations.</p>
            </div>
          ) : (
            suggestions.map((habit) => (
              <Card
                key={habit.id}
                className="hover-elevate"
                data-testid={`card-habit-suggestion-${habit.id}`}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3 flex-1">
                      <div className={`p-2 rounded-lg ${getDifficultyColor(habit.difficulty)}`}>
                        {getCategoryIcon(habit.category)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <h4 className="font-medium">{habit.name}</h4>
                          <Badge variant="outline" className="text-xs capitalize">
                            {habit.difficulty}
                          </Badge>
                          <Badge variant="secondary" className="text-xs">
                            <Clock className="h-3 w-3 mr-1" />
                            {habit.frequency}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-2">
                          {habit.description}
                        </p>
                        {habit.benefits && habit.benefits.length > 0 && (
                          <div className="flex flex-wrap gap-2 mt-2">
                            {habit.benefits.slice(0, 3).map((benefit, i) => (
                              <div
                                key={i}
                                className="flex items-center gap-1 text-xs text-green-600 dark:text-green-400"
                              >
                                <TrendingUp className="h-3 w-3" />
                                {benefit}
                              </div>
                            ))}
                          </div>
                        )}
                        {habit.tips && habit.tips.length > 0 && (
                          <div className="mt-2 p-2 rounded bg-muted/50 text-xs">
                            <strong>Tip:</strong> {habit.tips[0]}
                          </div>
                        )}
                      </div>
                    </div>
                    <Button
                      variant={addedHabits.has(habit.id) ? 'outline' : 'default'}
                      size="sm"
                      onClick={() => handleAddHabit(habit)}
                      disabled={addedHabits.has(habit.id) || addingHabit === habit.id}
                      className="shrink-0"
                      data-testid={`button-add-habit-${habit.id}`}
                    >
                      {addingHabit === habit.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : addedHabits.has(habit.id) ? (
                        <>
                          <CheckCircle2 className="h-4 w-4 mr-1" />
                          Added
                        </>
                      ) : (
                        <>
                          <Plus className="h-4 w-4 mr-1" />
                          Add
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>

        <DialogFooter className="flex items-center justify-between border-t pt-4">
          {experienceId && (
            <AIFeedbackButtons
              experienceId={experienceId}
              patientId={patientId}
            />
          )}
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
