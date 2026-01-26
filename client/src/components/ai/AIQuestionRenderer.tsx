import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Checkbox } from '@/components/ui/checkbox';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Brain, Loader2, Sparkles, ChevronRight } from 'lucide-react';
import type { AIQuestion } from '@/hooks/usePatientAI';

interface AIQuestionRendererProps {
  questions: AIQuestion[];
  isLoading?: boolean;
  onSubmit: (answers: Record<string, unknown>) => void;
  isSubmitting?: boolean;
  showAIBadge?: boolean;
}

export function AIQuestionRenderer({
  questions,
  isLoading,
  onSubmit,
  isSubmitting,
  showAIBadge = true,
}: AIQuestionRendererProps) {
  const [answers, setAnswers] = useState<Record<string, unknown>>({});

  const updateAnswer = (questionId: string, value: unknown) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const handleSubmit = () => {
    onSubmit(answers);
  };

  if (isLoading) {
    return (
      <Card data-testid="card-ai-questions-loading">
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-64 mt-2" />
        </CardHeader>
        <CardContent className="space-y-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  if (!questions || questions.length === 0) {
    return null;
  }

  const renderQuestion = (question: AIQuestion) => {
    const value = answers[question.id];

    switch (question.type) {
      case 'text':
        return (
          <Textarea
            placeholder="Type your answer..."
            value={(value as string) || ''}
            onChange={(e) => updateAnswer(question.id, e.target.value)}
            className="min-h-[80px]"
            data-testid={`input-question-${question.id}`}
          />
        );

      case 'scale':
        return (
          <div className="space-y-3">
            <Slider
              value={[(value as number) || question.min || 0]}
              min={question.min || 0}
              max={question.max || 10}
              step={1}
              onValueChange={([v]) => updateAnswer(question.id, v)}
              data-testid={`slider-question-${question.id}`}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{question.min || 0}</span>
              <span className="font-medium text-foreground">{value ?? '-'}</span>
              <span>{question.max || 10}</span>
            </div>
          </div>
        );

      case 'select':
        return (
          <RadioGroup
            value={(value as string) || ''}
            onValueChange={(v) => updateAnswer(question.id, v)}
            className="space-y-2"
            data-testid={`radio-question-${question.id}`}
          >
            {question.options?.map((option) => (
              <div key={option} className="flex items-center space-x-2">
                <RadioGroupItem value={option} id={`${question.id}-${option}`} />
                <Label htmlFor={`${question.id}-${option}`} className="font-normal cursor-pointer">
                  {option}
                </Label>
              </div>
            ))}
          </RadioGroup>
        );

      case 'multiselect':
        const selectedOptions = (value as string[]) || [];
        return (
          <div className="space-y-2" data-testid={`checkbox-question-${question.id}`}>
            {question.options?.map((option) => (
              <div key={option} className="flex items-center space-x-2">
                <Checkbox
                  id={`${question.id}-${option}`}
                  checked={selectedOptions.includes(option)}
                  onCheckedChange={(checked) => {
                    if (checked) {
                      updateAnswer(question.id, [...selectedOptions, option]);
                    } else {
                      updateAnswer(question.id, selectedOptions.filter((o) => o !== option));
                    }
                  }}
                />
                <Label htmlFor={`${question.id}-${option}`} className="font-normal cursor-pointer">
                  {option}
                </Label>
              </div>
            ))}
          </div>
        );

      case 'boolean':
        return (
          <div className="flex items-center space-x-3" data-testid={`switch-question-${question.id}`}>
            <Switch
              checked={(value as boolean) || false}
              onCheckedChange={(v) => updateAnswer(question.id, v)}
            />
            <Label className="font-normal">{value ? 'Yes' : 'No'}</Label>
          </div>
        );

      default:
        return (
          <Input
            placeholder="Type your answer..."
            value={(value as string) || ''}
            onChange={(e) => updateAnswer(question.id, e.target.value)}
            data-testid={`input-question-${question.id}`}
          />
        );
    }
  };

  return (
    <Card data-testid="card-ai-questions">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            Personalized Questions
          </CardTitle>
          {showAIBadge && (
            <Badge variant="secondary" className="gap-1">
              <Sparkles className="h-3 w-3" />
              AI-Powered
            </Badge>
          )}
        </div>
        <CardDescription>
          Answer these questions to help us understand your health better
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {questions.map((question, index) => (
          <div key={question.id} className="space-y-3">
            <div className="flex items-start gap-3">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary text-sm font-medium">
                {index + 1}
              </div>
              <div className="flex-1 space-y-3">
                <Label className="text-base font-medium">
                  {question.text}
                  {question.required && <span className="text-destructive ml-1">*</span>}
                </Label>
                {question.category && (
                  <Badge variant="outline" className="text-xs">
                    {question.category}
                  </Badge>
                )}
                {renderQuestion(question)}
              </div>
            </div>
          </div>
        ))}
        <Button
          onClick={handleSubmit}
          disabled={isSubmitting}
          className="w-full mt-4"
          data-testid="button-submit-questions"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              Continue
              <ChevronRight className="h-4 w-4 ml-2" />
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}
