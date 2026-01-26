import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Textarea } from '@/components/ui/textarea';
import { ThumbsUp, ThumbsDown, Loader2, CheckCircle2 } from 'lucide-react';
import { useAIFeedback } from '@/hooks/usePatientAI';
import { useToast } from '@/hooks/use-toast';

interface AIFeedbackButtonsProps {
  experienceId: string;
  patientId: string;
  onFeedbackSubmitted?: (rating: string) => void;
  size?: 'sm' | 'default';
  className?: string;
}

export function AIFeedbackButtons({
  experienceId,
  patientId,
  onFeedbackSubmitted,
  size = 'sm',
  className = '',
}: AIFeedbackButtonsProps) {
  const { toast } = useToast();
  const [submitted, setSubmitted] = useState<string | null>(null);
  const [showReasonPopover, setShowReasonPopover] = useState(false);
  const [additionalContext, setAdditionalContext] = useState('');
  const feedbackMutation = useAIFeedback();

  const handleFeedback = async (rating: 'helpful' | 'not_helpful') => {
    if (submitted) return;
    
    try {
      await feedbackMutation.mutateAsync({
        experience_id: experienceId,
        patient_id: patientId,
        rating,
        additional_context: additionalContext || undefined,
      });
      
      setSubmitted(rating);
      setShowReasonPopover(false);
      toast({
        title: 'Feedback received',
        description: 'Thank you for helping improve our AI recommendations.',
      });
      onFeedbackSubmitted?.(rating);
    } catch {
      toast({
        title: 'Error',
        description: 'Failed to submit feedback. Please try again.',
        variant: 'destructive',
      });
    }
  };

  if (submitted) {
    return (
      <div className={`flex items-center gap-2 text-muted-foreground ${className}`}>
        <CheckCircle2 className="h-4 w-4 text-green-500" />
        <span className="text-sm">Thanks for your feedback!</span>
      </div>
    );
  }

  const buttonSize = size === 'sm' ? 'h-8 w-8' : 'h-9 w-9';
  const iconSize = size === 'sm' ? 'h-4 w-4' : 'h-5 w-5';

  return (
    <div className={`flex items-center gap-2 ${className}`} data-testid="ai-feedback-buttons">
      <span className="text-sm text-muted-foreground">Was this helpful?</span>
      <Button
        variant="ghost"
        size="icon"
        className={buttonSize}
        onClick={() => handleFeedback('helpful')}
        disabled={feedbackMutation.isPending}
        data-testid="button-feedback-helpful"
      >
        {feedbackMutation.isPending ? (
          <Loader2 className={`${iconSize} animate-spin`} />
        ) : (
          <ThumbsUp className={iconSize} />
        )}
      </Button>
      <Popover open={showReasonPopover} onOpenChange={setShowReasonPopover}>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className={buttonSize}
            disabled={feedbackMutation.isPending}
            data-testid="button-feedback-not-helpful"
          >
            <ThumbsDown className={iconSize} />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-80" align="end">
          <div className="space-y-3">
            <p className="text-sm font-medium">What could be improved?</p>
            <Textarea
              placeholder="Optional: Tell us how we can improve..."
              value={additionalContext}
              onChange={(e) => setAdditionalContext(e.target.value)}
              className="min-h-[80px]"
              data-testid="input-feedback-reason"
            />
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowReasonPopover(false)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={() => handleFeedback('not_helpful')}
                disabled={feedbackMutation.isPending}
                data-testid="button-submit-feedback"
              >
                {feedbackMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Submit
              </Button>
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
