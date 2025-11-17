import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { 
  Wind, 
  Hand, 
  Eye, 
  User, 
  Activity, 
  MessageSquare, 
  AlertCircle,
  Play,
  SkipForward,
  Check
} from "lucide-react";

export type ExamType = 
  | 'respiratory'
  | 'skin_pallor'
  | 'eye_sclera'
  | 'swelling'
  | 'tremor'
  | 'tongue'
  | 'custom';

interface ExamPrepStepProps {
  examType: ExamType;
  sequenceNumber: number;
  totalExams: number;
  onStartRecording: () => void;
  onSkipExam: () => void;
  prepDuration?: number; // seconds, default 30
}

const EXAM_CONFIG: Record<ExamType, {
  title: string;
  icon: typeof Wind;
  instructions: string[];
  tips: string[];
  color: string;
}> = {
  respiratory: {
    title: "Respiratory Rate & Pattern",
    icon: Wind,
    instructions: [
      "Position yourself so your chest and face are clearly visible",
      "Sit in a relaxed position with good lighting",
      "Breathe normally - don't alter your breathing pattern",
      "The camera will record for 30 seconds to analyze your breathing"
    ],
    tips: [
      "Remove any tight clothing that restricts chest movement",
      "Avoid talking during the recording",
      "Stay as still as possible while breathing normally"
    ],
    color: "text-blue-500"
  },
  skin_pallor: {
    title: "Skin Pallor & Color Changes",
    icon: Hand,
    instructions: [
      "Show your palms to the camera in good lighting",
      "Then show the soles of your feet (if possible)",
      "Hold each position steady for 10-15 seconds",
      "Ensure natural lighting for accurate color detection"
    ],
    tips: [
      "Avoid colored lighting or filters",
      "Keep hands relaxed and flat",
      "Compare both hands for symmetry"
    ],
    color: "text-rose-500"
  },
  eye_sclera: {
    title: "Eye Sclera Examination",
    icon: Eye,
    instructions: [
      "Look directly at the camera with eyes wide open",
      "Ensure your eyes are well-lit (no shadows)",
      "Hold steady for 10 seconds",
      "The camera will analyze the white part of your eyes"
    ],
    tips: [
      "Remove glasses if wearing any",
      "Avoid squinting or straining",
      "Natural daylight works best"
    ],
    color: "text-amber-500"
  },
  swelling: {
    title: "Facial & Leg Swelling",
    icon: User,
    instructions: [
      "First, show your full face to the camera (10 seconds)",
      "Then, show your lower legs and ankles (10 seconds)",
      "Turn slowly to show both sides if possible",
      "Keep the camera steady and well-lit"
    ],
    tips: [
      "Press gently on swollen areas before recording",
      "Note any pitting edema (indentation remains)",
      "Compare both sides of face and legs"
    ],
    color: "text-purple-500"
  },
  tremor: {
    title: "Tremor Detection",
    icon: Activity,
    instructions: [
      "Hold your hands out in front of you, fingers spread",
      "Keep your arms extended and steady",
      "Hold this position for 20 seconds",
      "The camera will detect any involuntary movements"
    ],
    tips: [
      "Try to stay as still as possible",
      "Relax your muscles - don't strain",
      "If tremor affects other body parts, show those areas too"
    ],
    color: "text-green-500"
  },
  tongue: {
    title: "Tongue Examination",
    icon: MessageSquare,
    instructions: [
      "Open your mouth wide and stick out your tongue",
      "Hold steady for 10 seconds",
      "Ensure good lighting inside your mouth",
      "The camera will analyze color and coating"
    ],
    tips: [
      "Avoid eating or drinking for 30 minutes before",
      "Look for any discoloration or coating",
      "Note any pain or unusual sensations"
    ],
    color: "text-pink-500"
  },
  custom: {
    title: "Custom Abnormality",
    icon: AlertCircle,
    instructions: [
      "You'll describe the location and what to look for",
      "Position the camera to clearly show the area of concern",
      "Hold steady for 15-20 seconds",
      "Ensure good lighting on the area"
    ],
    tips: [
      "Be specific about location (e.g., 'left forearm')",
      "Describe what you notice (color, size, texture)",
      "Compare with other side if applicable"
    ],
    color: "text-orange-500"
  }
};

export function ExamPrepStep({
  examType,
  sequenceNumber,
  totalExams,
  onStartRecording,
  onSkipExam,
  prepDuration = 30
}: ExamPrepStepProps) {
  const [countdown, setCountdown] = useState(prepDuration);
  const [isCountingDown, setIsCountingDown] = useState(false);
  const config = EXAM_CONFIG[examType];
  const Icon = config.icon;

  useEffect(() => {
    if (!isCountingDown) return;

    if (countdown === 0) {
      onStartRecording();
      return;
    }

    const timer = setTimeout(() => {
      setCountdown(prev => prev - 1);
    }, 1000);

    return () => clearTimeout(timer);
  }, [countdown, isCountingDown, onStartRecording]);

  const handleStartPrep = () => {
    setIsCountingDown(true);
  };

  const handleSkipPrep = () => {
    setIsCountingDown(false);
    setCountdown(0);
    onStartRecording();
  };

  const progress = ((prepDuration - countdown) / prepDuration) * 100;

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2" data-testid="text-exam-title">
            <Icon className={`h-6 w-6 ${config.color}`} />
            {config.title}
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Examination {sequenceNumber} of {totalExams}
          </p>
        </div>
        <Button 
          variant="outline" 
          size="sm" 
          onClick={onSkipExam}
          data-testid="button-skip-exam"
        >
          <SkipForward className="h-4 w-4 mr-2" />
          Skip This Exam
        </Button>
      </div>

      {/* Prep Progress */}
      {isCountingDown && (
        <Card className="bg-primary/5 border-primary/20">
          <CardContent className="p-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Preparing for examination...</span>
                <span className="text-2xl font-bold text-primary" data-testid="text-countdown">
                  {countdown}s
                </span>
              </div>
              <Progress value={progress} className="h-2" />
              <Button
                variant="secondary"
                onClick={handleSkipPrep}
                className="w-full"
                data-testid="button-skip-prep"
              >
                <SkipForward className="h-4 w-4 mr-2" />
                Skip Preparation & Start Recording Now
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Instructions */}
      <Card data-testid="card-instructions">
        <CardContent className="p-6 space-y-6">
          {/* Main Instructions */}
          <div>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Check className="h-4 w-4 text-chart-2" />
              How to Perform This Examination
            </h3>
            <ol className="space-y-2 text-sm">
              {config.instructions.map((instruction, idx) => (
                <li key={idx} className="flex gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">
                    {idx + 1}
                  </span>
                  <span className="text-muted-foreground pt-0.5">{instruction}</span>
                </li>
              ))}
            </ol>
          </div>

          {/* Tips */}
          <div className="border-t pt-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-amber-500" />
              Important Tips
            </h3>
            <ul className="space-y-2 text-sm">
              {config.tips.map((tip, idx) => (
                <li key={idx} className="flex gap-2">
                  <span className="text-amber-500 flex-shrink-0">•</span>
                  <span className="text-muted-foreground">{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Action Button */}
          {!isCountingDown && (
            <Button
              onClick={handleStartPrep}
              size="lg"
              className="w-full"
              data-testid="button-start-prep"
            >
              <Play className="h-5 w-5 mr-2" />
              Start {prepDuration} Second Preparation
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Wellness Disclaimer */}
      <div className="text-xs text-muted-foreground text-center p-4 bg-muted/30 rounded-lg">
        This is wellness monitoring only. Not medical diagnosis. Discuss any concerns with your healthcare provider.
      </div>
    </div>
  );
}
