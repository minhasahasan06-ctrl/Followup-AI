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
      "🪑 SITTING POSITION (RECOMMENDED): Sit upright in a chair with back support, feet flat on floor, hands on thighs. Relax shoulders.",
      "🛏️ LYING DOWN (ALTERNATIVE): Lie flat on back, arms at sides, thin pillow under head (optional). Relax completely.",
      "📹 Position camera at chest level (sitting) or above chest (lying down)",
      "🫁 Breathe naturally for 60-90 seconds - don't alter your breathing",
      "💡 Ensure good lighting on your chest area"
    ],
    tips: [
      "Sitting position provides most accurate measurements (recommended for Asthma, COPD, Heart Failure patients)",
      "Remove tight clothing that restricts chest movement",
      "Avoid talking during recording - stay still while breathing normally",
      "Camera will analyze: breathing rate, rhythm stability, neck muscle use, chest shape, breathing coordination"
    ],
    color: "text-blue-500"
  },
  skin_pallor: {
    title: "Skin, Nails & Nail Bed Examination",
    icon: Hand,
    instructions: [
      "Show your palms to the camera in good lighting (10 seconds)",
      "Show the backs of your hands with focus on nails and nail beds (10 seconds)",
      "Show the soles of your feet if possible (10 seconds)",
      "Hold each position steady for clear AI analysis",
      "Ensure natural lighting for accurate color detection"
    ],
    tips: [
      "AI checks for: anaemia (pale nail beds), nicotine stains, burns, discoloration",
      "Compare both hands/feet for symmetry",
      "Avoid colored lighting or nail polish if possible",
      "Keep hands relaxed and fingers spread to show nails clearly"
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
    title: "Edema (Swelling) Assessment",
    icon: User,
    instructions: [
      "📹 STEP 1 - Show Both Sides (30 sec): Position camera to show both legs/feet/ankles side-by-side for symmetry comparison",
      "👆 STEP 2 - Pitting Test (Optional, 15-30 sec): Gently press finger on swollen area for 5-15 seconds, then release",
      "⏱️ STEP 3 - Record Rebound: Keep camera on pressed area after releasing to measure how fast dimple disappears",
      "🔄 STEP 4 - Show Face (if swelling present): Front view + side views to assess facial edema",
      "💡 Ensure good lighting and clear view of skin surface"
    ],
    tips: [
      "AI analyzes: Location (legs/ankles/feet/face), Symmetry (one side vs both), Pitting grade (1-4), Volume change from baseline",
      "Pitting Grade Scale: Grade 1 (immediate rebound, 2mm pit), Grade 2 (<15 sec, 3-4mm), Grade 3 (15-60 sec, 5-6mm), Grade 4 (2-3 min, 8mm)",
      "Compare left vs right - asymmetric swelling may indicate different causes",
      "Note any skin tightness, shininess, or color changes (redness)"
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
