import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  CheckCircle,
  Circle,
  Phone,
  MessageSquare,
  Video,
  AlertTriangle,
  Heart,
  Pill,
  Activity,
  Brain,
  Mic,
  User,
  Stethoscope,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface DemoStep {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  duration: number;
  category: "patient" | "ai" | "doctor" | "system";
  completed: boolean;
}

const DEMO_STEPS: DemoStep[] = [
  {
    id: "1",
    title: "Patient Opens App",
    description: "Sarah opens Followup AI to check in with Agent Clona",
    icon: User,
    duration: 3,
    category: "patient",
    completed: false,
  },
  {
    id: "2",
    title: "Voice Greeting",
    description: "Clona greets Sarah warmly and asks how she's feeling today",
    icon: Mic,
    duration: 5,
    category: "ai",
    completed: false,
  },
  {
    id: "3",
    title: "Symptom Report",
    description: "Sarah mentions she's been experiencing chest tightness and shortness of breath",
    icon: Heart,
    duration: 8,
    category: "patient",
    completed: false,
  },
  {
    id: "4",
    title: "Red Flag Detection",
    description: "System detects cardiovascular red flags in conversation",
    icon: AlertTriangle,
    duration: 2,
    category: "system",
    completed: false,
  },
  {
    id: "5",
    title: "Clarifying Questions",
    description: "Clona asks about severity, duration, and associated symptoms",
    icon: MessageSquare,
    duration: 10,
    category: "ai",
    completed: false,
  },
  {
    id: "6",
    title: "Risk Assessment",
    description: "AI calculates elevated risk score based on symptoms and history",
    icon: Activity,
    duration: 3,
    category: "system",
    completed: false,
  },
  {
    id: "7",
    title: "Doctor Alert",
    description: "Dr. Johnson receives urgent notification about Sarah's symptoms",
    icon: Stethoscope,
    duration: 2,
    category: "system",
    completed: false,
  },
  {
    id: "8",
    title: "Lysa Briefing",
    description: "Assistant Lysa provides Dr. Johnson with patient context and recommendations",
    icon: Brain,
    duration: 5,
    category: "ai",
    completed: false,
  },
  {
    id: "9",
    title: "Video Consultation",
    description: "Dr. Johnson initiates secure video call with Sarah",
    icon: Video,
    duration: 15,
    category: "doctor",
    completed: false,
  },
  {
    id: "10",
    title: "Prescription Update",
    description: "Doctor adjusts medication and schedules follow-up",
    icon: Pill,
    duration: 5,
    category: "doctor",
    completed: false,
  },
  {
    id: "11",
    title: "Care Plan Update",
    description: "System updates Sarah's care plan with new instructions",
    icon: CheckCircle,
    duration: 3,
    category: "system",
    completed: false,
  },
];

export default function DemoFlow() {
  const [steps, setSteps] = useState<DemoStep[]>(DEMO_STEPS);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const totalDuration = steps.reduce((sum, step) => sum + step.duration, 0);
  const completedDuration = steps
    .filter((s) => s.completed)
    .reduce((sum, step) => sum + step.duration, 0);
  const progress = (completedDuration / totalDuration) * 100;

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "patient":
        return "bg-blue-500";
      case "ai":
        return "bg-purple-500";
      case "doctor":
        return "bg-green-500";
      case "system":
        return "bg-orange-500";
      default:
        return "bg-gray-500";
    }
  };

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case "patient":
        return "Patient Action";
      case "ai":
        return "AI Agent";
      case "doctor":
        return "Doctor Action";
      case "system":
        return "System Process";
      default:
        return category;
    }
  };

  const handleNext = () => {
    if (currentStep < steps.length) {
      const newSteps = [...steps];
      newSteps[currentStep].completed = true;
      setSteps(newSteps);
      setCurrentStep(currentStep + 1);
    }
  };

  const handleReset = () => {
    setSteps(DEMO_STEPS.map((s) => ({ ...s, completed: false })));
    setCurrentStep(0);
    setIsPlaying(false);
    setElapsed(0);
  };

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const current = steps[currentStep];

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2" data-testid="text-demo-title">
          Followup AI Demo Flow
        </h1>
        <p className="text-muted-foreground">
          Experience the complete patient journey from symptom report to care resolution
        </p>
      </div>

      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">Demo Progress</span>
          <span className="text-sm text-muted-foreground">
            {Math.round(progress)}% complete
          </span>
        </div>
        <Progress value={progress} className="h-2" data-testid="progress-demo" />
      </div>

      <Card className="mb-6">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl">Demo Controls</CardTitle>
              <CardDescription>Control the demo playback</CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={togglePlay}
                data-testid="button-play-pause"
              >
                {isPlaying ? (
                  <Pause className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={handleNext}
                disabled={currentStep >= steps.length}
                data-testid="button-next-step"
              >
                <SkipForward className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={handleReset}
                data-testid="button-reset"
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {current ? (
            <div className="flex items-start gap-4 p-4 bg-muted/50 rounded-lg">
              <div
                className={cn(
                  "p-3 rounded-full",
                  getCategoryColor(current.category)
                )}
              >
                <current.icon className="h-6 w-6 text-white" />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-semibold text-lg">{current.title}</h3>
                  <Badge variant="secondary" className="text-xs">
                    {getCategoryLabel(current.category)}
                  </Badge>
                </div>
                <p className="text-muted-foreground">{current.description}</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Duration: ~{current.duration} seconds
                </p>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <h3 className="font-semibold text-lg mb-2">Demo Complete!</h3>
              <p className="text-muted-foreground mb-4">
                You've seen the complete patient journey through Followup AI
              </p>
              <Button onClick={handleReset} data-testid="button-restart-demo">
                Restart Demo
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {["patient", "ai", "doctor", "system"].map((category) => (
          <Card key={category} className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <div
                className={cn(
                  "w-3 h-3 rounded-full",
                  getCategoryColor(category)
                )}
              />
              <span className="font-medium text-sm">
                {getCategoryLabel(category)}
              </span>
            </div>
            <p className="text-2xl font-bold">
              {steps.filter((s) => s.category === category && s.completed).length}
              <span className="text-muted-foreground text-sm font-normal">
                /{steps.filter((s) => s.category === category).length}
              </span>
            </p>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Demo Steps</CardTitle>
          <CardDescription>
            Follow the patient journey through each step
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {steps.map((step, index) => {
              const StepIcon = step.icon;
              const isCurrent = index === currentStep;
              const isPast = step.completed;
              
              return (
                <div
                  key={step.id}
                  className={cn(
                    "flex items-start gap-4 p-4 rounded-lg transition-colors",
                    isCurrent && "bg-muted border-l-4 border-l-primary",
                    isPast && "opacity-60"
                  )}
                  data-testid={`step-${step.id}`}
                >
                  <div className="flex flex-col items-center">
                    <div
                      className={cn(
                        "p-2 rounded-full",
                        isPast
                          ? "bg-green-100 dark:bg-green-900"
                          : isCurrent
                          ? getCategoryColor(step.category)
                          : "bg-muted"
                      )}
                    >
                      {isPast ? (
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      ) : (
                        <StepIcon
                          className={cn(
                            "h-5 w-5",
                            isCurrent ? "text-white" : "text-muted-foreground"
                          )}
                        />
                      )}
                    </div>
                    {index < steps.length - 1 && (
                      <div
                        className={cn(
                          "w-0.5 h-8 mt-2",
                          isPast ? "bg-green-500" : "bg-muted-foreground/20"
                        )}
                      />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h4 className="font-medium">{step.title}</h4>
                      <Badge
                        variant="outline"
                        className={cn(
                          "text-xs",
                          getCategoryColor(step.category).replace("bg-", "border-")
                        )}
                      >
                        {getCategoryLabel(step.category)}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      {step.description}
                    </p>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    ~{step.duration}s
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <Separator className="my-8" />

      <Card>
        <CardHeader>
          <CardTitle>Key Features Demonstrated</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="p-4 border rounded-lg">
              <Phone className="h-8 w-8 text-purple-500 mb-2" />
              <h4 className="font-medium mb-1">Voice Conversations</h4>
              <p className="text-sm text-muted-foreground">
                Natural voice interactions with AI agents
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <AlertTriangle className="h-8 w-8 text-red-500 mb-2" />
              <h4 className="font-medium mb-1">Red Flag Detection</h4>
              <p className="text-sm text-muted-foreground">
                Automatic detection of 13 emergency categories
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <Video className="h-8 w-8 text-blue-500 mb-2" />
              <h4 className="font-medium mb-1">Video Consultations</h4>
              <p className="text-sm text-muted-foreground">
                HIPAA-compliant doctor-patient video calls
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <Brain className="h-8 w-8 text-green-500 mb-2" />
              <h4 className="font-medium mb-1">AI Risk Assessment</h4>
              <p className="text-sm text-muted-foreground">
                Real-time health deterioration detection
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <Stethoscope className="h-8 w-8 text-orange-500 mb-2" />
              <h4 className="font-medium mb-1">Doctor Escalation</h4>
              <p className="text-sm text-muted-foreground">
                Automated alerts to healthcare providers
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <Pill className="h-8 w-8 text-pink-500 mb-2" />
              <h4 className="font-medium mb-1">Medication Management</h4>
              <p className="text-sm text-muted-foreground">
                Prescription updates and reminders
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
