import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bot,
  Heart,
  Pill,
  Activity,
  TrendingUp,
  MessageSquare,
  Calendar,
  Shield,
  Check,
  ArrowRight,
} from "lucide-react";

interface DemoMessage {
  role: "agent" | "user";
  content: string;
  delay: number;
}

const demoConversation: DemoMessage[] = [
  { role: "agent", content: "Good morning! How are you feeling today?", delay: 0 },
  { role: "user", content: "A bit tired, and my joints feel stiff.", delay: 1500 },
  { role: "agent", content: "I understand. Let me ask a few quick questions to better understand your symptoms.", delay: 3000 },
  { role: "agent", content: "On a scale of 1-10, how would you rate your fatigue?", delay: 4500 },
  { role: "user", content: "Maybe a 6 today.", delay: 6000 },
  { role: "agent", content: "Thanks for sharing. I've logged your symptoms. Based on your recent patterns, I recommend discussing medication timing with Dr. Smith during your next visit.", delay: 7500 },
];

export function MiniDemo() {
  const [currentStep, setCurrentStep] = useState(0);
  const [messages, setMessages] = useState<DemoMessage[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!isPlaying) return;

    if (currentStep >= demoConversation.length) {
      setIsPlaying(false);
      return;
    }

    const timer = setTimeout(() => {
      setMessages((prev) => [...prev, demoConversation[currentStep]]);
      setCurrentStep((prev) => prev + 1);
    }, currentStep === 0 ? 500 : demoConversation[currentStep].delay - demoConversation[currentStep - 1].delay);

    return () => clearTimeout(timer);
  }, [currentStep, isPlaying]);

  const startDemo = () => {
    setMessages([]);
    setCurrentStep(0);
    setIsPlaying(true);
  };

  const resetDemo = () => {
    setMessages([]);
    setCurrentStep(0);
    setIsPlaying(false);
  };

  return (
    <Card className="max-w-lg mx-auto" data-testid="mini-demo-card">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary">
              <Bot className="h-4 w-4 text-primary-foreground" />
            </div>
            <CardTitle className="text-base">Agent Clona Demo</CardTitle>
          </div>
          <Badge variant="secondary">Interactive</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="h-64 overflow-y-auto space-y-3 bg-muted/50 rounded-lg p-3" data-testid="demo-chat-container">
          <AnimatePresence>
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] px-3 py-2 rounded-lg text-sm ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-background border"
                  }`}
                >
                  {msg.content}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {messages.length === 0 && !isPlaying && (
            <div className="h-full flex flex-col items-center justify-center text-center text-muted-foreground">
              <Bot className="h-10 w-10 mb-3 opacity-50" />
              <p className="text-sm">Click "Start Demo" to see Agent Clona in action</p>
            </div>
          )}
          {isPlaying && currentStep > 0 && currentStep < demoConversation.length && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-2 text-muted-foreground text-sm"
            >
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </motion.div>
          )}
        </div>

        <div className="flex gap-2">
          {!isPlaying && messages.length === 0 && (
            <Button onClick={startDemo} className="flex-1" data-testid="button-start-demo">
              Start Demo
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          )}
          {!isPlaying && messages.length > 0 && (
            <Button onClick={resetDemo} variant="outline" className="flex-1" data-testid="button-reset-demo">
              Reset Demo
            </Button>
          )}
          {isPlaying && (
            <Button disabled variant="secondary" className="flex-1">
              Playing...
            </Button>
          )}
        </div>

        <div className="grid grid-cols-4 gap-2 pt-2 border-t">
          <div className="flex flex-col items-center text-center">
            <Heart className="h-5 w-5 text-primary mb-1" />
            <span className="text-xs text-muted-foreground">Symptoms</span>
          </div>
          <div className="flex flex-col items-center text-center">
            <Pill className="h-5 w-5 text-primary mb-1" />
            <span className="text-xs text-muted-foreground">Medications</span>
          </div>
          <div className="flex flex-col items-center text-center">
            <Activity className="h-5 w-5 text-primary mb-1" />
            <span className="text-xs text-muted-foreground">Trends</span>
          </div>
          <div className="flex flex-col items-center text-center">
            <Calendar className="h-5 w-5 text-primary mb-1" />
            <span className="text-xs text-muted-foreground">Follow-ups</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function FeatureHighlights() {
  const features = [
    {
      icon: MessageSquare,
      title: "Daily Check-ins",
      description: "Quick 2-minute conversations with Agent Clona to track your wellbeing",
    },
    {
      icon: TrendingUp,
      title: "Pattern Detection",
      description: "AI identifies trends in your symptoms before they become problems",
    },
    {
      icon: Shield,
      title: "HIPAA Secure",
      description: "Your health data is encrypted and protected to the highest standards",
    },
  ];

  return (
    <div className="grid md:grid-cols-3 gap-6 mt-8">
      {features.map((feature, idx) => (
        <Card key={idx} className="hover-elevate" data-testid={`feature-card-${idx}`}>
          <CardContent className="pt-6">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 mb-4">
              <feature.icon className="h-6 w-6 text-primary" />
            </div>
            <h4 className="font-semibold mb-2">{feature.title}</h4>
            <p className="text-sm text-muted-foreground">{feature.description}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export default MiniDemo;
