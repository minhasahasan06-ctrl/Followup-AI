import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Heart, Sun, Moon, Cloud, Sparkles, Music, BookOpen, Smile } from "lucide-react";
import { motion } from "framer-motion";

interface WelcomeContent {
  greeting: string;
  message: string;
  quote?: string;
  quoteAuthor?: string;
  healthInsight?: string;
  imageUrl?: string;
  musicNote?: string;
}

const INSPIRATIONAL_QUOTES = [
  {
    quote: "Every day is a new beginning. Take a deep breath and start again.",
    author: "Unknown"
  },
  {
    quote: "Your health is an investment, not an expense.",
    author: "Unknown"
  },
  {
    quote: "Small steps every day lead to big changes.",
    author: "Unknown"
  },
  {
    quote: "Be kind to yourself. You're doing better than you think.",
    author: "Unknown"
  },
  {
    quote: "Healing is a matter of time, but it is sometimes also a matter of opportunity.",
    author: "Hippocrates"
  },
  {
    quote: "The greatest wealth is health.",
    author: "Virgil"
  },
  {
    quote: "Take care of your body. It's the only place you have to live.",
    author: "Jim Rohn"
  },
  {
    quote: "A positive attitude gives you power over your circumstances.",
    author: "Joyce Meyer"
  },
  {
    quote: "Today is a gift. That's why it's called the present.",
    author: "Alice Morse Earle"
  },
  {
    quote: "You are stronger than you know, braver than you believe.",
    author: "Christopher Robin"
  }
];

const CALMING_IMAGES = [
  "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=400&fit=crop", // Mountain sunrise
  "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=400&fit=crop", // Forest path
  "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800&h=400&fit=crop", // Misty mountains
  "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=800&h=400&fit=crop", // Ocean sunset
  "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800&h=400&fit=crop", // Lake reflection
  "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=800&h=400&fit=crop", // Nature landscape
  "https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?w=800&h=400&fit=crop", // Flower field
  "https://images.unsplash.com/photo-1475924156734-496f6cac6ec1?w=800&h=400&fit=crop", // Peaceful beach
];

const MUSIC_NOTES = [
  "Morning calm meditation",
  "Gentle piano melody",
  "Nature sounds",
  "Soft acoustic guitar",
  "Peaceful ambience",
];

export default function DynamicWelcome({ userName }: { userName?: string }) {
  const [welcomeContent, setWelcomeContent] = useState<WelcomeContent | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date());

  const { data: recentJournals } = useQuery({
    queryKey: ['/api/journals/recent'],
  });

  const { data: healthMetrics } = useQuery({
    queryKey: ['/api/patient/profile'],
  });

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 60000); // Update every minute
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    generateWelcomeContent();
  }, [currentTime, healthMetrics, recentJournals]);

  const getTimeOfDay = () => {
    const hour = currentTime.getHours();
    if (hour < 12) return "morning";
    if (hour < 17) return "afternoon";
    if (hour < 21) return "evening";
    return "night";
  };

  const getGreeting = () => {
    const timeOfDay = getTimeOfDay();
    const name = userName ? `, ${userName}` : "";
    
    const greetings: Record<string, string[]> = {
      morning: [
        `Good morning${name}!`,
        `Rise and shine${name}!`,
        `Beautiful morning${name}!`,
      ],
      afternoon: [
        `Good afternoon${name}!`,
        `Hello${name}! Hope you're having a wonderful day!`,
        `Happy afternoon${name}!`,
      ],
      evening: [
        `Good evening${name}!`,
        `Evening${name}! Time to relax!`,
        `Hello${name}! Hope you had a great day!`,
      ],
      night: [
        `Good night${name}!`,
        `Evening${name}! Time to wind down!`,
        `Hello${name}! Hope you're well!`,
      ],
    };

    const timeGreetings = greetings[timeOfDay];
    return timeGreetings[Math.floor(Math.random() * timeGreetings.length)];
  };

  const getHealthInsight = () => {
    if (!healthMetrics) return null;

    const insights = [
      "Remember to stay hydrated today - your body will thank you!",
      "You're doing great with your health journey. Keep it up!",
      "Take a moment to breathe deeply. You deserve this peaceful moment.",
      "Your wellness matters. Be gentle with yourself today.",
      "Every small healthy choice you make is a victory!",
      "Listen to your body today - it knows what it needs.",
      "Remember: progress, not perfection. You're doing wonderfully!",
      "Take your medications as prescribed - you're taking great care of yourself!",
    ];

    return insights[Math.floor(Math.random() * insights.length)];
  };

  const getPersonalizedMessage = () => {
    const messages = [
      "I'm here to support you every step of the way.",
      "Let's make today a good day for your health!",
      "Your wellbeing is what matters most.",
      "Remember, I'm always here if you need to chat.",
      "Take things at your own pace today.",
      "You're doing an amazing job taking care of yourself!",
      "Every day is a fresh start for your health.",
      "I'm proud of you for prioritizing your wellness.",
    ];

    return messages[Math.floor(Math.random() * messages.length)];
  };

  const generateWelcomeContent = () => {
    const quote = INSPIRATIONAL_QUOTES[Math.floor(Math.random() * INSPIRATIONAL_QUOTES.length)];
    const image = CALMING_IMAGES[Math.floor(Math.random() * CALMING_IMAGES.length)];
    const music = MUSIC_NOTES[Math.floor(Math.random() * MUSIC_NOTES.length)];

    setWelcomeContent({
      greeting: getGreeting(),
      message: getPersonalizedMessage(),
      quote: quote.quote,
      quoteAuthor: quote.author,
      healthInsight: getHealthInsight(),
      imageUrl: image,
      musicNote: music,
    });
  };

  if (!welcomeContent) {
    return null;
  }

  const TimeIcon = () => {
    const timeOfDay = getTimeOfDay();
    switch (timeOfDay) {
      case "morning":
        return <Sun className="w-6 h-6 text-yellow-500" />;
      case "afternoon":
        return <Cloud className="w-6 h-6 text-blue-400" />;
      case "evening":
        return <Sparkles className="w-6 h-6 text-purple-400" />;
      case "night":
        return <Moon className="w-6 h-6 text-indigo-400" />;
      default:
        return <Sun className="w-6 h-6" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="mb-6"
      data-testid="container-dynamic-welcome"
    >
      <Card className="overflow-hidden">
        {/* Calming Background Image */}
        <div className="relative h-48 overflow-hidden">
          <img
            src={welcomeContent.imageUrl}
            alt="Calming scene"
            className="w-full h-full object-cover"
            data-testid="img-calming-scene"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/50 to-transparent" />
        </div>

        <CardContent className="relative -mt-24 space-y-4">
          {/* Greeting Card */}
          <Card className="bg-card/95 backdrop-blur">
            <CardContent className="pt-6">
              <div className="flex items-start gap-3">
                <TimeIcon />
                <div className="flex-1 space-y-3">
                  <h2 className="text-2xl font-semibold" data-testid="text-greeting">
                    {welcomeContent.greeting}
                  </h2>
                  <p className="text-lg text-muted-foreground" data-testid="text-message">
                    {welcomeContent.message}
                  </p>

                  {/* Health Insight */}
                  {welcomeContent.healthInsight && (
                    <div className="flex items-start gap-2 p-3 bg-primary/5 rounded-lg border border-primary/10">
                      <Heart className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
                      <p className="text-sm" data-testid="text-health-insight">
                        {welcomeContent.healthInsight}
                      </p>
                    </div>
                  )}

                  {/* Inspirational Quote */}
                  {welcomeContent.quote && (
                    <div className="flex items-start gap-2 p-4 bg-muted rounded-lg">
                      <BookOpen className="w-5 h-5 text-muted-foreground mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="italic text-sm leading-relaxed" data-testid="text-quote">
                          "{welcomeContent.quote}"
                        </p>
                        {welcomeContent.quoteAuthor && (
                          <p className="text-xs text-muted-foreground mt-2" data-testid="text-quote-author">
                            — {welcomeContent.quoteAuthor}
                          </p>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Music Note */}
                  {welcomeContent.musicNote && (
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="gap-2">
                        <Music className="w-3 h-3" />
                        <span data-testid="text-music-note">{welcomeContent.musicNote}</span>
                      </Badge>
                      <Badge variant="secondary" className="gap-2">
                        <Smile className="w-3 h-3" />
                        <span>You've got this!</span>
                      </Badge>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </motion.div>
  );
}
