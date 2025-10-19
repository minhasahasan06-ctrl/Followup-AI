import { useState } from "react";
import { ChatMessage } from "@/components/ChatMessage";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Bot } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  isGP?: boolean;
  entities?: Array<{
    text: string;
    type: "medication" | "symptom" | "diagnosis" | "dosage";
  }>;
}

export default function Chat() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant" as const,
      content: "Hello! I'm your AI GP Assistant. I'm here to help you manage your health as an immunocompromised patient. How are you feeling today?",
      timestamp: "10:00 AM",
      isGP: true,
    },
  ]);

  const handleSend = () => {
    if (!message.trim()) return;

    const userMsg = {
      role: "user" as const,
      content: message,
      timestamp: "Just now",
    };

    const aiResponse = {
      role: "assistant" as const,
      content: "I understand your concern. Let me help you with that. Based on your symptoms and medical history, I'd recommend...",
      timestamp: "Just now",
      isGP: true,
      entities: [
        { text: "Fatigue", type: "symptom" as const },
        { text: "Ibuprofen", type: "medication" as const },
      ],
    };

    setMessages([...messages, userMsg, aiResponse]);
    setMessage("");
    console.log("Message sent:", message);
  };

  return (
    <div className="h-full flex flex-col max-w-4xl mx-auto">
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-foreground">
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold">AI GP Assistant</h1>
            <p className="text-sm text-muted-foreground">
              Powered by GPT-4 with Azure Health Analytics
            </p>
          </div>
        </div>
      </div>

      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="border-b">
          <CardTitle className="text-base">Chat with AI GP</CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-0 min-h-0">
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4">
              {messages.map((msg, idx) => (
                <ChatMessage
                  key={idx}
                  role={msg.role}
                  content={msg.content}
                  timestamp={msg.timestamp}
                  isGP={msg.role === "assistant"}
                  entities={msg.entities}
                />
              ))}
            </div>
          </ScrollArea>
          
          <div className="border-t p-4">
            <div className="flex gap-2">
              <Textarea
                placeholder="Describe your symptoms or ask a question..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                className="resize-none"
                rows={3}
                data-testid="input-chat-message"
              />
              <Button
                size="icon"
                onClick={handleSend}
                disabled={!message.trim()}
                data-testid="button-send-message"
                className="h-auto"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              This AI assistant provides general health information. Always consult your doctor for medical decisions.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
