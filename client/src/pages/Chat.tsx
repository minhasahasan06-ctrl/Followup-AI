import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { ChatMessage } from "@/components/ChatMessage";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Bot } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAuth } from "@/hooks/useAuth";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { ChatMessage as ChatMessageType } from "@shared/schema";

export default function Chat() {
  const [message, setMessage] = useState("");
  const { user } = useAuth();
  const isDoctor = user?.role === "doctor";
  const agentType = isDoctor ? "lysa" : "clona";
  const agentName = isDoctor ? "Assistant Lysa" : "Agent Clona";

  const { data: chatMessages, isLoading } = useQuery<ChatMessageType[]>({
    queryKey: ["/api/chat/messages", { agent: agentType }],
  });

  const sendMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      return await apiRequest("POST", "/api/chat/send", { content, agentType });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/chat/messages"] });
      setMessage("");
    },
  });

  const handleSend = () => {
    if (!message.trim() || sendMessageMutation.isPending) return;
    sendMessageMutation.mutate(message);
  };

  return (
    <div className="h-full flex flex-col max-w-4xl mx-auto">
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <div className={`flex h-10 w-10 items-center justify-center rounded-full ${
            isDoctor ? "bg-accent" : "bg-primary"
          } text-primary-foreground`}>
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold">{agentName}</h1>
            <p className="text-sm text-muted-foreground">
              Powered by GPT-4 {isDoctor ? "- Clinical Insights" : "- Health Support"}
            </p>
          </div>
        </div>
      </div>

      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="border-b">
          <CardTitle className="text-base">Chat with {agentName}</CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-0 min-h-0">
          <ScrollArea className="flex-1 p-4">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
              </div>
            ) : (
              <div className="space-y-4">
                {chatMessages && chatMessages.length > 0 ? (
                  chatMessages.map((msg) => (
                    <ChatMessage
                      key={msg.id}
                      role={msg.role as "user" | "assistant"}
                      content={msg.content}
                      timestamp={new Date(msg.createdAt).toLocaleTimeString()}
                      isGP={msg.role === "assistant"}
                      entities={msg.medicalEntities}
                    />
                  ))
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <p>Start a conversation with {agentName}</p>
                  </div>
                )}
                {sendMessageMutation.isPending && (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    <span>{agentName} is typing...</span>
                  </div>
                )}
              </div>
            )}
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
              {isDoctor 
                ? "AI-powered clinical insights. Always use professional judgment."
                : "This AI assistant provides general health information. Always consult your doctor for medical decisions."}
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
