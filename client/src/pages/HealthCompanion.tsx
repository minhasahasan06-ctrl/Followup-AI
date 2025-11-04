import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Heart,
  MessageCircle,
  Sparkles,
  Calendar,
  TrendingUp,
  Send,
  Smile,
  Meh,
  Frown,
  Clock,
  Activity,
} from "lucide-react";
import { format } from "date-fns";

interface CheckIn {
  id: string;
  patientId: string;
  checkInType: string;
  userInput: string;
  companionResponse: string;
  empathyLevel: string;
  sentimentScore: number;
  concernsRaised: boolean;
  needsFollowup: boolean;
  followupReason: string | null;
  conversationSummary: string;
  extractedData: any;
  createdAt: Date;
}

interface Engagement {
  id: string;
  patientId: string;
  preferredCheckInTime: string | null;
  checkInFrequency: string;
  companionPersonality: string;
  preferredTone: string;
  currentStreak: number;
  longestStreak: number;
  totalCheckIns: number;
  lastCheckIn: Date | null;
  reminderEnabled: boolean;
}

export default function HealthCompanion() {
  const { toast } = useToast();
  const [message, setMessage] = useState("");
  const [checkInType, setCheckInType] = useState("spontaneous");
  const [activeTab, setActiveTab] = useState("check-in");

  const { data: recentCheckIns, isLoading: checkInsLoading } = useQuery<CheckIn[]>({
    queryKey: ["/api/companion/check-ins/recent"],
  });

  const { data: engagement, isLoading: engagementLoading } = useQuery<Engagement>({
    queryKey: ["/api/companion/engagement"],
  });

  const { data: suggestedPrompt } = useQuery<{ prompt: string }>({
    queryKey: ["/api/companion/prompt", { type: checkInType }],
    enabled: activeTab === "check-in",
  });

  const checkInMutation = useMutation({
    mutationFn: async (data: { userInput: string; checkInType: string }) => {
      return await apiRequest("POST", "/api/companion/check-in", data);
    },
    onSuccess: (response: any) => {
      queryClient.invalidateQueries({ queryKey: ["/api/companion/check-ins"] });
      queryClient.invalidateQueries({ queryKey: ["/api/companion/check-ins/recent"] });
      queryClient.invalidateQueries({ queryKey: ["/api/companion/engagement"] });
      setMessage("");
      
      toast({
        title: "Check-in Complete",
        description: "Your health companion has recorded your update.",
      });
    },
    onError: () => {
      toast({
        title: "Check-in Failed",
        description: "Unable to process your check-in. Please try again.",
        variant: "destructive",
      });
    },
  });

  const updateEngagementMutation = useMutation({
    mutationFn: async (data: Partial<Engagement>) => {
      return await apiRequest("PATCH", "/api/companion/engagement", data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/companion/engagement"] });
      toast({
        title: "Settings Updated",
        description: "Your companion preferences have been saved.",
      });
    },
    onError: () => {
      toast({
        title: "Update Failed",
        description: "Unable to update settings. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleCheckIn = () => {
    if (!message.trim() || checkInMutation.isPending) return;
    checkInMutation.mutate({
      userInput: message,
      checkInType,
    });
  };

  const getSentimentIcon = (score: number) => {
    if (score > 0.3) return <Smile className="w-5 h-5 text-green-600 dark:text-green-400" />;
    if (score < -0.3) return <Frown className="w-5 h-5 text-red-600 dark:text-red-400" />;
    return <Meh className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />;
  };

  const getEmpathyBadge = (level: string) => {
    const colors: Record<string, string> = {
      supportive: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
      empathetic: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
      encouraging: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
    };

    return (
      <Badge className={colors[level.toLowerCase()] || ""} variant="outline">
        {level}
      </Badge>
    );
  };

  const useSuggestedPrompt = () => {
    if (suggestedPrompt?.prompt) {
      setMessage(suggestedPrompt.prompt);
    }
  };

  return (
    <div className="container mx-auto py-8 max-w-5xl" data-testid="page-health-companion">
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-green-400 to-teal-500 text-white">
            <Heart className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-4xl font-bold" data-testid="heading-companion">
              Health Companion
            </h1>
            <p className="text-muted-foreground" data-testid="text-description">
              Your personal AI health companion for daily check-ins and support
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current Streak</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-current-streak">
              {engagementLoading ? "..." : `${engagement?.currentStreak || 0} days`}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Longest: {engagement?.longestStreak || 0} days
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Check-ins</CardTitle>
            <MessageCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-total-checkins">
              {engagementLoading ? "..." : engagement?.totalCheckIns || 0}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {engagement?.lastCheckIn
                ? `Last: ${format(new Date(engagement.lastCheckIn), "MMM d, h:mm a")}`
                : "No check-ins yet"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Companion Style</CardTitle>
            <Sparkles className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold capitalize" data-testid="text-personality">
              {engagementLoading ? "..." : engagement?.companionPersonality || "Empathetic"}
            </div>
            <p className="text-xs text-muted-foreground mt-2 capitalize">
              Tone: {engagement?.preferredTone || "warm"}
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="check-in" data-testid="tab-checkin">
            Check-in
          </TabsTrigger>
          <TabsTrigger value="history" data-testid="tab-history">
            History
          </TabsTrigger>
          <TabsTrigger value="settings" data-testid="tab-settings">
            Settings
          </TabsTrigger>
        </TabsList>

        <TabsContent value="check-in" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Daily Check-in</CardTitle>
              <CardDescription>
                Share how you're feeling today - your companion is here to listen and support
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">Check-in Type:</label>
                <div className="flex gap-2">
                  <Button
                    variant={checkInType === "morning" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setCheckInType("morning")}
                    data-testid="button-type-morning"
                  >
                    <Calendar className="w-3 h-3 mr-1" />
                    Morning
                  </Button>
                  <Button
                    variant={checkInType === "evening" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setCheckInType("evening")}
                    data-testid="button-type-evening"
                  >
                    <Clock className="w-3 h-3 mr-1" />
                    Evening
                  </Button>
                  <Button
                    variant={checkInType === "spontaneous" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setCheckInType("spontaneous")}
                    data-testid="button-type-spontaneous"
                  >
                    <MessageCircle className="w-3 h-3 mr-1" />
                    Anytime
                  </Button>
                </div>
              </div>

              {suggestedPrompt?.prompt && (
                <div className="p-3 bg-muted rounded-lg">
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <p className="text-sm text-muted-foreground">Suggested prompt:</p>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={useSuggestedPrompt}
                      data-testid="button-use-prompt"
                    >
                      Use this
                    </Button>
                  </div>
                  <p className="text-sm italic">{suggestedPrompt.prompt}</p>
                </div>
              )}

              <div className="relative">
                <Textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleCheckIn();
                    }
                  }}
                  placeholder="How are you feeling today? Share anything - your mood, energy, symptoms, meals, medications, or just what's on your mind..."
                  className="min-h-32 resize-none"
                  data-testid="textarea-checkin"
                />
              </div>

              <div className="flex justify-end">
                <Button
                  onClick={handleCheckIn}
                  disabled={!message.trim() || checkInMutation.isPending}
                  data-testid="button-submit-checkin"
                >
                  {checkInMutation.isPending ? (
                    <>Processing...</>
                  ) : (
                    <>
                      <Send className="w-4 h-4 mr-2" />
                      Check In
                    </>
                  )}
                </Button>
              </div>

              {checkInMutation.isSuccess && checkInMutation.data && (
                <Card className="border-green-200 dark:border-green-800">
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <Heart className="w-5 h-5 text-green-600 dark:text-green-400" />
                      <CardTitle className="text-base">Your Companion's Response</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="text-sm leading-relaxed" data-testid="text-companion-response">
                      {checkInMutation.data.response}
                    </p>
                    
                    <div className="flex flex-wrap items-center gap-2">
                      {getEmpathyBadge(checkInMutation.data.empathyLevel)}
                      {checkInMutation.data.concernsRaised && (
                        <Badge variant="outline" className="bg-yellow-50 dark:bg-yellow-950">
                          Concerns Noted
                        </Badge>
                      )}
                      {checkInMutation.data.needsFollowup && (
                        <Badge variant="outline" className="bg-blue-50 dark:bg-blue-950">
                          Follow-up Recommended
                        </Badge>
                      )}
                    </div>

                    {checkInMutation.data.conversationSummary && (
                      <div className="p-3 bg-muted rounded-lg text-sm">
                        <p className="font-medium mb-1">Summary:</p>
                        <p className="text-muted-foreground">
                          {checkInMutation.data.conversationSummary}
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Recent Check-ins</CardTitle>
              <CardDescription>Your conversation history with your health companion</CardDescription>
            </CardHeader>
            <CardContent>
              {checkInsLoading ? (
                <p className="text-sm text-muted-foreground">Loading check-ins...</p>
              ) : recentCheckIns && recentCheckIns.length > 0 ? (
                <ScrollArea className="h-[600px]">
                  <div className="space-y-4">
                    {recentCheckIns.map((checkIn) => (
                      <div
                        key={checkIn.id}
                        className="border rounded-lg p-4 space-y-3"
                        data-testid={`checkin-card-${checkIn.id}`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="capitalize">
                              {checkIn.checkInType}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {format(new Date(checkIn.createdAt), "MMM d, h:mm a")}
                            </span>
                          </div>
                          {getSentimentIcon(checkIn.sentimentScore)}
                        </div>

                        <div className="space-y-2">
                          <div>
                            <p className="text-sm font-medium text-muted-foreground mb-1">You:</p>
                            <p className="text-sm">{checkIn.userInput}</p>
                          </div>

                          <div>
                            <p className="text-sm font-medium text-muted-foreground mb-1">Companion:</p>
                            <p className="text-sm leading-relaxed">{checkIn.companionResponse}</p>
                          </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-2">
                          {getEmpathyBadge(checkIn.empathyLevel)}
                          {checkIn.concernsRaised && (
                            <Badge variant="outline" className="text-xs">
                              Concerns Noted
                            </Badge>
                          )}
                          {checkIn.needsFollowup && (
                            <Badge variant="outline" className="text-xs">
                              Follow-up: {checkIn.followupReason}
                            </Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-8">
                  <MessageCircle className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
                  <p className="text-muted-foreground mb-4">No check-ins yet</p>
                  <Button onClick={() => setActiveTab("check-in")} data-testid="button-start-checkin">
                    <Heart className="w-4 h-4 mr-2" />
                    Start Your First Check-in
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Companion Settings</CardTitle>
              <CardDescription>Customize your health companion's personality and behavior</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Companion Personality</label>
                <div className="grid grid-cols-3 gap-2">
                  {["empathetic", "motivating", "practical"].map((personality) => (
                    <Button
                      key={personality}
                      variant={
                        engagement?.companionPersonality === personality ? "default" : "outline"
                      }
                      onClick={() =>
                        updateEngagementMutation.mutate({ companionPersonality: personality })
                      }
                      disabled={updateEngagementMutation.isPending}
                      className="capitalize"
                      data-testid={`button-personality-${personality}`}
                    >
                      {personality}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Preferred Tone</label>
                <div className="grid grid-cols-3 gap-2">
                  {["warm", "professional", "casual"].map((tone) => (
                    <Button
                      key={tone}
                      variant={engagement?.preferredTone === tone ? "default" : "outline"}
                      onClick={() => updateEngagementMutation.mutate({ preferredTone: tone })}
                      disabled={updateEngagementMutation.isPending}
                      className="capitalize"
                      data-testid={`button-tone-${tone}`}
                    >
                      {tone}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Check-in Frequency</label>
                <div className="grid grid-cols-3 gap-2">
                  {["daily", "twice_daily", "flexible"].map((frequency) => (
                    <Button
                      key={frequency}
                      variant={engagement?.checkInFrequency === frequency ? "default" : "outline"}
                      onClick={() =>
                        updateEngagementMutation.mutate({ checkInFrequency: frequency })
                      }
                      disabled={updateEngagementMutation.isPending}
                      className="capitalize"
                      data-testid={`button-frequency-${frequency}`}
                    >
                      {frequency.replace("_", " ")}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="pt-4 border-t">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Daily Reminders</p>
                    <p className="text-sm text-muted-foreground">
                      Get reminded to check in with your companion
                    </p>
                  </div>
                  <Button
                    variant={engagement?.reminderEnabled ? "default" : "outline"}
                    onClick={() =>
                      updateEngagementMutation.mutate({
                        reminderEnabled: !engagement?.reminderEnabled,
                      })
                    }
                    disabled={updateEngagementMutation.isPending}
                    data-testid="button-toggle-reminders"
                  >
                    {engagement?.reminderEnabled ? "Enabled" : "Disabled"}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
