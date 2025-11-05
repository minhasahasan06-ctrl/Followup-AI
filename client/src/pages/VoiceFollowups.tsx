import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { VoiceRecorder } from "@/components/VoiceRecorder";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Mic,
  Calendar,
  TrendingUp,
  History,
  Heart,
  Smile,
  Meh,
  Frown,
  AlertCircle,
} from "lucide-react";
import { format } from "date-fns";

interface VoiceFollowup {
  id: string;
  audioFileUrl: string;
  audioDuration: number;
  transcription: string;
  extractedMood: string;
  moodScore: number;
  extractedSymptoms: Array<{ symptom: string; severity: string; confidence: number }>;
  medicationAdherence: Array<{ medication: string; taken: boolean; time?: string }>;
  sentimentScore: number;
  empathyLevel: string;
  concernsRaised: boolean;
  concernsSummary: string | null;
  aiResponse: string;
  conversationSummary: string;
  needsFollowup: boolean;
  followupReason: string | null;
  recommendedActions: string[];
  createdAt: Date;
}

export default function VoiceFollowups() {
  const [activeTab, setActiveTab] = useState("record");

  const { data: recentFollowups, refetch: refetchFollowups } = useQuery<VoiceFollowup[]>({
    queryKey: ["/api/voice-followup/recent"],
  });

  const handleUploadSuccess = () => {
    refetchFollowups();
    setActiveTab("history");
  };

  const getMoodIcon = (moodScore: number) => {
    if (moodScore > 0.3) return <Smile className="w-5 h-5 text-green-600 dark:text-green-400" />;
    if (moodScore < -0.3) return <Frown className="w-5 h-5 text-red-600 dark:text-red-400" />;
    return <Meh className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />;
  };

  const getEmpathyBadge = (level: string) => {
    const colors: Record<string, string> = {
      supportive: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
      empathetic: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
      encouraging: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
      concerned: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
    };

    return (
      <Badge className={colors[level.toLowerCase()] || ""} variant="outline">
        {level}
      </Badge>
    );
  };

  const totalCheckIns = recentFollowups?.length || 0;
  const avgMoodScore =
    recentFollowups && recentFollowups.length > 0
      ? (
          recentFollowups.reduce((sum, f) => sum + parseFloat(f.moodScore.toString()), 0) /
          recentFollowups.length
        ).toFixed(2)
      : "0.00";
  const concernsCount =
    recentFollowups?.filter((f) => f.concernsRaised).length || 0;

  return (
    <div className="container mx-auto py-8 max-w-6xl" data-testid="page-voice-followups">
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-pink-400 to-red-500 text-white">
            <Mic className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-4xl font-bold" data-testid="heading-voice-followups">
              Voice Follow-ups
            </h1>
            <p className="text-muted-foreground" data-testid="text-description">
              Quick 1-minute voice logs - more like having someone who cares check in
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Voice Logs</CardTitle>
            <History className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-total-logs">
              {totalCheckIns}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {recentFollowups && recentFollowups.length > 0
                ? `Last: ${format(new Date(recentFollowups[0].createdAt), "MMM d")}`
                : "No logs yet"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average Mood</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-avg-mood">
              {avgMoodScore}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {parseFloat(avgMoodScore) > 0 ? "Positive overall" : parseFloat(avgMoodScore) < 0 ? "Needs support" : "Neutral"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Concerns Raised</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-concerns-count">
              {concernsCount}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {concernsCount === 0 ? "All clear" : "Needs attention"}
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="record" data-testid="tab-record">
            <Mic className="w-4 h-4 mr-2" />
            Record
          </TabsTrigger>
          <TabsTrigger value="history" data-testid="tab-history">
            <Calendar className="w-4 h-4 mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="record">
          <VoiceRecorder onUploadSuccess={handleUploadSuccess} />
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Your Voice Log History</CardTitle>
              <CardDescription>All your recorded voice follow-ups</CardDescription>
            </CardHeader>
            <CardContent>
              {recentFollowups && recentFollowups.length > 0 ? (
                <ScrollArea className="h-[700px]">
                  <div className="space-y-4">
                    {recentFollowups.map((followup) => (
                      <div
                        key={followup.id}
                        className="border rounded-lg p-4 space-y-3"
                        data-testid={`followup-card-${followup.id}`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-muted-foreground">
                              {format(new Date(followup.createdAt), "MMM d, h:mm a")}
                            </span>
                            <Badge variant="outline" className="text-xs">
                              {followup.audioDuration}s
                            </Badge>
                          </div>
                          {getMoodIcon(followup.moodScore)}
                        </div>

                        <div className="space-y-2">
                          <div>
                            <p className="text-sm font-medium text-muted-foreground mb-1">
                              Transcription:
                            </p>
                            <p className="text-sm italic">{followup.transcription}</p>
                          </div>

                          <div>
                            <p className="text-sm font-medium text-muted-foreground mb-1">
                              Response:
                            </p>
                            <p className="text-sm leading-relaxed">{followup.aiResponse}</p>
                          </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-2">
                          {getEmpathyBadge(followup.empathyLevel)}
                          {followup.concernsRaised && (
                            <Badge variant="outline" className="text-xs">
                              <AlertCircle className="w-3 h-3 mr-1" />
                              Concerns Noted
                            </Badge>
                          )}
                          {followup.needsFollowup && (
                            <Badge variant="outline" className="text-xs">
                              Follow-up Required
                            </Badge>
                          )}
                          {followup.extractedMood && (
                            <Badge variant="secondary" className="text-xs capitalize">
                              {followup.extractedMood}
                            </Badge>
                          )}
                        </div>

                        {followup.extractedSymptoms && followup.extractedSymptoms.length > 0 && (
                          <div className="p-3 bg-muted rounded-lg text-sm">
                            <p className="font-medium mb-2">Symptoms:</p>
                            <div className="flex flex-wrap gap-1">
                              {followup.extractedSymptoms.map((symptom, idx) => (
                                <Badge key={idx} variant="secondary" className="text-xs">
                                  {symptom.symptom} ({symptom.severity})
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {followup.medicationAdherence &&
                          followup.medicationAdherence.length > 0 && (
                            <div className="p-3 bg-muted rounded-lg text-sm">
                              <p className="font-medium mb-2">Medication Adherence:</p>
                              <ul className="space-y-1">
                                {followup.medicationAdherence.map((med, idx) => (
                                  <li key={idx} className="flex items-center gap-2">
                                    <div
                                      className={`h-2 w-2 rounded-full ${
                                        med.taken ? "bg-green-500" : "bg-red-500"
                                      }`}
                                    />
                                    <span>
                                      {med.medication} - {med.taken ? "Taken" : "Missed"}
                                      {med.time && ` at ${med.time}`}
                                    </span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                        {followup.recommendedActions && followup.recommendedActions.length > 0 && (
                          <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg text-sm">
                            <p className="font-medium mb-2">Recommended Actions:</p>
                            <ul className="list-disc list-inside space-y-1">
                              {followup.recommendedActions.map((action, idx) => (
                                <li key={idx} className="text-muted-foreground">
                                  {action}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-12">
                  <Mic className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
                  <p className="text-muted-foreground mb-4">No voice logs yet</p>
                  <p className="text-sm text-muted-foreground">
                    Record your first voice message to get started
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
