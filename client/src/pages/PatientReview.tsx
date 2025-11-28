import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useParams } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import {
  Activity,
  Heart,
  Droplet,
  TrendingUp,
  Calendar,
  Bot,
  AlertCircle,
  CheckCircle,
  MessageSquare,
  Clock,
  FileText,
  Download,
  Bell,
  Brain,
} from "lucide-react";
import { startOfWeek, endOfWeek, subWeeks, format } from "date-fns";
import type { User, DailyFollowup, Medication, ChatMessage } from "@shared/schema";
import { PatientAIAlerts } from "@/components/patient-ai-alerts";
import { PatientMLTools } from "@/components/patient-ml-tools";

interface ChatSession {
  id: string;
  patientId: string;
  agentType: string;
  sessionTitle: string | null;
  startedAt: Date;
  endedAt: Date | null;
  messageCount: number | null;
  symptomsDiscussed: string[] | null;
  recommendations: string[] | null;
  healthInsights: any;
  aiSummary: string | null;
  createdAt: Date;
  updatedAt: Date;
  doctorNotes: string | null;
}

export default function PatientReview() {
  const params = useParams();
  const patientId = params.id;
  const { toast } = useToast();
  
  // PDF Report state
  const [selectedWeekOffset, setSelectedWeekOffset] = useState(0); // 0 = current week, 1 = last week, etc.
  const [referenceDate] = useState(() => new Date()); // FIX: Memoized to prevent inconsistent week ranges

  const { data: patient } = useQuery<User>({
    queryKey: [`/api/doctor/patients/${patientId}`],
  });

  const { data: followups } = useQuery<DailyFollowup[]>({
    queryKey: [`/api/doctor/patients/${patientId}/followups`],
  });

  const { data: medications } = useQuery<Medication[]>({
    queryKey: [`/api/doctor/patients/${patientId}/medications`],
  });

  const { data: aiInsights } = useQuery<ChatMessage[]>({
    queryKey: ["/api/chat/messages", { agent: "lysa", patientId }],
  });

  const { data: chatSessions } = useQuery<ChatSession[]>({
    queryKey: [`/api/doctor/patient-sessions/${patientId}`],
  });

  // PDF Generation Mutation
  const generatePDF = useMutation({
    mutationFn: async () => {
      const weekStart = startOfWeek(subWeeks(referenceDate, selectedWeekOffset), { weekStartsOn: 1 }); // Monday
      const weekEnd = endOfWeek(subWeeks(referenceDate, selectedWeekOffset), { weekStartsOn: 1 }); // Sunday

      // FIX: Handle blob response, not JSON
      const res = await fetch(`/api/v1/symptom-journal/generate-weekly-pdf/${patientId}?${new URLSearchParams({
        week_start: weekStart.toISOString(),
        week_end: weekEnd.toISOString(),
      })}`, {
        method: "POST",
        credentials: 'include', // Include session cookies
      });

      if (!res.ok) {
        const error = await res.text();
        throw new Error(error || "Failed to generate PDF");
      }

      // FIX: Backend streams PDF blob, not JSON
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      
      // Trigger download
      const link = document.createElement('a');
      link.href = url;
      link.download = `symptom-report-${format(weekStart, 'yyyy-MM-dd')}-to-${format(weekEnd, 'yyyy-MM-dd')}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      return { success: true };
    },
    onSuccess: () => {
      toast({
        title: "PDF Generated",
        description: "Weekly report has been downloaded successfully.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Generation Failed",
        description: error.message || "Failed to generate PDF report. Please try again.",
        variant: "destructive",
      });
    },
  });

  const getInitials = (firstName?: string | null, lastName?: string | null) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  return (
    <div className="space-y-6">
      {patient && (
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <Avatar className="h-16 w-16">
                <AvatarFallback className="bg-primary text-primary-foreground text-xl">
                  {getInitials(patient.firstName, patient.lastName)}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1">
                <h1 className="text-2xl font-semibold" data-testid="text-patient-name">
                  {patient.firstName} {patient.lastName}
                </h1>
                <p className="text-muted-foreground">{patient.email}</p>
              </div>
              <div className="flex gap-2">
                <Badge variant="secondary">Patient ID: {patient.id.slice(0, 8)}</Badge>
                <Badge variant="secondary">
                  <Activity className="h-3 w-3 mr-1" />
                  Active
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Tabs defaultValue="ai-alerts" className="space-y-4">
            <TabsList className="flex flex-wrap gap-1">
              <TabsTrigger value="ai-alerts" data-testid="tab-ai-alerts">
                <Bell className="h-4 w-4 mr-2" />
                AI Alerts
              </TabsTrigger>
              <TabsTrigger value="ml-tools" data-testid="tab-ml-tools">
                <Brain className="h-4 w-4 mr-2" />
                ML Tools
              </TabsTrigger>
              <TabsTrigger value="timeline" data-testid="tab-timeline">
                <Calendar className="h-4 w-4 mr-2" />
                Timeline
              </TabsTrigger>
              <TabsTrigger value="vitals" data-testid="tab-vitals">
                <Heart className="h-4 w-4 mr-2" />
                Vitals
              </TabsTrigger>
              <TabsTrigger value="medications" data-testid="tab-medications">
                <Droplet className="h-4 w-4 mr-2" />
                Medications
              </TabsTrigger>
              <TabsTrigger value="sessions" data-testid="tab-sessions">
                <MessageSquare className="h-4 w-4 mr-2" />
                Sessions
              </TabsTrigger>
              <TabsTrigger value="reports" data-testid="tab-reports">
                <FileText className="h-4 w-4 mr-2" />
                Reports
              </TabsTrigger>
            </TabsList>

            {/* AI Health Alerts Tab */}
            <TabsContent value="ai-alerts">
              {patientId && (
                <PatientAIAlerts 
                  patientId={patientId} 
                  patientName={patient ? `${patient.firstName} ${patient.lastName}` : undefined}
                />
              )}
            </TabsContent>

            {/* ML Prediction Tools Tab */}
            <TabsContent value="ml-tools">
              {patientId && (
                <PatientMLTools 
                  patientId={patientId} 
                  patientName={patient ? `${patient.firstName} ${patient.lastName}` : undefined}
                />
              )}
            </TabsContent>

            <TabsContent value="timeline">
              <Card>
                <CardHeader>
                  <CardTitle>Patient Timeline</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[600px] pr-4">
                    <div className="space-y-4">
                      {followups && followups.length > 0 ? (
                        followups.map((followup) => (
                          <Card key={followup.id} className="border-l-4 border-l-primary">
                            <CardContent className="p-4">
                              <div className="flex items-start justify-between mb-2">
                                <div>
                                  <p className="font-medium">
                                    {new Date(followup.date).toLocaleDateString()}
                                  </p>
                                  <p className="text-sm text-muted-foreground">Daily Follow-up</p>
                                </div>
                                {followup.completed ? (
                                  <Badge variant="secondary">
                                    <CheckCircle className="h-3 w-3 mr-1" />
                                    Completed
                                  </Badge>
                                ) : (
                                  <Badge variant="secondary">
                                    <AlertCircle className="h-3 w-3 mr-1" />
                                    Pending
                                  </Badge>
                                )}
                              </div>
                              <div className="grid grid-cols-2 gap-3 mt-3">
                                {followup.heartRate && (
                                  <div className="text-sm">
                                    <span className="text-muted-foreground">Heart Rate: </span>
                                    <span className="font-medium">{followup.heartRate} bpm</span>
                                  </div>
                                )}
                                {followup.temperature && (
                                  <div className="text-sm">
                                    <span className="text-muted-foreground">Temp: </span>
                                    <span className="font-medium">{followup.temperature}°F</span>
                                  </div>
                                )}
                                {followup.oxygenSaturation && (
                                  <div className="text-sm">
                                    <span className="text-muted-foreground">SpO2: </span>
                                    <span className="font-medium">{followup.oxygenSaturation}%</span>
                                  </div>
                                )}
                                {followup.stepsCount && (
                                  <div className="text-sm">
                                    <span className="text-muted-foreground">Steps: </span>
                                    <span className="font-medium">{followup.stepsCount}</span>
                                  </div>
                                )}
                              </div>
                            </CardContent>
                          </Card>
                        ))
                      ) : (
                        <div className="text-center py-8 text-muted-foreground">
                          <Calendar className="h-12 w-12 mx-auto mb-3 opacity-50" />
                          <p>No follow-up data available</p>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="vitals">
              <Card>
                <CardHeader>
                  <CardTitle>Vital Signs Trends</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-2">
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-muted-foreground">Heart Rate</span>
                          <Heart className="h-4 w-4 text-muted-foreground" />
                        </div>
                        <p className="text-2xl font-bold">72 bpm</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          <TrendingUp className="h-3 w-3 inline mr-1" />
                          Normal range
                        </p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-muted-foreground">Blood Pressure</span>
                          <Activity className="h-4 w-4 text-muted-foreground" />
                        </div>
                        <p className="text-2xl font-bold">120/80</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          <TrendingUp className="h-3 w-3 inline mr-1" />
                          Optimal
                        </p>
                      </CardContent>
                    </Card>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="medications">
              <Card>
                <CardHeader>
                  <CardTitle>Current Medications</CardTitle>
                </CardHeader>
                <CardContent>
                  {medications && medications.length > 0 ? (
                    <div className="space-y-3">
                      {medications.map((med) => (
                        <Card key={med.id}>
                          <CardContent className="p-4">
                            <div className="flex items-start justify-between">
                              <div>
                                <p className="font-medium">{med.name}</p>
                                <p className="text-sm text-muted-foreground">
                                  {med.dosage} • {med.frequency}
                                </p>
                                {med.aiSuggestion && (
                                  <p className="text-xs text-muted-foreground mt-2 italic">
                                    AI: {med.aiSuggestion}
                                  </p>
                                )}
                              </div>
                              {med.isOTC && <Badge variant="secondary">OTC</Badge>}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Droplet className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>No medications recorded</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="sessions">
              <Card>
                <CardHeader>
                  <CardTitle>Chat Sessions (Last Month)</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Recent conversations with Agent Clona
                  </p>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[600px] pr-4">
                    <div className="space-y-4">
                      {chatSessions && chatSessions.length > 0 ? (
                        chatSessions.map((session) => (
                          <Card key={session.id} className="border-l-4 border-l-chart-2">
                            <CardContent className="p-4">
                              <div className="flex items-start justify-between mb-3">
                                <div>
                                  <p className="font-medium mb-1">
                                    {session.sessionTitle || "Untitled Session"}
                                  </p>
                                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                    <Clock className="h-3 w-3" />
                                    {new Date(session.startedAt).toLocaleDateString('en-US', {
                                      month: 'short',
                                      day: 'numeric',
                                      year: 'numeric',
                                      hour: '2-digit',
                                      minute: '2-digit'
                                    })}
                                  </div>
                                </div>
                                {session.messageCount && (
                                  <Badge variant="secondary">
                                    {session.messageCount} messages
                                  </Badge>
                                )}
                              </div>

                              {session.symptomsDiscussed && session.symptomsDiscussed.length > 0 && (
                                <div className="mb-3">
                                  <p className="text-xs font-medium text-muted-foreground mb-1">
                                    Symptoms Discussed:
                                  </p>
                                  <div className="flex flex-wrap gap-1">
                                    {session.symptomsDiscussed.map((symptom, idx) => (
                                      <Badge key={idx} variant="outline" className="text-xs">
                                        {symptom}
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {session.aiSummary && (
                                <div className="mt-3 pt-3 border-t">
                                  <div className="flex items-center gap-2 mb-2">
                                    <FileText className="h-3 w-3 text-muted-foreground" />
                                    <p className="text-xs font-medium text-muted-foreground">
                                      AI Summary:
                                    </p>
                                  </div>
                                  <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                                    {session.aiSummary}
                                  </p>
                                </div>
                              )}
                            </CardContent>
                          </Card>
                        ))
                      ) : (
                        <div className="text-center py-8 text-muted-foreground">
                          <MessageSquare className="h-12 w-12 mx-auto mb-3 opacity-50" />
                          <p>No chat sessions in the last month</p>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="reports">
              <Card>
                <CardHeader>
                  <CardTitle>Weekly Symptom Journal Reports</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Generate comprehensive PDF reports of patient symptom tracking
                  </p>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Week Selection */}
                  <div>
                    <label className="text-sm font-medium mb-2 block">Select Week</label>
                    <div className="grid grid-cols-2 gap-2">
                      {[0, 1, 2, 3, 4, 5, 6, 7].map((offset) => {
                        const weekStart = startOfWeek(subWeeks(referenceDate, offset), { weekStartsOn: 1 });
                        const weekEnd = endOfWeek(subWeeks(referenceDate, offset), { weekStartsOn: 1 });
                        return (
                          <Button
                            key={offset}
                            variant={selectedWeekOffset === offset ? "default" : "outline"}
                            onClick={() => setSelectedWeekOffset(offset)}
                            className="justify-start"
                            data-testid={`button-week-${offset}`}
                          >
                            <Calendar className="mr-2 h-4 w-4" />
                            {offset === 0 ? "This Week" : offset === 1 ? "Last Week" : `${offset} Weeks Ago`}
                            <div className="ml-auto text-xs opacity-75">
                              {format(weekStart, "MMM d")} - {format(weekEnd, "MMM d")}
                            </div>
                          </Button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Generate Button */}
                  <div className="flex items-center gap-4 p-4 border rounded-lg bg-muted/50">
                    <FileText className="h-8 w-8 text-muted-foreground" />
                    <div className="flex-1">
                      <h4 className="font-medium">Generate PDF Report</h4>
                      <p className="text-sm text-muted-foreground">
                        Create a comprehensive weekly report including measurements, trends, and AI observations
                      </p>
                    </div>
                    <Button
                      onClick={() => generatePDF.mutate()}
                      disabled={generatePDF.isPending}
                      data-testid="button-generate-pdf"
                    >
                      {generatePDF.isPending ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Download className="mr-2 h-4 w-4" />
                          Generate PDF
                        </>
                      )}
                    </Button>
                  </div>

                  {/* Info Card */}
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">What's Included in the Report</h4>
                    <ul className="space-y-1 text-sm text-muted-foreground">
                      <li>• Patient symptom measurements with images</li>
                      <li>• Color and area change tracking</li>
                      <li>• Respiratory rate trends (if available)</li>
                      <li>• AI-generated observations and alerts</li>
                      <li>• Week-over-week comparison data</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <Card className="bg-gradient-to-br from-accent/20 to-accent/5">
            <CardHeader>
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent">
                  <Bot className="h-4 w-4" />
                </div>
                <CardTitle className="text-base">Assistant Lysa Insights</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px]">
                <div className="space-y-3">
                  <Card>
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <AlertCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-sm font-medium mb-1">Medication Adherence</p>
                          <p className="text-xs text-muted-foreground">
                            Patient shows excellent adherence with 95% completion rate over the past 30 days.
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <TrendingUp className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-sm font-medium mb-1">Positive Trend</p>
                          <p className="text-xs text-muted-foreground">
                            Vital signs trending positively. Heart rate variability improving.
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <Activity className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-sm font-medium mb-1">Activity Level</p>
                          <p className="text-xs text-muted-foreground">
                            Daily step count averaging 4,200 steps. Consider encouraging increased gentle activity.
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </ScrollArea>

              <Button className="w-full mt-4" variant="outline" data-testid="button-chat-lysa">
                <Bot className="h-4 w-4 mr-2" />
                Chat with Assistant Lysa
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
