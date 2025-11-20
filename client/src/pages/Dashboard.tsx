import { HealthMetricCard } from "@/components/HealthMetricCard";
import { MedicationCard } from "@/components/MedicationCard";
import { FollowUpCard } from "@/components/FollowUpCard";
import { ReminderCard } from "@/components/ReminderCard";
import { EmergencyAlert } from "@/components/EmergencyAlert";
import DynamicWelcome from "@/components/DynamicWelcome";
import { LegalDisclaimer } from "@/components/LegalDisclaimer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Heart, Activity, Droplet, Moon, TrendingUp, Calendar, CheckCircle, Brain, Video, Eye, Hand, Smile, Play, Wind, Palette, Zap, Users } from "lucide-react";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
import { Link } from "wouter";
import type { DailyFollowup, Medication, DynamicTask, BehavioralInsight } from "@shared/schema";

export default function Dashboard() {
  const [showEmergency, setShowEmergency] = useState(false);
  const { user } = useAuth();

  const { data: todayFollowup } = useQuery<DailyFollowup>({
    queryKey: ["/api/daily-followup/today"],
  });

  const { data: medications } = useQuery<Medication[]>({
    queryKey: ["/api/medications"],
  });

  const { data: tasks } = useQuery<DynamicTask[]>({
    queryKey: ["/api/tasks"],
  });

  const { data: insights } = useQuery<BehavioralInsight[]>({
    queryKey: ["/api/behavioral-insights"],
  });

  // Fetch latest video AI metrics via Express proxy
  const { data: latestVideoMetrics } = useQuery<any>({
    queryKey: ["/api/video-ai/latest-metrics"],
    enabled: !!user,
  });

  // Check if metrics are from today (12am-12am)
  const isMetricsFromToday = latestVideoMetrics?.created_at 
    ? new Date(latestVideoMetrics.created_at).toDateString() === new Date().toDateString()
    : false;

  return (
    <div className="space-y-6">
      {showEmergency && (
        <EmergencyAlert
          symptoms={[
            "Severe chest pain or pressure",
            "Difficulty breathing",
            "High fever (103°F+) with confusion",
          ]}
          onDismiss={() => setShowEmergency(false)}
        />
      )}

      {/* Dynamic Welcome Screen */}
      <DynamicWelcome userName={user?.firstName} />

      {/* Legal Disclaimer */}
      <LegalDisclaimer variant="wellness" />

      <div>
        <h2 className="text-2xl font-semibold mb-2" data-testid="text-health-summary">Today's Health Summary</h2>
        <p className="text-muted-foreground" data-testid="text-summary-subtitle">Your daily metrics and activities</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4" data-testid="grid-health-metrics">
        <HealthMetricCard
          title="Heart Rate"
          value="72"
          unit="bpm"
          icon={Heart}
          status="normal"
          trend="down"
          trendValue="3%"
          lastUpdated="2 mins ago"
          testId="card-metric-heart-rate"
        />
        <HealthMetricCard
          title="Steps Today"
          value="3,542"
          icon={Activity}
          status="normal"
          trend="up"
          trendValue="12%"
          lastUpdated="5 mins ago"
          testId="card-metric-steps"
        />
        <HealthMetricCard
          title="Water Intake"
          value="1.2"
          unit="L"
          icon={Droplet}
          status="warning"
          lastUpdated="1 hour ago"
          testId="card-metric-water"
        />
        <HealthMetricCard
          title="Sleep Quality"
          value="7.5"
          unit="hrs"
          icon={Moon}
          status="normal"
          trend="up"
          trendValue="0.5h"
          lastUpdated="Today"
          testId="card-metric-sleep"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <Card data-testid="card-todays-medications">
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
              <CardTitle data-testid="text-medications-title">Today's Medications</CardTitle>
              <Button variant="ghost" size="sm" data-testid="button-view-all-medications">
                View All
              </Button>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <MedicationCard
                name="Vitamin D3"
                dosage="2000 IU"
                frequency="Once daily"
                nextDose="2:00 PM"
                status="pending"
                isOTC={true}
                testId="card-medication-vitamin-d3"
              />
              <MedicationCard
                name="Prednisone"
                dosage="10mg"
                frequency="Twice daily"
                nextDose="Taken at 8:00 AM"
                status="taken"
                aiSuggestion="Consider reducing to 5mg based on recent improvements"
                testId="card-medication-prednisone"
              />
            </CardContent>
          </Card>

          <Card data-testid="card-daily-followup">
            <CardHeader>
              <CardTitle data-testid="text-followup-title">Daily Follow-up</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="device" className="space-y-4">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="device" data-testid="tab-device">Device Data</TabsTrigger>
                  <TabsTrigger value="video-ai" data-testid="tab-video-ai">Video AI</TabsTrigger>
                  <TabsTrigger value="bowel" data-testid="tab-bowel">Bowel/Bladder</TabsTrigger>
                </TabsList>
                <TabsContent value="device" className="space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div data-testid="data-heart-rate">
                      <span className="text-muted-foreground">Heart Rate: </span>
                      <span className="font-medium" data-testid="value-heart-rate">{todayFollowup?.heartRate || "--"} bpm</span>
                    </div>
                    <div data-testid="data-spo2">
                      <span className="text-muted-foreground">SpO2: </span>
                      <span className="font-medium" data-testid="value-spo2">{todayFollowup?.oxygenSaturation || "--"}%</span>
                    </div>
                    <div data-testid="data-temperature">
                      <span className="text-muted-foreground">Temp: </span>
                      <span className="font-medium" data-testid="value-temperature">{todayFollowup?.temperature || "--"}°F</span>
                    </div>
                    <div data-testid="data-steps">
                      <span className="text-muted-foreground">Steps: </span>
                      <span className="font-medium" data-testid="value-steps">{todayFollowup?.stepsCount || "--"}</span>
                    </div>
                  </div>
                </TabsContent>
                <TabsContent value="video-ai" className="space-y-3">
                  {isMetricsFromToday && latestVideoMetrics ? (
                    <>
                      {/* Today's Video AI Metrics */}
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium">Today's AI Analysis</p>
                          <Badge variant="secondary" className="text-xs">
                            {new Date(latestVideoMetrics.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </Badge>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {latestVideoMetrics.respiratory_rate_bpm && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Wind className="h-3 w-3 text-blue-500" />
                                <span className="text-muted-foreground">Respiratory</span>
                              </div>
                              <span className="font-medium">{latestVideoMetrics.respiratory_rate_bpm.toFixed(1)} bpm</span>
                            </div>
                          )}
                          
                          {latestVideoMetrics.skin_pallor_score !== null && latestVideoMetrics.skin_pallor_score !== undefined && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Palette className="h-3 w-3 text-amber-500" />
                                <span className="text-muted-foreground">Skin Pallor</span>
                              </div>
                              <span className="font-medium">{latestVideoMetrics.skin_pallor_score.toFixed(1)}/100</span>
                            </div>
                          )}
                          
                          {latestVideoMetrics.jaundice_risk_level && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Eye className="h-3 w-3 text-yellow-500" />
                                <span className="text-muted-foreground">Jaundice</span>
                              </div>
                              <span className="font-medium capitalize">{latestVideoMetrics.jaundice_risk_level}</span>
                            </div>
                          )}
                          
                          {latestVideoMetrics.facial_swelling_score !== null && latestVideoMetrics.facial_swelling_score !== undefined && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Users className="h-3 w-3 text-rose-500" />
                                <span className="text-muted-foreground">Swelling</span>
                              </div>
                              <span className="font-medium">{latestVideoMetrics.facial_swelling_score.toFixed(1)}/100</span>
                            </div>
                          )}
                          
                          {latestVideoMetrics.head_stability_score !== null && latestVideoMetrics.head_stability_score !== undefined && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Zap className="h-3 w-3 text-purple-500" />
                                <span className="text-muted-foreground">Tremor</span>
                              </div>
                              <span className="font-medium">{latestVideoMetrics.tremor_detected ? 'Detected' : 'None'}</span>
                            </div>
                          )}
                          
                          {latestVideoMetrics.tongue_color_index !== null && latestVideoMetrics.tongue_color_index !== undefined && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Smile className="h-3 w-3 text-pink-500" />
                                <span className="text-muted-foreground">Tongue</span>
                              </div>
                              <span className="font-medium">{latestVideoMetrics.tongue_coating_detected ? 'Coating' : 'Normal'}</span>
                            </div>
                          )}
                        </div>
                        
                        {/* Quality Metrics */}
                        {(latestVideoMetrics.lighting_quality_score || latestVideoMetrics.frames_analyzed || latestVideoMetrics.face_detection_confidence) && (
                          <div className="rounded-md border bg-muted/30 p-2 space-y-1">
                            <p className="text-xs font-medium">Analysis Quality</p>
                            <div className="grid grid-cols-3 gap-2 text-xs">
                              {latestVideoMetrics.lighting_quality_score && (
                                <div>
                                  <span className="text-muted-foreground">Lighting: </span>
                                  <span className="font-medium">{latestVideoMetrics.lighting_quality_score.toFixed(0)}/100</span>
                                </div>
                              )}
                              {latestVideoMetrics.frames_analyzed && (
                                <div>
                                  <span className="text-muted-foreground">Frames: </span>
                                  <span className="font-medium">{latestVideoMetrics.frames_analyzed}</span>
                                </div>
                              )}
                              {latestVideoMetrics.face_detection_confidence && (
                                <div>
                                  <span className="text-muted-foreground">Confidence: </span>
                                  <span className="font-medium">{(latestVideoMetrics.face_detection_confidence * 100).toFixed(0)}%</span>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        <Link href="/guided-video-exam">
                          <Button size="sm" variant="outline" className="w-full gap-2" data-testid="button-new-exam">
                            <Play className="h-3 w-3" />
                            Take Another Exam
                          </Button>
                        </Link>
                      </div>
                    </>
                  ) : (
                    <>
                      {/* No exam taken today - Prompt to take exam */}
                      <div className="rounded-lg border bg-gradient-to-br from-primary/5 to-primary/10 p-4 space-y-3">
                        <div className="flex items-center gap-2">
                          <div className="rounded-full bg-primary/10 p-2">
                            <Video className="h-4 w-4 text-primary" />
                          </div>
                          <div>
                            <p className="font-medium text-sm">No Examination Today</p>
                            <p className="text-xs text-muted-foreground">Complete your daily video exam</p>
                          </div>
                        </div>

                        <Link href="/guided-video-exam">
                          <Button size="sm" className="w-full gap-2" data-testid="button-start-video-exam">
                            <Play className="h-4 w-4" />
                            Start Video Examination
                          </Button>
                        </Link>
                        
                        <p className="text-xs text-center text-muted-foreground">
                          7-stage AI analysis • ~5-7 minutes • Camera required
                        </p>
                      </div>
                      
                      <div className="text-xs text-muted-foreground" data-testid="text-video-info">
                        <p className="font-medium mb-1">AI tracks:</p>
                        <ul className="space-y-0.5 ml-4 list-disc">
                          <li>Respiratory rate & breathing</li>
                          <li>Skin pallor (anemia detection)</li>
                          <li>Eye sclera yellowness (jaundice)</li>
                          <li>Facial swelling & edema</li>
                          <li>Head/hand tremor patterns</li>
                          <li>Tongue color & coating</li>
                        </ul>
                      </div>
                    </>
                  )}
                </TabsContent>
                <TabsContent value="bowel">
                  <div className="space-y-2 text-sm">
                    <div data-testid="data-bowel-movements">
                      <span className="text-muted-foreground">Bowel Movements: </span>
                      <span className="font-medium" data-testid="value-bowel-movements">{todayFollowup?.bowelMovements || "--"}</span>
                    </div>
                    <div data-testid="data-urine-frequency">
                      <span className="text-muted-foreground">Urine Frequency: </span>
                      <span className="font-medium" data-testid="value-urine-frequency">{todayFollowup?.urineFrequency || "--"}</span>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <Card data-testid="card-dynamic-tasks">
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
              <CardTitle data-testid="text-tasks-title">Dynamic Tasks</CardTitle>
              <Badge variant="secondary" data-testid="badge-tasks-pending">{tasks?.filter(t => !t.completed).length || 0} pending</Badge>
            </CardHeader>
            <CardContent>
              {tasks && tasks.length > 0 ? (
                <div className="space-y-2">
                  {tasks.slice(0, 5).map((task) => (
                    <div key={task.id} className="flex items-center gap-2 p-2 rounded-md hover-elevate" data-testid={`task-item-${task.id}`}>
                      {task.completed ? (
                        <CheckCircle className="h-4 w-4 text-chart-2 flex-shrink-0" data-testid={`icon-task-completed-${task.id}`} />
                      ) : (
                        <div className="h-4 w-4 rounded-full border-2 flex-shrink-0" data-testid={`icon-task-pending-${task.id}`} />
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate" data-testid={`text-task-title-${task.id}`}>{task.title}</p>
                        {task.description && (
                          <p className="text-xs text-muted-foreground truncate" data-testid={`text-task-description-${task.id}`}>{task.description}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground" data-testid="text-no-tasks">No tasks for today</p>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card data-testid="card-reminders">
            <CardHeader>
              <CardTitle className="flex items-center gap-2" data-testid="text-reminders-title">
                <Calendar className="h-5 w-5" />
                Reminders
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <ReminderCard
                type="water"
                title="Drink Water"
                time="2:00 PM"
                description="You've had 4 glasses today. Goal: 8 glasses"
                testId="reminder-water"
              />
              <ReminderCard
                type="exercise"
                title="Gentle Stretching"
                time="3:00 PM"
                description="15-minute low-impact session"
                testId="reminder-exercise"
              />
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-primary/5 to-primary/10" data-testid="card-behavioral-insights">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2" data-testid="text-insights-title">
                <Brain className="h-5 w-5" />
                Behavioral AI Insight
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {insights && insights.length > 0 ? (
                insights.slice(0, 2).map((insight, idx) => (
                  <div key={idx} className="flex items-start gap-2" data-testid={`insight-item-${idx}`}>
                    <TrendingUp className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <div className="text-sm">
                      <p className="font-medium mb-1" data-testid={`text-stress-level-${idx}`}>Stress Level: {insight.stressScore}/10</p>
                      <p className="text-muted-foreground" data-testid={`text-activity-level-${idx}`}>Activity: {insight.activityLevel}</p>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground" data-testid="text-insight-placeholder">
                  Complete your daily check-ins to unlock AI-powered behavioral insights and health trend detection.
                </p>
              )}
              <Link href="/behavioral-ai-insights">
                <Button variant="outline" size="sm" className="w-full" data-testid="button-view-insights">
                  View Full Insights
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card data-testid="card-medication-adherence">
            <CardHeader>
              <CardTitle className="text-base" data-testid="text-adherence-title">Medication Adherence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">This Week</span>
                  <span className="font-semibold text-chart-2" data-testid="value-adherence-percentage">92%</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2" data-testid="progress-adherence">
                  <div className="bg-chart-2 h-2 rounded-full" style={{ width: "92%" }} data-testid="progress-adherence-fill" />
                </div>
                <p className="text-xs text-muted-foreground" data-testid="text-active-medications">
                  {medications?.filter(m => m.active).length || 0} active medications
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
