import { HealthMetricCard } from "@/components/HealthMetricCard";
import { MedicationCard } from "@/components/MedicationCard";
import { FollowUpCard } from "@/components/FollowUpCard";
import { ReminderCard } from "@/components/ReminderCard";
import { EmergencyAlert } from "@/components/EmergencyAlert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Heart, Activity, Droplet, Moon, TrendingUp, Calendar, CheckCircle, Brain } from "lucide-react";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
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

      <div>
        <h1 className="text-4xl font-semibold mb-2">Welcome back, {user?.firstName || "Patient"}</h1>
        <p className="text-muted-foreground">Here's your health summary for today</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <HealthMetricCard
          title="Heart Rate"
          value="72"
          unit="bpm"
          icon={Heart}
          status="normal"
          trend="down"
          trendValue="3%"
          lastUpdated="2 mins ago"
        />
        <HealthMetricCard
          title="Steps Today"
          value="3,542"
          icon={Activity}
          status="normal"
          trend="up"
          trendValue="12%"
          lastUpdated="5 mins ago"
        />
        <HealthMetricCard
          title="Water Intake"
          value="1.2"
          unit="L"
          icon={Droplet}
          status="warning"
          lastUpdated="1 hour ago"
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
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
              <CardTitle>Today's Medications</CardTitle>
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
              />
              <MedicationCard
                name="Prednisone"
                dosage="10mg"
                frequency="Twice daily"
                nextDose="Taken at 8:00 AM"
                status="taken"
                aiSuggestion="Consider reducing to 5mg based on recent improvements"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Daily Follow-up</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="device" className="space-y-4">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="device" data-testid="tab-device">Device Data</TabsTrigger>
                  <TabsTrigger value="camera" data-testid="tab-camera">Camera</TabsTrigger>
                  <TabsTrigger value="bowel" data-testid="tab-bowel">Bowel/Bladder</TabsTrigger>
                </TabsList>
                <TabsContent value="device" className="space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-muted-foreground">Heart Rate: </span>
                      <span className="font-medium">{todayFollowup?.heartRate || "--"} bpm</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">SpO2: </span>
                      <span className="font-medium">{todayFollowup?.oxygenSaturation || "--"}%</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Temp: </span>
                      <span className="font-medium">{todayFollowup?.temperature || "--"}°F</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Steps: </span>
                      <span className="font-medium">{todayFollowup?.stepsCount || "--"}</span>
                    </div>
                  </div>
                </TabsContent>
                <TabsContent value="camera">
                  <p className="text-sm text-muted-foreground">Camera assessment coming soon</p>
                </TabsContent>
                <TabsContent value="bowel">
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-muted-foreground">Bowel Movements: </span>
                      <span className="font-medium">{todayFollowup?.bowelMovements || "--"}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Urine Frequency: </span>
                      <span className="font-medium">{todayFollowup?.urineFrequency || "--"}</span>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
              <CardTitle>Dynamic Tasks</CardTitle>
              <Badge variant="secondary">{tasks?.filter(t => !t.completed).length || 0} pending</Badge>
            </CardHeader>
            <CardContent>
              {tasks && tasks.length > 0 ? (
                <div className="space-y-2">
                  {tasks.slice(0, 5).map((task) => (
                    <div key={task.id} className="flex items-center gap-2 p-2 rounded-md hover-elevate" data-testid={`task-${task.id}`}>
                      {task.completed ? (
                        <CheckCircle className="h-4 w-4 text-chart-2 flex-shrink-0" />
                      ) : (
                        <div className="h-4 w-4 rounded-full border-2 flex-shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{task.title}</p>
                        {task.description && (
                          <p className="text-xs text-muted-foreground truncate">{task.description}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No tasks for today</p>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
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
              />
              <ReminderCard
                type="exercise"
                title="Gentle Stretching"
                time="3:00 PM"
                description="15-minute low-impact session"
              />
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-primary/5 to-primary/10">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Behavioral Insights
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {insights && insights.length > 0 ? (
                insights.slice(0, 2).map((insight, idx) => (
                  <div key={idx} className="flex items-start gap-2">
                    <TrendingUp className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <div className="text-sm">
                      <p className="font-medium mb-1">Stress Level: {insight.stressScore}/10</p>
                      <p className="text-muted-foreground">Activity: {insight.activityLevel}</p>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">
                  Your activity levels have improved by 15% this week. Keep up the great work!
                </p>
              )}
              <Button variant="outline" size="sm" className="w-full" data-testid="button-view-insights">
                View Full Insights
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Medication Adherence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">This Week</span>
                  <span className="font-semibold text-chart-2">92%</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div className="bg-chart-2 h-2 rounded-full" style={{ width: "92%" }} />
                </div>
                <p className="text-xs text-muted-foreground">
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
