import { HealthMetricCard } from "@/components/HealthMetricCard";
import { MedicationCard } from "@/components/MedicationCard";
import { FollowUpCard } from "@/components/FollowUpCard";
import { ReminderCard } from "@/components/ReminderCard";
import { EmergencyAlert } from "@/components/EmergencyAlert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Heart, Activity, Droplet, Moon, TrendingUp, Calendar } from "lucide-react";
import { useState } from "react";

export default function Dashboard() {
  const [showEmergency, setShowEmergency] = useState(false);

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
        <h1 className="text-4xl font-semibold mb-2">Welcome back, Sarah</h1>
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
            <CardContent className="space-y-4">
              <FollowUpCard
                date="Today"
                time="9:00 AM"
                type="Daily Visual Assessment"
                completed={false}
                onStartAssessment={() => console.log("Starting assessment")}
              />
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
              <CardTitle className="text-base">AI Health Insights</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start gap-2">
                <TrendingUp className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                <p className="text-sm">
                  Your activity levels have improved by 15% this week. Keep up the great work!
                </p>
              </div>
              <Button variant="outline" size="sm" className="w-full" data-testid="button-view-insights">
                View Full Insights
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
