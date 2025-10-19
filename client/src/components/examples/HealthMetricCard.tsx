import { HealthMetricCard } from "../HealthMetricCard";
import { Heart, Activity, Droplet, Moon } from "lucide-react";

export default function HealthMetricCardExample() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 p-4">
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
  );
}
