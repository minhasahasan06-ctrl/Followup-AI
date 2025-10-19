import { ReminderCard } from "../ReminderCard";

export default function ReminderCardExample() {
  return (
    <div className="grid gap-3 max-w-md p-4">
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
      <ReminderCard
        type="medication"
        title="Take Vitamin D3"
        time="4:00 PM"
        description="2000 IU daily supplement"
      />
    </div>
  );
}
