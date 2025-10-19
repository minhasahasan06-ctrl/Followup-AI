import { WellnessSessionCard } from "../WellnessSessionCard";

export default function WellnessSessionCardExample() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 p-4">
      <WellnessSessionCard
        title="Morning Calm"
        duration="10 min"
        difficulty="Easy"
        description="Start your day with gentle mindfulness meditation designed for immunocompromised patients"
        type="meditation"
        recommended={true}
      />
      <WellnessSessionCard
        title="Gentle Stretching"
        duration="15 min"
        difficulty="Easy"
        description="Low-impact stretching exercises to maintain flexibility and reduce joint stiffness"
        type="exercise"
        recommended={true}
      />
      <WellnessSessionCard
        title="Body Scan Meditation"
        duration="20 min"
        difficulty="Moderate"
        description="Deep relaxation technique to connect with your body and release tension"
        type="meditation"
      />
    </div>
  );
}
