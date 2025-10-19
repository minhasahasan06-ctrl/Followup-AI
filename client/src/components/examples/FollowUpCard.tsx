import { FollowUpCard } from "../FollowUpCard";

export default function FollowUpCardExample() {
  return (
    <div className="grid gap-4 md:grid-cols-2 p-4">
      <FollowUpCard
        date="Today"
        time="9:00 AM"
        type="Daily Visual Assessment"
        completed={false}
        onStartAssessment={() => console.log("Starting assessment")}
      />
      <FollowUpCard
        date="Yesterday"
        time="9:00 AM"
        type="Daily Visual Assessment"
        completed={true}
        results={[
          { condition: "Anemia", detected: false },
          { condition: "Jaundice", detected: false },
          { condition: "Edema", detected: true, confidence: 72 },
        ]}
      />
    </div>
  );
}
