import { EmergencyAlert } from "../EmergencyAlert";

export default function EmergencyAlertExample() {
  return (
    <div className="p-4">
      <EmergencyAlert
        symptoms={[
          "Severe chest pain or pressure",
          "Difficulty breathing",
          "High fever (103°F+) with confusion",
        ]}
        onDismiss={() => console.log("Alert dismissed")}
      />
    </div>
  );
}
