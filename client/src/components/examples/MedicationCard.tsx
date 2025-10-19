import { MedicationCard } from "../MedicationCard";

export default function MedicationCardExample() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 p-4">
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
      <MedicationCard
        name="Multivitamin"
        dosage="1 tablet"
        frequency="Once daily"
        nextDose="Missed - 9:00 AM"
        status="missed"
        isOTC={true}
      />
    </div>
  );
}
