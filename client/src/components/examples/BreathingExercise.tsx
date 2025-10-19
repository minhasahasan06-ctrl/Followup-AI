import { BreathingExercise } from "../BreathingExercise";

export default function BreathingExerciseExample() {
  return (
    <div className="grid gap-4 md:grid-cols-2 p-4">
      <BreathingExercise
        name="Box Breathing"
        description="Calm your nervous system with this balanced breathing technique"
        pattern={[4, 4, 4, 4]}
      />
      <BreathingExercise
        name="4-7-8 Technique"
        description="Reduce anxiety and promote relaxation before sleep"
        pattern={[4, 7, 8, 0]}
      />
    </div>
  );
}
