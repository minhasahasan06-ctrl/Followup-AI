import { WellnessSessionCard } from "@/components/WellnessSessionCard";
import { BreathingExercise } from "@/components/BreathingExercise";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Wind, Heart } from "lucide-react";

export default function Wellness() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold mb-2">Wellness Center</h1>
        <p className="text-muted-foreground">
          Personalized meditation, breathing exercises, and gentle movement for immunocompromised patients
        </p>
      </div>

      <Tabs defaultValue="meditation" className="space-y-6">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="meditation" data-testid="tab-meditation">
            <Wind className="h-4 w-4 mr-2" />
            Meditation
          </TabsTrigger>
          <TabsTrigger value="exercise" data-testid="tab-exercise">
            <Heart className="h-4 w-4 mr-2" />
            Exercise
          </TabsTrigger>
        </TabsList>

        <TabsContent value="meditation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Breathing Exercises</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
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
            </CardContent>
          </Card>

          <div>
            <h2 className="text-xl font-semibold mb-4">Guided Meditation Sessions</h2>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <WellnessSessionCard
                title="Morning Calm"
                duration="10 min"
                difficulty="Easy"
                description="Start your day with gentle mindfulness meditation designed for immunocompromised patients"
                type="meditation"
                recommended={true}
              />
              <WellnessSessionCard
                title="Body Scan Meditation"
                duration="20 min"
                difficulty="Moderate"
                description="Deep relaxation technique to connect with your body and release tension"
                type="meditation"
              />
              <WellnessSessionCard
                title="Stress Relief"
                duration="15 min"
                difficulty="Easy"
                description="Quick stress management session for busy days"
                type="meditation"
              />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="exercise" className="space-y-6">
          <div>
            <h2 className="text-xl font-semibold mb-4">Gentle Exercise Programs</h2>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <WellnessSessionCard
                title="Gentle Stretching"
                duration="15 min"
                difficulty="Easy"
                description="Low-impact stretching exercises to maintain flexibility and reduce joint stiffness"
                type="exercise"
                recommended={true}
              />
              <WellnessSessionCard
                title="Chair Yoga"
                duration="20 min"
                difficulty="Easy"
                description="Safe, seated yoga practice perfect for days when you need extra support"
                type="exercise"
                recommended={true}
              />
              <WellnessSessionCard
                title="Light Walking Routine"
                duration="25 min"
                difficulty="Moderate"
                description="Guided walking session with proper pacing for energy conservation"
                type="exercise"
              />
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
