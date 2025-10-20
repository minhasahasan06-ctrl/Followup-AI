import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Stethoscope, Heart, Users, TrendingUp, Shield, Bot } from "lucide-react";

export default function Landing() {
  const handleLogin = () => {
    window.location.href = "/api/login";
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Stethoscope className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-semibold">Followup AI</h1>
              <p className="text-xs text-muted-foreground">HIPAA-Compliant Health Platform</p>
            </div>
          </div>
          <Button onClick={handleLogin} data-testid="button-login">
            Sign In
          </Button>
        </div>
      </header>

      <main>
        <section className="py-20 px-6">
          <div className="max-w-6xl mx-auto text-center">
            <h2 className="text-5xl font-bold tracking-tight mb-6">
              AI-Powered Health Monitoring for Immunocompromised Patients
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
              Advanced daily follow-ups with Agent Clona, comprehensive health tracking, and research-backed insights 
              to help you manage your health with confidence.
            </p>
            <div className="flex gap-4 justify-center">
              <Button size="lg" onClick={handleLogin} data-testid="button-get-started">
                Get Started
              </Button>
              <Button size="lg" variant="outline" data-testid="button-learn-more">
                Learn More
              </Button>
            </div>
          </div>
        </section>

        <section className="py-16 px-6 bg-muted/50">
          <div className="max-w-7xl mx-auto">
            <h3 className="text-3xl font-semibold text-center mb-12">Who We Serve</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <Card data-testid="card-patient-role">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                      <Heart className="h-6 w-6 text-primary" />
                    </div>
                    <CardTitle className="text-2xl">For Patients</CardTitle>
                  </div>
                  <CardDescription className="text-base">
                    Comprehensive health monitoring tailored for immunocompromised individuals
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start gap-2">
                    <Bot className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">Agent Clona - Your AI Health Assistant</p>
                      <p className="text-sm text-muted-foreground">24/7 support with GPT-4 powered insights</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <TrendingUp className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">Daily Health Follow-ups</p>
                      <p className="text-sm text-muted-foreground">Track vitals, symptoms, and medication adherence</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Heart className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">Wellness Modules</p>
                      <p className="text-sm text-muted-foreground">Meditation, breathing exercises, and gentle movement</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card data-testid="card-doctor-role">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                      <Users className="h-6 w-6 text-primary" />
                    </div>
                    <CardTitle className="text-2xl">For Doctors</CardTitle>
                  </div>
                  <CardDescription className="text-base">
                    Advanced tools for patient management and epidemiological research
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start gap-2">
                    <Bot className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">Assistant Lysa - Clinical AI Partner</p>
                      <p className="text-sm text-muted-foreground">Pattern recognition and evidence-based insights</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Users className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">Patient Review Dashboard</p>
                      <p className="text-sm text-muted-foreground">Comprehensive timelines with AI-flagged concerns</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <TrendingUp className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">Research Center</p>
                      <p className="text-sm text-muted-foreground">AI-powered epidemiological analysis and reporting</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        <section className="py-16 px-6">
          <div className="max-w-7xl mx-auto">
            <h3 className="text-3xl font-semibold text-center mb-12">Why Followup AI?</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <Shield className="h-10 w-10 text-primary mb-3" />
                  <CardTitle>HIPAA Compliant</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Bank-level encryption and strict compliance with healthcare privacy regulations. Your data is secure.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <Bot className="h-10 w-10 text-primary mb-3" />
                  <CardTitle>AI-Powered Insights</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    GPT-4 integration with medical entity recognition and sentiment analysis for comprehensive health understanding.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <Heart className="h-10 w-10 text-primary mb-3" />
                  <CardTitle>Holistic Care</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Combines clinical tracking with wellness modules, behavioral insights, and auto-generated health journals.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t py-8 px-6 bg-muted/30">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-3">
            <div className="h-2 w-2 rounded-full bg-chart-2" />
            <span className="text-sm text-muted-foreground">HIPAA Compliant • FDA/CE Pending</span>
          </div>
          <p className="text-xs text-muted-foreground">
            © 2025 Followup AI. This platform provides health information and is not a substitute for professional medical advice.
          </p>
        </div>
      </footer>
    </div>
  );
}
