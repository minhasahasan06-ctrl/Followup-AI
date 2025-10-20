import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Stethoscope, Heart, Users, TrendingUp, Shield, Bot, Calendar, Camera, Activity, CheckCircle, ArrowRight, Sparkles } from "lucide-react";
import appStoreBadge from "@assets/stock_images/apple_app_store_down_a3189b22.jpg";
import googlePlayBadge from "@assets/stock_images/google_play_store_do_60dd571a.jpg";

export default function Landing() {
  const handlePatientSignup = () => {
    // Will redirect to Replit Auth with patient context
    window.location.href = "/api/login?role=patient&type=signup";
  };

  const handlePatientLogin = () => {
    // Will redirect to Replit Auth with patient context
    window.location.href = "/api/login?role=patient&type=login";
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Stethoscope className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-semibold">Followup AI</h1>
              <p className="text-xs text-muted-foreground">HIPAA-Compliant Health Platform</p>
            </div>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <Link href="/doctor-portal">
              <Button variant="ghost" data-testid="button-doctor-portal">
                Doctor Portal
              </Button>
            </Link>
            <Button variant="outline" onClick={handlePatientLogin} data-testid="button-patient-login">
              Patient Login
            </Button>
            <Button onClick={handlePatientSignup} data-testid="button-patient-signup">
              Patient Sign Up
            </Button>
          </div>
        </div>
      </header>

      <main>
        {/* Hero Section */}
        <section className="py-20 px-6">
          <div className="max-w-6xl mx-auto text-center">
            <h2 className="text-5xl font-bold tracking-tight mb-6">
              Personalized AI for Immunocompromised Patients
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
              Advanced daily follow-ups with Agent Clona, comprehensive health tracking, and research-backed insights 
              to help you manage your health with confidence.
            </p>
            <div className="flex gap-4 justify-center flex-wrap">
              <Button size="lg" onClick={handlePatientSignup} data-testid="button-get-started">
                Get Started Free
              </Button>
              <Button size="lg" variant="outline" onClick={handlePatientLogin} data-testid="button-patient-login-hero">
                Sign In
              </Button>
            </div>
          </div>
        </section>

        {/* Daily Follow-Ups Section */}
        <section className="py-20 px-6 bg-gradient-to-br from-primary/5 via-background to-primary/5">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary mb-6">
                <Sparkles className="h-4 w-4" />
                <span className="text-sm font-medium">Your Daily Health Companion</span>
              </div>
              <h3 className="text-4xl font-bold mb-4">What Are Daily Follow-Ups?</h3>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
                A personalized check-in every day with Agent Clona to monitor your health, catch early warning signs, 
                and keep you on track—because consistent care makes all the difference.
              </p>
            </div>

            {/* Interactive Timeline */}
            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <Card className="hover-elevate transition-all duration-300 group" data-testid="card-followup-step-1">
                <CardHeader>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                      <Calendar className="h-7 w-7 text-primary" />
                    </div>
                    <span className="text-sm font-semibold text-primary">Step 1</span>
                  </div>
                  <CardTitle className="text-xl">Morning Check-In</CardTitle>
                  <CardDescription className="text-base">
                    Start your day with a quick health snapshot
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <p className="text-sm">How did you sleep?</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <p className="text-sm">Any new symptoms?</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <p className="text-sm">Energy levels and mood</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="hover-elevate transition-all duration-300 group" data-testid="card-followup-step-2">
                <CardHeader>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                      <Camera className="h-7 w-7 text-primary" />
                    </div>
                    <span className="text-sm font-semibold text-primary">Step 2</span>
                  </div>
                  <CardTitle className="text-xl">Visual Assessment</CardTitle>
                  <CardDescription className="text-base">
                    AI-powered analysis for early detection
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <p className="text-sm">Skin changes or rashes</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <p className="text-sm">Wound healing progress</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <p className="text-sm">Inflammation monitoring</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="hover-elevate transition-all duration-300 group" data-testid="card-followup-step-3">
                <CardHeader>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                      <Activity className="h-7 w-7 text-primary" />
                    </div>
                    <span className="text-sm font-semibold text-primary">Step 3</span>
                  </div>
                  <CardTitle className="text-xl">AI Insights</CardTitle>
                  <CardDescription className="text-base">
                    Personalized recommendations from Agent Clona
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <p className="text-sm">Pattern recognition</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <p className="text-sm">Medication reminders</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <p className="text-sm">Actionable next steps</p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Why It Matters for Immunocompromised Patients */}
            <div className="bg-card rounded-lg p-8 border">
              <div className="flex items-center gap-3 mb-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                  <Heart className="h-6 w-6 text-primary" />
                </div>
                <h4 className="text-2xl font-semibold">Why This Matters for You</h4>
              </div>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                      <ArrowRight className="h-5 w-5 text-primary" />
                    </div>
                  </div>
                  <div>
                    <h5 className="font-semibold mb-2">Early Infection Detection</h5>
                    <p className="text-muted-foreground">
                      Catch subtle changes before they become serious. Your compromised immune system needs vigilant monitoring 
                      to detect infections at their earliest stages.
                    </p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                      <ArrowRight className="h-5 w-5 text-primary" />
                    </div>
                  </div>
                  <div>
                    <h5 className="font-semibold mb-2">Medication Adherence</h5>
                    <p className="text-muted-foreground">
                      Stay on track with your immunosuppressants and other critical medications. Consistency is vital 
                      for maintaining your health stability.
                    </p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                      <ArrowRight className="h-5 w-5 text-primary" />
                    </div>
                  </div>
                  <div>
                    <h5 className="font-semibold mb-2">Symptom Tracking</h5>
                    <p className="text-muted-foreground">
                      Document patterns over time. Agent Clona learns your baseline and alerts you to deviations that 
                      might indicate complications.
                    </p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                      <ArrowRight className="h-5 w-5 text-primary" />
                    </div>
                  </div>
                  <div>
                    <h5 className="font-semibold mb-2">Peace of Mind</h5>
                    <p className="text-muted-foreground">
                      Know that you're being monitored daily by AI that never sleeps. Reduce anxiety with consistent 
                      check-ins and expert guidance.
                    </p>
                  </div>
                </div>
              </div>
              <div className="mt-8 text-center">
                <Button size="lg" onClick={handlePatientSignup} className="gap-2" data-testid="button-start-followups">
                  Start Your Daily Follow-Ups
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </section>

        {/* Who We Serve Section */}
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
                <CardContent className="space-y-4">
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
                  <div className="pt-4 border-t">
                    <div className="flex gap-3">
                      <Button onClick={handlePatientSignup} className="flex-1" data-testid="button-patient-signup-card">
                        Sign Up
                      </Button>
                      <Button onClick={handlePatientLogin} variant="outline" className="flex-1" data-testid="button-patient-login-card">
                        Login
                      </Button>
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
                <CardContent className="space-y-4">
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
                  <div className="pt-4 border-t">
                    <Link href="/doctor-portal">
                      <Button className="w-full" data-testid="button-doctor-portal-card">
                        Access Doctor Portal
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* Why Followup AI Section */}
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

        {/* App Download Section */}
        <section className="py-16 px-6 bg-muted/50">
          <div className="max-w-6xl mx-auto text-center">
            <h3 className="text-3xl font-semibold mb-6">Download Our Mobile App</h3>
            <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
              Access your health data on the go. Track vitals, chat with Agent Clona, and manage medications 
              from your iOS or Android device.
            </p>
            <div className="flex gap-6 justify-center flex-wrap items-center">
              <a 
                href="https://apps.apple.com" 
                target="_blank" 
                rel="noopener noreferrer"
                data-testid="link-download-ios"
                className="hover-elevate active-elevate-2 transition-transform rounded-lg overflow-hidden"
              >
                <img 
                  src={appStoreBadge} 
                  alt="Download on the App Store" 
                  className="h-14 w-auto"
                />
              </a>
              <a 
                href="https://play.google.com" 
                target="_blank" 
                rel="noopener noreferrer"
                data-testid="link-download-android"
                className="hover-elevate active-elevate-2 transition-transform rounded-lg overflow-hidden"
              >
                <img 
                  src={googlePlayBadge} 
                  alt="Get it on Google Play" 
                  className="h-14 w-auto"
                />
              </a>
            </div>
            <p className="text-sm text-muted-foreground mt-4">
              Coming Soon • iOS 15+ and Android 10+ required
            </p>
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
