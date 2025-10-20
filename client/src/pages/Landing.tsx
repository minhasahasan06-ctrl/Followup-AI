import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Stethoscope, Heart, Users, TrendingUp, Shield, Bot, Calendar, Camera, Activity, CheckCircle, ArrowRight, Sparkles, ClipboardList, BarChart3, FileText, Bell, Lock, Brain, Leaf, Zap, Award, Globe } from "lucide-react";
import { SiLinkedin, SiX } from "react-icons/si";
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
            <Button onClick={handlePatientSignup} data-testid="button-patient-portal">
              Patient Portal
            </Button>
            <Link href="/doctor-portal">
              <Button variant="outline" data-testid="button-doctor-portal">
                Doctor Portal
              </Button>
            </Link>
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
              <Button size="lg" onClick={handlePatientSignup} data-testid="button-patient-portal-hero">
                Patient Portal
              </Button>
              <Link href="/doctor-portal">
                <Button size="lg" variant="outline" data-testid="button-doctor-portal-hero">
                  Doctor Portal
                </Button>
              </Link>
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
                A personalized check-in every day with Agent Clona, tailored to <strong>your specific condition</strong>—whether you're 
                managing post-transplant care, autoimmune disorders, cancer treatment, or HIV/AIDS. Agent Clona learns your unique 
                health patterns and adapts questions to what matters most for you.
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

            {/* Personalization Based on Condition */}
            <div className="bg-gradient-to-r from-primary/10 via-primary/5 to-primary/10 rounded-lg p-8 border-2 border-primary/20 mb-12">
              <div className="flex items-center gap-3 mb-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/20">
                  <Sparkles className="h-6 w-6 text-primary" />
                </div>
                <h4 className="text-2xl font-semibold">Personalized to Your Condition</h4>
              </div>
              <p className="text-muted-foreground mb-6">
                Agent Clona adapts daily follow-ups based on your specific immunocompromised condition, ensuring every question and recommendation 
                is relevant to your unique health journey.
              </p>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-background/60 backdrop-blur rounded-lg p-4">
                  <h5 className="font-semibold mb-2">Post-Transplant Patients</h5>
                  <p className="text-sm text-muted-foreground">
                    Monitors rejection signs, immunosuppressant levels, and infection risks specific to transplant recipients.
                  </p>
                </div>
                <div className="bg-background/60 backdrop-blur rounded-lg p-4">
                  <h5 className="font-semibold mb-2">Cancer Treatment</h5>
                  <p className="text-sm text-muted-foreground">
                    Tracks chemotherapy side effects, neutropenia symptoms, and energy levels during treatment cycles.
                  </p>
                </div>
                <div className="bg-background/60 backdrop-blur rounded-lg p-4">
                  <h5 className="font-semibold mb-2">Autoimmune Disorders</h5>
                  <p className="text-sm text-muted-foreground">
                    Monitors flare-ups, medication efficacy, and inflammatory markers tailored to your specific condition.
                  </p>
                </div>
              </div>
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

        {/* Why Followup AI Section - Enhanced */}
        <section className="py-20 px-6 bg-gradient-to-b from-background via-primary/5 to-background">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary mb-6">
                <Award className="h-4 w-4" />
                <span className="text-sm font-medium">What Sets Us Apart</span>
              </div>
              <h3 className="text-4xl font-bold mb-4">The Followup AI Advantage</h3>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
                Cutting-edge technology meets compassionate care—designed specifically for the unique challenges 
                faced by immunocompromised patients and their healthcare teams.
              </p>
            </div>

            {/* Top 3 Features Grid */}
            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <Card className="hover-elevate transition-all duration-300 group" data-testid="card-feature-security">
                <CardHeader>
                  <div className="flex items-center justify-between mb-4">
                    <Shield className="h-12 w-12 text-primary group-hover:scale-110 transition-transform" />
                    <Lock className="h-5 w-5 text-primary/60" />
                  </div>
                  <CardTitle className="text-xl">Military-Grade Security</CardTitle>
                  <CardDescription>HIPAA compliant with zero compromises</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">End-to-end AES-256 encryption for all data in transit and at rest</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Full HIPAA compliance with regular third-party security audits</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Patient-controlled data sharing with granular consent management</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Secure cloud infrastructure with automatic backups and disaster recovery</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="hover-elevate transition-all duration-300 group" data-testid="card-feature-ai">
                <CardHeader>
                  <div className="flex items-center justify-between mb-4">
                    <Brain className="h-12 w-12 text-primary group-hover:scale-110 transition-transform" />
                    <Zap className="h-5 w-5 text-primary/60" />
                  </div>
                  <CardTitle className="text-xl">Advanced AI Intelligence</CardTitle>
                  <CardDescription>GPT-4 powered medical insights</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Medical entity recognition to identify symptoms, medications, and conditions</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Real-time sentiment analysis to detect emotional distress and mental health concerns</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Pattern recognition across thousands of daily check-ins to predict complications</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Context-aware responses that remember your history and preferences</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="hover-elevate transition-all duration-300 group" data-testid="card-feature-holistic">
                <CardHeader>
                  <div className="flex items-center justify-between mb-4">
                    <Leaf className="h-12 w-12 text-primary group-hover:scale-110 transition-transform" />
                    <Heart className="h-5 w-5 text-primary/60" />
                  </div>
                  <CardTitle className="text-xl">Whole-Person Care</CardTitle>
                  <CardDescription>Beyond symptoms—supporting your well-being</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Guided meditation and breathing exercises for stress management</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Auto-generated health journals that track behavioral patterns and triggers</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Professional psychological counseling for patients and doctors</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-muted-foreground">Wellness activities tailored to your energy levels and physical capabilities</p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Additional Features Row */}
            <div className="grid md:grid-cols-3 gap-6">
              <Card className="hover-elevate transition-all duration-300" data-testid="card-feature-research">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                      <Globe className="h-5 w-5 text-primary" />
                    </div>
                    <CardTitle className="text-lg">Research-Backed</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Access to global health data from PubMed, WHO, and PhysioNet for evidence-based care and doctor-driven AI research.
                  </p>
                </CardContent>
              </Card>

              <Card className="hover-elevate transition-all duration-300" data-testid="card-feature-integration">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                      <Activity className="h-5 w-5 text-primary" />
                    </div>
                    <CardTitle className="text-lg">Wearable Integration</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Seamlessly connect Fitbit, Apple Health, and other devices with patient-controlled consent for comprehensive tracking.
                  </p>
                </CardContent>
              </Card>

              <Card className="hover-elevate transition-all duration-300" data-testid="card-feature-emergency">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                      <Bell className="h-5 w-5 text-primary" />
                    </div>
                    <CardTitle className="text-lg">Emergency Response</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Intelligent alerts notify doctors of critical changes, ensuring rapid intervention when you need it most.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* Doctor Section */}
        <section className="py-20 px-6 bg-muted/50">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary mb-6">
                <Stethoscope className="h-4 w-4" />
                <span className="text-sm font-medium">For Healthcare Providers</span>
              </div>
              <h3 className="text-4xl font-bold mb-4">Empower Your Practice with Assistant Lysa</h3>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
                Manage your immunocompromised patients more effectively with AI-powered insights, automated monitoring, 
                and comprehensive patient timelines—all while reducing your administrative burden.
              </p>
            </div>

            {/* Doctor Features Grid */}
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
              <Card className="hover-elevate transition-all duration-300" data-testid="card-doctor-feature-1">
                <CardHeader>
                  <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 mb-4">
                    <ClipboardList className="h-7 w-7 text-primary" />
                  </div>
                  <CardTitle className="text-lg">Patient Review Dashboard</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    See all your immunocompromised patients in one view with AI-flagged concerns and priority alerts.
                  </p>
                </CardContent>
              </Card>

              <Card className="hover-elevate transition-all duration-300" data-testid="card-doctor-feature-2">
                <CardHeader>
                  <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 mb-4">
                    <Bell className="h-7 w-7 text-primary" />
                  </div>
                  <CardTitle className="text-lg">Smart Alerts</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Get notified when patients show early signs of infection, medication non-adherence, or deterioration.
                  </p>
                </CardContent>
              </Card>

              <Card className="hover-elevate transition-all duration-300" data-testid="card-doctor-feature-3">
                <CardHeader>
                  <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 mb-4">
                    <BarChart3 className="h-7 w-7 text-primary" />
                  </div>
                  <CardTitle className="text-lg">Research Center</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Generate epidemiological reports and train AI models using public medical datasets (PubMed, WHO, PhysioNet).
                  </p>
                </CardContent>
              </Card>

              <Card className="hover-elevate transition-all duration-300" data-testid="card-doctor-feature-4">
                <CardHeader>
                  <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 mb-4">
                    <FileText className="h-7 w-7 text-primary" />
                  </div>
                  <CardTitle className="text-lg">Comprehensive Timelines</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Review complete patient histories with daily follow-ups, vitals, medications, and visual assessments in one place.
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Condition-Specific Monitoring */}
            <div className="bg-card rounded-lg p-8 border">
              <div className="flex items-center gap-3 mb-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                  <Activity className="h-6 w-6 text-primary" />
                </div>
                <h4 className="text-2xl font-semibold">Condition-Specific Patient Monitoring</h4>
              </div>
              <p className="text-muted-foreground mb-8">
                Assistant Lysa tailors monitoring protocols based on each patient's immunocompromised condition, 
                helping you catch complications early and intervene proactively.
              </p>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
                    <h5 className="font-semibold">Transplant Recipients</h5>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground ml-7">
                    <li>• Rejection marker tracking</li>
                    <li>• Immunosuppressant monitoring</li>
                    <li>• Infection risk assessment</li>
                    <li>• Organ-specific vitals</li>
                  </ul>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
                    <h5 className="font-semibold">Oncology Patients</h5>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground ml-7">
                    <li>• Infection pattern analysis</li>
                    <li>• Treatment side effects</li>
                    <li>• Energy level patterns</li>
                    <li>• Nutrition tracking</li>
                  </ul>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
                    <h5 className="font-semibold">Autoimmune Care</h5>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground ml-7">
                    <li>• Flare-up prediction</li>
                    <li>• Medication efficacy trends</li>
                    <li>• Inflammatory markers</li>
                    <li>• Quality of life metrics</li>
                  </ul>
                </div>
              </div>
              <div className="mt-8 flex gap-4 justify-center flex-wrap">
                <Link href="/doctor-portal">
                  <Button size="lg" className="gap-2" data-testid="button-doctor-portal-cta">
                    Access Doctor Portal
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
                <Button size="lg" variant="outline" onClick={() => window.location.href = "/api/login?role=doctor&type=signup"} data-testid="button-doctor-signup">
                  Sign Up as Doctor
                </Button>
              </div>
            </div>
          </div>
        </section>

        {/* App Download Section */}
        <section className="py-12 px-6">
          <div className="max-w-4xl mx-auto text-center">
            <h3 className="text-2xl font-semibold mb-4">Download Our Mobile App</h3>
            <div className="flex gap-4 justify-center flex-wrap items-center">
              <Link href="/coming-soon" data-testid="link-download-ios" className="hover-elevate active-elevate-2 transition-transform rounded-lg overflow-hidden block">
                <img 
                  src={appStoreBadge} 
                  alt="Download on the App Store" 
                  className="h-12 w-auto"
                />
              </Link>
              <Link href="/coming-soon" data-testid="link-download-android" className="hover-elevate active-elevate-2 transition-transform rounded-lg overflow-hidden block">
                <img 
                  src={googlePlayBadge} 
                  alt="Get it on Google Play" 
                  className="h-12 w-auto"
                />
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t py-12 px-6 bg-gradient-to-b from-muted/50 to-muted/80">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
            {/* Product Column */}
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-3">
                <li>
                  <button onClick={handlePatientSignup} className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-patient">
                    Patient Portal
                  </button>
                </li>
                <li>
                  <Link href="/doctor-portal" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-doctor">
                    Doctor Portal
                  </Link>
                </li>
                <li>
                  <Link href="/coming-soon" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-mobile">
                    Mobile Apps
                  </Link>
                </li>
                <li>
                  <span className="text-sm text-muted-foreground">Agent Clona</span>
                </li>
                <li>
                  <span className="text-sm text-muted-foreground">Assistant Lysa</span>
                </li>
              </ul>
            </div>

            {/* Resources Column */}
            <div>
              <h4 className="font-semibold mb-4">Resources</h4>
              <ul className="space-y-3">
                <li>
                  <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-faq">
                    FAQ
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-pricing">
                    Pricing
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-documentation">
                    Documentation
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-api">
                    API
                  </a>
                </li>
              </ul>
            </div>

            {/* Company Column */}
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-3">
                <li>
                  <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-terms">
                    Terms & Conditions
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-privacy">
                    Privacy Policy
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-hipaa">
                    HIPAA Compliance
                  </a>
                </li>
                <li>
                  <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors" data-testid="link-footer-contact">
                    Contact Us
                  </a>
                </li>
              </ul>
            </div>

            {/* Social Column */}
            <div>
              <h4 className="font-semibold mb-4">Social</h4>
              <div className="flex gap-3">
                <a 
                  href="https://twitter.com" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex h-9 w-9 items-center justify-center rounded-md bg-muted hover-elevate active-elevate-2 transition-all"
                  data-testid="link-footer-twitter"
                  aria-label="Twitter"
                >
                  <SiX className="h-4 w-4" />
                </a>
                <a 
                  href="https://linkedin.com" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex h-9 w-9 items-center justify-center rounded-md bg-muted hover-elevate active-elevate-2 transition-all"
                  data-testid="link-footer-linkedin"
                  aria-label="LinkedIn"
                >
                  <SiLinkedin className="h-4 w-4" />
                </a>
              </div>
            </div>
          </div>

          {/* Bottom Section */}
          <div className="pt-8 border-t border-border">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-chart-2" />
                <span className="text-sm text-muted-foreground">HIPAA Compliant • FDA/CE Pending</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Followup AI © 2025. All rights reserved.
              </p>
            </div>
            <p className="text-xs text-muted-foreground text-center md:text-left mt-4">
              This platform provides health information and is not a substitute for professional medical advice.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
