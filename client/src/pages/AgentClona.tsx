import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ArrowLeft, Stethoscope, Heart, Shield, Calendar, Pill, Activity, Camera, MessageSquare, Sparkles, CheckCircle, Moon, Bell, TrendingUp, Users, Smile } from "lucide-react";

export default function AgentClona() {
  const handlePatientSignup = () => {
    window.location.href = "/api/auth/login";
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <Link href="/">
            <div className="flex items-center gap-3 cursor-pointer">
              <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
                <Stethoscope className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">Followup AI</h1>
                <p className="text-xs text-muted-foreground">HIPAA-Compliant Health Platform</p>
              </div>
            </div>
          </Link>
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-2" data-testid="button-back-home">
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Button>
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-chart-2/20 text-chart-2 mb-6">
                <Heart className="h-4 w-4" />
                <span className="text-sm font-medium">Your Personal Health Companion</span>
              </div>
              <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">
                Agent Clona
              </h1>
              <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
                Your caring AI companion designed specifically for chronic care patients. 
                Clona provides daily check-ins, medication reminders, wellness support, and connects 
                you with your healthcare team—all while keeping you safe and informed.
              </p>
              <div className="flex gap-4">
                <Button size="lg" className="gap-2" onClick={handlePatientSignup} data-testid="button-get-started">
                  <Heart className="h-5 w-5" />
                  Get Started Free
                </Button>
                <Link href="/coming-soon">
                  <Button size="lg" variant="outline" data-testid="button-download">
                    Download App
                  </Button>
                </Link>
              </div>
            </div>

            <Card className="p-8 bg-chart-2/5 border-chart-2/20">
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/20 flex-shrink-0">
                    <Shield className="h-6 w-6 text-chart-2" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">24/7 Health Guardian</h3>
                    <p className="text-sm text-muted-foreground">
                      Continuous monitoring and instant alerts when something needs attention
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/20 flex-shrink-0">
                    <Smile className="h-6 w-6 text-chart-2" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Peace of Mind</h3>
                    <p className="text-sm text-muted-foreground">
                      Know you're being watched over by AI trained on chronic care conditions
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/20 flex-shrink-0">
                    <Users className="h-6 w-6 text-chart-2" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Connected Care</h3>
                    <p className="text-sm text-muted-foreground">
                      Your doctors see what you see, ensuring coordinated treatment
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Core Features */}
      <section className="py-16 px-6 bg-muted/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">How Agent Clona Supports You Every Day</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Clona is more than an app—she's your dedicated health partner who understands the unique 
              challenges of living with a compromised immune system.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/20 mb-4">
                  <Calendar className="h-6 w-6 text-chart-2" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Daily Health Check-Ins</h3>
                <p className="text-muted-foreground mb-4">
                  Start each day with a quick check-in. Clona asks about your symptoms, energy levels, 
                  and overall well-being—tracking patterns your doctors need to see.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Simple, conversational questions</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Visual symptom tracking with photos</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Trend insights over time</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/20 mb-4">
                  <Pill className="h-6 w-6 text-chart-2" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Smart Medication Management</h3>
                <p className="text-muted-foreground mb-4">
                  Never miss a dose. Clona reminds you when to take medications, tracks adherence, 
                  and alerts your doctor if there are concerns.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Personalized reminder schedule</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Drug interaction warnings</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Refill reminders</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/20 mb-4">
                  <MessageSquare className="h-6 w-6 text-chart-2" />
                </div>
                <h3 className="text-xl font-semibold mb-3">24/7 AI Companion</h3>
                <p className="text-muted-foreground mb-4">
                  Chat with Clona anytime about your health questions, concerns, or just to check in. 
                  She understands your condition and provides empathetic, evidence-based guidance.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Instant answers to health questions</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Emotional support and encouragement</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Knows when to escalate to your doctor</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/20 mb-4">
                  <Camera className="h-6 w-6 text-chart-2" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Visual Health Assessment</h3>
                <p className="text-muted-foreground mb-4">
                  Use your phone camera to document skin changes, rashes, or wounds. Clona's AI analyzes 
                  images and flags concerning patterns for your medical team.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>AI-powered image analysis</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Secure photo timeline</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Automatic doctor notification</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/20 mb-4">
                  <Moon className="h-6 w-6 text-chart-2" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Wellness & Mindfulness</h3>
                <p className="text-muted-foreground mb-4">
                  Healing isn't just physical. Access guided meditation, breathing exercises, 
                  gentle movement routines, and educational content tailored to your journey.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Guided meditation library</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Gentle exercise videos</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Stress management tools</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/20 mb-4">
                  <Activity className="h-6 w-6 text-chart-2" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Wearable Integration</h3>
                <p className="text-muted-foreground mb-4">
                  Connect your Fitbit, Apple Watch, or other devices. Clona analyzes heart rate, 
                  sleep, and activity data to spot early warning signs.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Apple Health & Google Fit sync</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Sleep quality tracking</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <span>Abnormal pattern alerts</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Your Typical Day with Agent Clona</h2>
            <p className="text-lg text-muted-foreground">
              See how Clona seamlessly integrates into your life
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-chart-2/20 mx-auto mb-4">
                  <span className="text-2xl font-bold text-chart-2">1</span>
                </div>
                <h3 className="font-semibold mb-2">Morning Check-In</h3>
                <p className="text-sm text-muted-foreground">
                  Start your day with a quick 2-minute health assessment. Take a photo if needed.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-chart-2/20 mx-auto mb-4">
                  <span className="text-2xl font-bold text-chart-2">2</span>
                </div>
                <h3 className="font-semibold mb-2">Medication Reminders</h3>
                <p className="text-sm text-muted-foreground">
                  Get gentle nudges throughout the day for each medication. Confirm when taken.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-chart-2/20 mx-auto mb-4">
                  <span className="text-2xl font-bold text-chart-2">3</span>
                </div>
                <h3 className="font-semibold mb-2">Wellness Activities</h3>
                <p className="text-sm text-muted-foreground">
                  Enjoy a 10-minute meditation or gentle stretch when you have time to relax.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-chart-2/20 mx-auto mb-4">
                  <span className="text-2xl font-bold text-chart-2">4</span>
                </div>
                <h3 className="font-semibold mb-2">Doctor Updates</h3>
                <p className="text-sm text-muted-foreground">
                  Your care team sees your progress automatically. They reach out if concerned.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Impact Section */}
      <section className="py-16 px-6 bg-muted/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Life-Changing Results</h2>
            <p className="text-lg text-muted-foreground">
              Real improvements in health and quality of life
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="text-center">
              <CardContent className="pt-8">
                <div className="text-5xl font-bold text-chart-2 mb-2">65%</div>
                <p className="text-sm text-muted-foreground mb-4">Better Adherence</p>
                <p className="text-sm">
                  Patients using Clona take medications on time consistently
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardContent className="pt-8">
                <div className="text-5xl font-bold text-chart-2 mb-2">50%</div>
                <p className="text-sm text-muted-foreground mb-4">Fewer ER Visits</p>
                <p className="text-sm">
                  Early detection prevents emergencies before they happen
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardContent className="pt-8">
                <div className="text-5xl font-bold text-chart-2 mb-2">4.8★</div>
                <p className="text-sm text-muted-foreground mb-4">User Rating</p>
                <p className="text-sm">
                  Patients love the peace of mind and support Clona provides
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Security Section */}
      <section className="py-16 px-6">
        <div className="max-w-4xl mx-auto">
          <Card className="p-8 bg-primary/5 border-primary/20">
            <div className="flex items-start gap-6">
              <div className="flex h-16 w-16 items-center justify-center rounded-lg bg-primary/10 flex-shrink-0">
                <Shield className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h3 className="text-2xl font-bold mb-3">Your Data is Safe with Us</h3>
                <p className="text-muted-foreground mb-4">
                  We take your privacy seriously. All health data is encrypted, HIPAA-compliant, 
                  and never shared without your explicit permission.
                </p>
                <ul className="grid md:grid-cols-2 gap-3 text-sm">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-primary flex-shrink-0" />
                    <span>End-to-end encryption</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-primary flex-shrink-0" />
                    <span>HIPAA compliant infrastructure</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-primary flex-shrink-0" />
                    <span>You control data sharing</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-primary flex-shrink-0" />
                    <span>SOC 2 Type II certified</span>
                  </li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-6 bg-chart-2/5">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Start Your Journey with Agent Clona Today</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Join thousands of chronic care patients who've found peace of mind and 
            better health outcomes with their AI companion.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Button size="lg" className="gap-2" onClick={handlePatientSignup} data-testid="button-cta-signup">
              <Heart className="h-5 w-5" />
              Get Started Free
            </Button>
            <Link href="/coming-soon">
              <Button size="lg" variant="outline" data-testid="button-cta-app">
                Download Mobile App
              </Button>
            </Link>
          </div>
          <p className="text-sm text-muted-foreground mt-6">
            Free to start • No credit card required • HIPAA Compliant
          </p>
        </div>
      </section>
    </div>
  );
}
