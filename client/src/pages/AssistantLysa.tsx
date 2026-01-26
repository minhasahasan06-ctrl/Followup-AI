import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ArrowLeft, Stethoscope, Brain, Clock, FileSearch, Users, TrendingUp, Shield, Sparkles, CheckCircle, Activity, MessageSquare, Database } from "lucide-react";

export default function AssistantLysa() {
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
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/50 text-accent-foreground mb-6">
                <Sparkles className="h-4 w-4" />
                <span className="text-sm font-medium">AI-Powered Clinical Assistant</span>
              </div>
              <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">
                Assistant Lysa
              </h1>
              <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
                Your intelligent clinical companion that transforms how you care for chronic care patients. 
                Lysa analyzes patient data, identifies patterns, and provides evidence-based insights—so you can 
                focus on what matters most: patient care.
              </p>
              <div className="flex gap-4">
                <Link href="/doctor-portal">
                  <Button size="lg" className="gap-2" data-testid="button-get-started">
                    <Brain className="h-5 w-5" />
                    Get Started
                  </Button>
                </Link>
                <Link href="/enterprise-contact">
                  <Button size="lg" variant="outline" data-testid="button-enterprise">
                    Enterprise Solutions
                  </Button>
                </Link>
              </div>
            </div>

            <Card className="p-8 bg-accent/5 border-accent/20">
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 flex-shrink-0">
                    <Clock className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Save 3+ Hours Daily</h3>
                    <p className="text-sm text-muted-foreground">
                      Automated chart reviews, trend analysis, and report generation
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 flex-shrink-0">
                    <TrendingUp className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Earlier Detection</h3>
                    <p className="text-sm text-muted-foreground">
                      AI identifies subtle infection patterns before they become critical
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 flex-shrink-0">
                    <Users className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Care More Patients</h3>
                    <p className="text-sm text-muted-foreground">
                      Efficient workflows let you manage 2x the patient load effectively
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
            <h2 className="text-3xl font-bold mb-4">How Assistant Lysa Transforms Your Practice</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Designed specifically for physicians managing chronic care patients, Lysa combines 
              AI intelligence with clinical expertise to elevate care quality.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mb-4">
                  <Activity className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Real-Time Patient Monitoring</h3>
                <p className="text-muted-foreground mb-4">
                  Lysa continuously analyzes vital signs, lab results, and daily check-ins from your patients. 
                  Get instant alerts when patterns indicate potential complications.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Automated vital sign trend analysis</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Infection pattern detection algorithms</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Priority-based patient queue</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mb-4">
                  <Brain className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">AI-Powered Insights</h3>
                <p className="text-muted-foreground mb-4">
                  Lysa reviews thousands of data points across your patient panel and surfaces actionable 
                  insights you might have missed.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Medication adherence tracking</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Treatment effectiveness analysis</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Risk stratification scoring</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mb-4">
                  <MessageSquare className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Clinical Conversation Partner</h3>
                <p className="text-muted-foreground mb-4">
                  Chat with Lysa about complex cases. She understands medical context and can help you 
                  think through clinical patterns and treatment considerations.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Evidence-based recommendations</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Drug interaction checking</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Clinical guideline references</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mb-4">
                  <FileSearch className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Automated Documentation</h3>
                <p className="text-muted-foreground mb-4">
                  Reduce documentation burden with AI-generated summaries, progress notes, and 
                  discharge planning—all HIPAA-compliant and ready for review.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Smart visit summaries</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Treatment plan templates</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>Billing code suggestions</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mb-4">
                  <Database className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Research & Analytics</h3>
                <p className="text-muted-foreground mb-4">
                  Access our Epidemiology Research Center to analyze population health trends, 
                  outcome patterns, and contribute to medical research.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>De-identified cohort analysis</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>PubMed & PhysioNet integration</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>AI-powered research insights</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mb-4">
                  <Shield className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">HIPAA-Compliant & Secure</h3>
                <p className="text-muted-foreground mb-4">
                  All data is encrypted end-to-end, with role-based access controls and 
                  comprehensive audit trails for complete peace of mind.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>AES-256 encryption at rest</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>TLS 1.3 for data in transit</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>SOC 2 Type II compliant</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Impact Section */}
      <section className="py-16 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Real Impact on Patient Outcomes</h2>
            <p className="text-lg text-muted-foreground">
              Assistant Lysa doesn't just save time—it saves lives
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="text-center">
              <CardContent className="pt-8">
                <div className="text-5xl font-bold text-primary mb-2">40%</div>
                <p className="text-sm text-muted-foreground mb-4">Faster Detection</p>
                <p className="text-sm">
                  Early identification of infection patterns in chronic care patients
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardContent className="pt-8">
                <div className="text-5xl font-bold text-primary mb-2">3.5hrs</div>
                <p className="text-sm text-muted-foreground mb-4">Time Saved Daily</p>
                <p className="text-sm">
                  Reduced documentation and chart review time per physician
                </p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardContent className="pt-8">
                <div className="text-5xl font-bold text-primary mb-2">2x</div>
                <p className="text-sm text-muted-foreground mb-4">Patient Capacity</p>
                <p className="text-sm">
                  Manage more patients without compromising care quality
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-6 bg-primary/5">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Transform Your Practice?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Join leading healthcare institutions already using Assistant Lysa to improve 
            patient outcomes and physician well-being.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Link href="/doctor-portal">
              <Button size="lg" className="gap-2" data-testid="button-cta-start">
                <Brain className="h-5 w-5" />
                Start Using Lysa
              </Button>
            </Link>
            <Link href="/enterprise-contact">
              <Button size="lg" variant="outline" data-testid="button-cta-contact">
                Contact Sales
              </Button>
            </Link>
          </div>
          <p className="text-sm text-muted-foreground mt-6">
            HIPAA Compliant • FDA/CE Certification Pending • SOC 2 Type II
          </p>
        </div>
      </section>
    </div>
  );
}
