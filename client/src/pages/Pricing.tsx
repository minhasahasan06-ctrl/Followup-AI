import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, Stethoscope, Check, TrendingUp, BarChart3 } from "lucide-react";

export default function Pricing() {
  const handlePatientSignup = () => {
    window.location.href = "/api/login";
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
      <section className="py-16 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-5xl font-bold mb-6">Sustainable Revenue Streams for Growth</h1>
          <p className="text-xl text-muted-foreground mb-12 max-w-3xl mx-auto">
            Choose the plan that fits your needs. From free basic access to advanced enterprise solutions.
          </p>
        </div>
      </section>

      {/* Patient Pricing Section */}
      <section className="py-12 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-chart-2/20 text-chart-2 mb-4">
              <TrendingUp className="h-5 w-5" />
              <span className="font-semibold">Patient Care Plans</span>
            </div>
            <h2 className="text-3xl font-bold mb-3">Agent Clona Premium</h2>
            <p className="text-muted-foreground">AI-powered health companion for immunocompromised patients</p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 mb-16">
            {/* Agent Clona Basic */}
            <Card className="relative">
              <CardHeader>
                <CardTitle className="text-center">
                  <div className="text-sm font-semibold text-muted-foreground mb-2">AGENT CLONA</div>
                  <div className="text-lg font-bold mb-3">Basic</div>
                  <div className="text-4xl font-bold mb-2">$0<span className="text-lg font-normal text-muted-foreground">/month</span></div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Unlimited chat for common health questions</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">General advice on wellness, nutrition and lifestyle</span>
                  </li>
                </ul>
                <Button className="w-full" onClick={handlePatientSignup} data-testid="button-basic-signup">
                  Get Started Free
                </Button>
              </CardContent>
            </Card>

            {/* Agent Clona Premium */}
            <Card className="relative border-primary shadow-lg">
              <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-primary text-primary-foreground text-xs font-semibold rounded-full">
                MOST POPULAR
              </div>
              <CardHeader>
                <CardTitle className="text-center">
                  <div className="text-sm font-semibold text-muted-foreground mb-2">AGENT CLONA</div>
                  <div className="text-lg font-bold mb-3">Premium</div>
                  <div className="text-4xl font-bold mb-2">$20<span className="text-lg font-normal text-muted-foreground">/month</span></div>
                  <div className="text-sm text-primary font-semibold">7-day free trial</div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm"><strong>20 consultation credits/month</strong> (1 doctor consultation)</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Unlimited AI chat with Agent Clona</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Daily health follow-ups and monitoring</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Personalized medication and lifestyle recommendations</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Wellness programs and counseling</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">AI-generated health summaries</span>
                  </li>
                </ul>
                <Button className="w-full" onClick={handlePatientSignup} data-testid="button-premium-signup">
                  Start Premium
                </Button>
              </CardContent>
            </Card>

          </div>
        </div>
      </section>

      {/* Research Center Pricing */}
      <section className="py-12 px-6 bg-muted/50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/20 text-primary mb-4">
              <BarChart3 className="h-5 w-5" />
              <span className="font-semibold">Research & Analytics</span>
            </div>
            <h2 className="text-3xl font-bold mb-3">Epidemiology Research Center</h2>
            <p className="text-muted-foreground">Advanced analytics and research tools for medical professionals</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Basic Research */}
            <Card>
              <CardHeader>
                <CardTitle className="text-center">
                  <div className="text-lg font-bold mb-3">Basic</div>
                  <div className="text-4xl font-bold mb-2">$29<span className="text-lg font-normal text-muted-foreground">/month</span></div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Digital patient registry for chronic diseases</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Automated data cleaning & basic analytics dashboard</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Simple trend reports (by patient, by clinic, by region)</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Exportable datasets for research teams</span>
                  </li>
                </ul>
                <Link href="/enterprise-contact">
                  <Button className="w-full" variant="outline" data-testid="button-research-basic">
                    Contact Sales
                  </Button>
                </Link>
              </CardContent>
            </Card>

            {/* Professional Research */}
            <Card className="relative border-primary shadow-lg">
              <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-primary text-primary-foreground text-xs font-semibold rounded-full">
                RECOMMENDED
              </div>
              <CardHeader>
                <CardTitle className="text-center">
                  <div className="text-lg font-bold mb-3">Professional</div>
                  <div className="text-4xl font-bold mb-2">$99<span className="text-lg font-normal text-muted-foreground">/month</span></div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">AI-based risk scoring & patient stratification</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">GIS mapping for disease distribution</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Wearable & IoT device integration for real-time monitoring</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Early warning alerts for worsening conditions at population level</span>
                  </li>
                </ul>
                <Link href="/enterprise-contact">
                  <Button className="w-full" data-testid="button-research-professional">
                    Contact Sales
                  </Button>
                </Link>
              </CardContent>
            </Card>

            {/* Enterprise Research */}
            <Card>
              <CardHeader>
                <CardTitle className="text-center">
                  <div className="text-lg font-bold mb-3">Enterprise</div>
                  <div className="text-4xl font-bold mb-2">$299<span className="text-lg font-normal text-muted-foreground">/month</span></div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">AI-driven disease progression forecasting (population & individual)</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Multi-source data fusion (EHRs, lifestyle, genomics)</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">Research-grade analytics with automated cohort building</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <Check className="h-5 w-5 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span className="text-sm">API access</span>
                  </li>
                </ul>
                <Link href="/enterprise-contact">
                  <Button className="w-full" variant="outline" data-testid="button-research-enterprise">
                    Contact Sales
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-16 px-6">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">Frequently Asked Questions</h2>
          <div className="space-y-6">
            <Card>
              <CardContent className="pt-6">
                <h3 className="font-semibold mb-2">Can I switch plans anytime?</h3>
                <p className="text-sm text-muted-foreground">
                  Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately, 
                  and billing is prorated.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <h3 className="font-semibold mb-2">Is there a free trial for premium plans?</h3>
                <p className="text-sm text-muted-foreground">
                  Yes! Agent Clona Premium includes a 7-day free trial. You can cancel anytime during the trial 
                  without being charged. Research Center plans include a demo upon request.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <h3 className="font-semibold mb-2">What payment methods do you accept?</h3>
                <p className="text-sm text-muted-foreground">
                  We accept all major credit cards, debit cards, and wire transfers for enterprise plans. 
                  All transactions are secure and HIPAA-compliant.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <h3 className="font-semibold mb-2">How do consultation credits work?</h3>
                <p className="text-sm text-muted-foreground">
                  Agent Clona Premium subscribers receive 20 consultation credits each month, which can be used 
                  for one 10-minute consultation with a licensed doctor. Credits expire when you cancel your subscription 
                  and don't roll over to the next billing period.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <h3 className="font-semibold mb-2">What about doctors?</h3>
                <p className="text-sm text-muted-foreground">
                  Doctors can access Assistant Lysa and all doctor features completely free. Doctors earn credits 
                  from patient consultations ($18 per 10-minute session) which can be withdrawn via Stripe. The platform 
                  retains a 10% fee to cover operational costs.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-6 bg-primary/5">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Get Started?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Choose your plan and start improving health outcomes today.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Button size="lg" onClick={handlePatientSignup} data-testid="button-cta-patient">
              Start as Patient
            </Button>
            <Link href="/doctor-portal">
              <Button size="lg" variant="outline" data-testid="button-cta-doctor">
                Start as Doctor
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
