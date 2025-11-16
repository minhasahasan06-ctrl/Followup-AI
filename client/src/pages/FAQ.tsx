import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ArrowLeft, Stethoscope, HelpCircle, MessageSquare, Shield, Users, CreditCard, Smartphone, Database } from "lucide-react";

export default function FAQ() {
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
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/20 text-primary mb-6">
            <HelpCircle className="h-5 w-5" />
            <span className="font-semibold">Help Center</span>
          </div>
          <h1 className="text-5xl font-bold mb-6">Frequently Asked Questions</h1>
          <p className="text-xl text-muted-foreground">
            Find answers to common questions about Followup AI, our AI agents, and how we support your health journey.
          </p>
        </div>
      </section>

      {/* FAQ Categories */}
      <section className="py-12 px-6">
        <div className="max-w-5xl mx-auto space-y-12">
          {/* General Questions */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <MessageSquare className="h-5 w-5 text-primary" />
              </div>
              <h2 className="text-2xl font-bold">General Questions</h2>
            </div>
            <div className="space-y-4">
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">What is Followup AI?</h3>
                  <p className="text-sm text-muted-foreground">
                    Followup AI is a HIPAA-compliant health platform designed specifically for immunocompromised patients. 
                    We feature two AI agents: Agent Clona (patient support) and Assistant Lysa (doctor assistance) that 
                    work together to provide comprehensive, continuous care monitoring and support.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Who should use Followup AI?</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Followup AI is ideal for:
                  </p>
                  <ul className="text-sm text-muted-foreground space-y-2 ml-4">
                    <li>• Immunocompromised patients requiring daily health monitoring</li>
                    <li>• Physicians managing high-risk patient populations</li>
                    <li>• Healthcare institutions seeking better patient outcomes</li>
                    <li>• Researchers studying chronic disease patterns and epidemiology</li>
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">What makes Followup AI different from other health apps?</h3>
                  <p className="text-sm text-muted-foreground">
                    Unlike generic health apps, Followup AI is purpose-built for immunocompromised patients with dual AI agents, 
                    real-time infection pattern detection, camera-based health assessments, wearable integration, and direct 
                    doctor-patient coordination—all while maintaining strict HIPAA compliance and enterprise-grade security.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* For Patients */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-2/20">
                <Users className="h-5 w-5 text-chart-2" />
              </div>
              <h2 className="text-2xl font-bold">For Patients</h2>
            </div>
            <div className="space-y-4">
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">How does Agent Clona help me?</h3>
                  <p className="text-sm text-muted-foreground">
                    Agent Clona provides daily check-ins, medication reminders, symptom tracking with photo documentation, 
                    24/7 health question support, wellness activities, and automatically alerts your doctor when patterns 
                    indicate potential complications. Think of Clona as your caring health companion who never sleeps.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Is Agent Clona free to use?</h3>
                  <p className="text-sm text-muted-foreground">
                    Yes! Agent Clona Basic is completely free and includes unlimited chat for common health questions and 
                    general wellness advice. For advanced features like personalized wellness insights, health pattern tracking, 
                    and trend analysis, upgrade to Premium for $19.99/month.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Can Agent Clona replace my doctor?</h3>
                  <p className="text-sm text-muted-foreground">
                    No. Agent Clona is designed to support—not replace—your healthcare team. While Clona provides guidance 
                    and monitoring, she works alongside your doctors and alerts them when professional medical attention is needed. 
                    Always consult your doctor for medical decisions.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Does my doctor see my data?</h3>
                  <p className="text-sm text-muted-foreground">
                    Only with your explicit consent. You control who sees your health data. When you connect with your doctor 
                    through Followup AI, they can access your check-ins, symptoms, and trends to provide better care. You can 
                    revoke access at any time.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* For Doctors */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Stethoscope className="h-5 w-5 text-primary" />
              </div>
              <h2 className="text-2xl font-bold">For Doctors</h2>
            </div>
            <div className="space-y-4">
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">What is Assistant Lysa?</h3>
                  <p className="text-sm text-muted-foreground">
                    Assistant Lysa is your AI clinical companion that analyzes patient data, identifies infection patterns, 
                    provides evidence-based insights, automates documentation, and helps you manage larger patient panels 
                    more effectively. Lysa saves physicians an average of 3.5 hours daily.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">How much does Assistant Lysa cost?</h3>
                  <p className="text-sm text-muted-foreground">
                    Assistant Lysa Premium operates on a consultation fee model—we charge 10% of each consultation fee. 
                    This aligns our success with yours. There are no upfront costs or monthly minimums.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Can I integrate Assistant Lysa with my existing EHR?</h3>
                  <p className="text-sm text-muted-foreground">
                    Yes! We support FHIR integration with major EHR systems including Epic, Cerner, and Allscripts. 
                    Contact our enterprise team for custom integration assistance.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">What is the Epidemiology Research Center?</h3>
                  <p className="text-sm text-muted-foreground">
                    Our Research Center provides AI-powered analytics tools for population health analysis, disease trend 
                    identification, and clinical research. Plans range from $29/month (Basic) to $299/month (Enterprise) 
                    with features like patient registries, GIS mapping, and automated cohort building.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Security & Privacy */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Shield className="h-5 w-5 text-primary" />
              </div>
              <h2 className="text-2xl font-bold">Security & Privacy</h2>
            </div>
            <div className="space-y-4">
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Is Followup AI HIPAA compliant?</h3>
                  <p className="text-sm text-muted-foreground">
                    Yes. We are fully HIPAA-compliant with end-to-end encryption (AES-256 at rest, TLS 1.3 in transit), 
                    comprehensive audit trails, role-based access controls, and SOC 2 Type II certification. Your health 
                    data security is our top priority.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">How is my data stored and protected?</h3>
                  <p className="text-sm text-muted-foreground">
                    All data is encrypted and stored on AWS Cloud infrastructure with multiple redundancy layers. We use 
                    industry-leading security practices, regular penetration testing, and maintain strict access controls. 
                    Your data is never sold or shared with third parties.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Can I export or delete my data?</h3>
                  <p className="text-sm text-muted-foreground">
                    Absolutely. You have full control over your data. You can export all your health information at any time 
                    in standard formats, and request complete data deletion if you choose to close your account.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Technical & Billing */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <CreditCard className="h-5 w-5 text-primary" />
              </div>
              <h2 className="text-2xl font-bold">Billing & Payments</h2>
            </div>
            <div className="space-y-4">
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">What payment methods do you accept?</h3>
                  <p className="text-sm text-muted-foreground">
                    We accept all major credit cards (Visa, Mastercard, American Express, Discover), debit cards, and 
                    wire transfers for enterprise plans. All transactions are processed securely through our PCI-compliant 
                    payment processor.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Can I cancel my subscription anytime?</h3>
                  <p className="text-sm text-muted-foreground">
                    Yes. You can cancel your subscription at any time with no cancellation fees. Your access will continue 
                    until the end of your current billing period, and you won't be charged again.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Do you offer refunds?</h3>
                  <p className="text-sm text-muted-foreground">
                    We offer a 14-day money-back guarantee for all premium plans. If you're not satisfied within the first 
                    14 days, contact our support team for a full refund.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Mobile & Platform */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Smartphone className="h-5 w-5 text-primary" />
              </div>
              <h2 className="text-2xl font-bold">Mobile & Platform</h2>
            </div>
            <div className="space-y-4">
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Is there a mobile app?</h3>
                  <p className="text-sm text-muted-foreground">
                    Our iOS and Android mobile apps are coming soon! In the meantime, our web platform is fully responsive 
                    and works seamlessly on mobile browsers. Sign up for our newsletter to be notified when the apps launch.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-3">Which devices and wearables are supported?</h3>
                  <p className="text-sm text-muted-foreground">
                    We integrate with Apple Watch, Fitbit, Google Fit, and other devices that sync with Apple Health or 
                    Google Fit. We continuously add support for new devices based on user demand.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* Still Have Questions */}
      <section className="py-16 px-6 bg-primary/5">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Still Have Questions?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Our support team is here to help. Reach out and we'll get back to you within 24 hours.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <a href="mailto:t@followupai.io">
              <Button size="lg" data-testid="button-contact-support">
                Contact Support
              </Button>
            </a>
            <Link href="/enterprise-contact">
              <Button size="lg" variant="outline" data-testid="button-enterprise-contact">
                Enterprise Inquiry
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
