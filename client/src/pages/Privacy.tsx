import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Stethoscope, Shield, Lock, Eye } from "lucide-react";

export default function Privacy() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
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

      <main className="max-w-4xl mx-auto px-6 py-12">
        <h1 className="text-4xl font-bold mb-2">Privacy Policy</h1>
        <p className="text-muted-foreground mb-8">Last Updated: January 2025</p>

        <div className="prose prose-slate dark:prose-invert max-w-none space-y-8">
          <section>
            <h2 className="text-2xl font-semibold mb-4">Our Commitment to Your Privacy</h2>
            <p className="text-muted-foreground leading-relaxed">
              Followup AI is committed to protecting the privacy and security of your health information. This Privacy 
              Policy explains how we collect, use, disclose, and safeguard your information in compliance with HIPAA 
              and other applicable privacy laws.
            </p>
          </section>

          <div className="grid md:grid-cols-3 gap-6 my-8">
            <div className="p-6 rounded-lg bg-muted/50 border">
              <Shield className="h-8 w-8 text-primary mb-3" />
              <h3 className="font-semibold mb-2">HIPAA Compliant</h3>
              <p className="text-sm text-muted-foreground">
                Full compliance with healthcare privacy regulations
              </p>
            </div>
            <div className="p-6 rounded-lg bg-muted/50 border">
              <Lock className="h-8 w-8 text-primary mb-3" />
              <h3 className="font-semibold mb-2">Encrypted Storage</h3>
              <p className="text-sm text-muted-foreground">
                End-to-end encryption for all health data
              </p>
            </div>
            <div className="p-6 rounded-lg bg-muted/50 border">
              <Eye className="h-8 w-8 text-primary mb-3" />
              <h3 className="font-semibold mb-2">Your Control</h3>
              <p className="text-sm text-muted-foreground">
                You decide what data to share and with whom
              </p>
            </div>
          </div>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Information We Collect</h2>
            <h3 className="text-lg font-semibold mt-6 mb-3">Health Information</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Daily health assessments (vital signs, symptoms, photos)</li>
              <li>Medication tracking and adherence data</li>
              <li>Wellness activity participation and progress</li>
              <li>Chat conversations with AI agents (Agent Clona and Assistant Lysa)</li>
              <li>Wearable device data (if you connect external health apps)</li>
            </ul>

            <h3 className="text-lg font-semibold mt-6 mb-3">Chat Session Storage</h3>
            <p className="text-muted-foreground leading-relaxed mb-3">
              All conversations with Agent Clona are automatically saved as medical history sessions. This includes:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Complete message transcripts between you and Agent Clona</li>
              <li>AI-generated session summaries for quick review</li>
              <li>Extracted medical entities (symptoms, medications discussed)</li>
              <li>Recommendations and health insights provided during conversations</li>
              <li>Consultation requests made during chat sessions</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-3">
              These sessions are stored securely with end-to-end encryption and can be reviewed by you and your assigned 
              doctors (with your consent). You can export or delete your chat history at any time from your profile settings.
            </p>

            <h3 className="text-lg font-semibold mt-6 mb-3">Account Information</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Email address and authentication credentials</li>
              <li>Name and profile information</li>
              <li>For doctors: Medical license number and verification details</li>
              <li>Subscription and billing information (for premium plans)</li>
              <li>Credit balance and transaction history (for consultation services)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">How We Use Your Information</h2>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>
                <strong>Personalized Health Support:</strong> Agent Clona uses your data to provide customized 
                health monitoring and wellness recommendations
              </li>
              <li>
                <strong>Doctor Access:</strong> With your explicit consent, assigned doctors can review your health 
                data through Assistant Lysa
              </li>
              <li>
                <strong>Platform Improvement:</strong> Aggregated, de-identified data helps us improve AI models 
                and platform features
              </li>
              <li>
                <strong>Research:</strong> Only with your explicit consent through our Research Consent program, 
                your anonymized data may be used for medical research
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Data Sharing and Disclosure</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              We never sell your personal health information. We only share your data in these circumstances:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><strong>With Your Consent:</strong> When you explicitly authorize data sharing</li>
              <li><strong>Healthcare Providers:</strong> With doctors you've assigned or authorized</li>
              <li>
                <strong>Consultation Services:</strong> When you request a paid consultation, your relevant chat history 
                and health data are shared with the consulting doctor for the duration of the session
              </li>
              <li>
                <strong>Doctor-to-Doctor Consultations:</strong> If your primary doctor requests a consultation with another 
                specialist, we require your explicit consent before sharing any health information with the consulting physician
              </li>
              <li><strong>Legal Requirements:</strong> When required by law or to protect health and safety</li>
              <li><strong>Service Providers:</strong> HIPAA-compliant vendors who help us operate the platform</li>
              <li><strong>Payment Processing:</strong> Stripe processes payments securely; they do not have access to your health data</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Your Privacy Rights</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">You have the right to:</p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Access your health information at any time</li>
              <li>Request corrections to your health records</li>
              <li>Download or export your data</li>
              <li>Revoke consent for data sharing or research participation</li>
              <li>Delete your account and associated data</li>
              <li>Receive a copy of this Privacy Policy</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Data Security</h2>
            <p className="text-muted-foreground leading-relaxed">
              We implement industry-standard security measures including:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground mt-3">
              <li>End-to-end encryption for data transmission and storage</li>
              <li>Secure authentication with Replit Auth</li>
              <li>Regular security audits and penetration testing</li>
              <li>HIPAA-compliant infrastructure and backup systems</li>
              <li>Employee training on privacy and security practices</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Consultation and Payment Services</h2>
            <h3 className="text-lg font-semibold mt-4 mb-3">Credit System</h3>
            <p className="text-muted-foreground leading-relaxed mb-3">
              Premium Agent Clona subscribers ($20/month) receive 20 consultation credits. Each 10-minute doctor consultation 
              costs 20 credits. Credits are non-refundable and expire with subscription cancellation.
            </p>
            <h3 className="text-lg font-semibold mt-4 mb-3">Doctor Earnings</h3>
            <p className="text-muted-foreground leading-relaxed mb-3">
              Doctors earn credits from patient consultations, which are securely stored in their account. Doctors can 
              withdraw earnings through Stripe payment gateway. All transactions are encrypted and HIPAA-compliant.
            </p>
            <h3 className="text-lg font-semibold mt-4 mb-3">Free Trial</h3>
            <p className="text-muted-foreground leading-relaxed">
              New users receive a 7-day free trial of premium features. Your payment method will be charged $20/month 
              after the trial unless you cancel. No data collected during your trial will be deleted upon cancellation.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Third-Party Integrations</h2>
            <p className="text-muted-foreground leading-relaxed">
              If you connect third-party health apps (Fitbit, Apple Health, Google Fit, etc.), you control what 
              data is shared through our Health Insight Consent Management system. You can revoke these permissions 
              at any time from your App Connections settings.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Contact Us</h2>
            <p className="text-muted-foreground leading-relaxed">
              For privacy-related questions, to exercise your rights, or to report a concern, please contact our 
              Privacy Officer at{" "}
              <a href="mailto:t@followupai.io" className="text-primary hover:underline">
                t@followupai.io
              </a>
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}
