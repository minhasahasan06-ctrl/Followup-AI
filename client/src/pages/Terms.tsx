import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Stethoscope } from "lucide-react";

export default function Terms() {
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
        <h1 className="text-4xl font-bold mb-2">Terms and Conditions</h1>
        <p className="text-muted-foreground mb-8">Last Updated: January 2025</p>

        <div className="prose prose-slate dark:prose-invert max-w-none space-y-8">
          <section>
            <h2 className="text-2xl font-semibold mb-4">1. Acceptance of Terms</h2>
            <p className="text-muted-foreground leading-relaxed">
              By accessing and using Followup AI, you accept and agree to be bound by the terms and provision of this agreement. 
              If you do not agree to these Terms and Conditions, please do not use our platform.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">2. Medical Disclaimer</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              Followup AI is a health monitoring and wellness platform designed to support immunocompromised patients. 
              Our AI agents (Agent Clona and Assistant Lysa) provide informational support only.
            </p>
            <p className="text-muted-foreground leading-relaxed font-semibold">
              This platform does not provide medical advice, diagnosis, or treatment. Always seek the advice of your 
              physician or other qualified health provider with any questions regarding a medical condition.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">3. User Accounts and Roles</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">
              <strong>Patient Accounts:</strong> Patients can access health tracking, Agent Clona, wellness modules, 
              and medication management features.
            </p>
            <p className="text-muted-foreground leading-relaxed">
              <strong>Doctor Accounts:</strong> Doctors must provide valid medical license information for verification. 
              Doctor accounts include access to Assistant Lysa, patient review dashboards, and research tools.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">4. Data Privacy and HIPAA Compliance</h2>
            <p className="text-muted-foreground leading-relaxed">
              We are committed to protecting your health information in accordance with HIPAA regulations. Your data is 
              encrypted, securely stored, and never shared without your explicit consent. Please review our Privacy Policy 
              for detailed information.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">5. Acceptable Use</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">You agree not to:</p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Use the platform for any unlawful purpose</li>
              <li>Attempt to gain unauthorized access to any part of the platform</li>
              <li>Share your account credentials with others</li>
              <li>Upload false or misleading health information</li>
              <li>Interfere with the proper functioning of the platform</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">6. AI-Generated Content and Medical Verification</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">
              Agent Clona and Assistant Lysa use artificial intelligence to provide health insights and support. 
              AI-generated content should be reviewed and verified by qualified healthcare professionals before 
              making any medical decisions.
            </p>
            <p className="text-muted-foreground leading-relaxed font-semibold">
              When Agent Clona suggests a treatment or medical recommendation, we strongly encourage you to verify this 
              information with a licensed healthcare provider. You can request an instant consultation through our platform 
              to connect with a doctor for professional medical advice.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">7. Chat Session Storage</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">
              All conversations with Agent Clona are automatically saved as medical history sessions to provide continuity 
              of care and enable doctors to better understand your health journey. By using the chat feature, you consent to:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Automatic storage of all chat messages and AI responses</li>
              <li>AI-generated summaries of your conversations for quick review</li>
              <li>Extraction of medical entities (symptoms, medications) for health tracking</li>
              <li>Access by doctors you authorize or consult with through the platform</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-3">
              You may request deletion of chat sessions from your profile settings at any time.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">8. Subscription, Payments, and Credits</h2>
            <h3 className="text-lg font-semibold mt-4 mb-3">Premium Agent Clona Subscription</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><strong>Price:</strong> $20/month billed monthly</li>
              <li><strong>Free Trial:</strong> 7-day free trial for new users; cancel anytime during trial with no charge</li>
              <li><strong>Inclusions:</strong> Unlimited AI conversations, wellness programs, 20 consultation credits per month</li>
              <li><strong>Renewal:</strong> Automatically renews monthly unless cancelled before renewal date</li>
              <li><strong>Refunds:</strong> No refunds for partial months; unused credits expire upon cancellation</li>
            </ul>

            <h3 className="text-lg font-semibold mt-6 mb-3">Consultation Credits</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Premium subscribers receive 20 consultation credits per month</li>
              <li>Each 10-minute doctor consultation costs 20 credits</li>
              <li>Credits are non-transferable and expire when subscription is cancelled</li>
              <li>Unused credits do not roll over to the next billing period</li>
            </ul>

            <h3 className="text-lg font-semibold mt-6 mb-3">Doctor Earnings and Withdrawals</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Doctors earn credits from completed consultations at a rate of $18 per 10-minute session</li>
              <li>Credits are securely stored in the doctor's account and can be viewed in their profile</li>
              <li>Withdrawals are processed through Stripe payment gateway with standard processing fees</li>
              <li>Minimum withdrawal amount is $50; processing time is 3-5 business days</li>
              <li>Followup AI retains a 10% platform fee to cover operational costs</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">9. Consultation Services</h2>
            <h3 className="text-lg font-semibold mt-4 mb-3">Patient-to-Doctor Consultations</h3>
            <p className="text-muted-foreground leading-relaxed mb-3">
              When you request a consultation through the platform:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>You consent to sharing relevant chat history and health data with the consulting doctor</li>
              <li>Consultations are scheduled based on doctor availability; not all requests can be immediately fulfilled</li>
              <li>Each consultation lasts 10 minutes; credits are deducted upon doctor acceptance</li>
              <li>Consultations are conducted via secure chat or video call (if enabled)</li>
              <li>Doctor responses do not constitute an official doctor-patient relationship unless explicitly established</li>
            </ul>

            <h3 className="text-lg font-semibold mt-6 mb-3">Doctor-to-Doctor Consultations</h3>
            <p className="text-muted-foreground leading-relaxed mb-3">
              If your doctor requests a consultation with another specialist about your case:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>You will receive a consent request before any health information is shared</li>
              <li>You have the right to approve or deny the consultation request</li>
              <li>If approved, only medically necessary information will be shared with the consulting physician</li>
              <li>All doctors involved are subject to HIPAA confidentiality requirements</li>
              <li>You can revoke consent at any time before the consultation occurs</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">10. Limitation of Liability</h2>
            <p className="text-muted-foreground leading-relaxed">
              Followup AI and its affiliates shall not be liable for any direct, indirect, incidental, special, 
              or consequential damages arising from your use of the platform. This includes but is not limited to 
              medical decisions made based on platform information, consultation outcomes, or technical issues 
              affecting service availability.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">11. Changes to Terms</h2>
            <p className="text-muted-foreground leading-relaxed">
              We reserve the right to modify these terms at any time. Users will be notified of significant changes 
              via email or platform notification. Continued use of the platform after changes constitutes acceptance 
              of modified terms.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">12. Contact Information</h2>
            <p className="text-muted-foreground leading-relaxed">
              For questions about these Terms and Conditions, please contact us at{" "}
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
