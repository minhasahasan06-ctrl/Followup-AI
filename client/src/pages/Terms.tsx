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
            <h2 className="text-2xl font-semibold mb-4">6. AI-Generated Content</h2>
            <p className="text-muted-foreground leading-relaxed">
              Agent Clona and Assistant Lysa use artificial intelligence to provide health insights and support. 
              AI-generated content should be reviewed and verified by qualified healthcare professionals before 
              making any medical decisions.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">7. Limitation of Liability</h2>
            <p className="text-muted-foreground leading-relaxed">
              Followup AI and its affiliates shall not be liable for any direct, indirect, incidental, special, 
              or consequential damages arising from your use of the platform. This includes but is not limited to 
              medical decisions made based on platform information.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">8. Changes to Terms</h2>
            <p className="text-muted-foreground leading-relaxed">
              We reserve the right to modify these terms at any time. Users will be notified of significant changes 
              via email or platform notification. Continued use of the platform after changes constitutes acceptance 
              of modified terms.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">9. Contact Information</h2>
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
