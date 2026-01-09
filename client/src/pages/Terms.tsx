import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Stethoscope } from "lucide-react";

export default function Terms() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      <header className="border-b bg-background/95 backdrop-blur sticky top-0 z-50">
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
        <p className="text-muted-foreground mb-8">Last Updated: January 2025 • Version: v2025-01</p>

        <div className="prose prose-slate dark:prose-invert max-w-none space-y-8">
          <section>
            <h2 className="text-2xl font-semibold mb-4">1. Acceptance of Terms</h2>
            <p className="text-muted-foreground leading-relaxed">
              By creating an account or using Followup AI, you agree to be bound by these Terms, our Privacy Policy, and other posted policies. If you do not agree, please do not use the Platform.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">2. Platform Description & Medical Disclaimer</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              <strong>Not medical advice:</strong> Followup AI is a wellness monitoring platform for chronic care patients. It is not a medical device and does not provide diagnosis, treatment, or clinical recommendations. AI outputs and trend analyses are informational only. Always consult a licensed healthcare professional for medical decisions. In emergencies, contact emergency services immediately.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">3. User Roles & Account Responsibilities</h2>
            <p className="text-muted-foreground leading-relaxed">
              Patient accounts can use daily follow-ups, medication tracking, Agent Clona, and research settings. Doctor accounts must provide valid license details and are responsible for professional conduct. Keep account credentials secure and report unauthorized access promptly.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">4. Data, Chat Sessions & Retention</h2>
            <p className="text-muted-foreground leading-relaxed">
              All chat conversations, transcripts, and AI summaries are stored as medical history to enable continuity of care. You may request export or deletion of your data, subject to legal obligations and the limits described in our Privacy Policy. Deletion cannot guarantee removal from backups or from aggregated, de-identified research datasets that have already been used.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">5. Research & Secondary Use</h2>
            <p className="text-muted-foreground leading-relaxed">
              Research participation is opt-in and managed in Settings. Data used for research is de-identified and processed with privacy safeguards (e.g., hashing, bucketing, and k-anonymity with k ≥ 25). We use Business Associate Agreements (BAA) when required. Withdrawals of consent stop future research use but cannot retroactively remove data from completed research or models trained before withdrawal.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">6. Privacy, HIPAA & Security</h2>
            <p className="text-muted-foreground leading-relaxed">
              We are committed to HIPAA-compliant data handling, AES-256 encryption at rest and in transit, regular audits, and access logging. We do not sell personal health information. For full details, see our Privacy Policy.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">7. Third-Party Integrations</h2>
            <p className="text-muted-foreground leading-relaxed">
              If you connect third-party apps (Fitbit, Apple Health), you control what is shared. External processors operate under BAAs when processing PHI; if operating in non-BAA mode, only heavily de-identified data is used.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">8. Subscriptions & Payments</h2>
            <p className="text-muted-foreground leading-relaxed">
              Premium Agent Clona subscribers are billed $20/month after a 7-day trial. Premium benefits and consultation credits are described in the platform; credits expire on subscription cancellation. Doctor earnings and withdrawals use Stripe, with a platform fee and payout policies described in the dashboard.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">9. Consultations</h2>
            <p className="text-muted-foreground leading-relaxed">
              Consultations share relevant health information only with your consent. Consultations do not automatically establish a formal doctor–patient relationship unless explicitly agreed by the parties.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">10. AI Insights</h2>
            <p className="text-muted-foreground leading-relaxed">
              AI-generated insights are informational and must be reviewed by a licensed clinician. We use reasonable efforts to indicate uncertainty in AI outputs but do not guarantee clinical accuracy.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">11. Acceptable Use</h2>
            <p className="text-muted-foreground leading-relaxed">
              Users must not use the Platform unlawfully, impersonate others, access other accounts, upload false information, or interfere with Platform operation.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">12. Liability & Indemnity</h2>
            <p className="text-muted-foreground leading-relaxed">
              To the fullest extent allowed by law, Followup AI is not liable for indirect or consequential damages arising from use of the Platform. Users agree to indemnify Followup AI for third-party claims arising from misuse.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">13. Termination</h2>
            <p className="text-muted-foreground leading-relaxed">
              We may suspend or terminate accounts for breaches. Data handling after termination follows the Privacy Policy and legal requirements.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">14. Changes to Terms</h2>
            <p className="text-muted-foreground leading-relaxed">
              We may modify Terms and will notify of significant changes. Continued use after notice constitutes acceptance.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">15. Governing Law & Contact</h2>
            <p className="text-muted-foreground leading-relaxed">
              These Terms are governed by the laws specified in our legal jurisdiction. Contact:{" "}
              <a href="mailto:t@followupai.io" className="text-primary hover:underline">
                t@followupai.io
              </a>
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">16. Recordkeeping</h2>
            <p className="text-muted-foreground leading-relaxed">
              We keep a record of Terms acceptance: user_id, terms_version, accepted_at, ip_address, and user_agent.
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}
