import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Stethoscope, Shield, Lock, FileCheck, Eye, Server, AlertTriangle } from "lucide-react";

export default function HIPAACompliance() {
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
        <div className="flex items-center gap-3 mb-2">
          <Shield className="h-10 w-10 text-primary" />
          <h1 className="text-4xl font-bold">HIPAA Compliance</h1>
        </div>
        <p className="text-muted-foreground mb-8">
          How Followup AI protects your health information in accordance with federal regulations
        </p>

        <div className="prose prose-slate dark:prose-invert max-w-none space-y-8">
          <section className="bg-primary/5 border-l-4 border-primary p-6 rounded-r-lg">
            <h2 className="text-xl font-semibold mb-3 flex items-center gap-2">
              <Shield className="h-6 w-6 text-primary" />
              Our HIPAA Commitment
            </h2>
            <p className="text-muted-foreground leading-relaxed">
              Followup AI is designed and operated in full compliance with the Health Insurance Portability and 
              Accountability Act (HIPAA) of 1996 and its implementing regulations. We take our responsibility to 
              protect your Protected Health Information (PHI) seriously.
            </p>
          </section>

          <div className="grid md:grid-cols-2 gap-6 my-8">
            <div className="p-6 rounded-lg bg-card border">
              <Lock className="h-8 w-8 text-primary mb-3" />
              <h3 className="font-semibold mb-2">Administrative Safeguards</h3>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Security officer oversight</li>
                <li>• Staff training programs</li>
                <li>• Access control policies</li>
                <li>• Incident response procedures</li>
              </ul>
            </div>
            <div className="p-6 rounded-lg bg-card border">
              <Server className="h-8 w-8 text-primary mb-3" />
              <h3 className="font-semibold mb-2">Physical Safeguards</h3>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Secure data centers</li>
                <li>• Controlled facility access</li>
                <li>• Device security controls</li>
                <li>• Backup and recovery systems</li>
              </ul>
            </div>
            <div className="p-6 rounded-lg bg-card border">
              <FileCheck className="h-8 w-8 text-primary mb-3" />
              <h3 className="font-semibold mb-2">Technical Safeguards</h3>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• End-to-end encryption (AES-256)</li>
                <li>• Secure authentication</li>
                <li>• Audit logs and monitoring</li>
                <li>• Automatic session timeouts</li>
              </ul>
            </div>
            <div className="p-6 rounded-lg bg-card border">
              <Eye className="h-8 w-8 text-primary mb-3" />
              <h3 className="font-semibold mb-2">Privacy Controls</h3>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Minimum necessary access</li>
                <li>• Patient consent management</li>
                <li>• Data use agreements</li>
                <li>• De-identification procedures</li>
              </ul>
            </div>
          </div>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Protected Health Information (PHI)</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              Under HIPAA, PHI includes any health information that can identify you. At Followup AI, we protect:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Health assessments and vital signs</li>
              <li>Medical history and conditions</li>
              <li>Medication information</li>
              <li>Photographs and visual assessments</li>
              <li>Communication with healthcare providers</li>
              <li>AI-generated health insights and recommendations</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Business Associate Agreements</h2>
            <p className="text-muted-foreground leading-relaxed">
              All third-party vendors and service providers who may have access to PHI have signed HIPAA-compliant 
              Business Associate Agreements (BAAs). This includes our cloud infrastructure providers, AI service 
              providers, and data analytics partners.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Patient Rights Under HIPAA</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">You have the right to:</p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>
                <strong>Access:</strong> View and obtain copies of your health information
              </li>
              <li>
                <strong>Amendment:</strong> Request corrections to your health records
              </li>
              <li>
                <strong>Accounting:</strong> Receive a list of certain disclosures of your PHI
              </li>
              <li>
                <strong>Restriction:</strong> Request limits on uses and disclosures of your PHI
              </li>
              <li>
                <strong>Confidential Communication:</strong> Request communication through specific means
              </li>
              <li>
                <strong>Notice:</strong> Receive a copy of our Privacy Practices notice
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Data Encryption and Security</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">
              We employ multiple layers of security to protect your PHI:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><strong>In Transit:</strong> TLS 1.3 encryption for all data transmission</li>
              <li><strong>At Rest:</strong> AES-256 encryption for stored data</li>
              <li><strong>Database:</strong> Encrypted PostgreSQL with role-based access control</li>
              <li><strong>Backups:</strong> Encrypted, geographically distributed backups</li>
              <li><strong>Sessions:</strong> Secure session management with automatic timeouts</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Breach Notification</h2>
            <p className="text-muted-foreground leading-relaxed">
              In the unlikely event of a breach of unsecured PHI, we will notify affected individuals within 60 days 
              as required by HIPAA. We maintain comprehensive incident response procedures and conduct regular security 
              assessments to prevent breaches.
            </p>
          </section>

          <section className="bg-amber-500/10 border-l-4 border-amber-500 p-6 rounded-r-lg">
            <h2 className="text-xl font-semibold mb-3 flex items-center gap-2">
              <AlertTriangle className="h-6 w-6 text-amber-600" />
              Report a Privacy Concern
            </h2>
            <p className="text-muted-foreground leading-relaxed">
              If you believe your privacy rights have been violated or you have concerns about our privacy practices, 
              please contact our Privacy Officer immediately at{" "}
              <a href="mailto:t@followupai.io" className="text-primary hover:underline font-medium">
                t@followupai.io
              </a>
              . You also have the right to file a complaint with the U.S. Department of Health and Human Services.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Compliance Certifications</h2>
            <div className="flex items-center gap-2 text-muted-foreground">
              <div className="h-2 w-2 rounded-full bg-chart-2" />
              <span>HIPAA Compliant</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground mt-2">
              <div className="h-2 w-2 rounded-full bg-chart-4" />
              <span>FDA/CE Certification Pending</span>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Questions?</h2>
            <p className="text-muted-foreground leading-relaxed">
              For questions about HIPAA compliance or our privacy practices, contact our Privacy Officer at{" "}
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
