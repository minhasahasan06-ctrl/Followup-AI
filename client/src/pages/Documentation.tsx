import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, Stethoscope, Book, Code, Activity, MessageSquare, Camera, Database, Zap, Shield } from "lucide-react";

export default function Documentation() {
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
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/20 text-primary mb-6">
            <Book className="h-5 w-5" />
            <span className="font-semibold">Developer Resources</span>
          </div>
          <h1 className="text-5xl font-bold mb-6">Documentation</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Complete guides and API references to integrate Followup AI into your healthcare workflow.
          </p>
        </div>
      </section>

      {/* Quick Start */}
      <section className="py-12 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Quick Start Guides</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="hover-elevate cursor-pointer" data-testid="card-patient-guide">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-2/20">
                    <Activity className="h-5 w-5 text-chart-2" />
                  </div>
                  <span>Patient Guide</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Learn how to set up Agent Clona, complete daily check-ins, and manage your health data.
                </p>
                <ul className="text-sm space-y-2">
                  <li>• Creating your account</li>
                  <li>• Daily health check-ins</li>
                  <li>• Medication tracking</li>
                  <li>• Connecting wearables</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="hover-elevate cursor-pointer" data-testid="card-doctor-guide">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <Stethoscope className="h-5 w-5 text-primary" />
                  </div>
                  <span>Doctor Guide</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Get started with Assistant Lysa, manage patient panels, and leverage AI insights.
                </p>
                <ul className="text-sm space-y-2">
                  <li>• Setting up your practice</li>
                  <li>• Patient enrollment</li>
                  <li>• AI-powered analytics</li>
                  <li>• Documentation automation</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="hover-elevate cursor-pointer" data-testid="card-api-guide">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <Code className="h-5 w-5 text-primary" />
                  </div>
                  <span>API Integration</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Integrate Followup AI with your existing systems using our RESTful API.
                </p>
                <ul className="text-sm space-y-2">
                  <li>• Authentication & API keys</li>
                  <li>• FHIR integration</li>
                  <li>• Webhooks & events</li>
                  <li>• Rate limits & best practices</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Core Features */}
      <section className="py-12 px-6 bg-muted/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Core Features Documentation</h2>
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <MessageSquare className="h-6 w-6 text-primary" />
                  <span>AI Chat & Conversation</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Both Agent Clona and Assistant Lysa use GPT-4-powered conversational AI to provide contextual, 
                  empathetic responses. The chat system maintains conversation history and understands medical context.
                </p>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <p className="text-xs font-mono mb-2 text-muted-foreground">Example: Patient Chat Flow</p>
                  <code className="text-xs block whitespace-pre-wrap">
{`1. Patient: "I've had a fever for 2 days"
2. Clona: "I understand. Let's track this carefully. 
   What's your current temperature?"
3. Patient: "101.5°F"
4. Clona: [Analyzes pattern, checks history]
   "Your temperature is elevated. Given your 
   immunocompromised status, I'm alerting 
   Dr. Smith. Can you take a photo of your 
   thermometer reading?"`}
                  </code>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Camera className="h-6 w-6 text-primary" />
                  <span>Camera-Based Health Assessment</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Patients can document symptoms visually using their phone camera. AI analyzes images for 
                  concerning patterns and creates a visual timeline for doctors.
                </p>
                <ul className="space-y-2 text-sm">
                  <li className="flex gap-2">
                    <span className="text-primary font-semibold">•</span>
                    <span><strong>Supported:</strong> Skin rashes, wounds, surgical sites, medication bottles, thermometer readings</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary font-semibold">•</span>
                    <span><strong>AI Detection:</strong> Color changes, size progression, infection signs</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary font-semibold">•</span>
                    <span><strong>Privacy:</strong> All images encrypted, only shared with authorized providers</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Activity className="h-6 w-6 text-primary" />
                  <span>Wearable Device Integration</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Seamlessly connect Apple Watch, Fitbit, and other wearables to provide continuous health monitoring.
                </p>
                <div className="grid md:grid-cols-2 gap-4 mt-4">
                  <div>
                    <h4 className="font-semibold text-sm mb-2">Supported Metrics</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Heart rate & variability</li>
                      <li>• Sleep duration & quality</li>
                      <li>• Daily step count</li>
                      <li>• Blood oxygen (SpO2)</li>
                      <li>• Activity levels</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold text-sm mb-2">AI Analysis</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Abnormal pattern detection</li>
                      <li>• Baseline comparison</li>
                      <li>• Early warning alerts</li>
                      <li>• Trend visualization</li>
                      <li>• Correlation analysis</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Database className="h-6 w-6 text-primary" />
                  <span>EHR Integration (FHIR)</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Connect with Epic, Cerner, Allscripts, and other FHIR-compliant systems to sync patient data seamlessly.
                </p>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <p className="text-xs font-mono mb-2 text-muted-foreground">Supported FHIR Resources</p>
                  <code className="text-xs block">
{`Patient, Observation, Condition, MedicationRequest,
Encounter, AllergyIntolerance, Immunization,
DiagnosticReport, Procedure`}
                  </code>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Zap className="h-6 w-6 text-primary" />
                  <span>Real-Time Alerts & Notifications</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Intelligent alert system notifies doctors when patients need attention based on AI-detected patterns.
                </p>
                <div className="space-y-3 mt-4">
                  <div className="flex gap-3">
                    <div className="w-2 h-2 rounded-full bg-red-500 mt-1.5 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-semibold">Critical Alert</p>
                      <p className="text-xs text-muted-foreground">Immediate attention required (fever spike, severe symptoms)</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="w-2 h-2 rounded-full bg-yellow-500 mt-1.5 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-semibold">Warning Alert</p>
                      <p className="text-xs text-muted-foreground">Review within 24 hours (trend changes, missed medications)</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="w-2 h-2 rounded-full bg-blue-500 mt-1.5 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-semibold">Info Alert</p>
                      <p className="text-xs text-muted-foreground">FYI updates (completed check-in, positive progress)</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Security & Compliance */}
      <section className="py-12 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Security & Compliance</h2>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <Shield className="h-6 w-6 text-primary" />
                <span>HIPAA Compliance & Data Security</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3">Encryption</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• AES-256 encryption at rest</li>
                    <li>• TLS 1.3 for data in transit</li>
                    <li>• End-to-end encrypted messaging</li>
                    <li>• Encrypted database backups</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-3">Access Control</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• Role-based permissions (RBAC)</li>
                    <li>• Multi-factor authentication (MFA)</li>
                    <li>• Session management & timeouts</li>
                    <li>• Audit logging for all access</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-3">Compliance</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• HIPAA compliant infrastructure</li>
                    <li>• SOC 2 Type II certified</li>
                    <li>• FDA/CE certification pending</li>
                    <li>• Regular security audits</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-3">Data Governance</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• Patient consent management</li>
                    <li>• Data retention policies</li>
                    <li>• Right to deletion (GDPR)</li>
                    <li>• Data portability support</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Research Center */}
      <section className="py-12 px-6 bg-muted/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Epidemiology Research Center</h2>
          <Card>
            <CardContent className="pt-6">
              <p className="text-muted-foreground mb-6">
                Access de-identified patient data for population health analysis, disease trend identification, 
                and clinical research. All data is anonymized and compliant with research ethics standards.
              </p>
              <div className="grid md:grid-cols-3 gap-6">
                <div>
                  <h4 className="font-semibold mb-3">Data Sources</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• Patient registries</li>
                    <li>• EHR integration</li>
                    <li>• Wearable device data</li>
                    <li>• PubMed integration</li>
                    <li>• PhysioNet datasets</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-3">Analytics Tools</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• Cohort building</li>
                    <li>• Trend analysis</li>
                    <li>• Risk stratification</li>
                    <li>• GIS mapping</li>
                    <li>• Predictive modeling</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-3">Export Formats</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• CSV/Excel</li>
                    <li>• JSON/XML</li>
                    <li>• FHIR bundles</li>
                    <li>• R/Python scripts</li>
                    <li>• API access</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 px-6 bg-primary/5">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Need More Help?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Access our full API documentation or contact our developer support team.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Link href="/api">
              <Button size="lg" data-testid="button-api-docs">
                View API Documentation
              </Button>
            </Link>
            <a href="mailto:t@followupai.io">
              <Button size="lg" variant="outline" data-testid="button-developer-support">
                Contact Developer Support
              </Button>
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}
