import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, Stethoscope, Code, Key, Zap, Database, Lock, Terminal } from "lucide-react";

export default function API() {
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
            <Terminal className="h-5 w-5" />
            <span className="font-semibold">Developer API</span>
          </div>
          <h1 className="text-5xl font-bold mb-6">Followup AI REST API</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8">
            Integrate patient monitoring, AI chat, health analytics, and EHR data into your applications 
            with our secure, HIPAA-compliant API.
          </p>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-muted text-sm font-mono">
            <Code className="h-4 w-4" />
            <span>https://api.followupai.io/v1</span>
          </div>
        </div>
      </section>

      {/* Quick Start */}
      <section className="py-12 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Quick Start</h2>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <Key className="h-6 w-6 text-primary" />
                <span>Authentication</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                All API requests require authentication using an API key. Include your key in the Authorization header.
              </p>
              <div className="bg-muted/50 p-4 rounded-lg">
                <p className="text-xs font-mono mb-2 text-muted-foreground">Example Request</p>
                <code className="text-xs block whitespace-pre-wrap font-mono">
{`curl -X GET https://api.followupai.io/v1/patients \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json"`}
                </code>
              </div>
              <div className="mt-4 p-4 bg-primary/5 border border-primary/20 rounded-lg">
                <p className="text-sm font-semibold mb-2">Get Your API Key</p>
                <p className="text-sm text-muted-foreground mb-3">
                  API keys are available for Enterprise and Research Center plans. Contact sales to get started.
                </p>
                <Link href="/enterprise-contact">
                  <Button size="sm" data-testid="button-get-api-key">
                    Request API Access
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Core Endpoints */}
      <section className="py-12 px-6 bg-muted/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Core API Endpoints</h2>
          <div className="space-y-6">
            {/* Patients */}
            <Card>
              <CardHeader>
                <CardTitle>Patients</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/patients</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">List all patients in your practice</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/patients/:id</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Get detailed patient information</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-semibold rounded">POST</span>
                      <code className="text-sm font-mono">/v1/patients</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Register a new patient</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-yellow-500/20 text-yellow-600 dark:text-yellow-400 text-xs font-semibold rounded">PATCH</span>
                      <code className="text-sm font-mono">/v1/patients/:id</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Update patient information</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Health Data */}
            <Card>
              <CardHeader>
                <CardTitle>Health Data</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/patients/:id/vitals</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Get patient vital signs history</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-semibold rounded">POST</span>
                      <code className="text-sm font-mono">/v1/patients/:id/vitals</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Record new vital signs</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/patients/:id/checkins</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Get daily check-in responses</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/patients/:id/photos</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Get health assessment photos timeline</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Medications */}
            <Card>
              <CardHeader>
                <CardTitle>Medications</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/patients/:id/medications</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">List patient medications</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-semibold rounded">POST</span>
                      <code className="text-sm font-mono">/v1/patients/:id/medications</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Add new medication to patient plan</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/patients/:id/adherence</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Get medication adherence stats</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* AI Chat */}
            <Card>
              <CardHeader>
                <CardTitle>AI Chat</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-semibold rounded">POST</span>
                      <code className="text-sm font-mono">/v1/chat/clona</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Send message to Agent Clona (patient AI)</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-semibold rounded">POST</span>
                      <code className="text-sm font-mono">/v1/chat/lysa</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Send message to Assistant Lysa (doctor AI)</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/patients/:id/conversations</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Get chat history for patient</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Analytics */}
            <Card>
              <CardHeader>
                <CardTitle>Analytics & Research</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-semibold rounded">POST</span>
                      <code className="text-sm font-mono">/v1/research/cohorts</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Create patient cohort for analysis</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/research/trends</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Get population health trends</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 text-xs font-semibold rounded">GET</span>
                      <code className="text-sm font-mono">/v1/research/risk-scores</code>
                    </div>
                    <p className="text-sm text-muted-foreground ml-16">Get AI-calculated risk stratification</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Example Request/Response */}
      <section className="py-12 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Example: Get Patient Vitals</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Request</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <code className="text-xs block whitespace-pre-wrap font-mono">
{`GET /v1/patients/12345/vitals?days=7
Authorization: Bearer sk_live_abc123...
Content-Type: application/json`}
                  </code>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Response</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <code className="text-xs block whitespace-pre-wrap font-mono">
{`{
  "patient_id": "12345",
  "vitals": [
    {
      "date": "2025-10-21",
      "temperature": 98.6,
      "heart_rate": 72,
      "blood_pressure": "120/80",
      "oxygen": 98
    },
    ...
  ],
  "ai_insights": {
    "status": "normal",
    "alerts": []
  }
}`}
                  </code>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Webhooks */}
      <section className="py-12 px-6 bg-muted/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Webhooks</h2>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <Zap className="h-6 w-6 text-primary" />
                <span>Real-Time Event Notifications</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                Configure webhooks to receive real-time notifications when important events occur.
              </p>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-sm mb-3">Available Events</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• <code className="text-xs bg-muted px-1 py-0.5 rounded">patient.checkin.completed</code></li>
                    <li>• <code className="text-xs bg-muted px-1 py-0.5 rounded">patient.alert.critical</code></li>
                    <li>• <code className="text-xs bg-muted px-1 py-0.5 rounded">patient.medication.missed</code></li>
                    <li>• <code className="text-xs bg-muted px-1 py-0.5 rounded">patient.photo.uploaded</code></li>
                    <li>• <code className="text-xs bg-muted px-1 py-0.5 rounded">patient.vitals.abnormal</code></li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-sm mb-3">Configuration</h4>
                  <div className="bg-muted/50 p-3 rounded-lg">
                    <code className="text-xs block whitespace-pre-wrap font-mono">
{`POST /v1/webhooks
{
  "url": "https://your-app/hook",
  "events": [
    "patient.alert.critical"
  ]
}`}
                    </code>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Rate Limits & Best Practices */}
      <section className="py-12 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Rate Limits & Best Practices</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Lock className="h-5 w-5 text-primary" />
                  <span>Rate Limits</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3 text-sm">
                  <li className="flex justify-between">
                    <span className="text-muted-foreground">Basic Tier</span>
                    <span className="font-semibold">100 req/min</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-muted-foreground">Professional Tier</span>
                    <span className="font-semibold">500 req/min</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-muted-foreground">Enterprise Tier</span>
                    <span className="font-semibold">Custom limits</span>
                  </li>
                </ul>
                <p className="text-xs text-muted-foreground mt-4">
                  Rate limit headers are included in every response for monitoring.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Database className="h-5 w-5 text-primary" />
                  <span>Best Practices</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Cache responses when possible</li>
                  <li>• Use webhooks for real-time updates</li>
                  <li>• Implement exponential backoff on errors</li>
                  <li>• Batch requests where supported</li>
                  <li>• Always validate webhook signatures</li>
                  <li>• Keep API keys secure and rotate regularly</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 px-6 bg-primary/5">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Build?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Get API access and start integrating Followup AI into your healthcare applications.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Link href="/enterprise-contact">
              <Button size="lg" data-testid="button-request-access">
                Request API Access
              </Button>
            </Link>
            <Link href="/documentation">
              <Button size="lg" variant="outline" data-testid="button-view-docs">
                View Full Documentation
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
