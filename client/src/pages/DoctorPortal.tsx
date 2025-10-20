import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Stethoscope, Users, Bot, TrendingUp, Shield, CheckCircle, FileText, Info } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";

export default function DoctorPortal() {
  const { user } = useAuth();

  const handleDoctorSignup = () => {
    // Will redirect to Replit Auth with doctor context
    window.location.href = "/api/login?role=doctor&type=signup";
  };

  const handleDoctorLogin = () => {
    // Will redirect to Replit Auth with doctor context
    window.location.href = "/api/login?role=doctor&type=login";
  };

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
                <p className="text-xs text-muted-foreground">Doctor Portal</p>
              </div>
            </div>
          </Link>
          <div className="flex items-center gap-3 flex-wrap">
            <Link href="/">
              <Button variant="ghost" data-testid="button-patient-portal">
                Patient Portal
              </Button>
            </Link>
            <Button variant="outline" onClick={handleDoctorLogin} data-testid="button-doctor-login">
              Login
            </Button>
            <Button onClick={handleDoctorSignup} data-testid="button-doctor-signup">
              Sign Up
            </Button>
          </div>
        </div>
      </header>

      <main>
        {/* Alert for already logged in users */}
        {user && (
          <section className="py-6 px-6">
            <div className="max-w-6xl mx-auto">
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  You're currently logged in as a {user.role === "doctor" ? "doctor" : "patient"}. 
                  {user.role === "patient" && " To access the doctor portal, please log out first using the button in the sidebar, then sign up or log in as a doctor."}
                  {user.role === "doctor" && " You already have access to all doctor features through your dashboard."}
                </AlertDescription>
              </Alert>
            </div>
          </section>
        )}

        {/* Hero Section */}
        <section className="py-20 px-6">
          <div className="max-w-6xl mx-auto text-center">
            <div className="flex items-center justify-center gap-3 mb-6">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                <Users className="h-8 w-8 text-primary" />
              </div>
            </div>
            <h2 className="text-5xl font-bold tracking-tight mb-6">
              Advanced Clinical Tools for Healthcare Providers
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
              Access Assistant Lysa, manage your immunocompromised patients with AI-powered insights, 
              and contribute to groundbreaking epidemiological research.
            </p>
            <div className="flex gap-4 justify-center flex-wrap">
              <Button size="lg" onClick={handleDoctorSignup} data-testid="button-doctor-signup-hero">
                Create Doctor Account
              </Button>
              <Button size="lg" variant="outline" onClick={handleDoctorLogin} data-testid="button-doctor-login-hero">
                Sign In
              </Button>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-16 px-6 bg-muted/50">
          <div className="max-w-7xl mx-auto">
            <h3 className="text-3xl font-semibold text-center mb-12">Powerful Tools for Modern Medicine</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <Bot className="h-10 w-10 text-primary mb-3" />
                  <CardTitle>Assistant Lysa</CardTitle>
                  <CardDescription>Your AI Clinical Partner</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    GPT-4 powered insights with pattern recognition, medical entity extraction, 
                    and evidence-based clinical recommendations.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <Users className="h-10 w-10 text-primary mb-3" />
                  <CardTitle>Patient Management</CardTitle>
                  <CardDescription>Comprehensive Care Coordination</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Review patient timelines, daily follow-ups, medication adherence, and AI-flagged 
                    health concerns in one unified dashboard.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <TrendingUp className="h-10 w-10 text-primary mb-3" />
                  <CardTitle>Research Center</CardTitle>
                  <CardDescription>Epidemiological Intelligence</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    AI research agent with automated report generation, correlation analysis, 
                    and population health insights.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <FileText className="h-10 w-10 text-primary mb-3" />
                  <CardTitle>Clinical Documentation</CardTitle>
                  <CardDescription>Streamlined Workflow</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Access patient lab results, imaging reports, and auto-generated health journals 
                    with intelligent search and filtering.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <Shield className="h-10 w-10 text-primary mb-3" />
                  <CardTitle>HIPAA Compliance</CardTitle>
                  <CardDescription>Security First</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    End-to-end encryption, audit logs, role-based access control, 
                    and full compliance with healthcare privacy regulations.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CheckCircle className="h-10 w-10 text-primary mb-3" />
                  <CardTitle>License Verification</CardTitle>
                  <CardDescription>Professional Credentials</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Automated medical license verification and optional LinkedIn profile 
                    integration for credentialing.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section className="py-16 px-6">
          <div className="max-w-5xl mx-auto">
            <h3 className="text-3xl font-semibold text-center mb-12">How It Works</h3>
            <div className="space-y-6">
              <Card>
                <CardHeader className="flex flex-row items-center gap-4 space-y-0">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground text-xl font-bold flex-shrink-0">
                    1
                  </div>
                  <div>
                    <CardTitle>Create Your Account</CardTitle>
                    <CardDescription>Sign up with your professional credentials</CardDescription>
                  </div>
                </CardHeader>
                <CardContent className="pl-20">
                  <p className="text-muted-foreground">
                    Register using your email or continue with Google/GitHub. Provide your medical license 
                    number for verification. Optionally link your LinkedIn profile for enhanced credentialing.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center gap-4 space-y-0">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground text-xl font-bold flex-shrink-0">
                    2
                  </div>
                  <div>
                    <CardTitle>License Verification</CardTitle>
                    <CardDescription>Automated credential validation</CardDescription>
                  </div>
                </CardHeader>
                <CardContent className="pl-20">
                  <p className="text-muted-foreground">
                    Our system verifies your medical license through secure databases. Once approved, 
                    you gain full access to patient data and research tools.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center gap-4 space-y-0">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground text-xl font-bold flex-shrink-0">
                    3
                  </div>
                  <div>
                    <CardTitle>Start Managing Patients</CardTitle>
                    <CardDescription>Access comprehensive care tools</CardDescription>
                  </div>
                </CardHeader>
                <CardContent className="pl-20">
                  <p className="text-muted-foreground">
                    Review patient timelines, chat with Assistant Lysa for clinical insights, access the 
                    Research Center, and coordinate care for your immunocompromised patients.
                  </p>
                </CardContent>
              </Card>
            </div>

            <div className="mt-12 text-center">
              <Button size="lg" onClick={handleDoctorSignup} data-testid="button-doctor-signup-cta">
                Get Started as a Doctor
              </Button>
              <p className="text-sm text-muted-foreground mt-4">
                Already have an account?{" "}
                <button onClick={handleDoctorLogin} className="text-primary hover:underline" data-testid="link-doctor-login">
                  Sign in here
                </button>
              </p>
            </div>
          </div>
        </section>

        {/* Requirements Section */}
        <section className="py-16 px-6 bg-muted/50">
          <div className="max-w-4xl mx-auto">
            <h3 className="text-3xl font-semibold text-center mb-8">Requirements</h3>
            <Card>
              <CardContent className="pt-6">
                <ul className="space-y-3">
                  <li className="flex items-start gap-3">
                    <CheckCircle className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">Valid Medical License</p>
                      <p className="text-sm text-muted-foreground">
                        Must hold an active medical license in good standing
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <CheckCircle className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">Professional Email</p>
                      <p className="text-sm text-muted-foreground">
                        Work email address for account verification
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <CheckCircle className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">HIPAA Training (Recommended)</p>
                      <p className="text-sm text-muted-foreground">
                        Familiarity with healthcare privacy regulations
                      </p>
                    </div>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>
      </main>

      <footer className="border-t py-8 px-6 bg-muted/30">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-3">
            <div className="h-2 w-2 rounded-full bg-chart-2" />
            <span className="text-sm text-muted-foreground">HIPAA Compliant • FDA/CE Pending</span>
          </div>
          <p className="text-xs text-muted-foreground">
            © 2025 Followup AI. For healthcare providers only. Unauthorized access is prohibited.
          </p>
        </div>
      </footer>
    </div>
  );
}
