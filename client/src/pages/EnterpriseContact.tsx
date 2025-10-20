import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, Stethoscope, Building2, Mail, User, Phone } from "lucide-react";
import { useState } from "react";
import { useToast } from "@/hooks/use-toast";

export default function EnterpriseContact() {
  const { toast } = useToast();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    organizationName: "",
    phone: "",
    message: ""
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Create email body with form data
    const emailBody = `
Enterprise Information Request

Name: ${formData.name}
Email: ${formData.email}
Organization: ${formData.organizationName}
Phone: ${formData.phone}

Message:
${formData.message}

---
This inquiry was submitted through the Followup AI Enterprise Contact form.
    `.trim();

    // Create mailto link with pre-filled data
    const mailtoLink = `mailto:t@followupai.io?subject=Enterprise Information Request from ${formData.organizationName}&body=${encodeURIComponent(emailBody)}`;
    
    // Open email client
    window.location.href = mailtoLink;
    
    toast({
      title: "Opening Email Client",
      description: "Your default email application will open with the pre-filled information.",
    });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
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
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Building2 className="h-6 w-6 text-primary" />
            </div>
            <h1 className="text-4xl font-bold">Enterprise Contact</h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Transform patient care at your institution with Followup AI. Fill out the form below and we'll get back to you shortly.
          </p>
        </div>

        <Card className="max-w-2xl mx-auto">
          <CardHeader>
            <CardTitle>Request Enterprise Information</CardTitle>
            <CardDescription>
              Tell us about your organization and how we can help improve patient outcomes together.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="name" className="flex items-center gap-2">
                  <User className="h-4 w-4 text-muted-foreground" />
                  Full Name *
                </Label>
                <Input
                  id="name"
                  name="name"
                  placeholder="Dr. Jane Smith"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  data-testid="input-name"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="email" className="flex items-center gap-2">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                  Email Address *
                </Label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  placeholder="jane.smith@hospital.org"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  data-testid="input-email"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="organizationName" className="flex items-center gap-2">
                  <Building2 className="h-4 w-4 text-muted-foreground" />
                  Organization Name *
                </Label>
                <Input
                  id="organizationName"
                  name="organizationName"
                  placeholder="Memorial Hospital Transplant Center"
                  value={formData.organizationName}
                  onChange={handleChange}
                  required
                  data-testid="input-organization"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="phone" className="flex items-center gap-2">
                  <Phone className="h-4 w-4 text-muted-foreground" />
                  Phone Number
                </Label>
                <Input
                  id="phone"
                  name="phone"
                  type="tel"
                  placeholder="+1 (555) 123-4567"
                  value={formData.phone}
                  onChange={handleChange}
                  data-testid="input-phone"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="message">
                  Additional Information
                </Label>
                <Textarea
                  id="message"
                  name="message"
                  placeholder="Tell us about your patient population, specific needs, or any questions you have..."
                  value={formData.message}
                  onChange={handleChange}
                  rows={5}
                  data-testid="input-message"
                />
                <p className="text-xs text-muted-foreground">
                  Let us know about your institution's needs, patient volume, or specific features you're interested in.
                </p>
              </div>

              <div className="pt-4">
                <Button 
                  type="submit" 
                  size="lg" 
                  className="w-full gap-2"
                  data-testid="button-submit-contact"
                >
                  <Mail className="h-4 w-4" />
                  Send Enterprise Inquiry
                </Button>
              </div>

              <p className="text-xs text-muted-foreground text-center">
                By submitting this form, your email client will open with the information pre-filled. 
                This ensures your inquiry is sent securely from your email address.
              </p>
            </form>
          </CardContent>
        </Card>

        {/* Features Reminder */}
        <div className="mt-12 grid md:grid-cols-3 gap-6">
          <Card className="text-center">
            <CardContent className="pt-6">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mx-auto mb-3">
                <Stethoscope className="h-6 w-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">HIPAA Compliant</h3>
              <p className="text-sm text-muted-foreground">
                Enterprise-grade security for sensitive patient data
              </p>
            </CardContent>
          </Card>

          <Card className="text-center">
            <CardContent className="pt-6">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mx-auto mb-3">
                <Building2 className="h-6 w-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">White-Label Options</h3>
              <p className="text-sm text-muted-foreground">
                Branded solutions for your institution
              </p>
            </CardContent>
          </Card>

          <Card className="text-center">
            <CardContent className="pt-6">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mx-auto mb-3">
                <Mail className="h-6 w-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">Dedicated Support</h3>
              <p className="text-sm text-muted-foreground">
                Direct access to our enterprise team
              </p>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
