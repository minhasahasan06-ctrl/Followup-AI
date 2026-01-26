import { useState } from "react";
import { useLocation } from "wouter";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Heart, Users, Stethoscope } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";

export default function RoleSelection() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [selectedRole, setSelectedRole] = useState<"patient" | "doctor" | null>(null);
  const [medicalLicenseNumber, setMedicalLicenseNumber] = useState("");
  const [termsAccepted, setTermsAccepted] = useState(false);

  const selectRoleMutation = useMutation({
    mutationFn: async (data: { role: string; medicalLicenseNumber?: string; termsAccepted: boolean }) => {
      const res = await apiRequest("/api/user/select-role", { method: "POST", json: data });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/auth/user"] });
      toast({
        title: "Role selected successfully",
        description: "Welcome to Followup AI!",
      });
      setLocation("/");
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to select role",
        variant: "destructive",
      });
    },
  });

  const handleRoleSelect = (role: "patient" | "doctor") => {
    setSelectedRole(role);
  };

  const handleSubmit = () => {
    if (!selectedRole) {
      toast({
        title: "Please select a role",
        description: "Choose whether you are a patient or doctor",
        variant: "destructive",
      });
      return;
    }

    if (selectedRole === "doctor" && !medicalLicenseNumber.trim()) {
      toast({
        title: "Medical license required",
        description: "Please enter your medical license number",
        variant: "destructive",
      });
      return;
    }

    if (!termsAccepted) {
      toast({
        title: "Terms acceptance required",
        description: "Please accept the Terms and Conditions to continue",
        variant: "destructive",
      });
      return;
    }

    selectRoleMutation.mutate({
      role: selectedRole,
      medicalLicenseNumber: selectedRole === "doctor" ? medicalLicenseNumber : undefined,
      termsAccepted,
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
      <div className="w-full max-w-4xl">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Stethoscope className="h-7 w-7" />
            </div>
            <h1 className="text-3xl font-semibold">Followup AI</h1>
          </div>
          <h2 className="text-2xl font-semibold mb-2">Welcome! Let's get you started</h2>
          <p className="text-muted-foreground">Please select your role to customize your experience</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <Card
            className={`cursor-pointer transition-all hover-elevate ${
              selectedRole === "patient" ? "ring-2 ring-primary" : ""
            }`}
            onClick={() => handleRoleSelect("patient")}
            data-testid="card-select-patient"
          >
            <CardHeader>
              <div className="flex items-center gap-3 mb-2">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                  <Heart className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-xl">I'm a Patient</CardTitle>
              </div>
              <CardDescription>
                Access Agent Clona, daily follow-ups, wellness modules, and health tracking
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Daily health assessments</li>
                <li>• AI-powered health assistant</li>
                <li>• Medication tracking</li>
                <li>• Wellness and meditation</li>
              </ul>
            </CardContent>
          </Card>

          <Card
            className={`cursor-pointer transition-all hover-elevate ${
              selectedRole === "doctor" ? "ring-2 ring-primary" : ""
            }`}
            onClick={() => handleRoleSelect("doctor")}
            data-testid="card-select-doctor"
          >
            <CardHeader>
              <div className="flex items-center gap-3 mb-2">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                  <Users className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-xl">I'm a Doctor</CardTitle>
              </div>
              <CardDescription>
                Access Assistant Lysa, patient management, and research tools
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Patient review dashboard</li>
                <li>• AI clinical insights</li>
                <li>• Research center</li>
                <li>• Epidemiological analysis</li>
              </ul>
            </CardContent>
          </Card>
        </div>

        {selectedRole === "doctor" && (
          <Card className="mb-6" data-testid="card-license-input">
            <CardHeader>
              <CardTitle>Medical License Verification</CardTitle>
              <CardDescription>Please provide your medical license number for verification</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Label htmlFor="license">Medical License Number</Label>
                <Input
                  id="license"
                  placeholder="Enter your medical license number"
                  value={medicalLicenseNumber}
                  onChange={(e) => setMedicalLicenseNumber(e.target.value)}
                  data-testid="input-medical-license"
                />
                <p className="text-xs text-muted-foreground">
                  Your license will be verified before granting full access to patient data and research tools.
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {selectedRole && (
          <Card className="mb-6" data-testid="card-terms-conditions">
            <CardContent className="pt-6">
              <div className="flex items-start gap-3">
                <Checkbox
                  id="terms"
                  checked={termsAccepted}
                  onCheckedChange={(checked) => setTermsAccepted(checked as boolean)}
                  data-testid="checkbox-terms"
                />
                <div className="space-y-1">
                  <Label htmlFor="terms" className="text-sm font-normal cursor-pointer">
                    I agree to the{" "}
                    <a
                      href="/terms"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline font-medium"
                      data-testid="link-terms"
                    >
                      Terms and Conditions
                    </a>{" "}
                    and{" "}
                    <a
                      href="/privacy"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline font-medium"
                      data-testid="link-privacy"
                    >
                      Privacy Policy
                    </a>
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    By accepting, you acknowledge that you have read and understood our HIPAA-compliant privacy practices and terms of service.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="flex justify-center">
          <Button
            size="lg"
            onClick={handleSubmit}
            disabled={!selectedRole || !termsAccepted || selectRoleMutation.isPending}
            data-testid="button-continue"
          >
            {selectRoleMutation.isPending ? "Setting up your account..." : "Continue"}
          </Button>
        </div>
      </div>
    </div>
  );
}
