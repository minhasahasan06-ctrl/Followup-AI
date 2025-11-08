import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircle, XCircle, Loader2 } from "lucide-react";

export default function VerifyEmail() {
  const [, setLocation] = useLocation();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");

    if (!token) {
      setStatus("error");
      setMessage("Invalid verification link");
      return;
    }

    fetch(`/api/auth/verify-email?token=${token}`)
      .then(async (res) => {
        const data = await res.json();
        if (res.ok) {
          setStatus("success");
          setMessage(data.message || "Email verified successfully!");
        } else {
          setStatus("error");
          setMessage(data.message || "Verification failed");
        }
      })
      .catch(() => {
        setStatus("error");
        setMessage("An error occurred during verification");
      });
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl text-center">Email Verification</CardTitle>
          <CardDescription className="text-center">
            {status === "loading" && "Verifying your email..."}
            {status === "success" && "Verification successful"}
            {status === "error" && "Verification failed"}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-4">
          {status === "loading" && <Loader2 className="h-16 w-16 animate-spin text-primary" />}
          {status === "success" && <CheckCircle className="h-16 w-16 text-green-500" />}
          {status === "error" && <XCircle className="h-16 w-16 text-red-500" />}
          
          <p className="text-center text-muted-foreground">{message}</p>
          
          {status === "success" && (
            <Button onClick={() => setLocation("/login")} className="w-full" data-testid="button-go-to-login">
              Go to Login
            </Button>
          )}
          
          {status === "error" && (
            <Button onClick={() => setLocation("/")} variant="outline" className="w-full" data-testid="button-go-home">
              Go to Home
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
