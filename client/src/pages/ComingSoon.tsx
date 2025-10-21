import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Smartphone, ArrowLeft, Bell } from "lucide-react";

export default function ComingSoon() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center px-6">
      <div className="max-w-2xl mx-auto text-center">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-primary/10 mb-8">
          <Smartphone className="h-10 w-10 text-primary" />
        </div>
        
        <h1 className="text-5xl font-bold mb-4">Coming Soon</h1>
        <p className="text-xl text-muted-foreground mb-8">
          Our mobile apps for iOS and Android are currently in development. 
          We're working hard to bring Followup AI to your pocket!
        </p>

        <div className="bg-card border rounded-lg p-8 mb-8">
          <div className="flex items-center gap-3 mb-4">
            <Bell className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Get Notified</h2>
          </div>
          <p className="text-muted-foreground mb-4">
            Want to be the first to know when our mobile apps launch?
          </p>
          <p className="text-sm text-muted-foreground">
            Sign up for the web platform now and we'll notify you as soon as the mobile apps are available.
          </p>
        </div>

        <div className="flex gap-4 justify-center flex-wrap">
          <Link href="/">
            <Button size="lg" variant="outline" className="gap-2" data-testid="button-back-home">
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Button>
          </Link>
          <Link href="/signup/patient">
            <Button size="lg" data-testid="button-signup-web">
              Sign Up for Web Platform
            </Button>
          </Link>
        </div>

        <p className="text-sm text-muted-foreground mt-8">
          In the meantime, access all features through our web platform on any device.
        </p>
      </div>
    </div>
  );
}
