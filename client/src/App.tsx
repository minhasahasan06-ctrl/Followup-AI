import { Switch, Route, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ThemeToggle } from "@/components/ThemeToggle";
import { useAuth } from "@/hooks/useAuth";
import { isUnauthorizedError } from "@/lib/authUtils";
import NotFound from "@/pages/not-found";
import Landing from "@/pages/Landing";
import DoctorPortal from "@/pages/DoctorPortal";
import RoleSelection from "@/pages/RoleSelection";
import Dashboard from "@/pages/Dashboard";
import Chat from "@/pages/Chat";
import Wellness from "@/pages/Wellness";
import Files from "@/pages/Files";
import Profile from "@/pages/Profile";
import DoctorDashboard from "@/pages/DoctorDashboard";
import PatientReview from "@/pages/PatientReview";
import ResearchCenter from "@/pages/ResearchCenter";
import Counseling from "@/pages/Counseling";
import ConsentManagement from "@/pages/ConsentManagement";

function PatientRouter() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/chat" component={Chat} />
      <Route path="/wellness/:type?" component={Wellness} />
      <Route path="/counseling" component={Counseling} />
      <Route path="/consents" component={ConsentManagement} />
      <Route path="/files" component={Files} />
      <Route path="/profile" component={Profile} />
      <Route component={NotFound} />
    </Switch>
  );
}

function DoctorRouter() {
  return (
    <Switch>
      <Route path="/" component={DoctorDashboard} />
      <Route path="/doctor/patient/:id" component={PatientReview} />
      <Route path="/research" component={ResearchCenter} />
      <Route path="/chat" component={Chat} />
      <Route path="/counseling" component={Counseling} />
      <Route path="/profile" component={Profile} />
      <Route component={NotFound} />
    </Switch>
  );
}

function AuthenticatedApp() {
  const { user, isLoading } = useAuth();
  const [location] = useLocation();
  const style = {
    "--sidebar-width": "16rem",
    "--sidebar-width-icon": "3rem",
  };

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto mb-4" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // Allow access to public routes even when authenticated
  if (!user || location === "/doctor-portal") {
    return (
      <Switch>
        <Route path="/doctor-portal" component={DoctorPortal} />
        <Route path="/" component={Landing} />
        <Route component={Landing} />
      </Switch>
    );
  }

  if (!user.role) {
    return <RoleSelection />;
  }

  const isDoctor = user.role === "doctor";

  return (
    <SidebarProvider style={style as React.CSSProperties}>
      <div className="flex h-screen w-full">
        <AppSidebar />
        <div className="flex flex-col flex-1 min-w-0">
          <header className="flex items-center justify-between p-4 border-b bg-background">
            <SidebarTrigger data-testid="button-sidebar-toggle" />
            <ThemeToggle />
          </header>
          <main className="flex-1 overflow-auto p-6">
            {isDoctor ? <DoctorRouter /> : <PatientRouter />}
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="light">
        <TooltipProvider>
          <AuthenticatedApp />
          <Toaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
