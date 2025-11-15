import { Switch, Route, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ThemeToggle } from "@/components/ThemeToggle";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import NotFound from "@/pages/not-found";
import Landing from "@/pages/Landing";
import DoctorPortal from "@/pages/DoctorPortal";
import Login from "@/pages/Login";
import DoctorSignup from "@/pages/DoctorSignup";
import PatientSignup from "@/pages/PatientSignup";
import VerifyEmail from "@/pages/VerifyEmail";
import VerifyPhone from "@/pages/VerifyPhone";
import ForgotPassword from "@/pages/ForgotPassword";
import ResetPassword from "@/pages/ResetPassword";
import RoleSelection from "@/pages/RoleSelection";
import ComingSoon from "@/pages/ComingSoon";
import Terms from "@/pages/Terms";
import Privacy from "@/pages/Privacy";
import HIPAACompliance from "@/pages/HIPAACompliance";
import EnterpriseContact from "@/pages/EnterpriseContact";
import AssistantLysa from "@/pages/AssistantLysa";
import AgentClona from "@/pages/AgentClona";
import Pricing from "@/pages/Pricing";
import FAQ from "@/pages/FAQ";
import Documentation from "@/pages/Documentation";
import API from "@/pages/API";
import Blog from "@/pages/Blog";
import Dashboard from "@/pages/Dashboard";
import Chat from "@/pages/Chat";
import Wellness from "@/pages/Wellness";
import Files from "@/pages/Files";
import Profile from "@/pages/Profile";
import DoctorDashboard from "@/pages/DoctorDashboard";
import DoctorWellness from "@/pages/DoctorWellness";
import ReceptionistDashboard from "@/pages/ReceptionistDashboard";
import PatientReview from "@/pages/PatientReview";
import ResearchCenter from "@/pages/ResearchCenter";
import Counseling from "@/pages/Counseling";
import ConsentManagement from "@/pages/ConsentManagement";
import PreviousSessions from "@/pages/PreviousSessions";
import EHRIntegrations from "@/pages/EHRIntegrations";
import WearableDevices from "@/pages/WearableDevices";
import Referrals from "@/pages/Referrals";
import Wallet from "@/pages/Wallet";
import AdminVerification from "@/pages/AdminVerification";
import TwoFactorAuth from "@/pages/TwoFactorAuth";
import MedicalDocuments from "@/pages/MedicalDocuments";
import DrugInteractions from "@/pages/DrugInteractions";
import ImmuneMonitoring from "@/pages/ImmuneMonitoring";
import EnvironmentalRiskMap from "@/pages/EnvironmentalRiskMap";
import CorrelationInsights from "@/pages/CorrelationInsights";
import NutritionInsights from "@/pages/NutritionInsights";
import HealthCompanion from "@/pages/HealthCompanion";
import VoiceFollowups from "@/pages/VoiceFollowups";
import Habits from "@/pages/Habits";
import DoctorSearch from "@/pages/DoctorSearch";
import DoctorProfile from "@/pages/DoctorProfile";
import MyDoctors from "@/pages/MyDoctors";
import ConsultationRequests from "@/pages/ConsultationRequests";
import PainDetection from "@/pages/PainDetection";
import SymptomJournal from "@/pages/SymptomJournal";
import ExamCoach from "@/pages/ExamCoach";

function PatientRouter() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/chat" component={Chat} />
      <Route path="/pain-detection" component={PainDetection} />
      <Route path="/symptom-journal" component={SymptomJournal} />
      <Route path="/exam-coach" component={ExamCoach} />
      <Route path="/habits" component={Habits} />
      <Route path="/previous-sessions" component={PreviousSessions} />
      <Route path="/doctor-search" component={DoctorSearch} />
      <Route path="/doctor/:doctorId" component={DoctorProfile} />
      <Route path="/my-doctors" component={MyDoctors} />
      <Route path="/consultation-requests" component={ConsultationRequests} />
      <Route path="/wellness/:type?" component={Wellness} />
      <Route path="/counseling" component={Counseling} />
      <Route path="/consents" component={ConsentManagement} />
      <Route path="/ehr-integrations" component={EHRIntegrations} />
      <Route path="/wearables" component={WearableDevices} />
      <Route path="/immune-monitoring" component={ImmuneMonitoring} />
      <Route path="/environmental-risk" component={EnvironmentalRiskMap} />
      <Route path="/correlation-insights" component={CorrelationInsights} />
      <Route path="/nutrition-insights" component={NutritionInsights} />
      <Route path="/health-companion" component={HealthCompanion} />
      <Route path="/voice-followups" component={VoiceFollowups} />
      <Route path="/medical-documents" component={MedicalDocuments} />
      <Route path="/drug-interactions" component={DrugInteractions} />
      <Route path="/referrals" component={Referrals} />
      <Route path="/wallet" component={Wallet} />
      <Route path="/files" component={Files} />
      <Route path="/security/2fa" component={TwoFactorAuth} />
      <Route path="/profile" component={Profile} />
      <Route component={NotFound} />
    </Switch>
  );
}

function DoctorRouter() {
  return (
    <Switch>
      <Route path="/" component={DoctorDashboard} />
      <Route path="/receptionist" component={ReceptionistDashboard} />
      <Route path="/doctor-wellness" component={DoctorWellness} />
      <Route path="/doctor/patient/:id" component={PatientReview} />
      <Route path="/research" component={ResearchCenter} />
      <Route path="/chat" component={Chat} />
      <Route path="/counseling" component={Counseling} />
      <Route path="/referrals" component={Referrals} />
      <Route path="/wallet" component={Wallet} />
      <Route path="/admin/verify-doctors" component={AdminVerification} />
      <Route path="/security/2fa" component={TwoFactorAuth} />
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

  // Public routes that don't require auth
  const publicRoutes = ["/login", "/signup/doctor", "/signup/patient", "/verify-email", "/verify-phone", "/forgot-password", "/reset-password", "/doctor-portal", "/coming-soon", "/terms", "/privacy", "/hipaa", "/enterprise-contact", "/assistant-lysa", "/agent-clona", "/pricing", "/faq", "/documentation", "/api", "/blog"];
  const isPublicRoute = publicRoutes.includes(location) || (!user && location === "/");

  // Show loading while checking auth
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

  // Public routes available for everyone
  if (isPublicRoute) {
    return (
      <Switch>
        <Route path="/login" component={Login} />
        <Route path="/signup/doctor" component={DoctorSignup} />
        <Route path="/signup/patient" component={PatientSignup} />
        <Route path="/verify-email" component={VerifyEmail} />
        <Route path="/verify-phone" component={VerifyPhone} />
        <Route path="/forgot-password" component={ForgotPassword} />
        <Route path="/reset-password" component={ResetPassword} />
        <Route path="/doctor-portal" component={DoctorPortal} />
        <Route path="/coming-soon" component={ComingSoon} />
        <Route path="/terms" component={Terms} />
        <Route path="/privacy" component={Privacy} />
        <Route path="/hipaa" component={HIPAACompliance} />
        <Route path="/enterprise-contact" component={EnterpriseContact} />
        <Route path="/assistant-lysa" component={AssistantLysa} />
        <Route path="/agent-clona" component={AgentClona} />
        <Route path="/pricing" component={Pricing} />
        <Route path="/faq" component={FAQ} />
        <Route path="/documentation" component={Documentation} />
        <Route path="/api" component={API} />
        <Route path="/blog" component={Blog} />
        <Route path="/" component={Landing} />
        <Route component={Landing} />
      </Switch>
    );
  }
  
  // Not authenticated - redirect to login
  if (!user) {
    window.location.href = "/login";
    return null;
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
      <AuthProvider>
        <ThemeProvider defaultTheme="light">
          <TooltipProvider>
            <AuthenticatedApp />
            <Toaster />
          </TooltipProvider>
        </ThemeProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}
