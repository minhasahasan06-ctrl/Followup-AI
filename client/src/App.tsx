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
import AgentHub from "@/pages/AgentHub";
import DemoFlow from "@/pages/DemoFlow";
import Wellness from "@/pages/Wellness";
import Files from "@/pages/Files";
import Profile from "@/pages/Profile";
import ReceptionistDashboard from "@/pages/ReceptionistDashboard";
import CalendarSync from "@/pages/CalendarSync";
import PatientReview from "@/pages/PatientReview";
import ResearchCenter from "@/pages/ResearchCenter";
import ConsentManagement from "@/pages/ConsentManagement";
import ResearchConsentSettings from "@/pages/ResearchConsentSettings";
import WearableDevices from "@/pages/WearableDevices";
import DeviceConnect from "@/pages/DeviceConnect";
import Referrals from "@/pages/Referrals";
import Wallet from "@/pages/Wallet";
import AdminVerification from "@/pages/AdminVerification";
import TwoFactorAuth from "@/pages/TwoFactorAuth";
import PatientRecords from "@/pages/PatientRecords";
import RiskExposures from "@/pages/RiskExposures";
import CorrelationInsights from "@/pages/CorrelationInsights";
import NutritionInsights from "@/pages/NutritionInsights";
import HealthCompanion from "@/pages/HealthCompanion";
import VoiceFollowups from "@/pages/VoiceFollowups";
import Habits from "@/pages/Habits";
import DoctorSearch from "@/pages/DoctorSearch";
import DoctorProfile from "@/pages/DoctorProfile";
import MyDoctors from "@/pages/MyDoctors";
import ConsultationRequests from "@/pages/ConsultationRequests";
import VideoConsultation from "@/pages/VideoConsultation";
import PainDetection from "@/pages/PainDetection";
import Medications from "@/pages/Medications";
import DoctorMedicationReport from "@/pages/DoctorMedicationReport";
import DeteriorationDashboard from "@/pages/DeteriorationDashboard";
import AIAlertsDashboard from "@/pages/AIAlertsDashboard";
import BehavioralAIInsights from "@/pages/BehavioralAIInsights";
import AIVideoDashboard from "@/pages/AIVideoDashboard";
import AIAudioDashboard from "@/pages/AIAudioDashboard";
import GuidedVideoExam from "@/pages/GuidedVideoExam";
import GuidedAudioExam from "@/pages/guided-audio-exam";
import DailyFollowup from "@/pages/DailyFollowup";
import DailyFollowupHistory from "@/pages/DailyFollowupHistory";
import MentalHealth from "@/pages/MentalHealth";
import Prescriptions from "@/pages/Prescriptions";
import MLMonitoring from "@/pages/MLMonitoring";
import MLInsightsPage from "@/pages/MLInsightsPage";
import AdminMLTrainingHub from "@/pages/AdminMLTrainingHub";
import MedicalNLPDashboard from "@/pages/MedicalNLPDashboard";
import TinkerDashboard from "@/pages/TinkerDashboard";
import PatientPrivacyResearch from "@/pages/PatientPrivacyResearch";
import { DevLogin } from "@/components/DevLogin";

function PatientRouter() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/dashboard" component={Dashboard} />
      <Route path="/daily-followup" component={DailyFollowupHistory} />
      <Route path="/daily-followup/video-exam" component={DailyFollowup} />
      <Route path="/agent-hub" component={AgentHub} />
      <Route path="/medications" component={Medications} />
      <Route path="/patient-records" component={PatientRecords} />
      <Route path="/prescriptions" component={PatientRecords} />
      <Route path="/medical-files" component={PatientRecords} />
      <Route path="/deterioration" component={DeteriorationDashboard} />
      <Route path="/ai-alerts" component={AIAlertsDashboard} />
      <Route path="/ml-insights" component={MLInsightsPage} />
      <Route path="/ai-video" component={AIVideoDashboard} />
      <Route path="/ai-audio" component={AIAudioDashboard} />
      <Route path="/guided-exam" component={GuidedVideoExam} />
      <Route path="/guided-video-exam" component={GuidedVideoExam} />
      <Route path="/guided-audio-exam" component={GuidedAudioExam} />
      <Route path="/behavioral-ai-insights" component={BehavioralAIInsights} />
      <Route path="/mental-health" component={MentalHealth} />
      <Route path="/habits" component={Habits} />
      <Route path="/doctor-search" component={DoctorSearch} />
      <Route path="/doctor/:doctorId" component={DoctorProfile} />
      <Route path="/my-doctors" component={MyDoctors} />
      <Route path="/consultation-requests" component={ConsultationRequests} />
      <Route path="/video-consultation/:consultationId" component={VideoConsultation} />
      <Route path="/wellness/:type?" component={Wellness} />
      <Route path="/consents" component={ConsentManagement} />
      <Route path="/research-consent" component={ResearchConsentSettings} />
      <Route path="/privacy-research" component={PatientPrivacyResearch} />
      <Route path="/wearables" component={WearableDevices} />
      <Route path="/device-connect" component={DeviceConnect} />
      <Route path="/risk-exposures" component={RiskExposures} />
      <Route path="/correlation-insights" component={CorrelationInsights} />
      <Route path="/nutrition-insights" component={NutritionInsights} />
      <Route path="/health-companion" component={HealthCompanion} />
      <Route path="/voice-followups" component={VoiceFollowups} />
      <Route path="/demo-flow" component={DemoFlow} />
      <Route path="/referrals" component={Referrals} />
      <Route path="/wallet" component={Wallet} />
      <Route path="/security/2fa" component={TwoFactorAuth} />
      <Route path="/profile" component={Profile} />
      <Route component={NotFound} />
    </Switch>
  );
}

function DoctorRouter() {
  return (
    <Switch>
      <Route path="/" component={ReceptionistDashboard} />
      <Route path="/dashboard" component={ReceptionistDashboard} />
      <Route path="/daily-followup" component={DailyFollowupHistory} />
      <Route path="/ai-alerts" component={AIAlertsDashboard} />
      <Route path="/prescriptions" component={Prescriptions} />
      <Route path="/calendar-sync" component={CalendarSync} />
      <Route path="/video-consultation/:consultationId" component={VideoConsultation} />
      <Route path="/doctor/patient/:id" component={PatientReview} />
      <Route path="/doctor/patient/:patientId/ml-insights" component={MLInsightsPage} />
      <Route path="/doctor/medication-report/:id" component={DoctorMedicationReport} />
      <Route path="/research" component={ResearchCenter} />
      <Route path="/ml-monitoring" component={MLMonitoring} />
      <Route path="/medical-nlp" component={MedicalNLPDashboard} />
      <Route path="/agent-hub" component={AgentHub} />
      <Route path="/referrals" component={Referrals} />
      <Route path="/wallet" component={Wallet} />
      <Route path="/admin/verify-doctors" component={AdminVerification} />
      <Route path="/security/2fa" component={TwoFactorAuth} />
      <Route path="/profile" component={Profile} />
      <Route component={NotFound} />
    </Switch>
  );
}

function AdminRouter() {
  return (
    <Switch>
      <Route path="/" component={AdminMLTrainingHub} />
      <Route path="/ml-training" component={AdminMLTrainingHub} />
      <Route path="/ml-monitoring" component={MLMonitoring} />
      <Route path="/medical-nlp" component={MedicalNLPDashboard} />
      <Route path="/tinker" component={TinkerDashboard} />
      <Route path="/research" component={ResearchCenter} />
      <Route path="/admin/verify-doctors" component={AdminVerification} />
      <Route path="/agent-hub" component={AgentHub} />
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
  
  // Not authenticated - show dev login in development, otherwise redirect to login
  if (!user) {
    // In development, show quick dev login
    if (import.meta.env.DEV) {
      return <DevLogin />;
    }
    // In production, redirect to login page
    window.location.href = "/login";
    return null;
  }

  const isDoctor = user.role === "doctor";
  const isAdmin = user.role === "admin";

  const getRouter = () => {
    if (isAdmin) return <AdminRouter />;
    if (isDoctor) return <DoctorRouter />;
    return <PatientRouter />;
  };

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
            {getRouter()}
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
