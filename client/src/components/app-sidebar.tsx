import {
  Activity,
  Bot,
  Calendar,
  Camera,
  FileText,
  Heart,
  Home,
  Pill,
  Settings,
  Stethoscope,
  User,
  Wind,
  Users,
  Beaker,
  LogOut,
  MessageCircle,
  Link as LinkIcon,
  Watch,
  Wallet as WalletIcon,
  Gift,
  ScanText,
  Shield,
  Target,
  Headphones,
  Search,
  UserPlus,
  Video,
  Mic,
  Bell,
  TrendingUp,
  Brain,
  Microscope,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useLocation } from "wouter";
import { useAuth } from "@/contexts/AuthContext";

const patientItems = [
  {
    title: "Dashboard",
    url: "/",
    icon: Home,
  },
  {
    title: "Daily Follow-up",
    url: "/daily-followup",
    icon: Activity,
    badge: "History",
  },
  {
    title: "Agent Clona",
    url: "/agent-hub",
    icon: Bot,
    badge: "AI",
  },
  {
    title: "Habit Tracker",
    url: "/habits",
    icon: Target,
    badge: "AI",
  },
  {
    title: "Health Alerts",
    url: "/ai-alerts",
    icon: Bell,
    badge: "AI",
  },
  {
    title: "Medications",
    url: "/medications",
    icon: Pill,
    badge: "AI",
  },
  {
    title: "Patient Records",
    url: "/patient-records",
    icon: FileText,
    badge: "Unified",
  },
  {
    title: "My Doctors",
    url: "/my-doctors",
    icon: UserPlus,
  },
  {
    title: "Wellness",
    url: "/wellness",
    icon: Wind,
  },
  {
    title: "App Connections",
    url: "/consents",
    icon: LinkIcon,
  },
  {
    title: "Research Consent",
    url: "/research-consent",
    icon: Microscope,
    badge: "HIPAA",
  },
  {
    title: "Wearable Devices",
    url: "/wearables",
    icon: Watch,
  },
  {
    title: "Risk & Exposures",
    url: "/risk-exposures",
    icon: Shield,
    badge: "Auto",
  },
  {
    title: "Referrals",
    url: "/referrals",
    icon: Gift,
  },
  {
    title: "Wallet",
    url: "/wallet",
    icon: WalletIcon,
  },
  {
    title: "Profile",
    url: "/profile",
    icon: User,
  },
];

const doctorItems = [
  {
    title: "Dashboard",
    url: "/",
    icon: Home,
  },
  {
    title: "Assistant Lysa",
    url: "/agent-hub",
    icon: Headphones,
    badge: "AI",
  },
  {
    title: "Research Center",
    url: "/research",
    icon: Beaker,
  },
  {
    title: "Referrals",
    url: "/referrals",
    icon: Gift,
  },
  {
    title: "Wallet",
    url: "/wallet",
    icon: WalletIcon,
  },
  {
    title: "Profile",
    url: "/profile",
    icon: User,
  },
];

export function AppSidebar() {
  const [location, setLocation] = useLocation();
  const { user, logout } = useAuth();
  const isDoctor = user?.role === "doctor";
  const menuItems = isDoctor ? doctorItems : patientItems;

  const handleLogout = () => {
    logout();
    setLocation("/login");
  };

  return (
    <Sidebar>
      <SidebarHeader className="p-4 border-b">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Stethoscope className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold">Followup AI</h2>
            <p className="text-xs text-muted-foreground">
              {isDoctor ? "Doctor Portal" : "Patient Portal"}
            </p>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>{isDoctor ? "Tools" : "Health"}</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location === item.url}
                    data-testid={`link-${item.title.toLowerCase().replace(/\s+/g, "-")}`}
                  >
                    <a href={item.url} className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <item.icon className="h-4 w-4" />
                        <span>{item.title}</span>
                      </div>
                      {item.badge && (
                        <Badge variant="default" className="text-xs h-5">
                          {item.badge}
                        </Badge>
                      )}
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="p-4 border-t space-y-3">
        <Button 
          variant="outline" 
          className="w-full justify-start gap-2"
          onClick={handleLogout}
          data-testid="button-logout"
        >
          <LogOut className="h-4 w-4" />
          <span>Log Out</span>
        </Button>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <div className="h-2 w-2 rounded-full bg-chart-2" />
          <span>HIPAA Compliant</span>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
