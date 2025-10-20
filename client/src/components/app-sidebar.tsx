import {
  Activity,
  Bot,
  Calendar,
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
import { useLocation } from "wouter";
import { useAuth } from "@/hooks/useAuth";

const patientItems = [
  {
    title: "Dashboard",
    url: "/",
    icon: Home,
  },
  {
    title: "Chat with Agent Clona",
    url: "/chat",
    icon: Bot,
    badge: "AI",
  },
  {
    title: "Wellness",
    url: "/wellness",
    icon: Wind,
  },
  {
    title: "Medical Files",
    url: "/files",
    icon: FileText,
  },
  {
    title: "Profile",
    url: "/profile",
    icon: User,
  },
];

const doctorItems = [
  {
    title: "All Patients",
    url: "/",
    icon: Users,
  },
  {
    title: "Research Center",
    url: "/research",
    icon: Beaker,
  },
  {
    title: "Chat with Assistant Lysa",
    url: "/chat",
    icon: Bot,
    badge: "AI",
  },
  {
    title: "Profile",
    url: "/profile",
    icon: User,
  },
];

export function AppSidebar() {
  const [location] = useLocation();
  const { user } = useAuth();
  const isDoctor = user?.role === "doctor";
  const menuItems = isDoctor ? doctorItems : patientItems;

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
      <SidebarFooter className="p-4 border-t">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <div className="h-2 w-2 rounded-full bg-chart-2" />
          <span>HIPAA Compliant</span>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
