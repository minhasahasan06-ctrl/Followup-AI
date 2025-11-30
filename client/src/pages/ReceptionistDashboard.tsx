import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Calendar, Clock, Mail, Phone, Plus, User, Video, Bot, MessageSquare, Stethoscope, Pill, FileText, ChevronRight, Sparkles, LayoutDashboard, Users, Activity, Link2Off, ExternalLink, Loader2, Send, RefreshCw, Check, X, Search, Reply, Forward, Trash2 } from "lucide-react";
import { SiGoogle, SiWhatsapp } from "react-icons/si";
import { format, startOfWeek, addDays, isSameDay, parseISO, differenceInMinutes, isToday, isTomorrow, isWithinInterval, addWeeks } from "date-fns";
import { LysaChatPanel } from "@/components/LysaChatPanel";
import { PatientManagementPanel } from "@/components/PatientManagementPanel";
import { DiagnosisHelper } from "@/components/DiagnosisHelper";
import { PrescriptionHelper } from "@/components/PrescriptionHelper";
import { AutomationStatusPanel, AutomationStatusBadge } from "@/components/AutomationStatusPanel";
import { BookAppointmentDialog } from "@/components/LysaCalendarBooking";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface Appointment {
  id: string;
  patientId: string;
  doctorId: string;
  startTime: string;
  endTime: string;
  appointmentType: string;
  status: string;
  confirmationStatus: string;
  notes?: string;
  patientName?: string;
  patientEmail?: string;
  patientPhone?: string;
  duration?: number;
  bookedByMethod?: string;
  description?: string;
}

interface CalendarEvent {
  id: string;
  summary: string;
  description?: string;
  location?: string;
  start: { dateTime?: string; date?: string };
  end: { dateTime?: string; date?: string };
  status: string;
}

interface ConnectorStatus {
  connected: boolean;
  calendarId?: string;
  calendarName?: string;
  email?: string;
  source: string;
}

interface EmailThread {
  id: string;
  doctorId: string;
  subject: string;
  category: string;
  priority: string;
  status: string;
  isRead: boolean;
  messageCount: number;
  lastMessageAt: string;
}

interface IntegrationStatus {
  gmail: { connected: boolean; email?: string; lastSync?: string };
  whatsapp: { connected: boolean; number?: string; lastSync?: string };
  twilio: { connected: boolean; number?: string; lastSync?: string };
}

interface SyncedEmail {
  id: string;
  doctorId: string;
  gmailMessageId: string;
  from: string;
  to: string;
  subject: string;
  body?: string;
  snippet?: string;
  category?: string;
  priority?: string;
  sentiment?: string;
  isRead: boolean;
  receivedAt: string;
}

interface WhatsAppMessage {
  id: string;
  fromNumber: string;
  toNumber: string;
  message: string;
  direction: string;
  status: string;
  createdAt: string;
  patientName?: string;
}

interface LysaPatient {
  id: string;
  firstName?: string | null;
  lastName?: string | null;
}

export default function ReceptionistDashboard() {
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [lysaExpanded, setLysaExpanded] = useState(true);
  const [lysaPatient, setLysaPatient] = useState<LysaPatient | null>(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [appointmentDialog, setAppointmentDialog] = useState(false);
  const [selectedAppointment, setSelectedAppointment] = useState<Appointment | null>(null);
  const [selectedEmail, setSelectedEmail] = useState<SyncedEmail | null>(null);
  const [emailReplyMode, setEmailReplyMode] = useState(false);
  const [emailReplyText, setEmailReplyText] = useState("");
  const [whatsappReplyNumber, setWhatsappReplyNumber] = useState<string | null>(null);
  const [whatsappReplyText, setWhatsappReplyText] = useState("");
  const { toast } = useToast();
  const weekStart = startOfWeek(selectedDate);
  const weekDays = Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));

  const { data: upcomingAppointments = [], isLoading: appointmentsLoading } = useQuery<Appointment[]>({
    queryKey: ["/api/v1/appointments/upcoming", { days: 7 }],
  });

  const { data: emailThreads = [], isLoading: emailsLoading } = useQuery<EmailThread[]>({
    queryKey: ["/api/v1/emails/threads", { limit: 10 }],
  });

  const { data: integrationStatus, isLoading: integrationLoading } = useQuery<IntegrationStatus>({
    queryKey: ["/api/v1/integrations/status"],
  });

  const { data: calendarConnector, isLoading: connectorLoading } = useQuery<ConnectorStatus>({
    queryKey: ["/api/v1/calendar/connector-status"],
  });

  const { data: calendarEventsData, isLoading: eventsLoading } = useQuery<{ events: CalendarEvent[] }>({
    queryKey: ["/api/v1/calendar/events"],
    enabled: calendarConnector?.connected === true,
  });

  const calendarEvents = calendarEventsData?.events || [];

  const { data: syncedEmails = [], refetch: refetchEmails } = useQuery<SyncedEmail[]>({
    queryKey: ["/api/v1/integrations/gmail/emails"],
    enabled: integrationStatus?.gmail.connected === true,
  });

  const { data: whatsappMessages = [], refetch: refetchWhatsapp } = useQuery<WhatsAppMessage[]>({
    queryKey: ["/api/v1/integrations/whatsapp/messages"],
    enabled: integrationStatus?.whatsapp.connected === true,
  });

  const getCalendarAuthUrl = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/calendar/auth-url");
      return response.json();
    },
    onSuccess: (data: { authUrl: string }) => {
      window.location.href = data.authUrl;
    },
    onError: () => {
      toast({
        title: "Connection Error",
        description: "Failed to get Google Calendar authorization URL.",
        variant: "destructive",
      });
    },
  });

  const getGmailAuthUrl = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/gmail/auth-url");
      return response.json();
    },
    onSuccess: (data: { authUrl: string }) => {
      window.location.href = data.authUrl;
    },
    onError: () => {
      toast({
        title: "Connection Error",
        description: "Failed to get Gmail authorization URL.",
        variant: "destructive",
      });
    },
  });

  const syncGmail = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/gmail/sync", { method: "POST" });
      return response.json();
    },
    onSuccess: () => {
      refetchEmails();
      toast({
        title: "Sync Complete",
        description: "Gmail inbox has been synced successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Sync Failed",
        description: "Failed to sync Gmail inbox.",
        variant: "destructive",
      });
    },
  });

  const disconnectGmail = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/gmail/disconnect", { method: "POST" });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/status"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/gmail/emails"] });
      toast({
        title: "Disconnected",
        description: "Gmail has been disconnected.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to disconnect Gmail.",
        variant: "destructive",
      });
    },
  });

  const sendEmailReply = useMutation({
    mutationFn: async ({ to, subject, body }: { to: string; subject: string; body: string }) => {
      const response = await apiRequest("/api/v1/integrations/gmail/send", { 
        method: "POST",
        body: JSON.stringify({ to, subject, body })
      });
      return response.json();
    },
    onSuccess: () => {
      setEmailReplyMode(false);
      setEmailReplyText("");
      toast({
        title: "Email Sent",
        description: "Your reply has been sent successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Send Failed",
        description: "Failed to send email reply.",
        variant: "destructive",
      });
    },
  });

  const sendWhatsAppMessage = useMutation({
    mutationFn: async ({ toNumber, message }: { toNumber: string; message: string }) => {
      const response = await apiRequest("/api/v1/integrations/whatsapp/send", { 
        method: "POST",
        body: JSON.stringify({ toNumber, message })
      });
      return response.json();
    },
    onSuccess: () => {
      setWhatsappReplyNumber(null);
      setWhatsappReplyText("");
      refetchWhatsapp();
      toast({
        title: "Message Sent",
        description: "WhatsApp message has been sent successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Send Failed",
        description: "Failed to send WhatsApp message.",
        variant: "destructive",
      });
    },
  });

  const todayAppointments = upcomingAppointments.filter(apt => 
    isSameDay(parseISO(apt.startTime), new Date())
  );

  const nextSevenDaysAppointments = upcomingAppointments.filter(apt => 
    !isSameDay(parseISO(apt.startTime), new Date()) &&
    isWithinInterval(parseISO(apt.startTime), {
      start: new Date(),
      end: addDays(new Date(), 7)
    })
  );

  const getAppointmentsForDay = (date: Date) => {
    return upcomingAppointments.filter(apt => 
      isSameDay(parseISO(apt.startTime), date)
    );
  };

  const getCalendarEventsForDay = (date: Date) => {
    return calendarEvents.filter(event => {
      const eventDate = event.start.dateTime ? parseISO(event.start.dateTime) : 
                       event.start.date ? parseISO(event.start.date) : null;
      return eventDate && isSameDay(eventDate, date);
    });
  };

  const getAllEventsForDay = (date: Date) => {
    const appointments = getAppointmentsForDay(date);
    const events = getCalendarEventsForDay(date);
    return {
      appointments,
      events,
      total: appointments.length + events.length
    };
  };

  const getBookingMethodInfo = (method?: string) => {
    const methods: Record<string, { icon: JSX.Element; label: string; color: string }> = {
      whatsapp: { icon: <SiWhatsapp className="h-3 w-3" />, label: "WhatsApp", color: "text-green-600" },
      email: { icon: <Mail className="h-3 w-3" />, label: "Email", color: "text-blue-600" },
      phone: { icon: <Phone className="h-3 w-3" />, label: "Phone", color: "text-orange-600" },
      online: { icon: <ExternalLink className="h-3 w-3" />, label: "Online", color: "text-purple-600" },
      "walk-in": { icon: <User className="h-3 w-3" />, label: "Walk-in", color: "text-gray-600" },
    };
    return methods[method || "online"] || methods.online;
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      urgent: 'destructive',
      appointment: 'default',
      prescription: 'secondary',
      general: 'outline'
    };
    return colors[category] || 'outline';
  };

  const getDateLabel = (dateStr: string) => {
    const date = parseISO(dateStr);
    if (isToday(date)) return "Today";
    if (isTomorrow(date)) return "Tomorrow";
    return format(date, "EEE, MMM d");
  };

  const groupedWhatsappMessages = whatsappMessages.reduce((acc, msg) => {
    const phone = msg.direction === 'inbound' ? msg.fromNumber : msg.toNumber;
    if (!acc[phone]) {
      acc[phone] = {
        phone,
        patientName: msg.patientName,
        messages: [],
        lastMessage: msg.createdAt,
        unreadCount: 0
      };
    }
    acc[phone].messages.push(msg);
    if (msg.direction === 'inbound' && msg.status === 'unread') {
      acc[phone].unreadCount++;
    }
    if (new Date(msg.createdAt) > new Date(acc[phone].lastMessage)) {
      acc[phone].lastMessage = msg.createdAt;
    }
    return acc;
  }, {} as Record<string, { phone: string; patientName?: string; messages: WhatsAppMessage[]; lastMessage: string; unreadCount: number }>);

  const conversationList = Object.values(groupedWhatsappMessages).sort(
    (a, b) => new Date(b.lastMessage).getTime() - new Date(a.lastMessage).getTime()
  );

  return (
    <div className="flex gap-6 h-full">
      <div className="flex-1 space-y-6 overflow-auto">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold mb-1 flex items-center gap-3" data-testid="text-receptionist-title">
              <div className="h-10 w-10 rounded-lg bg-accent flex items-center justify-center">
                <Bot className="h-5 w-5" />
              </div>
              Assistant Lysa
            </h1>
            <p className="text-muted-foreground">
              Your AI receptionist and doctor's assistant
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={() => setLysaExpanded(!lysaExpanded)} data-testid="button-toggle-lysa-panel">
              <MessageSquare className="h-4 w-4 mr-2" />
              {lysaExpanded ? "Hide Chat" : "Show Chat"}
            </Button>
            <Button onClick={() => setAppointmentDialog(true)} data-testid="button-book-appointment">
              <Plus className="h-4 w-4 mr-2" />
              Book Appointment
            </Button>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="mb-4 flex-wrap">
            <TabsTrigger value="overview" data-testid="tab-overview">
              <LayoutDashboard className="h-4 w-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="automation" data-testid="tab-automation">
              <Activity className="h-4 w-4 mr-2" />
              Automation
            </TabsTrigger>
            <TabsTrigger value="patients" data-testid="tab-patients">
              <Users className="h-4 w-4 mr-2" />
              Patients
            </TabsTrigger>
            <TabsTrigger value="diagnosis" data-testid="tab-diagnosis">
              <Stethoscope className="h-4 w-4 mr-2" />
              Diagnosis
            </TabsTrigger>
            <TabsTrigger value="prescription" data-testid="tab-prescription">
              <Pill className="h-4 w-4 mr-2" />
              Rx Builder
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6 mt-0">
            <div className="grid gap-6 lg:grid-cols-5">
              {/* Unified Calendar & Appointments Card - Takes 3 columns */}
              <Card className="lg:col-span-3" data-testid="card-unified-calendar">
                <CardHeader className="flex flex-row items-center justify-between gap-1 pb-2">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Calendar className="h-5 w-5" />
                      Calendar & Appointments
                      {calendarConnector?.connected && (
                        <Badge variant="outline" className="text-xs ml-2">
                          <SiGoogle className="h-3 w-3 mr-1" />
                          Synced
                        </Badge>
                      )}
                    </CardTitle>
                    <CardDescription>
                      {format(weekStart, "MMM d")} - {format(addDays(weekStart, 6), "MMM d, yyyy")}
                    </CardDescription>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {connectorLoading || appointmentsLoading || eventsLoading ? (
                    <div className="animate-pulse space-y-3">
                      {[1, 2, 3].map((i) => (
                        <div key={i} className="h-12 bg-muted rounded" />
                      ))}
                    </div>
                  ) : !calendarConnector?.connected ? (
                    <div className="text-center py-8">
                      <Link2Off className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                      <p className="text-muted-foreground mb-4">
                        Connect Google Calendar to sync your schedule and view appointments
                      </p>
                      <Button 
                        onClick={() => getCalendarAuthUrl.mutate()}
                        disabled={getCalendarAuthUrl.isPending}
                        data-testid="button-connect-calendar"
                      >
                        {getCalendarAuthUrl.isPending ? (
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        ) : (
                          <SiGoogle className="h-4 w-4 mr-2" />
                        )}
                        Connect Google Calendar
                      </Button>
                    </div>
                  ) : (
                    <>
                      {/* Weekly Calendar View */}
                      <div className="grid grid-cols-7 gap-2">
                        {weekDays.map((day) => {
                          const dayData = getAllEventsForDay(day);
                          const isCurrentDay = isSameDay(day, new Date());
                          
                          return (
                            <div
                              key={day.toISOString()}
                              className={`p-2 rounded-md border text-center ${
                                isCurrentDay ? "border-primary bg-primary/5" : "border-border"
                              }`}
                              data-testid={`calendar-day-${format(day, "yyyy-MM-dd")}`}
                            >
                              <div className="text-xs font-medium text-muted-foreground">
                                {format(day, "EEE")}
                              </div>
                              <div className={`text-lg font-bold ${isCurrentDay ? "text-primary" : ""}`}>
                                {format(day, "d")}
                              </div>
                              {dayData.total > 0 && (
                                <Badge variant={isCurrentDay ? "default" : "secondary"} className="text-xs mt-1">
                                  {dayData.total}
                                </Badge>
                              )}
                            </div>
                          );
                        })}
                      </div>

                      <Separator />

                      {/* Today's Appointments */}
                      <div>
                        <h4 className="font-medium text-sm mb-3 flex items-center gap-2">
                          <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                          Today · {todayAppointments.length} appointment{todayAppointments.length !== 1 ? 's' : ''}
                        </h4>
                        {todayAppointments.length === 0 ? (
                          <div className="text-center py-4 text-muted-foreground text-sm">
                            No appointments scheduled for today
                          </div>
                        ) : (
                          <div className="space-y-2">
                            {todayAppointments.slice(0, 4).map((appointment) => {
                              const methodInfo = getBookingMethodInfo(appointment.bookedByMethod);
                              return (
                                <div
                                  key={appointment.id}
                                  className="flex items-center justify-between p-2 rounded-md border hover-elevate active-elevate-2 cursor-pointer"
                                  onClick={() => setSelectedAppointment(appointment)}
                                  data-testid={`today-appointment-${appointment.id}`}
                                >
                                  <div className="flex items-center gap-2">
                                    <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                                      <User className="h-4 w-4 text-primary" />
                                    </div>
                                    <div>
                                      <p className="font-medium text-sm">
                                        {appointment.patientName || `Patient`}
                                      </p>
                                      <p className="text-xs text-muted-foreground flex items-center gap-1">
                                        <Clock className="h-3 w-3" />
                                        {format(parseISO(appointment.startTime), "h:mm a")}
                                      </p>
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <Badge variant={appointment.confirmationStatus === 'confirmed' ? 'default' : 'secondary'} className="text-xs">
                                      {appointment.confirmationStatus}
                                    </Badge>
                                    <span className={`text-xs flex items-center gap-1 ${methodInfo.color}`}>
                                      {methodInfo.icon}
                                    </span>
                                  </div>
                                </div>
                              );
                            })}
                            {todayAppointments.length > 4 && (
                              <p className="text-xs text-center text-muted-foreground">
                                +{todayAppointments.length - 4} more appointments today
                              </p>
                            )}
                          </div>
                        )}
                      </div>

                      <Separator />

                      {/* Next 7 Days */}
                      <div>
                        <h4 className="font-medium text-sm mb-3">
                          Next 7 Days · {nextSevenDaysAppointments.length} appointment{nextSevenDaysAppointments.length !== 1 ? 's' : ''}
                        </h4>
                        {nextSevenDaysAppointments.length === 0 ? (
                          <div className="text-center py-4 text-muted-foreground text-sm">
                            No appointments scheduled for the next 7 days
                          </div>
                        ) : (
                          <ScrollArea className="h-[180px]">
                            <div className="space-y-2 pr-3">
                              {nextSevenDaysAppointments.map((appointment) => {
                                const methodInfo = getBookingMethodInfo(appointment.bookedByMethod);
                                return (
                                  <div
                                    key={appointment.id}
                                    className="flex items-center justify-between p-2 rounded-md border hover-elevate active-elevate-2 cursor-pointer"
                                    onClick={() => setSelectedAppointment(appointment)}
                                    data-testid={`upcoming-appointment-${appointment.id}`}
                                  >
                                    <div className="flex items-center gap-2">
                                      <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center">
                                        <User className="h-4 w-4 text-muted-foreground" />
                                      </div>
                                      <div>
                                        <p className="font-medium text-sm">
                                          {appointment.patientName || `Patient`}
                                        </p>
                                        <p className="text-xs text-muted-foreground">
                                          {getDateLabel(appointment.startTime)} · {format(parseISO(appointment.startTime), "h:mm a")}
                                        </p>
                                      </div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                      <Badge variant="outline" className="text-xs">
                                        {appointment.appointmentType}
                                      </Badge>
                                      <span className={`text-xs flex items-center gap-1 ${methodInfo.color}`}>
                                        {methodInfo.icon}
                                      </span>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </ScrollArea>
                        )}
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>

              {/* Right Column - Email & WhatsApp Widgets - Takes 2 columns */}
              <div className="lg:col-span-2 space-y-6">
                {/* Email Widget */}
                <Card data-testid="card-email-widget">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center justify-between">
                      <span className="flex items-center gap-2">
                        <Mail className="h-5 w-5" />
                        Email
                      </span>
                      {integrationStatus?.gmail.connected && (
                        <div className="flex items-center gap-2">
                          <Button 
                            variant="ghost" 
                            size="icon"
                            onClick={() => syncGmail.mutate()}
                            disabled={syncGmail.isPending}
                            data-testid="button-sync-gmail"
                          >
                            <RefreshCw className={`h-4 w-4 ${syncGmail.isPending ? 'animate-spin' : ''}`} />
                          </Button>
                        </div>
                      )}
                    </CardTitle>
                    {integrationStatus?.gmail.connected && integrationStatus.gmail.email && (
                      <CardDescription className="flex items-center gap-1">
                        <SiGoogle className="h-3 w-3" />
                        {integrationStatus.gmail.email}
                      </CardDescription>
                    )}
                  </CardHeader>
                  <CardContent>
                    {integrationLoading ? (
                      <div className="animate-pulse space-y-3">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="h-12 bg-muted rounded" />
                        ))}
                      </div>
                    ) : !integrationStatus?.gmail.connected ? (
                      <div className="text-center py-6">
                        <Mail className="h-10 w-10 mx-auto mb-3 text-muted-foreground opacity-50" />
                        <p className="text-sm text-muted-foreground mb-4">
                          Connect your Gmail to manage emails
                        </p>
                        <Button 
                          onClick={() => getGmailAuthUrl.mutate()}
                          disabled={getGmailAuthUrl.isPending}
                          data-testid="button-connect-gmail"
                        >
                          {getGmailAuthUrl.isPending ? (
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          ) : (
                            <SiGoogle className="h-4 w-4 mr-2" />
                          )}
                          Connect Gmail
                        </Button>
                      </div>
                    ) : syncedEmails.length === 0 ? (
                      <div className="text-center py-6 text-muted-foreground">
                        <Mail className="h-10 w-10 mx-auto mb-3 opacity-50" />
                        <p className="text-sm">No emails in inbox</p>
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="mt-2"
                          onClick={() => syncGmail.mutate()}
                          disabled={syncGmail.isPending}
                          data-testid="button-initial-sync-gmail"
                        >
                          {syncGmail.isPending ? (
                            <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                          ) : (
                            <RefreshCw className="h-3 w-3 mr-2" />
                          )}
                          Sync Inbox
                        </Button>
                      </div>
                    ) : (
                      <ScrollArea className="h-[280px]">
                        <div className="space-y-2 pr-3">
                          {syncedEmails.slice(0, 10).map((email) => (
                            <div
                              key={email.id}
                              className={`p-3 rounded-md border hover-elevate active-elevate-2 cursor-pointer ${
                                !email.isRead ? "bg-primary/5 border-primary/20" : ""
                              }`}
                              onClick={() => setSelectedEmail(email)}
                              data-testid={`email-item-${email.id}`}
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex-1 min-w-0">
                                  <p className={`font-medium text-sm truncate ${!email.isRead ? "text-primary" : ""}`}>
                                    {email.subject || "(No Subject)"}
                                  </p>
                                  <p className="text-xs text-muted-foreground truncate">
                                    {email.from}
                                  </p>
                                  {email.snippet && (
                                    <p className="text-xs text-muted-foreground mt-1 line-clamp-1">
                                      {email.snippet}
                                    </p>
                                  )}
                                </div>
                                <div className="flex flex-col items-end gap-1">
                                  <span className="text-xs text-muted-foreground whitespace-nowrap">
                                    {format(parseISO(email.receivedAt), "MMM d")}
                                  </span>
                                  {email.priority === 'urgent' && (
                                    <Badge variant="destructive" className="text-xs">
                                      Urgent
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    )}
                    {integrationStatus?.gmail.connected && (
                      <div className="mt-3 pt-3 border-t">
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="w-full text-xs text-muted-foreground"
                          onClick={() => disconnectGmail.mutate()}
                          disabled={disconnectGmail.isPending}
                          data-testid="button-disconnect-gmail"
                        >
                          Disconnect Gmail
                        </Button>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* WhatsApp Widget */}
                <Card data-testid="card-whatsapp-widget">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center justify-between">
                      <span className="flex items-center gap-2">
                        <SiWhatsapp className="h-5 w-5 text-green-600" />
                        WhatsApp
                      </span>
                      {integrationStatus?.whatsapp.connected && (
                        <Badge variant="outline" className="text-xs">
                          <div className="h-1.5 w-1.5 rounded-full bg-green-500 mr-1.5" />
                          Connected
                        </Badge>
                      )}
                    </CardTitle>
                    {integrationStatus?.whatsapp.connected && integrationStatus.whatsapp.number && (
                      <CardDescription className="flex items-center gap-1">
                        <Phone className="h-3 w-3" />
                        {integrationStatus.whatsapp.number}
                      </CardDescription>
                    )}
                  </CardHeader>
                  <CardContent>
                    {integrationLoading ? (
                      <div className="animate-pulse space-y-3">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="h-12 bg-muted rounded" />
                        ))}
                      </div>
                    ) : !integrationStatus?.whatsapp.connected ? (
                      <div className="text-center py-6">
                        <SiWhatsapp className="h-10 w-10 mx-auto mb-3 text-green-600 opacity-50" />
                        <p className="text-sm text-muted-foreground mb-2">
                          Connect WhatsApp Business to manage patient messages
                        </p>
                        <p className="text-xs text-muted-foreground mb-4">
                          Configure your WhatsApp Business API credentials in Automation settings
                        </p>
                        <Button 
                          variant="outline"
                          onClick={() => setActiveTab("automation")}
                          data-testid="button-configure-whatsapp"
                        >
                          <Sparkles className="h-4 w-4 mr-2" />
                          Configure WhatsApp
                        </Button>
                      </div>
                    ) : conversationList.length === 0 ? (
                      <div className="text-center py-6 text-muted-foreground">
                        <MessageSquare className="h-10 w-10 mx-auto mb-3 opacity-50" />
                        <p className="text-sm">No WhatsApp conversations</p>
                        <p className="text-xs mt-1">Messages will appear here when patients contact you</p>
                      </div>
                    ) : (
                      <ScrollArea className="h-[220px]">
                        <div className="space-y-2 pr-3">
                          {conversationList.slice(0, 8).map((conversation) => (
                            <div
                              key={conversation.phone}
                              className="p-3 rounded-md border hover-elevate active-elevate-2 cursor-pointer"
                              onClick={() => setWhatsappReplyNumber(conversation.phone)}
                              data-testid={`whatsapp-conversation-${conversation.phone}`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="flex items-center gap-2 flex-1 min-w-0">
                                  <div className="h-8 w-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0">
                                    <User className="h-4 w-4 text-green-600" />
                                  </div>
                                  <div className="min-w-0">
                                    <p className="font-medium text-sm truncate">
                                      {conversation.patientName || conversation.phone}
                                    </p>
                                    <p className="text-xs text-muted-foreground truncate">
                                      {conversation.messages[0]?.message?.slice(0, 40) || "No message"}...
                                    </p>
                                  </div>
                                </div>
                                <div className="flex flex-col items-end gap-1">
                                  <span className="text-xs text-muted-foreground">
                                    {format(parseISO(conversation.lastMessage), "MMM d")}
                                  </span>
                                  {conversation.unreadCount > 0 && (
                                    <Badge className="bg-green-600 text-xs">
                                      {conversation.unreadCount}
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="automation" className="mt-0">
            <div className="grid gap-6 lg:grid-cols-2">
              <AutomationStatusPanel />
              <Card data-testid="card-automation-config">
                <CardHeader>
                  <CardTitle>Automation Settings</CardTitle>
                  <CardDescription>Configure Lysa's automated tasks</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 rounded-lg border hover-elevate">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-blue-500/10 flex items-center justify-center">
                          <Mail className="h-5 w-5 text-blue-600" />
                        </div>
                        <div>
                          <p className="font-medium">Email Automation</p>
                          <p className="text-sm text-muted-foreground">Auto-classify, reply & forward</p>
                        </div>
                      </div>
                      <Badge variant="outline" data-testid="badge-email-automation">Configure</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg border hover-elevate">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-green-500/10 flex items-center justify-center">
                          <MessageSquare className="h-5 w-5 text-green-600" />
                        </div>
                        <div>
                          <p className="font-medium">WhatsApp Automation</p>
                          <p className="text-sm text-muted-foreground">Auto-reply & templates</p>
                        </div>
                      </div>
                      <Badge variant="outline" data-testid="badge-whatsapp-automation">Configure</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg border hover-elevate">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-purple-500/10 flex items-center justify-center">
                          <Calendar className="h-5 w-5 text-purple-600" />
                        </div>
                        <div>
                          <p className="font-medium">Appointment Automation</p>
                          <p className="text-sm text-muted-foreground">Auto-book & confirm</p>
                        </div>
                      </div>
                      <Badge variant="outline" data-testid="badge-appointment-automation">Configure</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg border hover-elevate">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-amber-500/10 flex items-center justify-center">
                          <Clock className="h-5 w-5 text-amber-600" />
                        </div>
                        <div>
                          <p className="font-medium">Reminder Automation</p>
                          <p className="text-sm text-muted-foreground">Medication, appointments, follow-ups</p>
                        </div>
                      </div>
                      <Badge variant="outline" data-testid="badge-reminder-automation">Configure</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg border hover-elevate">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-pink-500/10 flex items-center justify-center">
                          <Stethoscope className="h-5 w-5 text-pink-600" />
                        </div>
                        <div>
                          <p className="font-medium">Clinical Automation</p>
                          <p className="text-sm text-muted-foreground">SOAP notes, ICD-10 suggestions</p>
                        </div>
                      </div>
                      <Badge variant="outline" data-testid="badge-clinical-automation">Configure</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="patients" className="mt-0">
            <PatientManagementPanel 
              onOpenLysa={(patient) => {
                setLysaPatient(patient);
                setLysaExpanded(true);
              }} 
            />
          </TabsContent>

          <TabsContent value="diagnosis" className="mt-0">
            <DiagnosisHelper />
          </TabsContent>

          <TabsContent value="prescription" className="mt-0">
            <PrescriptionHelper />
          </TabsContent>
        </Tabs>
      </div>

      {lysaExpanded && (
        <div className="w-96 flex-shrink-0">
          <LysaChatPanel 
            isExpanded={lysaExpanded}
            onMinimize={() => {
              setLysaExpanded(false);
              setLysaPatient(null);
            }}
            patientId={lysaPatient?.id}
            patientName={lysaPatient ? `${lysaPatient.firstName || ''} ${lysaPatient.lastName || ''}`.trim() : undefined}
            className="h-full sticky top-0"
          />
        </div>
      )}
      
      <BookAppointmentDialog open={appointmentDialog} onOpenChange={setAppointmentDialog} />

      {/* Appointment Details Dialog */}
      <Dialog open={!!selectedAppointment} onOpenChange={(open) => !open && setSelectedAppointment(null)}>
        <DialogContent className="max-w-lg" data-testid="dialog-appointment-details">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5" />
              Appointment Details
            </DialogTitle>
          </DialogHeader>
          {selectedAppointment && (
            <div className="space-y-4">
              <div className="flex items-center gap-4 p-4 bg-muted/50 rounded-lg">
                <div className="h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center">
                  <User className="h-7 w-7 text-primary" />
                </div>
                <div>
                  <p className="font-semibold text-lg">
                    {selectedAppointment.patientName || `Patient ${selectedAppointment.patientId?.slice(0, 8) || 'Unknown'}`}
                  </p>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    {selectedAppointment.patientEmail && (
                      <span className="flex items-center gap-1">
                        <Mail className="h-3 w-3" />
                        {selectedAppointment.patientEmail}
                      </span>
                    )}
                  </div>
                  {selectedAppointment.patientPhone && (
                    <p className="text-sm text-muted-foreground flex items-center gap-1 mt-0.5">
                      <Phone className="h-3 w-3" />
                      {selectedAppointment.patientPhone}
                    </p>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 border rounded-lg">
                  <p className="text-xs text-muted-foreground mb-1">Date & Time</p>
                  <p className="font-medium flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    {format(parseISO(selectedAppointment.startTime), "MMM d, yyyy")}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {format(parseISO(selectedAppointment.startTime), "h:mm a")} - {format(parseISO(selectedAppointment.endTime), "h:mm a")}
                  </p>
                </div>
                <div className="p-3 border rounded-lg">
                  <p className="text-xs text-muted-foreground mb-1">Duration</p>
                  <p className="font-medium">
                    {selectedAppointment.duration || 
                      differenceInMinutes(parseISO(selectedAppointment.endTime), parseISO(selectedAppointment.startTime))} minutes
                  </p>
                  <p className="text-sm text-muted-foreground capitalize">
                    {selectedAppointment.appointmentType}
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Status</p>
                  <div className="flex items-center gap-2">
                    <Badge variant={selectedAppointment.confirmationStatus === 'confirmed' ? 'default' : 'secondary'}>
                      {selectedAppointment.confirmationStatus}
                    </Badge>
                    <Badge variant="outline">{selectedAppointment.status}</Badge>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs text-muted-foreground mb-1">Booked via</p>
                  {(() => {
                    const methodInfo = getBookingMethodInfo(selectedAppointment.bookedByMethod);
                    return (
                      <span className={`text-sm flex items-center gap-1 justify-end ${methodInfo.color}`}>
                        {methodInfo.icon}
                        {methodInfo.label}
                      </span>
                    );
                  })()}
                </div>
              </div>

              {(selectedAppointment.notes || selectedAppointment.description) && (
                <div className="p-3 border rounded-lg">
                  <p className="text-xs text-muted-foreground mb-1">Notes</p>
                  <p className="text-sm">{selectedAppointment.notes || selectedAppointment.description}</p>
                </div>
              )}
            </div>
          )}
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => setSelectedAppointment(null)} data-testid="button-close-appointment-details">
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Email Details Dialog */}
      <Dialog open={!!selectedEmail} onOpenChange={(open) => !open && setSelectedEmail(null)}>
        <DialogContent className="max-w-2xl max-h-[80vh]" data-testid="dialog-email-details">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5" />
              Email Details
            </DialogTitle>
          </DialogHeader>
          {selectedEmail && (
            <div className="space-y-4">
              <div className="p-4 bg-muted/50 rounded-lg">
                <p className="font-semibold text-lg mb-2">{selectedEmail.subject || "(No Subject)"}</p>
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span>From: {selectedEmail.from}</span>
                  <span>To: {selectedEmail.to}</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {format(parseISO(selectedEmail.receivedAt), "MMMM d, yyyy 'at' h:mm a")}
                </p>
              </div>

              <ScrollArea className="h-[200px] border rounded-lg p-4">
                <div className="text-sm whitespace-pre-wrap">
                  {selectedEmail.body || selectedEmail.snippet || "No content available"}
                </div>
              </ScrollArea>

              {emailReplyMode ? (
                <div className="space-y-3">
                  <Textarea
                    placeholder="Type your reply..."
                    value={emailReplyText}
                    onChange={(e) => setEmailReplyText(e.target.value)}
                    className="min-h-[100px]"
                    data-testid="input-email-reply"
                  />
                  <div className="flex justify-end gap-2">
                    <Button 
                      variant="outline" 
                      onClick={() => { setEmailReplyMode(false); setEmailReplyText(""); }}
                      data-testid="button-cancel-reply"
                    >
                      Cancel
                    </Button>
                    <Button 
                      onClick={() => sendEmailReply.mutate({
                        to: selectedEmail.from,
                        subject: `Re: ${selectedEmail.subject}`,
                        body: emailReplyText
                      })}
                      disabled={!emailReplyText.trim() || sendEmailReply.isPending}
                      data-testid="button-send-reply"
                    >
                      {sendEmailReply.isPending ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Send className="h-4 w-4 mr-2" />
                      )}
                      Send Reply
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Button 
                    variant="outline" 
                    onClick={() => setEmailReplyMode(true)}
                    data-testid="button-reply-email"
                  >
                    <Reply className="h-4 w-4 mr-2" />
                    Reply
                  </Button>
                </div>
              )}
            </div>
          )}
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => { setSelectedEmail(null); setEmailReplyMode(false); setEmailReplyText(""); }} data-testid="button-close-email-details">
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* WhatsApp Conversation Dialog */}
      <Dialog open={!!whatsappReplyNumber} onOpenChange={(open) => !open && setWhatsappReplyNumber(null)}>
        <DialogContent className="max-w-lg max-h-[80vh]" data-testid="dialog-whatsapp-conversation">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <SiWhatsapp className="h-5 w-5 text-green-600" />
              WhatsApp Conversation
            </DialogTitle>
            {whatsappReplyNumber && (
              <DialogDescription>
                {groupedWhatsappMessages[whatsappReplyNumber]?.patientName || whatsappReplyNumber}
              </DialogDescription>
            )}
          </DialogHeader>
          {whatsappReplyNumber && groupedWhatsappMessages[whatsappReplyNumber] && (
            <div className="space-y-4">
              <ScrollArea className="h-[300px] border rounded-lg p-4">
                <div className="space-y-3">
                  {groupedWhatsappMessages[whatsappReplyNumber].messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`p-3 rounded-lg max-w-[80%] ${
                        msg.direction === 'outbound' 
                          ? 'ml-auto bg-green-100 dark:bg-green-900/30' 
                          : 'bg-muted'
                      }`}
                      data-testid={`whatsapp-message-${msg.id}`}
                    >
                      <p className="text-sm">{msg.message}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {format(parseISO(msg.createdAt), "MMM d, h:mm a")}
                      </p>
                    </div>
                  ))}
                </div>
              </ScrollArea>

              <div className="flex gap-2">
                <Input
                  placeholder="Type your message..."
                  value={whatsappReplyText}
                  onChange={(e) => setWhatsappReplyText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && whatsappReplyText.trim()) {
                      sendWhatsAppMessage.mutate({
                        toNumber: whatsappReplyNumber,
                        message: whatsappReplyText
                      });
                    }
                  }}
                  data-testid="input-whatsapp-reply"
                />
                <Button 
                  onClick={() => sendWhatsAppMessage.mutate({
                    toNumber: whatsappReplyNumber,
                    message: whatsappReplyText
                  })}
                  disabled={!whatsappReplyText.trim() || sendWhatsAppMessage.isPending}
                  data-testid="button-send-whatsapp"
                >
                  {sendWhatsAppMessage.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          )}
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => { setWhatsappReplyNumber(null); setWhatsappReplyText(""); }} data-testid="button-close-whatsapp">
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
