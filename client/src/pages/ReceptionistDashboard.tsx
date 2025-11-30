import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Calendar, Clock, Mail, Phone, Plus, User, Video, Bot, MessageSquare, Stethoscope, Pill, FileText, ChevronRight, Sparkles, LayoutDashboard, Users, Activity } from "lucide-react";
import { format, startOfWeek, addDays, isSameDay, parseISO } from "date-fns";
import { LysaChatPanel, LysaQuickActionsBar } from "@/components/LysaChatPanel";
import { EmailAIHelper } from "@/components/EmailAIHelper";
import { PatientManagementPanel } from "@/components/PatientManagementPanel";
import { DiagnosisHelper } from "@/components/DiagnosisHelper";
import { PrescriptionHelper } from "@/components/PrescriptionHelper";
import { AutomationStatusPanel, AutomationStatusBadge } from "@/components/AutomationStatusPanel";

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
  category?: string;
  priority?: string;
  sentiment?: string;
  isRead: boolean;
  receivedAt: string;
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
  const weekStart = startOfWeek(selectedDate);
  const weekDays = Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));

  const { data: upcomingAppointments = [], isLoading: appointmentsLoading } = useQuery<Appointment[]>({
    queryKey: ["/api/v1/appointments/upcoming", { days: 7 }],
  });

  const { data: emailThreads = [], isLoading: emailsLoading } = useQuery<EmailThread[]>({
    queryKey: ["/api/v1/emails/threads", { limit: 10 }],
  });

  // Fetch integration status
  const { data: integrationStatus } = useQuery<IntegrationStatus>({
    queryKey: ["/api/v1/integrations/status"],
  });

  // Fetch synced emails from connected Gmail
  const { data: syncedEmails = [] } = useQuery<SyncedEmail[]>({
    queryKey: ["/api/v1/integrations/gmail/emails"],
    enabled: integrationStatus?.gmail.connected === true,
  });


  // Combine email sources - synced emails take priority if Gmail is connected
  const combinedEmails = integrationStatus?.gmail.connected && syncedEmails.length > 0
    ? syncedEmails.map(email => ({
        id: email.id,
        doctorId: email.doctorId,
        subject: email.subject,
        category: email.category || 'general',
        priority: email.priority || 'normal',
        status: 'active',
        isRead: email.isRead,
        messageCount: 1,
        lastMessageAt: email.receivedAt,
        from: email.from,
      }))
    : emailThreads;

  const unreadEmails = combinedEmails.filter(thread => !thread.isRead).length;
  const urgentEmails = combinedEmails.filter(thread => thread.priority === 'urgent').length;
  const todayAppointments = upcomingAppointments.filter(apt => 
    isSameDay(parseISO(apt.startTime), new Date())
  );

  const getAppointmentsForDay = (date: Date) => {
    return upcomingAppointments.filter(apt => 
      isSameDay(parseISO(apt.startTime), date)
    );
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

  const getPriorityColor = (priority: string) => {
    const colors: Record<string, string> = {
      urgent: 'destructive',
      high: 'default',
      normal: 'secondary',
      low: 'outline'
    };
    return colors[priority] || 'outline';
  };

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
            <Button data-testid="button-book-appointment">
              <Plus className="h-4 w-4 mr-2" />
              Book Appointment
            </Button>
          </div>
        </div>
        
        <LysaQuickActionsBar />

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
            <TabsTrigger value="emails" data-testid="tab-emails">
              <Mail className="h-4 w-4 mr-2" />
              Email AI
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
        <div className="grid gap-4 md:grid-cols-4">
          <Card data-testid="card-today-appointments">
            <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Today's Appointments</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{todayAppointments.length}</div>
              <p className="text-xs text-muted-foreground">
                {upcomingAppointments.length} total this week
              </p>
            </CardContent>
          </Card>

          <Card data-testid="card-unread-emails">
            <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Unread Emails</CardTitle>
              <Mail className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{unreadEmails}</div>
              <p className="text-xs text-muted-foreground">
                {urgentEmails} urgent messages
              </p>
            </CardContent>
          </Card>

          <Card data-testid="card-lysa-status">
            <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Lysa Automation</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-sm font-medium text-green-600">Active</span>
              </div>
              <AutomationStatusBadge />
            </CardContent>
          </Card>

          <Card data-testid="card-quick-actions">
            <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Quick Actions</CardTitle>
              <Video className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-2">
                <Button variant="outline" size="sm" data-testid="button-view-calendar" className="justify-start">
                  <Calendar className="h-3 w-3 mr-2" />
                  Calendar
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card data-testid="card-week-calendar">
            <CardHeader>
              <CardTitle>Weekly Calendar</CardTitle>
              <CardDescription>
                {format(weekStart, "MMM d")} - {format(addDays(weekStart, 6), "MMM d, yyyy")}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {appointmentsLoading ? (
                <div className="animate-pulse space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-12 bg-muted rounded" />
                  ))}
                </div>
              ) : (
                <div className="grid grid-cols-7 gap-2">
                  {weekDays.map((day) => {
                    const dayAppointments = getAppointmentsForDay(day);
                    const isToday = isSameDay(day, new Date());
                    
                    return (
                      <div
                        key={day.toISOString()}
                        className={`p-3 rounded-md border ${
                          isToday ? "border-primary bg-primary/5" : "border-border"
                        }`}
                        data-testid={`calendar-day-${format(day, "yyyy-MM-dd")}`}
                      >
                        <div className="text-sm font-medium mb-2">
                          {format(day, "EEE")}
                        </div>
                        <div className="text-2xl font-bold mb-2">
                          {format(day, "d")}
                        </div>
                        {dayAppointments.length > 0 && (
                          <div className="space-y-1">
                            {dayAppointments.slice(0, 2).map((apt) => (
                              <div
                                key={apt.id}
                                className="text-xs p-1 rounded bg-primary/10 text-primary truncate"
                                data-testid={`appointment-${apt.id}`}
                              >
                                {format(parseISO(apt.startTime), "HH:mm")}
                              </div>
                            ))}
                            {dayAppointments.length > 2 && (
                              <div className="text-xs text-muted-foreground">
                                +{dayAppointments.length - 2} more
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          <Card data-testid="card-upcoming-appointments">
            <CardHeader>
              <CardTitle>Upcoming Appointments</CardTitle>
              <CardDescription>Next 7 days</CardDescription>
            </CardHeader>
            <CardContent>
              {appointmentsLoading ? (
                <div className="animate-pulse space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-16 bg-muted rounded" />
                  ))}
                </div>
              ) : upcomingAppointments.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Calendar className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No upcoming appointments</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {upcomingAppointments.slice(0, 5).map((appointment) => (
                    <div
                      key={appointment.id}
                      className="flex items-center justify-between p-3 rounded-md border hover-elevate active-elevate-2"
                      data-testid={`appointment-item-${appointment.id}`}
                    >
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                          <User className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <p className="font-medium">
                            {appointment.patientName || `Patient ${appointment.patientId.slice(0, 8)}`}
                          </p>
                          <p className="text-sm text-muted-foreground flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {format(parseISO(appointment.startTime), "MMM d, h:mm a")}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={appointment.confirmationStatus === 'confirmed' ? 'default' : 'secondary'}>
                          {appointment.confirmationStatus}
                        </Badge>
                        <Badge variant="outline">{appointment.appointmentType}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card data-testid="card-email-inbox">
            <CardHeader>
              <CardTitle>Email Inbox</CardTitle>
              <CardDescription>Recent email threads</CardDescription>
            </CardHeader>
            <CardContent>
              {emailsLoading ? (
                <div className="animate-pulse space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-16 bg-muted rounded" />
                  ))}
                </div>
              ) : combinedEmails.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Mail className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No email threads</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {combinedEmails.slice(0, 5).map((thread) => (
                    <div
                      key={thread.id}
                      className={`p-3 rounded-md border hover-elevate active-elevate-2 ${
                        !thread.isRead ? "bg-primary/5 border-primary/20" : ""
                      }`}
                      data-testid={`email-thread-${thread.id}`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <p className={`font-medium ${!thread.isRead ? "text-primary" : ""}`}>
                            {thread.subject}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {thread.messageCount} message{thread.messageCount !== 1 ? 's' : ''} · 
                            {format(parseISO(thread.lastMessageAt), " MMM d, h:mm a")}
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={getCategoryColor(thread.category) as any}>
                            {thread.category}
                          </Badge>
                          {thread.priority !== 'normal' && (
                            <Badge variant={getPriorityColor(thread.priority) as any}>
                              {thread.priority}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card data-testid="card-lysa-capabilities">
            <CardHeader>
              <CardTitle>Lysa Capabilities</CardTitle>
              <CardDescription>What your AI assistant can do</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="p-3 rounded-md border hover-elevate active-elevate-2">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-blue-500/10 flex items-center justify-center">
                      <Calendar className="h-5 w-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium">Appointment Booking</p>
                      <p className="text-sm text-muted-foreground">Schedule and manage patient appointments</p>
                    </div>
                  </div>
                </div>
                <div className="p-3 rounded-md border hover-elevate active-elevate-2">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-green-500/10 flex items-center justify-center">
                      <Mail className="h-5 w-5 text-green-600" />
                    </div>
                    <div>
                      <p className="font-medium">Email Drafting</p>
                      <p className="text-sm text-muted-foreground">AI-assisted professional responses</p>
                    </div>
                  </div>
                </div>
                <div className="p-3 rounded-md border hover-elevate active-elevate-2">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-purple-500/10 flex items-center justify-center">
                      <Users className="h-5 w-5 text-purple-600" />
                    </div>
                    <div>
                      <p className="font-medium">Patient Records</p>
                      <p className="text-sm text-muted-foreground">Quick access to patient information</p>
                    </div>
                  </div>
                </div>
                <div className="p-3 rounded-md border hover-elevate active-elevate-2">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-orange-500/10 flex items-center justify-center">
                      <Stethoscope className="h-5 w-5 text-orange-600" />
                    </div>
                    <div>
                      <p className="font-medium">Clinical Support</p>
                      <p className="text-sm text-muted-foreground">Diagnosis and prescription assistance</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
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

          <TabsContent value="emails" className="mt-0">
            <div className="grid gap-6 lg:grid-cols-2">
              <EmailAIHelper />
              <Card data-testid="card-email-stats">
                <CardHeader>
                  <CardTitle>Email Analytics</CardTitle>
                  <CardDescription>AI-powered email insights</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-blue-500/10 flex items-center justify-center">
                          <Mail className="h-5 w-5 text-blue-600" />
                        </div>
                        <div>
                          <p className="font-medium">Total Threads</p>
                          <p className="text-sm text-muted-foreground">Active email conversations</p>
                        </div>
                      </div>
                      <span className="text-2xl font-bold">{emailThreads.length}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-orange-500/10 flex items-center justify-center">
                          <Sparkles className="h-5 w-5 text-orange-600" />
                        </div>
                        <div>
                          <p className="font-medium">Urgent Messages</p>
                          <p className="text-sm text-muted-foreground">Requires attention</p>
                        </div>
                      </div>
                      <span className="text-2xl font-bold">{urgentEmails}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-green-500/10 flex items-center justify-center">
                          <MessageSquare className="h-5 w-5 text-green-600" />
                        </div>
                        <div>
                          <p className="font-medium">Unread</p>
                          <p className="text-sm text-muted-foreground">Awaiting review</p>
                        </div>
                      </div>
                      <span className="text-2xl font-bold">{unreadEmails}</span>
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
    </div>
  );
}
