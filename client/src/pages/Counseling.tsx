import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { insertCounselingSessionSchema, type CounselingSession } from "@shared/schema";
import { format } from "date-fns";
import { Calendar, Clock, User, Video, Phone, Users, AlertCircle, Plus, Trash2, Edit2, Star } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { z } from "zod";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useAuth } from "@/hooks/useAuth";

const formSchema = insertCounselingSessionSchema.extend({
  sessionDate: z.string().min(1, "Session date is required"),
});

type FormValues = z.infer<typeof formSchema>;

export default function Counseling() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [editingSession, setEditingSession] = useState<CounselingSession | null>(null);

  const { data: sessions = [], isLoading } = useQuery<CounselingSession[]>({
    queryKey: ["/api/counseling/sessions"],
  });

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      sessionDate: "",
      duration: 60,
      counselorName: "",
      sessionType: "individual",
      sessionMode: "video",
      mainConcerns: [],
      sessionNotes: "",
      followUpRequired: false,
      status: "scheduled",
    },
  });

  const createMutation = useMutation({
    mutationFn: async (data: FormValues) => {
      const sessionDate = new Date(data.sessionDate).toISOString();
      return apiRequest("/api/counseling/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...data,
          sessionDate,
        }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/counseling/sessions"] });
      setIsCreateDialogOpen(false);
      form.reset();
      toast({
        title: "Session created",
        description: "Your counseling session has been scheduled.",
      });
    },
  });

  const updateMutation = useMutation({
    mutationFn: async ({ id, data }: { id: string; data: Partial<CounselingSession> }) => {
      return apiRequest(`/api/counseling/sessions/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/counseling/sessions"] });
      toast({
        title: "Session updated",
        description: "Your counseling session has been updated.",
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiRequest(`/api/counseling/sessions/${id}`, {
        method: "DELETE",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/counseling/sessions"] });
      toast({
        title: "Session deleted",
        description: "Your counseling session has been removed.",
      });
    },
  });

  const handleSubmit = (data: FormValues) => {
    createMutation.mutate(data);
  };

  const handleStatusChange = (session: CounselingSession, status: string) => {
    updateMutation.mutate({ id: session.id, data: { status } });
  };

  const handleDelete = (id: string) => {
    if (confirm("Are you sure you want to delete this session?")) {
      deleteMutation.mutate(id);
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, { variant: "default" | "secondary" | "destructive" | "outline", label: string }> = {
      scheduled: { variant: "default", label: "Scheduled" },
      completed: { variant: "secondary", label: "Completed" },
      cancelled: { variant: "destructive", label: "Cancelled" },
      "no-show": { variant: "outline", label: "No-Show" },
    };
    const config = variants[status] || variants.scheduled;
    return <Badge variant={config.variant} data-testid={`badge-status-${status}`}>{config.label}</Badge>;
  };

  const getSessionIcon = (mode: string) => {
    switch (mode) {
      case "video": return <Video className="h-4 w-4" />;
      case "phone": return <Phone className="h-4 w-4" />;
      case "in-person": return <User className="h-4 w-4" />;
      default: return <Video className="h-4 w-4" />;
    }
  };

  const getSessionTypeIcon = (type: string) => {
    switch (type) {
      case "group": return <Users className="h-4 w-4" />;
      case "family": return <Users className="h-4 w-4" />;
      case "emergency": return <AlertCircle className="h-4 w-4" />;
      default: return <User className="h-4 w-4" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto mb-4" />
          <p className="text-muted-foreground">Loading sessions...</p>
        </div>
      </div>
    );
  }

  const upcomingSessions = sessions.filter(s => s.status === "scheduled" && new Date(s.sessionDate) >= new Date());
  const pastSessions = sessions.filter(s => s.status === "completed" || new Date(s.sessionDate) < new Date());

  const isDoctor = user?.role === "doctor";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight" data-testid="heading-counseling">
            Psychological Counseling
          </h1>
          <p className="text-muted-foreground mt-1">
            {isDoctor 
              ? "Manage your professional counseling sessions and mental health support"
              : "Manage your counseling sessions and mental health support"
            }
          </p>
        </div>
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-new-session">
              <Plus className="h-4 w-4 mr-2" />
              Schedule Session
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl" data-testid="dialog-new-session">
            <DialogHeader>
              <DialogTitle>Schedule Counseling Session</DialogTitle>
              <DialogDescription>
                Book a new counseling session with your therapist or counselor
              </DialogDescription>
            </DialogHeader>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="sessionDate"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Date & Time</FormLabel>
                        <FormControl>
                          <Input type="datetime-local" {...field} data-testid="input-session-date" />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="duration"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Duration (minutes)</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            {...field}
                            onChange={(e) => field.onChange(parseInt(e.target.value))}
                            data-testid="input-duration"
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="counselorName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Counselor Name</FormLabel>
                      <FormControl>
                        <Input {...field} placeholder="Dr. Smith" data-testid="input-counselor-name" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="sessionType"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Session Type</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value || undefined}>
                          <FormControl>
                            <SelectTrigger data-testid="select-session-type">
                              <SelectValue placeholder="Select type" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="individual">Individual</SelectItem>
                            <SelectItem value="group">Group</SelectItem>
                            <SelectItem value="family">Family</SelectItem>
                            <SelectItem value="emergency">Emergency</SelectItem>
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="sessionMode"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Session Mode</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value || undefined}>
                          <FormControl>
                            <SelectTrigger data-testid="select-session-mode">
                              <SelectValue placeholder="Select mode" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="video">Video Call</SelectItem>
                            <SelectItem value="phone">Phone Call</SelectItem>
                            <SelectItem value="in-person">In-Person</SelectItem>
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="sessionNotes"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Session Notes (Optional)</FormLabel>
                      <FormControl>
                        <Textarea
                          {...field}
                          placeholder="Any specific topics or concerns to discuss..."
                          className="min-h-24"
                          data-testid="textarea-session-notes"
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <DialogFooter>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setIsCreateDialogOpen(false)}
                    data-testid="button-cancel-session"
                  >
                    Cancel
                  </Button>
                  <Button type="submit" disabled={createMutation.isPending} data-testid="button-create-session">
                    {createMutation.isPending ? "Scheduling..." : "Schedule Session"}
                  </Button>
                </DialogFooter>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
      </div>

      {sessions.length === 0 ? (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            You don't have any counseling sessions scheduled yet. Click "Schedule Session" to book your first appointment.
          </AlertDescription>
        </Alert>
      ) : (
        <>
          {upcomingSessions.length > 0 && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold" data-testid="heading-upcoming-sessions">
                Upcoming Sessions
              </h2>
              <div className="grid gap-4">
                {upcomingSessions.map((session) => (
                  <Card key={session.id} data-testid={`card-session-${session.id}`}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                            {getSessionTypeIcon(session.sessionType || "individual")}
                          </div>
                          <div>
                            <CardTitle className="text-lg">
                              {session.counselorName || "Counseling Session"}
                            </CardTitle>
                            <CardDescription className="flex items-center gap-4 mt-1">
                              <span className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" />
                                {format(new Date(session.sessionDate), "PPP")}
                              </span>
                              <span className="flex items-center gap-1">
                                <Clock className="h-3 w-3" />
                                {format(new Date(session.sessionDate), "p")}
                              </span>
                              {session.duration && (
                                <span>{session.duration} min</span>
                              )}
                            </CardDescription>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {getStatusBadge(session.status || "scheduled")}
                          <Button
                            size="icon"
                            variant="ghost"
                            onClick={() => handleDelete(session.id)}
                            data-testid={`button-delete-${session.id}`}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center gap-4 text-sm">
                        <span className="flex items-center gap-1">
                          {getSessionIcon(session.sessionMode || "video")}
                          <span className="capitalize">{session.sessionMode}</span>
                        </span>
                        <span className="capitalize">{session.sessionType} Session</span>
                      </div>
                      {session.sessionNotes && (
                        <p className="text-sm text-muted-foreground border-l-2 border-primary pl-3">
                          {session.sessionNotes}
                        </p>
                      )}
                      <div className="flex gap-2 pt-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleStatusChange(session, "completed")}
                          data-testid={`button-complete-${session.id}`}
                        >
                          Mark Complete
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleStatusChange(session, "cancelled")}
                          data-testid={`button-cancel-${session.id}`}
                        >
                          Cancel Session
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {pastSessions.length > 0 && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold" data-testid="heading-past-sessions">
                Past Sessions
              </h2>
              <div className="grid gap-4">
                {pastSessions.map((session) => (
                  <Card key={session.id} className="opacity-75" data-testid={`card-past-session-${session.id}`}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-muted">
                            {getSessionTypeIcon(session.sessionType || "individual")}
                          </div>
                          <div>
                            <CardTitle className="text-lg">
                              {session.counselorName || "Counseling Session"}
                            </CardTitle>
                            <CardDescription className="flex items-center gap-4 mt-1">
                              <span className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" />
                                {format(new Date(session.sessionDate), "PPP")}
                              </span>
                              {session.duration && (
                                <span>{session.duration} min</span>
                              )}
                            </CardDescription>
                          </div>
                        </div>
                        {getStatusBadge(session.status || "scheduled")}
                      </div>
                    </CardHeader>
                    {session.sessionNotes && (
                      <CardContent>
                        <p className="text-sm text-muted-foreground">
                          {session.sessionNotes}
                        </p>
                      </CardContent>
                    )}
                  </Card>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
