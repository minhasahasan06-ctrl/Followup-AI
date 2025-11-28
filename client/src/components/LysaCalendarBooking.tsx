import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Calendar as CalendarUI } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { 
  Calendar, Clock, User, Plus, Check, X, Loader2, 
  AlertTriangle, CheckCircle2, RefreshCw, CalendarDays,
  Video, MapPin, Link2Off
} from "lucide-react";
import { SiGoogle } from "react-icons/si";
import { format, addDays, parseISO, isSameDay, addMinutes, startOfHour, setHours, setMinutes } from "date-fns";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { cn } from "@/lib/utils";

interface CalendarEvent {
  id: string;
  summary: string;
  description?: string;
  location?: string;
  start: { dateTime?: string; date?: string };
  end: { dateTime?: string; date?: string };
  status: string;
  hangoutLink?: string;
}

interface Patient {
  id: string;
  firstName: string;
  lastName: string;
  email?: string;
}

interface ConnectorStatus {
  connected: boolean;
  calendarId?: string;
  calendarName?: string;
  email?: string;
  source: string;
}

interface TimeSlot {
  start: Date;
  end: Date;
  available: boolean;
}

interface BookAppointmentDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface CheckAvailabilityDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function BookAppointmentDialog({ open, onOpenChange }: BookAppointmentDialogProps) {
  const { toast } = useToast();
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [selectedTime, setSelectedTime] = useState<string>("");
  const [duration, setDuration] = useState<string>("30");
  const [appointmentType, setAppointmentType] = useState<string>("consultation");
  const [patientSearch, setPatientSearch] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [notes, setNotes] = useState("");
  const [location, setLocation] = useState("");

  const { data: connectorStatus, isLoading: connectorLoading } = useQuery<ConnectorStatus>({
    queryKey: ["/api/v1/calendar/connector-status"],
    enabled: open,
  });

  const { data: eventsData, isLoading: eventsLoading } = useQuery<{ events: CalendarEvent[] }>({
    queryKey: ["/api/v1/calendar/events"],
    enabled: open && connectorStatus?.connected === true,
  });

  const { data: allPatients = [], isLoading: patientsLoading } = useQuery<Patient[]>({
    queryKey: ["/api/doctor/patients"],
    enabled: open,
  });

  const patients = patientSearch.length >= 2
    ? allPatients.filter(p => 
        `${p.firstName} ${p.lastName}`.toLowerCase().includes(patientSearch.toLowerCase())
      )
    : [];

  const bookAppointmentMutation = useMutation({
    mutationFn: async (appointmentData: any) => {
      return apiRequest("/api/v1/appointments", {
        method: "POST",
        json: appointmentData,
      });
    },
    onSuccess: () => {
      toast({
        title: "Appointment Booked",
        description: "The appointment has been successfully scheduled and synced to Google Calendar.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/calendar/events"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/appointments"] });
      onOpenChange(false);
      resetForm();
    },
    onError: (error: any) => {
      toast({
        title: "Booking Failed",
        description: error.message || "Failed to book appointment. Please try again.",
        variant: "destructive",
      });
    },
  });

  const resetForm = () => {
    setSelectedDate(new Date());
    setSelectedTime("");
    setDuration("30");
    setAppointmentType("consultation");
    setSelectedPatient(null);
    setPatientSearch("");
    setNotes("");
    setLocation("");
  };

  const generateTimeSlots = (): string[] => {
    const slots: string[] = [];
    for (let hour = 8; hour <= 18; hour++) {
      slots.push(`${hour.toString().padStart(2, '0')}:00`);
      slots.push(`${hour.toString().padStart(2, '0')}:30`);
    }
    return slots;
  };

  const isTimeSlotBusy = (time: string): boolean => {
    if (!eventsData?.events) return false;
    
    const [hours, minutes] = time.split(':').map(Number);
    const slotStart = setMinutes(setHours(selectedDate, hours), minutes);
    const slotEnd = addMinutes(slotStart, parseInt(duration));
    
    return eventsData.events.some(event => {
      if (!event.start?.dateTime || !event.end?.dateTime) return false;
      const eventStart = parseISO(event.start.dateTime);
      const eventEnd = parseISO(event.end.dateTime);
      
      if (!isSameDay(eventStart, selectedDate)) return false;
      
      return (slotStart < eventEnd && slotEnd > eventStart);
    });
  };

  const handleBook = () => {
    if (!selectedPatient || !selectedTime) {
      toast({
        title: "Missing Information",
        description: "Please select a patient and time slot.",
        variant: "destructive",
      });
      return;
    }

    const [hours, minutes] = selectedTime.split(':').map(Number);
    const startTime = setMinutes(setHours(selectedDate, hours), minutes);
    const endTime = addMinutes(startTime, parseInt(duration));

    bookAppointmentMutation.mutate({
      patientId: selectedPatient.id,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
      appointmentType,
      notes,
      location,
    });
  };

  const isConnected = connectorStatus?.connected === true;
  const timeSlots = generateTimeSlots();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto" data-testid="dialog-book-appointment">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5 text-primary" />
            Book Appointment
          </DialogTitle>
          <DialogDescription>
            Schedule a new appointment with Google Calendar sync
          </DialogDescription>
        </DialogHeader>

        {connectorLoading ? (
          <div className="space-y-4 py-4">
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        ) : !isConnected ? (
          <Alert className="my-4">
            <Link2Off className="h-4 w-4" />
            <AlertTitle>Google Calendar Not Connected</AlertTitle>
            <AlertDescription>
              Connect your Google Calendar via Replit integrations to enable appointment sync. 
              You can still book appointments manually without calendar sync.
            </AlertDescription>
          </Alert>
        ) : (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 my-2">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <span className="text-sm text-green-700 dark:text-green-400">
              Connected to {connectorStatus?.calendarName || "Google Calendar"}
            </span>
          </div>
        )}

        <div className="grid gap-4 py-4">
          <div className="space-y-2">
            <Label>Patient</Label>
            {selectedPatient ? (
              <div className="flex items-center justify-between p-3 rounded-md border bg-accent/50">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                    <User className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="font-medium">{selectedPatient.firstName} {selectedPatient.lastName}</p>
                    <p className="text-sm text-muted-foreground">{selectedPatient.email || "No email"}</p>
                  </div>
                </div>
                <Button 
                  variant="ghost" 
                  size="icon"
                  onClick={() => setSelectedPatient(null)}
                  data-testid="button-clear-patient"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <div className="space-y-2">
                <Input
                  placeholder="Search patient by name..."
                  value={patientSearch}
                  onChange={(e) => setPatientSearch(e.target.value)}
                  data-testid="input-patient-search"
                />
                {patientsLoading && (
                  <div className="flex items-center gap-2 p-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading patients...
                  </div>
                )}
                {patientSearch.length >= 2 && !patientsLoading && (
                  patients.length > 0 ? (
                    <ScrollArea className="h-32 rounded-md border">
                      <div className="p-2 space-y-1">
                        {patients.map((patient) => (
                          <Button
                            key={patient.id}
                            variant="ghost"
                            className="w-full justify-start h-auto py-2"
                            onClick={() => {
                              setSelectedPatient(patient);
                              setPatientSearch("");
                            }}
                            data-testid={`patient-option-${patient.id}`}
                          >
                            <User className="h-4 w-4 mr-2" />
                            {patient.firstName} {patient.lastName}
                          </Button>
                        ))}
                      </div>
                    </ScrollArea>
                  ) : (
                    <p className="text-sm text-muted-foreground p-2">No patients found matching "{patientSearch}"</p>
                  )
                )}
              </div>
            )}
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Date</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-full justify-start text-left font-normal",
                      !selectedDate && "text-muted-foreground"
                    )}
                    data-testid="button-select-date"
                  >
                    <CalendarDays className="mr-2 h-4 w-4" />
                    {selectedDate ? format(selectedDate, "PPP") : "Pick a date"}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <CalendarUI
                    mode="single"
                    selected={selectedDate}
                    onSelect={(date) => date && setSelectedDate(date)}
                    disabled={(date) => date < new Date()}
                    initialFocus
                  />
                </PopoverContent>
              </Popover>
            </div>

            <div className="space-y-2">
              <Label>Duration</Label>
              <Select value={duration} onValueChange={setDuration}>
                <SelectTrigger data-testid="select-duration">
                  <SelectValue placeholder="Select duration" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="15">15 minutes</SelectItem>
                  <SelectItem value="30">30 minutes</SelectItem>
                  <SelectItem value="45">45 minutes</SelectItem>
                  <SelectItem value="60">1 hour</SelectItem>
                  <SelectItem value="90">1.5 hours</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Available Time Slots</Label>
            {eventsLoading ? (
              <div className="grid grid-cols-4 gap-2">
                {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                  <Skeleton key={i} className="h-9" />
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-4 gap-2">
                {timeSlots.map((time) => {
                  const isBusy = isConnected && isTimeSlotBusy(time);
                  const isSelected = selectedTime === time;
                  return (
                    <Button
                      key={time}
                      variant={isSelected ? "default" : isBusy ? "ghost" : "outline"}
                      size="sm"
                      disabled={isBusy}
                      className={cn(
                        isBusy && "opacity-40 cursor-not-allowed line-through"
                      )}
                      onClick={() => setSelectedTime(time)}
                      data-testid={`time-slot-${time.replace(':', '')}`}
                    >
                      {time}
                    </Button>
                  );
                })}
              </div>
            )}
            {isConnected && (
              <p className="text-xs text-muted-foreground">
                Grayed slots are busy in your Google Calendar
              </p>
            )}
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Appointment Type</Label>
              <Select value={appointmentType} onValueChange={setAppointmentType}>
                <SelectTrigger data-testid="select-appointment-type">
                  <SelectValue placeholder="Select type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="consultation">Consultation</SelectItem>
                  <SelectItem value="follow-up">Follow-up</SelectItem>
                  <SelectItem value="checkup">Check-up</SelectItem>
                  <SelectItem value="emergency">Emergency</SelectItem>
                  <SelectItem value="telemedicine">Telemedicine</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Location (Optional)</Label>
              <div className="flex items-center gap-2">
                <MapPin className="h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Office, Room 101, etc."
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  data-testid="input-location"
                />
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Notes (Optional)</Label>
            <Textarea
              placeholder="Reason for visit, special instructions..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={3}
              data-testid="input-notes"
            />
          </div>
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)} data-testid="button-cancel-booking">
            Cancel
          </Button>
          <Button 
            onClick={handleBook}
            disabled={!selectedPatient || !selectedTime || bookAppointmentMutation.isPending}
            data-testid="button-confirm-booking"
          >
            {bookAppointmentMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Booking...
              </>
            ) : (
              <>
                <Check className="h-4 w-4 mr-2" />
                Book Appointment
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function CheckAvailabilityDialog({ open, onOpenChange }: CheckAvailabilityDialogProps) {
  const { toast } = useToast();
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [daysToShow, setDaysToShow] = useState<string>("3");

  const { data: connectorStatus, isLoading: connectorLoading, refetch: refetchConnector } = useQuery<ConnectorStatus>({
    queryKey: ["/api/v1/calendar/connector-status"],
    enabled: open,
  });

  const { data: eventsData, isLoading: eventsLoading, refetch: refetchEvents } = useQuery<{ events: CalendarEvent[] }>({
    queryKey: ["/api/v1/calendar/events"],
    enabled: open && connectorStatus?.connected === true,
  });

  const isConnected = connectorStatus?.connected === true;

  const generateAvailabilityForDay = (date: Date): TimeSlot[] => {
    const slots: TimeSlot[] = [];
    const events = eventsData?.events || [];
    
    for (let hour = 8; hour <= 17; hour++) {
      for (const minutes of [0, 30]) {
        const slotStart = setMinutes(setHours(date, hour), minutes);
        const slotEnd = addMinutes(slotStart, 30);
        
        const isBusy = events.some(event => {
          if (!event.start?.dateTime || !event.end?.dateTime) return false;
          const eventStart = parseISO(event.start.dateTime);
          const eventEnd = parseISO(event.end.dateTime);
          
          if (!isSameDay(eventStart, date)) return false;
          
          return (slotStart < eventEnd && slotEnd > eventStart);
        });
        
        slots.push({
          start: slotStart,
          end: slotEnd,
          available: !isBusy,
        });
      }
    }
    
    return slots;
  };

  const daysToDisplay = Array.from({ length: parseInt(daysToShow) }, (_, i) => addDays(selectedDate, i));

  const getEventsForDay = (date: Date): CalendarEvent[] => {
    if (!eventsData?.events) return [];
    return eventsData.events.filter(event => {
      if (!event.start?.dateTime) return false;
      return isSameDay(parseISO(event.start.dateTime), date);
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto" data-testid="dialog-check-availability">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-primary" />
            Check Availability
          </DialogTitle>
          <DialogDescription>
            View calendar availability and scheduled events
          </DialogDescription>
        </DialogHeader>

        {connectorLoading ? (
          <div className="space-y-4 py-4">
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-60 w-full" />
          </div>
        ) : !isConnected ? (
          <div className="space-y-4 py-4">
            <Alert>
              <Link2Off className="h-4 w-4" />
              <AlertTitle>Google Calendar Not Connected</AlertTitle>
              <AlertDescription>
                Connect your Google Calendar via Replit integrations to view real-time availability.
              </AlertDescription>
            </Alert>
            <div className="p-6 rounded-lg border border-dashed flex flex-col items-center justify-center gap-4 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                <SiGoogle className="h-8 w-8 text-muted-foreground" />
              </div>
              <div>
                <h3 className="font-medium">Connect Google Calendar</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Enable real-time availability checking with your calendar
                </p>
              </div>
              <Button 
                variant="outline" 
                onClick={() => refetchConnector()}
                data-testid="button-retry-connection"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Check Connection
              </Button>
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between gap-4 py-2">
              <div className="flex items-center gap-2 p-2 rounded-lg bg-green-50 dark:bg-green-900/20">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <span className="text-sm text-green-700 dark:text-green-400">
                  {connectorStatus?.calendarName || "Google Calendar"}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Select value={daysToShow} onValueChange={setDaysToShow}>
                  <SelectTrigger className="w-32" data-testid="select-days-to-show">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">Today</SelectItem>
                    <SelectItem value="3">3 Days</SelectItem>
                    <SelectItem value="5">5 Days</SelectItem>
                    <SelectItem value="7">1 Week</SelectItem>
                  </SelectContent>
                </Select>
                <Button 
                  variant="outline" 
                  size="icon"
                  onClick={() => refetchEvents()}
                  data-testid="button-refresh-availability"
                >
                  <RefreshCw className={cn("h-4 w-4", eventsLoading && "animate-spin")} />
                </Button>
              </div>
            </div>

            {eventsLoading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-40 w-full" />
                ))}
              </div>
            ) : (
              <ScrollArea className="h-[400px]">
                <div className="space-y-4">
                  {daysToDisplay.map((date) => {
                    const dayEvents = getEventsForDay(date);
                    const slots = generateAvailabilityForDay(date);
                    const availableSlots = slots.filter(s => s.available);
                    const isToday = isSameDay(date, new Date());
                    
                    return (
                      <Card key={date.toISOString()} data-testid={`day-card-${format(date, 'yyyy-MM-dd')}`}>
                        <CardHeader className="pb-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <CardTitle className="text-base">
                                {format(date, "EEEE, MMM d")}
                              </CardTitle>
                              {isToday && (
                                <Badge variant="secondary">Today</Badge>
                              )}
                            </div>
                            <Badge variant={availableSlots.length > 5 ? "default" : availableSlots.length > 0 ? "secondary" : "destructive"}>
                              {availableSlots.length} slots available
                            </Badge>
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          {dayEvents.length > 0 && (
                            <div className="space-y-2">
                              <p className="text-sm font-medium text-muted-foreground">Scheduled Events</p>
                              <div className="space-y-1">
                                {dayEvents.map((event) => (
                                  <div 
                                    key={event.id}
                                    className="flex items-center justify-between p-2 rounded-md bg-accent/50 text-sm"
                                    data-testid={`event-item-${event.id}`}
                                  >
                                    <div className="flex items-center gap-2">
                                      <Clock className="h-3 w-3 text-muted-foreground" />
                                      <span>
                                        {event.start?.dateTime 
                                          ? format(parseISO(event.start.dateTime), "h:mm a")
                                          : "All day"
                                        }
                                        {event.end?.dateTime && (
                                          <> - {format(parseISO(event.end.dateTime), "h:mm a")}</>
                                        )}
                                      </span>
                                    </div>
                                    <span className="font-medium truncate max-w-[200px]">
                                      {event.summary || "Busy"}
                                    </span>
                                    {event.hangoutLink && (
                                      <Video className="h-3 w-3 text-primary" />
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          <div className="space-y-2">
                            <p className="text-sm font-medium text-muted-foreground">Available Slots</p>
                            <div className="flex flex-wrap gap-1">
                              {availableSlots.length > 0 ? (
                                availableSlots.map((slot) => (
                                  <Badge 
                                    key={slot.start.toISOString()} 
                                    variant="outline"
                                    className="text-xs"
                                    data-testid={`available-slot-${format(slot.start, 'HHmm')}`}
                                  >
                                    {format(slot.start, "h:mm a")}
                                  </Badge>
                                ))
                              ) : (
                                <p className="text-sm text-muted-foreground">No available slots</p>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </ScrollArea>
            )}
          </>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} data-testid="button-close-availability">
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function CalendarStatusBadge() {
  const { data: connectorStatus, isLoading } = useQuery<ConnectorStatus>({
    queryKey: ["/api/v1/calendar/connector-status"],
    staleTime: 60000,
  });

  if (isLoading) {
    return <Skeleton className="h-5 w-20" />;
  }

  const isConnected = connectorStatus?.connected === true;

  return (
    <Badge 
      variant={isConnected ? "secondary" : "outline"}
      className="text-xs"
      data-testid="badge-calendar-status"
    >
      {isConnected ? (
        <>
          <SiGoogle className="h-3 w-3 mr-1" />
          Calendar Synced
        </>
      ) : (
        <>
          <Link2Off className="h-3 w-3 mr-1" />
          Not Connected
        </>
      )}
    </Badge>
  );
}
