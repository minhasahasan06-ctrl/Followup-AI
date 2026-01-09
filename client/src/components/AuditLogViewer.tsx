import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  FileText, 
  UserPlus, 
  Pill, 
  AlertTriangle, 
  Activity,
  ChevronLeft,
  ChevronRight,
  Clock,
  Shield
} from "lucide-react";
import { format } from "date-fns";
import { useState } from "react";

interface AuditLogEntry {
  id: number;
  event_type: string;
  event_data: Record<string, any>;
  actor_id?: string;
  created_at: string;
}

interface AuditLogResponse {
  logs: AuditLogEntry[];
  total: number;
  page: number;
  page_size: number;
}

const eventTypeConfig: Record<string, { icon: any; label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
  terms_accepted: { icon: FileText, label: "Terms Accepted", variant: "default" },
  research_consent_changed: { icon: Shield, label: "Research Consent", variant: "secondary" },
  doctor_assigned: { icon: UserPlus, label: "Doctor Assigned", variant: "default" },
  doctor_unassigned: { icon: UserPlus, label: "Doctor Removed", variant: "outline" },
  medications_updated: { icon: Pill, label: "Medications Updated", variant: "secondary" },
  allergies_updated: { icon: AlertTriangle, label: "Allergies Updated", variant: "destructive" },
  emergency_contacts_updated: { icon: Activity, label: "Emergency Contacts", variant: "secondary" },
  chronic_conditions_updated: { icon: Activity, label: "Conditions Updated", variant: "secondary" },
  profile_updated: { icon: Activity, label: "Profile Updated", variant: "outline" },
};

export function AuditLogViewer() {
  const [page, setPage] = useState(1);
  const pageSize = 10;

  const { data, isLoading } = useQuery<AuditLogResponse>({
    queryKey: ["/api/patient/audit-log", page, pageSize],
  });

  const logs = data?.logs || [];
  const total = data?.total || 0;
  const totalPages = Math.ceil(total / pageSize);

  const getEventConfig = (eventType: string) => {
    return eventTypeConfig[eventType] || { 
      icon: Activity, 
      label: eventType.replace(/_/g, " "), 
      variant: "outline" as const 
    };
  };

  if (isLoading) {
    return (
      <Card data-testid="audit-log-loading">
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center gap-4">
                <div className="h-10 w-10 bg-muted rounded-full" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-muted rounded w-1/3" />
                  <div className="h-3 bg-muted rounded w-1/2" />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="audit-log-viewer">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="h-5 w-5 text-primary" />
          Activity History
        </CardTitle>
        <CardDescription>
          Your account activity and changes
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {logs.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No activity recorded yet</p>
          </div>
        ) : (
          <>
            <ScrollArea className="h-80">
              <div className="space-y-3">
                {logs.map((log) => {
                  const config = getEventConfig(log.event_type);
                  const Icon = config.icon;
                  
                  return (
                    <div
                      key={log.id}
                      className="flex items-start gap-3 p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors"
                      data-testid={`audit-log-entry-${log.id}`}
                    >
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-background border">
                        <Icon className="h-4 w-4 text-primary" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge variant={config.variant}>{config.label}</Badge>
                          <span className="text-xs text-muted-foreground">
                            {format(new Date(log.created_at), "MMM d, yyyy 'at' h:mm a")}
                          </span>
                        </div>
                        {log.event_data && Object.keys(log.event_data).length > 0 && (
                          <p className="text-sm text-muted-foreground mt-1 truncate">
                            {Object.entries(log.event_data)
                              .filter(([k]) => k !== "field" && k !== "updated")
                              .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`)
                              .join(", ")}
                          </p>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </ScrollArea>

            {totalPages > 1 && (
              <div className="flex items-center justify-between pt-4 border-t">
                <p className="text-sm text-muted-foreground">
                  Page {page} of {totalPages}
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                    data-testid="button-prev-page"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    disabled={page >= totalPages}
                    data-testid="button-next-page"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default AuditLogViewer;
