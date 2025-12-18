import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Video, Clock, TrendingUp, FileText, Loader2, AlertCircle, DollarSign } from "lucide-react";
import { Link } from "wouter";

interface UsageSummary {
  doctor_id: string;
  billing_month: string;
  total_participant_minutes: number;
  included_minutes: number;
  overage_minutes: number;
  overage_cost_usd: number;
  total_sessions: number;
  total_appointments: number;
  subscription_tier: string;
}

interface Invoice {
  id: number;
  billing_month: string;
  status: string;
  total_participant_minutes: number;
  included_minutes: number;
  overage_minutes: number;
  overage_cost_usd: number;
  generated_at: string;
}

export default function VideoBillingCard() {
  const { data: usage, isLoading: loadingUsage, error: usageError } = useQuery<UsageSummary>({
    queryKey: ["/api/video/usage"],
    retry: false
  });

  const { data: invoices, isLoading: loadingInvoices } = useQuery<Invoice[]>({
    queryKey: ["/api/video/invoices"],
    retry: false
  });

  if (loadingUsage) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-32">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (usageError) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Video className="h-5 w-5 text-primary" />
            <CardTitle>Video Usage & Billing</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-muted-foreground">
            <AlertCircle className="h-4 w-4" />
            <span className="text-sm">Unable to load billing data</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const usagePercent = usage ? Math.min(100, (usage.total_participant_minutes / usage.included_minutes) * 100) : 0;
  const isOverLimit = usage && usage.overage_minutes > 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Video className="h-5 w-5 text-primary" />
            <CardTitle>Video Usage & Billing</CardTitle>
          </div>
          <Badge variant={isOverLimit ? "destructive" : "secondary"}>
            {usage?.subscription_tier || "Standard"}
          </Badge>
        </div>
        <CardDescription>
          Current billing period: {usage?.billing_month || "N/A"}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Participant-Minutes Used</span>
            <span className="font-medium">
              {usage?.total_participant_minutes || 0} / {usage?.included_minutes || 0}
            </span>
          </div>
          <Progress value={usagePercent} className={isOverLimit ? "[&>div]:bg-destructive" : ""} />
          <p className="text-xs text-muted-foreground">
            {isOverLimit 
              ? `${usage?.overage_minutes} minutes over limit` 
              : `${(usage?.included_minutes || 0) - (usage?.total_participant_minutes || 0)} minutes remaining`
            }
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-1 text-muted-foreground text-xs">
              <Clock className="h-3 w-3" />
              Total Sessions
            </div>
            <p className="text-2xl font-semibold">{usage?.total_sessions || 0}</p>
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-1 text-muted-foreground text-xs">
              <TrendingUp className="h-3 w-3" />
              Appointments
            </div>
            <p className="text-2xl font-semibold">{usage?.total_appointments || 0}</p>
          </div>
        </div>

        {isOverLimit && (
          <>
            <Separator />
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <DollarSign className="h-4 w-4 text-destructive" />
                  <span className="text-sm font-medium">Overage Charges</span>
                </div>
                <span className="text-lg font-semibold text-destructive">
                  ${usage?.overage_cost_usd?.toFixed(2) || "0.00"}
                </span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {usage?.overage_minutes} minutes at $0.0045/min
              </p>
            </div>
          </>
        )}

        <Separator />

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Recent Invoices</span>
            <Link href="/profile?tab=video">
              <Button variant="ghost" size="sm" className="gap-1" data-testid="link-view-all-invoices">
                <FileText className="h-3 w-3" />
                View All
              </Button>
            </Link>
          </div>
          {loadingInvoices ? (
            <div className="flex items-center justify-center h-12">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          ) : invoices && invoices.length > 0 ? (
            <div className="space-y-2">
              {invoices.slice(0, 3).map((invoice) => (
                <div
                  key={invoice.id}
                  className="flex items-center justify-between p-2 rounded border text-sm"
                  data-testid={`invoice-row-${invoice.id}`}
                >
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <span>{invoice.billing_month}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={invoice.status === "paid" ? "secondary" : "outline"}
                      className="text-xs"
                    >
                      {invoice.status}
                    </Badge>
                    {invoice.overage_cost_usd > 0 && (
                      <span className="text-xs text-muted-foreground">
                        ${invoice.overage_cost_usd.toFixed(2)}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-2">
              No invoices yet
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
