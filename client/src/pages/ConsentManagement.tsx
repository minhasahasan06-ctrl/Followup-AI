import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { insertHealthInsightConsentSchema, type HealthInsightConsent } from "@shared/schema";
import { Plus, Shield, ShieldAlert, AlertCircle, CheckCircle2, Pause, X } from "lucide-react";
import { format } from "date-fns";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";

const consentFormSchema = insertHealthInsightConsentSchema.extend({
  dataTypes: z.array(z.string()).min(1, "Select at least one data type"),
});

type ConsentFormValues = z.infer<typeof consentFormSchema>;

const popularApps = [
  { name: "Fitbit", category: "fitness", icon: "💪" },
  { name: "Apple Health", category: "fitness", icon: "🍎" },
  { name: "Google Fit", category: "fitness", icon: "🏃" },
  { name: "MyFitnessPal", category: "nutrition", icon: "🥗" },
  { name: "Headspace", category: "mental_health", icon: "🧘" },
  { name: "Sleep Cycle", category: "sleep", icon: "😴" },
  { name: "Medisafe", category: "medication", icon: "💊" },
];

const dataTypeOptions = [
  "heart_rate",
  "blood_pressure",
  "steps",
  "sleep",
  "weight",
  "blood_glucose",
  "oxygen_saturation",
  "temperature",
  "medications",
  "activity",
];

export default function ConsentManagement() {
  const { toast } = useToast();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [revokeDialogOpen, setRevokeDialogOpen] = useState(false);
  const [selectedConsent, setSelectedConsent] = useState<HealthInsightConsent | null>(null);
  const [revokeReason, setRevokeReason] = useState("");

  const { data: consents = [], isLoading } = useQuery<HealthInsightConsent[]>({
    queryKey: ["/api/consents"],
  });

  const form = useForm<ConsentFormValues>({
    resolver: zodResolver(consentFormSchema),
    defaultValues: {
      appName: "",
      appCategory: "fitness",
      consentGranted: true,
      dataTypes: [],
      purpose: "",
      sharingFrequency: "real-time",
      syncStatus: "active",
    },
  });

  const createMutation = useMutation({
    mutationFn: (data: ConsentFormValues) => apiRequest("POST", "/api/consents", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/consents"] });
      toast({
        title: "Consent granted",
        description: "Health data sharing permission has been granted.",
      });
      setIsDialogOpen(false);
      form.reset();
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to grant consent. Please try again.",
        variant: "destructive",
      });
    },
  });

  const revokeMutation = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      apiRequest("POST", `/api/consents/${id}/revoke`, { reason }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/consents"] });
      toast({
        title: "Consent revoked",
        description: "Health data sharing permission has been revoked.",
      });
      setRevokeDialogOpen(false);
      setSelectedConsent(null);
      setRevokeReason("");
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to revoke consent. Please try again.",
        variant: "destructive",
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiRequest("DELETE", `/api/consents/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/consents"] });
      toast({
        title: "Consent deleted",
        description: "Health data sharing record has been deleted.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to delete consent. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleSubmit = (data: ConsentFormValues) => {
    createMutation.mutate(data);
  };

  const handleRevoke = () => {
    if (selectedConsent) {
      revokeMutation.mutate({ id: selectedConsent.id, reason: revokeReason });
    }
  };

  const getStatusIcon = (consent: HealthInsightConsent) => {
    if (consent.syncStatus === "active") return <CheckCircle2 className="h-4 w-4 text-green-600" />;
    if (consent.syncStatus === "paused") return <Pause className="h-4 w-4 text-yellow-600" />;
    if (consent.syncStatus === "error") return <AlertCircle className="h-4 w-4 text-red-600" />;
    return <ShieldAlert className="h-4 w-4 text-gray-400" />;
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      active: "default",
      paused: "secondary",
      revoked: "destructive",
      error: "destructive",
    };
    return <Badge variant={variants[status] || "outline"}>{status}</Badge>;
  };

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader className="h-24 bg-muted" />
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">App Connections</h1>
          <p className="text-muted-foreground mt-2">
            Manage health data sharing with third-party apps
          </p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-add-consent">
              <Plus className="h-4 w-4 mr-2" />
              Connect App
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Connect a Health App</DialogTitle>
              <DialogDescription>
                Grant permission for a third-party app to access your health data
              </DialogDescription>
            </DialogHeader>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
                <FormField
                  control={form.control}
                  name="appName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>App Name</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger data-testid="select-app-name">
                            <SelectValue placeholder="Select an app" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {popularApps.map((app) => (
                            <SelectItem key={app.name} value={app.name}>
                              {app.icon} {app.name}
                            </SelectItem>
                          ))}
                          <SelectItem value="other">Other App</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="appCategory"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Category</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger data-testid="select-category">
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="fitness">Fitness</SelectItem>
                          <SelectItem value="nutrition">Nutrition</SelectItem>
                          <SelectItem value="mental_health">Mental Health</SelectItem>
                          <SelectItem value="sleep">Sleep</SelectItem>
                          <SelectItem value="medication">Medication</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="dataTypes"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Data Types to Share</FormLabel>
                      <FormDescription>
                        Select the health data types this app can access
                      </FormDescription>
                      <div className="grid grid-cols-2 gap-2 mt-2">
                        {dataTypeOptions.map((type) => (
                          <div key={type} className="flex items-center space-x-2">
                            <Switch
                              checked={field.value?.includes(type)}
                              onCheckedChange={(checked) => {
                                const current = field.value || [];
                                if (checked) {
                                  field.onChange([...current, type]);
                                } else {
                                  field.onChange(current.filter((t) => t !== type));
                                }
                              }}
                              data-testid={`switch-${type}`}
                            />
                            <label className="text-sm capitalize">
                              {type.replace(/_/g, " ")}
                            </label>
                          </div>
                        ))}
                      </div>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="purpose"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Purpose</FormLabel>
                      <FormControl>
                        <Textarea
                          {...field}
                          value={field.value || ""}
                          placeholder="Why does this app need access to your health data?"
                          data-testid="input-purpose"
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="sharingFrequency"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Sharing Frequency</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger data-testid="select-frequency">
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="real-time">Real-time</SelectItem>
                          <SelectItem value="daily">Daily</SelectItem>
                          <SelectItem value="weekly">Weekly</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <DialogFooter>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setIsDialogOpen(false)}
                    data-testid="button-cancel-consent"
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    disabled={createMutation.isPending}
                    data-testid="button-submit-consent"
                  >
                    {createMutation.isPending ? "Granting..." : "Grant Permission"}
                  </Button>
                </DialogFooter>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
      </div>

      {consents.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center py-12">
              <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No app connections yet</h3>
              <p className="text-muted-foreground mb-4">
                Connect health and fitness apps to sync your data with Followup AI
              </p>
              <Button onClick={() => setIsDialogOpen(true)} data-testid="button-add-first-consent">
                <Plus className="h-4 w-4 mr-2" />
                Connect Your First App
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {consents.map((consent) => (
            <Card key={consent.id} data-testid={`card-consent-${consent.id}`}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(consent)}
                    <div>
                      <CardTitle className="text-lg" data-testid={`text-app-name-${consent.id}`}>
                        {consent.appName}
                      </CardTitle>
                      <CardDescription>
                        {consent.appCategory && (
                          <Badge variant="outline" className="mr-2 capitalize">
                            {consent.appCategory}
                          </Badge>
                        )}
                        {getStatusBadge(consent.syncStatus || "active")}
                      </CardDescription>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    {consent.syncStatus === "active" && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setSelectedConsent(consent);
                          setRevokeDialogOpen(true);
                        }}
                        data-testid={`button-revoke-${consent.id}`}
                      >
                        Revoke
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => deleteMutation.mutate(consent.id)}
                      data-testid={`button-delete-${consent.id}`}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {consent.purpose && (
                  <div>
                    <p className="text-sm font-medium mb-1">Purpose</p>
                    <p className="text-sm text-muted-foreground">{consent.purpose}</p>
                  </div>
                )}
                {consent.dataTypes && consent.dataTypes.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Data Types Shared</p>
                    <div className="flex flex-wrap gap-2">
                      {consent.dataTypes.map((type) => (
                        <Badge key={type} variant="secondary" className="capitalize">
                          {type.replace(/_/g, " ")}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Sharing Frequency</p>
                    <p className="font-medium capitalize">{consent.sharingFrequency ? consent.sharingFrequency : "Real-time"}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Connected</p>
                    <p className="font-medium">
                      {format(new Date(consent.createdAt), "MMM d, yyyy")}
                    </p>
                  </div>
                  {consent.lastSyncedAt && (
                    <div>
                      <p className="text-muted-foreground">Last Synced</p>
                      <p className="font-medium">
                        {format(new Date(consent.lastSyncedAt), "MMM d, yyyy h:mm a")}
                      </p>
                    </div>
                  )}
                  {consent.revokedAt !== null && (
                    <div>
                      <p className="text-muted-foreground">Revoked</p>
                      <p className="font-medium">
                        {format(new Date(consent.revokedAt!), "MMM d, yyyy")}
                      </p>
                    </div>
                  )}
                </div>
                {consent.revokedReason && (
                  <div>
                    <p className="text-sm font-medium mb-1">Revocation Reason</p>
                    <p className="text-sm text-muted-foreground">{consent.revokedReason}</p>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Dialog open={revokeDialogOpen} onOpenChange={setRevokeDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Revoke App Permission</DialogTitle>
            <DialogDescription>
              This will stop {selectedConsent?.appName} from accessing your health data.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">Reason (optional)</label>
              <Textarea
                value={revokeReason}
                onChange={(e) => setRevokeReason(e.target.value)}
                placeholder="Why are you revoking this permission?"
                className="mt-2"
                data-testid="input-revoke-reason"
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setRevokeDialogOpen(false);
                setSelectedConsent(null);
                setRevokeReason("");
              }}
              data-testid="button-cancel-revoke"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleRevoke}
              disabled={revokeMutation.isPending}
              data-testid="button-confirm-revoke"
            >
              {revokeMutation.isPending ? "Revoking..." : "Revoke Permission"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
