import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { 
  Mail, 
  MessageCircle, 
  Phone, 
  CheckCircle2, 
  XCircle, 
  Loader2, 
  RefreshCw, 
  ExternalLink,
  Settings,
  Unplug,
  Link2
} from "lucide-react";

interface IntegrationStatus {
  gmail: { connected: boolean; email?: string; lastSync?: string };
  whatsapp: { connected: boolean; number?: string; lastSync?: string };
  twilio: { connected: boolean; number?: string; lastSync?: string };
}

export function IntegrationSettings() {
  const { toast } = useToast();
  const [whatsappDialogOpen, setWhatsappDialogOpen] = useState(false);
  const [twilioDialogOpen, setTwilioDialogOpen] = useState(false);
  
  // WhatsApp form state
  const [whatsappForm, setWhatsappForm] = useState({
    businessId: "",
    phoneNumberId: "",
    displayNumber: "",
    accessToken: ""
  });
  
  // Twilio form state
  const [twilioForm, setTwilioForm] = useState({
    accountSid: "",
    apiKey: "",
    apiSecret: "",
    phoneNumber: ""
  });

  // Fetch integration status
  const { data: status, isLoading } = useQuery<IntegrationStatus>({
    queryKey: ["/api/v1/integrations/status"],
  });

  // Gmail OAuth URL mutation
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
        title: "Error",
        description: "Failed to start Gmail connection. Please try again.",
        variant: "destructive",
      });
    },
  });

  // Gmail sync mutation
  const syncGmail = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/gmail/sync", { method: "POST" });
      return response.json();
    },
    onSuccess: (data: { emailsFetched: number }) => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/status"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/gmail/emails"] });
      toast({
        title: "Sync Complete",
        description: `Successfully synced ${data.emailsFetched} new emails.`,
      });
    },
    onError: () => {
      toast({
        title: "Sync Failed",
        description: "Failed to sync emails. Please try again.",
        variant: "destructive",
      });
    },
  });

  // Gmail disconnect mutation
  const disconnectGmail = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/gmail/disconnect", { method: "POST" });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/status"] });
      toast({
        title: "Disconnected",
        description: "Gmail has been disconnected successfully.",
      });
    },
  });

  // WhatsApp configure mutation
  const configureWhatsApp = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/whatsapp/configure", {
        method: "POST",
        json: whatsappForm,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/status"] });
      setWhatsappDialogOpen(false);
      setWhatsappForm({ businessId: "", phoneNumberId: "", displayNumber: "", accessToken: "" });
      toast({
        title: "Connected",
        description: "WhatsApp Business has been connected successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Connection Failed",
        description: "Failed to connect WhatsApp Business. Please check your credentials.",
        variant: "destructive",
      });
    },
  });

  // Twilio configure mutation
  const configureTwilio = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/twilio/configure", {
        method: "POST",
        json: twilioForm,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/status"] });
      setTwilioDialogOpen(false);
      setTwilioForm({ accountSid: "", apiKey: "", apiSecret: "", phoneNumber: "" });
      toast({
        title: "Connected",
        description: "Twilio phone has been connected successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Connection Failed",
        description: "Failed to connect Twilio. Please check your credentials.",
        variant: "destructive",
      });
    },
  });

  // Twilio sync mutation
  const syncTwilio = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/integrations/twilio/sync", { method: "POST" });
      return response.json();
    },
    onSuccess: (data: { callsSynced: number }) => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/status"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/integrations/twilio/calls"] });
      toast({
        title: "Sync Complete",
        description: `Successfully synced ${data.callsSynced} new calls.`,
      });
    },
    onError: () => {
      toast({
        title: "Sync Failed",
        description: "Failed to sync call logs. Please try again.",
        variant: "destructive",
      });
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold" data-testid="text-integration-title">Account Integrations</h2>
          <p className="text-muted-foreground">
            Connect your personal email, WhatsApp, and phone to automatically sync communications
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {/* Gmail Integration */}
        <Card data-testid="card-integration-gmail">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-10 w-10 rounded-lg bg-red-500/10 flex items-center justify-center">
                  <Mail className="h-5 w-5 text-red-600" />
                </div>
                <div>
                  <CardTitle className="text-base">Gmail</CardTitle>
                  <CardDescription className="text-xs">Personal email sync</CardDescription>
                </div>
              </div>
              {status?.gmail.connected ? (
                <Badge variant="default" className="bg-green-500/10 text-green-600 hover:bg-green-500/20">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Connected
                </Badge>
              ) : (
                <Badge variant="secondary">
                  <XCircle className="h-3 w-3 mr-1" />
                  Not Connected
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {status?.gmail.connected ? (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Connected as: <span className="font-medium text-foreground">{status.gmail.email}</span>
                </p>
                {status.gmail.lastSync && (
                  <p className="text-xs text-muted-foreground">
                    Last sync: {new Date(status.gmail.lastSync).toLocaleString()}
                  </p>
                )}
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => syncGmail.mutate()}
                    disabled={syncGmail.isPending}
                    data-testid="button-sync-gmail"
                  >
                    {syncGmail.isPending ? (
                      <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4 mr-1" />
                    )}
                    Sync Now
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={() => disconnectGmail.mutate()}
                    disabled={disconnectGmail.isPending}
                    data-testid="button-disconnect-gmail"
                  >
                    <Unplug className="h-4 w-4 mr-1" />
                    Disconnect
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Connect your Gmail to automatically sync patient emails
                </p>
                <Button 
                  onClick={() => getGmailAuthUrl.mutate()}
                  disabled={getGmailAuthUrl.isPending}
                  className="w-full"
                  data-testid="button-connect-gmail"
                >
                  {getGmailAuthUrl.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Link2 className="h-4 w-4 mr-2" />
                  )}
                  Connect Gmail
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* WhatsApp Integration */}
        <Card data-testid="card-integration-whatsapp">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-10 w-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                  <MessageCircle className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <CardTitle className="text-base">WhatsApp Business</CardTitle>
                  <CardDescription className="text-xs">Message sync</CardDescription>
                </div>
              </div>
              {status?.whatsapp.connected ? (
                <Badge variant="default" className="bg-green-500/10 text-green-600 hover:bg-green-500/20">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Connected
                </Badge>
              ) : (
                <Badge variant="secondary">
                  <XCircle className="h-3 w-3 mr-1" />
                  Not Connected
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {status?.whatsapp.connected ? (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Connected: <span className="font-medium text-foreground">{status.whatsapp.number}</span>
                </p>
                {status.whatsapp.lastSync && (
                  <p className="text-xs text-muted-foreground">
                    Last sync: {new Date(status.whatsapp.lastSync).toLocaleString()}
                  </p>
                )}
                <Button 
                  variant="outline" 
                  size="sm"
                  data-testid="button-settings-whatsapp"
                >
                  <Settings className="h-4 w-4 mr-1" />
                  Settings
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Connect WhatsApp Business to sync patient messages
                </p>
                <Dialog open={whatsappDialogOpen} onOpenChange={setWhatsappDialogOpen}>
                  <DialogTrigger asChild>
                    <Button className="w-full" data-testid="button-connect-whatsapp">
                      <Link2 className="h-4 w-4 mr-2" />
                      Connect WhatsApp
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Connect WhatsApp Business</DialogTitle>
                      <DialogDescription>
                        Enter your WhatsApp Business API credentials from Meta Business Suite
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 pt-4">
                      <div className="space-y-2">
                        <Label htmlFor="businessId">Business ID</Label>
                        <Input
                          id="businessId"
                          placeholder="Your WhatsApp Business ID"
                          value={whatsappForm.businessId}
                          onChange={(e) => setWhatsappForm(prev => ({ ...prev, businessId: e.target.value }))}
                          data-testid="input-whatsapp-businessid"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="phoneNumberId">Phone Number ID</Label>
                        <Input
                          id="phoneNumberId"
                          placeholder="Your Phone Number ID"
                          value={whatsappForm.phoneNumberId}
                          onChange={(e) => setWhatsappForm(prev => ({ ...prev, phoneNumberId: e.target.value }))}
                          data-testid="input-whatsapp-phoneid"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="displayNumber">Display Number</Label>
                        <Input
                          id="displayNumber"
                          placeholder="+1234567890"
                          value={whatsappForm.displayNumber}
                          onChange={(e) => setWhatsappForm(prev => ({ ...prev, displayNumber: e.target.value }))}
                          data-testid="input-whatsapp-number"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="accessToken">Access Token</Label>
                        <Input
                          id="accessToken"
                          type="password"
                          placeholder="Your permanent access token"
                          value={whatsappForm.accessToken}
                          onChange={(e) => setWhatsappForm(prev => ({ ...prev, accessToken: e.target.value }))}
                          data-testid="input-whatsapp-token"
                        />
                      </div>
                      <div className="flex justify-between items-center pt-2">
                        <a 
                          href="https://developers.facebook.com/docs/whatsapp/cloud-api/get-started" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-sm text-primary hover:underline flex items-center gap-1"
                        >
                          <ExternalLink className="h-3 w-3" />
                          Setup Guide
                        </a>
                        <Button 
                          onClick={() => configureWhatsApp.mutate()}
                          disabled={configureWhatsApp.isPending || !whatsappForm.businessId || !whatsappForm.phoneNumberId}
                          data-testid="button-submit-whatsapp"
                        >
                          {configureWhatsApp.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                          Connect
                        </Button>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Twilio Phone Integration */}
        <Card data-testid="card-integration-twilio">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-10 w-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                  <Phone className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <CardTitle className="text-base">Phone (Twilio)</CardTitle>
                  <CardDescription className="text-xs">Call logging</CardDescription>
                </div>
              </div>
              {status?.twilio.connected ? (
                <Badge variant="default" className="bg-green-500/10 text-green-600 hover:bg-green-500/20">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Connected
                </Badge>
              ) : (
                <Badge variant="secondary">
                  <XCircle className="h-3 w-3 mr-1" />
                  Not Connected
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {status?.twilio.connected ? (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Connected: <span className="font-medium text-foreground">{status.twilio.number}</span>
                </p>
                {status.twilio.lastSync && (
                  <p className="text-xs text-muted-foreground">
                    Last sync: {new Date(status.twilio.lastSync).toLocaleString()}
                  </p>
                )}
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => syncTwilio.mutate()}
                    disabled={syncTwilio.isPending}
                    data-testid="button-sync-twilio"
                  >
                    {syncTwilio.isPending ? (
                      <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4 mr-1" />
                    )}
                    Sync Calls
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    data-testid="button-settings-twilio"
                  >
                    <Settings className="h-4 w-4 mr-1" />
                    Settings
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Connect a Twilio number to log calls automatically
                </p>
                <Dialog open={twilioDialogOpen} onOpenChange={setTwilioDialogOpen}>
                  <DialogTrigger asChild>
                    <Button className="w-full" data-testid="button-connect-twilio">
                      <Link2 className="h-4 w-4 mr-2" />
                      Connect Phone
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Connect Twilio Phone</DialogTitle>
                      <DialogDescription>
                        Enter your Twilio account credentials and phone number
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 pt-4">
                      <div className="space-y-2">
                        <Label htmlFor="accountSid">Account SID</Label>
                        <Input
                          id="accountSid"
                          placeholder="ACxxxxxx..."
                          value={twilioForm.accountSid}
                          onChange={(e) => setTwilioForm(prev => ({ ...prev, accountSid: e.target.value }))}
                          data-testid="input-twilio-sid"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="apiKey">API Key</Label>
                        <Input
                          id="apiKey"
                          placeholder="SKxxxxxx..."
                          value={twilioForm.apiKey}
                          onChange={(e) => setTwilioForm(prev => ({ ...prev, apiKey: e.target.value }))}
                          data-testid="input-twilio-apikey"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="apiSecret">API Secret</Label>
                        <Input
                          id="apiSecret"
                          type="password"
                          placeholder="Your API secret"
                          value={twilioForm.apiSecret}
                          onChange={(e) => setTwilioForm(prev => ({ ...prev, apiSecret: e.target.value }))}
                          data-testid="input-twilio-secret"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="phoneNumber">Phone Number</Label>
                        <Input
                          id="phoneNumber"
                          placeholder="+1234567890"
                          value={twilioForm.phoneNumber}
                          onChange={(e) => setTwilioForm(prev => ({ ...prev, phoneNumber: e.target.value }))}
                          data-testid="input-twilio-phone"
                        />
                      </div>
                      <div className="flex justify-between items-center pt-2">
                        <a 
                          href="https://www.twilio.com/console" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-sm text-primary hover:underline flex items-center gap-1"
                        >
                          <ExternalLink className="h-3 w-3" />
                          Twilio Console
                        </a>
                        <Button 
                          onClick={() => configureTwilio.mutate()}
                          disabled={configureTwilio.isPending || !twilioForm.accountSid || !twilioForm.apiKey}
                          data-testid="button-submit-twilio"
                        >
                          {configureTwilio.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                          Connect
                        </Button>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Integration Info */}
      <Card className="bg-muted/50">
        <CardContent className="pt-6">
          <div className="flex items-start gap-4">
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
              <Settings className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold mb-1">How Integrations Work</h3>
              <p className="text-sm text-muted-foreground">
                When connected, your communications are automatically synced and categorized by AI. 
                Patient emails are linked to their records, WhatsApp booking requests are extracted, 
                and call logs with transcriptions are saved for your review.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                <Badge variant="outline">AI Email Categorization</Badge>
                <Badge variant="outline">Auto Patient Linking</Badge>
                <Badge variant="outline">Call Transcription</Badge>
                <Badge variant="outline">HIPAA Compliant</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
