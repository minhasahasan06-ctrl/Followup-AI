import { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { 
  Watch, CheckCircle2, XCircle, AlertCircle, Plus, Trash2, RefreshCw, 
  Battery, Activity, Dumbbell, Zap, Smartphone, Stethoscope, Heart,
  Thermometer, Droplets, Scale, Bluetooth, Wifi, Shield, Settings,
  ArrowRight, ExternalLink, Clock, TrendingUp, AlertTriangle, Info,
  ChevronRight, Check, X, Loader2, Link2, Unlink, BarChart2, FileText
} from "lucide-react";
import { SiApple, SiFitbit, SiGarmin, SiSamsung, SiGoogle } from "react-icons/si";

interface DeviceType {
  id: string;
  name: string;
  category: string;
  icon: string;
  description: string;
  metrics: string[];
  pairingMethods: string[];
}

interface ConnectedDevice {
  id: string;
  deviceType: string;
  deviceName: string;
  vendorId: string;
  connectionStatus: "connected" | "disconnected" | "syncing" | "error" | "pending";
  batteryLevel?: number;
  lastSyncAt?: string;
  firmwareVersion?: string;
  trackedMetrics: string[];
  consentGiven: boolean;
  consentTimestamp?: string;
  healthData?: {
    latestReadings: Record<string, any>;
    dailyAverage: Record<string, number>;
    trends: Record<string, string>;
  };
}

interface Vendor {
  id: string;
  name: string;
  status: "active" | "coming_soon" | "requires_partnership";
  pairingMethod: "oauth" | "ble" | "manual" | "qr";
  deviceTypes: string[];
  logoComponent?: React.ComponentType<any>;
}

const DEVICE_CATEGORIES = [
  { id: "smartwatch", name: "Smartwatch", icon: Watch, description: "Apple Watch, Fitbit, Garmin, Oura" },
  { id: "bp_monitor", name: "Blood Pressure Monitor", icon: Heart, description: "Withings, Omron, iHealth" },
  { id: "glucose_meter", name: "Glucose Meter", icon: Droplets, description: "Dexcom, Abbott Libre, iHealth" },
  { id: "scale", name: "Smart Scale", icon: Scale, description: "Withings, Fitbit Aria, Renpho" },
  { id: "thermometer", name: "Smart Thermometer", icon: Thermometer, description: "Withings Thermo, Kinsa" },
  { id: "stethoscope", name: "Digital Stethoscope", icon: Stethoscope, description: "Eko, Littmann Core" },
  { id: "pulse_oximeter", name: "Pulse Oximeter", icon: Activity, description: "Masimo, iHealth, Wellue" },
  { id: "activity_tracker", name: "Activity Tracker", icon: Dumbbell, description: "Fitbit, Whoop, Oura Ring" },
];

const VENDORS: Vendor[] = [
  { id: "apple_health", name: "Apple Health", status: "active", pairingMethod: "manual", deviceTypes: ["smartwatch", "all"], logoComponent: SiApple },
  { id: "fitbit", name: "Fitbit", status: "active", pairingMethod: "oauth", deviceTypes: ["smartwatch", "scale", "activity_tracker"], logoComponent: SiFitbit },
  { id: "withings", name: "Withings", status: "active", pairingMethod: "oauth", deviceTypes: ["bp_monitor", "scale", "thermometer", "smartwatch"] },
  { id: "oura", name: "Oura", status: "active", pairingMethod: "oauth", deviceTypes: ["activity_tracker"] },
  { id: "google_fit", name: "Google Fit", status: "active", pairingMethod: "oauth", deviceTypes: ["smartwatch", "all"], logoComponent: SiGoogle },
  { id: "ihealth", name: "iHealth", status: "active", pairingMethod: "oauth", deviceTypes: ["bp_monitor", "glucose_meter", "pulse_oximeter", "scale"] },
  { id: "garmin", name: "Garmin", status: "requires_partnership", pairingMethod: "oauth", deviceTypes: ["smartwatch", "activity_tracker"], logoComponent: SiGarmin },
  { id: "whoop", name: "Whoop", status: "requires_partnership", pairingMethod: "oauth", deviceTypes: ["activity_tracker"] },
  { id: "dexcom", name: "Dexcom", status: "requires_partnership", pairingMethod: "oauth", deviceTypes: ["glucose_meter"] },
  { id: "samsung", name: "Samsung Health", status: "requires_partnership", pairingMethod: "oauth", deviceTypes: ["smartwatch"], logoComponent: SiSamsung },
  { id: "eko", name: "Eko", status: "requires_partnership", pairingMethod: "ble", deviceTypes: ["stethoscope"] },
  { id: "abbott", name: "Abbott LibreView", status: "requires_partnership", pairingMethod: "oauth", deviceTypes: ["glucose_meter"] },
];

const METRIC_LABELS: Record<string, string> = {
  heart_rate: "Heart Rate",
  resting_heart_rate: "Resting HR",
  hrv: "HRV",
  steps: "Steps",
  calories: "Calories",
  sleep: "Sleep",
  spo2: "SpO2",
  bp: "Blood Pressure",
  glucose: "Glucose",
  weight: "Weight",
  temperature: "Temperature",
  respiratory_rate: "Respiratory Rate",
  stress: "Stress",
  recovery: "Recovery",
  readiness: "Readiness",
  vo2_max: "VO2 Max",
  active_minutes: "Active Minutes",
  body_battery: "Body Battery",
};

const DATA_CONSENT_TYPES = [
  { id: "heart_rate", label: "Heart Rate & HRV", description: "Continuous heart rate monitoring and heart rate variability" },
  { id: "blood_pressure", label: "Blood Pressure", description: "Systolic, diastolic, and pulse readings" },
  { id: "glucose", label: "Blood Glucose", description: "Fasting and postprandial glucose levels" },
  { id: "sleep", label: "Sleep Data", description: "Sleep duration, stages, quality, and breathing" },
  { id: "activity", label: "Activity & Steps", description: "Daily steps, calories, and active minutes" },
  { id: "spo2", label: "Blood Oxygen (SpO2)", description: "Oxygen saturation levels" },
  { id: "temperature", label: "Temperature", description: "Skin and body temperature readings" },
  { id: "weight", label: "Weight & Body Composition", description: "Weight, BMI, body fat, muscle mass" },
  { id: "respiratory", label: "Respiratory Rate", description: "Breathing rate and patterns" },
  { id: "stress", label: "Stress & Recovery", description: "Stress levels and recovery scores" },
];

export default function DeviceConnect() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("devices");
  const [showPairingWizard, setShowPairingWizard] = useState(false);
  const [showConsentModal, setShowConsentModal] = useState(false);
  const [showDeviceDetails, setShowDeviceDetails] = useState(false);
  const [selectedDevice, setSelectedDevice] = useState<ConnectedDevice | null>(null);
  const [pairingStep, setPairingStep] = useState(0);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedVendor, setSelectedVendor] = useState<string | null>(null);
  const [consentSelections, setConsentSelections] = useState<Record<string, boolean>>({});
  const [pairingStatus, setPairingStatus] = useState<"idle" | "connecting" | "success" | "error">("idle");
  const [manualDeviceName, setManualDeviceName] = useState("");

  const { data: devices = [], isLoading: devicesLoading, error: devicesError } = useQuery<ConnectedDevice[]>({
    queryKey: ["/api/v1/devices/connections"],
    retry: false,
  });

  const isAuthenticated = !devicesError || !(devicesError as Error)?.message?.includes("401");

  const { data: healthAnalytics } = useQuery({
    queryKey: ["/api/v1/devices/health-analytics"],
    enabled: devices.length > 0 && isAuthenticated,
    retry: false,
  });

  const { data: syncHistory } = useQuery({
    queryKey: ["/api/v1/devices/sync-history"],
    enabled: devices.length > 0 && isAuthenticated,
    retry: false,
  });

  const startPairingMutation = useMutation({
    mutationFn: async (data: { vendorId: string; deviceType: string; deviceName?: string }) => {
      return await apiRequest("/api/v1/devices/pair/start", {
        method: "POST",
        body: JSON.stringify(data),
      });
    },
    onSuccess: (response: any) => {
      if (response.pairing_method === "oauth" && response.oauth_url) {
        window.open(response.oauth_url, "_blank", "width=600,height=700");
        setPairingStatus("connecting");
      } else if (response.pairing_method === "manual") {
        setPairingStatus("success");
        toast({
          title: "Device Added",
          description: "Your device has been added. Data will sync automatically.",
        });
        queryClient.invalidateQueries({ queryKey: ["/api/v1/devices/connections"] });
        closePairingWizard();
      }
    },
    onError: () => {
      setPairingStatus("error");
      toast({
        title: "Pairing Failed",
        description: "Could not connect to the device. Please try again.",
        variant: "destructive",
      });
    },
  });

  const syncDeviceMutation = useMutation({
    mutationFn: async (deviceId: string) => {
      return await apiRequest(`/api/v1/devices/${deviceId}/sync`, {
        method: "POST",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/devices/connections"] });
      toast({
        title: "Sync Complete",
        description: "Your health data has been synchronized.",
      });
    },
    onError: () => {
      toast({
        title: "Sync Failed",
        description: "Could not sync device data. Please try again.",
        variant: "destructive",
      });
    },
  });

  const disconnectDeviceMutation = useMutation({
    mutationFn: async (deviceId: string) => {
      return await apiRequest(`/api/v1/devices/${deviceId}`, {
        method: "DELETE",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/devices/connections"] });
      setShowDeviceDetails(false);
      setSelectedDevice(null);
      toast({
        title: "Device Disconnected",
        description: "The device has been removed from your account.",
      });
    },
  });

  const updateConsentMutation = useMutation({
    mutationFn: async (data: { deviceId: string; consents: Record<string, boolean> }) => {
      return await apiRequest(`/api/v1/devices/${data.deviceId}/consent`, {
        method: "PATCH",
        body: JSON.stringify({ consent_types: data.consents }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/devices/connections"] });
      setShowConsentModal(false);
      toast({
        title: "Consent Updated",
        description: "Your data sharing preferences have been saved.",
      });
    },
  });

  const closePairingWizard = () => {
    setShowPairingWizard(false);
    setPairingStep(0);
    setSelectedCategory(null);
    setSelectedVendor(null);
    setConsentSelections({});
    setPairingStatus("idle");
    setManualDeviceName("");
  };

  const handleStartPairing = () => {
    if (!selectedVendor || !selectedCategory) return;
    
    const consentsArray = Object.entries(consentSelections)
      .filter(([_, enabled]) => enabled)
      .map(([id]) => id);
    
    startPairingMutation.mutate({
      vendorId: selectedVendor,
      deviceType: selectedCategory,
      deviceName: manualDeviceName || undefined,
    });
  };

  const getDeviceIcon = (deviceType: string) => {
    const category = DEVICE_CATEGORIES.find(c => c.id === deviceType);
    if (category) {
      const Icon = category.icon;
      return <Icon className="h-6 w-6" />;
    }
    return <Watch className="h-6 w-6" />;
  };

  const getVendorLogo = (vendorId: string) => {
    const vendor = VENDORS.find(v => v.id === vendorId);
    if (vendor?.logoComponent) {
      const Logo = vendor.logoComponent;
      return <Logo className="h-5 w-5" />;
    }
    return null;
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "connected":
        return <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">Connected</Badge>;
      case "syncing":
        return <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400">Syncing</Badge>;
      case "error":
        return <Badge variant="destructive">Error</Badge>;
      case "pending":
        return <Badge variant="secondary">Pending</Badge>;
      default:
        return <Badge variant="outline">Disconnected</Badge>;
    }
  };

  const getBatteryIcon = (level?: number) => {
    if (!level) return null;
    const color = level > 50 ? "text-green-600" : level > 20 ? "text-yellow-600" : "text-red-600";
    return (
      <div className="flex items-center gap-1">
        <Battery className={`h-4 w-4 ${color}`} />
        <span className="text-xs text-muted-foreground">{level}%</span>
      </div>
    );
  };

  const formatLastSync = (timestamp?: string) => {
    if (!timestamp) return "Never";
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return `${Math.floor(diffMins / 1440)}d ago`;
  };

  const connectedDevices = devices.filter((d: ConnectedDevice) => d.connectionStatus === "connected");
  const pendingDevices = devices.filter((d: ConnectedDevice) => d.connectionStatus === "pending");

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold" data-testid="text-page-title">Device Connect</h1>
          <p className="text-muted-foreground">
            Connect your health devices to enable continuous monitoring and AI-powered insights
          </p>
        </div>
        <Button onClick={() => setShowPairingWizard(true)} data-testid="button-add-device">
          <Plus className="mr-2 h-4 w-4" />
          Add Device
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-full bg-green-100 dark:bg-green-900/30">
                <Link2 className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="text-connected-count">{connectedDevices.length}</p>
                <p className="text-sm text-muted-foreground">Connected Devices</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-full bg-blue-100 dark:bg-blue-900/30">
                <BarChart2 className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="text-metrics-count">
                  {connectedDevices.reduce((acc, d) => acc + (d.trackedMetrics?.length || 0), 0)}
                </p>
                <p className="text-sm text-muted-foreground">Active Metrics</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-full bg-purple-100 dark:bg-purple-900/30">
                <Clock className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="text-last-sync">
                  {connectedDevices.length > 0 
                    ? formatLastSync(connectedDevices[0]?.lastSyncAt)
                    : "--"
                  }
                </p>
                <p className="text-sm text-muted-foreground">Last Sync</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-full bg-amber-100 dark:bg-amber-900/30">
                <Shield className="h-5 w-5 text-amber-600" />
              </div>
              <div>
                <p className="text-2xl font-bold" data-testid="text-consent-status">
                  {connectedDevices.filter(d => d.consentGiven).length}/{connectedDevices.length}
                </p>
                <p className="text-sm text-muted-foreground">Consents Active</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="devices" data-testid="tab-devices">My Devices</TabsTrigger>
          <TabsTrigger value="available" data-testid="tab-available">Available Devices</TabsTrigger>
          <TabsTrigger value="history" data-testid="tab-history">Sync History</TabsTrigger>
        </TabsList>

        <TabsContent value="devices" className="mt-6">
          {!isAuthenticated ? (
            <Card>
              <CardContent className="py-12 text-center">
                <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">Authentication Required</h3>
                <p className="text-muted-foreground mb-4">
                  Please sign in to view and manage your connected health devices
                </p>
                <Button asChild data-testid="button-login">
                  <a href="/login">Sign In</a>
                </Button>
              </CardContent>
            </Card>
          ) : devicesLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : devices.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <Watch className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No devices connected</h3>
                <p className="text-muted-foreground mb-4">
                  Connect your health devices to start tracking your vitals and receive AI-powered insights
                </p>
                <Button onClick={() => setShowPairingWizard(true)} data-testid="button-add-first-device">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Your First Device
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {devices.map((device: ConnectedDevice) => (
                <Card 
                  key={device.id} 
                  className="hover-elevate cursor-pointer"
                  onClick={() => {
                    setSelectedDevice(device);
                    setShowDeviceDetails(true);
                  }}
                  data-testid={`card-device-${device.id}`}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-muted">
                          {getDeviceIcon(device.deviceType)}
                        </div>
                        <div>
                          <CardTitle className="text-base">{device.deviceName}</CardTitle>
                          <CardDescription className="flex items-center gap-2">
                            {getVendorLogo(device.vendorId)}
                            <span>{VENDORS.find(v => v.id === device.vendorId)?.name || device.vendorId}</span>
                          </CardDescription>
                        </div>
                      </div>
                      {getStatusBadge(device.connectionStatus)}
                    </div>
                  </CardHeader>
                  <CardContent className="pb-3">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Clock className="h-4 w-4" />
                        <span>Last sync: {formatLastSync(device.lastSyncAt)}</span>
                      </div>
                      {getBatteryIcon(device.batteryLevel)}
                    </div>
                    
                    <div className="mt-3 flex flex-wrap gap-1">
                      {device.trackedMetrics?.slice(0, 4).map(metric => (
                        <Badge key={metric} variant="outline" className="text-xs">
                          {METRIC_LABELS[metric] || metric}
                        </Badge>
                      ))}
                      {(device.trackedMetrics?.length || 0) > 4 && (
                        <Badge variant="outline" className="text-xs">
                          +{device.trackedMetrics.length - 4} more
                        </Badge>
                      )}
                    </div>
                  </CardContent>
                  <CardFooter className="pt-3 border-t gap-2">
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation();
                        syncDeviceMutation.mutate(device.id);
                      }}
                      disabled={syncDeviceMutation.isPending}
                      data-testid={`button-sync-${device.id}`}
                    >
                      {syncDeviceMutation.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <RefreshCw className="mr-2 h-4 w-4" />
                      )}
                      Sync
                    </Button>
                    <Button 
                      size="sm" 
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedDevice(device);
                        setConsentSelections({});
                        setShowConsentModal(true);
                      }}
                      data-testid={`button-consent-${device.id}`}
                    >
                      <Shield className="mr-2 h-4 w-4" />
                      Consent
                    </Button>
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="available" className="mt-6">
          <div className="grid gap-6">
            {DEVICE_CATEGORIES.map(category => (
              <Card key={category.id}>
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-muted">
                      <category.icon className="h-5 w-5" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{category.name}</CardTitle>
                      <CardDescription>{category.description}</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {VENDORS.filter(v => v.deviceTypes.includes(category.id) || v.deviceTypes.includes("all"))
                      .map(vendor => (
                        <Badge 
                          key={vendor.id}
                          variant={vendor.status === "active" ? "default" : "secondary"}
                          className="cursor-pointer hover-elevate py-1.5 px-3"
                          onClick={() => {
                            if (vendor.status === "active") {
                              setSelectedCategory(category.id);
                              setSelectedVendor(vendor.id);
                              setPairingStep(2);
                              setShowPairingWizard(true);
                            } else {
                              toast({
                                title: "Coming Soon",
                                description: `${vendor.name} integration requires a partnership agreement.`,
                              });
                            }
                          }}
                          data-testid={`badge-vendor-${vendor.id}`}
                        >
                          {vendor.logoComponent && <vendor.logoComponent className="h-3.5 w-3.5 mr-1.5" />}
                          {vendor.name}
                          {vendor.status !== "active" && (
                            <span className="ml-1 text-xs opacity-60">
                              {vendor.status === "coming_soon" ? "(Soon)" : "(Partner)"}
                            </span>
                          )}
                        </Badge>
                      ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="history" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Sync History</CardTitle>
              <CardDescription>Recent data synchronization events</CardDescription>
            </CardHeader>
            <CardContent>
              {!syncHistory || (syncHistory as any[])?.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No sync history available</p>
                </div>
              ) : (
                <ScrollArea className="h-[400px]">
                  <div className="space-y-3">
                    {(syncHistory as any[])?.map((event: any, i: number) => (
                      <div key={i} className="flex items-center gap-4 p-3 rounded-lg bg-muted/50">
                        <div className={`p-2 rounded-full ${
                          event.status === "success" ? "bg-green-100 dark:bg-green-900/30" :
                          event.status === "error" ? "bg-red-100 dark:bg-red-900/30" :
                          "bg-blue-100 dark:bg-blue-900/30"
                        }`}>
                          {event.status === "success" ? (
                            <Check className="h-4 w-4 text-green-600" />
                          ) : event.status === "error" ? (
                            <X className="h-4 w-4 text-red-600" />
                          ) : (
                            <RefreshCw className="h-4 w-4 text-blue-600" />
                          )}
                        </div>
                        <div className="flex-1">
                          <p className="font-medium">{event.deviceName}</p>
                          <p className="text-sm text-muted-foreground">
                            {event.recordsProcessed} records synced
                          </p>
                        </div>
                        <div className="text-right text-sm text-muted-foreground">
                          <p>{new Date(event.timestamp).toLocaleDateString()}</p>
                          <p>{new Date(event.timestamp).toLocaleTimeString()}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Dialog open={showPairingWizard} onOpenChange={(open) => !open && closePairingWizard()}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Connect a Device</DialogTitle>
            <DialogDescription>
              {pairingStep === 0 && "Select the type of device you want to connect"}
              {pairingStep === 1 && "Choose your device manufacturer"}
              {pairingStep === 2 && "Configure data sharing preferences"}
              {pairingStep === 3 && "Complete the connection"}
            </DialogDescription>
          </DialogHeader>

          <div className="flex items-center justify-between py-4">
            {[0, 1, 2, 3].map(step => (
              <div key={step} className="flex items-center">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  pairingStep > step ? "bg-primary text-primary-foreground" :
                  pairingStep === step ? "bg-primary text-primary-foreground" :
                  "bg-muted text-muted-foreground"
                }`}>
                  {pairingStep > step ? <Check className="h-4 w-4" /> : step + 1}
                </div>
                {step < 3 && (
                  <div className={`w-16 h-0.5 mx-2 ${
                    pairingStep > step ? "bg-primary" : "bg-muted"
                  }`} />
                )}
              </div>
            ))}
          </div>

          {pairingStep === 0 && (
            <div className="grid gap-3 md:grid-cols-2 max-h-[400px] overflow-y-auto">
              {DEVICE_CATEGORIES.map(category => (
                <Card 
                  key={category.id}
                  className={`cursor-pointer hover-elevate ${
                    selectedCategory === category.id ? "ring-2 ring-primary" : ""
                  }`}
                  onClick={() => setSelectedCategory(category.id)}
                  data-testid={`card-category-${category.id}`}
                >
                  <CardContent className="pt-4 pb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-muted">
                        <category.icon className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="font-medium">{category.name}</p>
                        <p className="text-xs text-muted-foreground">{category.description}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {pairingStep === 1 && (
            <div className="grid gap-3 md:grid-cols-2 max-h-[400px] overflow-y-auto">
              {VENDORS.filter(v => 
                v.deviceTypes.includes(selectedCategory!) || v.deviceTypes.includes("all")
              ).map(vendor => (
                <Card 
                  key={vendor.id}
                  className={`cursor-pointer hover-elevate ${
                    selectedVendor === vendor.id ? "ring-2 ring-primary" : ""
                  } ${vendor.status !== "active" ? "opacity-60" : ""}`}
                  onClick={() => vendor.status === "active" && setSelectedVendor(vendor.id)}
                  data-testid={`card-vendor-${vendor.id}`}
                >
                  <CardContent className="pt-4 pb-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-muted">
                          {vendor.logoComponent ? (
                            <vendor.logoComponent className="h-5 w-5" />
                          ) : (
                            <Link2 className="h-5 w-5" />
                          )}
                        </div>
                        <div>
                          <p className="font-medium">{vendor.name}</p>
                          <p className="text-xs text-muted-foreground capitalize">
                            {vendor.pairingMethod === "oauth" ? "OAuth Connection" : 
                             vendor.pairingMethod === "ble" ? "Bluetooth" : "Manual Entry"}
                          </p>
                        </div>
                      </div>
                      {vendor.status !== "active" && (
                        <Badge variant="secondary" className="text-xs">
                          {vendor.status === "coming_soon" ? "Soon" : "Partner Only"}
                        </Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {pairingStep === 2 && (
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                <div className="flex gap-3">
                  <Info className="h-5 w-5 text-blue-600 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-blue-900 dark:text-blue-100">HIPAA Compliant Data Sharing</p>
                    <p className="text-sm text-blue-700 dark:text-blue-300">
                      Your data is encrypted and only shared with your healthcare providers with your explicit consent.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="font-medium">Select data to share:</span>
                <div className="flex gap-2">
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => {
                      const allOn: Record<string, boolean> = {};
                      DATA_CONSENT_TYPES.forEach(t => allOn[t.id] = true);
                      setConsentSelections(allOn);
                    }}
                    data-testid="button-select-all"
                  >
                    Select All
                  </Button>
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => setConsentSelections({})}
                    data-testid="button-clear-all"
                  >
                    Clear All
                  </Button>
                </div>
              </div>

              <ScrollArea className="h-[280px]">
                <div className="space-y-2">
                  {DATA_CONSENT_TYPES.map(type => (
                    <div 
                      key={type.id}
                      className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50"
                    >
                      <div>
                        <p className="font-medium">{type.label}</p>
                        <p className="text-sm text-muted-foreground">{type.description}</p>
                      </div>
                      <Switch
                        checked={consentSelections[type.id] || false}
                        onCheckedChange={(checked) => 
                          setConsentSelections(prev => ({ ...prev, [type.id]: checked }))
                        }
                        data-testid={`switch-consent-${type.id}`}
                      />
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          )}

          {pairingStep === 3 && (
            <div className="space-y-6">
              {VENDORS.find(v => v.id === selectedVendor)?.pairingMethod === "manual" && (
                <div className="space-y-4">
                  <Label htmlFor="device-name">Device Name (Optional)</Label>
                  <Input
                    id="device-name"
                    placeholder="My Apple Watch"
                    value={manualDeviceName}
                    onChange={(e) => setManualDeviceName(e.target.value)}
                    data-testid="input-device-name"
                  />
                  <div className="p-4 rounded-lg bg-muted">
                    <h4 className="font-medium mb-2">How to sync your data:</h4>
                    <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                      <li>Open the Health app on your iPhone</li>
                      <li>Tap your profile icon, then "Export All Health Data"</li>
                      <li>Share the exported file with Followup AI</li>
                      <li>Or use our iOS Shortcuts integration for automatic syncing</li>
                    </ol>
                  </div>
                </div>
              )}

              {VENDORS.find(v => v.id === selectedVendor)?.pairingMethod === "oauth" && (
                <div className="text-center space-y-4">
                  {pairingStatus === "idle" && (
                    <>
                      <div className="p-4 rounded-lg bg-muted inline-block mx-auto">
                        {VENDORS.find(v => v.id === selectedVendor)?.logoComponent ? (
                          (() => {
                            const Logo = VENDORS.find(v => v.id === selectedVendor)!.logoComponent!;
                            return <Logo className="h-12 w-12" />;
                          })()
                        ) : (
                          <Link2 className="h-12 w-12" />
                        )}
                      </div>
                      <p className="text-muted-foreground">
                        Click "Connect" to authorize access to your{" "}
                        {VENDORS.find(v => v.id === selectedVendor)?.name} account
                      </p>
                    </>
                  )}

                  {pairingStatus === "connecting" && (
                    <>
                      <Loader2 className="h-12 w-12 animate-spin mx-auto text-primary" />
                      <p>Waiting for authorization...</p>
                      <p className="text-sm text-muted-foreground">
                        Complete the authorization in the popup window
                      </p>
                    </>
                  )}

                  {pairingStatus === "success" && (
                    <>
                      <div className="p-4 rounded-full bg-green-100 dark:bg-green-900/30 inline-block">
                        <CheckCircle2 className="h-12 w-12 text-green-600" />
                      </div>
                      <p className="font-medium text-green-600">Successfully connected!</p>
                    </>
                  )}

                  {pairingStatus === "error" && (
                    <>
                      <div className="p-4 rounded-full bg-red-100 dark:bg-red-900/30 inline-block">
                        <XCircle className="h-12 w-12 text-red-600" />
                      </div>
                      <p className="font-medium text-red-600">Connection failed</p>
                      <p className="text-sm text-muted-foreground">
                        Please try again or contact support
                      </p>
                    </>
                  )}
                </div>
              )}

              {VENDORS.find(v => v.id === selectedVendor)?.pairingMethod === "ble" && (
                <div className="text-center space-y-4">
                  <div className="p-4 rounded-full bg-blue-100 dark:bg-blue-900/30 inline-block">
                    <Bluetooth className="h-12 w-12 text-blue-600" />
                  </div>
                  <p className="font-medium">Bluetooth Pairing</p>
                  <p className="text-muted-foreground">
                    Make sure your device is in pairing mode and nearby
                  </p>
                  <Button 
                    onClick={() => {
                      toast({
                        title: "Bluetooth Pairing",
                        description: "Web Bluetooth is available in Chrome. Scanning for devices...",
                      });
                    }}
                    data-testid="button-scan-bluetooth"
                  >
                    <Bluetooth className="mr-2 h-4 w-4" />
                    Scan for Devices
                  </Button>
                </div>
              )}
            </div>
          )}

          <DialogFooter className="gap-2">
            {pairingStep > 0 && (
              <Button 
                variant="outline" 
                onClick={() => setPairingStep(prev => prev - 1)}
                data-testid="button-wizard-back"
              >
                Back
              </Button>
            )}
            
            {pairingStep < 3 && (
              <Button 
                onClick={() => setPairingStep(prev => prev + 1)}
                disabled={
                  (pairingStep === 0 && !selectedCategory) ||
                  (pairingStep === 1 && !selectedVendor) ||
                  (pairingStep === 2 && Object.values(consentSelections).filter(Boolean).length === 0)
                }
                data-testid="button-wizard-next"
              >
                Next
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            )}
            
            {pairingStep === 3 && (
              <Button 
                onClick={handleStartPairing}
                disabled={startPairingMutation.isPending || pairingStatus === "connecting"}
                data-testid="button-connect-device"
              >
                {startPairingMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <Link2 className="mr-2 h-4 w-4" />
                    Connect
                  </>
                )}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showDeviceDetails} onOpenChange={setShowDeviceDetails}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-3">
              {selectedDevice && getDeviceIcon(selectedDevice.deviceType)}
              {selectedDevice?.deviceName}
            </DialogTitle>
            <DialogDescription>
              {VENDORS.find(v => v.id === selectedDevice?.vendorId)?.name}
            </DialogDescription>
          </DialogHeader>

          {selectedDevice && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 rounded-lg bg-muted">
                  <p className="text-sm text-muted-foreground">Status</p>
                  <div className="mt-1">{getStatusBadge(selectedDevice.connectionStatus)}</div>
                </div>
                <div className="p-3 rounded-lg bg-muted">
                  <p className="text-sm text-muted-foreground">Battery</p>
                  <p className="font-medium mt-1">
                    {selectedDevice.batteryLevel ? `${selectedDevice.batteryLevel}%` : "N/A"}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-muted">
                  <p className="text-sm text-muted-foreground">Last Sync</p>
                  <p className="font-medium mt-1">{formatLastSync(selectedDevice.lastSyncAt)}</p>
                </div>
                <div className="p-3 rounded-lg bg-muted">
                  <p className="text-sm text-muted-foreground">Firmware</p>
                  <p className="font-medium mt-1">{selectedDevice.firmwareVersion || "N/A"}</p>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">Tracked Metrics</h4>
                <div className="flex flex-wrap gap-2">
                  {selectedDevice.trackedMetrics?.map(metric => (
                    <Badge key={metric} variant="outline">
                      {METRIC_LABELS[metric] || metric}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-muted">
                <div className="flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  <span>Data Sharing Consent</span>
                </div>
                {selectedDevice.consentGiven ? (
                  <Badge className="bg-green-100 text-green-800">Active</Badge>
                ) : (
                  <Badge variant="secondary">Not Configured</Badge>
                )}
              </div>

              <Separator />

              <div className="flex gap-2">
                <Button 
                  className="flex-1"
                  onClick={() => syncDeviceMutation.mutate(selectedDevice.id)}
                  disabled={syncDeviceMutation.isPending}
                  data-testid="button-detail-sync"
                >
                  {syncDeviceMutation.isPending ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="mr-2 h-4 w-4" />
                  )}
                  Sync Now
                </Button>
                <Button 
                  variant="outline"
                  onClick={() => {
                    setConsentSelections({});
                    setShowConsentModal(true);
                  }}
                  data-testid="button-detail-consent"
                >
                  <Shield className="mr-2 h-4 w-4" />
                  Manage Consent
                </Button>
              </div>

              <Button 
                variant="destructive" 
                className="w-full"
                onClick={() => {
                  if (confirm("Are you sure you want to disconnect this device?")) {
                    disconnectDeviceMutation.mutate(selectedDevice.id);
                  }
                }}
                data-testid="button-disconnect-device"
              >
                <Unlink className="mr-2 h-4 w-4" />
                Disconnect Device
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={showConsentModal} onOpenChange={setShowConsentModal}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Data Sharing Preferences
            </DialogTitle>
            <DialogDescription>
              Choose what health data to share with your healthcare team
            </DialogDescription>
          </DialogHeader>

          <div className="p-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
            <div className="flex gap-3">
              <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0" />
              <p className="text-sm text-amber-800 dark:text-amber-200">
                Your data sharing preferences are protected under HIPAA regulations. 
                Changes are logged for compliance and audit purposes.
              </p>
            </div>
          </div>

          <ScrollArea className="h-[300px]">
            <div className="space-y-2">
              {DATA_CONSENT_TYPES.map(type => (
                <div 
                  key={type.id}
                  className="flex items-center justify-between p-3 rounded-lg border"
                >
                  <div>
                    <p className="font-medium">{type.label}</p>
                    <p className="text-sm text-muted-foreground">{type.description}</p>
                  </div>
                  <Switch
                    checked={consentSelections[type.id] || false}
                    onCheckedChange={(checked) => 
                      setConsentSelections(prev => ({ ...prev, [type.id]: checked }))
                    }
                    data-testid={`switch-update-consent-${type.id}`}
                  />
                </div>
              ))}
            </div>
          </ScrollArea>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowConsentModal(false)}>
              Cancel
            </Button>
            <Button 
              onClick={() => {
                if (selectedDevice) {
                  updateConsentMutation.mutate({
                    deviceId: selectedDevice.id,
                    consents: consentSelections,
                  });
                }
              }}
              disabled={updateConsentMutation.isPending}
              data-testid="button-save-consent"
            >
              {updateConsentMutation.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Check className="mr-2 h-4 w-4" />
              )}
              Save Preferences
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
