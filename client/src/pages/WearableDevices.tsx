import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Watch, CheckCircle2, XCircle, AlertCircle, Plus, Trash2, RefreshCw, Battery, Activity, Dumbbell, Zap, Smartphone, Stethoscope } from "lucide-react";

const WEARABLE_DEVICES = [
  { id: "amazfit", name: "Amazfit", description: "Smartwatches and fitness trackers", Icon: Watch },
  { id: "garmin", name: "Garmin", description: "Sports watches and fitness devices", Icon: Activity },
  { id: "whoop", name: "Whoop", description: "Performance optimization strap", Icon: Zap },
  { id: "samsung", name: "Samsung Galaxy Watch", description: "Galaxy smartwatch series", Icon: Smartphone },
  { id: "eko", name: "Eko Stethoscope", description: "Digital stethoscope for heart sounds", Icon: Stethoscope },
];

export default function WearableDevices() {
  const { toast } = useToast();
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [deviceModel, setDeviceModel] = useState("");

  const { data: devices, isLoading } = useQuery({
    queryKey: ["/api/wearables"],
  });

  const addDeviceMutation = useMutation({
    mutationFn: async (data: any) => {
      return await apiRequest("/api/wearables", {
        method: "POST",
        body: JSON.stringify(data),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/wearables"] });
      setShowAddDialog(false);
      setSelectedDevice("");
      setDeviceModel("");
      toast({
        title: "Device Added",
        description: "Your wearable device has been connected successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to add wearable device. Please try again.",
        variant: "destructive",
      });
    },
  });

  const syncMutation = useMutation({
    mutationFn: async (id: string) => {
      return await apiRequest(`/api/wearables/${id}/sync`, {
        method: "POST",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/wearables"] });
      toast({
        title: "Sync Complete",
        description: "Your health data has been synchronized.",
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      return await apiRequest(`/api/wearables/${id}`, {
        method: "DELETE",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/wearables"] });
      toast({
        title: "Device Removed",
        description: "The wearable device has been disconnected.",
      });
    },
  });

  const handleAddDevice = () => {
    if (!selectedDevice || !deviceModel) {
      toast({
        title: "Missing Information",
        description: "Please fill in all fields.",
        variant: "destructive",
      });
      return;
    }

    const device = WEARABLE_DEVICES.find((d) => d.id === selectedDevice);
    addDeviceMutation.mutate({
      deviceType: selectedDevice,
      deviceName: device?.name || "",
      deviceModel,
      connectionStatus: "pending",
      trackedMetrics: ["heart_rate", "steps", "sleep", "spo2"],
    });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "connected":
        return <CheckCircle2 className="h-5 w-5 text-green-600" />;
      case "error":
        return <XCircle className="h-5 w-5 text-red-600" />;
      case "pending":
        return <AlertCircle className="h-5 w-5 text-yellow-600" />;
      default:
        return <Watch className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: any = {
      connected: "default",
      error: "destructive",
      pending: "secondary",
      disconnected: "outline",
    };

    return (
      <Badge variant={variants[status] || "outline"}>
        {status?.charAt(0).toUpperCase() + status?.slice(1)}
      </Badge>
    );
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Wearable Devices</h1>
          <p className="text-muted-foreground">
            Connect your fitness trackers and health monitoring devices
          </p>
        </div>
        <Button onClick={() => setShowAddDialog(true)} data-testid="button-add-device">
          <Plus className="h-4 w-4 mr-2" />
          Add Device
        </Button>
      </div>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-4 bg-muted rounded w-1/2" />
                <div className="h-3 bg-muted rounded w-3/4 mt-2" />
              </CardHeader>
            </Card>
          ))}
        </div>
      ) : devices && devices.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {devices.map((device: any) => (
            <Card key={device.id} className="hover-elevate" data-testid={`card-device-${device.id}`}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="flex items-center gap-2" data-testid={`text-device-name-${device.id}`}>
                      {getStatusIcon(device.connectionStatus)}
                      {device.deviceName}
                    </CardTitle>
                    <CardDescription className="mt-1" data-testid={`text-device-model-${device.id}`}>
                      {device.deviceModel}
                    </CardDescription>
                  </div>
                  <div data-testid={`badge-status-${device.id}`}>
                    {getStatusBadge(device.connectionStatus)}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  {device.batteryLevel !== null && device.batteryLevel !== undefined && (
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground flex items-center gap-1">
                        <Battery className="h-3 w-3" />
                        Battery:
                      </span>
                      <span className="font-medium" data-testid={`text-battery-${device.id}`}>{device.batteryLevel}%</span>
                    </div>
                  )}
                  {device.lastSyncedAt && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Last Synced:</span>
                      <span data-testid={`text-last-synced-${device.id}`}>{new Date(device.lastSyncedAt).toLocaleDateString()}</span>
                    </div>
                  )}
                  {device.trackedMetrics && device.trackedMetrics.length > 0 && (
                    <div className="pt-2">
                      <div className="text-xs text-muted-foreground mb-1">Tracked Metrics:</div>
                      <div className="flex flex-wrap gap-1">
                        {device.trackedMetrics.slice(0, 4).map((metric: string) => (
                          <Badge key={metric} variant="outline" className="text-xs">
                            {metric.replace("_", " ")}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex gap-2 mt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() => syncMutation.mutate(device.id)}
                    disabled={syncMutation.isPending}
                    data-testid={`button-sync-${device.id}`}
                  >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Sync
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteMutation.mutate(device.id)}
                    data-testid={`button-delete-${device.id}`}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Watch className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Devices Connected</h3>
            <p className="text-muted-foreground text-center mb-4">
              Connect your wearable devices to automatically track your health metrics
            </p>
            <Button onClick={() => setShowAddDialog(true)} data-testid="button-add-device-empty">
              <Plus className="h-4 w-4 mr-2" />
              Connect Your First Device
            </Button>
          </CardContent>
        </Card>
      )}

      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Wearable Device</DialogTitle>
            <DialogDescription>
              Connect your fitness tracker or health monitoring device
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="device-type">Device Type</Label>
              <Select value={selectedDevice} onValueChange={setSelectedDevice}>
                <SelectTrigger id="device-type" data-testid="select-device-type">
                  <SelectValue placeholder="Select your device" />
                </SelectTrigger>
                <SelectContent>
                  {WEARABLE_DEVICES.map((device) => (
                    <SelectItem key={device.id} value={device.id}>
                      <div className="flex items-center gap-2">
                        <device.Icon className="h-4 w-4" />
                        <div>
                          <div className="font-medium">{device.name}</div>
                          <div className="text-xs text-muted-foreground">{device.description}</div>
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="device-model">Device Model</Label>
              <Input
                id="device-model"
                placeholder="e.g., GTR 3 Pro, Forerunner 945, Strap 4.0"
                value={deviceModel}
                onChange={(e) => setDeviceModel(e.target.value)}
                data-testid="input-device-model"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddDialog(false)} data-testid="button-cancel-device">
              Cancel
            </Button>
            <Button
              onClick={handleAddDevice}
              disabled={addDeviceMutation.isPending}
              data-testid="button-connect-device"
            >
              {addDeviceMutation.isPending ? "Connecting..." : "Connect Device"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
