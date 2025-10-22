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
import { Activity, CheckCircle2, XCircle, AlertCircle, Plus, Trash2, RefreshCw } from "lucide-react";

const EHR_SYSTEMS = [
  { id: "epic", name: "Epic MyChart", description: "Access your Epic health records" },
  { id: "cerner", name: "Oracle Cerner", description: "Connect to Oracle Health records" },
  { id: "athena", name: "Athena Health", description: "Sync with Athena Practice" },
  { id: "eclinicalworks", name: "eClinicalWorks", description: "Link eClinicalWorks EHR" },
  { id: "allscripts", name: "Allscripts", description: "Connect Allscripts system" },
  { id: "advancedmd", name: "AdvancedMD", description: "Sync AdvancedMD records" },
  { id: "meditech", name: "Meditech", description: "Access Meditech data" },
  { id: "nextgen", name: "NextGen Healthcare", description: "Link NextGen EHR" },
  { id: "drchrono", name: "DrChrono", description: "Connect DrChrono platform" },
];

export default function EHRIntegrations() {
  const { toast } = useToast();
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [selectedSystem, setSelectedSystem] = useState("");
  const [facilityName, setFacilityName] = useState("");
  const [patientId, setPatientId] = useState("");

  const { data: connections, isLoading } = useQuery({
    queryKey: ["/api/ehr/connections"],
  });

  const addConnectionMutation = useMutation({
    mutationFn: async (data: any) => {
      return await apiRequest("/api/ehr/connections", {
        method: "POST",
        body: JSON.stringify(data),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ehr/connections"] });
      setShowAddDialog(false);
      setSelectedSystem("");
      setFacilityName("");
      setPatientId("");
      toast({
        title: "Connection Added",
        description: "Your EHR connection has been created successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to add EHR connection. Please try again.",
        variant: "destructive",
      });
    },
  });

  const syncMutation = useMutation({
    mutationFn: async (id: string) => {
      return await apiRequest(`/api/ehr/connections/${id}/sync`, {
        method: "POST",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ehr/connections"] });
      toast({
        title: "Sync Complete",
        description: "Your health records have been synchronized.",
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      return await apiRequest(`/api/ehr/connections/${id}`, {
        method: "DELETE",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ehr/connections"] });
      toast({
        title: "Connection Removed",
        description: "The EHR connection has been deleted.",
      });
    },
  });

  const handleAddConnection = () => {
    if (!selectedSystem || !facilityName || !patientId) {
      toast({
        title: "Missing Information",
        description: "Please fill in all fields.",
        variant: "destructive",
      });
      return;
    }

    const system = EHR_SYSTEMS.find((s) => s.id === selectedSystem);
    addConnectionMutation.mutate({
      ehrSystem: selectedSystem,
      ehrSystemName: system?.name || "",
      facilityName,
      patientExternalId: patientId,
      connectionStatus: "pending",
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
        return <Activity className="h-5 w-5 text-gray-400" />;
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
          <h1 className="text-3xl font-bold mb-2">EHR Integrations</h1>
          <p className="text-muted-foreground">
            Connect your electronic health records from your healthcare providers
          </p>
        </div>
        <Button onClick={() => setShowAddDialog(true)} data-testid="button-add-ehr">
          <Plus className="h-4 w-4 mr-2" />
          Add Connection
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
      ) : connections && connections.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {connections.map((connection: any) => (
            <Card key={connection.id} className="hover-elevate" data-testid={`card-ehr-${connection.id}`}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="flex items-center gap-2" data-testid={`text-ehr-name-${connection.id}`}>
                      {getStatusIcon(connection.connectionStatus)}
                      {connection.ehrSystemName}
                    </CardTitle>
                    <CardDescription className="mt-1" data-testid={`text-facility-${connection.id}`}>
                      {connection.facilityName}
                    </CardDescription>
                  </div>
                  <div data-testid={`badge-status-${connection.id}`}>
                    {getStatusBadge(connection.connectionStatus)}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Patient ID:</span>
                    <span className="font-mono" data-testid={`text-patient-id-${connection.id}`}>{connection.patientExternalId}</span>
                  </div>
                  {connection.lastSyncedAt && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Last Synced:</span>
                      <span data-testid={`text-last-synced-${connection.id}`}>{new Date(connection.lastSyncedAt).toLocaleDateString()}</span>
                    </div>
                  )}
                </div>
                <div className="flex gap-2 mt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() => syncMutation.mutate(connection.id)}
                    disabled={syncMutation.isPending}
                    data-testid={`button-sync-${connection.id}`}
                  >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Sync
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteMutation.mutate(connection.id)}
                    data-testid={`button-delete-${connection.id}`}
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
            <Activity className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No EHR Connections</h3>
            <p className="text-muted-foreground text-center mb-4">
              Connect your electronic health records to sync your medical data automatically
            </p>
            <Button onClick={() => setShowAddDialog(true)} data-testid="button-add-ehr-empty">
              <Plus className="h-4 w-4 mr-2" />
              Add Your First Connection
            </Button>
          </CardContent>
        </Card>
      )}

      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add EHR Connection</DialogTitle>
            <DialogDescription>
              Connect to your healthcare provider's electronic health record system
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="ehr-system">EHR System</Label>
              <Select value={selectedSystem} onValueChange={setSelectedSystem}>
                <SelectTrigger id="ehr-system" data-testid="select-ehr-system">
                  <SelectValue placeholder="Select your EHR system" />
                </SelectTrigger>
                <SelectContent>
                  {EHR_SYSTEMS.map((system) => (
                    <SelectItem key={system.id} value={system.id}>
                      <div>
                        <div className="font-medium">{system.name}</div>
                        <div className="text-xs text-muted-foreground">{system.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="facility-name">Facility/Hospital Name</Label>
              <Input
                id="facility-name"
                placeholder="e.g., Memorial Hospital"
                value={facilityName}
                onChange={(e) => setFacilityName(e.target.value)}
                data-testid="input-facility-name"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="patient-id">Your Patient ID</Label>
              <Input
                id="patient-id"
                placeholder="e.g., 123456789"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                data-testid="input-patient-id"
              />
              <p className="text-xs text-muted-foreground">
                You can find this on your patient portal or medical documents
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddDialog(false)} data-testid="button-cancel-ehr">
              Cancel
            </Button>
            <Button
              onClick={handleAddConnection}
              disabled={addConnectionMutation.isPending}
              data-testid="button-connect-ehr"
            >
              {addConnectionMutation.isPending ? "Connecting..." : "Connect"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
