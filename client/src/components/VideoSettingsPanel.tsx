import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Video, ExternalLink, Shield, Loader2, Check, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface VideoSettings {
  doctor_id: string;
  preferred_provider: "daily" | "zoom" | "google_meet";
  external_video_enabled: boolean;
  default_zoom_link: string | null;
  default_meet_link: string | null;
  max_duration_minutes: number;
  recording_enabled: boolean;
  waiting_room_enabled: boolean;
  chat_enabled: boolean;
  screen_share_enabled: boolean;
  auto_transcription: boolean;
}

export default function VideoSettingsPanel() {
  const { toast } = useToast();
  const [hasChanges, setHasChanges] = useState(false);

  const { data: settings, isLoading, error } = useQuery<VideoSettings>({
    queryKey: ["/api/video/settings"],
    retry: false
  });

  const [formData, setFormData] = useState<Partial<VideoSettings>>({});

  const currentSettings = { ...settings, ...formData };

  const updateSettingsMutation = useMutation({
    mutationFn: async (data: Partial<VideoSettings>) => {
      return await apiRequest("PUT", "/api/video/settings", data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/video/settings"] });
      setHasChanges(false);
      setFormData({});
      toast({
        title: "Settings saved",
        description: "Your video consultation settings have been updated",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error saving settings",
        description: error.message || "Failed to save video settings",
        variant: "destructive",
      });
    },
  });

  const handleChange = <K extends keyof VideoSettings>(key: K, value: VideoSettings[K]) => {
    setFormData(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = () => {
    updateSettingsMutation.mutate(formData);
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-32">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Unable to load video settings. Please try again later.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Video className="h-5 w-5 text-primary" />
            <CardTitle>Video Consultation Settings</CardTitle>
          </div>
          <Badge variant="outline" className="gap-1">
            <Shield className="h-3 w-3" />
            HIPAA Compliant
          </Badge>
        </div>
        <CardDescription>
          Configure your video visit preferences for patient consultations
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Preferred Video Provider</Label>
            <Select
              value={currentSettings.preferred_provider || "daily"}
              onValueChange={(value) => handleChange("preferred_provider", value as VideoSettings["preferred_provider"])}
            >
              <SelectTrigger data-testid="select-video-provider">
                <SelectValue placeholder="Select provider" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="daily">
                  <div className="flex items-center gap-2">
                    <Video className="h-4 w-4" />
                    Daily.co (Built-in, HIPAA Compliant)
                  </div>
                </SelectItem>
                <SelectItem value="zoom">
                  <div className="flex items-center gap-2">
                    <ExternalLink className="h-4 w-4" />
                    Zoom (External Link)
                  </div>
                </SelectItem>
                <SelectItem value="google_meet">
                  <div className="flex items-center gap-2">
                    <ExternalLink className="h-4 w-4" />
                    Google Meet (External Link)
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Daily.co provides built-in video with usage tracking. External providers require you to manage your own links.
            </p>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Enable External Video Links</Label>
              <p className="text-xs text-muted-foreground">
                Allow using Zoom or Google Meet links for video visits
              </p>
            </div>
            <Switch
              checked={currentSettings.external_video_enabled || false}
              onCheckedChange={(checked) => handleChange("external_video_enabled", checked)}
              data-testid="switch-external-video"
            />
          </div>

          {currentSettings.external_video_enabled && (
            <div className="space-y-4 pl-4 border-l-2 border-muted">
              <div className="space-y-2">
                <Label htmlFor="zoom-link">Default Zoom Meeting URL</Label>
                <Input
                  id="zoom-link"
                  placeholder="https://zoom.us/j/..."
                  value={currentSettings.default_zoom_link || ""}
                  onChange={(e) => handleChange("default_zoom_link", e.target.value || null)}
                  data-testid="input-zoom-link"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="meet-link">Default Google Meet URL</Label>
                <Input
                  id="meet-link"
                  placeholder="https://meet.google.com/..."
                  value={currentSettings.default_meet_link || ""}
                  onChange={(e) => handleChange("default_meet_link", e.target.value || null)}
                  data-testid="input-meet-link"
                />
              </div>
            </div>
          )}
        </div>

        <Separator />

        <div className="space-y-4">
          <h4 className="text-sm font-medium">Call Settings</h4>
          
          <div className="space-y-2">
            <Label>Default Call Duration</Label>
            <Select
              value={String(currentSettings.max_duration_minutes || 30)}
              onValueChange={(value) => handleChange("max_duration_minutes", parseInt(value))}
            >
              <SelectTrigger data-testid="select-duration">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="15">15 minutes</SelectItem>
                <SelectItem value="30">30 minutes</SelectItem>
                <SelectItem value="45">45 minutes</SelectItem>
                <SelectItem value="60">60 minutes</SelectItem>
                <SelectItem value="90">90 minutes</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="flex items-center justify-between">
              <Label>Waiting Room</Label>
              <Switch
                checked={currentSettings.waiting_room_enabled || false}
                onCheckedChange={(checked) => handleChange("waiting_room_enabled", checked)}
                data-testid="switch-waiting-room"
              />
            </div>

            <div className="flex items-center justify-between">
              <Label>In-Call Chat</Label>
              <Switch
                checked={currentSettings.chat_enabled ?? true}
                onCheckedChange={(checked) => handleChange("chat_enabled", checked)}
                data-testid="switch-chat"
              />
            </div>

            <div className="flex items-center justify-between">
              <Label>Screen Sharing</Label>
              <Switch
                checked={currentSettings.screen_share_enabled ?? true}
                onCheckedChange={(checked) => handleChange("screen_share_enabled", checked)}
                data-testid="switch-screen-share"
              />
            </div>

            <div className="flex items-center justify-between">
              <Label>Auto Transcription</Label>
              <Switch
                checked={currentSettings.auto_transcription || false}
                onCheckedChange={(checked) => handleChange("auto_transcription", checked)}
                data-testid="switch-transcription"
              />
            </div>
          </div>
        </div>

        <Separator />

        <Alert>
          <Shield className="h-4 w-4" />
          <AlertDescription>
            All video sessions are end-to-end encrypted and HIPAA compliant. No PHI is stored in room URLs or session data.
          </AlertDescription>
        </Alert>

        <div className="flex justify-end gap-2">
          {hasChanges && (
            <Button
              variant="outline"
              onClick={() => {
                setFormData({});
                setHasChanges(false);
              }}
              data-testid="button-cancel-video-settings"
            >
              Cancel
            </Button>
          )}
          <Button
            onClick={handleSave}
            disabled={!hasChanges || updateSettingsMutation.isPending}
            data-testid="button-save-video-settings"
          >
            {updateSettingsMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : hasChanges ? null : (
              <Check className="h-4 w-4 mr-2" />
            )}
            {updateSettingsMutation.isPending ? "Saving..." : hasChanges ? "Save Settings" : "Saved"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
