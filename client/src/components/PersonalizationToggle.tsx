import { useMemo } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import type { UserSettings } from "@shared/schema";

export function PersonalizationToggle() {
  const { toast } = useToast();

  const { data: settings, isLoading } = useQuery<UserSettings | null>({
    queryKey: ["/api/user/settings"],
  });

  const personalizationEnabled = useMemo(
    () => settings?.personalizationEnabled ?? false,
    [settings]
  );

  const updatePersonalization = useMutation({
    mutationFn: async (enabled: boolean) => {
      const res = await apiRequest("/api/user/settings/personalization", {
        method: "POST",
        json: { enabled },
      });
      return (await res.json()) as UserSettings;
    },
    onSuccess: (data) => {
      queryClient.setQueryData(["/api/user/settings"], data);
      toast({
        title: data.personalizationEnabled
          ? "Personalization enabled"
          : "Personalization disabled",
        description: data.personalizationEnabled
          ? "Agents will adapt to your preferences and interaction history."
          : "Agents will respond with global defaults only.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Unable to update personalization",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-4">
          <div>
            <CardTitle>Personalization</CardTitle>
            <CardDescription>
              Decide whether Followup AI should tailor responses using your preferences and history.
            </CardDescription>
          </div>
          <Badge variant={personalizationEnabled ? "default" : "secondary"}>
            {personalizationEnabled ? "On" : "Off"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-medium">Adaptive responses</p>
            <p className="text-sm text-muted-foreground">
              Turn on to let agents use your past interactions, goals, and preferences.
            </p>
          </div>
          <Switch
            checked={personalizationEnabled}
            disabled={isLoading || updatePersonalization.isPending}
            onCheckedChange={(checked) => updatePersonalization.mutate(checked)}
            data-testid="switch-personalization"
          />
        </div>
        <Separator />
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="rounded-lg border p-3 text-sm">
            <p className="font-medium">When on</p>
            <p className="text-muted-foreground">
              Recommendations and agent plans adapt to your role, context, and prior feedback.
            </p>
          </div>
          <div className="rounded-lg border p-3 text-sm">
            <p className="font-medium">When off</p>
            <p className="text-muted-foreground">
              Agents use safe global defaults without storing new personalization data.
            </p>
          </div>
        </div>
        {!personalizationEnabled && (
          <div className="flex flex-col gap-2 rounded-lg border bg-muted/40 p-3 text-sm">
            <p className="font-medium">Need a quick personalization pass?</p>
            <p className="text-muted-foreground">
              You can re-enable personalization anytime. This does not delete prior data but stops new adaptive updates.
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => updatePersonalization.mutate(true)}
                disabled={updatePersonalization.isPending}
              >
                Turn personalization on
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
