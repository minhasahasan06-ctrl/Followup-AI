import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";

interface FeatureFlag {
  name: string;
  enabled: boolean;
  description?: string;
  context_triggers?: string[];
}

interface FeatureFlagsResponse {
  flags: Record<string, FeatureFlag>;
  user_overrides?: Record<string, boolean>;
  role_overrides?: Record<string, boolean>;
}

const FEATURE_FLAG_DEFAULTS: Record<string, boolean> = {
  showClonaCallIcons: true,
  enableVoice: true,
  enableMemory: true,
  enablePersonaMarketplace: false,
  enableHabitTracker: true,
  enableRedFlagDetection: true,
  enableEscalation: true,
  showResearchCenter: true,
  showEpidemiologyDashboard: true,
  enableTinkerIntegration: true,
  enableAdvancedVoiceFeatures: false,
  showDeveloperTools: false,
  enableBetaFeatures: false,
  useNewChatUI: false,
  enableOfflineMode: false,
  showPerformanceMetrics: false,
  enableA11yEnhancements: true,
};

export function useFeatureFlags() {
  const { data, isLoading, error } = useQuery<FeatureFlagsResponse>({
    queryKey: ["/api/feature-flags"],
    staleTime: 60000,
    retry: 1,
  });

  const isEnabled = (flagName: string): boolean => {
    if (data?.flags?.[flagName]) {
      return data.flags[flagName].enabled;
    }
    return FEATURE_FLAG_DEFAULTS[flagName] ?? false;
  };

  const getFlag = (flagName: string): FeatureFlag | null => {
    return data?.flags?.[flagName] ?? null;
  };

  const getAllFlags = (): Record<string, FeatureFlag> => {
    return data?.flags ?? {};
  };

  return {
    isEnabled,
    getFlag,
    getAllFlags,
    isLoading,
    error,
    flags: data?.flags ?? {},
  };
}

export function useFeatureFlagToggle() {
  return useMutation({
    mutationFn: async ({ flagName, enabled }: { flagName: string; enabled: boolean }) => {
      const response = await fetch(`/api/feature-flags/${flagName}/toggle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) throw new Error("Failed to toggle feature flag");
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/feature-flags"] });
    },
  });
}
