import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { MapPin, Shield, Loader2, CheckCircle2 } from "lucide-react";

interface EnvironmentalAutoCreateProps {
  patientId?: string;
}

export function EnvironmentalAutoCreate({ patientId }: EnvironmentalAutoCreateProps) {
  const { toast } = useToast();
  const [consentGiven, setConsentGiven] = useState(false);
  const [isGettingLocation, setIsGettingLocation] = useState(false);
  const [profileCreated, setProfileCreated] = useState(false);

  const createProfileMutation = useMutation({
    mutationFn: async (data: { lat: number; lon: number; consent: boolean; allow_store_latlon?: boolean }) => {
      const endpoint = `/api/v1/environment/patient/${patientId || "me"}/auto_create`;
      const res = await apiRequest(endpoint, {
        method: "POST",
        json: { ...data, allow_store_latlon: false }
      });
      return await res.json();
    },
    onSuccess: (data) => {
      setProfileCreated(true);
      queryClient.invalidateQueries({ queryKey: ["/api/environment/profile"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/environment"] });
      toast({
        title: "Profile created",
        description: data?.message || "Your environmental health profile has been set up based on your location.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to create environmental profile",
        variant: "destructive",
      });
    },
  });

  const handleAutoCreate = async () => {
    if (!consentGiven) {
      toast({
        title: "Consent required",
        description: "Please agree to the privacy terms before continuing.",
        variant: "destructive",
      });
      return;
    }

    setIsGettingLocation(true);

    if (!navigator.geolocation) {
      toast({
        title: "Location not supported",
        description: "Your browser does not support location services.",
        variant: "destructive",
      });
      setIsGettingLocation(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setIsGettingLocation(false);
        createProfileMutation.mutate({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
          consent: consentGiven,
        });
      },
      (error) => {
        setIsGettingLocation(false);
        toast({
          title: "Location error",
          description: error.message || "Unable to get your location. Please try again.",
          variant: "destructive",
        });
      },
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 }
    );
  };

  if (profileCreated) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center gap-3 text-green-600">
            <CheckCircle2 className="h-5 w-5" />
            <span className="font-medium">Environmental profile created successfully</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MapPin className="h-5 w-5" />
          Quick Environmental Setup
        </CardTitle>
        <CardDescription>
          Set up personalized environmental health monitoring based on your location
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-start gap-3 p-4 bg-muted/50 rounded-lg">
          <Shield className="h-5 w-5 text-primary mt-0.5" />
          <div className="space-y-2">
            <p className="text-sm font-medium">Privacy-First Design</p>
            <p className="text-sm text-muted-foreground">
              We only store your ZIP code - never your exact location. Your precise coordinates 
              are used once to look up your area, then immediately discarded.
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Checkbox
            id="consent"
            checked={consentGiven}
            onCheckedChange={(checked) => setConsentGiven(checked === true)}
            data-testid="checkbox-location-consent"
          />
          <div className="grid gap-1.5 leading-none">
            <Label htmlFor="consent" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
              I consent to location-based environmental monitoring
            </Label>
            <p className="text-xs text-muted-foreground">
              Allow us to use your location to provide air quality alerts, pollen forecasts, 
              and weather-related health recommendations.
            </p>
          </div>
        </div>

        <Button
          onClick={handleAutoCreate}
          disabled={!consentGiven || isGettingLocation || createProfileMutation.isPending}
          className="w-full"
          data-testid="button-auto-create-env-profile"
        >
          {isGettingLocation ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Getting location...
            </>
          ) : createProfileMutation.isPending ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Creating profile...
            </>
          ) : (
            <>
              <MapPin className="h-4 w-4 mr-2" />
              Set Up Environmental Monitoring
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}

export default EnvironmentalAutoCreate;
