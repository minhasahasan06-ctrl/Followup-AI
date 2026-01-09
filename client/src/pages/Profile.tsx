import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { User, Shield, Bell, Heart, Lock, Video, Clock, Stethoscope } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { PatientProfile, DoctorProfile } from "@shared/schema";
import PhoneVerification from "@/components/PhoneVerification";
import TrainingConsentSettings from "@/components/TrainingConsentSettings";
import { PersonalizationToggle } from "@/components/PersonalizationToggle";
import VideoSettingsPanel from "@/components/VideoSettingsPanel";
import { AssignedDoctorCard } from "@/components/AssignedDoctorCard";
import { AuditLogViewer } from "@/components/AuditLogViewer";
import PatientMedicalInfoForm from "@/components/PatientMedicalInfoForm";
import DoctorProfessionalInfoForm from "@/components/DoctorProfessionalInfoForm";

export default function Profile() {
  const { user } = useAuth();
  const { toast } = useToast();
  const isPatient = user?.role === "patient";

  const { data: patientProfile } = useQuery<PatientProfile>({
    queryKey: ["/api/patient/profile"],
    enabled: isPatient,
  });

  const { data: doctorProfile } = useQuery<DoctorProfile>({
    queryKey: ["/api/doctor/profile"],
    enabled: !isPatient,
  });

  const updatePatientProfileMutation = useMutation({
    mutationFn: async (data: Partial<PatientProfile>) => {
      return await apiRequest("POST", "/api/patient/profile", data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/patient/profile"] });
      toast({
        title: "Profile updated",
        description: "Your profile has been updated successfully",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to update profile",
        variant: "destructive",
      });
    },
  });

  const updateDoctorProfileMutation = useMutation({
    mutationFn: async (data: Partial<DoctorProfile>) => {
      return await apiRequest("POST", "/api/doctor/profile", data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/doctor/profile"] });
      toast({
        title: "Profile updated",
        description: "Your profile has been updated successfully",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to update profile",
        variant: "destructive",
      });
    },
  });

  const getInitials = (firstName?: string | null, lastName?: string | null) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold mb-2">Profile Settings</h1>
        <p className="text-muted-foreground">Manage your account information and preferences</p>
      </div>

      <Card>
        <CardContent className="p-6">
          <div className="flex items-center gap-4">
            <Avatar className="h-20 w-20">
              <AvatarFallback className="bg-primary text-primary-foreground text-2xl">
                {getInitials(user?.firstName, user?.lastName)}
              </AvatarFallback>
            </Avatar>
            <div className="flex-1">
              <h2 className="text-2xl font-semibold" data-testid="text-user-name">
                {user?.firstName} {user?.lastName}
              </h2>
              <p className="text-muted-foreground">{user?.email}</p>
              <div className="flex gap-2 mt-2">
                <Badge variant="secondary" data-testid="badge-user-role">
                  {user?.role === "patient" ? "Patient" : "Doctor"}
                </Badge>
                {user?.role === "doctor" && user?.licenseVerified && (
                  <Badge variant="secondary">
                    <Shield className="h-3 w-3 mr-1" />
                    Verified
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="account" className="space-y-6">
        <TabsList>
          <TabsTrigger value="account" data-testid="tab-account">
            <User className="h-4 w-4 mr-2" />
            Account
          </TabsTrigger>
          <TabsTrigger value="medical" data-testid="tab-medical">
            <Heart className="h-4 w-4 mr-2" />
            {isPatient ? "Medical Info" : "Professional Info"}
          </TabsTrigger>
          <TabsTrigger value="notifications" data-testid="tab-notifications">
            <Bell className="h-4 w-4 mr-2" />
            Notifications
          </TabsTrigger>
          <TabsTrigger value="personalization" data-testid="tab-personalization">
            <Shield className="h-4 w-4 mr-2" />
            Personalization
          </TabsTrigger>
          {isPatient && (
            <TabsTrigger value="privacy" data-testid="tab-privacy">
              <Lock className="h-4 w-4 mr-2" />
              Privacy & ML
            </TabsTrigger>
          )}
          {isPatient && (
            <TabsTrigger value="doctor" data-testid="tab-doctor">
              <Stethoscope className="h-4 w-4 mr-2" />
              My Doctor
            </TabsTrigger>
          )}
          <TabsTrigger value="activity" data-testid="tab-activity">
            <Clock className="h-4 w-4 mr-2" />
            Activity
          </TabsTrigger>
          {!isPatient && (
            <TabsTrigger value="video" data-testid="tab-video">
              <Video className="h-4 w-4 mr-2" />
              Video Visits
            </TabsTrigger>
          )}
        </TabsList>

        <TabsContent value="account">
          <Card>
            <CardHeader>
              <CardTitle>Account Information</CardTitle>
              <CardDescription>Update your basic account details</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="firstName">First Name</Label>
                  <Input
                    id="firstName"
                    defaultValue={user?.firstName || ""}
                    data-testid="input-first-name"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lastName">Last Name</Label>
                  <Input
                    id="lastName"
                    defaultValue={user?.lastName || ""}
                    data-testid="input-last-name"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" type="email" defaultValue={user?.email || ""} disabled />
                <p className="text-xs text-muted-foreground">Email cannot be changed</p>
              </div>
              <Button data-testid="button-save-account">Save Changes</Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="medical">
          {isPatient ? (
            <PatientMedicalInfoForm />
          ) : (
            <DoctorProfessionalInfoForm />
          )}
        </TabsContent>

        <TabsContent value="notifications">
          <div className="space-y-6">
            <PhoneVerification />
            
            <Card>
              <CardHeader>
                <CardTitle>Email Notification Preferences</CardTitle>
                <CardDescription>Manage email updates and alerts</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Email Notifications</p>
                    <p className="text-sm text-muted-foreground">Receive updates via email</p>
                  </div>
                  <Button variant="outline" size="sm" data-testid="button-toggle-email">
                    Enabled
                  </Button>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Daily Reminders</p>
                    <p className="text-sm text-muted-foreground">Get daily follow-up reminders</p>
                  </div>
                  <Button variant="outline" size="sm" data-testid="button-toggle-reminders">
                    Enabled
                  </Button>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Research Updates</p>
                    <p className="text-sm text-muted-foreground">Notifications about research participation</p>
                  </div>
                  <Button variant="outline" size="sm" data-testid="button-toggle-research">
                    Disabled
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="personalization">
          <PersonalizationToggle />
        </TabsContent>

        {isPatient && (
          <TabsContent value="privacy">
            <TrainingConsentSettings />
          </TabsContent>
        )}

        {isPatient && (
          <TabsContent value="doctor">
            <AssignedDoctorCard />
          </TabsContent>
        )}

        <TabsContent value="activity">
          <AuditLogViewer />
        </TabsContent>

        {!isPatient && (
          <TabsContent value="video">
            <VideoSettingsPanel />
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}
