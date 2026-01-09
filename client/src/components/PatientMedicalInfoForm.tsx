import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Plus, X, User, Pill, AlertTriangle, Activity, Phone, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface EmergencyContact {
  name: string;
  relationship: string;
  phone: string;
  preferred: boolean;
}

interface Medication {
  name: string;
  dose: string;
  frequency: string;
  start_date?: string;
  instructions?: string;
}

interface PatientProfileExtended {
  user_id: string;
  emergency_contacts: EmergencyContact[];
  medications: Medication[];
  allergies: string[];
  chronic_conditions: string[];
}

export default function PatientMedicalInfoForm() {
  const { toast } = useToast();
  
  const { data: profile, isLoading } = useQuery<PatientProfileExtended>({
    queryKey: ["/api/patient/profile/extended"],
  });

  const [emergencyContacts, setEmergencyContacts] = useState<EmergencyContact[]>([]);
  const [medications, setMedications] = useState<Medication[]>([]);
  const [allergies, setAllergies] = useState<string[]>([]);
  const [chronicConditions, setChronicConditions] = useState<string[]>([]);
  
  const [newAllergy, setNewAllergy] = useState("");
  const [newCondition, setNewCondition] = useState("");

  useEffect(() => {
    if (profile) {
      setEmergencyContacts(profile.emergency_contacts || []);
      setMedications(profile.medications || []);
      setAllergies(profile.allergies || []);
      setChronicConditions(profile.chronic_conditions || []);
    }
  }, [profile]);

  const updateProfileMutation = useMutation({
    mutationFn: async (data: Partial<PatientProfileExtended>) => {
      return await apiRequest("POST", "/api/patient/profile/extended", data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/patient/profile/extended"] });
      toast({
        title: "Profile updated",
        description: "Your medical information has been saved",
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

  const handleSave = () => {
    updateProfileMutation.mutate({
      emergency_contacts: emergencyContacts,
      medications: medications,
      allergies: allergies,
      chronic_conditions: chronicConditions,
    });
  };

  const addEmergencyContact = () => {
    setEmergencyContacts([...emergencyContacts, { name: "", relationship: "", phone: "", preferred: false }]);
  };

  const updateEmergencyContact = (index: number, field: keyof EmergencyContact, value: string | boolean) => {
    const updated = [...emergencyContacts];
    updated[index] = { ...updated[index], [field]: value };
    setEmergencyContacts(updated);
  };

  const removeEmergencyContact = (index: number) => {
    setEmergencyContacts(emergencyContacts.filter((_, i) => i !== index));
  };

  const addMedication = () => {
    setMedications([...medications, { name: "", dose: "", frequency: "" }]);
  };

  const updateMedication = (index: number, field: keyof Medication, value: string) => {
    const updated = [...medications];
    updated[index] = { ...updated[index], [field]: value };
    setMedications(updated);
  };

  const removeMedication = (index: number) => {
    setMedications(medications.filter((_, i) => i !== index));
  };

  const addAllergy = () => {
    if (newAllergy.trim() && !allergies.includes(newAllergy.trim())) {
      setAllergies([...allergies, newAllergy.trim()]);
      setNewAllergy("");
    }
  };

  const removeAllergy = (allergy: string) => {
    setAllergies(allergies.filter(a => a !== allergy));
  };

  const addChronicCondition = () => {
    if (newCondition.trim() && !chronicConditions.includes(newCondition.trim())) {
      setChronicConditions([...chronicConditions, newCondition.trim()]);
      setNewCondition("");
    }
  };

  const removeChronicCondition = (condition: string) => {
    setChronicConditions(chronicConditions.filter(c => c !== condition));
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Phone className="h-5 w-5 text-primary" />
            Emergency Contacts
          </CardTitle>
          <CardDescription>
            People to contact in case of a medical emergency
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {emergencyContacts.map((contact, index) => (
            <div key={index} className="p-4 border rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <User className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">Contact {index + 1}</span>
                  {contact.preferred && (
                    <Badge variant="secondary">Primary</Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => removeEmergencyContact(index)}
                  data-testid={`button-remove-contact-${index}`}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                <div className="space-y-1">
                  <Label htmlFor={`contact-name-${index}`}>Name</Label>
                  <Input
                    id={`contact-name-${index}`}
                    placeholder="Full name"
                    value={contact.name}
                    onChange={(e) => updateEmergencyContact(index, "name", e.target.value)}
                    data-testid={`input-contact-name-${index}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor={`contact-relationship-${index}`}>Relationship</Label>
                  <Input
                    id={`contact-relationship-${index}`}
                    placeholder="e.g., Spouse, Parent"
                    value={contact.relationship}
                    onChange={(e) => updateEmergencyContact(index, "relationship", e.target.value)}
                    data-testid={`input-contact-relationship-${index}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor={`contact-phone-${index}`}>Phone</Label>
                  <Input
                    id={`contact-phone-${index}`}
                    placeholder="(555) 123-4567"
                    value={contact.phone}
                    onChange={(e) => updateEmergencyContact(index, "phone", e.target.value)}
                    data-testid={`input-contact-phone-${index}`}
                  />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  checked={contact.preferred}
                  onCheckedChange={(checked) => updateEmergencyContact(index, "preferred", checked)}
                  data-testid={`switch-contact-preferred-${index}`}
                />
                <Label>Primary contact</Label>
              </div>
            </div>
          ))}
          <Button variant="outline" onClick={addEmergencyContact} data-testid="button-add-contact">
            <Plus className="h-4 w-4 mr-2" />
            Add Emergency Contact
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Pill className="h-5 w-5 text-primary" />
            Current Medications
          </CardTitle>
          <CardDescription>
            List all medications you are currently taking
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {medications.map((med, index) => (
            <div key={index} className="p-4 border rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-medium">Medication {index + 1}</span>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => removeMedication(index)}
                  data-testid={`button-remove-medication-${index}`}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                <div className="space-y-1">
                  <Label htmlFor={`med-name-${index}`}>Medication Name</Label>
                  <Input
                    id={`med-name-${index}`}
                    placeholder="e.g., Metformin"
                    value={med.name}
                    onChange={(e) => updateMedication(index, "name", e.target.value)}
                    data-testid={`input-medication-name-${index}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor={`med-dose-${index}`}>Dose</Label>
                  <Input
                    id={`med-dose-${index}`}
                    placeholder="e.g., 500mg"
                    value={med.dose}
                    onChange={(e) => updateMedication(index, "dose", e.target.value)}
                    data-testid={`input-medication-dose-${index}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor={`med-frequency-${index}`}>Frequency</Label>
                  <Input
                    id={`med-frequency-${index}`}
                    placeholder="e.g., Twice daily"
                    value={med.frequency}
                    onChange={(e) => updateMedication(index, "frequency", e.target.value)}
                    data-testid={`input-medication-frequency-${index}`}
                  />
                </div>
              </div>
              <div className="space-y-1">
                <Label htmlFor={`med-instructions-${index}`}>Special Instructions (optional)</Label>
                <Input
                  id={`med-instructions-${index}`}
                  placeholder="e.g., Take with food"
                  value={med.instructions || ""}
                  onChange={(e) => updateMedication(index, "instructions", e.target.value)}
                  data-testid={`input-medication-instructions-${index}`}
                />
              </div>
            </div>
          ))}
          <Button variant="outline" onClick={addMedication} data-testid="button-add-medication">
            <Plus className="h-4 w-4 mr-2" />
            Add Medication
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-destructive" />
            Allergies
          </CardTitle>
          <CardDescription>
            Drug allergies and sensitivities your care team should know about
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {allergies.map((allergy) => (
              <Badge key={allergy} variant="destructive" className="flex items-center gap-1">
                {allergy}
                <button
                  onClick={() => removeAllergy(allergy)}
                  className="ml-1 hover:bg-destructive-foreground/20 rounded-full p-0.5"
                  data-testid={`button-remove-allergy-${allergy}`}
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
            {allergies.length === 0 && (
              <p className="text-muted-foreground text-sm">No allergies listed</p>
            )}
          </div>
          <div className="flex gap-2">
            <Input
              placeholder="Add allergy (e.g., Penicillin)"
              value={newAllergy}
              onChange={(e) => setNewAllergy(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addAllergy()}
              data-testid="input-new-allergy"
            />
            <Button variant="outline" onClick={addAllergy} data-testid="button-add-allergy">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            Chronic Conditions
          </CardTitle>
          <CardDescription>
            Ongoing health conditions that require regular monitoring
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {chronicConditions.map((condition) => (
              <Badge key={condition} variant="secondary" className="flex items-center gap-1">
                {condition}
                <button
                  onClick={() => removeChronicCondition(condition)}
                  className="ml-1 hover:bg-secondary-foreground/20 rounded-full p-0.5"
                  data-testid={`button-remove-condition-${condition}`}
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
            {chronicConditions.length === 0 && (
              <p className="text-muted-foreground text-sm">No chronic conditions listed</p>
            )}
          </div>
          <div className="flex gap-2">
            <Input
              placeholder="Add condition (e.g., Type 2 Diabetes)"
              value={newCondition}
              onChange={(e) => setNewCondition(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addChronicCondition()}
              data-testid="input-new-condition"
            />
            <Button variant="outline" onClick={addChronicCondition} data-testid="button-add-condition">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      <Separator />

      <div className="flex justify-end">
        <Button 
          onClick={handleSave} 
          disabled={updateProfileMutation.isPending}
          data-testid="button-save-medical-info"
        >
          {updateProfileMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Save Medical Information
        </Button>
      </div>
    </div>
  );
}
