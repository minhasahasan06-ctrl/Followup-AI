import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Plus, X, Building2, Award, Globe, Video, DollarSign, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface DoctorProfileExtended {
  user_id: string;
  npi: string | null;
  hospital_affiliations: string[];
  board_certifications: string[];
  languages: string[];
  accepted_insurances: string[];
  telemedicine_available: boolean;
  telemedicine_fee: number | null;
  consultation_fee: number | null;
}

export default function DoctorProfessionalInfoForm() {
  const { toast } = useToast();
  
  const { data: profile, isLoading } = useQuery<DoctorProfileExtended>({
    queryKey: ["/api/doctor/profile/extended"],
  });

  const [npi, setNpi] = useState("");
  const [hospitalAffiliations, setHospitalAffiliations] = useState<string[]>([]);
  const [boardCertifications, setBoardCertifications] = useState<string[]>([]);
  const [languages, setLanguages] = useState<string[]>([]);
  const [acceptedInsurances, setAcceptedInsurances] = useState<string[]>([]);
  const [telemedicineAvailable, setTelemedicineAvailable] = useState(false);
  const [telemedicineFee, setTelemedicineFee] = useState("");
  const [consultationFee, setConsultationFee] = useState("");
  
  const [newAffiliation, setNewAffiliation] = useState("");
  const [newCertification, setNewCertification] = useState("");
  const [newLanguage, setNewLanguage] = useState("");
  const [newInsurance, setNewInsurance] = useState("");

  useEffect(() => {
    if (profile) {
      setNpi(profile.npi || "");
      setHospitalAffiliations(profile.hospital_affiliations || []);
      setBoardCertifications(profile.board_certifications || []);
      setLanguages(profile.languages || []);
      setAcceptedInsurances(profile.accepted_insurances || []);
      setTelemedicineAvailable(profile.telemedicine_available || false);
      setTelemedicineFee(profile.telemedicine_fee ? String(profile.telemedicine_fee / 100) : "");
      setConsultationFee(profile.consultation_fee ? String(profile.consultation_fee / 100) : "");
    }
  }, [profile]);

  const updateProfileMutation = useMutation({
    mutationFn: async (data: Partial<DoctorProfileExtended>) => {
      return await apiRequest("POST", "/api/doctor/profile/extended", data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/doctor/profile/extended"] });
      toast({
        title: "Profile updated",
        description: "Your professional information has been saved",
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
      npi: npi || null,
      hospital_affiliations: hospitalAffiliations,
      board_certifications: boardCertifications,
      languages: languages,
      accepted_insurances: acceptedInsurances,
      telemedicine_available: telemedicineAvailable,
      telemedicine_fee: telemedicineFee ? Math.round(parseFloat(telemedicineFee) * 100) : null,
      consultation_fee: consultationFee ? Math.round(parseFloat(consultationFee) * 100) : null,
    });
  };

  const addItem = (
    value: string,
    setter: React.Dispatch<React.SetStateAction<string>>,
    list: string[],
    listSetter: React.Dispatch<React.SetStateAction<string[]>>
  ) => {
    if (value.trim() && !list.includes(value.trim())) {
      listSetter([...list, value.trim()]);
      setter("");
    }
  };

  const removeItem = (
    item: string,
    list: string[],
    listSetter: React.Dispatch<React.SetStateAction<string[]>>
  ) => {
    listSetter(list.filter(i => i !== item));
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
            <Award className="h-5 w-5 text-primary" />
            Credentials
          </CardTitle>
          <CardDescription>
            Your NPI and board certifications
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="npi">NPI (National Provider Identifier)</Label>
            <Input
              id="npi"
              placeholder="10-digit NPI number"
              value={npi}
              onChange={(e) => setNpi(e.target.value)}
              maxLength={10}
              data-testid="input-npi"
            />
            <p className="text-xs text-muted-foreground">
              Your 10-digit NPI is required for insurance billing and provider verification
            </p>
          </div>
          
          <Separator />
          
          <div className="space-y-3">
            <Label>Board Certifications</Label>
            <div className="flex flex-wrap gap-2">
              {boardCertifications.map((cert) => (
                <Badge key={cert} variant="secondary" className="flex items-center gap-1">
                  {cert}
                  <button
                    onClick={() => removeItem(cert, boardCertifications, setBoardCertifications)}
                    className="ml-1 hover:bg-secondary-foreground/20 rounded-full p-0.5"
                    data-testid={`button-remove-certification-${cert}`}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
              {boardCertifications.length === 0 && (
                <p className="text-muted-foreground text-sm">No certifications listed</p>
              )}
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Add certification (e.g., ABIM Internal Medicine)"
                value={newCertification}
                onChange={(e) => setNewCertification(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && addItem(newCertification, setNewCertification, boardCertifications, setBoardCertifications)}
                data-testid="input-new-certification"
              />
              <Button 
                variant="outline" 
                onClick={() => addItem(newCertification, setNewCertification, boardCertifications, setBoardCertifications)}
                data-testid="button-add-certification"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5 text-primary" />
            Hospital Affiliations
          </CardTitle>
          <CardDescription>
            Hospitals and medical centers where you practice
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {hospitalAffiliations.map((hospital) => (
              <Badge key={hospital} variant="outline" className="flex items-center gap-1">
                <Building2 className="h-3 w-3" />
                {hospital}
                <button
                  onClick={() => removeItem(hospital, hospitalAffiliations, setHospitalAffiliations)}
                  className="ml-1 hover:bg-muted rounded-full p-0.5"
                  data-testid={`button-remove-affiliation-${hospital}`}
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
            {hospitalAffiliations.length === 0 && (
              <p className="text-muted-foreground text-sm">No hospital affiliations listed</p>
            )}
          </div>
          <div className="flex gap-2">
            <Input
              placeholder="Add hospital (e.g., Mayo Clinic)"
              value={newAffiliation}
              onChange={(e) => setNewAffiliation(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addItem(newAffiliation, setNewAffiliation, hospitalAffiliations, setHospitalAffiliations)}
              data-testid="input-new-affiliation"
            />
            <Button 
              variant="outline" 
              onClick={() => addItem(newAffiliation, setNewAffiliation, hospitalAffiliations, setHospitalAffiliations)}
              data-testid="button-add-affiliation"
            >
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Video className="h-5 w-5 text-primary" />
            Telemedicine Settings
          </CardTitle>
          <CardDescription>
            Configure your virtual visit availability and fees
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <p className="font-medium">Telemedicine Available</p>
              <p className="text-sm text-muted-foreground">
                Accept virtual video consultations from patients
              </p>
            </div>
            <Switch
              checked={telemedicineAvailable}
              onCheckedChange={setTelemedicineAvailable}
              data-testid="switch-telemedicine"
            />
          </div>
          
          {telemedicineAvailable && (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="telemedicine-fee">Telemedicine Visit Fee ($)</Label>
                <div className="relative">
                  <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="telemedicine-fee"
                    type="number"
                    placeholder="0.00"
                    value={telemedicineFee}
                    onChange={(e) => setTelemedicineFee(e.target.value)}
                    className="pl-9"
                    data-testid="input-telemedicine-fee"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="consultation-fee">Standard Consultation Fee ($)</Label>
                <div className="relative">
                  <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="consultation-fee"
                    type="number"
                    placeholder="0.00"
                    value={consultationFee}
                    onChange={(e) => setConsultationFee(e.target.value)}
                    className="pl-9"
                    data-testid="input-consultation-fee"
                  />
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Globe className="h-5 w-5 text-primary" />
            Languages & Insurance
          </CardTitle>
          <CardDescription>
            Languages spoken and insurance plans accepted
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label>Languages Spoken</Label>
            <div className="flex flex-wrap gap-2">
              {languages.map((lang) => (
                <Badge key={lang} variant="secondary" className="flex items-center gap-1">
                  {lang}
                  <button
                    onClick={() => removeItem(lang, languages, setLanguages)}
                    className="ml-1 hover:bg-secondary-foreground/20 rounded-full p-0.5"
                    data-testid={`button-remove-language-${lang}`}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
              {languages.length === 0 && (
                <p className="text-muted-foreground text-sm">No languages listed</p>
              )}
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Add language (e.g., Spanish)"
                value={newLanguage}
                onChange={(e) => setNewLanguage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && addItem(newLanguage, setNewLanguage, languages, setLanguages)}
                data-testid="input-new-language"
              />
              <Button 
                variant="outline" 
                onClick={() => addItem(newLanguage, setNewLanguage, languages, setLanguages)}
                data-testid="button-add-language"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <Separator />

          <div className="space-y-3">
            <Label>Accepted Insurance Plans</Label>
            <div className="flex flex-wrap gap-2">
              {acceptedInsurances.map((ins) => (
                <Badge key={ins} variant="outline" className="flex items-center gap-1">
                  {ins}
                  <button
                    onClick={() => removeItem(ins, acceptedInsurances, setAcceptedInsurances)}
                    className="ml-1 hover:bg-muted rounded-full p-0.5"
                    data-testid={`button-remove-insurance-${ins}`}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
              {acceptedInsurances.length === 0 && (
                <p className="text-muted-foreground text-sm">No insurance plans listed</p>
              )}
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Add insurance (e.g., Blue Cross Blue Shield)"
                value={newInsurance}
                onChange={(e) => setNewInsurance(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && addItem(newInsurance, setNewInsurance, acceptedInsurances, setAcceptedInsurances)}
                data-testid="input-new-insurance"
              />
              <Button 
                variant="outline" 
                onClick={() => addItem(newInsurance, setNewInsurance, acceptedInsurances, setAcceptedInsurances)}
                data-testid="button-add-insurance"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Separator />

      <div className="flex justify-end">
        <Button 
          onClick={handleSave} 
          disabled={updateProfileMutation.isPending}
          data-testid="button-save-professional-info"
        >
          {updateProfileMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Save Professional Information
        </Button>
      </div>
    </div>
  );
}
