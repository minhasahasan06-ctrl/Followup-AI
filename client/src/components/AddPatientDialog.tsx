import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  UserPlus,
  Search,
  CheckCircle,
  Clock,
  Send,
  Shield,
  AlertCircle,
  Loader2,
} from "lucide-react";

type SearchResult = {
  id: string;
  firstName: string | null;
  lastName: string | null;
  email: string;
  followupPatientId: string | null;
  hasAccess: boolean;
  hasPendingRequest: boolean;
};

interface AddPatientDialogProps {
  trigger?: React.ReactNode;
}

export function AddPatientDialog({ trigger }: AddPatientDialogProps) {
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<SearchResult | null>(null);
  const [requestMessage, setRequestMessage] = useState("");
  const [step, setStep] = useState<"search" | "request">("search");

  const {
    data: searchResults,
    isLoading: isSearching,
    refetch: searchPatients,
  } = useQuery<SearchResult[]>({
    queryKey: ["/api/doctor/patient-search", searchQuery],
    queryFn: async () => {
      if (searchQuery.trim().length < 3) return [];
      const res = await fetch(`/api/doctor/patient-search?q=${encodeURIComponent(searchQuery)}`);
      if (!res.ok) throw new Error("Search failed");
      return res.json();
    },
    enabled: false,
  });

  const sendConsentRequest = useMutation({
    mutationFn: async (data: { patientId: string; requestMessage: string }) => {
      const res = await apiRequest("POST", "/api/doctor/consent-requests", data);
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Access Request Sent",
        description: "The patient will receive a notification to approve or deny your request.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/doctor/consent-requests/pending"] });
      queryClient.invalidateQueries({ queryKey: ["/api/doctor/assigned-patients"] });
      handleClose();
    },
    onError: (error: Error) => {
      toast({
        title: "Request Failed",
        description: error.message || "Could not send the access request. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleSearch = () => {
    if (searchQuery.trim().length >= 3) {
      searchPatients();
    }
  };

  const handleSelectPatient = (patient: SearchResult) => {
    setSelectedPatient(patient);
    setRequestMessage(`I would like to request access to your health records to provide you with personalized care and treatment.`);
    setStep("request");
  };

  const handleSendRequest = () => {
    if (!selectedPatient) return;
    sendConsentRequest.mutate({
      patientId: selectedPatient.id,
      requestMessage,
    });
  };

  const handleClose = () => {
    setOpen(false);
    setSearchQuery("");
    setSelectedPatient(null);
    setRequestMessage("");
    setStep("search");
  };

  const getInitials = (firstName?: string | null, lastName?: string | null) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => {
      if (!isOpen) handleClose();
      else setOpen(true);
    }}>
      <DialogTrigger asChild>
        {trigger || (
          <Button data-testid="button-add-patient">
            <UserPlus className="h-4 w-4 mr-2" />
            Add Patient
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            {step === "search" ? "Find Patient" : "Request Access"}
          </DialogTitle>
          <DialogDescription>
            {step === "search" 
              ? "Search for a patient by their email, phone number, or Followup Patient ID."
              : "Send an access request to this patient. They must approve before you can view their health data."
            }
          </DialogDescription>
        </DialogHeader>

        {step === "search" ? (
          <div className="space-y-4">
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Email, phone, or Patient ID (e.g., FAI-XXXXXX)"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                  className="pl-10"
                  data-testid="input-patient-search"
                />
              </div>
              <Button 
                onClick={handleSearch} 
                disabled={searchQuery.trim().length < 3 || isSearching}
                data-testid="button-search-patients"
              >
                {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : "Search"}
              </Button>
            </div>

            {searchQuery.length > 0 && searchQuery.length < 3 && (
              <p className="text-sm text-muted-foreground">
                Enter at least 3 characters to search
              </p>
            )}

            {searchResults && searchResults.length > 0 && (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {searchResults.map((patient) => (
                  <Card 
                    key={patient.id}
                    className={`hover-elevate cursor-pointer ${patient.hasAccess || patient.hasPendingRequest ? 'opacity-60' : ''}`}
                    onClick={() => !patient.hasAccess && !patient.hasPendingRequest && handleSelectPatient(patient)}
                    data-testid={`search-result-${patient.id}`}
                  >
                    <CardContent className="p-3">
                      <div className="flex items-center gap-3">
                        <Avatar className="h-10 w-10">
                          <AvatarFallback className="bg-primary/10 text-primary">
                            {getInitials(patient.firstName, patient.lastName)}
                          </AvatarFallback>
                        </Avatar>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-medium truncate">
                              {patient.firstName} {patient.lastName}
                            </span>
                            {patient.followupPatientId && (
                              <Badge variant="outline" className="text-xs shrink-0">
                                {patient.followupPatientId}
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground truncate">
                            {patient.email}
                          </p>
                        </div>
                        {patient.hasAccess ? (
                          <Badge variant="secondary" className="shrink-0">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Connected
                          </Badge>
                        ) : patient.hasPendingRequest ? (
                          <Badge variant="outline" className="shrink-0">
                            <Clock className="h-3 w-3 mr-1" />
                            Pending
                          </Badge>
                        ) : (
                          <Button size="sm" variant="ghost" className="shrink-0">
                            Select
                          </Button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}

            {searchResults && searchResults.length === 0 && searchQuery.length >= 3 && (
              <div className="text-center py-8">
                <AlertCircle className="h-8 w-8 mx-auto text-muted-foreground opacity-50 mb-2" />
                <p className="text-sm text-muted-foreground">
                  No patients found matching your search.
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Ask the patient to share their Followup Patient ID with you.
                </p>
              </div>
            )}

            <Separator />

            <div className="bg-muted/50 rounded-lg p-4 text-sm">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <Shield className="h-4 w-4 text-primary" />
                HIPAA Consent Required
              </h4>
              <p className="text-muted-foreground">
                For patient privacy and HIPAA compliance, patients must explicitly consent 
                before you can access their health records. The patient will receive a 
                notification and can approve or deny your request.
              </p>
            </div>
          </div>
        ) : selectedPatient && (
          <div className="space-y-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <Avatar className="h-12 w-12">
                    <AvatarFallback className="bg-primary text-primary-foreground">
                      {getInitials(selectedPatient.firstName, selectedPatient.lastName)}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <h4 className="font-semibold">
                      {selectedPatient.firstName} {selectedPatient.lastName}
                    </h4>
                    <p className="text-sm text-muted-foreground">{selectedPatient.email}</p>
                    {selectedPatient.followupPatientId && (
                      <Badge variant="outline" className="text-xs mt-1">
                        {selectedPatient.followupPatientId}
                      </Badge>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="space-y-2">
              <Label htmlFor="requestMessage">Message to Patient (Optional)</Label>
              <Textarea
                id="requestMessage"
                placeholder="Explain why you're requesting access..."
                value={requestMessage}
                onChange={(e) => setRequestMessage(e.target.value)}
                rows={4}
                data-testid="input-request-message"
              />
              <p className="text-xs text-muted-foreground">
                This message will be shown to the patient when they review your request.
              </p>
            </div>

            <div className="flex gap-2">
              <Button 
                variant="outline" 
                onClick={() => {
                  setSelectedPatient(null);
                  setStep("search");
                }}
                className="flex-1"
                data-testid="button-back-to-search"
              >
                Back
              </Button>
              <Button 
                onClick={handleSendRequest}
                disabled={sendConsentRequest.isPending}
                className="flex-1"
                data-testid="button-send-request"
              >
                {sendConsentRequest.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Send className="h-4 w-4 mr-2" />
                )}
                Send Request
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
