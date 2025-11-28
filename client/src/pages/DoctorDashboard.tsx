import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Users, Search, TrendingUp, AlertCircle, Sparkles, Lightbulb, BookOpen, Beaker, Clock, CheckCircle, Bot } from "lucide-react";
import { useState } from "react";
import { useLocation } from "wouter";
import type { User } from "@shared/schema";
import { AddPatientDialog } from "@/components/AddPatientDialog";
import { LysaChatPanel } from "@/components/LysaChatPanel";

type Recommendation = {
  id: string;
  type: string;
  category: string;
  title: string;
  description: string;
  confidenceScore: string;
  priority: string;
  reasoning?: string;
};

export default function DoctorDashboard() {
  const [, setLocation] = useLocation();
  const [searchQuery, setSearchQuery] = useState("");
  const [lysaDialogOpen, setLysaDialogOpen] = useState(false);
  const [selectedPatientForLysa, setSelectedPatientForLysa] = useState<User | null>(null);

  const { data: patients, isLoading } = useQuery<User[]>({
    queryKey: ["/api/doctor/patients"],
  });

  const openLysaForPatient = (patient: User, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedPatientForLysa(patient);
    setLysaDialogOpen(true);
  };

  // Fetch Assistant Lysa recommendations for doctors
  const { data: recommendations = [], isLoading: recommendationsLoading, isError: recommendationsError } = useQuery<Recommendation[]>({
    queryKey: ['/api/v1/ml/recommendations', { agentType: 'lysa', limit: 6 }],
    queryFn: async () => {
      const res = await fetch('/api/v1/ml/recommendations?agentType=lysa&limit=6');
      if (!res.ok) throw new Error('Failed to fetch recommendations');
      return res.json();
    },
  });

  const clinicalRecommendations = recommendations.filter(r => 
    r.category === 'clinical_decision_support' || r.category === 'protocol_suggestion'
  );
  
  const researchRecommendations = recommendations.filter(r => 
    r.category === 'research' || r.category === 'literature_review'
  );

  const filteredPatients = patients?.filter((patient) =>
    `${patient.firstName} ${patient.lastName}`.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getInitials = (firstName?: string | null, lastName?: string | null) => {
    return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase() || "?";
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-semibold mb-2" data-testid="text-dashboard-title">Doctor Dashboard</h1>
          <p className="text-muted-foreground">Clinical insights and patient overview powered by Assistant Lysa</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="secondary" className="text-sm">
            <Users className="h-3 w-3 mr-1" />
            {patients?.length || 0} Patients
          </Badge>
          <AddPatientDialog />
        </div>
      </div>

      {/* Assistant Lysa Recommendations Section */}
      {recommendationsLoading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {[1, 2].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-5 bg-muted rounded w-1/2 mb-2" />
                <div className="h-4 bg-muted rounded w-3/4" />
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[1, 2, 3].map((j) => (
                    <div key={j} className="h-20 bg-muted rounded" />
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : recommendationsError ? (
        <Card>
          <CardContent className="py-8 text-center">
            <AlertCircle className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">
              Unable to load recommendations. Please try again later.
            </p>
          </CardContent>
        </Card>
      ) : (clinicalRecommendations.length > 0 || researchRecommendations.length > 0) && (
        <div className="grid gap-4 md:grid-cols-2">
          {/* Clinical Insights */}
          {clinicalRecommendations.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-primary" />
                  <CardTitle className="text-lg">Clinical Insights</CardTitle>
                </div>
                <CardDescription>
                  AI-powered information from Assistant Lysa for discussion with patients
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {clinicalRecommendations.slice(0, 3).map((rec) => (
                    <div key={rec.id} className="p-3 border rounded-lg hover-elevate" data-testid={`clinical-recommendation-${rec.id}`}>
                      <div className="flex items-start justify-between mb-1">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <Lightbulb className="w-4 h-4 text-yellow-500" />
                            <h4 className="font-medium text-sm">{rec.title}</h4>
                          </div>
                          <p className="text-xs text-muted-foreground">{rec.description}</p>
                        </div>
                        <Badge variant={rec.priority === 'high' ? 'default' : 'secondary'} className="text-xs ml-2">
                          {rec.priority}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 mt-2">
                        <span className="text-xs text-muted-foreground">
                          {Math.round(parseFloat(rec.confidenceScore) * 100)}% confidence
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Research Recommendations */}
          {researchRecommendations.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Beaker className="w-5 h-5 text-primary" />
                  <CardTitle className="text-lg">Research Insights</CardTitle>
                </div>
                <CardDescription>
                  Latest research and clinical literature
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {researchRecommendations.slice(0, 3).map((rec) => (
                    <div key={rec.id} className="p-3 border rounded-lg hover-elevate" data-testid={`research-recommendation-${rec.id}`}>
                      <div className="flex items-start justify-between mb-1">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <BookOpen className="w-4 h-4 text-blue-500" />
                            <h4 className="font-medium text-sm">{rec.title}</h4>
                          </div>
                          <p className="text-xs text-muted-foreground">{rec.description}</p>
                        </div>
                        <Badge variant={rec.priority === 'high' ? 'default' : 'secondary'} className="text-xs ml-2">
                          {rec.priority}
                        </Badge>
                      </div>
                      {rec.reasoning && (
                        <p className="text-xs text-muted-foreground mt-2 italic">
                          {rec.reasoning}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Patient List Section */}
      <div className="flex items-center gap-2">
        <Users className="w-5 h-5" />
        <h2 className="text-2xl font-semibold">All Patients</h2>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search patients by name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
                data-testid="input-search-patients"
              />
            </div>
          </div>
        </CardHeader>
      </Card>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-6">
                <div className="flex items-center gap-3">
                  <div className="h-12 w-12 rounded-full bg-muted" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-muted rounded w-3/4" />
                    <div className="h-3 bg-muted rounded w-1/2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : filteredPatients && filteredPatients.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredPatients.map((patient) => (
            <Card
              key={patient.id}
              className="hover-elevate cursor-pointer"
              onClick={() => setLocation(`/doctor/patient/${patient.id}`)}
              data-testid={`card-patient-${patient.id}`}
            >
              <CardContent className="p-6">
                <div className="flex items-start gap-3 mb-4">
                  <Avatar className="h-12 w-12">
                    <AvatarFallback className="bg-primary text-primary-foreground">
                      {getInitials(patient.firstName, patient.lastName)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold truncate" data-testid={`text-patient-name-${patient.id}`}>
                      {patient.firstName} {patient.lastName}
                    </h3>
                    <p className="text-sm text-muted-foreground truncate">{patient.email}</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Last Follow-up</span>
                    <span className="font-medium">Today</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Adherence</span>
                    <Badge variant="secondary" className="text-xs">
                      <TrendingUp className="h-3 w-3 mr-1" />
                      92%
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Alerts</span>
                    <Badge variant="secondary" className="text-xs">
                      <AlertCircle className="h-3 w-3 mr-1" />
                      0
                    </Badge>
                  </div>
                </div>

                <div className="flex gap-2 mt-4">
                  <Button 
                    className="flex-1" 
                    variant="outline" 
                    data-testid={`button-view-patient-${patient.id}`}
                  >
                    View Patient
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={(e) => openLysaForPatient(patient, e)}
                    data-testid={`button-lysa-patient-${patient.id}`}
                    className="shrink-0"
                    title="Open Lysa AI Assistant"
                  >
                    <Bot className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="p-12 text-center">
            <Users className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
            <h3 className="text-lg font-semibold mb-2">No patients found</h3>
            <p className="text-muted-foreground">
              {searchQuery ? "Try adjusting your search" : "No patients have been added yet"}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Lysa AI Assistant Dialog */}
      <Dialog open={lysaDialogOpen} onOpenChange={setLysaDialogOpen}>
        <DialogContent className="max-w-4xl h-[80vh]" data-testid="dialog-lysa-assistant">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-primary" />
              Lysa AI Assistant
              {selectedPatientForLysa && (
                <Badge variant="secondary" className="ml-2">
                  {selectedPatientForLysa.firstName} {selectedPatientForLysa.lastName}
                </Badge>
              )}
            </DialogTitle>
            <DialogDescription>
              AI-powered clinical assistant for patient care and decision support
            </DialogDescription>
          </DialogHeader>
          <div className="flex-1 overflow-hidden -mx-6 -mb-6">
            {selectedPatientForLysa && (
              <LysaChatPanel 
                patientId={selectedPatientForLysa.id}
                patientName={`${selectedPatientForLysa.firstName} ${selectedPatientForLysa.lastName}`}
                className="h-full border-0 rounded-none"
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
