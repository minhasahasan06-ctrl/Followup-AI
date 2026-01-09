import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { useToast } from "@/hooks/use-toast";
import { 
  Sparkles, FileText, CheckCircle2, XCircle, RefreshCw, 
  AlertTriangle, Loader2, ChevronRight, Clock, User, 
  Stethoscope, Activity, TestTube, Image as ImageIcon
} from "lucide-react";

interface DifferentialContent {
  case_discussion?: {
    key_positives?: string[];
    key_negatives?: string[];
    contextual_factors?: string[];
  };
  diagnostic_next_steps?: {
    labs?: string[];
    imaging?: string[];
    monitoring?: string[];
  };
  differential_diagnosis?: {
    most_likely?: string[];
    expanded_differential?: string[];
    cant_miss?: string[];
  };
  assessment_and_plan?: {
    clinical_impression?: string;
    problems?: Array<{
      problem?: string;
      assessment?: string;
      plan?: string[];
    }>;
  };
  references?: string[];
}

interface LysaDraft {
  id: string;
  patient_id: string;
  doctor_id: string;
  draft_type: string;
  status: string;
  question?: string;
  content: DifferentialContent;
  provenance?: Array<{ source: string; timestamp: string }>;
  created_at: string;
  approved_at?: string;
}

interface DraftDifferentialPanelProps {
  patientId: string;
  patientName?: string;
  onDraftApproved?: (draftId: string) => void;
}

export function DraftDifferentialPanel({ patientId, patientName, onDraftApproved }: DraftDifferentialPanelProps) {
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const [clinicalQuestion, setClinicalQuestion] = useState("");
  const [revisionNotes, setRevisionNotes] = useState("");
  const [selectedDraft, setSelectedDraft] = useState<LysaDraft | null>(null);

  const { data: drafts, isLoading, refetch } = useQuery<{ drafts: LysaDraft[] }>({
    queryKey: ["/api/v1/lysa/patient", patientId, "drafts"],
    queryFn: async () => {
      const response = await fetch(`/api/v1/lysa/patient/${patientId}/drafts`);
      if (!response.ok) throw new Error("Failed to fetch drafts");
      return response.json();
    },
    enabled: isOpen,
  });

  const generateMutation = useMutation({
    mutationFn: async () => {
      return await apiRequest("POST", `/api/v1/lysa/patient/${patientId}/differential`, {
        question: clinicalQuestion || undefined
      });
    },
    onSuccess: (data: LysaDraft) => {
      setSelectedDraft(data);
      refetch();
      toast({
        title: "Draft generated",
        description: "A new differential diagnosis draft has been created for your review.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to generate differential",
        variant: "destructive",
      });
    },
  });

  const reviseMutation = useMutation({
    mutationFn: async () => {
      if (!selectedDraft) throw new Error("No draft selected");
      return await apiRequest("POST", `/api/v1/lysa/drafts/${selectedDraft.id}/revise`, {
        revision_notes: revisionNotes
      });
    },
    onSuccess: (data: LysaDraft) => {
      setSelectedDraft(data);
      setRevisionNotes("");
      refetch();
      toast({
        title: "Draft revised",
        description: "The draft has been updated based on your notes.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to revise draft",
        variant: "destructive",
      });
    },
  });

  const approveMutation = useMutation({
    mutationFn: async () => {
      if (!selectedDraft) throw new Error("No draft selected");
      return await apiRequest("POST", `/api/v1/lysa/drafts/${selectedDraft.id}/approve`, {
        confirmation: true
      });
    },
    onSuccess: () => {
      refetch();
      onDraftApproved?.(selectedDraft!.id);
      toast({
        title: "Draft approved",
        description: "The draft has been approved and is ready for chart insertion.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to approve draft",
        variant: "destructive",
      });
    },
  });

  const renderDifferentialContent = (content: DifferentialContent) => {
    return (
      <div className="space-y-6">
        {content.case_discussion && (
          <section className="space-y-3">
            <h4 className="font-medium flex items-center gap-2">
              <User className="h-4 w-4" />
              Case Discussion
            </h4>
            {content.case_discussion.key_positives?.length ? (
              <div>
                <p className="text-sm font-medium text-green-600 dark:text-green-400">Key Positives</p>
                <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
                  {content.case_discussion.key_positives.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {content.case_discussion.key_negatives?.length ? (
              <div>
                <p className="text-sm font-medium text-red-600 dark:text-red-400">Key Negatives</p>
                <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
                  {content.case_discussion.key_negatives.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </section>
        )}

        {content.differential_diagnosis && (
          <section className="space-y-3">
            <h4 className="font-medium flex items-center gap-2">
              <Stethoscope className="h-4 w-4" />
              Differential Diagnosis
            </h4>
            {content.differential_diagnosis.cant_miss?.length ? (
              <div>
                <p className="text-sm font-medium text-red-600 dark:text-red-400 flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  Can't Miss Diagnoses
                </p>
                <ul className="list-disc list-inside text-sm space-y-1">
                  {content.differential_diagnosis.cant_miss.map((item, i) => (
                    <li key={i} className="text-red-600 dark:text-red-400">{item}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {content.differential_diagnosis.most_likely?.length ? (
              <div>
                <p className="text-sm font-medium">Most Likely</p>
                <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
                  {content.differential_diagnosis.most_likely.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </section>
        )}

        {content.diagnostic_next_steps && (
          <section className="space-y-3">
            <h4 className="font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Recommended Next Steps
            </h4>
            <div className="grid gap-3 md:grid-cols-3">
              {content.diagnostic_next_steps.labs?.length ? (
                <div className="p-3 border rounded-lg">
                  <p className="text-sm font-medium flex items-center gap-1 mb-2">
                    <TestTube className="h-3 w-3" />
                    Labs
                  </p>
                  <ul className="text-xs space-y-1 text-muted-foreground">
                    {content.diagnostic_next_steps.labs.map((item, i) => (
                      <li key={i}>• {item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {content.diagnostic_next_steps.imaging?.length ? (
                <div className="p-3 border rounded-lg">
                  <p className="text-sm font-medium flex items-center gap-1 mb-2">
                    <ImageIcon className="h-3 w-3" />
                    Imaging
                  </p>
                  <ul className="text-xs space-y-1 text-muted-foreground">
                    {content.diagnostic_next_steps.imaging.map((item, i) => (
                      <li key={i}>• {item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {content.diagnostic_next_steps.monitoring?.length ? (
                <div className="p-3 border rounded-lg">
                  <p className="text-sm font-medium flex items-center gap-1 mb-2">
                    <Activity className="h-3 w-3" />
                    Monitoring
                  </p>
                  <ul className="text-xs space-y-1 text-muted-foreground">
                    {content.diagnostic_next_steps.monitoring.map((item, i) => (
                      <li key={i}>• {item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </section>
        )}

        {content.assessment_and_plan?.problems?.length ? (
          <section className="space-y-3">
            <h4 className="font-medium flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Assessment & Plan
            </h4>
            {content.assessment_and_plan.clinical_impression && (
              <p className="text-sm italic text-muted-foreground">
                {content.assessment_and_plan.clinical_impression}
              </p>
            )}
            <div className="space-y-2">
              {content.assessment_and_plan.problems.map((problem, i) => (
                <div key={i} className="p-3 border rounded-lg">
                  <p className="font-medium text-sm">{problem.problem}</p>
                  {problem.assessment && (
                    <p className="text-xs text-muted-foreground mt-1">{problem.assessment}</p>
                  )}
                  {problem.plan?.length ? (
                    <ul className="text-xs mt-2 space-y-1">
                      {problem.plan.map((p, j) => (
                        <li key={j}>→ {p}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {content.references?.length ? (
          <section className="space-y-2">
            <h4 className="text-sm font-medium text-muted-foreground">References</h4>
            <ul className="text-xs space-y-1 text-muted-foreground">
              {content.references.map((ref, i) => (
                <li key={i}>{ref}</li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>
    );
  };

  const STATUS_BADGES: Record<string, { variant: "default" | "secondary" | "destructive" | "outline"; label: string }> = {
    draft: { variant: "outline", label: "Draft" },
    revised: { variant: "secondary", label: "Revised" },
    approved: { variant: "default", label: "Approved" },
    rejected: { variant: "destructive", label: "Rejected" },
    inserted_to_chart: { variant: "default", label: "In Chart" },
  };

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" className="gap-2" data-testid="button-open-differential-panel">
          <Sparkles className="h-4 w-4" />
          AI Differential
          <ChevronRight className="h-4 w-4" />
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[600px] sm:max-w-[600px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Lysa Clinical Drafts
          </SheetTitle>
          <SheetDescription>
            AI-generated clinical documentation for {patientName || "patient"}. 
            All drafts require your review and approval.
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="h-[calc(100vh-120px)] mt-4 pr-4">
          <Tabs defaultValue="generate" className="space-y-4">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="generate">Generate New</TabsTrigger>
              <TabsTrigger value="history">Draft History</TabsTrigger>
            </TabsList>

            <TabsContent value="generate" className="space-y-4">
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  These are AI-generated drafts for your review. They are not medical advice 
                  and must be verified before use.
                </AlertDescription>
              </Alert>

              <div className="space-y-2">
                <Label htmlFor="question">Clinical Question (Optional)</Label>
                <Textarea
                  id="question"
                  placeholder="e.g., What is causing this patient's shortness of breath?"
                  value={clinicalQuestion}
                  onChange={(e) => setClinicalQuestion(e.target.value)}
                  data-testid="textarea-clinical-question"
                />
              </div>

              <Button
                onClick={() => generateMutation.mutate()}
                disabled={generateMutation.isPending}
                className="w-full"
                data-testid="button-generate-differential"
              >
                {generateMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Generate Differential Diagnosis
                  </>
                )}
              </Button>

              {selectedDraft && (
                <Card className="mt-4">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">Generated Draft</CardTitle>
                      <Badge {...STATUS_BADGES[selectedDraft.status]}>
                        {STATUS_BADGES[selectedDraft.status]?.label || selectedDraft.status}
                      </Badge>
                    </div>
                    <CardDescription className="flex items-center gap-2">
                      <Clock className="h-3 w-3" />
                      {new Date(selectedDraft.created_at).toLocaleString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {renderDifferentialContent(selectedDraft.content)}

                    {selectedDraft.status === "draft" || selectedDraft.status === "revised" ? (
                      <div className="mt-6 space-y-4">
                        <Separator />
                        
                        <div className="space-y-2">
                          <Label>Revision Notes</Label>
                          <Textarea
                            placeholder="Add notes to revise the draft..."
                            value={revisionNotes}
                            onChange={(e) => setRevisionNotes(e.target.value)}
                            data-testid="textarea-revision-notes"
                          />
                        </div>

                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            onClick={() => reviseMutation.mutate()}
                            disabled={reviseMutation.isPending || !revisionNotes}
                            className="flex-1"
                          >
                            {reviseMutation.isPending ? (
                              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                            ) : (
                              <RefreshCw className="h-4 w-4 mr-2" />
                            )}
                            Revise
                          </Button>
                          <Button
                            onClick={() => approveMutation.mutate()}
                            disabled={approveMutation.isPending}
                            className="flex-1"
                            data-testid="button-approve-draft"
                          >
                            {approveMutation.isPending ? (
                              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                            ) : (
                              <CheckCircle2 className="h-4 w-4 mr-2" />
                            )}
                            Approve
                          </Button>
                        </div>
                      </div>
                    ) : null}
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="history" className="space-y-4">
              {isLoading ? (
                <div className="space-y-3">
                  <Skeleton className="h-20 w-full" />
                  <Skeleton className="h-20 w-full" />
                </div>
              ) : drafts?.drafts?.length ? (
                drafts.drafts.map((draft) => (
                  <Card
                    key={draft.id}
                    className="cursor-pointer hover:bg-muted/50 transition-colors"
                    onClick={() => setSelectedDraft(draft)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                          <p className="font-medium capitalize">{draft.draft_type.replace("_", " ")}</p>
                          <p className="text-xs text-muted-foreground flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {new Date(draft.created_at).toLocaleString()}
                          </p>
                        </div>
                        <Badge {...STATUS_BADGES[draft.status]}>
                          {STATUS_BADGES[draft.status]?.label || draft.status}
                        </Badge>
                      </div>
                      {draft.question && (
                        <p className="text-sm text-muted-foreground mt-2 italic">
                          "{draft.question}"
                        </p>
                      )}
                    </CardContent>
                  </Card>
                ))
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <FileText className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No drafts yet</p>
                  <p className="text-sm">Generate a differential to get started</p>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

export default DraftDifferentialPanel;
