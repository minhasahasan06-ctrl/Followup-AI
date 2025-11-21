import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { AlertTriangle, Check, CheckCircle2, ClipboardList, TrendingDown, TrendingUp, FileText, Calendar, Phone, Download, Printer, Loader2, Brain, Activity } from "lucide-react";
import { format } from "date-fns";
import { useToast } from "@/hooks/use-toast";
import { queryClient, apiRequest } from "@/lib/queryClient";

interface QuestionnaireTemplate {
  type: string;
  full_name: string;
  description: string;
  public_domain: boolean;
  timeframe: string;
  instructions: string;
  response_options: Array<{
    value: number;
    label: string;
  }>;
  questions: Array<{
    id: string;
    text: string;
    cluster: string;
    crisis_flag?: boolean;
    reverse_scored?: boolean;
  }>;
  scoring: {
    max_score: number;
    severity_levels: Array<{
      range: number[];
      level: string;
      description: string;
    }>;
  };
}

interface QuestionnaireResponse {
  response_id: string;
  questionnaire_type: string;
  score: {
    total_score: number;
    max_score: number;
    severity_level: string;
    severity_description: string;
    cluster_scores: Record<string, any>;
    neutral_summary: string;
    key_observations: string[];
  };
  crisis_intervention?: {
    crisis_detected: boolean;
    crisis_severity: string;
    intervention_message: string;
    crisis_hotlines: Array<{
      name: string;
      phone?: string;
      sms?: string;
      description: string;
      website: string;
    }>;
    next_steps: string[];
  };
  analysis_id?: string;
}

interface HistoryItem {
  response_id: string;
  questionnaire_type: string;
  completed_at: string;
  total_score: number;
  max_score: number;
  severity_level: string;
  crisis_detected: boolean;
}

export default function MentalHealth() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState<"questionnaires" | "history">("questionnaires");
  const [selectedQuestionnaire, setSelectedQuestionnaire] = useState<QuestionnaireTemplate | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [responses, setResponses] = useState<Record<string, { response: number; response_text: string }>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [completedResponse, setCompletedResponse] = useState<QuestionnaireResponse | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);

  // Fetch available questionnaires
  const { data: questionnairesData } = useQuery({
    queryKey: ["/api/v1/mental-health/questionnaires"],
  });

  // Fetch history
  const { data: historyData, refetch: refetchHistory } = useQuery({
    queryKey: ["/api/v1/mental-health/history"],
  });

  // Start questionnaire
  const handleStartQuestionnaire = (template: QuestionnaireTemplate) => {
    setSelectedQuestionnaire(template);
    setCurrentQuestionIndex(0);
    setResponses({});
    setCompletedResponse(null);
    setStartTime(Date.now());
  };

  // Handle response selection
  const handleResponseSelect = (questionId: string, value: number, label: string) => {
    setResponses(prev => ({
      ...prev,
      [questionId]: { response: value, response_text: label }
    }));
  };

  // Navigate questions
  const handleNext = () => {
    if (selectedQuestionnaire && currentQuestionIndex < selectedQuestionnaire.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  // Submit questionnaire
  const handleSubmit = async () => {
    if (!selectedQuestionnaire) return;

    const allQuestionsAnswered = selectedQuestionnaire.questions.every(q => responses[q.id]);
    if (!allQuestionsAnswered) {
      toast({
        title: "Incomplete Questionnaire",
        description: "Please answer all questions before submitting.",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);

    try {
      const durationSeconds = startTime ? Math.round((Date.now() - startTime) / 1000) : null;

      const formattedResponses = selectedQuestionnaire.questions.map(q => ({
        question_id: q.id,
        question_text: q.text,
        response: responses[q.id].response,
        response_text: responses[q.id].response_text,
      }));

      const result = await apiRequest<QuestionnaireResponse>("/api/v1/mental-health/submit", {
        method: "POST",
        body: JSON.stringify({
          questionnaire_type: selectedQuestionnaire.type,
          responses: formattedResponses,
          duration_seconds: durationSeconds,
          allow_storage: true,
          allow_clinical_sharing: true,
        }),
        headers: {
          "Content-Type": "application/json",
        },
      });

      setCompletedResponse(result);
      refetchHistory();
      
      toast({
        title: "Questionnaire Submitted",
        description: "Your responses have been recorded and analyzed.",
      });

    } catch (error: any) {
      toast({
        title: "Submission Failed",
        description: error.message || "Failed to submit questionnaire. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  // Reset and start new questionnaire
  const handleStartNew = () => {
    setSelectedQuestionnaire(null);
    setCurrentQuestionIndex(0);
    setResponses({});
    setCompletedResponse(null);
    setStartTime(null);
  };

  // Get severity color
  const getSeverityColor = (level: string) => {
    const colors: Record<string, string> = {
      minimal: "bg-green-500/10 text-green-700 dark:text-green-400",
      low: "bg-green-500/10 text-green-700 dark:text-green-400",
      mild: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
      moderate: "bg-orange-500/10 text-orange-700 dark:text-orange-400",
      moderately_severe: "bg-red-500/10 text-red-700 dark:text-red-400",
      severe: "bg-red-600/10 text-red-700 dark:text-red-400",
      high: "bg-red-500/10 text-red-700 dark:text-red-400",
    };
    return colors[level] || "bg-gray-500/10 text-gray-700 dark:text-gray-400";
  };

  const currentQuestion = selectedQuestionnaire?.questions[currentQuestionIndex];
  const progress = selectedQuestionnaire ? ((currentQuestionIndex + 1) / selectedQuestionnaire.questions.length) * 100 : 0;
  const isLastQuestion = selectedQuestionnaire && currentQuestionIndex === selectedQuestionnaire.questions.length - 1;
  const currentResponse = currentQuestion ? responses[currentQuestion.id] : null;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b p-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Brain className="w-8 h-8" />
          Mental Health Assessments
        </h1>
        <p className="text-muted-foreground mt-1">
          Track your mental wellness with standardized screening questionnaires
        </p>
        <Badge variant="outline" className="mt-2">
          Non-diagnostic screening tools only
        </Badge>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden p-6">
        <Tabs value={activeTab} onValueChange={(v: any) => setActiveTab(v)} className="h-full flex flex-col">
          <TabsList className="w-full max-w-md">
            <TabsTrigger value="questionnaires" className="flex-1" data-testid="tab-questionnaires">
              <ClipboardList className="w-4 h-4 mr-2" />
              Assessments
            </TabsTrigger>
            <TabsTrigger value="history" className="flex-1" data-testid="tab-history">
              <Calendar className="w-4 h-4 mr-2" />
              History
            </TabsTrigger>
          </TabsList>

          {/* Questionnaires Tab */}
          <TabsContent value="questionnaires" className="flex-1 mt-6 overflow-hidden flex flex-col">
            {!selectedQuestionnaire && !completedResponse && (
              <div className="grid gap-4 max-w-4xl">
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Important Disclaimer</AlertTitle>
                  <AlertDescription>
                    These are self-reported screening tools only and do not provide medical diagnosis. Always consult with a licensed healthcare provider for clinical evaluation and treatment.
                  </AlertDescription>
                </Alert>

                {questionnairesData?.questionnaires?.map((template: QuestionnaireTemplate) => (
                  <Card key={template.type} className="hover-elevate active-elevate-2 cursor-pointer" onClick={() => handleStartQuestionnaire(template)} data-testid={`card-questionnaire-${template.type}`}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="text-xl">{template.full_name}</CardTitle>
                          <CardDescription className="mt-1">{template.description}</CardDescription>
                        </div>
                        <Badge variant="outline">{template.questions.length} questions</Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="flex flex-wrap gap-2">
                        <Badge variant="secondary">
                          <Activity className="w-3 h-3 mr-1" />
                          {template.timeframe}
                        </Badge>
                        {template.public_domain && (
                          <Badge variant="secondary">Public Domain</Badge>
                        )}
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button className="w-full" data-testid={`button-start-${template.type}`}>
                        Start Assessment
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            )}

            {/* Active Questionnaire */}
            {selectedQuestionnaire && !completedResponse && (
              <div className="max-w-3xl mx-auto flex flex-col h-full">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>{selectedQuestionnaire.full_name}</CardTitle>
                        <CardDescription className="mt-1">
                          Question {currentQuestionIndex + 1} of {selectedQuestionnaire.questions.length}
                        </CardDescription>
                      </div>
                      <Button variant="ghost" size="sm" onClick={handleStartNew} data-testid="button-cancel">
                        Cancel
                      </Button>
                    </div>
                    <Progress value={progress} className="mt-4" />
                  </CardHeader>

                  <CardContent className="space-y-6">
                    <div className="space-y-4">
                      <p className="text-sm text-muted-foreground">{selectedQuestionnaire.instructions}</p>
                      
                      <div className="bg-muted/30 p-6 rounded-lg">
                        <p className="text-lg font-medium">{currentQuestion?.text}</p>
                        {currentQuestion?.crisis_flag && (
                          <Badge variant="outline" className="mt-2 bg-red-500/10">
                            <AlertTriangle className="w-3 h-3 mr-1" />
                            Important question
                          </Badge>
                        )}
                      </div>

                      <RadioGroup
                        value={currentResponse?.response?.toString()}
                        onValueChange={(val) => {
                          const option = selectedQuestionnaire.response_options.find(o => o.value.toString() === val);
                          if (option && currentQuestion) {
                            handleResponseSelect(currentQuestion.id, option.value, option.label);
                          }
                        }}
                        className="space-y-3"
                      >
                        {selectedQuestionnaire.response_options.map((option) => (
                          <div key={option.value} className="flex items-center space-x-3 p-3 rounded-md hover-elevate active-elevate-2" data-testid={`option-${option.value}`}>
                            <RadioGroupItem value={option.value.toString()} id={`option-${option.value}`} />
                            <Label htmlFor={`option-${option.value}`} className="flex-1 cursor-pointer font-normal">
                              {option.label}
                            </Label>
                          </div>
                        ))}
                      </RadioGroup>
                    </div>
                  </CardContent>

                  <CardFooter className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={handlePrevious}
                      disabled={currentQuestionIndex === 0}
                      data-testid="button-previous"
                    >
                      Previous
                    </Button>
                    {!isLastQuestion ? (
                      <Button
                        onClick={handleNext}
                        disabled={!currentResponse}
                        className="flex-1"
                        data-testid="button-next"
                      >
                        Next
                      </Button>
                    ) : (
                      <Button
                        onClick={handleSubmit}
                        disabled={!currentResponse || isSubmitting}
                        className="flex-1"
                        data-testid="button-submit"
                      >
                        {isSubmitting ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Submitting...
                          </>
                        ) : (
                          <>
                            <Check className="w-4 h-4 mr-2" />
                            Submit Assessment
                          </>
                        )}
                      </Button>
                    )}
                  </CardFooter>
                </Card>
              </div>
            )}

            {/* Results */}
            {completedResponse && (
              <div className="max-w-4xl mx-auto space-y-6">
                {/* Crisis Alert */}
                {completedResponse.crisis_intervention?.crisis_detected && (
                  <Alert variant="destructive" className="border-2">
                    <AlertTriangle className="h-5 w-5" />
                    <AlertTitle className="text-lg font-bold">Immediate Support Needed</AlertTitle>
                    <AlertDescription className="space-y-4 mt-2">
                      <p>{completedResponse.crisis_intervention.intervention_message}</p>
                      
                      <div className="space-y-2">
                        <p className="font-semibold">Crisis Resources:</p>
                        {completedResponse.crisis_intervention.crisis_hotlines.map((hotline, idx) => (
                          <div key={idx} className="bg-background p-3 rounded-md">
                            <p className="font-medium">{hotline.name}</p>
                            {hotline.phone && <p className="text-sm">Call: {hotline.phone}</p>}
                            {hotline.sms && <p className="text-sm">{hotline.sms}</p>}
                            <p className="text-sm text-muted-foreground">{hotline.description}</p>
                          </div>
                        ))}
                      </div>

                      <div className="space-y-2">
                        <p className="font-semibold">Next Steps:</p>
                        <ul className="list-disc list-inside space-y-1">
                          {completedResponse.crisis_intervention.next_steps.map((step, idx) => (
                            <li key={idx} className="text-sm">{step}</li>
                          ))}
                        </ul>
                      </div>
                    </AlertDescription>
                  </Alert>
                )}

                {/* Score Summary */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>Assessment Results</CardTitle>
                        <CardDescription>
                          Completed {format(new Date(), "PPP")}
                        </CardDescription>
                      </div>
                      <CheckCircle2 className="w-8 h-8 text-green-500" />
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Overall Score */}
                    <div className="flex items-center justify-between p-6 bg-muted/30 rounded-lg">
                      <div>
                        <p className="text-sm text-muted-foreground">Total Score</p>
                        <p className="text-4xl font-bold">{completedResponse.score.total_score}</p>
                        <p className="text-sm text-muted-foreground">out of {completedResponse.score.max_score}</p>
                      </div>
                      <Badge className={getSeverityColor(completedResponse.score.severity_level)}>
                        {completedResponse.score.severity_description}
                      </Badge>
                    </div>

                    {/* Neutral Summary */}
                    <div>
                      <h3 className="font-semibold mb-2">Summary</h3>
                      <p className="text-muted-foreground">{completedResponse.score.neutral_summary}</p>
                    </div>

                    {/* Key Observations */}
                    {completedResponse.score.key_observations && completedResponse.score.key_observations.length > 0 && (
                      <div>
                        <h3 className="font-semibold mb-2">Key Observations</h3>
                        <ul className="space-y-2">
                          {completedResponse.score.key_observations.map((obs, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2" />
                              <span className="text-sm text-muted-foreground">{obs}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Cluster Scores */}
                    {Object.keys(completedResponse.score.cluster_scores || {}).length > 0 && (
                      <div>
                        <h3 className="font-semibold mb-3">Symptom Areas</h3>
                        <div className="space-y-3">
                          {Object.entries(completedResponse.score.cluster_scores).map(([key, cluster]: [string, any]) => {
                            const percentage = (cluster.score / cluster.maxScore) * 100;
                            return (
                              <div key={key} className="space-y-1">
                                <div className="flex items-center justify-between text-sm">
                                  <span>{cluster.label}</span>
                                  <span className="text-muted-foreground">
                                    {cluster.score}/{cluster.maxScore}
                                  </span>
                                </div>
                                <Progress value={percentage} className="h-2" />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter className="flex gap-2">
                    <Button variant="outline" onClick={handleStartNew} data-testid="button-new-assessment">
                      New Assessment
                    </Button>
                    <Button variant="outline" onClick={() => setActiveTab("history")} data-testid="button-view-history">
                      <Calendar className="w-4 h-4 mr-2" />
                      View History
                    </Button>
                  </CardFooter>
                </Card>

                {/* Disclaimer */}
                <Alert>
                  <FileText className="h-4 w-4" />
                  <AlertTitle>Important Note</AlertTitle>
                  <AlertDescription>
                    This is a self-reported screening tool and does not constitute a medical diagnosis. These results should be reviewed by a qualified healthcare professional. Not to be used as the sole basis for treatment decisions.
                  </AlertDescription>
                </Alert>
              </div>
            )}
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history" className="flex-1 mt-6 overflow-hidden">
            <ScrollArea className="h-full">
              <div className="max-w-4xl space-y-4">
                {historyData?.history?.length === 0 ? (
                  <Card>
                    <CardContent className="py-12 text-center">
                      <ClipboardList className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                      <p className="text-muted-foreground">No assessment history yet</p>
                      <Button className="mt-4" onClick={() => setActiveTab("questionnaires")} data-testid="button-start-first">
                        Start Your First Assessment
                      </Button>
                    </CardContent>
                  </Card>
                ) : (
                  historyData?.history?.map((item: HistoryItem) => (
                    <Card key={item.response_id} data-testid={`history-item-${item.response_id}`}>
                      <CardHeader>
                        <div className="flex items-start justify-between">
                          <div>
                            <CardTitle className="text-lg">
                              {item.questionnaire_type === "PHQ9" && "PHQ-9 Depression Screening"}
                              {item.questionnaire_type === "GAD7" && "GAD-7 Anxiety Screening"}
                              {item.questionnaire_type === "PSS10" && "PSS-10 Stress Assessment"}
                            </CardTitle>
                            <CardDescription>
                              {format(new Date(item.completed_at), "PPP 'at' p")}
                            </CardDescription>
                          </div>
                          <div className="flex flex-col items-end gap-2">
                            <Badge className={getSeverityColor(item.severity_level)}>
                              {item.severity_level.replace("_", " ")}
                            </Badge>
                            {item.crisis_detected && (
                              <Badge variant="destructive">
                                <AlertTriangle className="w-3 h-3 mr-1" />
                                Crisis flag
                              </Badge>
                            )}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center gap-6 text-sm">
                          <div>
                            <p className="text-muted-foreground">Score</p>
                            <p className="font-semibold text-lg">
                              {item.total_score}/{item.max_score}
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))
                )}

                {/* Trends Summary */}
                {historyData?.trends && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Activity className="w-5 h-5" />
                        Trends
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-muted/30 rounded-lg">
                          <p className="text-sm text-muted-foreground">Overall Trend</p>
                          <div className="flex items-center gap-2 mt-1">
                            {historyData.trends.overall_trend.direction === "improving" && (
                              <TrendingDown className="w-5 h-5 text-green-500" />
                            )}
                            {historyData.trends.overall_trend.direction === "worsening" && (
                              <TrendingUp className="w-5 h-5 text-red-500" />
                            )}
                            <span className="font-semibold capitalize">{historyData.trends.overall_trend.direction}</span>
                          </div>
                        </div>
                        <div className="p-4 bg-muted/30 rounded-lg">
                          <p className="text-sm text-muted-foreground">Pattern</p>
                          <p className="font-semibold capitalize mt-1">{historyData.trends.variability.pattern}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
