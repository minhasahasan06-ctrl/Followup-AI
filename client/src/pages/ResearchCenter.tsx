import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TrendingUp, BarChart, FileText, Bot, Beaker } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { AIResearchReport } from "@shared/schema";

export default function ResearchCenter() {
  const { toast } = useToast();
  const [reportTitle, setReportTitle] = useState("");
  const [analysisType, setAnalysisType] = useState<string>("");

  const { data: reports, isLoading } = useQuery<AIResearchReport[]>({
    queryKey: ["/api/doctor/research-reports"],
  });

  const generateReportMutation = useMutation({
    mutationFn: async (data: { title: string; analysisType: string }) => {
      return await apiRequest("POST", "/api/doctor/research-reports", data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/doctor/research-reports"] });
      toast({
        title: "Research report generated",
        description: "Your AI-powered research analysis is ready",
      });
      setReportTitle("");
      setAnalysisType("");
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to generate report",
        variant: "destructive",
      });
    },
  });

  const handleGenerateReport = () => {
    if (!reportTitle.trim() || !analysisType) {
      toast({
        title: "Missing information",
        description: "Please provide a title and select an analysis type",
        variant: "destructive",
      });
      return;
    }

    generateReportMutation.mutate({
      title: reportTitle,
      analysisType,
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold mb-2">Epidemiology Research Center</h1>
        <p className="text-muted-foreground">
          AI-powered research analysis and data visualization for immunocompromised patient populations
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader>
            <BarChart className="h-8 w-8 text-primary mb-2" />
            <CardTitle className="text-base">Total Reports</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{reports?.length || 0}</p>
            <p className="text-xs text-muted-foreground mt-1">Generated research analyses</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <TrendingUp className="h-8 w-8 text-primary mb-2" />
            <CardTitle className="text-base">Patient Cohort</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">150</p>
            <p className="text-xs text-muted-foreground mt-1">Active participants in research</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <Beaker className="h-8 w-8 text-primary mb-2" />
            <CardTitle className="text-base">Active Studies</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">3</p>
            <p className="text-xs text-muted-foreground mt-1">Ongoing research projects</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="generate" className="space-y-6">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="generate" data-testid="tab-generate">
            <Bot className="h-4 w-4 mr-2" />
            Generate Report
          </TabsTrigger>
          <TabsTrigger value="reports" data-testid="tab-reports">
            <FileText className="h-4 w-4 mr-2" />
            View Reports
          </TabsTrigger>
        </TabsList>

        <TabsContent value="generate">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-accent">
                  <Bot className="h-5 w-5" />
                </div>
                <div>
                  <CardTitle>AI Research Agent</CardTitle>
                  <CardDescription>
                    Generate comprehensive epidemiological research reports using AI analysis
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="title">Report Title</Label>
                  <Input
                    id="title"
                    placeholder="e.g., Correlation between medication adherence and vital stability"
                    value={reportTitle}
                    onChange={(e) => setReportTitle(e.target.value)}
                    data-testid="input-report-title"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="analysis-type">Analysis Type</Label>
                  <Select value={analysisType} onValueChange={setAnalysisType}>
                    <SelectTrigger id="analysis-type" data-testid="select-analysis-type">
                      <SelectValue placeholder="Select analysis type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="correlation">Correlation Analysis</SelectItem>
                      <SelectItem value="regression">Regression Analysis</SelectItem>
                      <SelectItem value="survival">Survival Analysis</SelectItem>
                      <SelectItem value="pattern">Pattern Recognition</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Card className="bg-muted/50">
                  <CardContent className="p-4">
                    <h4 className="font-medium mb-2 text-sm">Data Sources</h4>
                    <div className="space-y-1 text-sm text-muted-foreground">
                      <p>• Daily follow-up assessments (150 patients)</p>
                      <p>• Medication adherence records</p>
                      <p>• Vital signs and wearable data</p>
                      <p>• Behavioral insights and sentiment analysis</p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Button
                onClick={handleGenerateReport}
                disabled={generateReportMutation.isPending}
                className="w-full"
                data-testid="button-generate-report"
              >
                <Bot className="h-4 w-4 mr-2" />
                {generateReportMutation.isPending ? "Generating Report..." : "Generate Research Report"}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reports">
          <Card>
            <CardHeader>
              <CardTitle>Generated Research Reports</CardTitle>
              <CardDescription>View and manage your AI-generated research analyses</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <Card key={i} className="animate-pulse">
                      <CardContent className="p-4">
                        <div className="space-y-2">
                          <div className="h-4 bg-muted rounded w-3/4" />
                          <div className="h-3 bg-muted rounded w-1/2" />
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : reports && reports.length > 0 ? (
                <ScrollArea className="h-[600px] pr-4">
                  <div className="space-y-4">
                    {reports.map((report) => (
                      <Card key={report.id} className="hover-elevate" data-testid={`card-report-${report.id}`}>
                        <CardContent className="p-6">
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex-1">
                              <h3 className="font-semibold mb-1" data-testid={`text-report-title-${report.id}`}>
                                {report.title}
                              </h3>
                              <p className="text-sm text-muted-foreground line-clamp-2">{report.summary}</p>
                            </div>
                            <div className="flex gap-2 flex-shrink-0 ml-4">
                              <Badge variant="secondary">{report.analysisType}</Badge>
                              {report.publishReady && <Badge variant="secondary">Ready to Publish</Badge>}
                            </div>
                          </div>

                          <div className="grid grid-cols-2 gap-3 mb-4">
                            <div className="text-sm">
                              <span className="text-muted-foreground">Cohort Size: </span>
                              <span className="font-medium">{report.patientCohortSize} patients</span>
                            </div>
                            <div className="text-sm">
                              <span className="text-muted-foreground">Date: </span>
                              <span className="font-medium">
                                {new Date(report.createdAt).toLocaleDateString()}
                              </span>
                            </div>
                          </div>

                          {report.findings && report.findings.length > 0 && (
                            <div className="space-y-2 mt-4">
                              <p className="text-sm font-medium">Key Findings:</p>
                              {report.findings.slice(0, 2).map((finding, idx) => (
                                <Card key={idx} className="bg-muted/50">
                                  <CardContent className="p-3">
                                    <div className="flex items-start gap-2">
                                      <TrendingUp className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                                      <div className="text-sm">
                                        <p className="font-medium mb-1">{finding.finding}</p>
                                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                          <span>Significance: {finding.significance}</span>
                                          <span>•</span>
                                          <span>Confidence: {(finding.confidence * 100).toFixed(0)}%</span>
                                        </div>
                                      </div>
                                    </div>
                                  </CardContent>
                                </Card>
                              ))}
                            </div>
                          )}

                          <Button
                            variant="outline"
                            className="w-full mt-4"
                            data-testid={`button-view-report-${report.id}`}
                          >
                            View Full Report
                          </Button>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No research reports generated yet</p>
                  <p className="text-sm mt-1">Use the AI Research Agent to create your first report</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
