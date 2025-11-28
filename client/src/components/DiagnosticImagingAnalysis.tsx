import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { 
  Image,
  FileImage,
  Scan,
  Brain,
  Loader2,
  AlertTriangle,
  CheckCircle,
  Clock,
  FileText,
  Eye,
  Activity,
  Stethoscope,
  RefreshCw
} from "lucide-react";
import { format, parseISO } from "date-fns";

interface PatientContext {
  id: string;
  firstName: string;
  lastName: string;
  allergies?: string[];
  comorbidities?: string[];
  currentMedications?: string[];
}

interface ImagingAnalysis {
  id: string;
  imageType: string;
  studyDescription?: string;
  clinicalContext?: string;
  sections: {
    technique: string;
    findings: string;
    impression: string;
    recommendations: string;
  };
  fullReport: string;
  confidence: number;
  aiModel: string;
  analyzedAt: string;
  disclaimer: string;
  _fallback?: boolean;
}

interface ImagingStudy {
  id: string;
  type: string;
  date: string;
  status: string;
  findings: string;
  radiologist: string;
  priority: string;
}

interface DiagnosticImagingAnalysisProps {
  patientContext: PatientContext;
  className?: string;
}

const imageTypes = [
  { value: 'xray', label: 'X-Ray' },
  { value: 'ct', label: 'CT Scan' },
  { value: 'mri', label: 'MRI' },
  { value: 'ultrasound', label: 'Ultrasound' },
  { value: 'pet', label: 'PET Scan' },
  { value: 'mammogram', label: 'Mammogram' },
  { value: 'echocardiogram', label: 'Echocardiogram' },
  { value: 'fluoroscopy', label: 'Fluoroscopy' }
];

export function DiagnosticImagingAnalysis({ patientContext, className }: DiagnosticImagingAnalysisProps) {
  const [selectedImageType, setSelectedImageType] = useState<string>('');
  const [studyDescription, setStudyDescription] = useState('');
  const [clinicalContext, setClinicalContext] = useState('');
  const [currentAnalysis, setCurrentAnalysis] = useState<ImagingAnalysis | null>(null);
  const { toast } = useToast();

  const { data: imagingHistory, isLoading: historyLoading, refetch: refetchHistory } = useQuery({
    queryKey: ['/api/v1/lysa/imaging-history', patientContext.id],
    queryFn: async () => {
      const response = await fetch(`/api/v1/lysa/imaging-history/${patientContext.id}`);
      if (!response.ok) {
        return { studies: [], totalCount: 0 };
      }
      return response.json();
    },
    enabled: !!patientContext.id
  });

  const analyzeImageMutation = useMutation({
    mutationFn: async (data: { imageType: string; studyDescription?: string; clinicalContext?: string }) => {
      const response = await apiRequest('/api/v1/lysa/imaging-analysis', {
        method: 'POST',
        body: JSON.stringify({
          patientId: patientContext.id,
          ...data
        })
      });
      return response;
    },
    onSuccess: (data: any) => {
      setCurrentAnalysis(data.analysis);
      toast({
        title: "Analysis Complete",
        description: data.analysis._fallback 
          ? "AI analysis temporarily unavailable - showing preliminary report"
          : "AI-powered imaging analysis generated successfully"
      });
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to analyze imaging study",
        variant: "destructive"
      });
    }
  });

  const handleAnalyze = () => {
    if (!selectedImageType) {
      toast({
        title: "Select Image Type",
        description: "Please select the type of imaging study to analyze",
        variant: "destructive"
      });
      return;
    }

    analyzeImageMutation.mutate({
      imageType: imageTypes.find(t => t.value === selectedImageType)?.label || selectedImageType,
      studyDescription,
      clinicalContext
    });
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="secondary" className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"><CheckCircle className="h-3 w-3 mr-1" />Completed</Badge>;
      case 'pending':
        return <Badge variant="secondary" className="bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"><Clock className="h-3 w-3 mr-1" />Pending</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case 'stat':
        return <Badge variant="destructive">STAT</Badge>;
      case 'urgent':
        return <Badge variant="secondary" className="bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400">Urgent</Badge>;
      default:
        return <Badge variant="outline">Routine</Badge>;
    }
  };

  return (
    <div className={className}>
      <Tabs defaultValue="analyze" className="space-y-4">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="analyze" data-testid="tab-imaging-analyze">
            <Scan className="h-4 w-4 mr-2" />
            New Analysis
          </TabsTrigger>
          <TabsTrigger value="history" data-testid="tab-imaging-history">
            <FileImage className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="analyze" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                  <Brain className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <CardTitle className="text-lg">AI-Powered Imaging Analysis</CardTitle>
                  <CardDescription>
                    Request AI interpretation of diagnostic imaging studies for {patientContext.firstName} {patientContext.lastName}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="image-type">Imaging Type *</Label>
                  <Select value={selectedImageType} onValueChange={setSelectedImageType}>
                    <SelectTrigger id="image-type" data-testid="select-image-type">
                      <SelectValue placeholder="Select imaging type" />
                    </SelectTrigger>
                    <SelectContent>
                      {imageTypes.map(type => (
                        <SelectItem key={type.value} value={type.value}>{type.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="study-description">Study Description</Label>
                  <Input
                    id="study-description"
                    placeholder="e.g., PA and Lateral chest"
                    value={studyDescription}
                    onChange={(e) => setStudyDescription(e.target.value)}
                    data-testid="input-study-description"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="clinical-context">Clinical Context</Label>
                <Textarea
                  id="clinical-context"
                  placeholder="Enter relevant clinical history, symptoms, or reason for study..."
                  value={clinicalContext}
                  onChange={(e) => setClinicalContext(e.target.value)}
                  className="min-h-[100px]"
                  data-testid="input-clinical-context"
                />
              </div>

              <Button 
                onClick={handleAnalyze}
                disabled={analyzeImageMutation.isPending || !selectedImageType}
                className="w-full"
                data-testid="button-analyze-image"
              >
                {analyzeImageMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Scan className="h-4 w-4 mr-2" />
                    Request AI Analysis
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {currentAnalysis && (
            <Card className="border-primary/50">
              <CardHeader>
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-primary" />
                    <CardTitle className="text-lg">{currentAnalysis.imageType} Analysis Report</CardTitle>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      AI Model: {currentAnalysis.aiModel}
                    </Badge>
                    <Badge 
                      variant="secondary" 
                      className={currentAnalysis.confidence > 0.7 
                        ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                        : "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                      }
                    >
                      Confidence: {Math.round(currentAnalysis.confidence * 100)}%
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {currentAnalysis._fallback && (
                  <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
                    <div className="flex items-center gap-2 text-yellow-700 dark:text-yellow-400">
                      <AlertTriangle className="h-4 w-4" />
                      <span className="font-medium text-sm">AI Analysis Temporarily Unavailable</span>
                    </div>
                    <p className="text-xs text-yellow-600 dark:text-yellow-500 mt-1">
                      Showing preliminary report. Please have study reviewed by qualified radiologist.
                    </p>
                  </div>
                )}

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <h4 className="font-semibold text-sm flex items-center gap-1">
                      <Activity className="h-4 w-4" /> Technique
                    </h4>
                    <p className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-md">
                      {currentAnalysis.sections.technique || 'Not specified'}
                    </p>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold text-sm flex items-center gap-1">
                      <Stethoscope className="h-4 w-4" /> Impression
                    </h4>
                    <p className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-md">
                      {currentAnalysis.sections.impression || 'Pending review'}
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="font-semibold text-sm flex items-center gap-1">
                    <Eye className="h-4 w-4" /> Findings
                  </h4>
                  <div className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-md whitespace-pre-wrap">
                    {currentAnalysis.sections.findings || 'No findings documented'}
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="font-semibold text-sm flex items-center gap-1">
                    <CheckCircle className="h-4 w-4" /> Recommendations
                  </h4>
                  <p className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-md">
                    {currentAnalysis.sections.recommendations || 'No specific recommendations'}
                  </p>
                </div>

                <div className="border-t pt-3">
                  <p className="text-xs text-muted-foreground italic">
                    {currentAnalysis.disclaimer}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Analyzed: {format(parseISO(currentAnalysis.analyzedAt), 'MMM d, yyyy h:mm a')}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <CardTitle className="text-lg">Imaging History</CardTitle>
                  <CardDescription>
                    Previous imaging studies for {patientContext.firstName} {patientContext.lastName}
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={() => refetchHistory()} data-testid="button-refresh-imaging">
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {historyLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map(i => (
                    <Skeleton key={i} className="h-24 w-full" />
                  ))}
                </div>
              ) : imagingHistory?.studies?.length > 0 ? (
                <ScrollArea className="h-[400px]">
                  <div className="space-y-3">
                    {imagingHistory.studies.map((study: ImagingStudy) => (
                      <Card key={study.id} className="hover-elevate">
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between gap-2 flex-wrap">
                            <div className="flex items-center gap-3">
                              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                                <Image className="h-5 w-5 text-primary" />
                              </div>
                              <div>
                                <p className="font-medium">{study.type}</p>
                                <p className="text-sm text-muted-foreground">
                                  {format(parseISO(study.date), 'MMM d, yyyy')}
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              {getPriorityBadge(study.priority)}
                              {getStatusBadge(study.status)}
                            </div>
                          </div>
                          <div className="mt-3 pl-13">
                            <p className="text-sm text-muted-foreground">
                              <span className="font-medium">Findings:</span> {study.findings}
                            </p>
                            <p className="text-xs text-muted-foreground mt-1">
                              Interpreted by: {study.radiologist}
                            </p>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <FileImage className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No imaging studies found</p>
                  <p className="text-sm">Request a new analysis using the "New Analysis" tab</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
