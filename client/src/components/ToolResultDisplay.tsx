import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  FileText,
  Activity,
  Pill,
  AlertTriangle,
  Calendar,
  Beaker,
  Image as ImageIcon,
  ChevronDown,
  ChevronUp,
  User,
  Heart,
  Thermometer,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  ExternalLink,
  Download,
  Loader2
} from "lucide-react";
import { cn } from "@/lib/utils";
import { format } from "date-fns";

interface ToolResult {
  id: string;
  toolName: string;
  status: "pending" | "running" | "completed" | "failed" | "pending_approval";
  result?: Record<string, unknown>;
  error?: string;
  executionTimeMs?: number;
  createdAt?: string;
}

interface ToolResultDisplayProps {
  result: ToolResult;
  onExpand?: (resultId: string) => void;
  className?: string;
}

export function ToolResultDisplay({ result, onExpand, className }: ToolResultDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getToolIcon = (toolName: string) => {
    switch (toolName) {
      case "ehr_fetch":
        return <FileText className="h-4 w-4" />;
      case "lab_fetch":
        return <Beaker className="h-4 w-4" />;
      case "imaging_linker":
        return <ImageIcon className="h-4 w-4" />;
      case "prescription_draft":
        return <Pill className="h-4 w-4" />;
      case "calendar":
        return <Calendar className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const getToolDisplayName = (toolName: string) => {
    const names: Record<string, string> = {
      ehr_fetch: "Patient Health Record",
      lab_fetch: "Laboratory Results",
      imaging_linker: "Medical Imaging",
      prescription_draft: "Prescription Draft",
      calendar: "Appointment",
      messaging: "Secure Message"
    };
    return names[toolName] || toolName;
  };

  const getStatusBadge = () => {
    switch (result.status) {
      case "completed":
        return (
          <Badge variant="outline" className="gap-1 text-green-600 border-green-600" data-testid={`badge-status-completed-${result.id}`}>
            <CheckCircle className="h-3 w-3" />
            Completed
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="outline" className="gap-1 text-red-600 border-red-600" data-testid={`badge-status-failed-${result.id}`}>
            <XCircle className="h-3 w-3" />
            Failed
          </Badge>
        );
      case "pending_approval":
        return (
          <Badge variant="outline" className="gap-1 text-yellow-600 border-yellow-600" data-testid={`badge-status-pending-approval-${result.id}`}>
            <AlertCircle className="h-3 w-3" />
            Pending Approval
          </Badge>
        );
      case "running":
        return (
          <Badge variant="outline" className="gap-1 text-blue-600 border-blue-600" data-testid={`badge-status-running-${result.id}`}>
            <Loader2 className="h-3 w-3 animate-spin" />
            Running
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="gap-1" data-testid={`badge-status-pending-${result.id}`}>
            <Clock className="h-3 w-3" />
            Pending
          </Badge>
        );
    }
  };

  const renderContent = () => {
    if (result.error) {
      return (
        <div className="p-3 bg-red-50 dark:bg-red-950/30 rounded border border-red-200 dark:border-red-900" data-testid={`error-display-${result.id}`}>
          <p className="text-sm text-red-600 dark:text-red-400" data-testid={`text-error-${result.id}`}>{result.error}</p>
        </div>
      );
    }

    if (!result.result) {
      return null;
    }

    switch (result.toolName) {
      case "ehr_fetch":
        return <EHRResultDisplay data={result.result} />;
      case "lab_fetch":
        return <LabResultDisplay data={result.result} />;
      case "imaging_linker":
        return <ImagingResultDisplay data={result.result} />;
      case "prescription_draft":
        return <PrescriptionResultDisplay data={result.result} />;
      case "calendar":
        return <CalendarResultDisplay data={result.result} />;
      default:
        return <GenericResultDisplay data={result.result} />;
    }
  };

  return (
    <Collapsible 
      open={isExpanded} 
      onOpenChange={(open) => {
        setIsExpanded(open);
        if (open && onExpand) onExpand(result.id);
      }}
    >
      <Card className={cn("w-full", className)} data-testid={`tool-result-${result.id}`}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover-elevate py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10 text-primary">
                  {getToolIcon(result.toolName)}
                </div>
                <div>
                  <CardTitle className="text-sm font-medium">
                    {getToolDisplayName(result.toolName)}
                  </CardTitle>
                  {result.executionTimeMs && (
                    <CardDescription className="text-xs">
                      Completed in {result.executionTimeMs}ms
                    </CardDescription>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                {getStatusBadge()}
                {isExpanded ? (
                  <ChevronUp className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                )}
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="pt-0">
            <Separator className="mb-4" />
            {renderContent()}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}

function EHRResultDisplay({ data }: { data: Record<string, unknown> }) {
  const patient = data.patient as Record<string, unknown> | undefined;
  const vitals = data.vitals as Record<string, unknown>[] | undefined;
  const diagnoses = data.diagnoses as Record<string, unknown>[] | undefined;
  const allergies = data.allergies as string[] | undefined;
  const medications = data.medications as Record<string, unknown>[] | undefined;

  return (
    <div className="space-y-4" data-testid="ehr-result-content">
      {patient && (
        <div className="space-y-2" data-testid="ehr-patient-info">
          <h4 className="font-medium text-sm flex items-center gap-2">
            <User className="h-4 w-4" />
            Patient Information
          </h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Name:</span>
              <span data-testid="text-patient-name">{patient.name as string}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Age:</span>
              <span data-testid="text-patient-age">{patient.age as string}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Gender:</span>
              <span data-testid="text-patient-gender">{patient.gender as string}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">MRN:</span>
              <span data-testid="text-patient-mrn">{patient.mrn as string}</span>
            </div>
          </div>
        </div>
      )}

      {vitals && vitals.length > 0 && (
        <div className="space-y-2" data-testid="ehr-vitals-section">
          <h4 className="font-medium text-sm flex items-center gap-2">
            <Heart className="h-4 w-4" />
            Recent Vitals
          </h4>
          <div className="grid grid-cols-3 gap-2">
            {vitals.slice(0, 6).map((vital, i) => (
              <div key={i} className="p-2 rounded bg-muted text-sm" data-testid={`vital-${i}`}>
                <div className="text-muted-foreground text-xs">{vital.name as string}</div>
                <div className="font-medium" data-testid={`vital-value-${i}`}>{vital.value as string} {vital.unit as string}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {diagnoses && diagnoses.length > 0 && (
        <div className="space-y-2" data-testid="ehr-diagnoses-section">
          <h4 className="font-medium text-sm">Active Diagnoses</h4>
          <div className="flex flex-wrap gap-1">
            {diagnoses.map((dx, i) => (
              <Badge key={i} variant="outline" className="text-xs" data-testid={`diagnosis-${i}`}>
                {dx.code as string}: {dx.description as string}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {allergies && allergies.length > 0 && (
        <div className="space-y-2" data-testid="ehr-allergies-section">
          <h4 className="font-medium text-sm flex items-center gap-2 text-red-600">
            <AlertTriangle className="h-4 w-4" />
            Allergies
          </h4>
          <div className="flex flex-wrap gap-1">
            {allergies.map((allergy, i) => (
              <Badge key={i} variant="destructive" className="text-xs" data-testid={`allergy-${i}`}>
                {allergy}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {medications && medications.length > 0 && (
        <div className="space-y-2" data-testid="ehr-medications-section">
          <h4 className="font-medium text-sm flex items-center gap-2">
            <Pill className="h-4 w-4" />
            Current Medications
          </h4>
          <ScrollArea className="h-32">
            <div className="space-y-1">
              {medications.map((med, i) => (
                <div key={i} className="text-sm p-2 rounded bg-muted" data-testid={`medication-${i}`}>
                  <span className="font-medium" data-testid={`medication-name-${i}`}>{med.name as string}</span>
                  <span className="text-muted-foreground ml-2" data-testid={`medication-dosage-${i}`}>{med.dosage as string}</span>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>
      )}
    </div>
  );
}

function LabResultDisplay({ data }: { data: Record<string, unknown> }) {
  const labs = data.results as Record<string, unknown>[] | undefined;
  const abnormalCount = labs?.filter(l => l.isAbnormal).length || 0;

  return (
    <div className="space-y-4" data-testid="lab-result-content">
      {abnormalCount > 0 && (
        <div className="p-3 bg-yellow-50 dark:bg-yellow-950/30 rounded border border-yellow-200 dark:border-yellow-900">
          <div className="flex items-center gap-2 text-yellow-600 dark:text-yellow-400">
            <AlertTriangle className="h-4 w-4" />
            <span className="font-medium text-sm">{abnormalCount} abnormal result(s) detected</span>
          </div>
        </div>
      )}

      {labs && labs.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 font-medium">Test</th>
                <th className="text-right py-2 font-medium">Result</th>
                <th className="text-right py-2 font-medium">Reference</th>
                <th className="text-center py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {labs.map((lab, i) => (
                <tr 
                  key={i} 
                  className={cn(
                    "border-b",
                    lab.isAbnormal && "bg-yellow-50 dark:bg-yellow-950/20"
                  )}
                  data-testid={`lab-row-${i}`}
                >
                  <td className="py-2" data-testid={`text-lab-name-${i}`}>{lab.testName as string}</td>
                  <td className="text-right py-2 font-mono" data-testid={`text-lab-value-${i}`}>
                    {lab.value as string} {lab.unit as string}
                  </td>
                  <td className="text-right py-2 text-muted-foreground" data-testid={`text-lab-reference-${i}`}>
                    {lab.referenceRange as string}
                  </td>
                  <td className="text-center py-2" data-testid={`text-lab-status-${i}`}>
                    {lab.isAbnormal ? (
                      <Badge variant="destructive" className="text-xs" data-testid={`badge-abnormal-${i}`}>Abnormal</Badge>
                    ) : (
                      <Badge variant="outline" className="text-xs text-green-600" data-testid={`badge-normal-${i}`}>Normal</Badge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">No laboratory results available</p>
      )}

      {data.collectedAt && (
        <p className="text-xs text-muted-foreground">
          Collected: {format(new Date(data.collectedAt as string), "PPp")}
        </p>
      )}
    </div>
  );
}

function ImagingResultDisplay({ data }: { data: Record<string, unknown> }) {
  const studies = data.studies as Record<string, unknown>[] | undefined;

  return (
    <div className="space-y-4" data-testid="imaging-result-content">
      {studies && studies.length > 0 ? (
        <div className="space-y-3" data-testid="imaging-studies-list">
          {studies.map((study, i) => (
            <Card key={i} data-testid={`imaging-study-${i}`}>
              <CardContent className="p-3">
                <div className="flex items-start justify-between">
                  <div>
                    <h5 className="font-medium text-sm" data-testid={`text-modality-${i}`}>{study.modality as string}</h5>
                    <p className="text-xs text-muted-foreground" data-testid={`text-body-part-${i}`}>{study.bodyPart as string}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {study.hasReport && (
                      <Button variant="ghost" size="sm" className="h-7 gap-1" data-testid={`button-report-${i}`}>
                        <FileText className="h-3 w-3" />
                        Report
                      </Button>
                    )}
                    <Button variant="outline" size="sm" className="h-7 gap-1" data-testid={`button-view-study-${i}`}>
                      <ExternalLink className="h-3 w-3" />
                      View
                    </Button>
                  </div>
                </div>
                {study.findings && (
                  <p className="text-sm mt-2 p-2 bg-muted rounded" data-testid={`text-findings-${i}`}>
                    {study.findings as string}
                  </p>
                )}
                <p className="text-xs text-muted-foreground mt-2" data-testid={`text-study-date-${i}`}>
                  {format(new Date(study.studyDate as string), "PPp")}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground" data-testid="text-no-imaging">No imaging studies available</p>
      )}
    </div>
  );
}

function PrescriptionResultDisplay({ data }: { data: Record<string, unknown> }) {
  const prescription = data.prescription as Record<string, unknown> | undefined;
  const interactions = data.interactions as Record<string, unknown>[] | undefined;

  return (
    <div className="space-y-4" data-testid="prescription-result-content">
      {prescription && (
        <div className="p-4 border rounded-lg" data-testid="prescription-details">
          <div className="flex items-center gap-2 mb-3">
            <Pill className="h-5 w-5 text-primary" />
            <h4 className="font-medium" data-testid="text-rx-drug-name">{prescription.drugName as string}</h4>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-muted-foreground">Dosage:</span>
              <span className="ml-2 font-medium" data-testid="text-rx-dosage">{prescription.dosage as string}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Frequency:</span>
              <span className="ml-2 font-medium" data-testid="text-rx-frequency">{prescription.frequency as string}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Duration:</span>
              <span className="ml-2 font-medium" data-testid="text-rx-duration">{prescription.duration as string}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Refills:</span>
              <span className="ml-2 font-medium" data-testid="text-rx-refills">{prescription.refills as number}</span>
            </div>
          </div>
          {prescription.instructions && (
            <div className="mt-3 p-2 bg-muted rounded text-sm" data-testid="text-rx-instructions">
              <strong>Instructions:</strong> {prescription.instructions as string}
            </div>
          )}
        </div>
      )}

      {interactions && interactions.length > 0 && (
        <div className="p-3 bg-red-50 dark:bg-red-950/30 rounded border border-red-200 dark:border-red-900" data-testid="rx-drug-interactions">
          <h5 className="font-medium text-sm text-red-600 dark:text-red-400 flex items-center gap-2 mb-2">
            <AlertTriangle className="h-4 w-4" />
            Drug Interactions Detected
          </h5>
          <ul className="space-y-1">
            {interactions.map((int, i) => (
              <li key={i} className="text-sm text-red-600 dark:text-red-400" data-testid={`text-rx-interaction-${i}`}>
                {int.drug1 as string} + {int.drug2 as string}: {int.severity as string}
              </li>
            ))}
          </ul>
        </div>
      )}

      {data.status === "pending_approval" && (
        <div className="p-3 bg-yellow-50 dark:bg-yellow-950/30 rounded border border-yellow-200 dark:border-yellow-900" data-testid="alert-pending-approval">
          <p className="text-sm text-yellow-600 dark:text-yellow-400">
            This prescription requires doctor approval before it can be sent to the pharmacy.
          </p>
        </div>
      )}
    </div>
  );
}

function CalendarResultDisplay({ data }: { data: Record<string, unknown> }) {
  const appointment = data.appointment as Record<string, unknown> | undefined;
  const availability = data.availability as Record<string, unknown>[] | undefined;

  return (
    <div className="space-y-4" data-testid="calendar-result-content">
      {appointment && (
        <div className="p-4 border rounded-lg" data-testid="appointment-details">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-medium flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              <span data-testid="text-appointment-title">{appointment.title as string}</span>
            </h4>
            <Badge data-testid="badge-appointment-status">{appointment.status as string}</Badge>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span data-testid="text-appointment-time">{format(new Date(appointment.startTime as string), "PPpp")}</span>
            </div>
            <div className="flex items-center gap-2">
              <User className="h-4 w-4 text-muted-foreground" />
              <span data-testid="text-appointment-participant">{appointment.patientName as string || appointment.doctorName as string}</span>
            </div>
          </div>
        </div>
      )}

      {availability && availability.length > 0 && (
        <div data-testid="availability-section">
          <h4 className="font-medium text-sm mb-2">Available Slots</h4>
          <div className="grid grid-cols-2 gap-2">
            {availability.slice(0, 6).map((slot, i) => (
              <Button key={i} variant="outline" size="sm" className="justify-start gap-2" data-testid={`button-slot-${i}`}>
                <Clock className="h-3 w-3" />
                {format(new Date(slot.startTime as string), "MMM d, HH:mm")}
              </Button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function GenericResultDisplay({ data }: { data: Record<string, unknown> }) {
  return (
    <div className="p-4 bg-muted rounded" data-testid="generic-result-content">
      <pre className="text-xs overflow-auto whitespace-pre-wrap">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

export default ToolResultDisplay;
