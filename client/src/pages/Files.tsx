import { FileUploadZone } from "@/components/FileUploadZone";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { FileText, Image, FlaskConical, Lock } from "lucide-react";

export default function Files() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold mb-2">Medical Files</h1>
        <p className="text-muted-foreground">
          Upload and manage your medical documents, lab results, and imaging files
        </p>
      </div>

      <div className="flex items-center gap-2 p-3 rounded-md bg-muted/50 border">
        <Lock className="h-4 w-4 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">
          All files are encrypted and HIPAA compliant. Your data is secure.
        </p>
      </div>

      <Tabs defaultValue="upload" className="space-y-6">
        <TabsList>
          <TabsTrigger value="upload" data-testid="tab-upload">
            Upload Files
          </TabsTrigger>
          <TabsTrigger value="documents" data-testid="tab-documents">
            Documents
          </TabsTrigger>
          <TabsTrigger value="imaging" data-testid="tab-imaging">
            Medical Imaging
          </TabsTrigger>
          <TabsTrigger value="lab-results" data-testid="tab-lab-results">
            Lab Results
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>Upload Medical Files</CardTitle>
            </CardHeader>
            <CardContent>
              <FileUploadZone />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="documents">
          <Card>
            <CardHeader>
              <CardTitle>Medical Documents</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No documents uploaded yet</p>
                <p className="text-sm mt-1">Upload your prescriptions, medical reports, and other documents</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="imaging">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Medical Imaging
                <Badge variant="secondary">AI Analysis Available</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                <Image className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No imaging files uploaded yet</p>
                <p className="text-sm mt-1">Upload X-rays, CT scans, MRIs, or ultrasounds for AI-powered analysis</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="lab-results">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Laboratory Results
                <Badge variant="secondary">AI Interpretation</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                <FlaskConical className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No lab results uploaded yet</p>
                <p className="text-sm mt-1">Upload blood work, urinalysis, and other diagnostic test results</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
