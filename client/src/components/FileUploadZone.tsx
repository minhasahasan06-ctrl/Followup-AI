import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, FileText, Image, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadedFile {
  id: string;
  name: string;
  type: string;
  size: string;
}

export function FileUploadZone() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileAdd = () => {
    const mockFile: UploadedFile = {
      id: Date.now().toString(),
      name: "lab-results-2024.pdf",
      type: "application/pdf",
      size: "2.4 MB",
    };
    setFiles([...files, mockFile]);
    console.log("File uploaded:", mockFile);
  };

  const handleRemove = (id: string) => {
    setFiles(files.filter(f => f.id !== id));
    console.log("File removed:", id);
  };

  return (
    <div className="space-y-4">
      <Card
        className={cn(
          "border-2 border-dashed p-8 text-center transition-colors cursor-pointer hover-elevate",
          isDragging && "border-primary bg-primary/5"
        )}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFileAdd();
        }}
        onClick={handleFileAdd}
        data-testid="dropzone-upload"
      >
        <Upload className="h-10 w-10 mx-auto mb-4 text-muted-foreground" />
        <p className="text-sm font-medium mb-1">
          Drop files here or click to upload
        </p>
        <p className="text-xs text-muted-foreground">
          Supports: Medical imaging (DICOM, JPEG), Lab reports (PDF), Prescriptions
        </p>
      </Card>

      {files.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Uploaded Files ({files.length})</p>
          <div className="grid gap-2">
            {files.map((file) => (
              <Card key={file.id} className="p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    {file.type.includes("pdf") ? (
                      <FileText className="h-5 w-5 text-destructive flex-shrink-0" />
                    ) : (
                      <Image className="h-5 w-5 text-primary flex-shrink-0" />
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium truncate">{file.name}</p>
                      <p className="text-xs text-muted-foreground">{file.size}</p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 flex-shrink-0"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemove(file.id);
                    }}
                    data-testid={`button-remove-file-${file.id}`}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
