import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  MessageSquare, 
  Loader2, 
  Plus,
  Calendar,
  Users,
  Database,
  CheckCircle2,
  Clock,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import { useFeedbackDatasets, useBuildFeedbackDataset, type FeedbackDataset } from '@/hooks/useMLAI';
import { useToast } from '@/hooks/use-toast';
import { format } from 'date-fns';

const FEEDBACK_TYPES = [
  { id: 'helpful', label: 'Helpful/Not Helpful' },
  { id: 'accuracy', label: 'Accuracy Ratings' },
  { id: 'relevance', label: 'Relevance Scores' },
  { id: 'correction', label: 'Corrections' },
  { id: 'preference', label: 'Preference Comparisons' },
];

export function FeedbackDatasetTab() {
  const { toast } = useToast();
  const [isCreating, setIsCreating] = useState(false);
  const [name, setName] = useState('');
  const [source, setSource] = useState<'patient' | 'doctor' | 'both'>('both');
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  
  const { data: datasets, isLoading, refetch } = useFeedbackDatasets();
  const buildMutation = useBuildFeedbackDataset();
  
  const handleBuild = async () => {
    if (!name.trim() || selectedTypes.length === 0) {
      toast({
        title: 'Missing information',
        description: 'Please provide a name and select at least one feedback type.',
        variant: 'destructive',
      });
      return;
    }
    
    try {
      await buildMutation.mutateAsync({
        name,
        source,
        feedback_types: selectedTypes,
      });
      toast({
        title: 'Dataset build started',
        description: 'Your feedback dataset is being built. This may take a few minutes.',
      });
      setIsCreating(false);
      setName('');
      setSelectedTypes([]);
      refetch();
    } catch {
      toast({
        title: 'Build failed',
        description: 'Unable to start dataset build. Please try again.',
        variant: 'destructive',
      });
    }
  };
  
  const toggleType = (typeId: string) => {
    setSelectedTypes(prev =>
      prev.includes(typeId)
        ? prev.filter(t => t !== typeId)
        : [...prev, typeId]
    );
  };
  
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'ready':
        return <Badge className="bg-green-500 gap-1"><CheckCircle2 className="h-3 w-3" />Ready</Badge>;
      case 'building':
        return <Badge className="bg-yellow-500 gap-1"><Clock className="h-3 w-3" />Building</Badge>;
      case 'failed':
        return <Badge variant="destructive" className="gap-1"><AlertTriangle className="h-3 w-3" />Failed</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };
  
  return (
    <div className="space-y-6" data-testid="feedback-dataset-tab">
      <Alert className="border-primary/50 bg-primary/5">
        <MessageSquare className="h-4 w-4 text-primary" />
        <AlertTitle className="font-semibold">RLHF Feedback Datasets</AlertTitle>
        <AlertDescription className="text-sm">
          Build datasets from patient and doctor feedback for Reinforcement Learning from Human Feedback.
          Use these datasets to fine-tune AI models based on real user interactions.
        </AlertDescription>
      </Alert>
      
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Feedback Datasets</h3>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => refetch()} size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button onClick={() => setIsCreating(true)} size="sm" data-testid="button-create-dataset">
            <Plus className="h-4 w-4 mr-2" />
            Build Dataset
          </Button>
        </div>
      </div>
      
      {isCreating && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5 text-primary" />
              Build New Feedback Dataset
            </CardTitle>
            <CardDescription>
              Configure the dataset parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Dataset Name</Label>
              <Input
                placeholder="e.g., Patient RLHF Q4 2025"
                value={name}
                onChange={(e) => setName(e.target.value)}
                data-testid="input-dataset-name"
              />
            </div>
            
            <div className="space-y-2">
              <Label>Feedback Source</Label>
              <Select value={source} onValueChange={(v) => setSource(v as typeof source)}>
                <SelectTrigger data-testid="select-feedback-source">
                  <SelectValue placeholder="Select source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="patient">Patient Feedback Only</SelectItem>
                  <SelectItem value="doctor">Doctor Feedback Only</SelectItem>
                  <SelectItem value="both">Both Patient & Doctor</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label>Feedback Types</Label>
              <div className="grid grid-cols-2 gap-3">
                {FEEDBACK_TYPES.map(type => (
                  <div key={type.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={type.id}
                      checked={selectedTypes.includes(type.id)}
                      onCheckedChange={() => toggleType(type.id)}
                    />
                    <label htmlFor={type.id} className="text-sm cursor-pointer">
                      {type.label}
                    </label>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
          <CardFooter className="flex gap-2">
            <Button variant="outline" onClick={() => setIsCreating(false)} className="flex-1">
              Cancel
            </Button>
            <Button
              onClick={handleBuild}
              disabled={buildMutation.isPending || !name.trim() || selectedTypes.length === 0}
              className="flex-1"
              data-testid="button-start-build"
            >
              {buildMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Starting...
                </>
              ) : (
                'Start Build'
              )}
            </Button>
          </CardFooter>
        </Card>
      )}
      
      {isLoading ? (
        <Card>
          <CardContent className="py-8">
            <div className="space-y-4">
              {[1, 2, 3].map(i => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      ) : datasets && datasets.length > 0 ? (
        <Card data-testid="card-datasets-list">
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead className="text-right">Records</TableHead>
                  <TableHead>Feedback Types</TableHead>
                  <TableHead>Date Range</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {datasets.map((dataset) => (
                  <TableRow key={dataset.id}>
                    <TableCell className="font-medium">{dataset.name}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="gap-1">
                        <Users className="h-3 w-3" />
                        {dataset.source}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {dataset.record_count.toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {dataset.feedback_types.slice(0, 2).map((type, i) => (
                          <Badge key={i} variant="secondary" className="text-xs">
                            {type}
                          </Badge>
                        ))}
                        {dataset.feedback_types.length > 2 && (
                          <Badge variant="secondary" className="text-xs">
                            +{dataset.feedback_types.length - 2}
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {dataset.date_range.start} - {dataset.date_range.end}
                      </div>
                    </TableCell>
                    <TableCell>{getStatusBadge(dataset.status)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="py-12 text-center">
            <Database className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
            <h3 className="text-lg font-semibold mb-2">No Feedback Datasets</h3>
            <p className="text-muted-foreground mb-4">
              Build your first dataset to start collecting RLHF training data.
            </p>
            <Button onClick={() => setIsCreating(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Build Dataset
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
