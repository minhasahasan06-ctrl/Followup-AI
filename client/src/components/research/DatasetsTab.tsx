import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Database, Search, GitBranch, History, Eye, Download, 
  ArrowRight, Clock, User, FileText, Layers, ChevronRight,
  RefreshCw, Filter
} from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

interface Dataset {
  id: string;
  name: string;
  description?: string;
  version: string;
  rowCount: number;
  columnCount?: number;
  phiLevel: 'none' | 'limited' | 'full';
  createdAt: string;
  createdBy: string;
  parentDatasetId?: string;
  studyId?: string;
  cohortId?: string;
}

interface DatasetVersion {
  version: string;
  rowCount: number;
  createdAt: string;
  createdBy: string;
  changeDescription?: string;
}

interface LineageNode {
  id: string;
  name: string;
  type: 'cohort' | 'dataset' | 'export' | 'study';
  createdAt: string;
}

interface LineageEdge {
  source: string;
  target: string;
  operation: string;
}

const PHI_LEVEL_CONFIG = {
  none: { label: 'De-identified', color: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100' },
  limited: { label: 'Limited PHI', color: 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-100' },
  full: { label: 'Full PHI', color: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100' },
};

export function DatasetsTab() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const { data: datasets = [], isLoading, refetch } = useQuery<Dataset[]>({
    queryKey: ['/python-api/v1/research-center/datasets'],
  });

  const { data: versions = [] } = useQuery<DatasetVersion[]>({
    queryKey: ['/python-api/v1/research-center/datasets', selectedDataset?.id, 'versions'],
    queryFn: async () => {
      if (!selectedDataset?.id) return [];
      const res = await fetch(`/python-api/v1/research-center/datasets/${selectedDataset.id}/versions`);
      if (!res.ok) throw new Error('Failed to fetch versions');
      return res.json();
    },
    enabled: !!selectedDataset,
  });

  const { data: lineage } = useQuery<{ nodes: LineageNode[]; edges: LineageEdge[] }>({
    queryKey: ['/python-api/v1/research-center/datasets', selectedDataset?.id, 'lineage'],
    queryFn: async () => {
      if (!selectedDataset?.id) return { nodes: [], edges: [] };
      const res = await fetch(`/python-api/v1/research-center/datasets/${selectedDataset.id}/lineage`);
      if (!res.ok) throw new Error('Failed to fetch lineage');
      return res.json();
    },
    enabled: !!selectedDataset,
  });

  const filteredDatasets = datasets.filter(ds => 
    ds.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    ds.description?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleViewDetails = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setDetailsOpen(true);
  };

  const totalRows = datasets.reduce((sum, ds) => sum + (ds.rowCount || 0), 0);
  const phiDatasets = datasets.filter(ds => ds.phiLevel !== 'none').length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold" data-testid="text-datasets-title">Research Datasets</h2>
          <p className="text-muted-foreground">Browse datasets with version history and lineage tracking</p>
        </div>
        <Button variant="outline" size="icon" onClick={() => refetch()} data-testid="button-refresh-datasets">
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card data-testid="card-total-datasets">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Database className="h-4 w-4 text-primary" />
              Total Datasets
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{datasets.length}</div>
          </CardContent>
        </Card>
        <Card data-testid="card-total-rows">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Layers className="h-4 w-4 text-blue-500" />
              Total Rows
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalRows.toLocaleString()}</div>
          </CardContent>
        </Card>
        <Card data-testid="card-versions">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <History className="h-4 w-4 text-purple-500" />
              Versions Tracked
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{datasets.reduce((sum, ds) => sum + parseInt(ds.version || '1'), 0)}</div>
          </CardContent>
        </Card>
        <Card data-testid="card-phi-datasets">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4 text-amber-500" />
              PHI Datasets
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{phiDatasets}</div>
          </CardContent>
        </Card>
      </div>

      <Card data-testid="card-datasets-list">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Dataset Browser</CardTitle>
              <CardDescription>Explore and manage research datasets</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input 
                  placeholder="Search datasets..." 
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-9 w-64"
                  data-testid="input-search-datasets"
                />
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-20 w-full" />)}
            </div>
          ) : filteredDatasets.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>{searchTerm ? 'No datasets match your search.' : 'No datasets available yet.'}</p>
            </div>
          ) : (
            <ScrollArea className="h-[500px]">
              <div className="space-y-3">
                {filteredDatasets.map((dataset) => (
                  <div 
                    key={dataset.id} 
                    className="p-4 border rounded-lg hover-elevate cursor-pointer transition-all"
                    onClick={() => handleViewDetails(dataset)}
                    data-testid={`dataset-card-${dataset.id}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-lg bg-primary/10">
                          <Database className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{dataset.name}</span>
                            <Badge variant="outline" className="text-xs">v{dataset.version}</Badge>
                            <Badge className={PHI_LEVEL_CONFIG[dataset.phiLevel].color}>
                              {PHI_LEVEL_CONFIG[dataset.phiLevel].label}
                            </Badge>
                          </div>
                          {dataset.description && (
                            <p className="text-sm text-muted-foreground mt-1 line-clamp-1">{dataset.description}</p>
                          )}
                          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Layers className="h-3 w-3" />
                              {dataset.rowCount?.toLocaleString() || 0} rows
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatDistanceToNow(new Date(dataset.createdAt), { addSuffix: true })}
                            </span>
                            <span className="flex items-center gap-1">
                              <User className="h-3 w-3" />
                              {dataset.createdBy}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button variant="ghost" size="icon" data-testid={`button-view-${dataset.id}`}>
                          <Eye className="h-4 w-4" />
                        </Button>
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              {selectedDataset?.name}
              <Badge variant="outline">v{selectedDataset?.version}</Badge>
            </DialogTitle>
          </DialogHeader>

          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
              <TabsTrigger value="versions" data-testid="tab-versions">Version History</TabsTrigger>
              <TabsTrigger value="lineage" data-testid="tab-lineage">Data Lineage</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="mt-4">
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 border rounded-lg">
                    <div className="text-sm text-muted-foreground">Row Count</div>
                    <div className="text-lg font-medium">{selectedDataset?.rowCount?.toLocaleString()}</div>
                  </div>
                  <div className="p-3 border rounded-lg">
                    <div className="text-sm text-muted-foreground">PHI Level</div>
                    <Badge className={selectedDataset ? PHI_LEVEL_CONFIG[selectedDataset.phiLevel].color : ''}>
                      {selectedDataset ? PHI_LEVEL_CONFIG[selectedDataset.phiLevel].label : ''}
                    </Badge>
                  </div>
                  <div className="p-3 border rounded-lg">
                    <div className="text-sm text-muted-foreground">Created</div>
                    <div className="text-sm">{selectedDataset && format(new Date(selectedDataset.createdAt), 'MMM d, yyyy h:mm a')}</div>
                  </div>
                  <div className="p-3 border rounded-lg">
                    <div className="text-sm text-muted-foreground">Created By</div>
                    <div className="text-sm">{selectedDataset?.createdBy}</div>
                  </div>
                </div>
                {selectedDataset?.description && (
                  <div className="p-3 border rounded-lg">
                    <div className="text-sm text-muted-foreground mb-1">Description</div>
                    <p className="text-sm">{selectedDataset.description}</p>
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="versions" className="mt-4">
              <ScrollArea className="h-[300px]">
                {versions.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <History className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No version history available</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {versions.map((v, idx) => (
                      <div key={v.version} className="flex items-start gap-3 p-3 border rounded-lg" data-testid={`version-${v.version}`}>
                        <div className={`p-2 rounded-full ${idx === 0 ? 'bg-primary text-primary-foreground' : 'bg-muted'}`}>
                          <GitBranch className="h-4 w-4" />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">Version {v.version}</span>
                            {idx === 0 && <Badge>Current</Badge>}
                          </div>
                          <div className="text-sm text-muted-foreground mt-1">
                            {v.rowCount.toLocaleString()} rows • {format(new Date(v.createdAt), 'MMM d, yyyy')}
                          </div>
                          {v.changeDescription && (
                            <p className="text-sm mt-1">{v.changeDescription}</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </TabsContent>

            <TabsContent value="lineage" className="mt-4">
              <div className="border rounded-lg p-4 min-h-[300px]">
                {lineage && lineage.nodes.length > 0 ? (
                  <div className="space-y-4">
                    <div className="text-sm text-muted-foreground mb-4">Data provenance and transformation chain</div>
                    <div className="flex items-center gap-2 flex-wrap">
                      {lineage.nodes.map((node, idx) => (
                        <div key={node.id} className="flex items-center gap-2">
                          <div className={`p-3 border rounded-lg ${
                            node.type === 'cohort' ? 'bg-blue-50 dark:bg-blue-950 border-blue-200' :
                            node.type === 'dataset' ? 'bg-green-50 dark:bg-green-950 border-green-200' :
                            node.type === 'export' ? 'bg-amber-50 dark:bg-amber-950 border-amber-200' :
                            'bg-purple-50 dark:bg-purple-950 border-purple-200'
                          }`}>
                            <div className="text-xs font-medium uppercase text-muted-foreground">{node.type}</div>
                            <div className="font-medium">{node.name}</div>
                          </div>
                          {idx < lineage.nodes.length - 1 && (
                            <ArrowRight className="h-5 w-5 text-muted-foreground" />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <GitBranch className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No lineage information available</p>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>
    </div>
  );
}
