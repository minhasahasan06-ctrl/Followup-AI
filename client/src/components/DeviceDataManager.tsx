import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { queryClient, apiRequest } from '@/lib/queryClient';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { useToast } from '@/hooks/use-toast';
import {
  Activity,
  Battery,
  Brain,
  Check,
  Clock,
  Droplets,
  Heart,
  Loader2,
  Moon,
  Plus,
  RefreshCw,
  Scale,
  Stethoscope,
  Thermometer,
  TrendingDown,
  TrendingUp,
  Watch,
  Wind,
  Zap,
  Smartphone,
  AlertTriangle,
  ChevronRight,
} from 'lucide-react';
import { format } from 'date-fns';

interface DeviceReading {
  id: string;
  deviceType: string;
  deviceBrand?: string;
  source: string;
  recordedAt: string;
  [key: string]: any;
}

const DEVICE_TYPES = [
  { id: 'bp_monitor', name: 'Blood Pressure Monitor', icon: Heart, color: 'text-rose-500', bgColor: 'bg-rose-500/10' },
  { id: 'glucose_meter', name: 'Glucose Meter', icon: Droplets, color: 'text-blue-500', bgColor: 'bg-blue-500/10' },
  { id: 'smart_scale', name: 'Smart Scale', icon: Scale, color: 'text-emerald-500', bgColor: 'bg-emerald-500/10' },
  { id: 'thermometer', name: 'Thermometer', icon: Thermometer, color: 'text-orange-500', bgColor: 'bg-orange-500/10' },
  { id: 'stethoscope', name: 'Smart Stethoscope', icon: Stethoscope, color: 'text-purple-500', bgColor: 'bg-purple-500/10' },
  { id: 'smartwatch', name: 'Smartwatch', icon: Watch, color: 'text-cyan-500', bgColor: 'bg-cyan-500/10' },
];

const SMARTWATCH_METRIC_CATEGORIES = [
  {
    id: 'heart',
    name: 'Heart & Cardiovascular',
    icon: Heart,
    metrics: ['heartRate', 'restingHeartRate', 'hrv', 'hrvSdnn', 'afibDetected', 'irregularRhythmAlert'],
  },
  {
    id: 'oxygen',
    name: 'Blood Oxygen & Respiratory',
    icon: Wind,
    metrics: ['spo2', 'spo2Min', 'respiratoryRate'],
  },
  {
    id: 'sleep',
    name: 'Sleep',
    icon: Moon,
    metrics: ['sleepDuration', 'sleepDeepMinutes', 'sleepRemMinutes', 'sleepLightMinutes', 'sleepAwakeMinutes', 'sleepScore', 'sleepEfficiency', 'sleepConsistency', 'sleepDebt', 'sleepNeed'],
  },
  {
    id: 'recovery',
    name: 'Recovery & Readiness',
    icon: Battery,
    metrics: ['recoveryScore', 'readinessScore', 'bodyBattery', 'strainScore', 'stressScore'],
  },
  {
    id: 'activity',
    name: 'Activity & Fitness',
    icon: Activity,
    metrics: ['steps', 'activeMinutes', 'caloriesBurned', 'distanceMeters', 'floorsClimbed', 'standingHours', 'vo2Max', 'trainingLoad', 'trainingStatus', 'trainingReadiness', 'fitnessAge'],
  },
  {
    id: 'performance',
    name: 'Performance & Running',
    icon: Zap,
    metrics: ['lactateThreshold', 'performanceCondition', 'runningDynamics'],
  },
  {
    id: 'temperature',
    name: 'Temperature',
    icon: Thermometer,
    metrics: ['skinTemperature'],
  },
  {
    id: 'womens_health',
    name: "Women's Health",
    icon: Heart,
    metrics: ['cycleDay', 'cyclePhase', 'predictedOvulation', 'periodLogged'],
  },
  {
    id: 'safety',
    name: 'Safety & Emergency',
    icon: AlertTriangle,
    metrics: ['fallDetected', 'emergencySOSTriggered'],
  },
];

function DeviceCard({ 
  device, 
  latestReading,
  onAddReading 
}: { 
  device: typeof DEVICE_TYPES[0]; 
  latestReading?: DeviceReading;
  onAddReading: () => void;
}) {
  const Icon = device.icon;
  
  const getValueDisplay = () => {
    if (!latestReading) return null;
    
    switch (device.id) {
      case 'bp_monitor':
        return latestReading.bpSystolic && latestReading.bpDiastolic 
          ? `${latestReading.bpSystolic}/${latestReading.bpDiastolic} mmHg` 
          : null;
      case 'glucose_meter':
        return latestReading.glucoseValue 
          ? `${latestReading.glucoseValue} ${latestReading.glucoseUnit || 'mg/dL'}` 
          : null;
      case 'smart_scale':
        return latestReading.weight 
          ? `${latestReading.weight} ${latestReading.weightUnit || 'kg'}` 
          : null;
      case 'thermometer':
        return latestReading.temperature 
          ? `${latestReading.temperature}°${latestReading.temperatureUnit || 'F'}` 
          : null;
      case 'stethoscope':
        return latestReading.heartSoundsAnalysis 
          ? 'Recording available' 
          : null;
      case 'smartwatch':
        const metrics = [];
        if (latestReading.heartRate) metrics.push(`${latestReading.heartRate} bpm`);
        if (latestReading.spo2) metrics.push(`${latestReading.spo2}% SpO2`);
        if (latestReading.steps) metrics.push(`${latestReading.steps.toLocaleString()} steps`);
        return metrics.length > 0 ? metrics.join(' • ') : null;
      default:
        return null;
    }
  };

  return (
    <Card className="hover-elevate transition-all">
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg ${device.bgColor} flex items-center justify-center`}>
              <Icon className={`h-5 w-5 ${device.color}`} />
            </div>
            <div>
              <h3 className="font-medium text-sm">{device.name}</h3>
              {latestReading ? (
                <div className="space-y-1">
                  <p className="text-lg font-semibold">{getValueDisplay() || 'No data'}</p>
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {format(new Date(latestReading.recordedAt), 'MMM d, h:mm a')}
                  </p>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">No readings yet</p>
              )}
            </div>
          </div>
          <Button 
            variant="outline" 
            size="sm"
            onClick={onAddReading}
            data-testid={`button-add-${device.id}-reading`}
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
        {latestReading?.source === 'auto_sync' && (
          <Badge variant="secondary" className="mt-2 text-xs">
            <RefreshCw className="h-3 w-3 mr-1" />
            Auto-synced
          </Badge>
        )}
      </CardContent>
    </Card>
  );
}

function BPMonitorForm({ onSubmit, isLoading }: { onSubmit: (data: any) => void; isLoading: boolean }) {
  const [formData, setFormData] = useState({
    bpSystolic: '',
    bpDiastolic: '',
    bpPulse: '',
    bpIrregularHeartbeat: false,
    bpBodyPosition: 'seated',
    bpArmUsed: 'left',
    notes: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      deviceType: 'bp_monitor',
      bpSystolic: parseInt(formData.bpSystolic),
      bpDiastolic: parseInt(formData.bpDiastolic),
      bpPulse: formData.bpPulse ? parseInt(formData.bpPulse) : undefined,
      bpIrregularHeartbeat: formData.bpIrregularHeartbeat,
      bpBodyPosition: formData.bpBodyPosition,
      bpArmUsed: formData.bpArmUsed,
      notes: formData.notes || undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="bp-systolic">Systolic (mmHg) *</Label>
          <Input
            id="bp-systolic"
            type="number"
            placeholder="120"
            value={formData.bpSystolic}
            onChange={(e) => setFormData({ ...formData, bpSystolic: e.target.value })}
            required
            min={60}
            max={250}
            data-testid="input-bp-systolic"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="bp-diastolic">Diastolic (mmHg) *</Label>
          <Input
            id="bp-diastolic"
            type="number"
            placeholder="80"
            value={formData.bpDiastolic}
            onChange={(e) => setFormData({ ...formData, bpDiastolic: e.target.value })}
            required
            min={40}
            max={150}
            data-testid="input-bp-diastolic"
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="bp-pulse">Pulse (bpm)</Label>
          <Input
            id="bp-pulse"
            type="number"
            placeholder="72"
            value={formData.bpPulse}
            onChange={(e) => setFormData({ ...formData, bpPulse: e.target.value })}
            min={30}
            max={200}
            data-testid="input-bp-pulse"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="bp-arm">Arm Used</Label>
          <Select value={formData.bpArmUsed} onValueChange={(v) => setFormData({ ...formData, bpArmUsed: v })}>
            <SelectTrigger data-testid="select-bp-arm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="left">Left Arm</SelectItem>
              <SelectItem value="right">Right Arm</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Switch
          id="bp-irregular"
          checked={formData.bpIrregularHeartbeat}
          onCheckedChange={(checked) => setFormData({ ...formData, bpIrregularHeartbeat: checked })}
          data-testid="switch-bp-irregular"
        />
        <Label htmlFor="bp-irregular">Irregular Heartbeat Detected</Label>
      </div>
      <div className="space-y-2">
        <Label htmlFor="bp-notes">Notes</Label>
        <Textarea
          id="bp-notes"
          placeholder="Any additional observations..."
          value={formData.notes}
          onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
          data-testid="textarea-bp-notes"
        />
      </div>
      <Button type="submit" className="w-full" disabled={isLoading} data-testid="button-submit-bp">
        {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Check className="h-4 w-4 mr-2" />}
        Save Reading
      </Button>
    </form>
  );
}

function GlucoseMeterForm({ onSubmit, isLoading }: { onSubmit: (data: any) => void; isLoading: boolean }) {
  const [formData, setFormData] = useState({
    glucoseValue: '',
    glucoseContext: 'fasting',
    glucoseUnit: 'mg/dL',
    notes: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      deviceType: 'glucose_meter',
      glucoseValue: parseFloat(formData.glucoseValue),
      glucoseContext: formData.glucoseContext,
      glucoseUnit: formData.glucoseUnit,
      notes: formData.notes || undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="glucose-value">Blood Glucose *</Label>
          <Input
            id="glucose-value"
            type="number"
            step="0.1"
            placeholder="100"
            value={formData.glucoseValue}
            onChange={(e) => setFormData({ ...formData, glucoseValue: e.target.value })}
            required
            data-testid="input-glucose-value"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="glucose-unit">Unit</Label>
          <Select value={formData.glucoseUnit} onValueChange={(v) => setFormData({ ...formData, glucoseUnit: v })}>
            <SelectTrigger data-testid="select-glucose-unit">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="mg/dL">mg/dL</SelectItem>
              <SelectItem value="mmol/L">mmol/L</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="space-y-2">
        <Label htmlFor="glucose-context">Measurement Context</Label>
        <Select value={formData.glucoseContext} onValueChange={(v) => setFormData({ ...formData, glucoseContext: v })}>
          <SelectTrigger data-testid="select-glucose-context">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="fasting">Fasting</SelectItem>
            <SelectItem value="before_meal">Before Meal</SelectItem>
            <SelectItem value="after_meal">After Meal (1-2 hrs)</SelectItem>
            <SelectItem value="bedtime">Bedtime</SelectItem>
            <SelectItem value="random">Random</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label htmlFor="glucose-notes">Notes</Label>
        <Textarea
          id="glucose-notes"
          placeholder="What did you eat? Any symptoms?"
          value={formData.notes}
          onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
          data-testid="textarea-glucose-notes"
        />
      </div>
      <Button type="submit" className="w-full" disabled={isLoading} data-testid="button-submit-glucose">
        {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Check className="h-4 w-4 mr-2" />}
        Save Reading
      </Button>
    </form>
  );
}

function SmartScaleForm({ onSubmit, isLoading }: { onSubmit: (data: any) => void; isLoading: boolean }) {
  const [formData, setFormData] = useState({
    weight: '',
    weightUnit: 'kg',
    bmi: '',
    bodyFatPercentage: '',
    muscleMass: '',
    boneMass: '',
    waterPercentage: '',
    visceralFat: '',
    metabolicAge: '',
    notes: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      deviceType: 'smart_scale',
      weight: parseFloat(formData.weight),
      weightUnit: formData.weightUnit,
      bmi: formData.bmi ? parseFloat(formData.bmi) : undefined,
      bodyFatPercentage: formData.bodyFatPercentage ? parseFloat(formData.bodyFatPercentage) : undefined,
      muscleMass: formData.muscleMass ? parseFloat(formData.muscleMass) : undefined,
      boneMass: formData.boneMass ? parseFloat(formData.boneMass) : undefined,
      waterPercentage: formData.waterPercentage ? parseFloat(formData.waterPercentage) : undefined,
      visceralFat: formData.visceralFat ? parseInt(formData.visceralFat) : undefined,
      metabolicAge: formData.metabolicAge ? parseInt(formData.metabolicAge) : undefined,
      notes: formData.notes || undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="scale-weight">Weight *</Label>
          <Input
            id="scale-weight"
            type="number"
            step="0.1"
            placeholder="70.5"
            value={formData.weight}
            onChange={(e) => setFormData({ ...formData, weight: e.target.value })}
            required
            data-testid="input-scale-weight"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="scale-unit">Unit</Label>
          <Select value={formData.weightUnit} onValueChange={(v) => setFormData({ ...formData, weightUnit: v })}>
            <SelectTrigger data-testid="select-scale-unit">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="kg">kg</SelectItem>
              <SelectItem value="lbs">lbs</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="scale-body-fat">Body Fat %</Label>
          <Input
            id="scale-body-fat"
            type="number"
            step="0.1"
            placeholder="22.5"
            value={formData.bodyFatPercentage}
            onChange={(e) => setFormData({ ...formData, bodyFatPercentage: e.target.value })}
            data-testid="input-scale-body-fat"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="scale-muscle">Muscle Mass (kg)</Label>
          <Input
            id="scale-muscle"
            type="number"
            step="0.1"
            placeholder="32.0"
            value={formData.muscleMass}
            onChange={(e) => setFormData({ ...formData, muscleMass: e.target.value })}
            data-testid="input-scale-muscle"
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="scale-bmi">BMI</Label>
          <Input
            id="scale-bmi"
            type="number"
            step="0.1"
            placeholder="24.5"
            value={formData.bmi}
            onChange={(e) => setFormData({ ...formData, bmi: e.target.value })}
            data-testid="input-scale-bmi"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="scale-bone">Bone Mass (kg)</Label>
          <Input
            id="scale-bone"
            type="number"
            step="0.1"
            placeholder="2.8"
            value={formData.boneMass}
            onChange={(e) => setFormData({ ...formData, boneMass: e.target.value })}
            data-testid="input-scale-bone"
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="scale-water">Water %</Label>
          <Input
            id="scale-water"
            type="number"
            step="0.1"
            placeholder="55.0"
            value={formData.waterPercentage}
            onChange={(e) => setFormData({ ...formData, waterPercentage: e.target.value })}
            data-testid="input-scale-water"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="scale-visceral">Visceral Fat (1-30)</Label>
          <Input
            id="scale-visceral"
            type="number"
            placeholder="8"
            value={formData.visceralFat}
            onChange={(e) => setFormData({ ...formData, visceralFat: e.target.value })}
            min={1}
            max={30}
            data-testid="input-scale-visceral"
          />
        </div>
      </div>
      <div className="space-y-2">
        <Label htmlFor="scale-metabolic-age">Metabolic Age</Label>
        <Input
          id="scale-metabolic-age"
          type="number"
          placeholder="35"
          value={formData.metabolicAge}
          onChange={(e) => setFormData({ ...formData, metabolicAge: e.target.value })}
          data-testid="input-scale-metabolic-age"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="scale-notes">Notes</Label>
        <Textarea
          id="scale-notes"
          placeholder="Morning weigh-in, after workout, etc."
          value={formData.notes}
          onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
          data-testid="textarea-scale-notes"
        />
      </div>
      <Button type="submit" className="w-full" disabled={isLoading} data-testid="button-submit-scale">
        {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Check className="h-4 w-4 mr-2" />}
        Save Reading
      </Button>
    </form>
  );
}

function ThermometerForm({ onSubmit, isLoading }: { onSubmit: (data: any) => void; isLoading: boolean }) {
  const [formData, setFormData] = useState({
    temperature: '',
    temperatureUnit: 'F',
    temperatureLocation: 'oral',
    notes: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      deviceType: 'thermometer',
      temperature: parseFloat(formData.temperature),
      temperatureUnit: formData.temperatureUnit,
      temperatureLocation: formData.temperatureLocation,
      notes: formData.notes || undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="therm-temp">Temperature *</Label>
          <Input
            id="therm-temp"
            type="number"
            step="0.1"
            placeholder="98.6"
            value={formData.temperature}
            onChange={(e) => setFormData({ ...formData, temperature: e.target.value })}
            required
            data-testid="input-therm-temp"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="therm-unit">Unit</Label>
          <Select value={formData.temperatureUnit} onValueChange={(v) => setFormData({ ...formData, temperatureUnit: v })}>
            <SelectTrigger data-testid="select-therm-unit">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="F">Fahrenheit (°F)</SelectItem>
              <SelectItem value="C">Celsius (°C)</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="space-y-2">
        <Label htmlFor="therm-location">Measurement Location</Label>
        <Select value={formData.temperatureLocation} onValueChange={(v) => setFormData({ ...formData, temperatureLocation: v })}>
          <SelectTrigger data-testid="select-therm-location">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="oral">Oral</SelectItem>
            <SelectItem value="forehead">Forehead</SelectItem>
            <SelectItem value="ear">Ear (Tympanic)</SelectItem>
            <SelectItem value="underarm">Underarm (Axillary)</SelectItem>
            <SelectItem value="rectal">Rectal</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label htmlFor="therm-notes">Notes</Label>
        <Textarea
          id="therm-notes"
          placeholder="Any symptoms, time of day, etc."
          value={formData.notes}
          onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
          data-testid="textarea-therm-notes"
        />
      </div>
      <Button type="submit" className="w-full" disabled={isLoading} data-testid="button-submit-therm">
        {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Check className="h-4 w-4 mr-2" />}
        Save Reading
      </Button>
    </form>
  );
}

function StethoscopeForm({ onSubmit, isLoading }: { onSubmit: (data: any) => void; isLoading: boolean }) {
  const [formData, setFormData] = useState({
    stethoscopeLocation: 'aortic',
    stethoscopeAudioUrl: '',
    heartSoundsAnalysis: '',
    lungSoundsAnalysis: '',
    notes: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      deviceType: 'stethoscope',
      stethoscopeLocation: formData.stethoscopeLocation,
      stethoscopeAudioUrl: formData.stethoscopeAudioUrl || undefined,
      heartSoundsAnalysis: formData.heartSoundsAnalysis || undefined,
      lungSoundsAnalysis: formData.lungSoundsAnalysis || undefined,
      notes: formData.notes || undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="steth-location">Auscultation Location</Label>
        <Select value={formData.stethoscopeLocation} onValueChange={(v) => setFormData({ ...formData, stethoscopeLocation: v })}>
          <SelectTrigger data-testid="select-steth-location">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="aortic">Aortic (Heart)</SelectItem>
            <SelectItem value="pulmonic">Pulmonic (Heart)</SelectItem>
            <SelectItem value="tricuspid">Tricuspid (Heart)</SelectItem>
            <SelectItem value="mitral">Mitral (Heart)</SelectItem>
            <SelectItem value="lung_upper_left">Upper Left Lung</SelectItem>
            <SelectItem value="lung_upper_right">Upper Right Lung</SelectItem>
            <SelectItem value="lung_lower_left">Lower Left Lung</SelectItem>
            <SelectItem value="lung_lower_right">Lower Right Lung</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label htmlFor="steth-audio-url">Audio Recording URL</Label>
        <Input
          id="steth-audio-url"
          type="url"
          placeholder="https://s3.amazonaws.com/.../recording.wav"
          value={formData.stethoscopeAudioUrl}
          onChange={(e) => setFormData({ ...formData, stethoscopeAudioUrl: e.target.value })}
          data-testid="input-steth-audio-url"
        />
        <p className="text-xs text-muted-foreground">Paste URL from your smart stethoscope app</p>
      </div>
      <div className="space-y-2">
        <Label htmlFor="steth-heart-analysis">Heart Sounds Analysis</Label>
        <Textarea
          id="steth-heart-analysis"
          placeholder="S1, S2 normal. No murmurs, rubs, or gallops..."
          value={formData.heartSoundsAnalysis}
          onChange={(e) => setFormData({ ...formData, heartSoundsAnalysis: e.target.value })}
          data-testid="textarea-steth-heart-analysis"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="steth-lung-analysis">Lung Sounds Analysis</Label>
        <Textarea
          id="steth-lung-analysis"
          placeholder="Clear breath sounds bilaterally. No wheezes, rales, or rhonchi..."
          value={formData.lungSoundsAnalysis}
          onChange={(e) => setFormData({ ...formData, lungSoundsAnalysis: e.target.value })}
          data-testid="textarea-steth-lung-analysis"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="steth-notes">Additional Notes</Label>
        <Textarea
          id="steth-notes"
          placeholder="Any additional observations..."
          value={formData.notes}
          onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
          data-testid="textarea-steth-notes"
        />
      </div>
      <Button type="submit" className="w-full" disabled={isLoading} data-testid="button-submit-steth">
        {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Check className="h-4 w-4 mr-2" />}
        Save Recording
      </Button>
    </form>
  );
}

function SmartwatchForm({ onSubmit, isLoading }: { onSubmit: (data: any) => void; isLoading: boolean }) {
  const [formData, setFormData] = useState({
    deviceBrand: 'apple',
    heartRate: '',
    restingHeartRate: '',
    hrv: '',
    hrvSdnn: '',
    afibDetected: false,
    irregularRhythmAlert: false,
    spo2: '',
    spo2Min: '',
    respiratoryRate: '',
    sleepDuration: '',
    sleepDeepMinutes: '',
    sleepRemMinutes: '',
    sleepLightMinutes: '',
    sleepAwakeMinutes: '',
    sleepScore: '',
    sleepEfficiency: '',
    sleepConsistency: '',
    sleepDebt: '',
    sleepNeed: '',
    recoveryScore: '',
    readinessScore: '',
    bodyBattery: '',
    strainScore: '',
    stressScore: '',
    steps: '',
    activeMinutes: '',
    caloriesBurned: '',
    distanceMeters: '',
    floorsClimbed: '',
    standingHours: '',
    vo2Max: '',
    trainingLoad: '',
    trainingStatus: '',
    trainingReadiness: '',
    fitnessAge: '',
    lactateThreshold: '',
    performanceCondition: '',
    skinTemperature: '',
    cycleDay: '',
    cyclePhase: '',
    predictedOvulation: '',
    periodLogged: false,
    runningDynamics: '',
    fallDetected: false,
    emergencySOSTriggered: false,
    notes: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const data: any = {
      deviceType: 'smartwatch',
      deviceBrand: formData.deviceBrand,
      notes: formData.notes || undefined,
    };
    
    if (formData.heartRate) data.heartRate = parseInt(formData.heartRate);
    if (formData.restingHeartRate) data.restingHeartRate = parseInt(formData.restingHeartRate);
    if (formData.hrv) data.hrv = parseInt(formData.hrv);
    if (formData.hrvSdnn) data.hrvSdnn = parseInt(formData.hrvSdnn);
    if (formData.afibDetected) data.afibDetected = formData.afibDetected;
    if (formData.irregularRhythmAlert) data.irregularRhythmAlert = formData.irregularRhythmAlert;
    if (formData.spo2) data.spo2 = parseInt(formData.spo2);
    if (formData.spo2Min) data.spo2Min = parseInt(formData.spo2Min);
    if (formData.respiratoryRate) data.respiratoryRate = parseInt(formData.respiratoryRate);
    if (formData.sleepDuration) data.sleepDuration = parseInt(formData.sleepDuration);
    if (formData.sleepDeepMinutes) data.sleepDeepMinutes = parseInt(formData.sleepDeepMinutes);
    if (formData.sleepRemMinutes) data.sleepRemMinutes = parseInt(formData.sleepRemMinutes);
    if (formData.sleepLightMinutes) data.sleepLightMinutes = parseInt(formData.sleepLightMinutes);
    if (formData.sleepAwakeMinutes) data.sleepAwakeMinutes = parseInt(formData.sleepAwakeMinutes);
    if (formData.sleepScore) data.sleepScore = parseInt(formData.sleepScore);
    if (formData.sleepEfficiency) data.sleepEfficiency = parseFloat(formData.sleepEfficiency);
    if (formData.sleepConsistency) data.sleepConsistency = parseInt(formData.sleepConsistency);
    if (formData.sleepDebt) data.sleepDebt = parseInt(formData.sleepDebt);
    if (formData.sleepNeed) data.sleepNeed = parseInt(formData.sleepNeed);
    if (formData.recoveryScore) data.recoveryScore = parseInt(formData.recoveryScore);
    if (formData.readinessScore) data.readinessScore = parseInt(formData.readinessScore);
    if (formData.bodyBattery) data.bodyBattery = parseInt(formData.bodyBattery);
    if (formData.strainScore) data.strainScore = parseFloat(formData.strainScore);
    if (formData.stressScore) data.stressScore = parseInt(formData.stressScore);
    if (formData.steps) data.steps = parseInt(formData.steps);
    if (formData.activeMinutes) data.activeMinutes = parseInt(formData.activeMinutes);
    if (formData.caloriesBurned) data.caloriesBurned = parseInt(formData.caloriesBurned);
    if (formData.distanceMeters) data.distanceMeters = parseInt(formData.distanceMeters);
    if (formData.floorsClimbed) data.floorsClimbed = parseInt(formData.floorsClimbed);
    if (formData.standingHours) data.standingHours = parseInt(formData.standingHours);
    if (formData.vo2Max) data.vo2Max = parseFloat(formData.vo2Max);
    if (formData.trainingLoad) data.trainingLoad = parseInt(formData.trainingLoad);
    if (formData.trainingStatus) data.trainingStatus = formData.trainingStatus;
    if (formData.trainingReadiness) data.trainingReadiness = parseInt(formData.trainingReadiness);
    if (formData.fitnessAge) data.fitnessAge = parseInt(formData.fitnessAge);
    if (formData.lactateThreshold) data.lactateThreshold = parseInt(formData.lactateThreshold);
    if (formData.performanceCondition) data.performanceCondition = parseInt(formData.performanceCondition);
    if (formData.skinTemperature) data.skinTemperature = parseFloat(formData.skinTemperature);
    if (formData.cycleDay) data.cycleDay = parseInt(formData.cycleDay);
    if (formData.cyclePhase) data.cyclePhase = formData.cyclePhase;
    if (formData.predictedOvulation) data.predictedOvulation = formData.predictedOvulation;
    if (formData.periodLogged) data.periodLogged = formData.periodLogged;
    if (formData.runningDynamics) {
      try {
        data.runningDynamics = JSON.parse(formData.runningDynamics);
      } catch {
        // Invalid JSON - skip field
      }
    }
    if (formData.fallDetected) data.fallDetected = formData.fallDetected;
    if (formData.emergencySOSTriggered) data.emergencySOSTriggered = formData.emergencySOSTriggered;
    
    onSubmit(data);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="watch-brand">Smartwatch Brand</Label>
        <Select value={formData.deviceBrand} onValueChange={(v) => setFormData({ ...formData, deviceBrand: v })}>
          <SelectTrigger data-testid="select-watch-brand">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="apple">Apple Watch</SelectItem>
            <SelectItem value="garmin">Garmin</SelectItem>
            <SelectItem value="whoop">Whoop</SelectItem>
            <SelectItem value="oura">Oura Ring</SelectItem>
            <SelectItem value="fitbit">Fitbit</SelectItem>
            <SelectItem value="samsung">Samsung Galaxy Watch</SelectItem>
            <SelectItem value="google">Google Pixel Watch</SelectItem>
            <SelectItem value="other">Other</SelectItem>
          </SelectContent>
        </Select>
      </div>
      
      <Accordion type="multiple" className="w-full">
        <AccordionItem value="heart">
          <AccordionTrigger className="hover:no-underline" data-testid="accordion-watch-heart">
            <div className="flex items-center gap-2">
              <Heart className="h-4 w-4 text-rose-500" />
              Heart & Cardiovascular
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-hr" className="text-xs">Heart Rate (bpm)</Label>
                <Input
                  id="watch-hr"
                  type="number"
                  placeholder="72"
                  value={formData.heartRate}
                  onChange={(e) => setFormData({ ...formData, heartRate: e.target.value })}
                  data-testid="input-watch-hr"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-rhr" className="text-xs">Resting HR (bpm)</Label>
                <Input
                  id="watch-rhr"
                  type="number"
                  placeholder="60"
                  value={formData.restingHeartRate}
                  onChange={(e) => setFormData({ ...formData, restingHeartRate: e.target.value })}
                  data-testid="input-watch-rhr"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-hrv" className="text-xs">HRV RMSSD (ms)</Label>
                <Input
                  id="watch-hrv"
                  type="number"
                  placeholder="45"
                  value={formData.hrv}
                  onChange={(e) => setFormData({ ...formData, hrv: e.target.value })}
                  data-testid="input-watch-hrv"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-hrv-sdnn" className="text-xs">HRV SDNN (ms)</Label>
                <Input
                  id="watch-hrv-sdnn"
                  type="number"
                  placeholder="50"
                  value={formData.hrvSdnn}
                  onChange={(e) => setFormData({ ...formData, hrvSdnn: e.target.value })}
                  data-testid="input-watch-hrv-sdnn"
                />
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <Switch
                  id="watch-afib"
                  checked={formData.afibDetected}
                  onCheckedChange={(checked) => setFormData({ ...formData, afibDetected: checked })}
                  data-testid="switch-watch-afib"
                />
                <Label htmlFor="watch-afib" className="text-xs">AFib Detected</Label>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  id="watch-irregular"
                  checked={formData.irregularRhythmAlert}
                  onCheckedChange={(checked) => setFormData({ ...formData, irregularRhythmAlert: checked })}
                  data-testid="switch-watch-irregular"
                />
                <Label htmlFor="watch-irregular" className="text-xs">Irregular Rhythm Alert</Label>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="oxygen">
          <AccordionTrigger className="hover:no-underline" data-testid="accordion-watch-oxygen">
            <div className="flex items-center gap-2">
              <Wind className="h-4 w-4 text-blue-500" />
              Blood Oxygen & Respiratory
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-4">
            <div className="grid grid-cols-3 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-spo2" className="text-xs">SpO2 (%)</Label>
                <Input
                  id="watch-spo2"
                  type="number"
                  placeholder="98"
                  value={formData.spo2}
                  onChange={(e) => setFormData({ ...formData, spo2: e.target.value })}
                  min={70}
                  max={100}
                  data-testid="input-watch-spo2"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-spo2-min" className="text-xs">SpO2 Min (%)</Label>
                <Input
                  id="watch-spo2-min"
                  type="number"
                  placeholder="92"
                  value={formData.spo2Min}
                  onChange={(e) => setFormData({ ...formData, spo2Min: e.target.value })}
                  min={70}
                  max={100}
                  data-testid="input-watch-spo2-min"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-resp" className="text-xs">Respiratory Rate</Label>
                <Input
                  id="watch-resp"
                  type="number"
                  placeholder="14"
                  value={formData.respiratoryRate}
                  onChange={(e) => setFormData({ ...formData, respiratoryRate: e.target.value })}
                  data-testid="input-watch-resp"
                />
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="sleep">
          <AccordionTrigger className="hover:no-underline" data-testid="accordion-watch-sleep">
            <div className="flex items-center gap-2">
              <Moon className="h-4 w-4 text-indigo-500" />
              Sleep
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-dur" className="text-xs">Sleep Duration (min)</Label>
                <Input
                  id="watch-sleep-dur"
                  type="number"
                  placeholder="420"
                  value={formData.sleepDuration}
                  onChange={(e) => setFormData({ ...formData, sleepDuration: e.target.value })}
                  data-testid="input-watch-sleep-dur"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-score" className="text-xs">Sleep Score (0-100)</Label>
                <Input
                  id="watch-sleep-score"
                  type="number"
                  placeholder="85"
                  value={formData.sleepScore}
                  onChange={(e) => setFormData({ ...formData, sleepScore: e.target.value })}
                  min={0}
                  max={100}
                  data-testid="input-watch-sleep-score"
                />
              </div>
            </div>
            <div className="grid grid-cols-4 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-deep" className="text-xs">Deep (min)</Label>
                <Input
                  id="watch-sleep-deep"
                  type="number"
                  placeholder="90"
                  value={formData.sleepDeepMinutes}
                  onChange={(e) => setFormData({ ...formData, sleepDeepMinutes: e.target.value })}
                  data-testid="input-watch-sleep-deep"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-rem" className="text-xs">REM (min)</Label>
                <Input
                  id="watch-sleep-rem"
                  type="number"
                  placeholder="100"
                  value={formData.sleepRemMinutes}
                  onChange={(e) => setFormData({ ...formData, sleepRemMinutes: e.target.value })}
                  data-testid="input-watch-sleep-rem"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-light" className="text-xs">Light (min)</Label>
                <Input
                  id="watch-sleep-light"
                  type="number"
                  placeholder="200"
                  value={formData.sleepLightMinutes}
                  onChange={(e) => setFormData({ ...formData, sleepLightMinutes: e.target.value })}
                  data-testid="input-watch-sleep-light"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-awake" className="text-xs">Awake (min)</Label>
                <Input
                  id="watch-sleep-awake"
                  type="number"
                  placeholder="30"
                  value={formData.sleepAwakeMinutes}
                  onChange={(e) => setFormData({ ...formData, sleepAwakeMinutes: e.target.value })}
                  data-testid="input-watch-sleep-awake"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-efficiency" className="text-xs">Efficiency (%)</Label>
                <Input
                  id="watch-sleep-efficiency"
                  type="number"
                  step="0.1"
                  placeholder="92.5"
                  value={formData.sleepEfficiency}
                  onChange={(e) => setFormData({ ...formData, sleepEfficiency: e.target.value })}
                  data-testid="input-watch-sleep-efficiency"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-consistency" className="text-xs">Consistency (0-100)</Label>
                <Input
                  id="watch-sleep-consistency"
                  type="number"
                  placeholder="80"
                  value={formData.sleepConsistency}
                  onChange={(e) => setFormData({ ...formData, sleepConsistency: e.target.value })}
                  min={0}
                  max={100}
                  data-testid="input-watch-sleep-consistency"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-debt" className="text-xs">Sleep Debt (hrs)</Label>
                <Input
                  id="watch-sleep-debt"
                  type="number"
                  placeholder="2"
                  value={formData.sleepDebt}
                  onChange={(e) => setFormData({ ...formData, sleepDebt: e.target.value })}
                  data-testid="input-watch-sleep-debt"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-sleep-need" className="text-xs">Sleep Need (hrs)</Label>
                <Input
                  id="watch-sleep-need"
                  type="number"
                  placeholder="8"
                  value={formData.sleepNeed}
                  onChange={(e) => setFormData({ ...formData, sleepNeed: e.target.value })}
                  data-testid="input-watch-sleep-need"
                />
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="recovery">
          <AccordionTrigger className="hover:no-underline" data-testid="accordion-watch-recovery">
            <div className="flex items-center gap-2">
              <Battery className="h-4 w-4 text-emerald-500" />
              Recovery & Stress
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-4">
            <div className="grid grid-cols-3 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-recovery" className="text-xs">Recovery Score</Label>
                <Input
                  id="watch-recovery"
                  type="number"
                  placeholder="75"
                  value={formData.recoveryScore}
                  onChange={(e) => setFormData({ ...formData, recoveryScore: e.target.value })}
                  min={0}
                  max={100}
                  data-testid="input-watch-recovery"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-readiness" className="text-xs">Readiness Score</Label>
                <Input
                  id="watch-readiness"
                  type="number"
                  placeholder="80"
                  value={formData.readinessScore}
                  onChange={(e) => setFormData({ ...formData, readinessScore: e.target.value })}
                  min={0}
                  max={100}
                  data-testid="input-watch-readiness"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-body-battery" className="text-xs">Body Battery</Label>
                <Input
                  id="watch-body-battery"
                  type="number"
                  placeholder="65"
                  value={formData.bodyBattery}
                  onChange={(e) => setFormData({ ...formData, bodyBattery: e.target.value })}
                  min={0}
                  max={100}
                  data-testid="input-watch-body-battery"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-strain" className="text-xs">Strain Score (0-21)</Label>
                <Input
                  id="watch-strain"
                  type="number"
                  step="0.1"
                  placeholder="12.5"
                  value={formData.strainScore}
                  onChange={(e) => setFormData({ ...formData, strainScore: e.target.value })}
                  min={0}
                  max={21}
                  data-testid="input-watch-strain"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-stress" className="text-xs">Stress Score (0-100)</Label>
                <Input
                  id="watch-stress"
                  type="number"
                  placeholder="35"
                  value={formData.stressScore}
                  onChange={(e) => setFormData({ ...formData, stressScore: e.target.value })}
                  min={0}
                  max={100}
                  data-testid="input-watch-stress"
                />
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="activity">
          <AccordionTrigger className="hover:no-underline" data-testid="accordion-watch-activity">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-amber-500" />
              Activity & Fitness
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-4">
            <div className="grid grid-cols-3 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-steps" className="text-xs">Steps</Label>
                <Input
                  id="watch-steps"
                  type="number"
                  placeholder="8500"
                  value={formData.steps}
                  onChange={(e) => setFormData({ ...formData, steps: e.target.value })}
                  data-testid="input-watch-steps"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-active" className="text-xs">Active Minutes</Label>
                <Input
                  id="watch-active"
                  type="number"
                  placeholder="45"
                  value={formData.activeMinutes}
                  onChange={(e) => setFormData({ ...formData, activeMinutes: e.target.value })}
                  data-testid="input-watch-active"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-calories" className="text-xs">Calories Burned</Label>
                <Input
                  id="watch-calories"
                  type="number"
                  placeholder="2200"
                  value={formData.caloriesBurned}
                  onChange={(e) => setFormData({ ...formData, caloriesBurned: e.target.value })}
                  data-testid="input-watch-calories"
                />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-distance" className="text-xs">Distance (m)</Label>
                <Input
                  id="watch-distance"
                  type="number"
                  placeholder="7500"
                  value={formData.distanceMeters}
                  onChange={(e) => setFormData({ ...formData, distanceMeters: e.target.value })}
                  data-testid="input-watch-distance"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-floors" className="text-xs">Floors Climbed</Label>
                <Input
                  id="watch-floors"
                  type="number"
                  placeholder="10"
                  value={formData.floorsClimbed}
                  onChange={(e) => setFormData({ ...formData, floorsClimbed: e.target.value })}
                  data-testid="input-watch-floors"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-standing" className="text-xs">Standing Hours</Label>
                <Input
                  id="watch-standing"
                  type="number"
                  placeholder="8"
                  value={formData.standingHours}
                  onChange={(e) => setFormData({ ...formData, standingHours: e.target.value })}
                  max={24}
                  data-testid="input-watch-standing"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="watch-vo2" className="text-xs">VO2 Max (ml/kg/min)</Label>
              <Input
                id="watch-vo2"
                type="number"
                step="0.1"
                placeholder="42.5"
                value={formData.vo2Max}
                onChange={(e) => setFormData({ ...formData, vo2Max: e.target.value })}
                data-testid="input-watch-vo2"
              />
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="temp">
          <AccordionTrigger className="hover:no-underline" data-testid="accordion-watch-temp">
            <div className="flex items-center gap-2">
              <Thermometer className="h-4 w-4 text-orange-500" />
              Temperature
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-4">
            <div className="space-y-2">
              <Label htmlFor="watch-skin-temp" className="text-xs">Skin Temperature Deviation (°C)</Label>
              <Input
                id="watch-skin-temp"
                type="number"
                step="0.01"
                placeholder="-0.2"
                value={formData.skinTemperature}
                onChange={(e) => setFormData({ ...formData, skinTemperature: e.target.value })}
                data-testid="input-watch-skin-temp"
              />
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="performance">
          <AccordionTrigger className="hover:no-underline" data-testid="accordion-watch-performance">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-purple-500" />
              Performance & Training
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-training-load" className="text-xs">Training Load</Label>
                <Input
                  id="watch-training-load"
                  type="number"
                  placeholder="150"
                  value={formData.trainingLoad}
                  onChange={(e) => setFormData({ ...formData, trainingLoad: e.target.value })}
                  data-testid="input-watch-training-load"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-training-readiness" className="text-xs">Training Readiness (0-100)</Label>
                <Input
                  id="watch-training-readiness"
                  type="number"
                  placeholder="85"
                  value={formData.trainingReadiness}
                  onChange={(e) => setFormData({ ...formData, trainingReadiness: e.target.value })}
                  min={0}
                  max={100}
                  data-testid="input-watch-training-readiness"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-lactate" className="text-xs">Lactate Threshold (bpm)</Label>
                <Input
                  id="watch-lactate"
                  type="number"
                  placeholder="165"
                  value={formData.lactateThreshold}
                  onChange={(e) => setFormData({ ...formData, lactateThreshold: e.target.value })}
                  data-testid="input-watch-lactate"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-performance" className="text-xs">Performance Condition (-20 to +20)</Label>
                <Input
                  id="watch-performance"
                  type="number"
                  placeholder="5"
                  value={formData.performanceCondition}
                  onChange={(e) => setFormData({ ...formData, performanceCondition: e.target.value })}
                  min={-20}
                  max={20}
                  data-testid="input-watch-performance"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-fitness-age" className="text-xs">Fitness Age</Label>
                <Input
                  id="watch-fitness-age"
                  type="number"
                  placeholder="35"
                  value={formData.fitnessAge}
                  onChange={(e) => setFormData({ ...formData, fitnessAge: e.target.value })}
                  data-testid="input-watch-fitness-age"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-training-status" className="text-xs">Training Status</Label>
                <Select value={formData.trainingStatus} onValueChange={(v) => setFormData({ ...formData, trainingStatus: v })}>
                  <SelectTrigger data-testid="select-watch-training-status">
                    <SelectValue placeholder="Select status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="productive">Productive</SelectItem>
                    <SelectItem value="peaking">Peaking</SelectItem>
                    <SelectItem value="maintaining">Maintaining</SelectItem>
                    <SelectItem value="recovery">Recovery</SelectItem>
                    <SelectItem value="unproductive">Unproductive</SelectItem>
                    <SelectItem value="overreaching">Overreaching</SelectItem>
                    <SelectItem value="detraining">Detraining</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="watch-running-dynamics" className="text-xs">Running Dynamics (JSON)</Label>
              <Textarea
                id="watch-running-dynamics"
                placeholder='{"groundContactTime": 250, "strideLength": 1.2, "verticalOscillation": 8.5}'
                value={formData.runningDynamics}
                onChange={(e) => setFormData({ ...formData, runningDynamics: e.target.value })}
                data-testid="textarea-watch-running-dynamics"
              />
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="womens-health">
          <AccordionTrigger className="hover:no-underline" data-testid="accordion-watch-womens-health">
            <div className="flex items-center gap-2">
              <Heart className="h-4 w-4 text-pink-500" />
              Women's Health
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="watch-cycle-day" className="text-xs">Cycle Day</Label>
                <Input
                  id="watch-cycle-day"
                  type="number"
                  placeholder="14"
                  value={formData.cycleDay}
                  onChange={(e) => setFormData({ ...formData, cycleDay: e.target.value })}
                  min={1}
                  max={45}
                  data-testid="input-watch-cycle-day"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="watch-cycle-phase" className="text-xs">Cycle Phase</Label>
                <Select value={formData.cyclePhase} onValueChange={(v) => setFormData({ ...formData, cyclePhase: v })}>
                  <SelectTrigger data-testid="select-watch-cycle-phase">
                    <SelectValue placeholder="Select phase" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="menstrual">Menstrual</SelectItem>
                    <SelectItem value="follicular">Follicular</SelectItem>
                    <SelectItem value="ovulation">Ovulation</SelectItem>
                    <SelectItem value="luteal">Luteal</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="watch-predicted-ovulation" className="text-xs">Predicted Ovulation Date</Label>
              <Input
                id="watch-predicted-ovulation"
                type="date"
                value={formData.predictedOvulation}
                onChange={(e) => setFormData({ ...formData, predictedOvulation: e.target.value })}
                data-testid="input-watch-predicted-ovulation"
              />
            </div>
            <div className="flex items-center gap-2">
              <Switch
                id="watch-period"
                checked={formData.periodLogged}
                onCheckedChange={(checked) => setFormData({ ...formData, periodLogged: checked })}
                data-testid="switch-watch-period"
              />
              <Label htmlFor="watch-period">Period Logged Today</Label>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="safety">
          <AccordionTrigger className="hover:no-underline" data-testid="accordion-watch-safety">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              Safety & Emergency
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4 pt-4">
            <div className="flex items-center gap-2">
              <Switch
                id="watch-fall"
                checked={formData.fallDetected}
                onCheckedChange={(checked) => setFormData({ ...formData, fallDetected: checked })}
                data-testid="switch-watch-fall"
              />
              <Label htmlFor="watch-fall">Fall Detected</Label>
            </div>
            <div className="flex items-center gap-2">
              <Switch
                id="watch-sos"
                checked={formData.emergencySOSTriggered}
                onCheckedChange={(checked) => setFormData({ ...formData, emergencySOSTriggered: checked })}
                data-testid="switch-watch-sos"
              />
              <Label htmlFor="watch-sos">Emergency SOS Triggered</Label>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      <div className="space-y-2">
        <Label htmlFor="watch-notes">Notes</Label>
        <Textarea
          id="watch-notes"
          placeholder="How are you feeling today?"
          value={formData.notes}
          onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
          data-testid="textarea-watch-notes"
        />
      </div>
      <Button type="submit" className="w-full" disabled={isLoading} data-testid="button-submit-watch">
        {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Check className="h-4 w-4 mr-2" />}
        Save Smartwatch Data
      </Button>
    </form>
  );
}

export function DeviceDataManager() {
  const { toast } = useToast();
  const [activeDialog, setActiveDialog] = useState<string | null>(null);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['/api/device-readings/summary'],
  });

  const { data: recentReadings, isLoading: readingsLoading } = useQuery({
    queryKey: ['/api/device-readings', { limit: 20 }],
  });

  const createReading = useMutation({
    mutationFn: async (data: any) => {
      return apiRequest('/api/device-readings', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: { 'Content-Type': 'application/json' },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/device-readings'] });
      toast({
        title: 'Reading Saved',
        description: 'Your device reading has been recorded successfully.',
      });
      setActiveDialog(null);
    },
    onError: (error: any) => {
      toast({
        title: 'Error',
        description: error.message || 'Failed to save reading',
        variant: 'destructive',
      });
    },
  });

  const getFormForDevice = (deviceId: string) => {
    switch (deviceId) {
      case 'bp_monitor':
        return <BPMonitorForm onSubmit={createReading.mutate} isLoading={createReading.isPending} />;
      case 'glucose_meter':
        return <GlucoseMeterForm onSubmit={createReading.mutate} isLoading={createReading.isPending} />;
      case 'smart_scale':
        return <SmartScaleForm onSubmit={createReading.mutate} isLoading={createReading.isPending} />;
      case 'thermometer':
        return <ThermometerForm onSubmit={createReading.mutate} isLoading={createReading.isPending} />;
      case 'stethoscope':
        return <StethoscopeForm onSubmit={createReading.mutate} isLoading={createReading.isPending} />;
      case 'smartwatch':
        return <SmartwatchForm onSubmit={createReading.mutate} isLoading={createReading.isPending} />;
      default:
        return null;
    }
  };

  const getDeviceInfo = (deviceId: string) => {
    return DEVICE_TYPES.find(d => d.id === deviceId);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Medical Device Data</h2>
          <p className="text-sm text-muted-foreground">
            Track readings from your medical devices and wearables
          </p>
        </div>
        <Badge variant="outline" className="gap-1">
          <Smartphone className="h-3 w-3" />
          6 Device Types
        </Badge>
      </div>

      {summaryLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <Card key={i}>
              <CardContent className="p-4">
                <div className="animate-pulse flex items-center gap-3">
                  <div className="w-10 h-10 bg-muted rounded-lg" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-muted rounded w-24" />
                    <div className="h-3 bg-muted rounded w-16" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {DEVICE_TYPES.map((device) => (
            <Dialog 
              key={device.id} 
              open={activeDialog === device.id} 
              onOpenChange={(open) => setActiveDialog(open ? device.id : null)}
            >
              <DeviceCard
                device={device}
                latestReading={summary?.[device.id]?.lastReading}
                onAddReading={() => setActiveDialog(device.id)}
              />
              <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle className="flex items-center gap-2">
                    <device.icon className={`h-5 w-5 ${device.color}`} />
                    Add {device.name} Reading
                  </DialogTitle>
                  <DialogDescription>
                    Enter the values from your {device.name.toLowerCase()}.
                  </DialogDescription>
                </DialogHeader>
                {getFormForDevice(device.id)}
              </DialogContent>
            </Dialog>
          ))}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Recent Readings
          </CardTitle>
          <CardDescription>
            Your latest device data across all devices
          </CardDescription>
        </CardHeader>
        <CardContent>
          {readingsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : recentReadings?.length > 0 ? (
            <ScrollArea className="h-[300px]">
              <div className="space-y-3">
                {recentReadings.map((reading: DeviceReading) => {
                  const device = getDeviceInfo(reading.deviceType);
                  if (!device) return null;
                  const Icon = device.icon;
                  
                  return (
                    <div 
                      key={reading.id} 
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover-elevate"
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-md ${device.bgColor} flex items-center justify-center`}>
                          <Icon className={`h-4 w-4 ${device.color}`} />
                        </div>
                        <div>
                          <p className="font-medium text-sm">{device.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {format(new Date(reading.recordedAt), 'MMM d, yyyy h:mm a')}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {reading.source === 'auto_sync' && (
                          <Badge variant="secondary" className="text-xs">
                            <RefreshCw className="h-3 w-3 mr-1" />
                            Synced
                          </Badge>
                        )}
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </div>
                  );
                })}
              </div>
            </ScrollArea>
          ) : (
            <div className="text-center py-8">
              <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-3">
                <Activity className="h-6 w-6 text-muted-foreground" />
              </div>
              <p className="text-sm text-muted-foreground">
                No device readings yet. Add your first reading above!
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-primary/20 bg-primary/5">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
              <AlertTriangle className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="font-medium text-sm">AI Health Monitoring Active</h3>
              <p className="text-xs text-muted-foreground mt-1">
                Your device readings are analyzed by our AI to detect potential health changes and generate personalized alerts. 
                Data is routed to the appropriate health sections (Hypertension, Diabetes, Cardiovascular, etc.) for comprehensive monitoring.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
