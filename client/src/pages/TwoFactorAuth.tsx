import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { Shield, CheckCircle, XCircle, AlertTriangle, Copy, Download } from "lucide-react";

export default function TwoFactorAuth() {
  const { toast } = useToast();
  const [setupStep, setSetupStep] = useState<'initial' | 'setup' | 'verify'>('initial');
  const [qrCode, setQrCode] = useState('');
  const [secret, setSecret] = useState('');
  const [backupCodes, setBackupCodes] = useState<string[]>([]);
  const [verificationToken, setVerificationToken] = useState('');
  const [showDisableDialog, setShowDisableDialog] = useState(false);
  const [disablePassword, setDisablePassword] = useState('');

  const { data: status, isLoading } = useQuery({
    queryKey: ['/api/2fa/status'],
  });

  const setupMutation = useMutation({
    mutationFn: async () => {
      return await apiRequest('/api/2fa/setup', {
        method: 'POST',
      });
    },
    onSuccess: (data) => {
      setQrCode(data.qrCode);
      setSecret(data.secret);
      setBackupCodes(data.backupCodes);
      setSetupStep('setup');
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to setup 2FA. Please try again.",
        variant: "destructive",
      });
    },
  });

  const enableMutation = useMutation({
    mutationFn: async (token: string) => {
      return await apiRequest('/api/2fa/enable', {
        method: 'POST',
        body: JSON.stringify({ token }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/2fa/status'] });
      setSetupStep('initial');
      setVerificationToken('');
      toast({
        title: "2FA Enabled",
        description: "Two-factor authentication has been successfully enabled.",
      });
    },
    onError: () => {
      toast({
        title: "Verification Failed",
        description: "Invalid verification code. Please try again.",
        variant: "destructive",
      });
    },
  });

  const disableMutation = useMutation({
    mutationFn: async (password: string) => {
      return await apiRequest('/api/2fa/disable', {
        method: 'POST',
        body: JSON.stringify({ password }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/2fa/status'] });
      setShowDisableDialog(false);
      setDisablePassword('');
      toast({
        title: "2FA Disabled",
        description: "Two-factor authentication has been disabled.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to disable 2FA. Please check your password.",
        variant: "destructive",
      });
    },
  });

  const handleSetup = () => {
    setupMutation.mutate();
  };

  const handleVerify = () => {
    if (verificationToken.length === 6) {
      enableMutation.mutate(verificationToken);
    }
  };

  const handleDisable = () => {
    if (disablePassword) {
      disableMutation.mutate(disablePassword);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied",
      description: "Copied to clipboard",
    });
  };

  const downloadBackupCodes = () => {
    const text = backupCodes.join('\n');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'followupai-backup-codes.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div className="container max-w-4xl py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Shield className="w-8 h-8" />
          Two-Factor Authentication
        </h1>
        <p className="text-muted-foreground mt-2">
          Add an extra layer of security to your account
        </p>
      </div>

      {setupStep === 'initial' && (
        <Card data-testid="card-2fa-status">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Security Status</CardTitle>
                <CardDescription className="mt-2">
                  {status?.enabled
                    ? "Your account is protected with two-factor authentication"
                    : "Enable 2FA to secure your account"}
                </CardDescription>
              </div>
              <Badge
                variant={status?.enabled ? "default" : "secondary"}
                data-testid={`badge-2fa-${status?.enabled ? 'enabled' : 'disabled'}`}
              >
                {status?.enabled ? (
                  <span className="flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" />
                    Enabled
                  </span>
                ) : (
                  <span className="flex items-center gap-1">
                    <XCircle className="w-3 h-3" />
                    Disabled
                  </span>
                )}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert>
              <AlertTriangle className="w-4 h-4" />
              <AlertDescription>
                Two-factor authentication adds an extra layer of security by requiring a verification code
                from your authenticator app in addition to your password.
              </AlertDescription>
            </Alert>

            {!status?.enabled ? (
              <Button
                onClick={handleSetup}
                disabled={setupMutation.isPending}
                className="w-full"
                data-testid="button-enable-2fa"
              >
                <Shield className="w-4 h-4 mr-2" />
                Enable Two-Factor Authentication
              </Button>
            ) : (
              <Button
                variant="destructive"
                onClick={() => setShowDisableDialog(true)}
                className="w-full"
                data-testid="button-disable-2fa"
              >
                <XCircle className="w-4 h-4 mr-2" />
                Disable Two-Factor Authentication
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      {setupStep === 'setup' && (
        <div className="space-y-6">
          <Card data-testid="card-2fa-setup">
            <CardHeader>
              <CardTitle>Step 1: Scan QR Code</CardTitle>
              <CardDescription>
                Use an authenticator app like Google Authenticator or Authy to scan this QR code
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-center">
                <img src={qrCode} alt="2FA QR Code" className="w-64 h-64" data-testid="img-qr-code" />
              </div>
              
              <div className="bg-muted p-4 rounded-md">
                <Label className="text-sm font-medium">Manual Entry Code</Label>
                <div className="flex items-center gap-2 mt-2">
                  <code className="flex-1 text-sm font-mono" data-testid="text-secret">
                    {secret}
                  </code>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => copyToClipboard(secret)}
                    data-testid="button-copy-secret"
                  >
                    <Copy className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card data-testid="card-backup-codes">
            <CardHeader>
              <CardTitle>Step 2: Save Backup Codes</CardTitle>
              <CardDescription>
                Store these backup codes in a safe place. You can use them to access your account if you lose your device.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-2 bg-muted p-4 rounded-md">
                {backupCodes.map((code, index) => (
                  <code key={index} className="text-sm font-mono" data-testid={`text-backup-code-${index}`}>
                    {code}
                  </code>
                ))}
              </div>
              
              <Button
                variant="outline"
                onClick={downloadBackupCodes}
                className="w-full"
                data-testid="button-download-codes"
              >
                <Download className="w-4 h-4 mr-2" />
                Download Backup Codes
              </Button>
              
              <Button
                onClick={() => setSetupStep('verify')}
                className="w-full"
                data-testid="button-continue-verify"
              >
                Continue to Verification
              </Button>
            </CardContent>
          </Card>
        </div>
      )}

      {setupStep === 'verify' && (
        <Card data-testid="card-2fa-verify">
          <CardHeader>
            <CardTitle>Step 3: Verify Setup</CardTitle>
            <CardDescription>
              Enter the 6-digit code from your authenticator app to complete setup
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="verification-code">Verification Code</Label>
              <Input
                id="verification-code"
                type="text"
                maxLength={6}
                placeholder="000000"
                value={verificationToken}
                onChange={(e) => setVerificationToken(e.target.value.replace(/\D/g, ''))}
                className="text-center text-2xl font-mono tracking-widest"
                data-testid="input-verification-code"
              />
            </div>
            
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => setSetupStep('setup')}
                className="flex-1"
                data-testid="button-back"
              >
                Back
              </Button>
              <Button
                onClick={handleVerify}
                disabled={verificationToken.length !== 6 || enableMutation.isPending}
                className="flex-1"
                data-testid="button-verify-enable"
              >
                Verify & Enable
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Disable 2FA Dialog */}
      <Dialog open={showDisableDialog} onOpenChange={setShowDisableDialog}>
        <DialogContent data-testid="dialog-disable-2fa">
          <DialogHeader>
            <DialogTitle>Disable Two-Factor Authentication</DialogTitle>
            <DialogDescription>
              Enter your password to confirm disabling 2FA. This will make your account less secure.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              placeholder="Enter your password"
              value={disablePassword}
              onChange={(e) => setDisablePassword(e.target.value)}
              data-testid="input-disable-password"
            />
          </div>
          
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowDisableDialog(false);
                setDisablePassword('');
              }}
              data-testid="button-cancel-disable"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDisable}
              disabled={!disablePassword || disableMutation.isPending}
              data-testid="button-confirm-disable"
            >
              Disable 2FA
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
