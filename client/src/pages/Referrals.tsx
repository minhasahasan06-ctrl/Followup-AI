import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { Gift, Copy, Check, Users, CalendarClock } from "lucide-react";

export default function Referrals() {
  const { toast } = useToast();
  const [copied, setCopied] = useState(false);

  const { data: referralCode } = useQuery({
    queryKey: ["/api/referrals/my-code"],
  });

  const { data: referrals } = useQuery({
    queryKey: ["/api/referrals/my-referrals"],
  });

  const handleCopyLink = () => {
    if (referralCode?.referralLink) {
      navigator.clipboard.writeText(referralCode.referralLink);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast({
        title: "Link Copied!",
        description: "Share this link with friends to earn free trial time.",
      });
    }
  };

  const handleCopyCode = () => {
    if (referralCode?.referralCode) {
      navigator.clipboard.writeText(referralCode.referralCode);
      toast({
        title: "Code Copied!",
        description: "Share this code to give friends 1 month free trial.",
      });
    }
  };

  const getStatusBadge = (status: string) => {
    const config: any = {
      pending: { label: "Pending", variant: "secondary" },
      signed_up: { label: "Signed Up", variant: "default" },
      trial_activated: { label: "Trial Active", variant: "default" },
      trial_completed: { label: "Completed", variant: "outline" },
      expired: { label: "Expired", variant: "outline" },
    };

    const { label, variant } = config[status] || { label: status, variant: "outline" };
    return <Badge variant={variant}>{label}</Badge>;
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Referral Program</h1>
        <p className="text-muted-foreground">
          Share Followup AI with friends and both get 1 month free trial
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2 mb-8">
        <Card className="border-primary/50 bg-gradient-to-br from-primary/5 to-transparent">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Gift className="h-5 w-5 text-primary" />
              Your Referral Code
            </CardTitle>
            <CardDescription>
              Share this code with friends for 1 month free trial
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {referralCode ? (
              <>
                <div className="flex items-center gap-2">
                  <Input
                    value={referralCode.referralCode}
                    readOnly
                    className="font-mono text-lg font-bold text-center"
                    data-testid="input-referral-code"
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handleCopyCode}
                    data-testid="button-copy-code"
                  >
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <div className="flex items-center gap-2">
                  <Input
                    value={referralCode.referralLink}
                    readOnly
                    className="text-sm"
                    data-testid="input-referral-link"
                  />
                  <Button
                    variant="default"
                    size="icon"
                    onClick={handleCopyLink}
                    data-testid="button-copy-link"
                  >
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </>
            ) : (
              <div className="text-center py-4 text-muted-foreground">
                Loading your referral code...
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CalendarClock className="h-5 w-5" />
              How It Works
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="space-y-3 text-sm">
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">
                  1
                </span>
                <div>
                  <p className="font-medium">Share Your Link</p>
                  <p className="text-muted-foreground">
                    Send your referral code or link to friends
                  </p>
                </div>
              </li>
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">
                  2
                </span>
                <div>
                  <p className="font-medium">Friend Signs Up</p>
                  <p className="text-muted-foreground">
                    They create an account using your referral code
                  </p>
                </div>
              </li>
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">
                  3
                </span>
                <div>
                  <p className="font-medium">Both Get 1 Month Free</p>
                  <p className="text-muted-foreground">
                    You and your friend both receive 30 days free trial
                  </p>
                </div>
              </li>
            </ol>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Your Referrals ({referrals?.length || 0})
          </CardTitle>
          <CardDescription>
            Track your successful referrals and rewards
          </CardDescription>
        </CardHeader>
        <CardContent>
          {referrals && referrals.length > 0 ? (
            <div className="space-y-3">
              {referrals.map((referral: any) => (
                <div
                  key={referral.id}
                  className="flex items-center justify-between p-3 rounded-lg border hover-elevate"
                >
                  <div className="flex-1">
                    <div className="font-medium">
                      {referral.refereeEmail || referral.refereeId || "Unknown User"}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {referral.signedUpAt
                        ? `Signed up ${new Date(referral.signedUpAt).toLocaleDateString()}`
                        : referral.clickedAt
                        ? `Clicked ${new Date(referral.clickedAt).toLocaleDateString()}`
                        : "Pending signup"}
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {getStatusBadge(referral.status)}
                    {referral.referrerTrialExtended && (
                      <Badge variant="default" className="bg-green-600">
                        +30 days
                      </Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Users className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No referrals yet. Start sharing your code to earn free trial time!</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
