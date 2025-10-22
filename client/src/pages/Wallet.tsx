import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { Wallet as WalletIcon, Plus, ArrowDownToLine, ArrowUpRight, Coins } from "lucide-react";

export default function Wallet() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [showPurchaseDialog, setShowPurchaseDialog] = useState(false);
  const [showWithdrawDialog, setShowWithdrawDialog] = useState(false);
  const [purchaseAmount, setPurchaseAmount] = useState("");
  const [withdrawAmount, setWithdrawAmount] = useState("");

  const { data: balance } = useQuery({
    queryKey: ["/api/wallet/balance"],
  });

  const { data: transactions, isLoading } = useQuery({
    queryKey: ["/api/wallet/transactions"],
  });

  const purchaseMutation = useMutation({
    mutationFn: async (amount: number) => {
      return await apiRequest("/api/wallet/purchase", {
        method: "POST",
        body: JSON.stringify({ amount }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/wallet/balance"] });
      queryClient.invalidateQueries({ queryKey: ["/api/wallet/transactions"] });
      setShowPurchaseDialog(false);
      setPurchaseAmount("");
      toast({
        title: "Credits Purchased",
        description: "Credits have been added to your wallet.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to purchase credits. Please try again.",
        variant: "destructive",
      });
    },
  });

  const withdrawMutation = useMutation({
    mutationFn: async (amount: number) => {
      return await apiRequest("/api/wallet/withdraw", {
        method: "POST",
        body: JSON.stringify({ amount }),
      });
    },
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ["/api/wallet/balance"] });
      queryClient.invalidateQueries({ queryKey: ["/api/wallet/transactions"] });
      setShowWithdrawDialog(false);
      setWithdrawAmount("");
      toast({
        title: "Withdrawal Requested",
        description: data.message || "Your withdrawal has been processed.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to process withdrawal.",
        variant: "destructive",
      });
    },
  });

  const handlePurchase = () => {
    const amount = parseInt(purchaseAmount);
    if (!amount || amount <= 0) {
      toast({
        title: "Invalid Amount",
        description: "Please enter a valid credit amount.",
        variant: "destructive",
      });
      return;
    }
    purchaseMutation.mutate(amount);
  };

  const handleWithdraw = () => {
    const amount = parseInt(withdrawAmount);
    if (!amount || amount <= 0) {
      toast({
        title: "Invalid Amount",
        description: "Please enter a valid withdrawal amount.",
        variant: "destructive",
      });
      return;
    }
    if (balance && amount > balance.balance) {
      toast({
        title: "Insufficient Credits",
        description: "You don't have enough credits for this withdrawal.",
        variant: "destructive",
      });
      return;
    }
    withdrawMutation.mutate(amount);
  };

  const getTransactionIcon = (type: string) => {
    switch (type) {
      case "purchased":
        return <Plus className="h-4 w-4 text-green-600" />;
      case "earned":
        return <ArrowDownToLine className="h-4 w-4 text-green-600" />;
      case "spent":
        return <ArrowUpRight className="h-4 w-4 text-red-600" />;
      case "withdrawn":
        return <ArrowUpRight className="h-4 w-4 text-red-600" />;
      default:
        return <Coins className="h-4 w-4" />;
    }
  };

  const getTransactionColor = (type: string) => {
    return type === "purchased" || type === "earned" ? "text-green-600" : "text-red-600";
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Wallet</h1>
        <p className="text-muted-foreground">
          Manage your consultation credits and earnings
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3 mb-8">
        <Card className="lg:col-span-2 border-primary/50 bg-gradient-to-br from-primary/5 to-transparent">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <WalletIcon className="h-5 w-5" />
              Available Credits
            </CardTitle>
            <CardDescription>
              {user?.role === "patient"
                ? "Use credits for doctor consultations"
                : "Earn credits from consultations"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2 mb-6">
              <span className="text-5xl font-bold">{balance?.balance || 0}</span>
              <span className="text-2xl text-muted-foreground">credits</span>
            </div>
            <div className="flex gap-3">
              {user?.role === "patient" ? (
                <Button
                  onClick={() => setShowPurchaseDialog(true)}
                  className="flex-1"
                  data-testid="button-purchase-credits"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Purchase Credits
                </Button>
              ) : (
                <Button
                  onClick={() => setShowWithdrawDialog(true)}
                  variant="default"
                  className="flex-1"
                  disabled={(balance?.balance || 0) < 100}
                  data-testid="button-withdraw-credits"
                >
                  <ArrowUpRight className="h-4 w-4 mr-2" />
                  Withdraw Earnings
                </Button>
              )}
            </div>
            {user?.role === "doctor" && (balance?.balance || 0) < 100 && (
              <p className="text-xs text-muted-foreground mt-2">
                Minimum withdrawal amount: 100 credits
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Credit Value</CardTitle>
            <CardDescription>Pricing information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">20 credits</span>
              <span className="font-medium">= 1 consultation (10 min)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">100 credits</span>
              <span className="font-medium">= $20 USD</span>
            </div>
            {user?.role === "doctor" && (
              <div className="flex justify-between pt-2 border-t">
                <span className="text-muted-foreground">Your earnings</span>
                <span className="font-bold">${((balance?.balance || 0) / 5).toFixed(2)}</span>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Transaction History</CardTitle>
          <CardDescription>
            View all your credit transactions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center gap-4 p-3 rounded-lg border animate-pulse">
                  <div className="h-8 w-8 bg-muted rounded-full" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-muted rounded w-1/4" />
                    <div className="h-3 bg-muted rounded w-1/2" />
                  </div>
                </div>
              ))}
            </div>
          ) : transactions && transactions.length > 0 ? (
            <div className="space-y-2">
              {transactions.map((tx: any) => (
                <div
                  key={tx.id}
                  className="flex items-center justify-between p-3 rounded-lg border hover-elevate"
                >
                  <div className="flex items-center gap-3 flex-1">
                    <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center">
                      {getTransactionIcon(tx.transactionType)}
                    </div>
                    <div className="flex-1">
                      <div className="font-medium">
                        {tx.description || tx.transactionType}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {new Date(tx.createdAt).toLocaleDateString()} at{" "}
                        {new Date(tx.createdAt).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-lg font-bold ${getTransactionColor(tx.transactionType)}`}>
                      {tx.amount > 0 ? "+" : ""}
                      {tx.amount}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Balance: {tx.balanceAfter}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Coins className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No transactions yet</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Purchase Dialog */}
      <Dialog open={showPurchaseDialog} onOpenChange={setShowPurchaseDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Purchase Credits</DialogTitle>
            <DialogDescription>
              Add credits to your wallet for consultations (100 credits = $20)
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="purchase-amount">Number of Credits</Label>
              <Input
                id="purchase-amount"
                type="number"
                placeholder="100"
                value={purchaseAmount}
                onChange={(e) => setPurchaseAmount(e.target.value)}
                data-testid="input-purchase-amount"
              />
              {purchaseAmount && parseInt(purchaseAmount) > 0 && (
                <p className="text-sm text-muted-foreground">
                  Cost: ${(parseInt(purchaseAmount) / 5).toFixed(2)} USD
                </p>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowPurchaseDialog(false)} data-testid="button-cancel-purchase">
              Cancel
            </Button>
            <Button
              onClick={handlePurchase}
              disabled={purchaseMutation.isPending}
              data-testid="button-confirm-purchase"
            >
              {purchaseMutation.isPending ? "Processing..." : "Purchase"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Withdraw Dialog */}
      <Dialog open={showWithdrawDialog} onOpenChange={setShowWithdrawDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Withdraw Earnings</DialogTitle>
            <DialogDescription>
              Transfer your credits to your bank account (100 credits = $20)
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="withdraw-amount">Number of Credits</Label>
              <Input
                id="withdraw-amount"
                type="number"
                placeholder="100"
                value={withdrawAmount}
                onChange={(e) => setWithdrawAmount(e.target.value)}
                data-testid="input-withdraw-amount"
              />
              {withdrawAmount && parseInt(withdrawAmount) > 0 && (
                <p className="text-sm text-muted-foreground">
                  You'll receive: ${(parseInt(withdrawAmount) / 5).toFixed(2)} USD
                </p>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Funds will be transferred to your account within 2-3 business days
            </p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowWithdrawDialog(false)} data-testid="button-cancel-withdraw">
              Cancel
            </Button>
            <Button
              onClick={handleWithdraw}
              disabled={withdrawMutation.isPending}
              data-testid="button-confirm-withdraw"
            >
              {withdrawMutation.isPending ? "Processing..." : "Withdraw"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
