import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Trophy,
  Star,
  Flame,
  Target,
  Award,
  Crown,
  Shield,
  Heart,
  Users,
  Zap,
  Lock,
  CheckCircle,
  TrendingUp,
} from "lucide-react";

interface Badge {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  xp_reward: number;
  rarity: string;
  unlocked: boolean;
}

interface LevelInfo {
  current: number;
  name: string;
  xpInLevel: number;
  xpForNextLevel: number | null;
  progress: number;
  nextLevelName: string | null;
}

interface GamificationState {
  userId: string;
  level: LevelInfo;
  totalXp: number;
  streakBonus: number;
  completionPoints: number;
  daysActive: number;
  perfectDays: number;
  growthStage: string;
  badges: {
    unlocked: Badge[];
    locked: Badge[];
    totalUnlocked: number;
    totalAvailable: number;
  };
  nextMilestones: Array<{
    badge: Badge;
    progress: number;
    remaining: number;
  }>;
}

const ICON_MAP: Record<string, typeof Trophy> = {
  trophy: Trophy,
  star: Star,
  flame: Flame,
  fire: Flame,
  medal: Award,
  crown: Crown,
  shield: Shield,
  heart: Heart,
  users: Users,
  award: Award,
  gem: Zap,
  "check-circle": CheckCircle,
  "calendar-check": CheckCircle,
  "calendar-star": Star,
  sunrise: Zap,
  moon: Star,
  refresh: TrendingUp,
  "shield-check": Shield,
  "star-shield": Shield,
  "crown-shield": Crown,
  collection: Target,
  footprints: Target,
};

const RARITY_COLORS: Record<string, string> = {
  common: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
  uncommon: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  rare: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  epic: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  legendary: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
};

const GROWTH_STAGE_VISUALS: Record<string, { emoji: string; color: string }> = {
  seed: { emoji: "🌱", color: "text-gray-500" },
  sprout: { emoji: "🌿", color: "text-green-400" },
  growing: { emoji: "🌳", color: "text-green-500" },
  blooming: { emoji: "🌸", color: "text-pink-500" },
  flourishing: { emoji: "🌺", color: "text-purple-500" },
};

function BadgeIcon({ icon, className }: { icon: string; className?: string }) {
  const IconComponent = ICON_MAP[icon] || Trophy;
  return <IconComponent className={className} />;
}

function BadgeCard({ badge, isLocked }: { badge: Badge; isLocked: boolean }) {
  const rarityClass = RARITY_COLORS[badge.rarity] || RARITY_COLORS.common;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          className={`relative p-3 rounded-lg border transition-all ${
            isLocked
              ? "bg-muted/50 border-muted opacity-60"
              : "bg-card border-border hover-elevate"
          }`}
          data-testid={`badge-${badge.id}-${isLocked ? "locked" : "unlocked"}`}
        >
          <div className="flex flex-col items-center gap-2">
            <div
              className={`p-2 rounded-full ${
                isLocked ? "bg-muted" : "bg-primary/10"
              }`}
            >
              {isLocked ? (
                <Lock className="w-5 h-5 text-muted-foreground" />
              ) : (
                <BadgeIcon
                  icon={badge.icon}
                  className="w-5 h-5 text-primary"
                />
              )}
            </div>
            <span className="text-xs font-medium text-center line-clamp-2">
              {badge.name}
            </span>
            <Badge variant="outline" className={`text-[10px] ${rarityClass}`}>
              {badge.rarity}
            </Badge>
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-[200px]">
        <p className="font-medium">{badge.name}</p>
        <p className="text-xs text-muted-foreground">{badge.description}</p>
        <p className="text-xs mt-1">+{badge.xp_reward} XP</p>
      </TooltipContent>
    </Tooltip>
  );
}

export function HabitGamificationPanel() {
  const { data, isLoading, error } = useQuery<GamificationState>({
    queryKey: ["/api/habits/gamification/state"],
    queryFn: async () => {
      const res = await fetch("/api/habits/gamification/state", {
        credentials: "include",
      });
      if (!res.ok) throw new Error("Failed to fetch gamification state");
      return res.json();
    },
  });

  if (isLoading) {
    return (
      <Card data-testid="gamification-panel-loading">
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-muted rounded w-1/3" />
            <div className="h-4 bg-muted rounded w-full" />
            <div className="grid grid-cols-4 gap-2">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="h-20 bg-muted rounded" />
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card data-testid="gamification-panel-error">
        <CardContent className="p-6 text-center text-muted-foreground">
          Unable to load gamification data
        </CardContent>
      </Card>
    );
  }

  const growthVisual = GROWTH_STAGE_VISUALS[data.growthStage] || GROWTH_STAGE_VISUALS.seed;

  return (
    <Card data-testid="gamification-panel">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Trophy className="w-5 h-5 text-primary" />
            <span>Your Progress</span>
          </div>
          <span className={`text-2xl ${growthVisual.color}`}>
            {growthVisual.emoji}
          </span>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-2" data-testid="level-progress">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary text-primary-foreground font-bold">
                {data.level.current}
              </div>
              <div>
                <p className="font-semibold" data-testid="text-level-name">
                  {data.level.name}
                </p>
                <p className="text-xs text-muted-foreground">
                  {data.totalXp.toLocaleString()} XP total
                </p>
              </div>
            </div>
            {data.level.nextLevelName && (
              <div className="text-right text-sm text-muted-foreground">
                <p>Next: {data.level.nextLevelName}</p>
                <p className="text-xs">
                  {data.level.xpForNextLevel
                    ? `${(data.level.xpForNextLevel - data.level.xpInLevel).toLocaleString()} XP to go`
                    : "Max level!"}
                </p>
              </div>
            )}
          </div>
          <Progress
            value={data.level.progress * 100}
            className="h-2"
            data-testid="progress-level"
          />
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="text-center p-3 rounded-lg bg-muted/50">
            <p className="text-2xl font-bold text-primary" data-testid="text-days-active">
              {data.daysActive}
            </p>
            <p className="text-xs text-muted-foreground">Days Active</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-muted/50">
            <p className="text-2xl font-bold text-orange-500" data-testid="text-streak-bonus">
              {data.streakBonus}
            </p>
            <p className="text-xs text-muted-foreground">Streak Bonus</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-muted/50">
            <p className="text-2xl font-bold text-green-500" data-testid="text-perfect-days">
              {data.perfectDays}
            </p>
            <p className="text-xs text-muted-foreground">Perfect Days</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-muted/50">
            <p className="text-2xl font-bold text-purple-500" data-testid="text-badges-count">
              {data.badges.totalUnlocked}/{data.badges.totalAvailable}
            </p>
            <p className="text-xs text-muted-foreground">Badges</p>
          </div>
        </div>

        {data.nextMilestones.length > 0 && (
          <div className="space-y-2" data-testid="next-milestones">
            <h4 className="text-sm font-medium">Next Milestones</h4>
            <div className="space-y-2">
              {data.nextMilestones.map((milestone, i) => (
                <div
                  key={i}
                  className="flex items-center gap-3 p-2 rounded-lg bg-muted/30"
                >
                  <BadgeIcon
                    icon={milestone.badge.icon}
                    className="w-4 h-4 text-muted-foreground"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {milestone.badge.name}
                    </p>
                    <Progress
                      value={milestone.progress * 100}
                      className="h-1 mt-1"
                    />
                  </div>
                  <span className="text-xs text-muted-foreground whitespace-nowrap">
                    {milestone.remaining} to go
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <Tabs defaultValue="unlocked" className="w-full">
          <TabsList className="w-full">
            <TabsTrigger value="unlocked" className="flex-1" data-testid="tab-badges-unlocked">
              Unlocked ({data.badges.totalUnlocked})
            </TabsTrigger>
            <TabsTrigger value="locked" className="flex-1" data-testid="tab-badges-locked">
              Locked ({data.badges.locked.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="unlocked">
            <ScrollArea className="h-[200px]">
              {data.badges.unlocked.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Trophy className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No badges unlocked yet</p>
                  <p className="text-xs">Complete habits to earn your first badge!</p>
                </div>
              ) : (
                <div className="grid grid-cols-4 gap-2 p-1">
                  {data.badges.unlocked.map((badge) => (
                    <BadgeCard key={badge.id} badge={badge} isLocked={false} />
                  ))}
                </div>
              )}
            </ScrollArea>
          </TabsContent>

          <TabsContent value="locked">
            <ScrollArea className="h-[200px]">
              <div className="grid grid-cols-4 gap-2 p-1">
                {data.badges.locked.map((badge) => (
                  <BadgeCard key={badge.id} badge={badge} isLocked={true} />
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

export default HabitGamificationPanel;
