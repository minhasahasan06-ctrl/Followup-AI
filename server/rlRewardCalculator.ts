/**
 * Reinforcement Learning Reward Calculator
 * 
 * Calculates rewards from user interactions to train ML models
 * Uses epsilon-greedy exploration strategy
 */

import type { DailyEngagement, HabitCompletion, MLRecommendation } from '@shared/schema';

export interface RLState {
  userContext: {
    currentStreak: number;
    habitsCompleted: number;
    engagementScore: number;
    recentSentiment: number;
  };
  conversationContext: string[];
  healthMetrics: Record<string, any>;
  recentActions: string[];
}

export interface RLAction {
  type: string;
  content: string;
  parameters: Record<string, any>;
}

export interface RewardFactors {
  engagement: number;
  completion: number;
  satisfaction: number;
  healthOutcome: number;
}

export class RLRewardCalculator {
  // Reward weights (tunable hyperparameters)
  private weights = {
    engagement: 0.25,
    completion: 0.30,
    satisfaction: 0.25,
    healthOutcome: 0.20,
  };

  // Epsilon for epsilon-greedy exploration
  private epsilon = 0.1;

  /**
   * Calculate reward for a conversation interaction
   */
  calculateConversationReward(
    state: RLState,
    action: RLAction,
    userFeedback?: 'helpful' | 'not_helpful' | 'irrelevant'
  ): number {
    const factors: RewardFactors = {
      engagement: this.calculateEngagementReward(state, action),
      completion: 0, // Not applicable for conversations
      satisfaction: this.calculateSatisfactionReward(userFeedback),
      healthOutcome: 0, // Measured over longer time horizons
    };

    return this.computeWeightedReward(factors);
  }

  /**
   * Calculate reward for a habit completion
   */
  calculateHabitCompletionReward(
    state: RLState,
    completion: HabitCompletion,
    streakIncreased: boolean
  ): number {
    const factors: RewardFactors = {
      engagement: streakIncreased ? 0.5 : 0.2, // Higher reward for maintaining streaks
      completion: 1.0, // Full reward for completing habit
      satisfaction: this.mapMoodToSatisfaction(completion.mood),
      healthOutcome: 0.3, // Partial immediate health outcome
    };

    // Bonus for difficulty
    const difficultyBonus = (completion.difficultyLevel || 3) / 5;
    factors.completion *= (0.8 + 0.4 * difficultyBonus); // 0.8 to 1.2 multiplier

    return this.computeWeightedReward(factors);
  }

  /**
   * Calculate reward for recommendation acceptance/completion
   */
  calculateRecommendationReward(
    recommendation: MLRecommendation,
    status: 'accepted' | 'declined' | 'completed' | 'dismissed'
  ): number {
    const statusRewards = {
      accepted: 0.5,    // Positive signal
      declined: -0.2,   // Negative signal
      completed: 1.0,   // Very positive signal
      dismissed: -0.1,  // Small negative signal
    };

    const baseReward = statusRewards[status];

    // Apply confidence score multiplier
    const confidenceMultiplier = parseFloat(String(recommendation.confidenceScore)) || 0.5;

    // Apply personalization score multiplier
    const personalizationMultiplier = parseFloat(String(recommendation.personalizationScore)) || 0.5;

    return baseReward * confidenceMultiplier * personalizationMultiplier;
  }

  /**
   * Calculate reward for daily engagement
   */
  calculateDailyEngagementReward(
    currentEngagement: DailyEngagement,
    previousEngagement?: DailyEngagement
  ): number {
    const factors: RewardFactors = {
      engagement: this.normalizeEngagementScore(currentEngagement.engagementScore || 0),
      completion: this.normalizeHabitCompletion(
        currentEngagement.habitsCompleted || 0,
        currentEngagement.habitsSkipped || 0
      ),
      satisfaction: this.normalizeSentiment(parseFloat(String(currentEngagement.clonaSentiment)) || 0),
      healthOutcome: 0, // Calculated separately
    };

    // Bonus for streak continuation
    if (currentEngagement.isStreakDay) {
      factors.engagement += 0.2;
    }

    // Bonus for improvement over previous day
    if (previousEngagement) {
      const improvementBonus = this.calculateImprovementBonus(currentEngagement, previousEngagement);
      factors.engagement += improvementBonus;
    }

    return this.computeWeightedReward(factors);
  }

  /**
   * Calculate reward for milestone achievement
   */
  calculateMilestoneReward(
    milestoneType: string,
    targetValue: number,
    category: string
  ): number {
    const baseRewards = {
      streak: 0.8,
      total_completions: 0.6,
      health_goal: 1.0,
      engagement: 0.5,
      wellness: 0.7,
      doctor_wellness: 0.7,
    };

    const baseReward = baseRewards[milestoneType as keyof typeof baseRewards] || 0.5;

    // Scale reward by target difficulty
    const difficultyMultiplier = Math.min(1 + Math.log10(targetValue) / 2, 2);

    return baseReward * difficultyMultiplier;
  }

  /**
   * Calculate exploration bonus (epsilon-greedy)
   */
  shouldExplore(): boolean {
    return Math.random() < this.epsilon;
  }

  /**
   * Update epsilon (decay over time for less exploration)
   */
  updateEpsilon(decay: number = 0.995) {
    this.epsilon = Math.max(0.01, this.epsilon * decay);
  }

  /**
   * Private helper methods
   */
  private calculateEngagementReward(state: RLState, action: RLAction): number {
    // Reward based on engagement score
    return this.normalizeEngagementScore(state.userContext.engagementScore);
  }

  private calculateSatisfactionReward(feedback?: string): number {
    if (!feedback) return 0.5; // Neutral

    const feedbackRewards = {
      helpful: 1.0,
      not_helpful: 0.0,
      irrelevant: -0.2,
    };

    return feedbackRewards[feedback as keyof typeof feedbackRewards] || 0.5;
  }

  private mapMoodToSatisfaction(mood?: string | null): number {
    const moodScores = {
      great: 1.0,
      good: 0.7,
      okay: 0.4,
      struggling: 0.1,
    };

    return mood ? (moodScores[mood as keyof typeof moodScores] || 0.5) : 0.5;
  }

  private normalizeEngagementScore(score: number): number {
    // Engagement score is 0-100, normalize to 0-1
    return Math.min(Math.max(score / 100, 0), 1);
  }

  private normalizeHabitCompletion(completed: number, skipped: number): number {
    const total = completed + skipped;
    if (total === 0) return 0.5; // Neutral if no habits

    return completed / total;
  }

  private normalizeSentiment(sentiment: number): number {
    // Sentiment is -1 to 1, normalize to 0 to 1
    return (sentiment + 1) / 2;
  }

  private calculateImprovementBonus(
    current: DailyEngagement,
    previous: DailyEngagement
  ): number {
    const currentScore = current.engagementScore || 0;
    const previousScore = previous.engagementScore || 0;

    if (currentScore > previousScore) {
      return Math.min((currentScore - previousScore) / 100, 0.3);
    }

    return 0;
  }

  private computeWeightedReward(factors: RewardFactors): number {
    const reward =
      factors.engagement * this.weights.engagement +
      factors.completion * this.weights.completion +
      factors.satisfaction * this.weights.satisfaction +
      factors.healthOutcome * this.weights.healthOutcome;

    // Clip reward to [-1, 1]
    return Math.max(-1, Math.min(1, reward));
  }

  /**
   * Batch reward calculation for training
   */
  async calculateBatchRewards(
    interactions: Array<{
      state: RLState;
      action: RLAction;
      outcome: any;
    }>
  ): Promise<number[]> {
    return interactions.map(interaction => {
      // Determine reward based on interaction type
      if (interaction.outcome.type === 'habit_completion') {
        return this.calculateHabitCompletionReward(
          interaction.state,
          interaction.outcome.completion,
          interaction.outcome.streakIncreased
        );
      } else if (interaction.outcome.type === 'recommendation') {
        return this.calculateRecommendationReward(
          interaction.outcome.recommendation,
          interaction.outcome.status
        );
      } else if (interaction.outcome.type === 'conversation') {
        return this.calculateConversationReward(
          interaction.state,
          interaction.action,
          interaction.outcome.feedback
        );
      }

      return 0;
    });
  }

  /**
   * Calculate cumulative reward for an episode
   */
  calculateEpisodeReturn(
    rewards: number[],
    discountFactor: number = 0.99
  ): number {
    let discountedReturn = 0;
    let discount = 1;

    for (const reward of rewards) {
      discountedReturn += reward * discount;
      discount *= discountFactor;
    }

    return discountedReturn;
  }
}

// Export singleton instance
export const rlRewardCalculator = new RLRewardCalculator();
