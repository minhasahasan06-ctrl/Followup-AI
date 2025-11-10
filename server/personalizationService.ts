/**
 * Personalization Service - Rule-Based Recommendation & RAG Integration
 * 
 * Provides personalized recommendations using rule-based heuristics and
 * integrates learned preferences into OpenAI agent prompts via RAG.
 * 
 * Note: This is NOT a deep learning system. It uses simple, explainable
 * rules that actually work rather than untrained neural networks.
 */

import { storage } from './storage';
import { rlRewardCalculator } from './rlRewardCalculator';
import type { UserLearningProfile, MLRecommendation, Habit } from '@shared/schema';

/**
 * User Preference Tracker
 * Learns user preferences from interactions without pretending to use ML
 */
export class PreferenceTracker {
  /**
   * Update user learning profile based on interaction
   */
  async trackInteraction(
    userId: string,
    agentType: 'clona' | 'lysa',
    interaction: {
      type: string;
      liked?: boolean;
      category?: string;
      sentiment?: number;
    }
  ): Promise<void> {
    const profile = await storage.getUserLearningProfile(userId, agentType) || {
      userId,
      agentType,
      totalInteractions: 0,
      avgSentiment: 0,
      favoriteTopics: [],
      strugglingAreas: [],
      motivationalStyle: 'encouragement',
      lastUpdated: new Date(),
    };

    // Update interaction count
    profile.totalInteractions = (profile.totalInteractions || 0) + 1;

    // Update sentiment (moving average)
    if (interaction.sentiment !== undefined) {
      const oldSentiment = profile.avgSentiment || 0;
      profile.avgSentiment = (oldSentiment * 0.9 + interaction.sentiment * 0.1);
    }

    // Track liked categories
    if (interaction.liked && interaction.category) {
      const favorites = profile.favoriteTopics || [];
      if (!favorites.includes(interaction.category)) {
        favorites.push(interaction.category);
      }
      profile.favoriteTopics = favorites;
    }

    // Track struggling areas (from negative sentiment)
    if (interaction.sentiment !== undefined && interaction.sentiment < -0.3) {
      const struggling = profile.strugglingAreas || [];
      if (interaction.category && !struggling.includes(interaction.category)) {
        struggling.push(interaction.category);
      }
      profile.strugglingAreas = struggling;
    }

    await storage.upsertUserLearningProfile(profile);
  }

  /**
   * Get personalized context for RAG integration with OpenAI
   */
  async getPersonalizedContext(userId: string, agentType: 'clona' | 'lysa'): Promise<string> {
    const profile = await storage.getUserLearningProfile(userId, agentType);
    if (!profile) {
      return '';
    }

    const context: string[] = [];

    // Add preference insights
    if (profile.favoriteTopics && profile.favoriteTopics.length > 0) {
      context.push(`User prefers topics: ${profile.favoriteTopics.join(', ')}`);
    }

    if (profile.strugglingAreas && profile.strugglingAreas.length > 0) {
      context.push(`User struggles with: ${profile.strugglingAreas.join(', ')}`);
    }

    // Add motivational style
    if (profile.motivationalStyle) {
      context.push(`Motivational style: ${profile.motivationalStyle}`);
    }

    // Add sentiment trend
    const sentiment = profile.avgSentiment || 0;
    if (sentiment > 0.3) {
      context.push('User is generally positive and engaged');
    } else if (sentiment < -0.3) {
      context.push('User may need extra support and encouragement');
    }

    // Add habit insights for Clona
    if (agentType === 'clona') {
      const habits = await storage.getHabits(userId);
      if (habits.length > 0) {
        const activeHabits = habits.filter(h => h.currentStreak && h.currentStreak > 0);
        context.push(`Active habits: ${activeHabits.length}/${habits.length}`);
        
        const maxStreak = Math.max(...habits.map(h => h.currentStreak || 0));
        if (maxStreak > 7) {
          context.push(`Longest streak: ${maxStreak} days`);
        }
      }
    }

    return context.join('. ');
  }
}

/**
 * Rule-Based Recommendation Engine
 * Uses simple, explainable heuristics instead of untrained ML models
 */
export class RecommendationEngine {
  /**
   * Generate recommendations for Agent Clona (patients)
   */
  async generateClonaRecommendations(userId: string, limit: number = 5): Promise<Partial<MLRecommendation>[]> {
    const [profile, habits, completions] = await Promise.all([
      storage.getUserLearningProfile(userId, 'clona'),
      storage.getHabits(userId),
      storage.getRecentHabitCompletions(userId, 30),
    ]);

    const recommendations: Partial<MLRecommendation>[] = [];

    // Rule 1: Suggest new habits if user has high completion rate
    if (habits.length > 0 && completions.length > 0) {
      const completionRate = completions.length / (habits.length * 30);
      
      if (completionRate > 0.7) {
        recommendations.push({
          userId,
          agentType: 'clona',
          type: 'habit',
          category: 'wellness',
          title: 'Ready for a new challenge?',
          description: 'You\'ve been crushing your current habits! Consider adding a meditation or journaling practice.',
          confidenceScore: '0.85',
          personalizationScore: '0.90',
          priority: 'medium',
          reasoning: `High completion rate (${(completionRate * 100).toFixed(0)}%)`,
        });
      }
    }

    // Rule 2: Encourage struggling users
    if (profile && profile.avgSentiment && profile.avgSentiment < -0.2) {
      recommendations.push({
        userId,
        agentType: 'clona',
        type: 'health_tip',
        category: 'wellness',
        title: 'You\'re doing better than you think',
        description: 'Small steps every day add up to big changes. Be kind to yourself.',
        confidenceScore: '0.75',
        personalizationScore: '0.95',
        priority: 'high',
        reasoning: 'User sentiment suggests need for encouragement',
      });
    }

    // Rule 3: Streak milestones
    for (const habit of habits) {
      if (habit.currentStreak === 7 || habit.currentStreak === 30 || habit.currentStreak === 90) {
        recommendations.push({
          userId,
          agentType: 'clona',
          type: 'achievement',
          category: 'wellness',
          title: `${habit.currentStreak}-day streak on ${habit.name}!`,
          description: 'This is a major milestone. Celebrate your consistency!',
          confidenceScore: '1.0',
          personalizationScore: '1.0',
          priority: 'high',
          reasoning: 'Milestone achievement',
        });
      }
    }

    // Rule 4: Personalized wellness activities
    const favoriteTopics = profile?.favoriteTopics || [];
    if (favoriteTopics.includes('exercise')) {
      recommendations.push({
        userId,
        agentType: 'clona',
        type: 'activity',
        category: 'exercise',
        title: 'Try a new workout',
        description: 'Since you enjoy exercise, explore yoga or swimming this week',
        confidenceScore: '0.70',
        personalizationScore: '0.85',
        priority: 'low',
        reasoning: 'Based on interest in exercise',
      });
    }

    return recommendations.slice(0, limit);
  }

  /**
   * Generate recommendations for Assistant Lysa (doctors)
   */
  async generateLysaRecommendations(userId: string, limit: number = 5): Promise<Partial<MLRecommendation>[]> {
    const [profile, wellnessHistory] = await Promise.all([
      storage.getUserLearningProfile(userId, 'lysa'),
      storage.getDoctorWellnessHistory(userId, 7),
    ]);

    const recommendations: Partial<MLRecommendation>[] = [];

    // Rule 1: Doctor wellness check
    if (wellnessHistory.length > 0) {
      const avgStress = wellnessHistory.reduce((sum, w) => sum + (w.stressLevel || 0), 0) / wellnessHistory.length;
      
      if (avgStress > 7) {
        recommendations.push({
          userId,
          agentType: 'lysa',
          type: 'activity',
          category: 'wellness',
          title: 'High Stress Alert',
          description: 'Your stress levels have been elevated. Consider a mindfulness break or brief walk.',
          confidenceScore: '0.90',
          personalizationScore: '0.95',
          priority: 'high',
          reasoning: `Average stress level: ${avgStress.toFixed(1)}/10`,
        });
      }
    }

    // Rule 2: Research recommendations based on specialty
    const researchInterests = profile?.researchInterests || [];
    if (researchInterests.length > 0) {
      recommendations.push({
        userId,
        agentType: 'lysa',
        type: 'research',
        category: 'research',
        title: `New research in ${researchInterests[0]}`,
        description: 'Recent publications matching your interests',
        confidenceScore: '0.75',
        personalizationScore: '0.85',
        priority: 'medium',
        reasoning: 'Based on research interests',
      });
    }

    // Rule 3: Clinical protocol suggestions
    recommendations.push({
      userId,
      agentType: 'lysa',
      type: 'protocol',
      category: 'clinical',
      title: 'Updated Treatment Guidelines',
      description: 'Review recent updates to immunocompromised patient protocols',
      confidenceScore: '0.80',
      personalizationScore: '0.70',
      priority: 'medium',
      reasoning: 'Routine protocol review',
    });

    return recommendations.slice(0, limit);
  }
}

/**
 * OpenAI RAG Integration Helper
 * Injects learned preferences into agent prompts
 */
export class RAGIntegration {
  private preferenceTracker: PreferenceTracker;

  constructor() {
    this.preferenceTracker = new PreferenceTracker();
  }

  /**
   * Enhance OpenAI prompt with personalized context
   */
  async enhancePrompt(
    userId: string,
    agentType: 'clona' | 'lysa',
    basePrompt: string
  ): Promise<string> {
    const personalizedContext = await this.preferenceTracker.getPersonalizedContext(userId, agentType);

    if (!personalizedContext) {
      return basePrompt;
    }

    // Inject personalization into system prompt
    const enhancedPrompt = `${basePrompt}

PERSONALIZATION CONTEXT:
${personalizedContext}

Use this context to personalize your responses while maintaining your core personality and medical accuracy.`;

    return enhancedPrompt;
  }

  /**
   * Process user feedback to improve future recommendations
   */
  async processFeedback(
    userId: string,
    agentType: 'clona' | 'lysa',
    feedback: {
      messageId?: string;
      helpful?: boolean;
      sentiment?: number;
      category?: string;
    }
  ): Promise<void> {
    // Track the interaction
    await this.preferenceTracker.trackInteraction(userId, agentType, {
      type: 'feedback',
      liked: feedback.helpful,
      sentiment: feedback.sentiment,
      category: feedback.category,
    });

    // Calculate RL reward if helpful/not helpful
    if (feedback.helpful !== undefined) {
      const reward = rlRewardCalculator.calculateConversationReward(
        {
          userContext: {
            currentStreak: 0,
            habitsCompleted: 0,
            engagementScore: 0,
            recentSentiment: feedback.sentiment || 0,
          },
          conversationContext: [],
          healthMetrics: {},
          recentActions: [],
        },
        {
          type: 'response',
          content: '',
          parameters: {},
        },
        feedback.helpful ? 'helpful' : 'not_helpful'
      );

      // Store the reward for future analysis
      await storage.createRLReward({
        userId,
        agentType,
        rewardValue: reward.toString(),
        actionType: 'conversation',
        stateSnapshot: {},
        actionSnapshot: {},
        timestamp: new Date(),
      });
    }
  }
}

// Export singleton instances
export const personalizationService = {
  preferenceTracker: new PreferenceTracker(),
  recommendationEngine: new RecommendationEngine(),
  ragIntegration: new RAGIntegration(),

  /**
   * Get personalized recommendations
   */
  async getRecommendations(userId: string, agentType: 'clona' | 'lysa', limit?: number) {
    if (agentType === 'clona') {
      return this.recommendationEngine.generateClonaRecommendations(userId, limit);
    } else {
      return this.recommendationEngine.generateLysaRecommendations(userId, limit);
    }
  },

  /**
   * Enhance agent prompt with personalization
   */
  async personalizeAgentPrompt(userId: string, agentType: 'clona' | 'lysa', basePrompt: string): Promise<string> {
    return this.ragIntegration.enhancePrompt(userId, agentType, basePrompt);
  },

  /**
   * Track user interaction for learning
   */
  async trackUserInteraction(userId: string, agentType: 'clona' | 'lysa', interaction: any): Promise<void> {
    await this.preferenceTracker.trackInteraction(userId, agentType, interaction);
  },

  /**
   * Process feedback to improve recommendations
   */
  async processFeedback(userId: string, agentType: 'clona' | 'lysa', feedback: any): Promise<void> {
    await this.ragIntegration.processFeedback(userId, agentType, feedback);
  },
};
