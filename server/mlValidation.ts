/**
 * Zod validation schemas for ML/RL API endpoints
 * Ensures HIPAA-compliant input sanitization before storage
 */

import { z } from 'zod';

// Sanitization helpers with prompt injection prevention
const sanitizeText = (text: string): string => {
  // Strip HTML tags, control characters, and limit length
  // Remove newlines, carriage returns, tabs to prevent prompt injection
  return text
    .replace(/<[^>]*>/g, '')                    // Remove HTML
    .replace(/[\x00-\x1F\x7F]/g, '')           // Remove control characters
    .replace(/(\r\n|\n|\r)/g, ' ')             // Replace newlines with spaces
    .replace(/\s+/g, ' ')                      // Normalize whitespace
    .trim()
    .slice(0, 2000);
};

const sanitizeName = (name: string): string => {
  // Allow only alphanumeric, spaces, dashes, underscores
  return name.replace(/[^a-zA-Z0-9\s\-_]/g, '').slice(0, 100);
};

// POST /api/v1/ml/habits
export const createHabitSchema = z.object({
  name: z.string().min(1).max(100).transform(sanitizeName),
  description: z.string().max(500).optional().transform(val => val ? sanitizeText(val) : undefined),
  category: z.enum(['health', 'medication', 'exercise', 'wellness', 'nutrition', 'sleep', 'other']),
  frequency: z.enum(['daily', 'weekly', 'custom']),
  goalCount: z.number().int().min(1).max(10).optional().default(1),
});

// POST /api/v1/ml/habits/:id/complete
export const completeHabitSchema = z.object({
  mood: z.enum(['great', 'good', 'okay', 'struggling']).optional(),
  notes: z.string().max(500).optional().transform(val => val ? sanitizeText(val) : undefined),
  difficultyLevel: z.number().int().min(1).max(5).optional().default(3),
});

// POST /api/v1/ml/feedback
export const feedbackSchema = z.object({
  agentType: z.enum(['clona', 'lysa']),
  helpful: z.boolean().optional(),
  sentiment: z.number().min(-1).max(1).optional(),
  category: z.string().max(50).optional().transform(val => val ? sanitizeName(val) : undefined),
  messageId: z.string().uuid().optional(),
});

// POST /api/v1/ml/agent-prompts
export const agentPromptSchema = z.object({
  agentType: z.enum(['clona', 'lysa']),
  basePrompt: z.string().min(10).max(5000).transform(sanitizeText),
});

// POST /api/v1/ml/doctor-wellness
export const doctorWellnessSchema = z.object({
  stressLevel: z.number().int().min(1).max(10),
  hoursWorked: z.number().min(0).max(24),
  patientsToday: z.number().int().min(0).max(100),
  burnoutRisk: z.enum(['low', 'moderate', 'high', 'critical']).optional(),
  notes: z.string().max(1000).optional().transform(val => val ? sanitizeText(val) : undefined),
});

// Type exports
export type CreateHabitInput = z.infer<typeof createHabitSchema>;
export type CompleteHabitInput = z.infer<typeof completeHabitSchema>;
export type FeedbackInput = z.infer<typeof feedbackSchema>;
export type AgentPromptInput = z.infer<typeof agentPromptSchema>;
export type DoctorWellnessInput = z.infer<typeof doctorWellnessSchema>;
