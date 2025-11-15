"""
Respiratory Rate Estimation Service
Analyzes chest movement from video to estimate breathing rate
Uses AI-powered video analysis (no diagnosis)
"""

import base64
from typing import Dict, List, Optional
from app.services.openai_service import OpenAIService


class RespiratoryAnalysisService:
    """
    Estimates respiratory rate from video frames
    IMPORTANT: This provides monitoring estimates, NOT clinical diagnoses
    """
    
    def __init__(self):
        self.openai_service = OpenAIService()
    
    async def analyze_chest_movement(
        self,
        video_frames: List[str],  # Base64 encoded frames
        duration_seconds: float
    ) -> Dict:
        """
        Analyze chest movement from video frames to estimate respiratory rate
        
        Args:
            video_frames: List of base64 encoded video frames
            duration_seconds: Duration of the video recording
            
        Returns:
            Dictionary with respiratory rate estimate and confidence
        """
        # Sample frames evenly throughout the video
        sampled_frames = self._sample_frames(video_frames, max_frames=10)
        
        # Prepare prompt for AI analysis
        prompt = f"""Analyze this {duration_seconds}-second video of chest/torso movement to estimate respiratory rate.

IMPORTANT INSTRUCTIONS:
1. Count visible chest rise/fall movements
2. Look for rhythmic expansion and contraction
3. Estimate breaths per minute (BPM)
4. Assess movement quality (regular/irregular, shallow/deep)
5. Note any visible respiratory effort

Video contains {len(sampled_frames)} frames over {duration_seconds} seconds.

Respond in JSON format:
{{
    "chest_movements_detected": <number>,
    "estimated_respiratory_rate_bpm": <number>,
    "confidence_score": <0-1>,
    "movement_pattern": "regular|irregular|shallow|deep",
    "respiratory_effort": "normal|increased|decreased",
    "observations": "<brief non-diagnostic observations>",
    "quality_assessment": "good|acceptable|poor",
    "quality_issues": ["<list any visibility/quality issues>"]
}}

Remember: Provide monitoring observations ONLY, not medical diagnoses."""
        
        try:
            # Call OpenAI Vision API with sampled frames
            response = await self.openai_service.analyze_image_with_context(
                image_data=sampled_frames[0],  # Primary frame
                context=prompt,
                additional_images=sampled_frames[1:] if len(sampled_frames) > 1 else []
            )
            
            # Parse response
            import json
            result = json.loads(response)
            
            # Add metadata
            result["analysis_duration_seconds"] = duration_seconds
            result["frames_analyzed"] = len(sampled_frames)
            result["method"] = "ai_video_analysis"
            
            return result
            
        except Exception as e:
            # Return fallback response if AI analysis fails
            return {
                "chest_movements_detected": 0,
                "estimated_respiratory_rate_bpm": None,
                "confidence_score": 0.0,
                "movement_pattern": "unknown",
                "respiratory_effort": "unknown",
                "observations": f"Analysis could not be completed: {str(e)}",
                "quality_assessment": "poor",
                "quality_issues": ["analysis_error"],
                "error": str(e)
            }
    
    def _sample_frames(self, frames: List[str], max_frames: int = 10) -> List[str]:
        """Sample frames evenly from the video"""
        if len(frames) <= max_frames:
            return frames
        
        # Sample evenly
        step = len(frames) / max_frames
        sampled = [frames[int(i * step)] for i in range(max_frames)]
        return sampled
    
    def calculate_respiratory_rate(
        self,
        movement_count: int,
        duration_seconds: float
    ) -> float:
        """
        Calculate breaths per minute from movement count
        
        Args:
            movement_count: Number of chest rise/fall cycles detected
            duration_seconds: Duration of observation
            
        Returns:
            Respiratory rate in breaths per minute
        """
        if duration_seconds == 0:
            return 0.0
        
        # Convert to breaths per minute
        breaths_per_minute = (movement_count / duration_seconds) * 60
        
        # Round to 1 decimal place
        return round(breaths_per_minute, 1)
    
    def assess_respiratory_pattern(
        self,
        respiratory_rate: float,
        patient_age: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Assess respiratory pattern (monitoring context, NOT diagnosis)
        
        Args:
            respiratory_rate: Breaths per minute
            patient_age: Patient age (optional, for context)
            
        Returns:
            Assessment with monitoring observations
        """
        assessment = {
            "rate_category": "",
            "observation": "",
            "monitoring_note": ""
        }
        
        # Adult normal range: 12-20 BPM
        # These are reference ranges for monitoring, NOT diagnostic thresholds
        if respiratory_rate < 12:
            assessment["rate_category"] = "below_typical_range"
            assessment["observation"] = "Breathing rate below typical adult range"
            assessment["monitoring_note"] = "Consider discussing with healthcare provider"
        elif 12 <= respiratory_rate <= 20:
            assessment["rate_category"] = "within_typical_range"
            assessment["observation"] = "Breathing rate within typical adult range"
            assessment["monitoring_note"] = "Continue routine monitoring"
        elif 20 < respiratory_rate <= 25:
            assessment["rate_category"] = "slightly_elevated"
            assessment["observation"] = "Breathing rate slightly above typical range"
            assessment["monitoring_note"] = "Monitor for changes, discuss if persistent"
        else:
            assessment["rate_category"] = "notably_elevated"
            assessment["observation"] = "Breathing rate notably above typical range"
            assessment["monitoring_note"] = "Recommend prompt healthcare provider discussion"
        
        return assessment
    
    def compare_respiratory_trends(
        self,
        current_rate: float,
        previous_rates: List[float],
        days_range: int = 7
    ) -> Dict:
        """
        Compare current respiratory rate to recent history
        
        Args:
            current_rate: Current respiratory rate
            previous_rates: List of previous measurements
            days_range: Number of days to analyze
            
        Returns:
            Trend analysis
        """
        if not previous_rates:
            return {
                "trend": "no_baseline",
                "change_percent": 0.0,
                "observation": "First measurement - establishing baseline"
            }
        
        # Calculate average of previous measurements
        avg_previous = sum(previous_rates) / len(previous_rates)
        
        # Calculate percent change
        if avg_previous > 0:
            change_percent = ((current_rate - avg_previous) / avg_previous) * 100
        else:
            change_percent = 0.0
        
        # Determine trend
        if abs(change_percent) < 10:
            trend = "stable"
            observation = f"Respiratory rate stable (within 10% of {days_range}-day average)"
        elif change_percent >= 10:
            trend = "increasing"
            observation = f"Respiratory rate increased {abs(change_percent):.1f}% from {days_range}-day average"
        else:
            trend = "decreasing"
            observation = f"Respiratory rate decreased {abs(change_percent):.1f}% from {days_range}-day average"
        
        return {
            "trend": trend,
            "current_rate": current_rate,
            "average_previous": round(avg_previous, 1),
            "change_percent": round(change_percent, 1),
            "observation": observation,
            "measurements_compared": len(previous_rates)
        }
