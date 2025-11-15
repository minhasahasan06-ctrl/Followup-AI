"""
OpenAI Service for symptom image analysis
Uses GPT-4o Vision API for non-diagnostic change detection
"""

import os
import base64
from typing import Dict, List, Optional
from openai import AsyncOpenAI

from app.models.symptom_journal import BodyArea


# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYMPTOM_ANALYSIS_PROMPT = """You are a medical image analysis assistant helping patients track visual changes over time.

IMPORTANT: You are NOT making medical diagnoses. You are only describing observable changes in appearance.

Analyze this image of the patient's {body_area} and provide:
1. Objective observations about color, appearance, and visible features
2. If previous observations are provided, note any changes

Use non-diagnostic language:
- Instead of "jaundice", say "yellowish coloration change"
- Instead of "edema", say "increased swelling or puffiness"
- Instead of "anemia", say "paler color compared to baseline"
- Instead of "cyanosis", say "bluish tint observed"

Focus on:
- Color changes (overall tone, brightness, uniformity)
- Size/shape changes (swelling, asymmetry)
- Texture changes (smoothness, roughness)
- Visible patterns

Always end with: "These observations are for tracking purposes only. Please discuss any concerns with your healthcare provider."

Previous observations (if any):
{previous_observations}

Provide your response in JSON format:
{{
    "observations": "detailed objective observations",
    "detected_changes": [
        {{
            "type": "color|swelling|texture",
            "description": "specific change observed",
            "severity": "mild|moderate|significant"
        }}
    ]
}}
"""


async def analyze_symptom_image(
    image_data: bytes,
    body_area: BodyArea,
    previous_observations: Optional[str] = None
) -> Dict:
    """
    Analyze symptom image using OpenAI Vision API
    Returns non-diagnostic observations and detected changes
    """
    try:
        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare prompt
        prompt = SYMPTOM_ANALYSIS_PROMPT.format(
            body_area=body_area.value,
            previous_observations=previous_observations or "None"
        )
        
        # Call OpenAI Vision API
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical image analysis assistant focused on change detection, not diagnosis."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        
        # Parse response
        result = response.choices[0].message.content
        
        import json
        parsed_result = json.loads(result)
        
        return {
            "observations": parsed_result.get("observations", ""),
            "detected_changes": parsed_result.get("detected_changes", [])
        }
        
    except Exception as e:
        print(f"Error analyzing image with OpenAI: {e}")
        return {
            "observations": "Unable to analyze image at this time.",
            "detected_changes": []
        }


async def generate_weekly_summary(
    patient_id: str,
    measurements: List[Dict],
    alerts: List[Dict]
) -> str:
    """
    Generate a structured weekly summary for clinicians using GPT-4o
    """
    try:
        # Prepare summary data
        summary_prompt = f"""Generate a concise clinical summary for a doctor based on this patient's symptom tracking data from the past week.

Patient ID: {patient_id}
Number of measurements: {len(measurements)}
Number of alerts: {len(alerts)}

Measurements data:
{measurements}

Alerts:
{alerts}

Provide a structured summary including:
1. Overall pattern summary
2. Significant changes detected
3. Areas requiring clinical attention
4. Recommended follow-up actions

Keep it concise and actionable (200-300 words max).
"""
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant generating structured summaries for clinicians."
                },
                {
                    "role": "user",
                    "content": summary_prompt
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating weekly summary: {e}")
        return "Unable to generate summary at this time."


class OpenAIService:
    """
    OpenAI Service wrapper for exam coach and other AI-powered features
    HIPAA Compliance: Uses BAA-protected OpenAI API
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def analyze_image_with_context(
        self,
        image_data: str,
        context: str,
        additional_images: List[str] = None,
        prompt: str = None
    ) -> str:
        """
        Analyze image(s) with given context and optional custom prompt
        Used for exam coach real-time feedback and respiratory analysis
        
        Args:
            image_data: Base64 encoded primary image
            context: Context about what's being analyzed (can include prompt)
            additional_images: Optional list of additional base64 images (for video frames)
            prompt: Optional separate prompt (if not included in context)
        
        Returns:
            String response from AI (caller should parse JSON if needed)
        """
        try:
            # Build message content
            message_content = []
            
            # Add text context/prompt
            text_content = context
            if prompt:
                text_content = f"{context}\n\n{prompt}"
            message_content.append({"type": "text", "text": text_content})
            
            # Add primary image
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })
            
            # Add additional images if provided (for video frames)
            if additional_images:
                for img_data in additional_images[:9]:  # Limit to 10 total images
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}"
                        }
                    })
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical assistant providing monitoring feedback and observations. Focus on objective observations, NOT medical diagnoses."
                    },
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            raise Exception(f"Image analysis failed: {str(e)}")
