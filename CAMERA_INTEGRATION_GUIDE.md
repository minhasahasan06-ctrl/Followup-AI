# Live Physical Examination Camera Integration - HIPAA Compliant

## Overview
This guide explains how to integrate **Daily.co** for HIPAA-compliant live video consultations with physical examination monitoring in the Followup AI platform.

## Why Daily.co?

After researching HIPAA-compliant video SDKs (Daily.co, Twilio Video, Vonage), **Daily.co is the recommended choice** for the following reasons:

### ✅ Advantages
1. **BAA Available at No Extra Cost** - Signs Business Associate Agreement with no additional fees
2. **Automatic HIPAA Compliance** - Turn on HIPAA mode and compliance features are automatic
3. **Simple Setup** - Minimal configuration required compared to alternatives
4. **End-to-End Encryption** - All calls encrypted by default
5. **No PHI in URLs** - Randomized room names prevent Patient Health Information leakage
6. **No Browser Cookies** - HIPAA mode disables tracking
7. **HIPAA-Compliant Text Chat** - Built-in secure messaging
8. **Recording Disabled by Default** - Prevents accidental PHI storage
9. **Pay-as-you-go Pricing** - No enterprise contract required

### 📊 Comparison with Alternatives

| Feature | Daily.co | Twilio Video | Vonage Video |
|---------|----------|--------------|--------------|
| **BAA Cost** | ✅ Free | ❌ Enterprise only | ❓ Verify |
| **Auto HIPAA** | ✅ Yes | ❌ Manual setup | ❓ Unknown |
| **Complexity** | ⭐ Easy | ⭐⭐⭐ Complex | ❓ Unknown |

## Setup Instructions

### 1. Sign Up for Daily.co

1. Create account at https://www.daily.co/
2. Add credit card to enable paid features
3. **Request HIPAA BAA** via Daily.co portal form
4. Enable **Healthcare add-on** in your dashboard
5. Get your API key from Settings → Developers

### 2. Set Environment Variable

```bash
export DAILY_API_KEY="your_daily_api_key_here"
```

Add to your `.env` file or Replit Secrets:
```
DAILY_API_KEY=your_api_key_here
```

### 3. Install Daily.co Client SDK (Frontend)

The frontend uses Daily.co's JavaScript SDK:

```bash
npm install @daily-co/daily-js
```

### 4. Backend Implementation (Python)

The Python backend service is already implemented in:
- `app/services/daily_video_service.py` - Core Daily.co integration
- Creates HIPAA-compliant rooms
- Generates patient and doctor access tokens
- Manages room lifecycle

### 5. API Endpoints

Create these FastAPI endpoints (router not yet implemented):

```python
# app/routers/video_consultation.py

@router.post("/consultations/{consultation_id}/start-video")
async def start_video_consultation(consultation_id: int):
    """
    Create a HIPAA-compliant video room for a consultation.
    Returns room URL and access tokens for patient and doctor.
    """
    pass

@router.delete("/consultations/{consultation_id}/end-video")
async def end_video_consultation(consultation_id: int):
    """
    End video consultation and delete room (HIPAA best practice).
    """
    pass
```

### 6. Frontend Implementation

Example React component for video consultation:

```typescript
import DailyIframe from '@daily-co/daily-js';

function VideoConsultation({ roomUrl, token }: { roomUrl: string; token: string }) {
  const [callFrame, setCallFrame] = useState(null);

  useEffect(() => {
    const frame = DailyIframe.createFrame({
      showLeaveButton: true,
      iframeStyle: {
        position: 'fixed',
        width: '100%',
        height: '100%',
        border: 0,
      },
    });

    frame.join({ url: roomUrl, token });
    setCallFrame(frame);

    return () => {
      frame.destroy();
    };
  }, [roomUrl, token]);

  return <div id="video-container" />;
}
```

## HIPAA Compliance Checklist

Before going to production, ensure:

- [ ] **BAA Signed** - Business Associate Agreement executed with Daily.co
- [ ] **Healthcare Add-on Enabled** - HIPAA mode activated in Daily.co dashboard
- [ ] **No PHI in Room Names** - Use randomized room names only
- [ ] **Access Controls** - Only authenticated users get meeting tokens
- [ ] **Audit Logging** - Log all video session creation/deletion events
- [ ] **Session Timeout** - Rooms auto-expire after consultation
- [ ] **Immediate Deletion** - Delete rooms after consultation ends
- [ ] **Staff Training** - Educate team on HIPAA video best practices
- [ ] **Admin Policies** - Document video consultation procedures

## Security Features

### Automatic HIPAA Features (Daily.co)
- ✅ End-to-end encryption
- ✅ No cookies or local storage
- ✅ Randomized room names
- ✅ User data scrubbing
- ✅ Recording disabled by default
- ✅ Live streaming disabled
- ✅ HIPAA-compliant text chat

### Application-Level Security
- ✅ Authentication required (AWS Cognito)
- ✅ Role-based access control (patient/doctor only)
- ✅ Session tokens expire with room
- ✅ Audit logging for all video events
- ✅ Database records consultation metadata only (NO video data)

## Cost Estimates

**Daily.co Pricing:**
- $15 free credits for new accounts
- Healthcare add-on required (contact Daily.co for pricing)
- Pay-as-you-go after free credits
- Estimated cost: $0.004 per participant-minute

**Development Costs:**
- Integration development: ~2-3 days
- Testing and compliance review: ~1-2 days
- Total estimated cost: $100K+ including legal review

## Implementation Roadmap

### Phase 1: Basic Video (2-3 days)
1. Create Daily.co account and get BAA
2. Implement backend service (✅ DONE)
3. Create video consultation API endpoints
4. Build frontend video component
5. Test basic video calls

### Phase 2: Physical Exam Features (3-5 days)
1. Add camera controls (zoom, focus, flash)
2. Implement screen sharing for test results
3. Add annotation tools for highlighting areas
4. Create consultation notes integration
5. Implement recording (with HIPAA-compliant storage)

### Phase 3: AI Enhancements (5-7 days)
1. Integrate OpenAI Whisper for voice-to-text transcription
2. Add real-time symptom detection from video
3. Generate AI consultation summaries
4. Implement anomaly detection in physical exams
5. Create diagnostic assistance features

## Alternative SDKs (If Daily.co Doesn't Work)

### Backup Option 1: Twilio Video
- **Pros:** Multi-channel (video + SMS + voice), enterprise-grade
- **Cons:** Requires Enterprise Edition, complex setup
- **Use If:** You need integrated SMS/voice with video

### Backup Option 2: Dyte
- **Pros:** 10K free credits/month, used by healthcare platforms
- **Cons:** Less established than Daily.co or Twilio
- **Use If:** Budget is very tight

### Backup Option 3: Zoom Video SDK
- **Pros:** Familiar interface, proven reliability
- **Cons:** Different from consumer Zoom, requires paid healthcare plan
- **Use If:** Clients specifically request Zoom branding

## Support Resources

- **Daily.co Docs:** https://docs.daily.co/guides/privacy-and-security/hipaa
- **Daily.co Support:** https://www.daily.co/contact
- **HIPAA Compliance Guide:** https://www.hhs.gov/hipaa/index.html
- **BAA Template:** Available from Daily.co after signup

## Next Steps

1. **Sign up for Daily.co** and request BAA
2. **Create video consultation router** in `app/routers/video_consultation.py`
3. **Build frontend video component** using `@daily-co/daily-js`
4. **Test with mock consultation** between test patient and doctor accounts
5. **Conduct security review** before production deployment

## Questions?

Contact Daily.co support for:
- BAA signing process
- Healthcare add-on pricing
- HIPAA compliance verification
- Technical integration assistance
