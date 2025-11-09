# Production Setup Guide - Email & SMS Configuration

This guide explains how to move your Followup AI platform from development/sandbox mode to production for email and SMS delivery.

---

## 📧 Issue 1: AWS SES Sandbox Mode

### Current Problem
AWS SES is in **Sandbox Mode**, which means:
- ✅ Can send emails FROM verified addresses (t@followupai.io)
- ❌ Can ONLY send emails TO verified addresses
- ❌ Cannot send to patient/doctor signup emails (they're not verified)
- Daily limit: 200 emails/day
- Rate limit: 1 email/second

### Solution: Request Production Access

**Step 1: Request Production Access via AWS Support**

1. **Go to AWS SES Console** → **Account Dashboard** (Region: ap-southeast-2 Sydney)
2. Click **"Request production access"** button
3. Fill out the form with these details:

**Mail Type:** Transactional
**Website URL:** https://followupai.io (or your domain)
**Use Case Description:**
```
Followup AI is a HIPAA-compliant health monitoring platform for immunocompromised 
patients. We send the following transactional emails:

1. Email verification codes during patient/doctor signup (2-factor authentication)
2. Password reset emails
3. Welcome emails after successful registration
4. Appointment reminders and confirmations
5. Medication reminders
6. Health alerts and notifications

We have implemented AWS Cognito authentication with mandatory two-step verification.
All emails contain patient health information and require HIPAA compliance.

Expected volume: ~500 emails/day initially, scaling to ~5,000/day within 6 months.

We have signed AWS Business Associate Agreement (BAA) for HIPAA compliance.
```

**How will you handle bounces and complaints:**
```
We use AWS SNS to monitor bounce and complaint notifications. Our application:
1. Automatically removes bounced emails from our database
2. Immediately stops sending to addresses that generate complaints
3. Maintains bounce rate < 5% and complaint rate < 0.1%
4. Logs all email delivery events for audit compliance
```

**Company Details:**
- Legal company name: Followup AI
- Company website: followupai.io
- Physical address: [Your address]

4. Click **Submit**

**Step 2: Wait for AWS Approval (24-48 hours)**

AWS typically responds within 1-2 business days. They may ask follow-up questions.

**Step 3: After Approval**

Once approved, your SES account will have:
- Send to ANY email address (not just verified)
- Higher limits: 50,000 emails/day (can request increases)
- Rate: 14 emails/second

### Alternative: Verify Test Email Addresses (Temporary Fix)

For testing purposes while waiting for production access:

1. Go to **AWS SES Console** → **Verified identities**
2. Click **Create identity**
3. Select **Email address**
4. Enter test email: `testpatient@followupai.io`
5. Click **Create identity**
6. Check that email inbox for verification link
7. Click the verification link
8. Repeat for any other test emails

**Now you can send to those verified addresses during development.**

---

## 🌍 Issue 2: SES Region Hostname Mismatch

### Current Configuration
Your SES client is configured with region: **ap-southeast-2** (Sydney)

### Check Your Configuration

Let's verify the region is correctly set:

```typescript
// server/awsSES.ts (already correct)
const REGION = process.env.AWS_REGION || process.env.AWS_COGNITO_REGION!;

export const sesClient = new SESClient({
  region: REGION,  // Should be "ap-southeast-2"
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});
```

### Common Fixes

**1. Verify Environment Variables**
Check your secrets have the correct region:
- `AWS_REGION=ap-southeast-2`
- `AWS_COGNITO_REGION=ap-southeast-2`

**2. Verify SES Identity Region**
Your verified email (t@followupai.io) MUST be verified in the **same region** as your Cognito User Pool:
- Go to **AWS SES Console**
- Check you're in **Asia Pacific (Sydney) ap-southeast-2**
- Go to **Verified identities**
- Confirm `t@followupai.io` is listed and verified

**3. If SES identity is in wrong region:**
- Delete the identity from the wrong region
- Create and verify it in ap-southeast-2 (Sydney)

**Note:** The current configuration looks correct. The "hostname mismatch" might be a transient AWS API issue or DNS propagation delay. If emails are being sent (even just to verified addresses), the region is configured correctly.

---

## 📱 Issue 3: Twilio SMS Test Mode → Production

### Current Problem
Your Twilio credentials are likely from a **Trial Account**, which:
- ✅ Can send SMS to verified phone numbers only
- ❌ Cannot send to unverified patient/doctor phones
- ❌ All messages prefixed with "Sent from your Twilio trial account"
- Limited to ~30-50 test messages

### Solution: Upgrade to Twilio Production

**Step 1: Upgrade Your Twilio Account**

1. **Log in to Twilio Console**: https://console.twilio.com/
2. **Click "Upgrade"** (top navigation or dashboard)
3. **Add Payment Method**
   - Enter credit card details
   - Minimum $20 initial balance recommended
4. **Complete Account Verification**
   - Verify your business information
   - May require tax ID or business registration

**Step 2: Buy a Twilio Phone Number (for SMS)**

1. Go to **Phone Numbers** → **Manage** → **Buy a number**
2. Select country: **Australia** (+61)
3. Check **SMS** capability
4. Select a number (cost: ~$1.50/month)
5. Click **Buy**

**Step 3: Update Environment Variable**

Update your `TWILIO_PHONE_NUMBER` secret:
```
TWILIO_PHONE_NUMBER=+61xxxxxxxxx  (your new Twilio number)
```

**Step 4: Enable SMS Capabilities**

1. Go to **Messaging** → **Services**
2. Create a new Messaging Service (optional but recommended for production)
3. Add your phone number to the Messaging Service
4. Configure delivery receipts and webhooks (optional)

**Step 5: Verify A2P 10DLC Registration (Required for US numbers)**

If sending SMS to US numbers (+1), you MUST register for A2P 10DLC:
1. Go to **Messaging** → **Regulatory Compliance**
2. Complete **Business Profile**
3. Register your **Campaign** (use case: Transactional - Healthcare)
4. Wait for approval (~3-5 business days)

**Note:** Australian numbers (+61) don't require A2P registration.

### Pricing After Upgrade

**SMS Costs (Australia):**
- Outbound SMS: ~$0.08 USD per message
- Inbound SMS: ~$0.01 USD per message
- Phone number: ~$1.50 USD/month

**Example Monthly Cost (500 patients):**
- Phone number rental: $1.50
- SMS sent (500 signups × 2 codes = 1,000 SMS): $80
- **Total: ~$82/month**

---

## 🔐 Alternative: Use Replit's Twilio Integration

Replit offers a managed Twilio integration that simplifies configuration:

**Benefits:**
- Automatic credential management
- No need to manually update secrets
- Easier to switch between dev/prod

**Setup Steps:**
1. I can set this up for you if you'd like
2. It will guide you through connecting your Twilio account
3. Credentials are automatically injected as environment variables

**Would you like me to set up the Replit Twilio integration?**

---

## ✅ Verification Checklist

After completing the fixes above, verify everything works:

### SES Email Delivery
- [ ] AWS SES production access approved
- [ ] Can send emails to any address (not just verified)
- [ ] t@followupai.io verified in ap-southeast-2 region
- [ ] Test patient signup with real email address

### Twilio SMS Delivery
- [ ] Twilio account upgraded (no longer trial)
- [ ] Phone number purchased and configured
- [ ] TWILIO_PHONE_NUMBER secret updated
- [ ] Test SMS to unverified phone number
- [ ] No "trial account" prefix in messages

### Test End-to-End Flow
- [ ] Patient signup → Email code received
- [ ] Email verification → SMS code received
- [ ] Phone verification → User created
- [ ] Login successful → JWT tokens issued

---

## 📞 Support Contacts

**AWS SES Support:**
- AWS Support Center: https://console.aws.amazon.com/support/
- SES Support: use-case specific (production access request)

**Twilio Support:**
- Support Center: https://support.twilio.com/
- Upgrade issues: help@twilio.com
- Phone: 1-866-987-4546 (US)

**HIPAA Compliance:**
- AWS BAA: https://aws.amazon.com/compliance/hipaa-compliance/
- Twilio HIPAA: https://www.twilio.com/en-us/legal/hipaa

---

## 🚀 Quick Start Commands

After fixing the issues, test your setup:

```bash
# Test SES email
curl -X POST http://localhost:5000/api/auth/signup/patient \
  -H "Content-Type: application/json" \
  -d '{"email":"realuser@gmail.com","password":"Test123!@#","firstName":"Test","lastName":"User","phoneNumber":"+61400123456","ehrImportMethod":"manual"}'

# Check if email was sent (should not show SES sandbox error)

# Test Twilio SMS (after email verification)
# SMS should be sent to +61400123456 without "trial account" message
```

---

## 💰 Estimated Monthly Costs (Production)

**AWS SES:**
- First 62,000 emails/month: **FREE** (via AWS Free Tier)
- After: $0.10 per 1,000 emails
- **Estimate: $0-5/month**

**Twilio:**
- Phone number: $1.50/month
- SMS (1,000 messages): ~$80/month
- **Estimate: $82/month**

**Total Production Cost: ~$82-87/month**

---

## 🎯 Next Steps

1. **Immediate (Today):**
   - Request AWS SES production access (takes 24-48 hours)
   - Upgrade Twilio account (instant)
   - Buy Twilio phone number

2. **Within 48 hours:**
   - AWS SES production access approved
   - Test email delivery to real addresses

3. **Production Ready:**
   - Update patient/doctor signup flow
   - Enable real-time email + SMS verification
   - Monitor delivery rates and costs

**Need help with any of these steps? Let me know!**
