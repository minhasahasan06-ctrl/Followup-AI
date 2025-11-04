import twilio from 'twilio';

if (!process.env.TWILIO_ACCOUNT_SID || !process.env.TWILIO_AUTH_TOKEN || !process.env.TWILIO_PHONE_NUMBER) {
  console.warn('[TWILIO] Warning: Twilio credentials not configured. SMS features will be disabled.');
}

const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;
const twilioPhoneNumber = process.env.TWILIO_PHONE_NUMBER;

export const twilioClient = accountSid && authToken ? twilio(accountSid, authToken) : null;

export interface SendSMSOptions {
  to: string;
  message: string;
}

export interface SendVerificationCodeOptions {
  to: string;
  channel?: 'sms' | 'call';
}

export interface VerifyCodeOptions {
  to: string;
  code: string;
}

export async function sendSMS({ to, message }: SendSMSOptions): Promise<boolean> {
  if (!twilioClient || !twilioPhoneNumber) {
    console.error('[TWILIO] SMS not configured');
    return false;
  }

  try {
    const result = await twilioClient.messages.create({
      body: message,
      from: twilioPhoneNumber,
      to: to,
    });
    
    console.log(`[TWILIO] SMS sent successfully to ${to}. SID: ${result.sid}`);
    return true;
  } catch (error: any) {
    console.error('[TWILIO] Error sending SMS:', error.message);
    return false;
  }
}

export async function sendVerificationCode({ to, channel = 'sms' }: SendVerificationCodeOptions): Promise<{ success: boolean; code?: string }> {
  if (!twilioClient) {
    console.error('[TWILIO] Verify not configured');
    return { success: false };
  }

  try {
    const code = Math.floor(100000 + Math.random() * 900000).toString();
    
    const message = `Your Followup AI verification code is: ${code}. This code will expire in 10 minutes.`;
    
    if (channel === 'sms') {
      const success = await sendSMS({ to, message });
      return { success, code: success ? code : undefined };
    } else {
      const result = await twilioClient.calls.create({
        twiml: `<Response><Say>Your Followup AI verification code is ${code.split('').join(', ')}. Again, your code is ${code.split('').join(', ')}.</Say></Response>`,
        to: to,
        from: twilioPhoneNumber!,
      });
      
      console.log(`[TWILIO] Voice call initiated to ${to}. SID: ${result.sid}`);
      return { success: true, code };
    }
  } catch (error: any) {
    console.error('[TWILIO] Error sending verification code:', error.message);
    return { success: false };
  }
}

export async function sendMedicationReminder(to: string, medicationName: string, dosage: string, time: string): Promise<boolean> {
  const message = `Medication Reminder: It's time to take ${medicationName} ${dosage}. Scheduled for ${time}. - Followup AI`;
  return sendSMS({ to, message });
}

export async function sendAppointmentConfirmation(to: string, doctorName: string, date: string, time: string): Promise<boolean> {
  const message = `Appointment Confirmed: Your consultation with Dr. ${doctorName} is scheduled for ${date} at ${time}. - Followup AI`;
  return sendSMS({ to, message });
}

export async function sendAppointmentReminder(to: string, doctorName: string, timeUntil: string): Promise<boolean> {
  const message = `Reminder: Your appointment with Dr. ${doctorName} is in ${timeUntil}. - Followup AI`;
  return sendSMS({ to, message });
}

export async function sendEmergencyAlert(to: string, alertMessage: string): Promise<boolean> {
  const message = `⚠️ HEALTH ALERT: ${alertMessage} Please contact your healthcare provider immediately. - Followup AI`;
  return sendSMS({ to, message });
}

export async function sendDailyFollowup(to: string, userName: string): Promise<boolean> {
  const message = `Good morning ${userName}! Time for your daily health check-in. How are you feeling today? Log in to Followup AI to update your status. - Agent Clona`;
  return sendSMS({ to, message });
}

export async function sendHealthInsight(to: string, insight: string): Promise<boolean> {
  const message = `Health Insight: ${insight} - Followup AI`;
  return sendSMS({ to, message });
}

export async function sendWelcomeSMS(to: string, firstName: string): Promise<boolean> {
  const message = `Welcome to Followup AI, ${firstName}! We're here to support your health journey. Your caring AI companion Agent Clona is ready to help. Get started at your dashboard.`;
  return sendSMS({ to, message });
}

export async function sendPasswordResetSMS(to: string, code: string): Promise<boolean> {
  const message = `Your Followup AI password reset code is: ${code}. This code expires in 1 hour. If you didn't request this, please ignore this message.`;
  return sendSMS({ to, message });
}

export async function sendConsultationRequest(to: string, patientName: string, requestDetails: string): Promise<boolean> {
  const message = `New Consultation Request from ${patientName}: ${requestDetails}. Check your Followup AI dashboard for details.`;
  return sendSMS({ to, message });
}

export async function sendLabResultsNotification(to: string): Promise<boolean> {
  const message = `Your lab results are now available in your Followup AI dashboard. Please log in to review them. - Followup AI`;
  return sendSMS({ to, message });
}
