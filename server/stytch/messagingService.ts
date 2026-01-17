import { getStytchClient, isStytchConfigured } from "./stytchClient";

export interface SendSMSOptions {
  to: string;
  message: string;
}

export interface SendEmailOptions {
  to: string | string[];
  subject: string;
  htmlBody: string;
  textBody?: string;
  from?: string;
}

export async function sendSMS({ to, message }: SendSMSOptions): Promise<boolean> {
  if (!isStytchConfigured()) {
    console.warn("[STYTCH] SMS not configured - falling back to log only");
    console.log(`[SMS] Would send to ${to}: ${message}`);
    return false;
  }

  try {
    const client = getStytchClient();
    
    const response = await client.otps.sms.send({
      phone_number: to,
      expiration_minutes: 10,
    });

    console.log(`[STYTCH] SMS sent to ${to}, phone_id: ${response.phone_id}`);
    return true;
  } catch (error: any) {
    console.error("[STYTCH] SMS send error:", error.message);
    return false;
  }
}

export async function sendVerificationSMS(to: string): Promise<{ success: boolean; phoneId?: string }> {
  if (!isStytchConfigured()) {
    console.warn("[STYTCH] SMS not configured");
    return { success: false };
  }

  try {
    const client = getStytchClient();
    const response = await client.otps.sms.send({
      phone_number: to,
      expiration_minutes: 10,
    });

    console.log(`[STYTCH] Verification SMS sent to ${to}`);
    return { success: true, phoneId: response.phone_id };
  } catch (error: any) {
    console.error("[STYTCH] Verification SMS error:", error.message);
    return { success: false };
  }
}

export async function sendMedicationReminder(
  to: string,
  medicationName: string,
  dosage: string,
  time: string
): Promise<boolean> {
  console.log(`[SMS] Medication reminder to ${to}: Take ${medicationName} ${dosage} at ${time}`);
  return sendSMS({
    to,
    message: `Medication Reminder: It's time to take ${medicationName} ${dosage}. Scheduled for ${time}. - Followup AI`,
  });
}

export async function sendAppointmentConfirmation(
  to: string,
  doctorName: string,
  date: string,
  time: string
): Promise<boolean> {
  console.log(`[SMS] Appointment confirmation to ${to}: Dr. ${doctorName} on ${date} at ${time}`);
  return sendSMS({
    to,
    message: `Appointment Confirmed: Your consultation with Dr. ${doctorName} is scheduled for ${date} at ${time}. - Followup AI`,
  });
}

export async function sendAppointmentReminder(
  to: string,
  doctorName: string,
  timeUntil: string
): Promise<boolean> {
  return sendSMS({
    to,
    message: `Reminder: Your appointment with Dr. ${doctorName} is in ${timeUntil}. - Followup AI`,
  });
}

export async function sendEmergencyAlert(to: string, alertMessage: string): Promise<boolean> {
  return sendSMS({
    to,
    message: `⚠️ HEALTH ALERT: ${alertMessage} Please contact your healthcare provider immediately. - Followup AI`,
  });
}

export async function sendWelcomeSMS(to: string, firstName: string): Promise<boolean> {
  return sendSMS({
    to,
    message: `Welcome to Followup AI, ${firstName}! We're here to support your health journey. Your caring AI companion Agent Clona is ready to help.`,
  });
}

export async function sendEmail({ to, subject, htmlBody, textBody }: SendEmailOptions): Promise<boolean> {
  if (!isStytchConfigured()) {
    console.warn("[STYTCH] Email not configured - falling back to log only");
    const recipients = Array.isArray(to) ? to.join(", ") : to;
    console.log(`[EMAIL] Would send to ${recipients}: ${subject}`);
    return false;
  }

  try {
    const client = getStytchClient();
    const recipients = Array.isArray(to) ? to : [to];

    for (const recipient of recipients) {
      await client.magicLinks.email.send({
        email: recipient,
        login_magic_link_url: process.env.APP_URL || "https://followupai.io",
        signup_magic_link_url: process.env.APP_URL || "https://followupai.io",
      });
    }

    console.log(`[STYTCH] Email sent to ${recipients.join(", ")}: ${subject}`);
    return true;
  } catch (error: any) {
    console.error("[STYTCH] Email send error:", error.message);
    return false;
  }
}

export async function sendVerificationEmail(email: string, code: string): Promise<boolean> {
  console.log(`[EMAIL] Verification email to ${email} with code: ${code}`);
  return sendEmail({
    to: email,
    subject: "Verify Your Email - Followup AI",
    htmlBody: generateVerificationEmailHtml(code),
    textBody: `Your Followup AI verification code is: ${code}. This code will expire in 24 hours.`,
  });
}

export async function sendPasswordResetEmail(email: string, code: string): Promise<boolean> {
  console.log(`[EMAIL] Password reset email to ${email}`);
  return sendEmail({
    to: email,
    subject: "Reset Your Password - Followup AI",
    htmlBody: generatePasswordResetEmailHtml(code),
    textBody: `Your Followup AI password reset code is: ${code}. This code expires in 1 hour.`,
  });
}

export async function sendWelcomeEmail(email: string, firstName: string): Promise<boolean> {
  return sendEmail({
    to: email,
    subject: "Welcome to Followup AI!",
    htmlBody: generateWelcomeEmailHtml(firstName),
    textBody: `Welcome to Followup AI, ${firstName}! We're excited to have you on board.`,
  });
}

export async function sendAppointmentConfirmationEmail(
  email: string,
  doctorName: string,
  date: string,
  time: string,
  location: string
): Promise<boolean> {
  return sendEmail({
    to: email,
    subject: `Appointment Confirmed with Dr. ${doctorName}`,
    htmlBody: generateAppointmentEmailHtml(doctorName, date, time, location),
    textBody: `Your appointment with Dr. ${doctorName} is confirmed for ${date} at ${time}. Location: ${location}`,
  });
}

export async function sendLabResultsNotificationEmail(email: string, patientName: string): Promise<boolean> {
  return sendEmail({
    to: email,
    subject: "Your Lab Results Are Ready - Followup AI",
    htmlBody: generateLabResultsEmailHtml(patientName),
    textBody: `Hello ${patientName}, your lab results are now available in your Followup AI dashboard.`,
  });
}

function generateVerificationEmailHtml(code: string): string {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: white; padding: 40px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }
        .code { background: #f3f4f6; padding: 20px; text-align: center; font-size: 32px; font-weight: bold; letter-spacing: 4px; border-radius: 8px; margin: 20px 0; }
        .footer { text-align: center; padding: 20px; color: #6b7280; font-size: 14px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Verify Your Email</h1>
        </div>
        <div class="content">
          <p>Welcome to Followup AI!</p>
          <p>Please use the verification code below to confirm your email address:</p>
          <div class="code">${code}</div>
          <p>This code will expire in 24 hours.</p>
          <p>If you didn't create an account with Followup AI, please ignore this email.</p>
        </div>
        <div class="footer">
          <p>&copy; ${new Date().getFullYear()} Followup AI. All rights reserved.</p>
          <p>HIPAA-Compliant Health Monitoring Platform</p>
        </div>
      </div>
    </body>
    </html>
  `;
}

function generatePasswordResetEmailHtml(code: string): string {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: white; padding: 40px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }
        .code { background: #f3f4f6; padding: 20px; text-align: center; font-size: 32px; font-weight: bold; letter-spacing: 4px; border-radius: 8px; margin: 20px 0; }
        .footer { text-align: center; padding: 20px; color: #6b7280; font-size: 14px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Reset Your Password</h1>
        </div>
        <div class="content">
          <p>We received a request to reset your password.</p>
          <p>Use the code below to reset your password:</p>
          <div class="code">${code}</div>
          <p>This code expires in 1 hour.</p>
          <p>If you didn't request a password reset, please ignore this email.</p>
        </div>
        <div class="footer">
          <p>&copy; ${new Date().getFullYear()} Followup AI. All rights reserved.</p>
        </div>
      </div>
    </body>
    </html>
  `;
}

function generateWelcomeEmailHtml(firstName: string): string {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: white; padding: 40px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }
        .footer { text-align: center; padding: 20px; color: #6b7280; font-size: 14px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Welcome to Followup AI!</h1>
        </div>
        <div class="content">
          <p>Hello ${firstName},</p>
          <p>Welcome to Followup AI! We're excited to have you on board.</p>
          <p>Your AI health companion, Agent Clona, is ready to help you on your health journey.</p>
          <p>Get started by:</p>
          <ul>
            <li>Setting up your health profile</li>
            <li>Tracking your daily habits</li>
            <li>Connecting with healthcare providers</li>
          </ul>
        </div>
        <div class="footer">
          <p>&copy; ${new Date().getFullYear()} Followup AI. All rights reserved.</p>
        </div>
      </div>
    </body>
    </html>
  `;
}

function generateAppointmentEmailHtml(doctorName: string, date: string, time: string, location: string): string {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: white; padding: 40px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }
        .details { background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .footer { text-align: center; padding: 20px; color: #6b7280; font-size: 14px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Appointment Confirmed</h1>
        </div>
        <div class="content">
          <p>Your appointment has been confirmed!</p>
          <div class="details">
            <p><strong>Doctor:</strong> Dr. ${doctorName}</p>
            <p><strong>Date:</strong> ${date}</p>
            <p><strong>Time:</strong> ${time}</p>
            <p><strong>Location:</strong> ${location}</p>
          </div>
          <p>Please arrive 15 minutes early for check-in.</p>
        </div>
        <div class="footer">
          <p>&copy; ${new Date().getFullYear()} Followup AI. All rights reserved.</p>
        </div>
      </div>
    </body>
    </html>
  `;
}

function generateLabResultsEmailHtml(patientName: string): string {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: white; padding: 40px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }
        .footer { text-align: center; padding: 20px; color: #6b7280; font-size: 14px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Lab Results Available</h1>
        </div>
        <div class="content">
          <p>Hello ${patientName},</p>
          <p>Your lab results are now available in your Followup AI dashboard.</p>
          <p>Please log in to review your results. If you have any questions, contact your healthcare provider.</p>
        </div>
        <div class="footer">
          <p>&copy; ${new Date().getFullYear()} Followup AI. All rights reserved.</p>
        </div>
      </div>
    </body>
    </html>
  `;
}
