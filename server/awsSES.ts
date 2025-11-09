import { SESClient, SendEmailCommand } from "@aws-sdk/client-ses";

const REGION = process.env.AWS_REGION || process.env.AWS_COGNITO_REGION!;

export const sesClient = new SESClient({
  region: REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

interface EmailOptions {
  to: string | string[];
  subject: string;
  htmlBody: string;
  textBody?: string;
  from?: string;
}

export async function sendEmail({
  to,
  subject,
  htmlBody,
  textBody,
  from = "t@followupai.io", // Verified SES email
}: EmailOptions) {
  const recipients = Array.isArray(to) ? to : [to];

  const command = new SendEmailCommand({
    Source: from,
    Destination: {
      ToAddresses: recipients,
    },
    Message: {
      Subject: {
        Data: subject,
        Charset: "UTF-8",
      },
      Body: {
        Html: {
          Data: htmlBody,
          Charset: "UTF-8",
        },
        ...(textBody && {
          Text: {
            Data: textBody,
            Charset: "UTF-8",
          },
        }),
      },
    },
  });

  try {
    const response = await sesClient.send(command);
    console.log("Email sent successfully:", response.MessageId);
    return response;
  } catch (error) {
    console.error("Error sending email:", error);
    throw error;
  }
}

// Email templates
export async function sendVerificationEmail(email: string, code: string) {
  const htmlBody = `
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

  const textBody = `
    Verify Your Email
    
    Welcome to Followup AI!
    
    Your verification code is: ${code}
    
    This code will expire in 24 hours.
    
    If you didn't create an account with Followup AI, please ignore this email.
  `;

  return sendEmail({
    to: email,
    subject: "Verify Your Email - Followup AI",
    htmlBody,
    textBody,
  });
}

export async function sendPasswordResetEmail(email: string, code: string) {
  const htmlBody = `
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
          <p>This code will expire in 1 hour.</p>
          <p>If you didn't request a password reset, please ignore this email and ensure your account is secure.</p>
        </div>
        <div class="footer">
          <p>&copy; ${new Date().getFullYear()} Followup AI. All rights reserved.</p>
          <p>HIPAA-Compliant Health Monitoring Platform</p>
        </div>
      </div>
    </body>
    </html>
  `;

  const textBody = `
    Reset Your Password
    
    We received a request to reset your password.
    
    Your password reset code is: ${code}
    
    This code will expire in 1 hour.
    
    If you didn't request a password reset, please ignore this email.
  `;

  return sendEmail({
    to: email,
    subject: "Reset Your Password - Followup AI",
    htmlBody,
    textBody,
  });
}

export async function sendDoctorApprovedEmail(email: string, firstName: string) {
  const htmlBody = `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: white; padding: 40px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }
        .button { display: inline-block; background: #8B5CF6; color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; font-weight: 600; margin: 20px 0; }
        .footer { text-align: center; padding: 20px; color: #6b7280; font-size: 14px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>✓ Application Approved!</h1>
        </div>
        <div class="content">
          <p>Dear Dr. ${firstName},</p>
          <p>Congratulations! Your doctor application has been approved by our verification team.</p>
          <p>You can now log in to your Followup AI account and start helping immunocompromised patients with AI-powered health monitoring and insights.</p>
          <a href="${process.env.REPLIT_DOMAINS ? 'https://' + process.env.REPLIT_DOMAINS.split(',')[0] + '/login' : 'http://localhost:5000/login'}" class="button">Log In to Your Account</a>
          <p>If you have any questions or need assistance, please don't hesitate to contact our support team.</p>
        </div>
        <div class="footer">
          <p>&copy; ${new Date().getFullYear()} Followup AI. All rights reserved.</p>
          <p>HIPAA-Compliant Health Monitoring Platform</p>
        </div>
      </div>
    </body>
    </html>
  `;

  const textBody = `
    Application Approved!
    
    Dear Dr. ${firstName},
    
    Congratulations! Your doctor application has been approved by our verification team.
    
    You can now log in to your Followup AI account and start helping immunocompromised patients.
    
    Log in at: ${process.env.REPLIT_DOMAINS ? 'https://' + process.env.REPLIT_DOMAINS.split(',')[0] + '/login' : 'http://localhost:5000/login'}
    
    If you have any questions, please contact our support team.
  `;

  return sendEmail({
    to: email,
    subject: "Your Doctor Application Has Been Approved - Followup AI",
    htmlBody,
    textBody,
  });
}

export async function sendDoctorRejectedEmail(email: string, firstName: string, reason: string) {
  const htmlBody = `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: white; padding: 40px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }
        .reason { background: #FEF2F2; border-left: 4px solid #EF4444; padding: 16px; margin: 20px 0; border-radius: 4px; }
        .footer { text-align: center; padding: 20px; color: #6b7280; font-size: 14px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Application Update</h1>
        </div>
        <div class="content">
          <p>Dear Dr. ${firstName},</p>
          <p>Thank you for your interest in joining Followup AI as a healthcare provider.</p>
          <p>After careful review, we are unable to approve your application at this time.</p>
          <div class="reason">
            <strong>Reason:</strong><br>
            ${reason}
          </div>
          <p>If you believe this decision was made in error or if you have additional documentation to provide, please contact our verification team at verification@followupai.com.</p>
          <p>We appreciate your understanding.</p>
        </div>
        <div class="footer">
          <p>&copy; ${new Date().getFullYear()} Followup AI. All rights reserved.</p>
          <p>HIPAA-Compliant Health Monitoring Platform</p>
        </div>
      </div>
    </body>
    </html>
  `;

  const textBody = `
    Application Update
    
    Dear Dr. ${firstName},
    
    Thank you for your interest in joining Followup AI as a healthcare provider.
    
    After careful review, we are unable to approve your application at this time.
    
    Reason: ${reason}
    
    If you believe this decision was made in error or if you have additional documentation, please contact our verification team at verification@followupai.com.
  `;

  return sendEmail({
    to: email,
    subject: "Update on Your Doctor Application - Followup AI",
    htmlBody,
    textBody,
  });
}

export async function sendWelcomeEmail(email: string, firstName: string) {
  const htmlBody = `
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
          <p>Hi ${firstName},</p>
          <p>Thank you for joining Followup AI, your HIPAA-compliant health monitoring platform.</p>
          <p>We're excited to help you on your health journey with our AI-powered agents:</p>
          <ul>
            <li><strong>Agent Clona</strong> - Your personal health companion</li>
            <li><strong>Assistant Lysa</strong> - Medical research assistant for doctors</li>
          </ul>
          <p>Get started by completing your profile and connecting your health devices.</p>
        </div>
        <div class="footer">
          <p>&copy; ${new Date().getFullYear()} Followup AI. All rights reserved.</p>
          <p>HIPAA-Compliant Health Monitoring Platform</p>
        </div>
      </div>
    </body>
    </html>
  `;

  const textBody = `
    Welcome to Followup AI!
    
    Hi ${firstName},
    
    Thank you for joining Followup AI, your HIPAA-compliant health monitoring platform.
    
    We're excited to help you on your health journey with our AI-powered agents:
    - Agent Clona: Your personal health companion
    - Assistant Lysa: Medical research assistant for doctors
    
    Get started by completing your profile and connecting your health devices.
  `;

  return sendEmail({
    to: email,
    subject: "Welcome to Followup AI!",
    htmlBody,
    textBody,
  });
}
