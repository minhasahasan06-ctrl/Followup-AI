import { google, gmail_v1 } from 'googleapis';
import twilio from 'twilio';
import type { Storage } from './storage';
import type { DoctorIntegration, DoctorEmail, DoctorWhatsappMessage, CallLog } from '@shared/schema';
import OpenAI from 'openai';
import crypto from 'crypto';
import jwt from 'jsonwebtoken';

interface GoogleOAuthConfig {
  clientId: string;
  clientSecret: string;
  redirectUri: string;
}

interface TwilioConfig {
  accountSid: string;
  authToken: string;
}

// HIPAA-Compliant Encryption for OAuth Credentials
// Uses AES-256-GCM with a derived key from SESSION_SECRET
const ENCRYPTION_ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;
const SALT_LENGTH = 32;

function getEncryptionKey(): Buffer {
  const secret = process.env.SESSION_SECRET || 'followup-ai-default-secret-key';
  return crypto.scryptSync(secret, 'oauth-credential-salt', 32);
}

function encryptCredential(text: string): string {
  if (!text) return '';
  const iv = crypto.randomBytes(IV_LENGTH);
  const cipher = crypto.createCipheriv(ENCRYPTION_ALGORITHM, getEncryptionKey(), iv);
  let encrypted = cipher.update(text, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  const authTag = cipher.getAuthTag();
  return `${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;
}

function decryptCredential(encryptedText: string): string {
  if (!encryptedText) return '';
  
  // Check if this looks like an encrypted credential (IV:AuthTag:Encrypted format)
  if (!encryptedText.includes(':')) {
    // Legacy plaintext token - return as-is for backward compatibility
    // The token will be re-encrypted on next update
    return encryptedText;
  }
  
  try {
    const parts = encryptedText.split(':');
    if (parts.length !== 3) {
      // Not our encryption format, might be a legacy token with colons
      return encryptedText;
    }
    
    const [ivHex, authTagHex, encrypted] = parts;
    
    // Validate hex format (IV should be 32 hex chars = 16 bytes)
    if (ivHex.length !== 32 || !/^[a-f0-9]+$/i.test(ivHex)) {
      // Not our encryption format
      return encryptedText;
    }
    
    const iv = Buffer.from(ivHex, 'hex');
    const authTag = Buffer.from(authTagHex, 'hex');
    const decipher = crypto.createDecipheriv(ENCRYPTION_ALGORITHM, getEncryptionKey(), iv);
    decipher.setAuthTag(authTag);
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  } catch (error) {
    // Decryption failed - might be legacy plaintext with colons
    console.error('Credential decryption failed, returning as plaintext:', error);
    return encryptedText;
  }
}

// JWT-Signed OAuth State Token for CSRF Protection
const STATE_TOKEN_SECRET = process.env.SESSION_SECRET || 'oauth-state-secret';
const STATE_TOKEN_EXPIRY = '10m';

interface OAuthStatePayload {
  doctorId: string;
  type: string;
  nonce: string;
  iat?: number;
  exp?: number;
}

function createSignedStateToken(doctorId: string, type: string): string {
  const nonce = crypto.randomBytes(16).toString('hex');
  const payload: OAuthStatePayload = { doctorId, type, nonce };
  return jwt.sign(payload, STATE_TOKEN_SECRET, { expiresIn: STATE_TOKEN_EXPIRY });
}

function verifyStateToken(token: string): OAuthStatePayload | null {
  try {
    return jwt.verify(token, STATE_TOKEN_SECRET) as OAuthStatePayload;
  } catch (error) {
    console.error('OAuth state token verification failed:', error);
    return null;
  }
}

export class DoctorIntegrationService {
  private googleConfig: GoogleOAuthConfig;
  private twilioConfig: TwilioConfig;
  private openai: OpenAI;

  constructor(private storage: Storage) {
    this.googleConfig = {
      clientId: process.env.GOOGLE_CLIENT_ID || '',
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || '',
      redirectUri: `${process.env.REPLIT_DEV_DOMAIN ? `https://${process.env.REPLIT_DEV_DOMAIN}` : 'http://localhost:5000'}/api/integrations/gmail/callback`,
    };

    this.twilioConfig = {
      accountSid: process.env.TWILIO_ACCOUNT_SID || '',
      authToken: process.env.TWILIO_AUTH_TOKEN || '',
    };

    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }

  // ============================================
  // INTEGRATION STATUS
  // ============================================

  async getDoctorIntegrations(doctorId: string): Promise<DoctorIntegration[]> {
    return this.storage.getDoctorIntegrations(doctorId);
  }

  async getIntegrationStatus(doctorId: string): Promise<{
    gmail: { connected: boolean; email?: string; lastSync?: Date };
    whatsapp: { connected: boolean; number?: string; lastSync?: Date };
    twilio: { connected: boolean; number?: string; lastSync?: Date };
  }> {
    const integrations = await this.getDoctorIntegrations(doctorId);
    
    const gmail = integrations.find(i => i.integrationType === 'gmail');
    const whatsapp = integrations.find(i => i.integrationType === 'whatsapp_business');
    const twilioInt = integrations.find(i => i.integrationType === 'twilio');

    return {
      gmail: {
        connected: gmail?.status === 'connected',
        email: gmail?.providerAccountEmail || undefined,
        lastSync: gmail?.lastSyncAt || undefined,
      },
      whatsapp: {
        connected: whatsapp?.status === 'connected',
        number: whatsapp?.whatsappDisplayNumber || undefined,
        lastSync: whatsapp?.lastSyncAt || undefined,
      },
      twilio: {
        connected: twilioInt?.status === 'connected',
        number: twilioInt?.twilioPhoneNumber || undefined,
        lastSync: twilioInt?.lastSyncAt || undefined,
      },
    };
  }

  // ============================================
  // GMAIL OAUTH INTEGRATION
  // ============================================

  getGmailAuthUrl(doctorId: string): string {
    const oauth2Client = new google.auth.OAuth2(
      this.googleConfig.clientId,
      this.googleConfig.clientSecret,
      this.googleConfig.redirectUri
    );

    const scopes = [
      'https://www.googleapis.com/auth/gmail.readonly',
      'https://www.googleapis.com/auth/gmail.send',
      'https://www.googleapis.com/auth/gmail.modify',
      'https://www.googleapis.com/auth/userinfo.email',
    ];

    // Use signed JWT state token for CSRF protection (HIPAA-compliant)
    const signedState = createSignedStateToken(doctorId, 'gmail');

    return oauth2Client.generateAuthUrl({
      access_type: 'offline',
      scope: scopes,
      state: signedState,
      prompt: 'consent',
    });
  }

  async handleGmailCallback(code: string, stateToken: string): Promise<{ success: boolean; email?: string; error?: string }> {
    try {
      // Verify the signed state token to prevent CSRF attacks
      const statePayload = verifyStateToken(stateToken);
      if (!statePayload) {
        return { success: false, error: 'Invalid or expired OAuth state token' };
      }

      if (statePayload.type !== 'gmail') {
        return { success: false, error: 'Invalid OAuth callback type' };
      }

      const doctorId = statePayload.doctorId;

      const oauth2Client = new google.auth.OAuth2(
        this.googleConfig.clientId,
        this.googleConfig.clientSecret,
        this.googleConfig.redirectUri
      );

      const { tokens } = await oauth2Client.getToken(code);
      oauth2Client.setCredentials(tokens);

      // Get user email
      const oauth2 = google.oauth2({ version: 'v2', auth: oauth2Client });
      const userInfo = await oauth2.userinfo.get();
      const email = userInfo.data.email || '';

      // HIPAA-Compliant: Encrypt OAuth credentials before storage
      const encryptedAccessToken = encryptCredential(tokens.access_token || '');
      const encryptedRefreshToken = encryptCredential(tokens.refresh_token || '');

      // Check if integration exists
      const existing = await this.storage.getDoctorIntegrationByType(doctorId, 'gmail');

      if (existing) {
        // When falling back to existing refresh token, ensure it's encrypted
        let finalRefreshToken = encryptedRefreshToken;
        if (!tokens.refresh_token && existing.refreshToken) {
          // Check if existing token is already encrypted (contains ':' from AES-GCM format)
          if (existing.refreshToken.includes(':')) {
            finalRefreshToken = existing.refreshToken;
          } else {
            // Legacy plaintext token - encrypt it now
            finalRefreshToken = encryptCredential(existing.refreshToken);
          }
        }
        
        await this.storage.updateDoctorIntegration(existing.id, {
          status: 'connected',
          accessToken: encryptedAccessToken,
          refreshToken: finalRefreshToken,
          tokenExpiresAt: tokens.expiry_date ? new Date(tokens.expiry_date) : undefined,
          tokenScope: tokens.scope || '',
          providerAccountEmail: email,
          lastErrorMessage: null,
        });
      } else {
        await this.storage.createDoctorIntegration({
          doctorId,
          integrationType: 'gmail',
          status: 'connected',
          accessToken: encryptedAccessToken,
          refreshToken: encryptedRefreshToken,
          tokenExpiresAt: tokens.expiry_date ? new Date(tokens.expiry_date) : undefined,
          tokenScope: tokens.scope || '',
          providerAccountEmail: email,
          syncEnabled: true,
        });
      }

      // Log HIPAA audit event for OAuth connection
      console.log(`[HIPAA-AUDIT] Gmail OAuth connected for doctor ${doctorId} (${email})`);

      return { success: true, email };
    } catch (error) {
      console.error('Gmail OAuth callback error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async syncGmailEmails(doctorId: string): Promise<{ success: boolean; emailsFetched: number; error?: string }> {
    try {
      const integration = await this.storage.getDoctorIntegrationByType(doctorId, 'gmail');
      if (!integration || integration.status !== 'connected') {
        return { success: false, emailsFetched: 0, error: 'Gmail not connected' };
      }

      const oauth2Client = new google.auth.OAuth2(
        this.googleConfig.clientId,
        this.googleConfig.clientSecret,
        this.googleConfig.redirectUri
      );

      // HIPAA-Compliant: Decrypt credentials only when needed in memory
      const decryptedAccessToken = decryptCredential(integration.accessToken || '');
      const decryptedRefreshToken = decryptCredential(integration.refreshToken || '');

      oauth2Client.setCredentials({
        access_token: decryptedAccessToken,
        refresh_token: decryptedRefreshToken,
        expiry_date: integration.tokenExpiresAt?.getTime(),
      });

      const gmail = google.gmail({ version: 'v1', auth: oauth2Client });

      // Fetch recent emails
      const query = integration.lastSyncAt
        ? `after:${Math.floor(integration.lastSyncAt.getTime() / 1000)}`
        : 'in:inbox newer_than:7d';

      const response = await gmail.users.messages.list({
        userId: 'me',
        q: query,
        maxResults: 50,
      });

      const messages = response.data.messages || [];
      let emailsFetched = 0;

      for (const message of messages) {
        if (!message.id) continue;

        // Check if already synced
        const existing = await this.storage.getDoctorEmailByProviderId(doctorId, message.id);
        if (existing) continue;

        const fullMessage = await gmail.users.messages.get({
          userId: 'me',
          id: message.id,
          format: 'full',
        });

        await this.processAndStoreEmail(fullMessage.data, doctorId, integration.id);
        emailsFetched++;
      }

      // Update last sync
      await this.storage.updateDoctorIntegration(integration.id, {
        lastSyncAt: new Date(),
        lastErrorMessage: null,
      });

      return { success: true, emailsFetched };
    } catch (error) {
      console.error('Gmail sync error:', error);
      
      const integration = await this.storage.getDoctorIntegrationByType(doctorId, 'gmail');
      if (integration) {
        await this.storage.updateDoctorIntegration(integration.id, {
          status: 'error',
          lastErrorAt: new Date(),
          lastErrorMessage: error instanceof Error ? error.message : 'Unknown error',
        });
      }

      return { success: false, emailsFetched: 0, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  private async processAndStoreEmail(message: gmail_v1.Schema$Message, doctorId: string, integrationId: string): Promise<void> {
    const headers = message.payload?.headers || [];
    const subject = headers.find(h => h.name?.toLowerCase() === 'subject')?.value || '(No Subject)';
    const from = headers.find(h => h.name?.toLowerCase() === 'from')?.value || '';
    const to = headers.find(h => h.name?.toLowerCase() === 'to')?.value || '';
    const cc = headers.find(h => h.name?.toLowerCase() === 'cc')?.value || '';
    const date = headers.find(h => h.name?.toLowerCase() === 'date')?.value;

    // Parse from email
    const fromEmailMatch = from.match(/<([^>]+)>/) || [null, from];
    const fromEmail = fromEmailMatch[1] || from;
    const fromName = from.replace(/<[^>]+>/, '').trim() || undefined;

    // Parse body
    let bodyPlain = '';
    let bodyHtml = '';
    if (message.payload?.body?.data) {
      bodyPlain = Buffer.from(message.payload.body.data, 'base64').toString('utf-8');
    } else if (message.payload?.parts) {
      const textPart = message.payload.parts.find(p => p.mimeType === 'text/plain');
      const htmlPart = message.payload.parts.find(p => p.mimeType === 'text/html');
      if (textPart?.body?.data) {
        bodyPlain = Buffer.from(textPart.body.data, 'base64').toString('utf-8');
      }
      if (htmlPart?.body?.data) {
        bodyHtml = Buffer.from(htmlPart.body.data, 'base64').toString('utf-8');
      }
    }

    // AI categorization
    const aiAnalysis = await this.analyzeEmailWithAI(subject, bodyPlain || bodyHtml);

    await this.storage.createDoctorEmail({
      doctorId,
      integrationId,
      providerMessageId: message.id || '',
      threadId: message.threadId || undefined,
      subject,
      fromEmail,
      fromName,
      toEmails: to.split(',').map(e => e.trim()),
      ccEmails: cc ? cc.split(',').map(e => e.trim()) : undefined,
      snippet: message.snippet || undefined,
      bodyPlain: bodyPlain || undefined,
      bodyHtml: bodyHtml || undefined,
      isRead: !(message.labelIds || []).includes('UNREAD'),
      isStarred: (message.labelIds || []).includes('STARRED'),
      labels: message.labelIds || undefined,
      aiCategory: aiAnalysis.category,
      aiPriority: aiAnalysis.priority,
      aiSummary: aiAnalysis.summary,
      aiSuggestedReply: aiAnalysis.suggestedReply,
      aiExtractedInfo: aiAnalysis.extractedInfo,
      receivedAt: date ? new Date(date) : new Date(parseInt(message.internalDate || '0')),
    });
  }

  private async analyzeEmailWithAI(subject: string, body: string): Promise<{
    category: string;
    priority: string;
    summary: string;
    suggestedReply?: string;
    extractedInfo?: any;
  }> {
    try {
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: `You are a medical receptionist AI assistant. Analyze the email and provide:
1. category: One of 'appointment_request', 'medical_question', 'prescription_refill', 'test_results', 'billing', 'general', 'spam'
2. priority: One of 'urgent', 'high', 'normal', 'low'
3. summary: A brief 1-2 sentence summary
4. suggestedReply: A professional reply draft if appropriate
5. extractedInfo: Any extracted appointment requests, symptoms, or medication mentions

Respond in JSON format.`
          },
          {
            role: 'user',
            content: `Subject: ${subject}\n\nBody: ${body.substring(0, 2000)}`
          }
        ],
        response_format: { type: 'json_object' },
      });

      const result = JSON.parse(response.choices[0]?.message?.content || '{}');
      return {
        category: result.category || 'general',
        priority: result.priority || 'normal',
        summary: result.summary || '',
        suggestedReply: result.suggestedReply,
        extractedInfo: result.extractedInfo,
      };
    } catch (error) {
      console.error('AI email analysis error:', error);
      return { category: 'general', priority: 'normal', summary: '' };
    }
  }

  async sendEmailFromDoctor(doctorId: string, to: string, subject: string, body: string, replyToMessageId?: string): Promise<{ success: boolean; error?: string }> {
    try {
      const integration = await this.storage.getDoctorIntegrationByType(doctorId, 'gmail');
      if (!integration || integration.status !== 'connected') {
        return { success: false, error: 'Gmail not connected' };
      }

      const oauth2Client = new google.auth.OAuth2(
        this.googleConfig.clientId,
        this.googleConfig.clientSecret,
        this.googleConfig.redirectUri
      );

      // HIPAA-Compliant: Decrypt credentials only when needed in memory
      const decryptedAccessToken = decryptCredential(integration.accessToken || '');
      const decryptedRefreshToken = decryptCredential(integration.refreshToken || '');

      oauth2Client.setCredentials({
        access_token: decryptedAccessToken,
        refresh_token: decryptedRefreshToken,
        expiry_date: integration.tokenExpiresAt?.getTime(),
      });

      const gmail = google.gmail({ version: 'v1', auth: oauth2Client });

      const emailLines = [
        `To: ${to}`,
        `Subject: ${subject}`,
        'Content-Type: text/html; charset=utf-8',
        '',
        body,
      ];

      const encodedEmail = Buffer.from(emailLines.join('\r\n')).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

      await gmail.users.messages.send({
        userId: 'me',
        requestBody: {
          raw: encodedEmail,
          threadId: replyToMessageId,
        },
      });

      return { success: true };
    } catch (error) {
      console.error('Gmail send error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async disconnectGmail(doctorId: string): Promise<{ success: boolean }> {
    try {
      const integration = await this.storage.getDoctorIntegrationByType(doctorId, 'gmail');
      if (integration) {
        await this.storage.updateDoctorIntegration(integration.id, {
          status: 'disconnected',
          accessToken: null,
          refreshToken: null,
          tokenExpiresAt: null,
        });
      }
      return { success: true };
    } catch (error) {
      console.error('Gmail disconnect error:', error);
      return { success: false };
    }
  }

  // ============================================
  // WHATSAPP BUSINESS API INTEGRATION
  // ============================================

  async configureWhatsAppBusiness(
    doctorId: string,
    businessId: string,
    phoneNumberId: string,
    displayNumber: string,
    accessToken: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      // HIPAA-Compliant: Encrypt access token before storage
      const encryptedAccessToken = encryptCredential(accessToken);
      
      const existing = await this.storage.getDoctorIntegrationByType(doctorId, 'whatsapp_business');

      if (existing) {
        await this.storage.updateDoctorIntegration(existing.id, {
          status: 'connected',
          whatsappBusinessId: businessId,
          whatsappPhoneNumberId: phoneNumberId,
          whatsappDisplayNumber: displayNumber,
          accessToken: encryptedAccessToken,
          lastErrorMessage: null,
        });
      } else {
        await this.storage.createDoctorIntegration({
          doctorId,
          integrationType: 'whatsapp_business',
          status: 'connected',
          whatsappBusinessId: businessId,
          whatsappPhoneNumberId: phoneNumberId,
          whatsappDisplayNumber: displayNumber,
          accessToken: encryptedAccessToken,
          syncEnabled: true,
        });
      }

      // Log HIPAA audit event
      console.log(`[HIPAA-AUDIT] WhatsApp Business connected for doctor ${doctorId} (${displayNumber})`)

      return { success: true };
    } catch (error) {
      console.error('WhatsApp config error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async processIncomingWhatsAppMessage(
    doctorId: string,
    messageId: string,
    fromNumber: string,
    toNumber: string,
    messageType: string,
    textContent?: string,
    contactName?: string
  ): Promise<void> {
    const integration = await this.storage.getDoctorIntegrationByType(doctorId, 'whatsapp_business');
    if (!integration) return;

    // AI analysis for WhatsApp message
    const aiAnalysis = await this.analyzeWhatsAppMessage(textContent || '');

    await this.storage.createDoctorWhatsappMessage({
      doctorId,
      integrationId: integration.id,
      waMessageId: messageId,
      direction: 'inbound',
      fromNumber,
      toNumber,
      contactName,
      messageType,
      textContent,
      aiCategory: aiAnalysis.category,
      aiPriority: aiAnalysis.priority,
      aiSuggestedReply: aiAnalysis.suggestedReply,
      aiExtractedInfo: aiAnalysis.extractedInfo,
      receivedAt: new Date(),
    });
  }

  private async analyzeWhatsAppMessage(content: string): Promise<{
    category: string;
    priority: string;
    suggestedReply?: string;
    extractedInfo?: any;
  }> {
    try {
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: `You are a medical receptionist AI. Analyze the WhatsApp message and provide:
1. category: One of 'appointment_request', 'question', 'confirmation', 'cancellation', 'general'
2. priority: One of 'urgent', 'high', 'normal', 'low'
3. suggestedReply: A brief, friendly reply suggestion
4. extractedInfo: Any appointment details or medical concerns mentioned

Respond in JSON format.`
          },
          {
            role: 'user',
            content
          }
        ],
        response_format: { type: 'json_object' },
      });

      return JSON.parse(response.choices[0]?.message?.content || '{}');
    } catch (error) {
      return { category: 'general', priority: 'normal' };
    }
  }

  async sendWhatsAppMessage(doctorId: string, toNumber: string, message: string): Promise<{ success: boolean; error?: string }> {
    try {
      const integration = await this.storage.getDoctorIntegrationByType(doctorId, 'whatsapp_business');
      if (!integration || integration.status !== 'connected') {
        return { success: false, error: 'WhatsApp Business not connected' };
      }

      // HIPAA-Compliant: Decrypt access token only when needed in memory
      const decryptedAccessToken = decryptCredential(integration.accessToken || '');

      // Use WhatsApp Cloud API
      const response = await fetch(
        `https://graph.facebook.com/v18.0/${integration.whatsappPhoneNumberId}/messages`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${decryptedAccessToken}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messaging_product: 'whatsapp',
            to: toNumber,
            type: 'text',
            text: { body: message },
          }),
        }
      );

      if (!response.ok) {
        const error = await response.json();
        return { success: false, error: error.error?.message || 'Failed to send' };
      }

      // Store outbound message
      const result = await response.json() as any;
      await this.storage.createDoctorWhatsappMessage({
        doctorId,
        integrationId: integration.id,
        waMessageId: result.messages?.[0]?.id || '',
        direction: 'outbound',
        fromNumber: integration.whatsappDisplayNumber || '',
        toNumber,
        messageType: 'text',
        textContent: message,
        status: 'sent',
        receivedAt: new Date(),
      });

      return { success: true };
    } catch (error) {
      console.error('WhatsApp send error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  // ============================================
  // TWILIO PHONE INTEGRATION
  // ============================================

  async configureTwilioAccount(
    doctorId: string,
    accountSid: string,
    apiKey: string,
    apiSecret: string,
    phoneNumber: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      // Verify credentials by making a test API call
      const client = twilio(apiKey, apiSecret, { accountSid });
      await client.api.accounts(accountSid).fetch();

      // HIPAA-Compliant: Encrypt sensitive credentials before storage
      const encryptedApiKey = encryptCredential(apiKey);
      const encryptedApiSecret = encryptCredential(apiSecret);

      const existing = await this.storage.getDoctorIntegrationByType(doctorId, 'twilio');

      if (existing) {
        await this.storage.updateDoctorIntegration(existing.id, {
          status: 'connected',
          twilioAccountSid: accountSid, // Account SID is not sensitive (used for routing)
          twilioApiKey: encryptedApiKey,
          twilioApiSecret: encryptedApiSecret,
          twilioPhoneNumber: phoneNumber,
          lastErrorMessage: null,
        });
      } else {
        await this.storage.createDoctorIntegration({
          doctorId,
          integrationType: 'twilio',
          status: 'connected',
          twilioAccountSid: accountSid,
          twilioApiKey: encryptedApiKey,
          twilioApiSecret: encryptedApiSecret,
          twilioPhoneNumber: phoneNumber,
          syncEnabled: true,
        });
      }

      // Log HIPAA audit event
      console.log(`[HIPAA-AUDIT] Twilio phone connected for doctor ${doctorId} (${phoneNumber})`);

      return { success: true };
    } catch (error) {
      console.error('Twilio config error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Invalid credentials' };
    }
  }

  async processIncomingCall(
    doctorId: string,
    callSid: string,
    callerPhone: string,
    callerName: string | undefined,
    direction: 'inbound' | 'outbound',
    status: string,
    duration?: number,
    recordingUrl?: string,
    transcription?: string
  ): Promise<void> {
    // AI analysis of transcription if available
    let aiAnalysis: any = {};
    if (transcription) {
      aiAnalysis = await this.analyzeCallTranscription(transcription);
    }

    // Try to link to patient by phone number
    const patient = await this.storage.getUserByPhone(callerPhone);

    await this.storage.createCallLog({
      doctorId,
      patientId: patient?.id,
      callerPhone,
      callerName,
      direction,
      callType: aiAnalysis.intent || 'inquiry',
      startTime: new Date(),
      duration,
      status,
      twilioCallSid: callSid,
      recordingUrl,
      transcription,
      aiSummary: aiAnalysis.summary,
      aiIntent: aiAnalysis.intent,
      aiSentiment: aiAnalysis.sentiment,
      aiExtractedInfo: aiAnalysis.extractedInfo,
      requiresFollowup: aiAnalysis.requiresFollowup || false,
    });
  }

  private async analyzeCallTranscription(transcription: string): Promise<{
    summary?: string;
    intent?: string;
    sentiment?: string;
    extractedInfo?: any;
    requiresFollowup?: boolean;
  }> {
    try {
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: `Analyze this phone call transcription from a medical practice:
1. summary: Brief summary of the call
2. intent: One of 'book_appointment', 'question', 'emergency', 'followup', 'prescription', 'test_results', 'other'
3. sentiment: One of 'positive', 'neutral', 'negative', 'urgent'
4. extractedInfo: Any appointment requests, symptoms, or action items
5. requiresFollowup: Boolean if doctor needs to follow up

Respond in JSON format.`
          },
          {
            role: 'user',
            content: transcription
          }
        ],
        response_format: { type: 'json_object' },
      });

      return JSON.parse(response.choices[0]?.message?.content || '{}');
    } catch (error) {
      return {};
    }
  }

  async makeOutboundCall(doctorId: string, toNumber: string): Promise<{ success: boolean; callSid?: string; error?: string }> {
    try {
      const integration = await this.storage.getDoctorIntegrationByType(doctorId, 'twilio');
      if (!integration || integration.status !== 'connected') {
        return { success: false, error: 'Twilio not connected' };
      }

      // HIPAA-Compliant: Decrypt credentials only when needed in memory
      const decryptedApiKey = decryptCredential(integration.twilioApiKey || '');
      const decryptedApiSecret = decryptCredential(integration.twilioApiSecret || '');

      const client = twilio(
        decryptedApiKey,
        decryptedApiSecret,
        { accountSid: integration.twilioAccountSid || '' }
      );

      const call = await client.calls.create({
        to: toNumber,
        from: integration.twilioPhoneNumber || '',
        url: `${process.env.REPLIT_DEV_DOMAIN ? `https://${process.env.REPLIT_DEV_DOMAIN}` : 'http://localhost:5000'}/api/twilio/twiml/outbound`,
      });

      return { success: true, callSid: call.sid };
    } catch (error) {
      console.error('Twilio call error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async syncTwilioCallHistory(doctorId: string): Promise<{ success: boolean; callsSynced: number; error?: string }> {
    try {
      const integration = await this.storage.getDoctorIntegrationByType(doctorId, 'twilio');
      if (!integration || integration.status !== 'connected') {
        return { success: false, callsSynced: 0, error: 'Twilio not connected' };
      }

      // HIPAA-Compliant: Decrypt credentials only when needed in memory
      const decryptedApiKey = decryptCredential(integration.twilioApiKey || '');
      const decryptedApiSecret = decryptCredential(integration.twilioApiSecret || '');

      const client = twilio(
        decryptedApiKey,
        decryptedApiSecret,
        { accountSid: integration.twilioAccountSid || '' }
      );

      // Fetch recent calls
      const calls = await client.calls.list({
        to: integration.twilioPhoneNumber || undefined,
        limit: 50,
      });

      let callsSynced = 0;
      for (const call of calls) {
        // Check if already logged
        const existing = await this.storage.getCallLogByTwilioSid(call.sid);
        if (existing) continue;

        await this.processIncomingCall(
          doctorId,
          call.sid,
          call.from,
          undefined,
          call.direction === 'inbound' ? 'inbound' : 'outbound',
          call.status,
          parseInt(call.duration) || undefined
        );
        callsSynced++;
      }

      await this.storage.updateDoctorIntegration(integration.id, {
        lastSyncAt: new Date(),
      });

      return { success: true, callsSynced };
    } catch (error) {
      console.error('Twilio sync error:', error);
      return { success: false, callsSynced: 0, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  // ============================================
  // DATA RETRIEVAL
  // ============================================

  async getDoctorEmails(doctorId: string, options?: {
    category?: string;
    isRead?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<DoctorEmail[]> {
    return this.storage.getDoctorEmails(doctorId, options);
  }

  async getDoctorWhatsappMessages(doctorId: string, options?: {
    status?: string;
    limit?: number;
  }): Promise<DoctorWhatsappMessage[]> {
    return this.storage.getDoctorWhatsappMessages(doctorId, options);
  }

  async getDoctorCallLogs(doctorId: string, options?: {
    status?: string;
    limit?: number;
  }): Promise<CallLog[]> {
    return this.storage.getCallLogs(doctorId, options);
  }
}

export let doctorIntegrationService: DoctorIntegrationService;

export function initDoctorIntegrationService(storage: Storage) {
  doctorIntegrationService = new DoctorIntegrationService(storage);
}
