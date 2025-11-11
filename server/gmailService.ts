import { google, gmail_v1 } from 'googleapis';
import type { Storage } from './storage';

interface GmailConfig {
  clientId: string;
  clientSecret: string;
  redirectUri: string;
}

class GmailService {
  private oauth2Client: any;
  private config: GmailConfig;

  constructor(private storage: Storage) {
    this.config = {
      clientId: process.env.GOOGLE_CLIENT_ID || '',
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || '',
      redirectUri: process.env.GOOGLE_REDIRECT_URI || 'http://localhost:5000/api/gmail/oauth/callback',
    };

    this.oauth2Client = new google.auth.OAuth2(
      this.config.clientId,
      this.config.clientSecret,
      this.config.redirectUri
    );
  }

  getAuthUrl(doctorId: string): string {
    const scopes = [
      'https://www.googleapis.com/auth/gmail.readonly',
      'https://www.googleapis.com/auth/gmail.send',
      'https://www.googleapis.com/auth/gmail.modify',
    ];

    return this.oauth2Client.generateAuthUrl({
      access_type: 'offline',
      scope: scopes,
      state: doctorId,
      prompt: 'consent',
    });
  }

  async handleOAuthCallback(code: string, doctorId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const { tokens } = await this.oauth2Client.getToken(code);
      
      const gmailSync = await this.storage.getGmailSync(doctorId);
      
      if (gmailSync) {
        await this.storage.updateGmailSync(doctorId, {
          accessToken: tokens.access_token,
          refreshToken: tokens.refresh_token || gmailSync.refreshToken,
          tokenExpiry: tokens.expiry_date ? new Date(tokens.expiry_date) : undefined,
          syncEnabled: true,
        });
      } else {
        await this.storage.createGmailSync({
          doctorId,
          accessToken: tokens.access_token,
          refreshToken: tokens.refresh_token,
          tokenExpiry: tokens.expiry_date ? new Date(tokens.expiry_date) : undefined,
          syncEnabled: true,
        });
      }

      return { success: true };
    } catch (error) {
      console.error('Gmail OAuth error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async syncEmails(doctorId: string): Promise<{ success: boolean; emailsFetched: number; error?: string }> {
    try {
      const gmailSync = await this.storage.getGmailSync(doctorId);
      if (!gmailSync || !gmailSync.accessToken) {
        return { success: false, emailsFetched: 0, error: 'Gmail not connected' };
      }

      this.oauth2Client.setCredentials({
        access_token: gmailSync.accessToken,
        refresh_token: gmailSync.refreshToken,
        expiry_date: gmailSync.tokenExpiry?.getTime(),
      });

      const gmail = google.gmail({ version: 'v1', auth: this.oauth2Client });

      const query = gmailSync.lastSyncAt
        ? `after:${Math.floor(gmailSync.lastSyncAt.getTime() / 1000)}`
        : 'in:inbox';

      const response = await gmail.users.messages.list({
        userId: 'me',
        q: query,
        maxResults: 50,
      });

      const messages = response.data.messages || [];
      let emailsFetched = 0;

      for (const message of messages) {
        if (!message.id) continue;

        const fullMessage = await gmail.users.messages.get({
          userId: 'me',
          id: message.id,
          format: 'full',
        });

        await this.processGmailMessage(fullMessage.data, doctorId);
        emailsFetched++;
      }

      await this.storage.updateGmailSync(doctorId, {
        lastSyncAt: new Date(),
        lastSyncStatus: 'success',
      });

      return { success: true, emailsFetched };
    } catch (error) {
      console.error('Gmail sync error:', error);
      await this.storage.updateGmailSync(doctorId, {
        lastSyncStatus: 'error',
        lastSyncError: error instanceof Error ? error.message : 'Unknown error',
      });
      return { success: false, emailsFetched: 0, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  private async processGmailMessage(message: gmail_v1.Schema$Message, doctorId: string): Promise<void> {
    const headers = message.payload?.headers || [];
    const subject = headers.find(h => h.name?.toLowerCase() === 'subject')?.value || '(No Subject)';
    const from = headers.find(h => h.name?.toLowerCase() === 'from')?.value || '';
    const to = headers.find(h => h.name?.toLowerCase() === 'to')?.value || '';
    const threadId = message.threadId || message.id || '';

    let thread = await this.storage.getEmailThreadByExternalId(threadId);
    
    if (!thread) {
      thread = await this.storage.createEmailThread({
        doctorId,
        subject,
        externalThreadId: threadId,
        participants: [from, to],
        status: 'open',
        priority: 'normal',
        category: 'general',
        isRead: false,
      });
    }

    let body = '';
    if (message.payload?.body?.data) {
      body = Buffer.from(message.payload.body.data, 'base64').toString('utf-8');
    } else if (message.payload?.parts) {
      const textPart = message.payload.parts.find(p => p.mimeType === 'text/plain' || p.mimeType === 'text/html');
      if (textPart?.body?.data) {
        body = Buffer.from(textPart.body.data, 'base64').toString('utf-8');
      }
    }

    await this.storage.createEmailMessage({
      threadId: thread.id,
      externalMessageId: message.id || '',
      fromEmail: from,
      toEmail: to,
      subject,
      body,
      isFromDoctor: from.includes(to),
      sentAt: message.internalDate ? new Date(parseInt(message.internalDate)) : new Date(),
    });
  }

  async sendEmail(doctorId: string, to: string, subject: string, body: string): Promise<{ success: boolean; error?: string }> {
    try {
      const gmailSync = await this.storage.getGmailSync(doctorId);
      if (!gmailSync || !gmailSync.accessToken) {
        return { success: false, error: 'Gmail not connected' };
      }

      this.oauth2Client.setCredentials({
        access_token: gmailSync.accessToken,
        refresh_token: gmailSync.refreshToken,
        expiry_date: gmailSync.tokenExpiry?.getTime(),
      });

      const gmail = google.gmail({ version: 'v1', auth: this.oauth2Client });

      const email = [
        `To: ${to}`,
        `Subject: ${subject}`,
        '',
        body,
      ].join('\n');

      const encodedEmail = Buffer.from(email).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

      await gmail.users.messages.send({
        userId: 'me',
        requestBody: {
          raw: encodedEmail,
        },
      });

      return { success: true };
    } catch (error) {
      console.error('Gmail send error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async disconnect(doctorId: string): Promise<{ success: boolean }> {
    try {
      await this.storage.deleteGmailSync(doctorId);
      return { success: true };
    } catch (error) {
      console.error('Gmail disconnect error:', error);
      return { success: false };
    }
  }
}

export let gmailService: GmailService;

export function initGmailService(storage: Storage) {
  gmailService = new GmailService(storage);
}
