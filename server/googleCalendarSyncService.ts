import { google } from 'googleapis';
import type { calendar_v3 } from 'googleapis';
import { storage } from './storage';
import type { InsertGoogleCalendarSyncLog } from '@shared/schema';

const SCOPES = ['https://www.googleapis.com/auth/calendar'];

interface GoogleCalendarConfig {
  clientId?: string;
  clientSecret?: string;
  redirectUri?: string;
}

// Replit Google Calendar Connector - Integration Reference: connection:conn_google-calendar_01K9BV2Y4BKVD2BBGRWBYMVKPG
let connectionSettings: any;

async function getReplitAccessToken(): Promise<string | null> {
  try {
    if (connectionSettings && connectionSettings.settings?.expires_at && 
        new Date(connectionSettings.settings.expires_at).getTime() > Date.now()) {
      return connectionSettings.settings.access_token;
    }
    
    const hostname = process.env.REPLIT_CONNECTORS_HOSTNAME;
    const xReplitToken = process.env.REPL_IDENTITY 
      ? 'repl ' + process.env.REPL_IDENTITY 
      : process.env.WEB_REPL_RENEWAL 
      ? 'depl ' + process.env.WEB_REPL_RENEWAL 
      : null;

    if (!xReplitToken || !hostname) {
      console.log('[GoogleCalendarSync] Replit connector not available');
      return null;
    }

    connectionSettings = await fetch(
      'https://' + hostname + '/api/v2/connection?include_secrets=true&connector_names=google-calendar',
      {
        headers: {
          'Accept': 'application/json',
          'X_REPLIT_TOKEN': xReplitToken
        }
      }
    ).then(res => res.json()).then(data => data.items?.[0]);

    const accessToken = connectionSettings?.settings?.access_token || 
                       connectionSettings?.settings?.oauth?.credentials?.access_token;

    if (!connectionSettings || !accessToken) {
      console.log('[GoogleCalendarSync] Google Calendar not connected via Replit');
      return null;
    }
    return accessToken;
  } catch (error) {
    console.error('[GoogleCalendarSync] Error getting Replit access token:', error);
    return null;
  }
}

// Get Google Calendar client using Replit connector (for platform-level operations)
export async function getUncachableGoogleCalendarClient(): Promise<calendar_v3.Calendar | null> {
  try {
    const accessToken = await getReplitAccessToken();
    if (!accessToken) return null;

    const oauth2Client = new google.auth.OAuth2();
    oauth2Client.setCredentials({
      access_token: accessToken
    });

    return google.calendar({ version: 'v3', auth: oauth2Client });
  } catch (error) {
    console.error('[GoogleCalendarSync] Error creating calendar client:', error);
    return null;
  }
}

// Check if Replit connector is available
export async function isReplitConnectorAvailable(): Promise<boolean> {
  const token = await getReplitAccessToken();
  return token !== null;
}

/**
 * Google Calendar Sync Service
 * Provides bidirectional sync between Followup AI appointments and Google Calendar
 */
export class GoogleCalendarSyncService {
  private config: GoogleCalendarConfig;

  constructor(config: GoogleCalendarConfig = {}) {
    this.config = {
      clientId: config.clientId || process.env.GOOGLE_CALENDAR_CLIENT_ID,
      clientSecret: config.clientSecret || process.env.GOOGLE_CALENDAR_CLIENT_SECRET,
      redirectUri: config.redirectUri || process.env.GOOGLE_CALENDAR_REDIRECT_URI || 'http://localhost:5000/api/calendar/oauth/callback',
    };
  }

  /**
   * Generate OAuth URL for doctor to connect their Google Calendar
   */
  getAuthUrl(doctorId: string): string {
    const oauth2Client = new google.auth.OAuth2(
      this.config.clientId,
      this.config.clientSecret,
      this.config.redirectUri
    );

    const authUrl = oauth2Client.generateAuthUrl({
      access_type: 'offline',
      scope: SCOPES,
      state: doctorId, // Pass doctorId in state for callback
      prompt: 'consent', // Force consent screen to get refresh token
    });

    return authUrl;
  }

  /**
   * Exchange authorization code for tokens and store them
   */
  async handleOAuthCallback(
    code: string,
    doctorId: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const oauth2Client = new google.auth.OAuth2(
        this.config.clientId,
        this.config.clientSecret,
        this.config.redirectUri
      );

      // Exchange code for tokens
      const { tokens } = await oauth2Client.getToken(code);
      
      if (!tokens.refresh_token) {
        throw new Error('No refresh token received. User may have already authorized.');
      }

      oauth2Client.setCredentials(tokens);

      // Get calendar info
      const calendar = google.calendar({ version: 'v3', auth: oauth2Client });
      const calendarList = await calendar.calendarList.get({ calendarId: 'primary' });
      
      const calendarId = calendarList.data.id || '';
      const calendarName = calendarList.data.summary || 'Primary Calendar';

      // Store tokens and calendar info
      const existingSync = await storage.getGoogleCalendarSync(doctorId);
      
      if (existingSync) {
        await storage.updateGoogleCalendarSync(doctorId, {
          accessToken: tokens.access_token || null,
          refreshToken: tokens.refresh_token,
          tokenExpiry: tokens.expiry_date ? new Date(tokens.expiry_date) : null,
          calendarId,
          calendarName,
          syncEnabled: true,
          lastSyncStatus: null,
          lastSyncError: null,
        });
      } else {
        await storage.createGoogleCalendarSync({
          doctorId,
          accessToken: tokens.access_token || null,
          refreshToken: tokens.refresh_token,
          tokenExpiry: tokens.expiry_date ? new Date(tokens.expiry_date) : null,
          calendarId,
          calendarName,
          syncEnabled: true,
          syncDirection: 'bidirectional',
        });
      }

      // Perform initial sync
      await this.performSync(doctorId, 'full');

      return { success: true };
    } catch (error) {
      console.error('[GoogleCalendarSync] OAuth callback error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Get authenticated OAuth2 client for a doctor
   */
  private async getAuthClient(doctorId: string): Promise<any> {
    const syncConfig = await storage.getGoogleCalendarSync(doctorId);
    
    if (!syncConfig || !syncConfig.refreshToken) {
      throw new Error('Google Calendar not connected for this doctor');
    }

    const oauth2Client = new google.auth.OAuth2(
      this.config.clientId,
      this.config.clientSecret,
      this.config.redirectUri
    );

    oauth2Client.setCredentials({
      refresh_token: syncConfig.refreshToken,
      access_token: syncConfig.accessToken || undefined,
      expiry_date: syncConfig.tokenExpiry ? syncConfig.tokenExpiry.getTime() : undefined,
    });

    // Handle token refresh automatically
    oauth2Client.on('tokens', async (tokens) => {
      if (tokens.refresh_token) {
        await storage.updateGoogleCalendarSync(doctorId, {
          accessToken: tokens.access_token || null,
          refreshToken: tokens.refresh_token,
          tokenExpiry: tokens.expiry_date ? new Date(tokens.expiry_date) : null,
        });
      } else if (tokens.access_token) {
        await storage.updateGoogleCalendarSync(doctorId, {
          accessToken: tokens.access_token,
          tokenExpiry: tokens.expiry_date ? new Date(tokens.expiry_date) : null,
        });
      }
    });

    return oauth2Client;
  }

  /**
   * Perform full or incremental sync
   */
  async performSync(
    doctorId: string,
    syncType: 'full' | 'incremental' = 'incremental'
  ): Promise<{ success: boolean; stats: any; error?: string }> {
    const startTime = Date.now();
    const stats = {
      eventsCreated: 0,
      eventsUpdated: 0,
      eventsDeleted: 0,
      conflictsDetected: 0,
    };

    try {
      const auth = await this.getAuthClient(doctorId);
      const calendar = google.calendar({ version: 'v3', auth });
      const syncConfig = await storage.getGoogleCalendarSync(doctorId);

      if (!syncConfig || !syncConfig.syncEnabled) {
        throw new Error('Sync not enabled');
      }

      const calendarId = syncConfig.calendarId || 'primary';

      // Determine sync type
      let events: calendar_v3.Schema$Event[] = [];
      let newSyncToken: string | null = null;

      if (syncType === 'full' || !syncConfig.syncToken) {
        // Full sync
        ({ events, newSyncToken } = await this.fullSync(calendar, calendarId));
      } else {
        // Incremental sync
        try {
          ({ events, newSyncToken } = await this.incrementalSync(calendar, calendarId, syncConfig.syncToken));
        } catch (error: any) {
          if (error.code === 410) {
            // Sync token expired - fall back to full sync
            console.log('[GoogleCalendarSync] Sync token expired, performing full sync');
            ({ events, newSyncToken } = await this.fullSync(calendar, calendarId));
          } else {
            throw error;
          }
        }
      }

      // Process events based on sync direction
      if (syncConfig.syncDirection === 'from_google' || syncConfig.syncDirection === 'bidirectional') {
        const processStats = await this.processEventsFromGoogle(doctorId, events, syncConfig.conflictResolution || 'google_wins');
        stats.eventsCreated += processStats.created;
        stats.eventsUpdated += processStats.updated;
        stats.eventsDeleted += processStats.deleted;
        stats.conflictsDetected += processStats.conflicts;
      }

      // Update sync token and status
      await storage.updateGoogleCalendarSync(doctorId, {
        syncToken: newSyncToken,
        lastSyncAt: new Date(),
        lastSyncStatus: 'success',
        lastSyncError: null,
        totalEventsSynced: (syncConfig.totalEventsSynced || 0) + events.length,
        lastEventSyncedAt: events.length > 0 ? new Date() : syncConfig.lastEventSyncedAt,
      });

      // Log sync
      const durationMs = Date.now() - startTime;
      await this.logSync(doctorId, syncConfig.id, syncType, syncConfig.syncDirection || 'bidirectional', 'success', stats, durationMs);

      return { success: true, stats };
    } catch (error) {
      console.error('[GoogleCalendarSync] Sync error:', error);
      
      const durationMs = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      await storage.updateGoogleCalendarSync(doctorId, {
        lastSyncAt: new Date(),
        lastSyncStatus: 'failed',
        lastSyncError: errorMessage,
      });

      const syncConfig = await storage.getGoogleCalendarSync(doctorId);
      await this.logSync(doctorId, syncConfig?.id || null, syncType, 'bidirectional', 'failed', stats, durationMs, errorMessage);

      return {
        success: false,
        stats,
        error: errorMessage
      };
    }
  }

  /**
   * Full sync - fetch all events
   */
  private async fullSync(
    calendar: calendar_v3.Calendar,
    calendarId: string
  ): Promise<{ events: calendar_v3.Schema$Event[]; newSyncToken: string | null }> {
    let allEvents: calendar_v3.Schema$Event[] = [];
    let pageToken: string | undefined = undefined;
    let newSyncToken: string | null = null;

    do {
      const response = await calendar.events.list({
        calendarId,
        maxResults: 250,
        singleEvents: true,
        orderBy: 'startTime',
        pageToken,
      });

      if (response.data.items) {
        allEvents = allEvents.concat(response.data.items);
      }

      pageToken = response.data.nextPageToken || undefined;
      
      if (!pageToken && response.data.nextSyncToken) {
        newSyncToken = response.data.nextSyncToken;
      }
    } while (pageToken);

    return { events: allEvents, newSyncToken };
  }

  /**
   * Incremental sync - fetch only changed events
   */
  private async incrementalSync(
    calendar: calendar_v3.Calendar,
    calendarId: string,
    syncToken: string
  ): Promise<{ events: calendar_v3.Schema$Event[]; newSyncToken: string | null }> {
    let changedEvents: calendar_v3.Schema$Event[] = [];
    let pageToken: string | undefined = undefined;
    let newSyncToken: string | null = null;

    do {
      const response = await calendar.events.list({
        calendarId,
        syncToken: pageToken ? undefined : syncToken,
        pageToken,
        maxResults: 250,
        showDeleted: true,
      });

      if (response.data.items) {
        changedEvents = changedEvents.concat(response.data.items);
      }

      pageToken = response.data.nextPageToken || undefined;
      
      if (!pageToken && response.data.nextSyncToken) {
        newSyncToken = response.data.nextSyncToken;
      }
    } while (pageToken);

    return { events: changedEvents, newSyncToken };
  }

  /**
   * Process events from Google Calendar to local database
   */
  private async processEventsFromGoogle(
    doctorId: string,
    events: calendar_v3.Schema$Event[],
    conflictResolution: string
  ): Promise<{ created: number; updated: number; deleted: number; conflicts: number }> {
    let created = 0;
    let updated = 0;
    let deleted = 0;
    let conflicts = 0;

    for (const event of events) {
      try {
        if (!event.id) continue;

        // Check if event is deleted
        if (event.status === 'cancelled') {
          const existingAppointment = await storage.getAppointmentByGoogleEventId(event.id);
          if (existingAppointment) {
            await storage.updateAppointment(existingAppointment.id, {
              status: 'cancelled',
              cancelledBy: doctorId,
              cancellationReason: 'Cancelled in Google Calendar',
              cancelledAt: new Date(),
            });
            deleted++;
          }
          continue;
        }

        // Check if event already exists
        const existingAppointment = await storage.getAppointmentByGoogleEventId(event.id);

        if (existingAppointment) {
          // Update existing appointment
          await storage.updateAppointment(existingAppointment.id, {
            title: event.summary || existingAppointment.title,
            description: event.description || existingAppointment.description || undefined,
            startTime: event.start?.dateTime ? new Date(event.start.dateTime) : existingAppointment.startTime,
            endTime: event.end?.dateTime ? new Date(event.end.dateTime) : existingAppointment.endTime,
            location: event.location || existingAppointment.location || undefined,
            meetingLink: event.hangoutLink || existingAppointment.meetingLink || undefined,
          });
          updated++;
        } else {
          // Create new appointment
          if (!event.start?.dateTime || !event.end?.dateTime) continue;

          const startTime = new Date(event.start.dateTime);
          const endTime = new Date(event.end.dateTime);
          const duration = Math.round((endTime.getTime() - startTime.getTime()) / 60000);

          await storage.createAppointment({
            doctorId,
            title: event.summary || 'Untitled Event',
            description: event.description || undefined,
            appointmentType: 'consultation',
            startTime,
            endTime,
            duration,
            location: event.location || undefined,
            meetingLink: event.hangoutLink || undefined,
            googleCalendarEventId: event.id,
            status: 'scheduled',
          });
          created++;
        }
      } catch (error) {
        console.error('[GoogleCalendarSync] Error processing event:', event.id, error);
        conflicts++;
      }
    }

    return { created, updated, deleted, conflicts };
  }

  /**
   * Sync local appointment to Google Calendar
   */
  async syncAppointmentToGoogle(
    appointmentId: string
  ): Promise<{ success: boolean; eventId?: string; error?: string }> {
    try {
      const appointment = await storage.getAppointment(appointmentId);
      if (!appointment) {
        throw new Error('Appointment not found');
      }

      const syncConfig = await storage.getGoogleCalendarSync(appointment.doctorId);
      if (!syncConfig || !syncConfig.syncEnabled) {
        throw new Error('Google Calendar sync not enabled');
      }

      if (syncConfig.syncDirection === 'from_google') {
        // Read-only from Google
        return { success: false, error: 'Sync direction is from_google only' };
      }

      const auth = await this.getAuthClient(appointment.doctorId);
      const calendar = google.calendar({ version: 'v3', auth });
      const calendarId = syncConfig.calendarId || 'primary';

      const event: calendar_v3.Schema$Event = {
        summary: appointment.title,
        description: appointment.description || undefined,
        location: appointment.location || undefined,
        start: {
          dateTime: appointment.startTime.toISOString(),
          timeZone: 'UTC',
        },
        end: {
          dateTime: appointment.endTime.toISOString(),
          timeZone: 'UTC',
        },
      };

      if (appointment.meetingLink) {
        event.conferenceData = {
          conferenceSolution: {
            name: 'Google Meet',
          },
          entryPoints: [{
            entryPointType: 'video',
            uri: appointment.meetingLink,
          }],
        };
      }

      let googleEventId: string;

      if (appointment.googleCalendarEventId) {
        // Update existing event
        const response = await calendar.events.update({
          calendarId,
          eventId: appointment.googleCalendarEventId,
          requestBody: event,
        });
        googleEventId = response.data.id!;
      } else {
        // Create new event
        const response = await calendar.events.insert({
          calendarId,
          requestBody: event,
        });
        googleEventId = response.data.id!;

        // Update appointment with Google event ID
        await storage.updateAppointment(appointmentId, {
          googleCalendarEventId: googleEventId,
        });
      }

      return { success: true, eventId: googleEventId };
    } catch (error) {
      console.error('[GoogleCalendarSync] Error syncing to Google:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Delete event from Google Calendar
   */
  async deleteEventFromGoogle(appointmentId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const appointment = await storage.getAppointment(appointmentId);
      if (!appointment || !appointment.googleCalendarEventId) {
        return { success: true }; // Nothing to delete
      }

      const syncConfig = await storage.getGoogleCalendarSync(appointment.doctorId);
      if (!syncConfig || !syncConfig.syncEnabled) {
        return { success: true }; // Sync not enabled
      }

      if (syncConfig.syncDirection === 'from_google') {
        return { success: false, error: 'Sync direction is from_google only' };
      }

      const auth = await this.getAuthClient(appointment.doctorId);
      const calendar = google.calendar({ version: 'v3', auth });
      const calendarId = syncConfig.calendarId || 'primary';

      await calendar.events.delete({
        calendarId,
        eventId: appointment.googleCalendarEventId,
      });

      return { success: true };
    } catch (error) {
      console.error('[GoogleCalendarSync] Error deleting from Google:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Log sync activity
   */
  private async logSync(
    doctorId: string,
    syncId: string | null,
    syncType: 'full' | 'incremental' | 'manual',
    syncDirection: string,
    status: 'success' | 'partial' | 'failed',
    stats: { eventsCreated: number; eventsUpdated: number; eventsDeleted: number; conflictsDetected: number },
    durationMs: number,
    error?: string
  ): Promise<void> {
    try {
      const logData: InsertGoogleCalendarSyncLog = {
        doctorId,
        syncId,
        syncType,
        syncDirection,
        status,
        eventsCreated: stats.eventsCreated,
        eventsUpdated: stats.eventsUpdated,
        eventsDeleted: stats.eventsDeleted,
        conflictsDetected: stats.conflictsDetected,
        durationMs,
        error: error || undefined,
      };

      await storage.createGoogleCalendarSyncLog(logData);
    } catch (error) {
      console.error('[GoogleCalendarSync] Error logging sync:', error);
    }
  }

  /**
   * Disconnect Google Calendar for a doctor
   */
  async disconnect(doctorId: string): Promise<{ success: boolean }> {
    try {
      await storage.updateGoogleCalendarSync(doctorId, {
        syncEnabled: false,
        accessToken: null,
        refreshToken: null,
        tokenExpiry: null,
      });
      return { success: true };
    } catch (error) {
      console.error('[GoogleCalendarSync] Error disconnecting:', error);
      return { success: false };
    }
  }
}

// Export singleton instance
export const googleCalendarSyncService = new GoogleCalendarSyncService();
