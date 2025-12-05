/**
 * Audit Logging for HIPAA Compliance
 * 
 * HIPAA Security: Maintains comprehensive audit trails of all
 * PHI access, modifications, and security events.
 */

export enum AuditEventType {
  API_REQUEST = 'api_request',
  API_RESPONSE = 'api_response',
  API_ERROR = 'api_error',
  AUTH_SUCCESS = 'auth_success',
  AUTH_FAILURE = 'auth_failure',
  PHI_ACCESS = 'phi_access',
  PHI_MODIFICATION = 'phi_modification',
  SECURITY_VIOLATION = 'security_violation',
  RATE_LIMIT_EXCEEDED = 'rate_limit_exceeded',
  CSRF_VIOLATION = 'csrf_violation',
  SSRF_ATTEMPT = 'ssrf_attempt',
}

export interface AuditLogEntry {
  timestamp: number;
  eventType: AuditEventType;
  userId?: string;
  sessionId?: string;
  url?: string;
  method?: string;
  statusCode?: number;
  error?: string;
  metadata?: Record<string, any>;
  ipAddress?: string;
  userAgent?: string;
}

class AuditLogger {
  private logs: AuditLogEntry[] = [];
  private readonly maxLogs = 1000; // Keep last 1000 logs in memory
  private readonly enableConsoleLogging = import.meta.env.DEV;
  private readonly enableRemoteLogging = !import.meta.env.DEV;

  /**
   * Logs an audit event
   */
  log(entry: Omit<AuditLogEntry, 'timestamp'>): void {
    const fullEntry: AuditLogEntry = {
      ...entry,
      timestamp: Date.now(),
      ipAddress: this.getClientIP(),
      userAgent: navigator.userAgent,
      sessionId: this.getSessionId(),
    };

    // Add to in-memory log
    this.logs.push(fullEntry);
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }

    // Console logging in development
    if (this.enableConsoleLogging) {
      console.log('[AUDIT]', fullEntry);
    }

    // Remote logging in production
    if (this.enableRemoteLogging) {
      this.sendToRemoteLogger(fullEntry).catch(error => {
        console.error('[AUDIT] Failed to send log to remote:', error);
      });
    }
  }

  /**
   * Gets recent logs
   */
  getRecentLogs(count: number = 100): AuditLogEntry[] {
    return this.logs.slice(-count);
  }

  /**
   * Gets logs by event type
   */
  getLogsByType(eventType: AuditEventType): AuditLogEntry[] {
    return this.logs.filter(log => log.eventType === eventType);
  }

  /**
   * Gets client IP (if available)
   */
  private getClientIP(): string {
    // In browser, we can't get real IP, but we can track session
    return 'browser-client';
  }

  /**
   * Gets session ID
   */
  private getSessionId(): string {
    // Try to get from sessionStorage or generate one
    if (typeof window === 'undefined') {
      return 'server-side';
    }

    let sessionId = sessionStorage.getItem('audit-session-id');
    if (!sessionId) {
      sessionId = this.generateSessionId();
      sessionStorage.setItem('audit-session-id', sessionId);
    }
    return sessionId;
  }

  /**
   * Generates a session ID
   */
  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
  }

  /**
   * Sends log to remote logger (e.g., backend endpoint)
   */
  private async sendToRemoteLogger(entry: AuditLogEntry): Promise<void> {
    try {
      // Send to backend audit endpoint
      await fetch('/api/v1/audit/log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(entry),
        credentials: 'include',
      });
    } catch (error) {
      // Silently fail - we don't want audit logging to break the app
      // But we keep it in memory
    }
  }
}

// Global audit logger instance
export const auditLogger = new AuditLogger();

/**
 * Helper functions for common audit events
 */
export const audit = {
  apiRequest: (url: string, method: string, userId?: string) => {
    auditLogger.log({
      eventType: AuditEventType.API_REQUEST,
      url,
      method,
      userId,
    });
  },

  apiResponse: (
    url: string,
    method: string,
    statusCode: number,
    userId?: string
  ) => {
    auditLogger.log({
      eventType: AuditEventType.API_RESPONSE,
      url,
      method,
      statusCode,
      userId,
    });
  },

  apiError: (
    url: string,
    method: string,
    error: string,
    statusCode?: number,
    userId?: string
  ) => {
    auditLogger.log({
      eventType: AuditEventType.API_ERROR,
      url,
      method,
      error,
      statusCode,
      userId,
    });
  },

  securityViolation: (
    violationType: string,
    details: Record<string, any>,
    userId?: string
  ) => {
    auditLogger.log({
      eventType: AuditEventType.SECURITY_VIOLATION,
      error: violationType,
      metadata: details,
      userId,
    });
  },

  phiAccess: (resourceType: string, resourceId: string, userId: string) => {
    auditLogger.log({
      eventType: AuditEventType.PHI_ACCESS,
      url: `/${resourceType}/${resourceId}`,
      method: 'GET',
      userId,
      metadata: { resourceType, resourceId },
    });
  },

  phiModification: (
    resourceType: string,
    resourceId: string,
    userId: string,
    action: string
  ) => {
    auditLogger.log({
      eventType: AuditEventType.PHI_MODIFICATION,
      url: `/${resourceType}/${resourceId}`,
      method: action,
      userId,
      metadata: { resourceType, resourceId, action },
    });
  },
};

