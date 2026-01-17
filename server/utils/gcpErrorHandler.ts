/**
 * Standardized GCP Error Handler
 * 
 * Provides consistent error handling and logging for all GCP operations.
 * Includes HIPAA audit logging for PHI-related operations.
 */

import { Response } from "express";
import { GCP_ERROR_CODES, HIPAA_AUDIT_ACTIONS } from "../config/gcpConstants";

export interface GCPError {
  code: number;
  message: string;
  details?: any;
  operation?: string;
  resourceType?: string;
  resourceId?: string;
}

export interface AuditLogEntry {
  timestamp: string;
  action: string;
  userId?: string;
  resourceType: string;
  resourceId?: string;
  ipAddress?: string;
  userAgent?: string;
  success: boolean;
  errorMessage?: string;
  metadata?: Record<string, any>;
}

class GCPErrorHandler {
  private static formatError(error: any, operation: string): GCPError {
    const gcpError: GCPError = {
      code: GCP_ERROR_CODES.INTERNAL,
      message: "An unexpected error occurred",
      operation,
    };

    if (error.code) {
      gcpError.code = typeof error.code === "number" ? error.code : GCP_ERROR_CODES.INTERNAL;
    }

    if (error.message) {
      gcpError.message = error.message;
    }

    if (error.details) {
      gcpError.details = error.details;
    }

    return gcpError;
  }

  static handleStorageError(error: any, operation: string): GCPError {
    const gcpError = this.formatError(error, operation);

    if (error.code === 404 || error.message?.includes("No such object")) {
      gcpError.code = GCP_ERROR_CODES.NOT_FOUND;
      gcpError.message = "File not found";
    } else if (error.code === 403 || error.message?.includes("Permission denied")) {
      gcpError.code = GCP_ERROR_CODES.PERMISSION_DENIED;
      gcpError.message = "Access denied to storage resource";
    } else if (error.code === 429) {
      gcpError.code = GCP_ERROR_CODES.RESOURCE_EXHAUSTED;
      gcpError.message = "Rate limit exceeded. Please try again later.";
    }

    console.error(`[GCP Storage Error] ${operation}:`, gcpError);
    return gcpError;
  }

  static handleKMSError(error: any, operation: string): GCPError {
    const gcpError = this.formatError(error, operation);

    if (error.message?.includes("not found")) {
      gcpError.code = GCP_ERROR_CODES.NOT_FOUND;
      gcpError.message = "Encryption key not found";
    } else if (error.message?.includes("Permission denied")) {
      gcpError.code = GCP_ERROR_CODES.PERMISSION_DENIED;
      gcpError.message = "Access denied to encryption service";
    }

    console.error(`[GCP KMS Error] ${operation}:`, gcpError);
    return gcpError;
  }

  static handleHealthcareError(error: any, operation: string): GCPError {
    const gcpError = this.formatError(error, operation);

    if (error.response?.status === 404) {
      gcpError.code = GCP_ERROR_CODES.NOT_FOUND;
      gcpError.message = "Healthcare resource not found";
    } else if (error.response?.status === 403) {
      gcpError.code = GCP_ERROR_CODES.PERMISSION_DENIED;
      gcpError.message = "Access denied to healthcare data";
    }

    console.error(`[GCP Healthcare Error] ${operation}:`, gcpError);
    return gcpError;
  }

  static handleDocumentAIError(error: any, operation: string): GCPError {
    const gcpError = this.formatError(error, operation);

    if (error.message?.includes("Invalid document")) {
      gcpError.code = GCP_ERROR_CODES.INVALID_ARGUMENT;
      gcpError.message = "Invalid document format";
    }

    console.error(`[GCP Document AI Error] ${operation}:`, gcpError);
    return gcpError;
  }

  static sendErrorResponse(res: Response, error: GCPError): void {
    const statusCode = error.code >= 400 && error.code < 600 ? error.code : 500;
    res.status(statusCode).json({
      success: false,
      error: error.message,
      code: error.code,
      operation: error.operation,
    });
  }

  static createAuditLog(
    action: keyof typeof HIPAA_AUDIT_ACTIONS,
    resourceType: string,
    success: boolean,
    options: {
      userId?: string;
      resourceId?: string;
      ipAddress?: string;
      userAgent?: string;
      errorMessage?: string;
      metadata?: Record<string, any>;
    } = {}
  ): AuditLogEntry {
    const entry: AuditLogEntry = {
      timestamp: new Date().toISOString(),
      action: HIPAA_AUDIT_ACTIONS[action],
      resourceType,
      success,
      ...options,
    };

    console.log(`[HIPAA Audit] ${JSON.stringify(entry)}`);
    return entry;
  }

  static async withErrorHandling<T>(
    operation: string,
    fn: () => Promise<T>,
    errorHandler: (error: any, operation: string) => GCPError = this.handleStorageError.bind(this)
  ): Promise<{ success: true; data: T } | { success: false; error: GCPError }> {
    try {
      const data = await fn();
      return { success: true, data };
    } catch (error) {
      const gcpError = errorHandler(error, operation);
      return { success: false, error: gcpError };
    }
  }
}

export const gcpErrorHandler = GCPErrorHandler;
export default GCPErrorHandler;
