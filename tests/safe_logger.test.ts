/**
 * Safe Logger Tests
 * 
 * These tests verify the safe_logger correctly redacts
 * PHI and sensitive information from log output.
 */

import { redactString, redactObject, safeLogger, PHI_PATTERNS, SENSITIVE_FIELDS } from '../server/safe_logger';

describe('Safe Logger', () => {
  describe('redactString', () => {
    it('should redact email addresses', () => {
      const input = 'Contact john.doe@email.com for info';
      const result = redactString(input);
      expect(result).not.toContain('john.doe@email.com');
      expect(result).toContain('[EMAIL_REDACTED]');
    });

    it('should redact phone numbers', () => {
      const input = 'Call me at 555-123-4567';
      const result = redactString(input);
      expect(result).not.toContain('555-123-4567');
      expect(result).toContain('[PHONE_REDACTED]');
    });

    it('should redact SSN', () => {
      const input = 'SSN: 123-45-6789';
      const result = redactString(input);
      expect(result).not.toContain('123-45-6789');
      expect(result).toContain('[SSN_REDACTED]');
    });

    it('should redact medical record numbers', () => {
      const input = 'MRN: 12345678';
      const result = redactString(input);
      expect(result).not.toContain('12345678');
      expect(result).toContain('[MRN_REDACTED]');
    });

    it('should redact credit card numbers', () => {
      const input = 'Card: 4111-1111-1111-1111';
      const result = redactString(input);
      expect(result).not.toContain('4111-1111-1111-1111');
      expect(result).toContain('[CARD_REDACTED]');
    });

    it('should redact JWT tokens', () => {
      const input = 'Token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123xyz';
      const result = redactString(input);
      expect(result).not.toContain('eyJhbGciOiJIUzI1NiJ9');
      expect(result).toContain('[JWT_REDACTED]');
    });

    it('should redact API keys', () => {
      const input = 'Key: sk_live_abc123defg456hijklmnop';
      const result = redactString(input);
      expect(result).not.toContain('sk_live_');
      expect(result).toContain('[API_KEY_REDACTED]');
    });

    it('should redact IP addresses', () => {
      const input = 'Request from 192.168.1.100';
      const result = redactString(input);
      expect(result).not.toContain('192.168.1.100');
      expect(result).toContain('[IP_REDACTED]');
    });

    it('should redact dates of birth', () => {
      const input = 'DOB: 01/15/1990';
      const result = redactString(input);
      expect(result).not.toContain('01/15/1990');
      expect(result).toContain('[DOB_REDACTED]');
    });

    it('should handle multiple patterns', () => {
      const input = 'User john@test.com called 555-123-4567 with SSN 123-45-6789';
      const result = redactString(input);
      expect(result).not.toContain('john@test.com');
      expect(result).not.toContain('555-123-4567');
      expect(result).not.toContain('123-45-6789');
      expect(result).toContain('[EMAIL_REDACTED]');
      expect(result).toContain('[PHONE_REDACTED]');
      expect(result).toContain('[SSN_REDACTED]');
    });

    it('should handle empty strings', () => {
      expect(redactString('')).toBe('');
    });

    it('should handle null/undefined', () => {
      expect(redactString(null as any)).toBe(null);
      expect(redactString(undefined as any)).toBe(undefined);
    });
  });

  describe('redactObject', () => {
    it('should redact sensitive field values', () => {
      const input = {
        email: 'john@test.com',
        password: 'secret123',
        name: 'John Doe'
      };
      const result = redactObject(input);
      expect(result.email).toBe('[FIELD_REDACTED]');
      expect(result.password).toBe('[FIELD_REDACTED]');
    });

    it('should redact nested objects', () => {
      const input = {
        user: {
          email: 'test@example.com',
          ssn: '123-45-6789'
        }
      };
      const result = redactObject(input);
      expect(result.user.email).toBe('[FIELD_REDACTED]');
      expect(result.user.ssn).toBe('[FIELD_REDACTED]');
    });

    it('should redact arrays', () => {
      const input = {
        emails: ['john@test.com', 'jane@test.com']
      };
      const result = redactObject(input);
      expect(result.emails[0]).toContain('[EMAIL_REDACTED]');
      expect(result.emails[1]).toContain('[EMAIL_REDACTED]');
    });

    it('should handle deep nesting with depth limit', () => {
      let deep: any = { value: 'test@email.com' };
      for (let i = 0; i < 15; i++) {
        deep = { nested: deep };
      }
      const result = redactObject(deep);
      expect(result).toBeDefined();
    });

    it('should preserve non-sensitive values', () => {
      const input = {
        id: 123,
        status: 'active',
        count: 5
      };
      const result = redactObject(input);
      expect(result.id).toBe(123);
      expect(result.status).toBe('active');
      expect(result.count).toBe(5);
    });
  });

  describe('safeLogger.containsPHI', () => {
    it('should detect PHI patterns', () => {
      expect(safeLogger.containsPHI('john@email.com')).toBe(true);
      expect(safeLogger.containsPHI('555-123-4567')).toBe(true);
      expect(safeLogger.containsPHI('123-45-6789')).toBe(true);
    });

    it('should return false for safe strings', () => {
      expect(safeLogger.containsPHI('hello world')).toBe(false);
      expect(safeLogger.containsPHI('order-123')).toBe(false);
    });
  });

  describe('safeLogger.redact', () => {
    it('should redact strings', () => {
      const result = safeLogger.redact('Email: test@example.com');
      expect(result).toContain('[EMAIL_REDACTED]');
    });

    it('should redact objects', () => {
      const result = safeLogger.redact({ email: 'test@example.com' });
      expect(result.email).toBe('[FIELD_REDACTED]');
    });
  });
});

// Simple test runner for environments without Jest
if (typeof describe === 'undefined') {
  console.log('Running basic safe logger validation...');
  
  // Test 1: Email redaction
  const email = redactString('Contact: john.doe@email.com');
  console.log(`Test 1 (email redaction): ${email.includes('[EMAIL_REDACTED]') ? 'PASS' : 'FAIL'}`);
  
  // Test 2: Phone redaction
  const phone = redactString('Call: 555-123-4567');
  console.log(`Test 2 (phone redaction): ${phone.includes('[PHONE_REDACTED]') ? 'PASS' : 'FAIL'}`);
  
  // Test 3: SSN redaction
  const ssn = redactString('SSN: 123-45-6789');
  console.log(`Test 3 (SSN redaction): ${ssn.includes('[SSN_REDACTED]') ? 'PASS' : 'FAIL'}`);
  
  // Test 4: Object sensitive field redaction
  const obj = redactObject({ email: 'test@email.com', password: 'secret' });
  console.log(`Test 4 (object field redaction): ${obj.email === '[FIELD_REDACTED]' ? 'PASS' : 'FAIL'}`);
  
  // Test 5: Multiple patterns
  const multi = redactString('User john@test.com called 555-123-4567');
  const multiPass = multi.includes('[EMAIL_REDACTED]') && multi.includes('[PHONE_REDACTED]');
  console.log(`Test 5 (multiple patterns): ${multiPass ? 'PASS' : 'FAIL'}`);
  
  console.log('Safe logger validation complete');
}
