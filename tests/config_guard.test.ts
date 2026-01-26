/**
 * Config Guard Tests
 * 
 * These tests verify the config guard correctly detects
 * production identifiers and blocks application startup.
 */

import { runConfigGuard, matchesSuffix, matchesExact, containsSecretPattern } from '../server/config_guard';

describe('Config Guard', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  describe('matchesSuffix', () => {
    it('should detect matching suffixes', () => {
      expect(matchesSuffix('db.prod.neon.tech', ['.prod.neon.tech'])).toBe(true);
      expect(matchesSuffix('postgres://user:pass@db-prod.neon.tech/db', ['-prod.neon.tech'])).toBe(true);
    });

    it('should not match non-matching suffixes', () => {
      expect(matchesSuffix('db.dev.neon.tech', ['.prod.neon.tech'])).toBe(false);
      expect(matchesSuffix('db-staging.neon.tech', ['-prod.neon.tech'])).toBe(false);
    });

    it('should be case insensitive', () => {
      expect(matchesSuffix('DB.PROD.NEON.TECH', ['.prod.neon.tech'])).toBe(true);
    });
  });

  describe('matchesExact', () => {
    it('should detect exact matches', () => {
      expect(matchesExact('followupai-prod', ['followupai-prod'])).toBe(true);
      expect(matchesExact('followupai-prod.auth0.com', ['followupai-prod.auth0.com'])).toBe(true);
    });

    it('should not match partial strings', () => {
      expect(matchesExact('followupai-prod-extra', ['followupai-prod'])).toBe(false);
    });

    it('should be case insensitive', () => {
      expect(matchesExact('FOLLOWUPAI-PROD', ['followupai-prod'])).toBe(true);
    });
  });

  describe('containsSecretPattern', () => {
    it('should detect secret patterns', () => {
      expect(containsSecretPattern('sk_live_abc123xyz', ['sk_live_'])).toBe(true);
      expect(containsSecretPattern('Bearer token123', ['Bearer '])).toBe(true);
      expect(containsSecretPattern('-----BEGIN PRIVATE KEY-----', ['-----BEGIN'])).toBe(true);
    });

    it('should not match safe values', () => {
      expect(containsSecretPattern('sk_test_abc123', ['sk_live_'])).toBe(false);
      expect(containsSecretPattern('hello world', ['Bearer '])).toBe(false);
    });
  });

  describe('runConfigGuard', () => {
    it('should pass with no production identifiers', () => {
      process.env.GCP_PROJECT_ID = 'myapp-dev-123';
      process.env.DATABASE_URL = 'postgres://user:pass@db.dev.neon.tech/devdb';
      process.env.AUTH0_DOMAIN = 'myapp-dev.auth0.com';
      process.env.API_BASE_URL = 'https://dev.myapp.local';

      const result = runConfigGuard(false);
      expect(result.passed).toBe(true);
    });

    it('should fail with production GCP project ID', () => {
      process.env.GCP_PROJECT_ID = 'followupai-prod';

      const result = runConfigGuard(false);
      expect(result.passed).toBe(false);
      expect(result.checks.find(c => c.name === 'GCP_PROJECT_ID')?.passed).toBe(false);
    });

    it('should fail with production database URL', () => {
      process.env.DATABASE_URL = 'postgres://user:pass@db.prod.neon.tech/proddb';

      const result = runConfigGuard(false);
      expect(result.passed).toBe(false);
      expect(result.checks.find(c => c.name === 'DATABASE_URL')?.passed).toBe(false);
    });

    it('should fail with production Auth0 domain', () => {
      process.env.AUTH0_DOMAIN = 'followupai-prod.auth0.com';

      const result = runConfigGuard(false);
      expect(result.passed).toBe(false);
      expect(result.checks.find(c => c.name === 'AUTH0_DOMAIN')?.passed).toBe(false);
    });

    it('should fail with PROD_ prefixed env vars', () => {
      process.env.PROD_API_KEY = 'some-value';

      const result = runConfigGuard(false);
      expect(result.passed).toBe(false);
      expect(result.checks.find(c => c.name === 'BLOCKED_PREFIX_ENV_VAR')?.passed).toBe(false);
    });

    it('should fail with production API URL', () => {
      process.env.API_BASE_URL = 'https://api.followupai.com/v1';

      const result = runConfigGuard(false);
      expect(result.passed).toBe(false);
      expect(result.checks.find(c => c.name === 'API_BASE_URL')?.passed).toBe(false);
    });
  });
});

// Simple test runner for environments without Jest
if (typeof describe === 'undefined') {
  console.log('Running basic config guard validation...');
  
  // Test 1: Should pass with dev environment
  const devEnv = { ...process.env };
  devEnv.GCP_PROJECT_ID = 'test-dev';
  devEnv.DATABASE_URL = 'postgres://localhost/testdb';
  process.env = devEnv;
  
  const result1 = runConfigGuard(false);
  console.log(`Test 1 (dev env should pass): ${result1.passed ? 'PASS' : 'FAIL'}`);

  // Test 2: Should fail with prod identifier
  const prodEnv = { ...devEnv };
  prodEnv.GCP_PROJECT_ID = 'followupai-prod';
  process.env = prodEnv;
  
  const result2 = runConfigGuard(false);
  console.log(`Test 2 (prod env should fail): ${!result2.passed ? 'PASS' : 'FAIL'}`);

  console.log('Config guard validation complete');
}
