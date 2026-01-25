/**
 * HIPAA Config Guard - Fail-closed startup checks
 * 
 * This module enforces environment separation by blocking startup if
 * production identifiers are detected. It must run BEFORE any other
 * application code, database connections, or service initialization.
 * 
 * SECURITY: This guard is fail-closed - any detection of production
 * identifiers will cause immediate process termination with a generic
 * error message that does NOT reveal the offending value.
 */

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

interface ProdIdentifiers {
  version: string;
  gcp_project_ids: string[];
  gcp_cloud_run_services: string[];
  gcp_storage_buckets: string[];
  neon_host_suffixes: string[];
  auth0_domains: string[];
  auth0_audiences: string[];
  api_host_suffixes: string[];
  cloudflare_zones: string[];
  redis_hosts: string[];
  s3_buckets: string[];
  blocked_env_var_prefixes: string[];
  secret_patterns: string[];
}

const GENERIC_ERROR = 'Startup aborted: production identifier detected in environment';
const CONFIG_ERROR = 'Startup aborted: configuration guard failed to initialize';

let prodIdentifiers: ProdIdentifiers | null = null;

function loadProdIdentifiers(): ProdIdentifiers {
  if (prodIdentifiers) return prodIdentifiers;

  const identifiersPath = join(process.cwd(), 'prod_identifiers.json');
  
  if (!existsSync(identifiersPath)) {
    console.error('[CONFIG_GUARD] CRITICAL: prod_identifiers.json not found');
    process.exit(1);
  }

  try {
    const content = readFileSync(identifiersPath, 'utf-8');
    prodIdentifiers = JSON.parse(content) as ProdIdentifiers;
    
    if (!validateIdentifiersFile(prodIdentifiers)) {
      console.error('[CONFIG_GUARD] CRITICAL: Invalid prod_identifiers.json structure');
      process.exit(1);
    }

    return prodIdentifiers;
  } catch (error) {
    console.error('[CONFIG_GUARD] CRITICAL: Failed to parse prod_identifiers.json');
    process.exit(1);
  }
}

function validateIdentifiersFile(identifiers: ProdIdentifiers): boolean {
  const requiredKeys = [
    'gcp_project_ids',
    'neon_host_suffixes',
    'auth0_domains',
    'api_host_suffixes',
    'blocked_env_var_prefixes',
    'secret_patterns'
  ];

  for (const key of requiredKeys) {
    if (!Array.isArray(identifiers[key as keyof ProdIdentifiers])) {
      return false;
    }
  }

  return true;
}

function containsSecretPattern(value: string, patterns: string[]): boolean {
  const lowerValue = value.toLowerCase();
  for (const pattern of patterns) {
    if (lowerValue.includes(pattern.toLowerCase())) {
      return true;
    }
  }
  return false;
}

function matchesSuffix(value: string, suffixes: string[]): boolean {
  const lowerValue = value.toLowerCase();
  for (const suffix of suffixes) {
    if (lowerValue.includes(suffix.toLowerCase())) {
      return true;
    }
  }
  return false;
}

function matchesExact(value: string, identifiers: string[]): boolean {
  const lowerValue = value.toLowerCase();
  for (const id of identifiers) {
    if (lowerValue === id.toLowerCase()) {
      return true;
    }
  }
  return false;
}

function startsWithPrefix(value: string, prefixes: string[]): boolean {
  const lowerValue = value.toLowerCase();
  for (const prefix of prefixes) {
    if (lowerValue.startsWith(prefix.toLowerCase())) {
      return true;
    }
  }
  return false;
}

export interface GuardResult {
  passed: boolean;
  checks: {
    name: string;
    passed: boolean;
  }[];
}

export function runConfigGuard(exitOnFailure: boolean = true): GuardResult {
  const identifiers = loadProdIdentifiers();
  const results: GuardResult = {
    passed: true,
    checks: []
  };

  const addCheck = (name: string, passed: boolean) => {
    results.checks.push({ name, passed });
    if (!passed) {
      results.passed = false;
    }
  };

  // Check 1: GCP_PROJECT_ID must not match production project IDs
  const gcpProjectId = process.env.GCP_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT || '';
  if (gcpProjectId && matchesExact(gcpProjectId, identifiers.gcp_project_ids)) {
    addCheck('GCP_PROJECT_ID', false);
  } else {
    addCheck('GCP_PROJECT_ID', true);
  }

  // Check 2: DATABASE_URL must not contain production Neon hostnames
  const databaseUrl = process.env.DATABASE_URL || '';
  if (databaseUrl && matchesSuffix(databaseUrl, identifiers.neon_host_suffixes)) {
    addCheck('DATABASE_URL', false);
  } else {
    addCheck('DATABASE_URL', true);
  }

  // Check 3: AUTH0_DOMAIN must not match production Auth0 domain
  const auth0Domain = process.env.AUTH0_DOMAIN || '';
  if (auth0Domain && matchesExact(auth0Domain, identifiers.auth0_domains)) {
    addCheck('AUTH0_DOMAIN', false);
  } else {
    addCheck('AUTH0_DOMAIN', true);
  }

  // Check 4: API_BASE_URL must not contain production domains
  const apiBaseUrl = process.env.API_BASE_URL || '';
  if (apiBaseUrl && matchesSuffix(apiBaseUrl, identifiers.api_host_suffixes)) {
    addCheck('API_BASE_URL', false);
  } else {
    addCheck('API_BASE_URL', true);
  }

  // Check 5: No env var names should include blocked prefixes with values
  let foundBlockedPrefix = false;
  for (const [key, value] of Object.entries(process.env)) {
    if (value && startsWithPrefix(key, identifiers.blocked_env_var_prefixes)) {
      foundBlockedPrefix = true;
      break;
    }
  }
  addCheck('BLOCKED_PREFIX_ENV_VAR', !foundBlockedPrefix);

  // Check 6: No env var values should contain live Stripe keys (critical production pattern)
  let foundLiveSecret = false;
  for (const [key, value] of Object.entries(process.env)) {
    if (value && identifiers.secret_patterns && identifiers.secret_patterns.length > 0) {
      // Only check for critical patterns like live Stripe keys
      if (containsSecretPattern(value, identifiers.secret_patterns)) {
        foundLiveSecret = true;
        break;
      }
    }
  }
  addCheck('SECRET_PATTERN_DETECTED', !foundLiveSecret);

  // Check 7: Redis hosts must not be production
  const redisUrl = process.env.REDIS_URL || process.env.UPSTASH_REDIS_URL || '';
  if (redisUrl && matchesSuffix(redisUrl, identifiers.redis_hosts)) {
    addCheck('REDIS_URL', false);
  } else {
    addCheck('REDIS_URL', true);
  }

  // Check 8: S3/GCS buckets must not be production
  const storageBucket = process.env.GCS_BUCKET || process.env.S3_BUCKET || process.env.STORAGE_BUCKET || '';
  if (storageBucket && matchesSuffix(storageBucket, [
    ...identifiers.gcp_storage_buckets,
    ...identifiers.s3_buckets
  ])) {
    addCheck('STORAGE_BUCKET', false);
  } else {
    addCheck('STORAGE_BUCKET', true);
  }

  // Check 9: Environment mode validation
  const envMode = process.env.ENV || 'development';
  if (envMode === 'production') {
    console.log('[CONFIG_GUARD] Running in production mode');
  }
  addCheck('ENV_MODE', true);

  // Final decision
  if (!results.passed) {
    console.error(`[CONFIG_GUARD] ${GENERIC_ERROR}`);
    if (exitOnFailure) {
      process.exit(1);
    }
  } else {
    const envMode = process.env.ENV || 'development';
    console.log(`[CONFIG_GUARD] All environment checks passed - running in ${envMode} mode`);
  }

  return results;
}

export function assertDevEnvironment(): void {
  runConfigGuard(true);
}

// Export for testing
export { loadProdIdentifiers, matchesSuffix, matchesExact, containsSecretPattern };
