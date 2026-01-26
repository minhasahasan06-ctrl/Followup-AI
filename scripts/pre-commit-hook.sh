#!/bin/bash
#
# HIPAA Pre-Commit Hook
# 
# This script scans staged files for production identifiers,
# secrets, and potential PHI before allowing commits.
#
# Installation:
#   cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Or use with pre-commit framework by adding to .pre-commit-config.yaml

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running HIPAA pre-commit checks...${NC}"

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR)

if [ -z "$STAGED_FILES" ]; then
    echo -e "${GREEN}No staged files to check${NC}"
    exit 0
fi

ERRORS=0
WARNINGS=0

# Production identifier patterns to block
PROD_PATTERNS=(
    "api.followupai.com"
    "prod.followupai.com"
    "production.followupai.com"
    "followupai-prod.auth0.com"
    "followupai-production.auth0.com"
    ".prod.neon.tech"
    "-prod.neon.tech"
)

# Secret patterns to block
SECRET_PATTERNS=(
    "sk_live_"
    "pk_live_"
    "AKIA[A-Z0-9]{16}"
    "-----BEGIN PRIVATE KEY-----"
    "-----BEGIN RSA PRIVATE KEY-----"
    "-----BEGIN CERTIFICATE-----"
    "client_secret.*=.*['\"][^'\"]{20,}"
)

# PHI patterns to warn about (not block)
PHI_PATTERNS=(
    "[0-9]{3}-[0-9]{2}-[0-9]{4}"  # SSN
    "MRN[:\s#]*[0-9]{5,}"          # Medical Record Number
)

echo "Checking staged files for production identifiers..."

for file in $STAGED_FILES; do
    # Skip certain files
    if [[ "$file" == "prod_identifiers.json" ]] || \
       [[ "$file" == ".github/workflows/"* ]] || \
       [[ "$file" == "scripts/pre-commit-hook.sh" ]] || \
       [[ "$file" == "node_modules/"* ]] || \
       [[ "$file" == "*.md" ]]; then
        continue
    fi

    # Check for production identifiers
    for pattern in "${PROD_PATTERNS[@]}"; do
        if git show ":$file" 2>/dev/null | grep -q "$pattern"; then
            echo -e "${RED}ERROR: Production identifier found in $file${NC}"
            echo "  Pattern: $pattern"
            ERRORS=$((ERRORS + 1))
        fi
    done

    # Check for secrets
    for pattern in "${SECRET_PATTERNS[@]}"; do
        if git show ":$file" 2>/dev/null | grep -E "$pattern" 2>/dev/null; then
            echo -e "${RED}ERROR: Secret pattern found in $file${NC}"
            echo "  Pattern: $pattern"
            ERRORS=$((ERRORS + 1))
        fi
    done

    # Check for PHI patterns (warning only)
    for pattern in "${PHI_PATTERNS[@]}"; do
        if git show ":$file" 2>/dev/null | grep -E "$pattern" 2>/dev/null; then
            echo -e "${YELLOW}WARNING: Possible PHI pattern in $file${NC}"
            echo "  Pattern: $pattern"
            WARNINGS=$((WARNINGS + 1))
        fi
    done
done

# Check for .env files being committed
for file in $STAGED_FILES; do
    if [[ "$file" == ".env" ]] || \
       [[ "$file" == ".env.local" ]] || \
       [[ "$file" == ".env.production" ]] || \
       [[ "$file" == ".env.prod" ]]; then
        echo -e "${RED}ERROR: Attempting to commit .env file: $file${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check for service account JSON files
for file in $STAGED_FILES; do
    if [[ "$file" == *"service_account"* ]] || \
       [[ "$file" == *"serviceAccount"* ]] || \
       [[ "$file" == *"credentials"*.json ]]; then
        echo -e "${RED}ERROR: Attempting to commit credentials file: $file${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "================================================"

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}COMMIT BLOCKED: $ERRORS error(s) found${NC}"
    echo "Please remove production identifiers, secrets, or PHI before committing."
    echo ""
    echo "If this is a false positive, you can bypass with:"
    echo "  git commit --no-verify"
    echo ""
    echo "WARNING: Bypassing these checks may violate HIPAA compliance!"
    exit 1
fi

if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}WARNINGS: $WARNINGS potential issue(s) found${NC}"
    echo "Please review the warnings above to ensure no PHI is being committed."
fi

echo -e "${GREEN}All HIPAA pre-commit checks passed!${NC}"
exit 0
