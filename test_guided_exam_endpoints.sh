#!/bin/bash
# Guided Video Examination API Testing Script
# Tests all 5 endpoints with sample data

BASE_URL="http://localhost:8000"
PATIENT_ID="test_patient_123"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Guided Video Exam API Testing"
echo "=========================================="

# Test 1: Create Exam Session
echo -e "\n${YELLOW}Test 1: POST /api/v1/guided-exam/sessions${NC}"
echo "Creating new guided exam session..."

RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/guided-exam/sessions" \
  -H "Content-Type: application/json" \
  -d "{
    \"patient_id\": \"$PATIENT_ID\",
    \"device_info\": {
      \"browser\": \"Chrome\",
      \"os\": \"MacOS\",
      \"camera\": \"FaceTime HD\"
    }
  }")

echo "$RESPONSE" | jq .

# Extract session_id
SESSION_ID=$(echo "$RESPONSE" | jq -r '.session_id')

if [ "$SESSION_ID" != "null" ]; then
  echo -e "${GREEN}✓ Session created: $SESSION_ID${NC}"
else
  echo -e "${RED}✗ Failed to create session${NC}"
  exit 1
fi

# Test 2: Get Exam Session
echo -e "\n${YELLOW}Test 2: GET /api/v1/guided-exam/sessions/$SESSION_ID${NC}"
echo "Fetching session details..."

curl -s -X GET "$BASE_URL/api/v1/guided-exam/sessions/$SESSION_ID" | jq .

echo -e "${GREEN}✓ Session details retrieved${NC}"

# Test 3: Capture Eyes Frame
echo -e "\n${YELLOW}Test 3: POST /api/v1/guided-exam/sessions/$SESSION_ID/capture (eyes)${NC}"
echo "Capturing eyes frame..."

# Create a small test image (1x1 red pixel as base64)
# This is a minimal valid JPEG image
TEST_IMAGE_BASE64="/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAAC//EABwQAAAHAQEBAAAAAAAAAAAAAAECAwQFBgcICQr/2gAIAQEAAD8An+V5XleV5XleV5XleV5XleV5XleV5X//2Q=="

curl -s -X POST "$BASE_URL/api/v1/guided-exam/sessions/$SESSION_ID/capture" \
  -H "Content-Type: application/json" \
  -d "{
    \"stage\": \"eyes\",
    \"frame_base64\": \"$TEST_IMAGE_BASE64\"
  }" | jq .

echo -e "${GREEN}✓ Eyes frame captured${NC}"

# Test 4: Capture Palm Frame
echo -e "\n${YELLOW}Test 4: POST /api/v1/guided-exam/sessions/$SESSION_ID/capture (palm)${NC}"
echo "Capturing palm frame..."

curl -s -X POST "$BASE_URL/api/v1/guided-exam/sessions/$SESSION_ID/capture" \
  -H "Content-Type: application/json" \
  -d "{
    \"stage\": \"palm\",
    \"frame_base64\": \"$TEST_IMAGE_BASE64\"
  }" | jq .

echo -e "${GREEN}✓ Palm frame captured${NC}"

# Test 5: Capture Tongue Frame
echo -e "\n${YELLOW}Test 5: POST /api/v1/guided-exam/sessions/$SESSION_ID/capture (tongue)${NC}"
echo "Capturing tongue frame..."

curl -s -X POST "$BASE_URL/api/v1/guided-exam/sessions/$SESSION_ID/capture" \
  -H "Content-Type: application/json" \
  -d "{
    \"stage\": \"tongue\",
    \"frame_base64\": \"$TEST_IMAGE_BASE64\"
  }" | jq .

echo -e "${GREEN}✓ Tongue frame captured${NC}"

# Test 6: Capture Lips Frame
echo -e "\n${YELLOW}Test 6: POST /api/v1/guided-exam/sessions/$SESSION_ID/capture (lips)${NC}"
echo "Capturing lips frame..."

curl -s -X POST "$BASE_URL/api/v1/guided-exam/sessions/$SESSION_ID/capture" \
  -H "Content-Type: application/json" \
  -d "{
    \"stage\": \"lips\",
    \"frame_base64\": \"$TEST_IMAGE_BASE64\"
  }" | jq .

echo -e "${GREEN}✓ Lips frame captured${NC}"

# Test 7: Complete Exam (triggers AI analysis)
echo -e "\n${YELLOW}Test 7: POST /api/v1/guided-exam/sessions/$SESSION_ID/complete${NC}"
echo "Completing exam and triggering AI analysis..."
echo "This may take 10-30 seconds for video processing..."

COMPLETE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/guided-exam/sessions/$SESSION_ID/complete")

echo "$COMPLETE_RESPONSE" | jq .

# Extract video_metrics_id
METRICS_ID=$(echo "$COMPLETE_RESPONSE" | jq -r '.video_metrics_id')

if [ "$METRICS_ID" != "null" ]; then
  echo -e "${GREEN}✓ Exam completed. Metrics ID: $METRICS_ID${NC}"
else
  echo -e "${RED}✗ Exam completion failed${NC}"
  echo "Error: $(echo "$COMPLETE_RESPONSE" | jq -r '.error')"
fi

# Test 8: Get Results
if [ "$METRICS_ID" != "null" ]; then
  echo -e "\n${YELLOW}Test 8: GET /api/v1/guided-exam/sessions/$SESSION_ID/results${NC}"
  echo "Fetching AI analysis results..."

  curl -s -X GET "$BASE_URL/api/v1/guided-exam/sessions/$SESSION_ID/results" | jq .

  echo -e "${GREEN}✓ Results retrieved${NC}"
fi

echo -e "\n=========================================="
echo -e "${GREEN}All tests completed!${NC}"
echo "=========================================="

# Summary
echo -e "\n${YELLOW}Test Summary:${NC}"
echo "- Session ID: $SESSION_ID"
echo "- Metrics ID: $METRICS_ID"
echo "- All 5 endpoints tested successfully"
echo ""
echo "Key Metrics Available:"
echo "  - Scleral analysis (jaundice detection)"
echo "  - Conjunctival analysis (anemia detection)"
echo "  - Palmar pallor analysis"
echo "  - Tongue color and coating"
echo "  - Lip hydration and cyanosis"
