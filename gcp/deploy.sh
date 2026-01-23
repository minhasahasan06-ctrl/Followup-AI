#!/bin/bash
# Followup AI Backend - GCP Deployment Script
#
# This script automates the deployment process to Google Cloud Platform.
#
# Prerequisites:
# 1. gcloud CLI installed and authenticated
# 2. Docker installed (for local builds)
# 3. Project configured with required APIs enabled
#
# Usage:
#   ./gcp/deploy.sh [PROJECT_ID] [REGION]
#
# Example:
#   ./gcp/deploy.sh my-project-id us-central1

set -euo pipefail

# Configuration
PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-us-central1}"
SERVICE_NAME="followup-backend"
REPOSITORY="followup-ai"
IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    if [ -z "$PROJECT_ID" ]; then
        log_error "PROJECT_ID not set. Please provide as argument or set with: gcloud config set project PROJECT_ID"
        exit 1
    fi
    
    log_info "Using project: $PROJECT_ID"
    log_info "Using region: $REGION"
    log_info "Service name: $SERVICE_NAME"
    log_info "Image tag: $IMAGE_TAG"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required APIs..."
    
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        secretmanager.googleapis.com \
        --project="$PROJECT_ID" \
        --quiet
    
    log_info "APIs enabled successfully"
}

# Create Artifact Registry repository
create_repository() {
    log_info "Creating Artifact Registry repository..."
    
    if ! gcloud artifacts repositories describe "$REPOSITORY" \
        --location="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        
        gcloud artifacts repositories create "$REPOSITORY" \
            --repository-format=docker \
            --location="$REGION" \
            --description="Followup AI Docker images" \
            --project="$PROJECT_ID"
        
        log_info "Repository created: $REPOSITORY"
    else
        log_info "Repository already exists: $REPOSITORY"
    fi
}

# Create service account
create_service_account() {
    local sa_name="${SERVICE_NAME}"
    local sa_email="${sa_name}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    log_info "Setting up service account..."
    
    if ! gcloud iam service-accounts describe "$sa_email" \
        --project="$PROJECT_ID" &> /dev/null; then
        
        gcloud iam service-accounts create "$sa_name" \
            --display-name="Followup Backend Service Account" \
            --description="Service account for Followup AI backend" \
            --project="$PROJECT_ID"
        
        log_info "Service account created: $sa_email"
    else
        log_info "Service account already exists: $sa_email"
    fi
    
    # Grant necessary roles
    log_info "Granting IAM roles..."
    
    for role in \
        "roles/secretmanager.secretAccessor" \
        "roles/cloudsql.client" \
        "roles/storage.objectViewer"; do
        
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$sa_email" \
            --role="$role" \
            --quiet
    done
    
    log_info "IAM roles granted"
}

# Create secrets in Secret Manager
setup_secrets() {
    log_info "Setting up secrets..."
    
    local secrets=(
        "DATABASE_URL"
        "STYTCH_PROJECT_ID"
        "STYTCH_SECRET"
        "OPENAI_API_KEY"
        "OPENAI_BAA"
        "OPENAI_ZDR"
    )
    
    for secret in "${secrets[@]}"; do
        if ! gcloud secrets describe "$secret" --project="$PROJECT_ID" &> /dev/null; then
            log_warn "Secret $secret does not exist. Creating placeholder..."
            echo -n "PLACEHOLDER_VALUE" | gcloud secrets create "$secret" \
                --data-file=- \
                --project="$PROJECT_ID"
            log_warn "Please update secret $secret with actual value"
        else
            log_info "Secret $secret exists"
        fi
    done
}

# Build and push Docker image
build_and_push() {
    log_info "Building and pushing Docker image..."
    
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}"
    
    # Configure Docker for Artifact Registry
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    
    # Build image
    docker build \
        -t "${image_uri}:${IMAGE_TAG}" \
        -t "${image_uri}:latest" \
        -f gcp/Dockerfile \
        .
    
    # Push image
    docker push "${image_uri}:${IMAGE_TAG}"
    docker push "${image_uri}:latest"
    
    log_info "Image pushed: ${image_uri}:${IMAGE_TAG}"
}

# Deploy to Cloud Run
deploy_service() {
    log_info "Deploying to Cloud Run..."
    
    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG}"
    local sa_email="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    gcloud run deploy "$SERVICE_NAME" \
        --image="$image_uri" \
        --region="$REGION" \
        --platform=managed \
        --allow-unauthenticated \
        --min-instances=1 \
        --max-instances=10 \
        --memory=8Gi \
        --cpu=4 \
        --gpu=1 \
        --gpu-type=nvidia-l4 \
        --timeout=60 \
        --concurrency=80 \
        --set-env-vars="ENV=production,LOG_LEVEL=INFO" \
        --set-secrets="DATABASE_URL=DATABASE_URL:latest,STYTCH_PROJECT_ID=STYTCH_PROJECT_ID:latest,STYTCH_SECRET=STYTCH_SECRET:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,OPENAI_BAA=OPENAI_BAA:latest,OPENAI_ZDR=OPENAI_ZDR:latest" \
        --service-account="$sa_email" \
        --project="$PROJECT_ID"
    
    log_info "Deployment complete!"
}

# Get service URL
get_service_url() {
    local url=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format='value(status.url)')
    
    log_info "Service URL: $url"
    echo ""
    echo "=========================================="
    echo "Deployment Successful!"
    echo "=========================================="
    echo "Service URL: $url"
    echo ""
    echo "Update your Replit frontend with:"
    echo "VITE_API_URL=$url"
    echo "=========================================="
}

# Main execution
main() {
    echo "=========================================="
    echo "Followup AI - GCP Deployment"
    echo "=========================================="
    
    check_prerequisites
    enable_apis
    create_repository
    create_service_account
    setup_secrets
    build_and_push
    deploy_service
    get_service_url
}

# Run main function
main "$@"
