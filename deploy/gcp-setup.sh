#!/usr/bin/env bash
# One-time Google Cloud setup for the CD workflow (see docs/DEPLOY.md).
#
# Run it in Cloud Shell (gcloud is installed and signed in there) or wherever gcloud is installed:
#   bash deploy/gcp-setup.sh <PROJECT_ID> [REGION] [GITHUB_REPOSITORY]
#
# It creates, or reuses when they already exist:
#   - the Cloud Run, Artifact Registry, IAM and Security Token Service APIs;
#   - an Artifact Registry Docker repository that keeps only the 3 most recent images;
#   - a "github-deployer" service account, allowed to deploy Cloud Run services and push images, nothing else;
#   - a "meteoreading-run" service account the app runs as, with no role at all;
#   - a Workload Identity pool and provider that trust GitHub's OIDC tokens for this repository only:
#     GitHub Actions gets short-lived credentials, no service account key is ever created.
# It ends by printing the GitHub variables to set.
set -euo pipefail

PROJECT_ID="${1:?usage: bash deploy/gcp-setup.sh <PROJECT_ID> [REGION] [GITHUB_REPOSITORY]}"
REGION="${2:-europe-west1}"
GITHUB_REPOSITORY="${3:-ElliotGalaor/MeteoReading}"
REPOSITORY=meteoreading
DEPLOY_SA_NAME=github-deployer
RUNTIME_SA_NAME=meteoreading-run
POOL=github
PROVIDER=github-oidc

gcloud config set project "$PROJECT_ID" > /dev/null
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
DEPLOY_SA="$DEPLOY_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
RUNTIME_SA="$RUNTIME_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
PROVIDER_NAME="projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER"

echo "1/5 Enabling the APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com iam.googleapis.com \
    iamcredentials.googleapis.com sts.googleapis.com

echo "2/5 Artifact Registry repository '$REPOSITORY' ($REGION)"
if ! gcloud artifacts repositories describe "$REPOSITORY" --location "$REGION" > /dev/null 2>&1; then
    gcloud artifacts repositories create "$REPOSITORY" --repository-format=docker --location "$REGION" \
        --description="Météo Analytics images deployed by GitHub Actions"
fi
# Keep the 3 most recent images (enough to roll back), delete the older ones: storage stays small
policy=$(mktemp)
cat > "$policy" << 'JSON'
[
  {"name": "keep-3-most-recent", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 3}},
  {"name": "delete-the-rest", "action": {"type": "Delete"}, "condition": {"tagState": "any"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies "$REPOSITORY" --location "$REGION" --policy "$policy" --no-dry-run > /dev/null
rm -f "$policy"

echo "3/5 Service accounts"
gcloud iam service-accounts describe "$DEPLOY_SA" > /dev/null 2>&1 \
    || gcloud iam service-accounts create "$DEPLOY_SA_NAME" --display-name="GitHub Actions deployer ($GITHUB_REPOSITORY)"
gcloud iam service-accounts describe "$RUNTIME_SA" > /dev/null 2>&1 \
    || gcloud iam service-accounts create "$RUNTIME_SA_NAME" --display-name="Météo Analytics on Cloud Run (no role)"

echo "4/5 Permissions of the deployer"
# Deploy Cloud Run services and make them public (--allow-unauthenticated)
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$DEPLOY_SA" \
    --role roles/run.admin --condition=None > /dev/null
# Push images to this repository only
gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" --location "$REGION" \
    --member "serviceAccount:$DEPLOY_SA" --role roles/artifactregistry.writer > /dev/null
# Start services as the runtime account (and only that one)
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --member "serviceAccount:$DEPLOY_SA" \
    --role roles/iam.serviceAccountUser > /dev/null

echo "5/5 Workload Identity Federation (GitHub OIDC, repository $GITHUB_REPOSITORY only)"
gcloud iam workload-identity-pools describe "$POOL" --location global > /dev/null 2>&1 \
    || gcloud iam workload-identity-pools create "$POOL" --location global --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers describe "$PROVIDER" --location global \
        --workload-identity-pool "$POOL" > /dev/null 2>&1 \
    || gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location global \
        --workload-identity-pool "$POOL" --display-name "GitHub OIDC" \
        --issuer-uri "https://token.actions.githubusercontent.com" \
        --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
        --attribute-condition "assertion.repository == '$GITHUB_REPOSITORY'"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" --role roles/iam.workloadIdentityUser \
    --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$GITHUB_REPOSITORY" \
    > /dev/null

cat << EOF

Done. Set these GitHub repository variables (Settings > Secrets and variables > Actions > Variables),
or run these commands where the GitHub CLI is signed in:

gh variable set GCP_PROJECT_ID --repo $GITHUB_REPOSITORY --body "$PROJECT_ID"
gh variable set GCP_REGION --repo $GITHUB_REPOSITORY --body "$REGION"
gh variable set GCP_ARTIFACT_REPOSITORY --repo $GITHUB_REPOSITORY --body "$REPOSITORY"
gh variable set GCP_WORKLOAD_IDENTITY_PROVIDER --repo $GITHUB_REPOSITORY --body "$PROVIDER_NAME"
gh variable set GCP_DEPLOY_SERVICE_ACCOUNT --repo $GITHUB_REPOSITORY --body "$DEPLOY_SA"
gh variable set GCP_RUNTIME_SERVICE_ACCOUNT --repo $GITHUB_REPOSITORY --body "$RUNTIME_SA"
EOF
