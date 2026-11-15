#!/usr/bin/env bash
# Caps the monthly cost of the project (see docs/DEPLOY.md, "Costs").
#
# Run it once the services exist (after the first deploy), in Cloud Shell or wherever gcloud is installed,
# signed in as a project owner who can manage the budgets of the billing account:
#   bash deploy/gcp-budget.sh <PROJECT_ID> [AMOUNT] [REGION]
# AMOUNT is in the currency of the billing account: 4.30 by default, about 5 USD for an account in EUR.
#
# A Google Cloud budget only sends alerts, it never stops anything. This script sets up:
#   - a monthly budget for the project, with e-mail alerts to the billing admins at 50%, 90% and 100%;
#   - a Pub/Sub topic where Cloud Billing posts the state of that budget several times a day;
#   - the "budget-guard" Cloud Function (deploy/budget_guard), subscribed to that topic: it takes the Cloud Run
#     services offline when the cost of the month reaches half the budget, and puts them back online when the
#     next month starts;
#   - a "budget-guard" service account, allowed to change those services and nothing else, and a
#     "budget-guard-build" one that builds the function.
# It can be run again safely, for instance with another amount.
set -euo pipefail

PROJECT_ID="${1:?usage: bash deploy/gcp-budget.sh <PROJECT_ID> [AMOUNT] [REGION] [SERVICE...]}"
AMOUNT="${2:-4.30}"
REGION="${3:-europe-west1}"
# The Cloud Run services to close: the default names, or the ones given after the region (if CLOUD_RUN_SERVICE
# renames them in the GitHub environments, pass the same names here)
if [ $# -gt 3 ]; then
    SERVICES=("${@:4}")
else
    SERVICES=(meteoreading-staging meteoreading)
fi
CLOSE_AT=0.5
REPOSITORY=meteoreading
RUNTIME_SA_NAME=meteoreading-run
GUARD=budget-guard
BUILD_SA_NAME=budget-guard-build
TOPIC=budget-alerts
BUDGET_NAME="meteoreading monthly cap"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/budget_guard"

gcloud config set project "$PROJECT_ID" > /dev/null
BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingAccountName)')
BILLING_ACCOUNT="${BILLING_ACCOUNT#billingAccounts/}"
if [ -z "$BILLING_ACCOUNT" ]; then
    echo "Billing is not enabled on $PROJECT_ID" >&2
    exit 1
fi
CURRENCY=$(gcloud billing accounts describe "$BILLING_ACCOUNT" --format='value(currencyCode)')
GUARD_SA="$GUARD@$PROJECT_ID.iam.gserviceaccount.com"
BUILD_SA="$BUILD_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
RUNTIME_SA="$RUNTIME_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

echo "1/6 Enabling the APIs"
gcloud services enable billingbudgets.googleapis.com pubsub.googleapis.com cloudfunctions.googleapis.com \
    cloudbuild.googleapis.com eventarc.googleapis.com run.googleapis.com artifactregistry.googleapis.com

echo "2/6 Pub/Sub topic '$TOPIC'"
gcloud pubsub topics describe "$TOPIC" > /dev/null 2>&1 || gcloud pubsub topics create "$TOPIC"
# Cloud Billing publishes the budget notifications as this Google-managed account
gcloud pubsub topics add-iam-policy-binding "$TOPIC" --role roles/pubsub.publisher \
    --member serviceAccount:billing-budget-alert@system.gserviceaccount.com > /dev/null

echo "3/6 Budget: $AMOUNT $CURRENCY per month for $PROJECT_ID"
# gcloud reads the decimals as nanos ("4.30" would be 4 units and 30 nanos): give all nine of them
budget_amount="$(awk "BEGIN { printf \"%.9f\", $AMOUNT }")$CURRENCY"
budget=$(gcloud billing budgets list --billing-account "$BILLING_ACCOUNT" \
    --filter "displayName=\"$BUDGET_NAME\"" --format 'value(name)')
if [ -z "$budget" ]; then
    gcloud billing budgets create --billing-account "$BILLING_ACCOUNT" --display-name "$BUDGET_NAME" \
        --budget-amount "$budget_amount" --calendar-period month --filter-projects "projects/$PROJECT_ID" \
        --threshold-rule percent=0.5 --threshold-rule percent=0.9 --threshold-rule percent=1.0 \
        --notifications-rule-pubsub-topic "projects/$PROJECT_ID/topics/$TOPIC" > /dev/null
else
    gcloud billing budgets update "${budget##*/}" --billing-account "$BILLING_ACCOUNT" \
        --budget-amount "$budget_amount" --notifications-rule-pubsub-topic "projects/$PROJECT_ID/topics/$TOPIC" > /dev/null
fi

echo "4/6 Service accounts"
gcloud iam service-accounts describe "$GUARD_SA" > /dev/null 2>&1 \
    || gcloud iam service-accounts create "$GUARD" --display-name "Budget guard: closes the Cloud Run services over budget"
gcloud iam service-accounts describe "$BUILD_SA" > /dev/null 2>&1 \
    || gcloud iam service-accounts create "$BUILD_SA_NAME" --display-name "Builds the budget-guard function"

echo "5/6 Permissions"
# The guard may change the guarded services and nothing else (updating a service means acting as its account)
for service in "${SERVICES[@]}"; do
    if gcloud run services describe "$service" --region "$REGION" > /dev/null 2>&1; then
        gcloud run services add-iam-policy-binding "$service" --region "$REGION" \
            --member "serviceAccount:$GUARD_SA" --role roles/run.developer > /dev/null
    else
        echo "  Service '$service' not found: deploy it first (CD), then run this script again" >&2
    fi
done
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --member "serviceAccount:$GUARD_SA" \
    --role roles/iam.serviceAccountUser > /dev/null
# Cloud Run checks that whoever updates a service can read its image, even when the image does not change
gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" --location "$REGION" \
    --member "serviceAccount:$GUARD_SA" --role roles/artifactregistry.reader > /dev/null
# The build of the function reads its sources, writes its logs and pushes its image to the repository
for role in roles/logging.logWriter roles/storage.objectViewer; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$BUILD_SA" \
        --role "$role" --condition None > /dev/null
done
gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" --location "$REGION" \
    --member "serviceAccount:$BUILD_SA" --role roles/artifactregistry.writer > /dev/null

echo "6/6 Cloud Function '$GUARD' (a few minutes)"
services_list=$(IFS=,; echo "${SERVICES[*]}")
# Its image goes to the same repository, whose cleanup policy keeps a few versions only.
# Internal ingress: it is only ever called by Eventarc, which delivers the Pub/Sub messages.
gcloud functions deploy "$GUARD" --gen2 --region "$REGION" --runtime python313 --quiet \
    --source "$SOURCE_DIR" --entry-point handle \
    --trigger-topic "$TOPIC" --trigger-service-account "$GUARD_SA" --service-account "$GUARD_SA" \
    --build-service-account "projects/$PROJECT_ID/serviceAccounts/$BUILD_SA" \
    --docker-repository "projects/$PROJECT_ID/locations/$REGION/repositories/$REPOSITORY" \
    --memory 256Mi --max-instances 1 --timeout 300 --ingress-settings internal-only \
    --set-env-vars "^@^PROJECT_ID=$PROJECT_ID@REGION=$REGION@GUARDED_SERVICES=$services_list@CLOSE_AT=$CLOSE_AT" > /dev/null
# Eventarc calls the function as the guard account
gcloud functions add-invoker-policy-binding "$GUARD" --region "$REGION" --member "serviceAccount:$GUARD_SA" > /dev/null

limit=$(awk "BEGIN { printf \"%.2f\", $AMOUNT * $CLOSE_AT }")
cat << EOF

Done. Budget: $AMOUNT $CURRENCY per month; the services go offline at $limit $CURRENCY and come back next month.

Test it on staging only: this message says the budget is spent, and only the service in "services" is touched.
  gcloud pubsub topics publish $TOPIC --attribute services=meteoreading-staging \\
      --message '{"costAmount": $AMOUNT, "budgetAmount": $AMOUNT, "currencyCode": "$CURRENCY"}'
Then put it back online:
  gcloud pubsub topics publish $TOPIC --attribute services=meteoreading-staging \\
      --message '{"costAmount": 0, "budgetAmount": $AMOUNT, "currencyCode": "$CURRENCY"}'
The function's log: gcloud functions logs read $GUARD --region $REGION
EOF
