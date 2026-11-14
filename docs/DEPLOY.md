# Deploying on Google Cloud Run

The dashboard runs on [Cloud Run](https://cloud.google.com/run), Google's serverless container platform:
HTTPS out of the box, scales to zero between visits. GitHub Actions builds and tests the image, then promotes
that same image to a staging service and, after approval, to production.

```mermaid
flowchart LR
    PR["Pull request"] --> CI["CI: lint, tests,<br>image with data,<br>smoke test"]
    M["Push to main /<br>Monday data refresh"] --> CI
    CI -->|"push the tested image"| GHCR["GitHub Container<br>Registry"]
    GHCR --> CD["CD"]
    CD -->|"1. automatic"| S["Cloud Run<br>meteoreading-staging"]
    CD -->|"2. after approval"| P["Cloud Run<br>meteoreading"]
```

## What is in the image

The image holds the dashboard **and its data**: during the build, the pipeline downloads the Météo-France files,
builds the DuckDB warehouse (~25 million rows), the figures and the results, and only the warehouse and the
results are kept in the final image (~300 MB of data). The container needs nothing else to start, and the
same image runs anywhere:

```bash
docker run -p 8080:8080 ghcr.io/elliotgalaor/meteoreading:latest
```

The data is downloaded again at most once a day (the CI passes the date as `DATA_VERSION`), and **every Monday**
a scheduled CI run rebuilds the image with the latest observations; CD then deploys it like any other image.

## Costs

For a portfolio with occasional visitors, this stays within Google Cloud's free tier:

- **Cloud Run** (tier 1 regions such as `europe-west1`): a monthly free allowance of CPU time, memory time and
  requests. The services scale to zero, so nothing runs between visits; a visit starts an instance in a few
  seconds (cold start).
- **Artifact Registry**: 0.5 GB of free storage, then a few cents per GB and per month. The repository keeps only
  the 5 most recent images, and they share most of their layers.

A billing account is still required by Google. Setting a **budget alert** (Billing > Budgets & alerts) is a good
habit.

## 1. Prepare Google Cloud (once)

You need a Google Cloud project with billing enabled. Then open **[Cloud Shell](https://shell.cloud.google.com)**
(the terminal icon at the top of the console: `gcloud` is installed and signed in) and run:

```bash
git clone https://github.com/ElliotGalaor/MeteoReading.git
bash MeteoReading/deploy/gcp-setup.sh <PROJECT_ID> europe-west1
```

The script ([deploy/gcp-setup.sh](../deploy/gcp-setup.sh)) can be run again safely. It creates:

| Resource | Purpose |
|---|---|
| Artifact Registry repository `meteoreading` | Cloud Run pulls images from Google's registry: CD copies the tested image there (same digest) |
| Service account `github-deployer` | What GitHub Actions acts as: deploys Cloud Run services, pushes to that repository, nothing else |
| Service account `meteoreading-run` | What the app runs as: no role at all, the app only calls public APIs |
| Workload Identity pool `github` + provider `github-oidc` | Lets GitHub Actions, for **this repository only**, exchange its OIDC token for short-lived Google credentials: **no service account key** to create, store or rotate |

It ends by printing the GitHub variables to set.

## 2. Configure GitHub (once)

1. **Repository variables** (**Settings** → **Secrets and variables** → **Actions** → **Variables**), from the
   output of the script: `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_ARTIFACT_REPOSITORY`,
   `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_DEPLOY_SERVICE_ACCOUNT`, `GCP_RUNTIME_SERVICE_ACCOUNT`. These are
   identifiers, not secrets: thanks to Workload Identity Federation, nothing secret is stored on GitHub.
2. **`staging` environment**: **Settings** → **Environments** → **New environment**, name `staging`.
3. **`production` environment**: the same, name `production`, then:
   - **Required reviewers**: tick it, add yourself, **Save protection rules** (leave **Prevent self-review**
     unticked if you are the only reviewer);
   - **Deployment branches and tags**: **Selected branches and tags**, add `main`.

The Cloud Run services are created by the first deploy: `meteoreading-staging` and `meteoreading`. To use other
names, add a `CLOUD_RUN_SERVICE` variable to an environment.

Without `GCP_PROJECT_ID`, the deploy jobs do nothing and their log says so (forks, before the setup).

## 3. Automatic updates

**The image is built once, then promoted from one environment to the next**:

1. **CI** (`.github/workflows/ci.yml`), on every pull request, every push to `main` and every Monday: `ruff`
   and `actionlint`, the tests on Python 3.12, 3.13 and 3.14, then the Docker image is built with fresh data and
   smoke-tested (health endpoint, row count of the warehouse, the dashboard rendered headless with Streamlit
   `AppTest`). Except for pull requests, that exact image is pushed to GitHub Container Registry, tagged
   `run-<CI run id>`, `sha-<commit>` (pushes) and `latest`.
2. **CD** (`.github/workflows/cd.yml`), after every successful CI run on `main`: resolves the image to its digest,
   copies it to Artifact Registry, deploys it to **staging**, checks that the service answers, then waits for a
   reviewer and deploys the same digest to **production**.

| GitHub environment | Cloud Run service | Deployed |
|---|---|---|
| `staging` | `meteoreading-staging` | automatically, after each successful CI run on `main` |
| `production` | `meteoreading` | after approval, with the image staging received |

The deploy itself lives in a reusable composite action:
[.github/actions/deploy-cloud-run](../.github/actions/deploy-cloud-run/action.yml). The services are public, scale
from 0 to 3 instances (2 GiB, 1 vCPU, CPU boost at start-up), and use session affinity with a long timeout,
because Streamlit keeps one WebSocket open per visitor.

### Approve a production deploy

After staging, the run shows **Review deployments** and GitHub notifies the reviewers. Open the staging URL (in the
summary of the staging job), then tick `production` → **Approve and deploy**. When a newer image arrives while one
is waiting, the older one is cancelled: only the newest image waits for approval.

### Redeploy or roll back

**Actions** → **CD** → **Run workflow**, with the tag of any published image (`sha-<commit>`, `run-<id>`,
`latest` or a version): it goes through the same steps, staging then production after approval. The tags are listed
on the [package page](https://github.com/ElliotGalaor/MeteoReading/pkgs/container/meteoreading).

## Limits

- **Cold start**: after a while without visits, the first page takes a few seconds to load.
- **Route weather** calls the free Open-Meteo and OpenStreetMap routing services from the server: fine for a
  portfolio, not for heavy traffic (their fair-use limits apply to the server's address).
- The data is a snapshot taken at build time, refreshed every Monday.

## Troubleshooting

- **The deploy jobs say "GCP_PROJECT_ID variable not set"**: the repository variables are missing (step 2).
- **`google-github-actions/auth` fails with "permission denied" or "unable to acquire impersonated credentials"**:
  `GCP_WORKLOAD_IDENTITY_PROVIDER` or `GCP_DEPLOY_SERVICE_ACCOUNT` is wrong, or the repository name given to the
  setup script is not this repository: run the script again with the right name.
- **"Permission 'artifactregistry.repositories.uploadArtifacts' denied"**: the region of `GCP_REGION` is not the
  one of the repository created by the script.
- **"Deploy to production" stays "Waiting"**: it waits for a reviewer, see
  [Approve a production deploy](#approve-a-production-deploy).
- **The service answers but a page shows an error**: **Cloud Run** → the service → **Logs** in the Google Cloud
  console.
