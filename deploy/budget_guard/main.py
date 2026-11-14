"""Budget guard: takes the Cloud Run services offline when the monthly budget is nearly spent.

A Google Cloud budget only sends alerts, it never stops anything. Cloud Billing also posts the state of the
budget to a Pub/Sub topic several times a day: this Cloud Function reads each of those messages and, for every
guarded service:

- closes it when the cost of the month reaches CLOSE_AT x the budget: its ingress becomes "internal", so it no
  longer answers on the Internet, starts no instance and costs nothing more;
- opens it again once the cost is back under that level: when the next month starts (the cost of the period
  restarts from zero) or after the budget was raised.

It only reopens the services it closed itself, which carry the label budget-guard=closed: a service made
internal by hand stays internal.

Deployed by deploy/gcp-budget.sh, with these environment variables:
  PROJECT_ID, REGION   where the services are
  GUARDED_SERVICES     comma-separated names of the Cloud Run services to guard
  CLOSE_AT             fraction of the budget at which they close (default 0.5: billing data can be hours
                       late, and the other half of the budget absorbs what is spent in the meantime)

A message can carry a "services" attribute to act on some of the guarded services only (to test on staging).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass

try:
    import functions_framework
except ImportError:  # the unit tests only need the decision logic, not the Cloud Functions runtime
    functions_framework = None

LABEL = "budget-guard"
OPEN = "INGRESS_TRAFFIC_ALL"
CLOSED = "INGRESS_TRAFFIC_INTERNAL_ONLY"


@dataclass(frozen=True)
class Budget:
    """What a budget notification says: the cost of the current period, against the budget."""

    cost: float
    amount: float
    currency: str
    period_start: str

    @classmethod
    def from_notification(cls, data: bytes) -> Budget:
        body = json.loads(data)
        return cls(
            cost=float(body["costAmount"]),
            amount=float(body["budgetAmount"]),
            currency=body.get("currencyCode", ""),
            period_start=body.get("costIntervalStart", ""),
        )

    def over(self, close_at: float) -> bool:
        return self.amount > 0 and self.cost >= close_at * self.amount


def decide(ingress: str, labels: dict[str, str], over_budget: bool) -> str | None:
    """What to do with a service, given its ingress and labels: "close", "open" or None (nothing)."""
    if over_budget and ingress != CLOSED:
        return "close"
    if not over_budget and labels.get(LABEL) == "closed":
        return "open"
    return None


def targets(guarded: list[str], requested: str | None) -> list[str]:
    """The services to check: all the guarded ones, or those named in the message (never any other)."""
    if not requested:
        return guarded
    names = {name.strip() for name in requested.split(",")}
    return [name for name in guarded if name in names]


def log(severity: str, message: str, **fields) -> None:
    # One JSON object per line: Cloud Logging reads the severity and keeps the other fields
    print(json.dumps({"severity": severity, "message": message, **fields}), flush=True)


def services_client():
    from google.cloud import run_v2  # imported here: the tests run without the Google Cloud libraries

    return run_v2.ServicesClient()


def apply(client, name: str, over_budget: bool) -> str | None:
    """Closes or reopens one service if needed; returns what was done."""
    from google.cloud import run_v2

    service = client.get_service(name=f"projects/{os.environ['PROJECT_ID']}/locations/{os.environ['REGION']}/services/{name}")
    action = decide(run_v2.IngressTraffic(service.ingress).name, dict(service.labels), over_budget)
    if action == "close":
        service.ingress = run_v2.IngressTraffic.INGRESS_TRAFFIC_INTERNAL_ONLY
        service.labels[LABEL] = "closed"
    elif action == "open":
        service.ingress = run_v2.IngressTraffic.INGRESS_TRAFFIC_ALL
        del service.labels[LABEL]
    if action:
        # Ingress and labels belong to the service, not to the revision: no new revision, no restart. The request
        # is validated right away; the function does not wait for the rollout (that would need a project-wide
        # permission): every notification reads the actual state again, so a change that did not apply is
        # simply made again.
        client.update_service(service=service)
    return action


def handle(cloud_event) -> None:
    message = cloud_event.data["message"]
    budget = Budget.from_notification(base64.b64decode(message["data"]))
    close_at = float(os.environ.get("CLOSE_AT", "0.5"))
    over_budget = budget.over(close_at)
    guarded = [name.strip() for name in os.environ["GUARDED_SERVICES"].split(",") if name.strip()]
    names = targets(guarded, (message.get("attributes") or {}).get("services"))
    state = {"cost": budget.cost, "budget": budget.amount, "currency": budget.currency,
             "period_start": budget.period_start, "close_at": close_at, "over_budget": over_budget}

    client, failed = services_client(), []
    for name in names:
        try:
            action = apply(client, name, over_budget)
        except Exception as error:  # one failing service must not keep the others open
            log("ERROR", f"{name}: {error}", service=name, **state)
            failed.append(name)
            continue
        if action:
            verb = "closed (internal only)" if action == "close" else "reopened to the Internet"
            log("WARNING" if action == "close" else "NOTICE", f"{name} {verb}: cost {budget.cost:.2f} "
                f"of {budget.amount:.2f} {budget.currency} this month", service=name, action=action, **state)
    if not failed:
        log("INFO", f"Cost {budget.cost:.2f} of {budget.amount:.2f} {budget.currency}, services checked: "
            f"{', '.join(names) or 'none'}", **state)
    else:
        # The execution is marked as failed; the next notification (minutes later) tries again
        raise RuntimeError(f"Could not update {', '.join(failed)}")


if functions_framework is not None:
    handle = functions_framework.cloud_event(handle)
