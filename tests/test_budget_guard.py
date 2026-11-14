"""Decision logic of the budget guard (deploy/budget_guard): when to close and reopen the services."""

import base64
import json
from types import SimpleNamespace

import pytest

from deploy.budget_guard import main as guard

# Shape of a Cloud Billing budget notification (data of the Pub/Sub message)
NOTIFICATION = {
    "budgetDisplayName": "meteoreading monthly cap",
    "alertThresholdExceeded": 0.5,
    "costAmount": 2.4,
    "costIntervalStart": "2026-11-01T07:00:00Z",
    "budgetAmount": 4.3,
    "budgetAmountType": "SPECIFIED_AMOUNT",
    "currencyCode": "EUR",
}


def test_parses_a_notification():
    budget = guard.Budget.from_notification(json.dumps(NOTIFICATION).encode())
    assert budget == guard.Budget(cost=2.4, amount=4.3, currency="EUR", period_start="2026-11-01T07:00:00Z")


@pytest.mark.parametrize(("cost", "over"), [(0.0, False), (2.14, False), (2.15, True), (9.0, True)])
def test_over_budget_from_half_the_budget(cost, over):
    assert guard.Budget(cost, 4.3, "EUR", "").over(0.5) is over


def test_no_budget_amount_never_closes():
    assert not guard.Budget(10.0, 0.0, "EUR", "").over(0.5)


@pytest.mark.parametrize(
    ("ingress", "labels", "over", "action"),
    [
        (guard.OPEN, {}, True, "close"),
        (guard.OPEN, {}, False, None),
        (guard.CLOSED, {guard.LABEL: "closed"}, True, None),  # already closed
        (guard.CLOSED, {guard.LABEL: "closed"}, False, "open"),  # new month, or budget raised
        (guard.CLOSED, {}, False, None),  # made internal by hand: left alone
        (guard.CLOSED, {}, True, None),
    ],
)
def test_decide(ingress, labels, over, action):
    assert guard.decide(ingress, labels, over) == action


def test_targets_are_limited_to_the_guarded_services():
    guarded = ["meteoreading-staging", "meteoreading"]
    assert guard.targets(guarded, None) == guarded
    assert guard.targets(guarded, "meteoreading-staging") == ["meteoreading-staging"]
    assert guard.targets(guarded, "meteoreading-staging, other-service") == ["meteoreading-staging"]


def event(body, **attributes):
    data = base64.b64encode(json.dumps(body).encode()).decode()
    return SimpleNamespace(data={"message": {"data": data, "attributes": attributes}})


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setenv("PROJECT_ID", "project")
    monkeypatch.setenv("REGION", "europe-west1")
    monkeypatch.setenv("GUARDED_SERVICES", "meteoreading-staging,meteoreading")
    monkeypatch.delenv("CLOSE_AT", raising=False)
    monkeypatch.setattr(guard, "services_client", lambda: None)
    monkeypatch.setattr(guard, "apply", lambda client, name, over: seen.append((name, over)))
    return seen


def test_handle_checks_every_guarded_service(calls):
    guard.handle(event(NOTIFICATION))
    assert calls == [("meteoreading-staging", True), ("meteoreading", True)]


def test_handle_test_message_on_staging_only(calls):
    guard.handle(event({**NOTIFICATION, "costAmount": 0}, services="meteoreading-staging"))
    assert calls == [("meteoreading-staging", False)]


def test_handle_keeps_going_when_a_service_fails(monkeypatch, calls):
    def flaky(client, name, over):
        if name == "meteoreading-staging":
            raise PermissionError("denied")
        calls.append((name, over))

    monkeypatch.setattr(guard, "apply", flaky)
    with pytest.raises(RuntimeError, match="meteoreading-staging"):
        guard.handle(event(NOTIFICATION))
    assert calls == [("meteoreading", True)]
