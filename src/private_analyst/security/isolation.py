from __future__ import annotations

from private_analyst.exceptions import IsolationError


def assert_same_org(resource_org: str, user_org: str) -> None:
    if resource_org != user_org:
        raise IsolationError("Cross-organization access is not allowed.")


def scoped_query_params(user_org: str) -> tuple[str, tuple[str]]:
    return "org_id = ?", (user_org,)
