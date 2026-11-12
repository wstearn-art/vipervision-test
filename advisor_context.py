"""Shared display evidence and calculator state; never changes catalog eligibility."""
import hashlib
import json
from urllib.parse import urlparse


CAPABILITY_LABELS = {
    "crew_dispatch": "crew scheduling and job assignment",
    "personal_task_scheduling": "personal task scheduling",
    "employee_scheduling": "employee scheduling",
    "shift_swap_management": "shift swaps",
    "labor_cost_management": "labor cost management",
    "advanced_labor_forecasting": "advanced labor forecasting",
    "appointment_booking": "appointment booking",
    "customer_appointment_updates": "customer appointment reminders",
    "inbound_call_answering": "answering incoming calls",
    "customer_question_answering": "answering customer questions",
    "draft_customer_responses": "drafting customer replies",
    "review_response_drafting": "drafting review responses",
    "social_content_creation": "social media content",
    "marketing_content_generation": "marketing content",
    "local_customer_acquisition": "attracting local customers",
    "accounting_document_processing": "processing accounting documents",
    "property_measurement_takeoff": "property measurements and takeoffs",
    "landscape_concept_design": "landscape design concepts",
    "inventory_management": "inventory management",
    "invoice_processing": "invoice processing",
    "food_demand_forecasting": "food demand forecasting",
    "inventory_forecasting": "inventory forecasting",
    "kitchen_production_planning": "kitchen production planning",
    "general_business_assistance": "general business assistance",
    "generic_content_drafting": "drafting general content",
}


def capability_label(capability):
    return CAPABILITY_LABELS.get(capability, capability.replace("_", " "))


def display_text(text):
    """Translate IDs in engine explanations without changing machine identifiers."""
    for capability in sorted(CAPABILITY_LABELS, key=len, reverse=True):
        text = text.replace(capability, capability_label(capability))
    return text


def team_label(count):
    return f"{count} {'person' if count == 1 else 'people'}"


def monthly_cost_label(cost, billing_term=""):
    suffix = " monthly equivalent (annual billing)" if "annual" in str(billing_term).lower() else "/month"
    return f"${cost:,.2f}{suffix}"


def evidence_conflicts(plans):
    """Flag overlapping provider records with incompatible free-usage assertions.

    A shared domain is a review signal, not proof that products/plans are identical.
    Retain both records and their dates/sources rather than resolving the conflict.
    """
    groups = {}
    for plan in plans:
        domain = urlparse(plan.get("evidence_url", "")).netloc.lower().removeprefix("www.")
        if domain:
            groups.setdefault(domain, []).append(plan)
    result = {}
    for domain, rows in groups.items():
        free = [p for p in rows if str(p.get("monthly_price")) in ("0", "0.0")]
        if len({p["tool_name"] for p in free}) < 2:
            continue
        allowances = {(p.get("included_usage", ""), p.get("usage_unit", "")) for p in free}
        if len(allowances) < 2:
            continue
        names = sorted({p["tool_name"] for p in rows})
        note = ("Evidence needs reconciliation: " + " and ".join(names)
                + " are overlapping catalog records from " + domain + ". Their free AI allowance "
                "is stated differently (a numeric allowance versus an unspecified limit). "
                "The exact current free quota is not verified consistently; do not rely on a numeric "
                "quota before checking the provider. Paid monthly and annual-equivalent figures "
                "describe different billing terms, not interchangeable offers. Product/feature scope "
                "and regional terms need verification; these are not independent vendor alternatives.")
        for name in names:
            result[name] = note
    return result


def plan_evidence(name, plans):
    """Lossless relevant plan rows: catalog statements are not verified totals."""
    return [dict(p) for p in plans if p["tool_name"] == name]


def roi_candidate(entry, revision):
    pricing = entry.get("pricing") or entry
    return {
        "profile_revision": revision, "name": str(entry["tool"]["tool_name"]),
        "plan": pricing.get("plan", entry.get("plan", "")),
        "cost": pricing.get("cost") if pricing.get("status") == "confirmed" else None,
        "status": pricing.get("status", "unknown"),
        "billing_term": pricing.get("billing_term", ""),
        "fit_classification": entry.get("fit_classification"),
        "covered_capabilities": sorted(entry.get("covered_capabilities", [])),
        "missing_capabilities": sorted(entry.get("missing_capabilities", [])),
        "unresolved_requirements": list(entry.get("unresolved_requirements", [])),
    }


def review_roi_scope(session, selected):
    """Keep numeric inputs; require owner review when the selected scope changes."""
    fields = ("name", "plan", "covered_capabilities", "missing_capabilities", "unresolved_requirements")
    signature = json.dumps({key: selected.get(key) for key in fields}, sort_keys=True)
    previous = session.get("vee_roi_scope")
    if previous is None or previous["signature"] != signature:
        pending = previous is not None or bool(selected.get("missing_capabilities") or selected.get("unresolved_requirements"))
        session["vee_roi_scope"] = {"signature": signature, "needs_review": pending,
                                    "generation": (previous or {}).get("generation", 0) + 1}
    return session["vee_roi_scope"]


def roi_review_key(scope):
    return "roi_review_" + hashlib.sha256(scope["signature"].encode()).hexdigest()[:16] + "_" + str(scope["generation"])


def roi_snapshot(selected, hours, hourly_value, reduction, cost, cost_known, needs_review):
    """The one calculator formula shared by rendering and advisor context."""
    time_value = hours * hourly_value * 4.33
    savings = time_value * reduction / 100 if not needs_review else None
    return {**selected, "weekly_hours": hours, "hourly_value": hourly_value,
            "time_reduction_percent": reduction, "monthly_time_value": time_value,
            "monthly_savings": savings, "monthly_cost": cost if cost_known else None,
            "monthly_net": savings - cost if cost_known and savings is not None else None,
            "needs_assumption_review": needs_review, "illustrative_not_measured": True,
            "excluded_costs": "onboarding, taxes, unmodeled usage, extra seats and other costs"}
