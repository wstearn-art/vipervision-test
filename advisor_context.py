"""Shared display evidence and calculator state; never changes catalog eligibility."""
import hashlib
import json
import math
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


ROI_WIDGETS = {
    "choice": "roi_tool_choice", "weekly_hours": "roi_hours", "hourly_value": "roi_hourly",
    "time_reduction_percent": "roi_reduction", "custom_cost": "roi_custom_cost",
    "custom_cost_confirmed": "roi_custom_confirmed",
}
ROI_DEFAULTS = {"choice": None, "weekly_hours": 5.0, "hourly_value": 25.0,
                "time_reduction_percent": 30, "custom_cost": 0.0, "custom_cost_confirmed": False}
ROI_LIMITS = {"weekly_hours": 168, "hourly_value": 10000,
              "time_reduction_percent": 100, "custom_cost": 100000}


def _valid_roi_input(field, value):
    if field == "choice":
        return value is None or isinstance(value, str)
    if field == "custom_cost_confirmed":
        return type(value) is bool
    return (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= ROI_LIMITS[field]
            and (field != "time_reduction_percent" or type(value) is int))


def roi_state(session):
    """Durable scenario inputs, independent of Streamlit's disposable widget keys."""
    if "vee_roi" not in session:
        saved = session.get("vee_roi_snapshot") or {}
        values = dict(ROI_DEFAULTS)
        for field, default in values.items():
            previous = session.get(ROI_WIDGETS[field], saved.get(field, default))
            if _valid_roi_input(field, previous):
                values[field] = previous
        values["choice"] = session.get("roi_tool_choice", saved.get("name", values["choice"]))
        legacy_cost_key = saved.get("cost_widget_key")
        if legacy_cost_key:
            cost = session.get(legacy_cost_key, saved.get("monthly_cost"))
            if _valid_roi_input("custom_cost", cost):
                values["custom_cost"] = cost
                values["custom_cost_confirmed"] = bool(session.get(legacy_cost_key + "_confirmed", cost is not None))
        session["vee_roi"] = {**values, "signature": None, "approved_signature": None,
                              "generation": 0, "rendered": bool(saved)}
        # Legacy approvals did not bind the numeric inputs and cannot be trusted.
        session.pop("vee_roi_scope", None)
    return session["vee_roi"]


def set_roi_input(session, field, value):
    if field not in ROI_WIDGETS or not _valid_roi_input(field, value):
        raise ValueError("Invalid ROI input")
    model = roi_state(session)
    if model[field] != value:
        model[field] = value
        model["signature"] = model["approved_signature"] = None
        if field == "custom_cost":
            model["custom_cost_confirmed"] = False


def save_roi_widget(session, field):
    """One callback for every calculator input; runs before the Streamlit rerun."""
    set_roi_input(session, field, session[ROI_WIDGETS[field]])


def restore_roi_widgets(session):
    model = roi_state(session)
    for field, key in ROI_WIDGETS.items():
        session[key] = model[field]


def confirm_roi(session, signature, widget_key=None):
    model = roi_state(session)
    if signature is not None and model["signature"] == signature and (widget_key is None or session.get(widget_key)):
        model["approved_signature"] = signature


def roi_calculation(session, candidates, profile):
    """Resolve current eligibility, exact approval and the single Python calculation."""
    model = roi_state(session)
    choices = [c["name"] for c in candidates]
    if model["choice"] not in ["Enter my own cost"] + choices:
        set_roi_input(session, "choice", choices[0] if choices else "Enter my own cost")
    selected = next((c for c in candidates if c["name"] == model["choice"]), None)
    if selected:
        selected = dict(selected)
    else:
        selected = {"name": "Enter my own cost", "plan": "Owner's cost scenario",
                    "covered_capabilities": profile.get("essential_capabilities", []),
                    "unresolved_requirements": profile.get("unresolved_requirements", []),
                    "fit_classification": "Owner scenario; coverage is not verified"}
    catalog_known = selected.get("status") == "confirmed" and selected.get("cost") is not None
    cost = selected["cost"] if catalog_known else model["custom_cost"]
    known = catalog_known or model["custom_cost_confirmed"]
    cost_source = "catalog" if catalog_known else "owner_scenario" if known else "unknown"
    # A confirmed hypothetical amount never changes catalog evidence or fit.
    selected["cost_source"] = cost_source
    selected["profile_revision"] = profile.get("revision", 0)
    # Metadata such as an owner's name or the revision counter alone does not change eligibility.
    eligibility_fields = ("business", "employees", "budget_amount", "seats", "locations", "channels",
                          "monthly_usage", "usage_unit", "monthly_billing_only", "prefer_free",
                          "requirements", "essential_capabilities", "optional_capabilities", "unresolved_requirements")
    plan_fields = ("name", "plan", "status", "cost", "billing_term", "fit_classification",
                   "covered_capabilities", "missing_capabilities", "unresolved_requirements")
    signature = json.dumps({"plan": {k: selected.get(k) for k in plan_fields},
        "profile": {k: profile.get(k) for k in eligibility_fields},
        "inputs": {k: model[k] for k in ("weekly_hours", "hourly_value", "time_reduction_percent")},
        "cost": cost, "cost_known": known, "cost_source": cost_source}, sort_keys=True, separators=(",", ":"))
    if model["signature"] != signature:
        model["signature"] = signature
        model["approved_signature"] = None
        model["generation"] += 1
    pending = model["approved_signature"] != signature
    return roi_snapshot(selected, model["weekly_hours"], model["hourly_value"],
                        model["time_reduction_percent"], cost, known, pending)


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
            "needs_assumption_review": needs_review, "assumptions_confirmed": not needs_review,
            "illustrative_not_measured": True, "available": True,
            "excluded_costs": "onboarding, taxes, unmodeled usage, extra seats and other costs"}
