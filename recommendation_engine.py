"""Deterministic tool affordability and fit classification."""
from __future__ import annotations

import csv
import math
import re

FIT_STRONG = "Strong Fit"
FIT_PARTIAL = "Partial Fit"
FIT_RESEARCH = "Research Lead"


def load_plan_catalog(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    required = {
        "tool_name", "plan_name", "monthly_price", "minimum_known_monthly_cost",
        "base_subscription_required", "base_subscription_monthly_cost", "add_on_cost",
        "per_user_cost", "per_location_cost", "per_channel_cost", "included_usage", "usage_unit",
        "usage_overage_cost", "capabilities", "excluded_capabilities",
        "included_users", "evidence_url", "evidence_status", "verified_date", "limitations",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Plan catalog is missing required fields.")
    for row in rows:
        key = (row["tool_name"].strip(), row["plan_name"].strip())
        if not all(key) or key in seen:
            raise ValueError("Every plan must have a unique tool and plan name.")
        seen.add(key)
    return rows


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _tags(value):
    return {item.strip() for item in str(value or "").split(";") if item.strip()}


def required_capabilities(job, challenge="", goal=""):
    text = f"{challenge} {goal}".casefold()
    if job == "Scheduling":
        if re.search(r"\b(crew|dispatch|route|field team|field staff)\b", text):
            return ["crew_dispatch"]
        if re.search(r"\b(double[- ]book(?:ing|ed)?|appointment conflicts?|booking conflicts?)\b", text) and not re.search(r"\b(staff|employees?|workers?|shifts?)\b", text):
            return ["appointment_booking"]
        if re.search(r"\b(forecast|forecasting|predict(?:ion|ive)?)\b", text):
            caps = ["employee_scheduling", "advanced_labor_forecasting"]
            if re.search(r"\b(labor cost|labor costs|labor spend|overtime|labor planning)\b", text):
                caps.append("labor_cost_management")
            if re.search(r"\b(swap|swaps|shift pool|shift trade)\b", text):
                caps.append("shift_swap_management")
            return caps
        if re.search(r"\b(labor cost|labor costs|labor spend|overtime|labor planning)\b", text):
            caps = ["employee_scheduling", "labor_cost_management"]
            if re.search(r"\b(swap|swaps|shift pool|shift trade)\b", text):
                caps.append("shift_swap_management")
            if re.search(r"\b(forecast|forecasting|predict(?:ion|ive)?)\b", text):
                caps.append("advanced_labor_forecasting")
            return caps
        caps = ["employee_scheduling"]
        if re.search(r"\b(swap|swaps|shift pool|shift trade)\b", text):
            caps.append("shift_swap_management")
        return caps
    if job == "Bookings":
        if re.search(r"\b(crew|dispatch|route)\b", text):
            return ["crew_dispatch"]
        return ["appointment_booking"]
    if job == "Customer Questions":
        if re.search(r"\b(draft|write|compose)\b.*\b(repl(?:y|ies)|response|responses|message|messages)\b", text):
            return ["draft_customer_responses"]
        if re.search(r"\b(call|calls|phone|missed call|answering while|incoming call)\b", text):
            caps = ["inbound_call_answering"]
            if re.search(r"\b(question|questions|faq|common customer)\b", text):
                caps.append("customer_question_answering")
            return caps
        return ["customer_question_answering"]
    if job == "Reviews":
        return ["review_response_drafting"]
    if job == "Social media":
        return ["social_content_creation"]
    if job == "Marketing":
        if re.search(r"\b(foot traffic|increase traffic|more customers|attract customers|customer acquisition|bring in customers)\b", text):
            return ["local_customer_acquisition"]
        return ["marketing_content_generation"]
    if job == "Accounting":
        return ["accounting_document_processing"]
    if job == "Estimating":
        return ["property_measurement_takeoff"]
    if job == "Landscape Design":
        return ["landscape_concept_design"]
    if job == "Daily operations":
        if re.search(r"\b(inventory|food cost|invoice|forecast|production)\b", text):
            caps = []
            if re.search(r"\b(inventory|food cost)\b", text): caps.append("inventory_management")
            if re.search(r"\b(invoice|invoices)\b", text): caps.append("invoice_processing")
            if re.search(r"\b(forecast|forecasting)\b", text): caps.append("food_demand_forecasting")
            if re.search(r"\b(production|prep)\b", text): caps.append("kitchen_production_planning")
            return caps
        return ["general_business_assistance"]
    return []


CAPABILITY_WORDS = {
    "employee_scheduling": ("employee scheduling", "staff scheduling", "shift schedule", "schedule shifts"),
    "shift_swap_management": ("shift swap", "shift swaps", "shift trade", "shift trades"),
    "labor_cost_management": ("labor cost", "labor costs", "overtime", "labor planning"),
    "advanced_labor_forecasting": ("forecast", "forecasting", "predictive labor"),
    "appointment_booking": ("appointment booking", "book appointments", "double booking"),
    "inbound_call_answering": ("answer calls", "missed calls", "phone calls", "call answering"),
    "customer_question_answering": ("answer customer questions", "customer faqs", "answer inquiries"),
    "draft_customer_responses": ("draft customer responses", "write customer replies"),
    "social_content_creation": ("social posts", "social content", "instagram posts", "facebook posts"),
    "marketing_content_generation": ("marketing content", "marketing copy", "promotional content"),
}


def split_requirements(job, challenge="", goal="", explicit_preferred=None):
    """Separate explicitly preferred features from essential workflow needs."""
    text = f"{challenge} {goal}".casefold()
    essential = required_capabilities(job, challenge, goal)
    preferred = list(explicit_preferred or [])
    for capability in list(essential):
        phrases = CAPABILITY_WORDS.get(capability, (capability.replace("_", " "),))
        for phrase in phrases:
            for match in re.finditer(re.escape(phrase), text):
                prefix = text[max(0, match.start() - 55):match.start()]
                if re.search(r"\b(prefer|would like|nice to have|optional|not essential)\b[^.;!?]*$", prefix):
                    essential.remove(capability)
                    if capability not in preferred:
                        preferred.append(capability)
                    break
            if capability not in essential:
                break
    return essential, preferred


def _required_seats(req):
    value = _number(req.get("seats"))
    return int(value) if value is not None and value >= 1 else None


def _plan_cost(plan, req):
    fixed = _number(plan.get("monthly_price"))
    base = _number(plan.get("base_subscription_monthly_cost"))
    addon = _number(plan.get("add_on_cost"))
    per_user = _number(plan.get("per_user_cost"))
    per_location = _number(plan.get("per_location_cost"))
    per_channel = _number(plan.get("per_channel_cost"))
    minimum = _number(plan.get("minimum_known_monthly_cost"))
    usage_rate = _number(plan.get("usage_overage_cost"))
    total = 0.0
    unknown = []
    parts = []
    basis = str(plan.get("price_basis", "fixed")).strip().casefold()

    if basis.startswith("per_channel") and per_channel is not None:
        channels = _number(req.get("channels"))
        if channels is None:
            unknown.append("number of paid channels is unknown")
        else:
            total += per_channel * channels
            parts.append(f"${per_channel:,.2f} × {channels:g} channels")
    elif basis.startswith("per_user") and per_user is not None:
        seats = _required_seats(req)
        if seats is None:
            unknown.append("required paid seat count is unknown")
        else:
            total += per_user * seats
            parts.append(f"${per_user:,.2f} × {seats} seats")
    elif basis.startswith("per_location") and per_location is not None:
        locations = _number(req.get("locations")) or 1
        total += per_location * locations
        parts.append(f"${per_location:,.2f} × {locations:g} locations")
    elif fixed is not None:
        total += fixed
        parts.append(f"${fixed:,.2f} plan price")
    elif per_user is not None:
        seats = _required_seats(req)
        if seats is None:
            unknown.append("required paid seat count is unknown")
        else:
            total += per_user * seats
            parts.append(f"${per_user:,.2f} × {seats} seats")
    elif per_location is not None:
        locations = _number(req.get("locations")) or 1
        total += per_location * locations
        parts.append(f"${per_location:,.2f} × {locations:g} locations")
    elif basis in {"quote", "unknown"}:
        unknown.append("plan price is not published")
    else:
        unknown.append("monthly plan price is not verified")

    if str(plan.get("base_subscription_required", "false")).casefold() == "true":
        if base is None:
            unknown.append("required base subscription cost is unknown")
        else:
            total += base
            parts.append(f"${base:,.2f} required base subscription")
    if addon is not None:
        total += addon
        parts.append(f"${addon:,.2f} add-on")
    included = _number(plan.get("included_usage"))
    usage = _number(req.get("monthly_usage"))
    usage_unit = str(req.get("usage_unit") or "").casefold()
    plan_unit = str(plan.get("usage_unit") or "").casefold()
    aliases = {"callers": "unique_customers", "customers": "unique_customers",
               "outcomes": "outcomes", "resolved_questions": "outcomes",
               "conversations": "conversations", "inquiries": "conversations"}
    if usage is not None and plan_unit and usage_unit and aliases.get(usage_unit, usage_unit) != aliases.get(plan_unit, plan_unit):
        unknown.append("requested usage unit does not match the verified plan limit")
    elif usage is not None and included is not None and usage > included:
        if usage_rate is None:
            unknown.append("usage exceeds included volume and overage pricing is unknown")
        else:
            extra = (usage - included) * usage_rate
            total += extra
            parts.append(f"${extra:,.2f} estimated usage overage")
    elif included is None and plan_unit:
        unknown.append("usage limit is not verified")
    elif usage is None and included is not None and plan_unit:
        unknown.append("required monthly usage is unstated, so plan capacity is unverified")

    term = (str(plan.get("billing_term", "")) + " " + basis).casefold()
    if req.get("monthly_billing_only") and any(token in term for token in ("annual", "12-month", "12 month", "commitment")):
        return {"status": "ineligible", "cost": None, "lower_bound": minimum or total,
                "reason": "plan billing term conflicts with the monthly-only requirement"}
    employees = _number(req.get("employees"))
    employee_limit = _number(plan.get("included_employees"))
    if employees is not None and employee_limit is not None and employees > employee_limit:
        return {"status": "ineligible", "cost": None, "lower_bound": minimum or total,
                "reason": f"plan limit is {employee_limit:g} employees; profile has {employees:g}"}
    seats = _number(req.get("seats"))
    user_limit = _number(plan.get("included_users"))
    if seats is not None and user_limit is not None and seats > user_limit:
        return {"status": "ineligible", "cost": None, "lower_bound": minimum or total,
                "reason": f"plan limit is {user_limit:g} users; profile requires {seats:g}"}
    locations = _number(req.get("locations")) or 1
    location_limit = _number(plan.get("included_locations"))
    if location_limit is not None and locations > location_limit:
        return {"status": "ineligible", "cost": None, "lower_bound": minimum or total,
                "reason": f"plan limit is {location_limit:g} locations; profile has {locations:g}"}

    known_floor = max(minimum or 0.0, total)
    cap = _number(req.get("budget"))
    if cap is not None and known_floor > cap:
        return {"status": "ineligible", "cost": None, "lower_bound": known_floor,
                "reason": "; ".join(parts + [f"known minimum ${known_floor:,.2f} exceeds ${cap:,.2f} budget"])}
    if unknown:
        return {"status": "unknown", "cost": None, "lower_bound": known_floor or minimum,
                "reason": "; ".join(parts + unknown)}
    if cap is not None and total > cap:
        return {"status": "ineligible", "cost": total, "lower_bound": total,
                "reason": f"required plan total ${total:,.2f} exceeds ${cap:,.2f} budget"}
    return {"status": "confirmed", "cost": total, "lower_bound": total,
            "reason": "; ".join(parts) or "verified free plan"}


def _coverage(required, capabilities):
    """Only exact, plan-documented capabilities count as verified coverage."""
    return set(required) & set(capabilities)


def _classify_plan(tool, plan, req, required, preferred):
    caps = _tags(plan.get("capabilities"))
    excluded = _tags(plan.get("excluded_capabilities"))
    exact = _coverage(required, caps)
    missing = set(required) - exact
    cost = _plan_cost(plan, req)
    evidence = str(plan.get("evidence_status", "unknown"))
    documented = evidence in {"official_verified", "catalog_source"}
    record = {
        **cost,
        "plan": plan["plan_name"],
        "capabilities": sorted(caps),
        "excluded_capabilities": sorted(excluded),
        "missing_capabilities": sorted(missing),
        "covered_capabilities": sorted(exact),
        "limitations": plan.get("limitations", ""),
        "evidence_status": evidence,
        "evidence_url": plan.get("evidence_url", ""),
        "verified_date": plan.get("verified_date", ""),
        "billing_term": plan.get("billing_term", ""),
        "required_capabilities": list(required),
        "preferred_capabilities": list(preferred),
    }

    if not required:
        record.update(fit_classification=FIT_RESEARCH,
                      fit_reason="The customer's required workflow could not be identified.")
    elif cost["status"] == "ineligible":
        record.update(fit_classification="Ineligible",
                      fit_reason="The known required plan cost or limit exceeds the customer's requirement.")
    elif cost["status"] == "unknown" or not documented:
        if exact:
            record.update(fit_classification=FIT_RESEARCH,
                          fit_reason="A required capability is documented, but plan cost or evidence is unverified.")
        else:
            record.update(fit_classification="Ineligible",
                          fit_reason="No documented capability supports the stated workflow.")
    elif cost["status"] == "confirmed" and not missing:
        record.update(fit_classification=FIT_STRONG,
                      fit_reason="Required capabilities and the selected plan's total cost are documented within budget.")
    elif cost["status"] == "confirmed" and exact:
        unknown_missing = missing - excluded
        if not unknown_missing:
            record.update(fit_classification=FIT_PARTIAL,
                          fit_reason="The affordable plan supports part of the workflow; it does not include: "
                          + ", ".join(sorted(missing)) + ".")
        else:
            record.update(fit_classification=FIT_RESEARCH,
                          fit_reason="Some needed capabilities are not established by the plan evidence: "
                          + ", ".join(sorted(unknown_missing)) + ".")
    else:
        record.update(fit_classification="Ineligible",
                      fit_reason="No documented capability supports the stated workflow.")
    record["reasons"] = [record["fit_reason"], f"Selected plan: {record['plan']}.", cost["reason"]]
    return record


def _fit_order(item):
    rank = {FIT_STRONG: 0, FIT_PARTIAL: 1, FIT_RESEARCH: 2}
    return (rank.get(item.get("fit_classification"), 3),
            -len(item.get("covered_capabilities", [])),
            -len(set(item.get("preferred_capabilities", [])) & set(item.get("capabilities", []))),
            item.get("cost") if item.get("cost") is not None else float("inf"),
            str(item.get("tool_name", "")).casefold())


def evaluate_tool(tool, plans, req, required=None, preferred=None):
    """Choose the best affordable plan and assign a plain-language fit class."""
    name = str(tool.get("tool_name", ""))
    required = list(required or [])
    preferred = list(preferred or [])
    candidates = [plan for plan in plans if plan["tool_name"] == name]
    if not candidates:
        return {"status": "unknown", "fit_classification": FIT_RESEARCH,
                "plan": "No structured plan record", "cost": None,
                "lower_bound": None, "capabilities": [], "covered_capabilities": [],
                "required_capabilities": required,
                "fit_reason": "Plan pricing and capability evidence are unavailable.",
                "reason": "No plan-level evidence in tool_plans.csv"}
    evaluated = [_classify_plan(tool, plan, req, required, preferred) for plan in candidates]
    eligible = [item for item in evaluated if item["fit_classification"] in {FIT_STRONG, FIT_PARTIAL}]
    if eligible:
        eligible.sort(key=lambda item: _fit_order({**item, "tool_name": name}))
        return eligible[0]
    leads = [item for item in evaluated if item["fit_classification"] == FIT_RESEARCH]
    if leads:
        # Known over-budget plans are not research leads; only genuinely unknown plans remain.
        leads.sort(key=lambda item: (-len(item.get("covered_capabilities", [])),
                                     item["lower_bound"] if item["lower_bound"] is not None else float("inf")))
        return leads[0]
    evaluated.sort(key=lambda item: (item["lower_bound"] if item["lower_bound"] is not None else float("inf"),
                                     -len(item.get("covered_capabilities", []))))
    return evaluated[0]


def score_tools(data, job, cap, comfort, requirements, plans):
    """Return only affordable Strong Fit and Partial Fit products."""
    req = dict(requirements or {})
    req["budget"] = cap
    required, preferred = split_requirements(job, req.get("challenge", ""), req.get("goal", ""),
                                             req.get("preferred_capabilities"))
    preference_text = f"{req.get('challenge', '')} {req.get('goal', '')}".casefold()
    if re.search(r"\b(prefer|would like|nice to have|optional)\b.*\b(free|prefer free)\b", preference_text):
        req["prefer_free"] = True
    results = []
    for _, series in data.iterrows():
        tool = series.to_dict()
        plan = evaluate_tool(tool, plans, req, required, preferred)
        if plan.get("fit_classification") not in {FIT_STRONG, FIT_PARTIAL}:
            continue
        item = {**plan, "tool": series, "eligibility": plan["fit_classification"],
                "pricing": plan, "reasons": plan.get("reasons", [])}
        results.append(item)
    results.sort(key=lambda item: _fit_order({**item, "tool_name": item["tool"]["tool_name"]}))
    return results


def research_leads(data, job, cap, comfort, requirements, plans):
    req = dict(requirements or {})
    req["budget"] = cap
    required, preferred = split_requirements(job, req.get("challenge", ""), req.get("goal", ""),
                                             req.get("preferred_capabilities"))
    output = []
    for _, series in data.iterrows():
        tool = series.to_dict()
        plan = evaluate_tool(tool, plans, req, required, preferred)
        if plan.get("fit_classification") != FIT_RESEARCH or plan.get("status") == "ineligible":
            continue
        if not plan.get("covered_capabilities"):
            continue
        output.append({**plan, "tool": series, "eligibility": FIT_RESEARCH,
                       "pricing": plan, "reasons": plan.get("reasons", [])})
    output.sort(key=lambda item: (-len(item.get("covered_capabilities", [])),
                                  item.get("lower_bound") if item.get("lower_bound") is not None else float("inf"),
                                  str(item["tool"]["tool_name"]).casefold()))
    return output
