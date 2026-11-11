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


# Interpret workflow wording once, before evaluating catalog plans.
CAPABILITY_PATTERNS = {
    "crew_dispatch": r"\b(?:crews? schedul\w*|schedul\w* crews?|coordinat\w* crews?|assign\w* (?:landscaping )?crews?|crew dispatch|dispatch|routing|field teams? schedul\w*)\b",
    "personal_task_scheduling": r"\b(?:personal (?:calendar|schedul\w*|task planning)|schedule tasks)\b",
    "employee_scheduling": r"\b(?:(?:employee|staff|shift|workforce)\s+(?:shift\s+)?schedul\w*|schedul\w*\s+(?:employees|staff|shifts)|timekeeping|timesheets?|time clocks?)\b",
    "shift_swap_management": r"\b(?:shift (?:swaps?|swapping|trades?|trading|pool)|swap shifts)\b",
    "labor_cost_management": r"\b(?:labou?r (?:costs?|spend|planning)|overtime)\b",
    "advanced_labor_forecasting": r"\b(?:(?:advanced |hourly |predictive )?labou?r forecast\w*|forecast\w* (?:labou?r|staffing))\b",
    "appointment_booking": r"\b(?:appointment bookings?|book appointments?|booking appointments?|double book\w*|appointment conflicts?|booking conflicts?)\b",
    "customer_appointment_updates": r"\b(?:(?:automated |customer |appointment )?appointment (?:reminders?|updates?)|customer reminders?|customer updates?)\b",
    "inbound_call_answering": r"\b(?:missed calls?|answer(?:ing)? (?:customer |incoming )?calls?|call answering|phone calls?|incoming calls?)\b",
    "customer_question_answering": r"\b(?:customer (?:questions?|faqs)|answer inquiries|faq)\b",
    "draft_customer_responses": r"\b(?:draft|write|compose)\w* (?:customer )?(?:replies|responses?|messages?)\b",
    "review_response_drafting": r"\b(?:reviews?|review responses?)\b",
    "social_content_creation": r"\b(?:social media|social posts?|social content|instagram|facebook|tiktok|reels?)\b",
    "marketing_content_generation": r"\b(?:marketing|marketing copy|promotional content|campaigns?|advertising)\b",
    "local_customer_acquisition": r"\b(?:foot traffic|more (?:local )?(?:customers|visitors)|attract\w* (?:(?:more|new|local) )*customers|customer acquisition|bring in customers|increase traffic)\b",
    "accounting_document_processing": r"\b(?:accounting|bookkeeping|receipts?|accounting documents?)\b",
    "property_measurement_takeoff": r"\b(?:property measurements?|takeoffs?|estimating|estimates?)\b",
    "landscape_concept_design": r"\b(?:landscape design|landscaping designs?|design concepts?)\b",
    "inventory_management": r"\b(?:inventory|food cost|track materials|tracking materials)\b",
    "invoice_processing": r"\b(?:invoices?|invoice processing)\b",
    "food_demand_forecasting": r"\b(?:(?:food|restaurant|kitchen|meal) (?:demand )?forecast\w*|forecast\w* (?:food|meal) demand)\b",
    "inventory_forecasting": r"\b(?:inventory forecast\w*|forecast\w* inventory)\b",
    "kitchen_production_planning": r"\b(?:kitchen production|production planning|prep planning)\b",
    "general_business_assistance": r"\b(?:general business (?:help|assistance)|daily operations)\b",
}


def _capability_matches(text):
    """Prefer specific phrases over matches contained inside them."""
    matches = []
    for capability, pattern in CAPABILITY_PATTERNS.items():
        for match in re.finditer(pattern, text):
            prefix = re.split(r"[.;!?]|\b(?:but|and)\b", text[:match.start()])[-1]
            suffix = text[match.end():].split(".")[0]
            if re.search(r"\b(?:don't|do not|no longer) (?:need|want|require)\b|\b(?:forget|remove|drop|not)\s*$", prefix):
                continue
            if re.match(r"\s+(?:alone\s+)?(?:will not solve|is not needed|isn't needed|not required)\b", suffix):
                continue
            matches.append((capability, match))
    return [(capability, match) for capability, match in matches
            if not any(other != capability and outer.start() <= match.start()
                       and match.end() <= outer.end() and outer.span() != match.span()
                       for other, outer in matches)]


def required_capabilities(job, challenge="", goal=""):
    """Read all explicit workflows, independently of industry or category."""
    text = re.sub(r"[-–—]", " ", f"{challenge} {goal}".casefold())
    caps = sorted({capability for capability, _ in _capability_matches(text)})
    if "local_customer_acquisition" in caps and not re.search(r"\b(?:copy|content|campaigns?)\b", text):
        caps = [c for c in caps if c != "marketing_content_generation"]
    if re.search(r"\bforecast(?:ing)?\b", text) and not any("forecast" in c for c in caps):
        if "inventory_management" in caps:
            caps.append("inventory_forecasting")
        elif "employee_scheduling" in caps or "labor_cost_management" in caps:
            caps.append("advanced_labor_forecasting")
    # Empty legacy calls support catalog diagnostics, never a saved profile.
    if not text.strip():
        defaults = {"Scheduling": "employee_scheduling", "Bookings": "appointment_booking",
                    "Customer Questions": "customer_question_answering", "Reviews": "review_response_drafting",
                    "Social media": "social_content_creation", "Marketing": "marketing_content_generation",
                    "Accounting": "accounting_document_processing", "Estimating": "property_measurement_takeoff",
                    "Landscape Design": "landscape_concept_design", "Daily operations": "general_business_assistance"}
        return [defaults[job]] if job in defaults else []
    return sorted(set(caps))


def split_requirements(job, challenge="", goal="", explicit_preferred=None):
    """Separate optional preferences from essential capabilities."""
    text = re.sub(r"[-–—]", " ", f"{challenge} {goal}".casefold())
    essential = required_capabilities(job, challenge, goal)
    preferred = set(explicit_preferred or [])
    for capability in list(essential):
        matches = [match for name, match in _capability_matches(text) if name == capability]
        if matches and all(re.search(r"\b(?:prefer|would like|nice to have|optional|not essential)\b(?:(?!\b(?:need|require|must)\b)[^.;!?])*$",
                                    text[max(0, m.start()-70):m.start()]) for m in matches):
            essential.remove(capability)
            preferred.add(capability)
    return essential, sorted(preferred - set(essential))


def profile_capabilities(req, job=None):
    """Saved capabilities take precedence over compatibility-only text inputs."""
    if "essential_capabilities" in req:
        return list(req["essential_capabilities"]), list(req.get("optional_capabilities", []))
    return split_requirements(job, req.get("challenge", ""), req.get("goal", ""),
                              req.get("preferred_capabilities"))


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

    if cost["status"] == "ineligible":
        record.update(fit_classification="Ineligible",
                      fit_reason="The known required plan cost or limit exceeds the customer's requirement.")
    elif not required:
        record.update(fit_classification=FIT_RESEARCH,
                      fit_reason="The customer's required workflow could not be identified.")
    elif cost["status"] == "unknown" or not documented:
        if exact:
            record.update(fit_classification=FIT_RESEARCH,
                          fit_reason="A required capability is documented, but plan cost or evidence is unverified.")
        else:
            record.update(fit_classification="Ineligible",
                          fit_reason="No documented capability supports the stated workflow.")
    elif req.get("unresolved_requirements") and exact:
        record.update(fit_classification=FIT_RESEARCH,
                      fit_reason="Customer requirements still need clarification: "
                      + "; ".join(req["unresolved_requirements"]))
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
            not (item.get("prefer_free") and item.get("cost") == 0),
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


def evaluate_catalog(data, req, plans):
    """Evaluate every product against the same saved customer requirements."""
    required, preferred = profile_capabilities(req, req.get("job"))
    output = []
    for _, series in data.iterrows():
        plan = evaluate_tool(series.to_dict(), plans, req, required, preferred)
        output.append({**plan, "tool": series, "eligibility": plan["fit_classification"],
                       "pricing": plan, "reasons": plan.get("reasons", []),
                       "profile_revision": req.get("revision", 0), "prefer_free": req.get("prefer_free", False)})
    output.sort(key=lambda item: _fit_order({**item, "tool_name": item["tool"]["tool_name"]}))
    return output


def score_tools(data, job, cap, comfort, requirements, plans):
    """Compatibility API; no numerical scoring or category-specific evaluation."""
    req = {**(requirements or {}), "budget": cap}
    req.setdefault("job", job)
    return [item for item in evaluate_catalog(data, req, plans)
            if item["fit_classification"] in {FIT_STRONG, FIT_PARTIAL}]


def research_leads(data, job, cap, comfort, requirements, plans):
    req = {**(requirements or {}), "budget": cap}
    req.setdefault("job", job)
    return [item for item in evaluate_catalog(data, req, plans)
            if item["fit_classification"] == FIT_RESEARCH
            and item["status"] != "ineligible" and item.get("covered_capabilities")]


def plan_price_summary(plan):
    """Use the same complete-cost calculation for catalog summaries."""
    cost = _plan_cost(plan, {})
    if cost["status"] == "confirmed":
        summary = f"${cost['cost']:,.2f}/month total ({cost['reason']})"
    else:
        floor = cost.get("lower_bound")
        summary = (f"known minimum ${floor:,.2f}/month; " if floor is not None else "")
        summary += "total unverified (" + cost["reason"] + ")"
    if plan.get("billing_term"):
        summary += "; " + plan["billing_term"]
    return summary
