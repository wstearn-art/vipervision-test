"""Deterministic tool affordability and fit classification."""
from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import re

import pandas as pd

CATALOG_CODE_REVISION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

FIT_STRONG = "Strong Fit"
FIT_PARTIAL = "Partial Fit"
FIT_RESEARCH = "Research Lead"


PRODUCT_COLUMNS = 'product_id,tool_name,provider_id,aliases,industry_tags,categories,description,setup_effort,strengths,limitations,industry_examples,website,evidence_urls,checked_on,research_notes'.split(',')
PLAN_COLUMNS = 'product_id,plan_id,plan_name,currency,billing_period,required_components,fixed_monthly_cost,base_monthly_cost,addon_monthly_cost,per_user_monthly_cost,per_location_monthly_cost,per_channel_monthly_cost,known_cost_floor,included_usage,usage_unit,usage_overage_cost,usage_limit_status,max_users,max_employees,max_locations,capabilities,excluded_capabilities,pricing_status,capability_status,eligibility_status,evidence_urls,checked_on,valid_until,limitations'.split(',')
CHARGE_FIELDS = {'fixed':'fixed_monthly_cost','base':'base_monthly_cost','addon':'addon_monthly_cost',
                 'user':'per_user_monthly_cost','location':'per_location_monthly_cost','channel':'per_channel_monthly_cost'}
QUANTITY_FIELDS = {'user':'seats','location':'locations','channel':'channels'}


def _read_rows(content, columns):
    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    if reader.fieldnames != columns:
        raise ValueError('Catalog headers must match the documented schema: ' + ','.join(columns))
    rows = list(reader)
    for number, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'CSV row {number} has missing or extra cells')
        for key, value in row.items():
            row[key] = value.strip()
    return rows


def _validate_catalog(products, plans):
    def require(condition, where, message):
        if not condition: raise ValueError(where + ': ' + message)
    ids, names, plan_ids, offers = set(), set(), set(), set()
    for row in products:
        identity=row['product_id'];where='Product '+identity
        require(bool(re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',identity)),where,'invalid ID')
        require(identity not in ids and bool(row['tool_name']) and row['tool_name'].casefold() not in names,where,'duplicate or empty identity')
        ids.add(identity);names.add(row['tool_name'].casefold())
        require(row['setup_effort'] in {'easy','medium','high','unknown'},where,'invalid setup effort')
        require(bool(row['industry_tags']) and bool(row['categories']) and bool(row['description']),where,'missing product description/tags')
        if row['industry_examples']:
            try: examples=json.loads(row['industry_examples'])
            except ValueError: raise ValueError(where+': invalid industry_examples JSON')
            require(isinstance(examples,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in examples.items()),where,'invalid examples')
    for row in plans:
        where='Plan '+row['plan_id']
        require(row['product_id'] in ids,where,'unknown product ID')
        require(bool(re.fullmatch(r'[a-z0-9-]+',row['plan_id'])) and row['plan_id'] not in plan_ids and bool(row['plan_name']),where,'duplicate/empty plan identity')
        plan_ids.add(row['plan_id'])
        offer=(row['product_id'],row['plan_name'].casefold(),row['billing_period'])
        require(offer not in offers,where,'duplicate named billing offer')
        offers.add(offer)
        active=_tags(row['required_components'])
        require(bool(active) and active <= set(CHARGE_FIELDS)|{'unknown'} and not ('unknown' in active and len(active)>1),where,'invalid charge components')
        require(len(row['required_components'].split(';'))==len(active),where,'duplicate charge component')
        for key in list(CHARGE_FIELDS.values())+['known_cost_floor','included_usage','usage_overage_cost']:
            if row[key]: require(_number(row[key]) is not None and _number(row[key])>=0,where,'invalid '+key)
        for component,key in CHARGE_FIELDS.items():
            require(component in active or not row[key],where,'inactive charge has an amount: '+key)
        for key in ('max_users','max_employees','max_locations'):
            value=row[key]
            require(not value or value in {'unlimited','not_applicable'} or (value.isdigit() and int(value)>0),where,'invalid '+key)
        require(row['currency']=='USD',where,'only researched USD assessments supported')
        require(row['billing_period'] in {'monthly','annual','custom','unknown'},where,'invalid billing period')
        require(row['usage_limit_status'] in {'known_limit','unlimited','not_applicable','unknown'},where,'invalid quota status')
        if row['usage_limit_status']=='known_limit':
            require(bool(row['included_usage']) and bool(row['usage_unit']),where,'known quota needs amount and unit')
        require(row['usage_limit_status']=='known_limit' or not row['included_usage'],where,'quota contradicts status')
        if row['usage_limit_status'] in {'unlimited','not_applicable'}:
            require(not row['usage_overage_cost'],where,'unlimited/unmetered quota contradicts overage charge')
        if row['usage_limit_status']=='not_applicable':require(not row['usage_unit'],where,'unmetered quota contradicts usage unit')
        if row['usage_overage_cost']:require(bool(row['usage_unit']),where,'overage needs unit')
        for key,allowed in [('pricing_status',{'verified','documented','unverified','quote'}),
                            ('capability_status',{'verified','documented','unverified'}),
                            ('eligibility_status',{'verified','documented','conditional','unverified'})]:
            require(row[key] in allowed,where,'invalid '+key)
        require(not (_tags(row['capabilities'])&_tags(row['excluded_capabilities'])),where,'capability included and excluded')
        require((_tags(row['capabilities'])|_tags(row['excluded_capabilities']))<=set(CAPABILITY_DEFINITIONS),where,'unknown capability ID')
        if active <= {'fixed','base','addon'} and all(row[CHARGE_FIELDS[c]] for c in active):
            total=sum(Decimal(row[CHARGE_FIELDS[c]]) for c in active)
            require(not row['known_cost_floor'] or total>=Decimal(row['known_cost_floor']),where,'complete total below known cost floor')
    require(bool(products) and bool(plans), 'Catalog','must not be empty')
    require({p['product_id'] for p in plans}==ids,'Catalog','every product needs plans')
    for row in products+plans:
        where=row.get('plan_id',row['product_id'])
        for key in ('checked_on','valid_until'):
            if row.get(key):
                try:require(date.fromisoformat(row[key]).isoformat()==row[key],where,'invalid date')
                except ValueError:raise ValueError(where+': invalid '+key)
                if key=='checked_on':require(date.fromisoformat(row[key])<=date.today(),where,'research date is in the future')
        urls=([row['website']] if 'website' in row else [])+list(_tags(row['evidence_urls']))
        require(bool(row['evidence_urls']) and bool(row['checked_on']),where,'research source/date required')
        for url in urls:require(bool(re.fullmatch(r'https://[^\s/]+(?:/[^\s]*)?',url)),where,'invalid HTTPS source')


def _plan_view(row, names):
    """Derived display aliases for existing controls; CSV v2 remains authoritative."""
    p=dict(row)
    fixed=row['fixed_monthly_cost']
    p.update(tool_name=names[row['product_id']],monthly_price=fixed,
        evidence_url=row['evidence_urls'].split(';')[0],verified_date=row['checked_on'],
        evidence_status={'verified':'official_verified','documented':'catalog_source'}.get(row['capability_status'],'unverified'),
        billing_term='annual billing; monthly equivalent' if row['billing_period']=='annual' else row['billing_period'])
    return p


def product_view(row, plans):
    """Adapt product facts to existing UI field names without a second price source."""
    item=dict(row)
    if 'product_id' not in item:return item
    relevant=plans.get(item['tool_name'],[]) if isinstance(plans,dict) else [p for p in plans if p.get('product_id')==item['product_id']]
    free=any(_tags(p['required_components'])=={'fixed'} and _number(p['fixed_monthly_cost'])==0
             and p['pricing_status'] in {'verified','documented'} for p in relevant)
    costs=[_number(p['known_cost_floor']) for p in relevant if _number(p['known_cost_floor']) is not None]
    item.update(business_type=', '.join(item['industry_tags'].split(';')),challenge=item['categories'],
        effort=item['setup_effort'].title(),pros=item['strengths'],cons=item['limitations'],
        source=item['evidence_urls'],checked_date=item['checked_on'],
        free_price=0.0 if free else float('nan'),monthly_cost=min(costs) if costs else float('nan'),
        price_note=('Free starting option; plan restrictions apply.' if free else
            (f'Published minimum ${min(costs):,.2f}/month equivalent; billing terms, seats, usage and required subscriptions may change the total.' if costs else
             'Total price is not verified; provider research is required.')))
    return item


@lru_cache(maxsize=4)
def _parse_catalog(product_bytes, plan_bytes):
    products=_read_rows(product_bytes,PRODUCT_COLUMNS);plans=_read_rows(plan_bytes,PLAN_COLUMNS)
    _validate_catalog(products,plans)
    names={p['product_id']:p['tool_name'] for p in products}
    return products,[_plan_view(p,names) for p in plans]


def load_catalog(product_path,plan_path):
    product_bytes=Path(product_path).read_bytes();plan_bytes=Path(plan_path).read_bytes()
    products,parsed=_parse_catalog(product_bytes,plan_bytes)
    plans=[dict(p) for p in parsed]
    index={}
    for p in plans:index.setdefault(p['tool_name'],[]).append(p)
    tools=pd.DataFrame([product_view(p,index) for p in products])
    tools.attrs['catalog_revision']=hashlib.sha256(product_bytes+b'\0'+plan_bytes).hexdigest()+':'+CATALOG_CODE_REVISION+':'+date.today().isoformat()
    return tools,plans


def load_plan_catalog(path):
    return load_catalog(Path(path).with_name('ai_tools.csv'),path)[1]


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


CAPABILITY_DEFINITIONS = {key:key.replace('_',' ') for key in CAPABILITY_PATTERNS}
CAPABILITY_DEFINITIONS.update({
    'crew_dispatch':'Schedule field crews and assign jobs; distinct from employee shift rosters or personal calendars.',
    'employee_scheduling':'Create employee shift rosters; does not establish field-job dispatch.',
    'personal_task_scheduling':'Schedule one person\'s tasks/calendar; does not establish team shift scheduling.',
    'appointment_booking':'Book customer appointments; does not establish workforce scheduling.',
    'inventory_management':'Track stock or inventory; does not establish forecasting future demand.',
    'inventory_forecasting':'Forecast future inventory demand; separate from tracking stock or food-specific forecasting.',
    'property_measurement_takeoff':'Measure property areas/quantities; does not establish a complete cost estimate workflow.',
    'generic_content_drafting':'Draft general text; does not establish integrated workflow execution.',
    'review_monitoring':'Monitor customer reviews; does not include drafting replies.',
    'estimating_workflow':'Prepare estimates; distinct from property measurement alone.',
    'multi_step_workflows':'Execute a multi-step workflow only within documented plan scope.',
    'ai_concierge':'Provider-specific concierge work; do not infer booking or answering unless separately documented.',
})


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


def _plan_cost(plan, req):
    """One additive cost calculation; unknown quantities never become defaults."""
    active=_tags(plan['required_components']); total=Decimal('0'); unknown=[];parts=[]
    if 'unknown' in active:unknown.append('required plan charges are not verified')
    for component in sorted(active-{'unknown'}):
        amount=_number(plan.get(CHARGE_FIELDS[component]))
        if amount is None or amount<0:
            unknown.append(component+' charge is not verified');continue
        quantity=1
        if component in QUANTITY_FIELDS:
            quantity=_number(req.get(QUANTITY_FIELDS[component]))
            if quantity is None or quantity<1:
                unknown.append('required '+QUANTITY_FIELDS[component]+' count is unstated');continue
        total+=Decimal(str(amount))*Decimal(str(quantity))
        parts.append(f"${amount:,.2f}"+(f" × {quantity:g} {QUANTITY_FIELDS[component]}" if component in QUANTITY_FIELDS else ' '+component+' charge'))
    included=_number(plan.get('included_usage'));usage=_number(req.get('monthly_usage'))
    unit=str(req.get('usage_unit') or '').casefold();plan_unit=str(plan.get('usage_unit') or '').casefold()
    aliases={'callers':'unique_customers','customers':'unique_customers','resolved_questions':'outcomes','inquiries':'conversations'}
    quota=plan.get('usage_limit_status','unknown')
    if usage is not None and quota!='not_applicable' and quota!='unlimited':
        if not unit or not plan_unit or aliases.get(unit,unit)!=aliases.get(plan_unit,plan_unit):
            unknown.append('requested usage unit is unstated or incompatible with the plan')
        elif quota!='known_limit' or included is None:
            unknown.append('usage limit is not verified')
        elif usage>included:
            rate=_number(plan.get('usage_overage_cost'))
            if rate is None:unknown.append('usage exceeds included volume and overage price is unknown')
            else:
                extra=Decimal(str(usage-included))*Decimal(str(rate));total+=extra
                parts.append(f"${extra:,.2f} estimated usage overage")
    elif quota=='known_limit' or (quota=='unknown' and plan_unit):
        unknown.append('required monthly usage or plan capacity is unverified')
    minimum=_number(plan.get('known_cost_floor'));floor=max(minimum or 0,float(total))
    def result(status,reason):
        return {'status':status,'cost':float(total) if status=='confirmed' else None,
                'lower_bound':floor if minimum is not None or total>0 or status=='confirmed' else None,
                'reason':'; '.join(parts+reason)}
    for field,limit_field in (('employees','max_employees'),('seats','max_users'),('locations','max_locations')):
        limit=_number(plan.get(limit_field));quantity=_number(req.get(field))
        if limit is not None:
            if quantity is None:unknown.append('required '+field+' count is unstated')
            elif quantity>limit:return result('ineligible',[f'plan limit is {limit:g} {field}; profile requires {quantity:g}'])
        elif not plan.get(limit_field):
            component={'seats':'user','locations':'location'}.get(field)
            workforce=field=='employees' and bool(_tags(plan.get('capabilities'))&{
                'employee_scheduling','shift_swap_management','labor_cost_management','advanced_labor_forecasting'})
            if quantity is not None and quantity>1 and (component in active or workforce):
                unknown.append('plan capacity for multiple '+field+' is unverified')
    term=plan.get('billing_period','unknown')
    if req.get('monthly_billing_only'):
        if term=='annual':return result('ineligible',['annual billing conflicts with monthly-only requirement'])
        if term!='monthly':unknown.append('monthly billing availability is unverified')
    if plan.get('pricing_status') in {'unverified','quote'}:unknown.append('plan pricing evidence is unverified')
    if plan.get('eligibility_status') in {'conditional','unverified'}:unknown.append('essential plan restrictions need verification')
    if plan.get('valid_until') and plan['valid_until']<str(req.get('as_of') or date.today()):unknown.append('researched offer has expired')
    if minimum is not None and float(total)<minimum and not unknown:unknown.append('computed total is below the researched cost floor')
    cap=_number(req.get('budget'))
    if cap is not None and floor>cap:return result('ineligible',[f'known minimum ${floor:,.2f} exceeds ${cap:,.2f} budget'])
    return result('unknown',unknown) if unknown else result('confirmed',[])


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
    candidates = plans.get(name, []) if isinstance(plans, dict) else [plan for plan in plans if plan["tool_name"] == name]
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
    index={}
    for plan in plans:index.setdefault(plan['tool_name'],[]).append(plan)
    for _, series in data.iterrows():
        series=pd.Series(product_view(series.to_dict(),index)) if 'business_type' not in series else series
        plan = evaluate_tool(series.to_dict(), index, req, required, preferred)
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
