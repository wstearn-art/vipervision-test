import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import requests
from vee_chat import render_vee_chat
st.set_page_config(
    page_title="VipervisionAI | Find the right AI tool",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ai_tools.csv"
VEE_FILE = BASE_DIR / "vee.png"
EXPECTED_TOOL_COUNT = 22
PRICE_CHECK_DATE = "Nov 6, 2026"  # replaced by the newest checked_date in the catalog

REQUIRED_COLUMNS = [
    "tool_name",
    "business_type",
    "challenge",
    "free_price",
    "monthly_cost",
    "price_note",
    "effort",
    "pros",
    "cons",
    "website",
    "source",
    "checked_date",
]

BUDGET_CAP = {
    "Free only": 0,
    "Under $20 a month": 20,
    "Under $50 a month": 50,
    "Under $100 a month": 100,
    "Over $100 a month": float("inf"),
}

JOBS = [
    "Reviews",
    "Customer Questions",
    "Social media",
    "Marketing",
    "Bookings",
    "Daily operations",
    "Scheduling",
    "Accounting",
    "Estimating",
    "Landscape Design",
]

# Plain words that map typed text in the chat helper to one catalog job.
JOB_WORDS = {
    "Reviews": ["review", "reviews", "rating", "ratings", "reputation", "stars"],
    "Customer Questions": [
        "question", "questions", "inbox", "message", "messages", "texts", "calls", "phone", "caller",
        "chatbot", "answer", "answers", "inquiries",
    ],
    "Social media": [
        "social", "instagram", "facebook", "tiktok", "post", "posts",
        "flyer", "flyers", "design", "graphics",
    ],
    "Marketing": [
        "marketing", "email", "emails", "newsletter", "campaign",
        "campaigns", "promotion", "promotions", "loyalty",
    ],
    "Bookings": [
        "booking", "bookings", "book", "reservation", "reservations",
        "tables", "appointment", "appointments",
    ],
    "Daily operations": [
        "pos", "orders", "ordering", "payments", "register", "checkout",
        "operations", "sales",
    ],
    "Scheduling": [
        "schedule", "schedules", "scheduling", "shifts", "staff",
        "employees", "timesheets", "hours",
    ],
    "Accounting": [
        "accounting", "bookkeeping", "taxes", "invoices", "expenses", "books",
    ],
    "Estimating": ["estimate", "estimates", "estimating", "quote", "quotes", "measurements", "measure", "acreage", "takeoff", "takeoffs"],
    "Landscape Design": ["landscape", "landscaping", "garden", "gardens", "yard", "yards", "visualize", "renderings"],
}

BUSINESS_TYPES = ["Restaurant / Cafe", "Coffee Shop", "Landscaping"]
BUDGET_OPTIONS = list(BUDGET_CAP)
COMFORT_OPTIONS = ["Keep it simple", "I can learn a bit"]

# The team fills these in before the final. Lines show only when text is added.
TEAM_CONTRIBUTIONS = {}  # example: {"Cypress": "Built the catalog research"}
OUTSIDE_COACHING = []  # example: ["Dr. Name, Smith School, advice on scoring"]

def money(amount):
    """Plain text money for metrics, with no minus sign."""
    if amount < 0:
        return f"Minus ${abs(amount):,.0f}"
    return f"${amount:,.0f}"

def md(text):
    """Escape dollar signs so Streamlit does not read price text as math."""
    return str(text).replace("$", "\\$")

def date_label(value):
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return PRICE_CHECK_DATE
    return f"{stamp.strftime('%b')} {stamp.day}, {stamp.year}"

def show_table(data):
    """Full width table that works on older and newer Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(data, use_container_width=True, hide_index=True)

@st.cache_data
def load_tools(path, catalog_revision=None):
    data = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("The catalog is missing required columns: " + ", ".join(missing))
    data = data.copy()
    data["free_price"] = pd.to_numeric(data["free_price"], errors="coerce")
    data["monthly_cost"] = pd.to_numeric(data["monthly_cost"], errors="coerce")
    data["business_type"] = data["business_type"].fillna("")
    data["challenge"] = data["challenge"].fillna("").astype(str).str.strip()
    data["effort"] = data["effort"].fillna("").astype(str).str.strip()
    if data["tool_name"].isna().any() or data["tool_name"].duplicated().any():
        raise ValueError("Every catalog tool must have one unique name.")
    return data

try:
    tools = load_tools(DATA_FILE, (DATA_FILE.stat().st_mtime_ns, DATA_FILE.stat().st_size))
except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
    st.error("The tool catalog could not be loaded. Check ai_tools.csv and try again.")
    st.caption(str(error))
    st.stop()

newest_check = pd.to_datetime(tools["checked_date"], errors="coerce").max()
if pd.notna(newest_check):
    PRICE_CHECK_DATE = date_label(newest_check)

if len(tools) < EXPECTED_TOOL_COUNT:
    st.warning(
        f"The catalog currently contains {len(tools)} tools; "
        f"the original baseline contained {EXPECTED_TOOL_COUNT}."
    )

def business_tools(data, business_type):
    """Return tools whose catalog business type includes the selected type."""
    return data[
        data["business_type"].apply(
            lambda value: business_type in [part.strip() for part in str(value).split(",")]
        )
    ].copy()

def price_fits(row, cap):
    """Check whether a published free starting option or monthly price fits."""
    if pd.notna(row["free_price"]) and row["free_price"] == 0:
        return True
    if pd.isna(row["monthly_cost"]):
        return False
    return float(row["monthly_cost"]) <= cap

def starting_cost(tool):
    """Monthly cost to start: zero for a free start, else the published price."""
    if pd.notna(tool["free_price"]) and tool["free_price"] == 0:
        return 0.0
    if pd.isna(tool["monthly_cost"]):
        return None
    return float(tool["monthly_cost"])

def evaluate_plan_cost(tool, requirements=None):
    """Conservative, catalog-specific monthly cost assessment for recommendations and ROI.

    'confirmed' means the documented tier covers the KNOWN requirements, not a
    live provider quote. 'unknown' NEVER means free; 'ineligible' means the
    documented minimum is already over budget. Only the catalog is consulted.
    """
    req = requirements or {}
    name = str(tool.get("tool_name", ""))
    note = str(tool.get("price_note", ""))
    cap = req.get("budget", float("inf"))
    seats = req.get("seats")
    locations = req.get("locations") or 1
    usage = req.get("monthly_usage")
    usage_unit = str(req.get("usage_unit") or "")
    monthly_only = bool(req.get("monthly_billing_only"))
    free = pd.notna(tool.get("free_price")) and float(tool["free_price"]) == 0
    base = None if pd.isna(tool.get("monthly_cost")) else float(tool["monthly_cost"])

    cost = None
    plan = "Catalog starting option"
    reasons = []
    # These are explicitly documented plan conditions from the supplied CSV.
    if name == "Intercom Fin AI Agent":
        plan = "Fin usage-based outcomes (base subscription additional/uncertain)"
        if usage is not None and usage_unit in ("outcomes", "resolved_questions", "inquiries"):
            minimum = 0.99 * usage
            reasons.append(f"{usage:g} resolved outcomes at $0.99 each cost at least ${minimum:,.2f}, before seats/base plans")
            if minimum > cap:
                return {"status": "ineligible", "cost": None, "lower_bound": minimum,
                        "plan": plan, "reason": "; ".join(reasons)}
        reasons.append("Final total depends on resolved outcomes and required base/seat charges")
    elif name == "Tidio Lyro AI":
        plan = "Lyro 50-conversation starting tier"
        if usage is not None and usage_unit in ("conversations", "inquiries", "resolved_questions", "outcomes") and usage > 50:
            reasons.append(f"{usage:g} conversations exceed the 50-conversation starting tier")
        else:
            reasons.append("AI conversation limits and eligible plan must be verified before pricing the workload")
    elif name == "Goodcall":
        plan = "Starter with 100 unique customers"
        if usage is not None and usage_unit in ("callers", "customers", "unique_customers"):
            cost = 79 + max(0, usage - 100) * 0.79
            reasons.append(f"{usage:g} unique customers; 100 included, $0.79 each additional")
        else:
            reasons.append("Number of unique customers needed to determine overages")
    elif name == "Reclaim AI":
        if seats is None:
            reasons.append("Number of software seats unknown; free Lite is one user only")
        elif seats == 1:
            cost = 0.0
            plan = "Lite free, one user"
        elif 2 <= seats <= 10:
            cost = 12.0 * seats if monthly_only else None
            plan = "Starter, up to 10 users"
            if not monthly_only:
                reasons.append("Starter price depends on monthly versus annual billing ($12 monthly; $10 annual per seat)")
        else:
            reasons.append("Starter is limited to 10 users; an eligible higher tier is not priced here")
    elif name == "7shifts":
        plan = "Premium advanced forecasting"
        cost = 149.99 * locations
        reasons.append(f"${149.99:,.2f} per location across {locations} location(s)")
    elif name == "Cactus AI":
        plan = "Growth (AI forecasting features)"
        cost = 149.0 * locations
        reasons.append(f"$149 per location across {locations} location(s)")
    elif name == "SiteRecon Scout AI":
        plan = "Scout paid credits"
        if usage is not None and usage_unit in ("property_acre_credits", "credits"):
            if usage <= 20:
                cost = 39.0
                plan = "Scout 20 property-acre credits"
            elif usage <= 50:
                cost = 79.0
                plan = "Scout 50 property-acre credits"
            else:
                reasons.append("More than 50 credits; higher-volume cost not in catalog")
        else:
            reasons.append("Property-acre credit need unknown; free two-measurement offer does not establish paid-credit eligibility")
    elif name == "Buffer AI Assistant":
        plan = "Free: three channels; 10 queued posts per channel"
        channels = req.get("channels")
        if channels is not None and channels <= 3:
            cost = 0.0
        else:
            # The catalog lists $5/channel as annual equivalent, not monthly-billed.
            reasons.append("Free plan covers up to three channels; confirm channels and queue requirements")
        if monthly_only and cost is None:
            reasons.append("$5/channel figure is annually billed; monthly-billed price is not in catalog")
    elif name == "Copy.ai":
        plan = "Chat, 5 included seats"
        if seats is None:
            reasons.append("Number of required software seats not established (five included)")
        elif seats > 5:
            reasons.append("Requested software seats exceed five included Chat seats")
        else:
            cost = 29.0
    elif name == "Calendly Callie":
        plan = "Callie AI bundle plus eligible per-seat plan"
        reasons.append("$18/seat and $8 AI add-on are annual prices; monthly AI bundle cost unverified")
    elif name == "Jobber AI Receptionist":
        plan = "AI Receptionist add-on"
        reasons.append("$29 add-on requires a qualifying Jobber subscription; total unknown")
    elif name == "Canva AI":
        plan = "Limited free AI use"
        reasons.append("Free AI allowance and monthly-billed price not confirmed for stated output needs")
    elif name == "Canva Magic Studio":
        plan = "Free with limited AI operations"
        # Do not claim free covers a business's requested output volume.
        reasons.append("Free AI usage allowance may not cover the required content volume")
    elif name == "Predis.ai":
        plan = "Core monthly, limited generation credits"
        reasons.append("Generation credit needs not established; Core does not include auto-posting")
    elif name == "Dext AI Assist":
        plan = "Base subscription plus AI Assist"
        cost = 42.0
    elif name == "Jasper":
        plan = "Pro per software seat"
        if seats is not None:
            cost = 69.0 * seats
        else:
            reasons.append("Per-seat subscription; required seat count unknown")
    elif name == "CafeSquad AI":
        plan = "Growth, 30 AI review drafts"
        if usage is not None and usage_unit in ("drafts", "reviews") and usage > 30:
            reasons.append("Requested draft volume exceeds 30 included Growth drafts")
        else:
            cost = 299.0
    elif name == "Podium AI":
        plan = "Core annual commitment plus 10DLC fee"
        if not monthly_only:
            reasons.append("Published $399 base plus $5 10DLC; annual commitment required")
        else:
            reasons.append("12-month commitment conflicts with monthly-only preference")
    elif name == "Neighborbrite":
        plan = "Landscaper estimate workflow"
        cost = 149.0 if "estimat" in str(req.get("task_text", "")).lower() else 49.0
    elif name == "ChatGPT":
        plan = "Free or Plus general assistant"
        cost = 0.0
    elif name == "Birdeye Reviews AI" or name == "PreciTaste":
        reasons.append("Quote-only deployment; no verified monthly total")
    elif free:
        reasons.append("Free tier available but eligibility for the requested capabilities is not established")
    elif base is not None:
        # Only fixed, unqualified monthly catalog prices are confidently usable.
        if not re.search(r"per.seat|/seat|per.location|/location|usage|additional|add.on|annual|billed|credit|conversation|outcome|customer|quote", note, re.I):
            cost = base
        else:
            reasons.append("Starting figure has plan, billing, seat, or usage conditions")
    else:
        reasons.append("No verified monthly price in the catalog")

    if cost is None:
        return {"status": "unknown", "cost": None, "lower_bound": None,
                "plan": plan, "reason": "; ".join(reasons) or "Applicable total monthly cost unverified"}
    if cost > cap:
        return {"status": "ineligible", "cost": cost, "lower_bound": cost,
                "plan": plan, "reason": "; ".join(reasons) or "Required plan exceeds budget"}
    return {"status": "confirmed", "cost": cost, "lower_bound": cost,
            "plan": plan, "reason": "; ".join(reasons) or "Catalog-priced plan fits the known requirements"}


def score_tools(data, job, cap, comfort, requirements=None):
    """Explainable job 3 / price 2 / effort 1, with contextual cost checks."""
    scored = []
    for _, row in data.iterrows():
        if row["challenge"] != job:
            continue
        pricing = None
        if requirements is not None:
            pricing = evaluate_plan_cost(row, {**requirements, "budget": cap})
            if pricing["status"] == "ineligible":
                continue
            if cap == 0 and pricing["status"] != "confirmed":
                continue
        elif not price_fits(row, cap):
            continue

        score = 3
        breakdown = {"Job fit": 3, "Price fit": 0, "Effort fit": 0}
        reasons = [f"Matches catalog job: {job}"]
        if pricing is not None:
            if pricing["status"] == "confirmed":
                score += 2
                breakdown["Price fit"] = 2
                reasons.append(f"Applicable catalog plan: {pricing['plan']} (${pricing['cost']:,.2f}/month)")
            else:
                reasons.append("Total price/plan eligibility not verified; budget fit remains unknown")
        elif price_fits(row, cap):
            score += 2
            breakdown["Price fit"] = 2
            reasons.append("Published starting option fits the selected budget; plan terms may apply")
        if row["effort"] == "Easy":
            score += 1
            breakdown["Effort fit"] = 1
            reasons.append("Setup is marked easy in the catalog")
        elif row["effort"] == "Medium" and comfort == "I can learn a bit":
            score += 1
            breakdown["Effort fit"] = 1
            reasons.append("Medium setup matches your comfort level")
        elif row["effort"] == "Medium":
            reasons.append("Setup is medium and may need extra learning")
        scored.append({"score": score, "tool": row, "reasons": reasons,
                       "breakdown": breakdown, "pricing": pricing})
    scored.sort(key=lambda item: (-item["score"], str(item["tool"]["tool_name"]).casefold()))
    return scored

@st.cache_data
def run_scorer_checks(data):
    """Run the scorer on every business type, job, budget, and comfort pair."""
    total = 0
    passed = 0
    for business_type in BUSINESS_TYPES:
        pool = business_tools(data, business_type)
        for job in JOBS:
            for cap in BUDGET_CAP.values():
                for comfort in COMFORT_OPTIONS:
                    ranked = score_tools(pool, job, cap, comfort)
                    scores = [item["score"] for item in ranked]
                    ok = (
                        all(3 <= score <= 6 for score in scores)
                        and scores == sorted(scores, reverse=True)
                        and all(item["tool"]["challenge"] == job for item in ranked)
                        and all(
                            sum(item["breakdown"].values()) == item["score"]
                            for item in ranked
                        )
                    )
                    total += 1
                    passed += int(ok)
    return passed, total

def team_size_note(tool, employees):
    """Compare team size with a free plan limit stated in the catalog price note.
    This note is information only. It does not change the score.
    """
    match = re.search(r"Free up to (\d+) (team members|employees)", str(tool["price_note"]))
    if not match:
        return None
    limit = int(match.group(1))
    if employees <= limit:
        return f"The free plan covers up to {limit} {match.group(2)}, so a team of {employees} fits."
    return (
        f"The free plan covers up to {limit} {match.group(2)}, and your team of "
        f"{employees} is above that, so expect to need a paid plan."
    )

def vee_avatar():
    return str(VEE_FILE) if VEE_FILE.exists() else "assistant"

def show_tool_link(url, name=None):
    """Display a readable link without exposing a raw URL as page text."""
    label = f"Visit {name} website" if name else "Visit provider website"
    st.markdown(f"[{label}]({url})")

def show_tool_result(item, number, employees=None):
    tool = item["tool"]
    st.markdown(f"#### {number}. {tool['tool_name']}")
    st.write(f"**Fit score:** {item['score']} out of 6")
    st.write(
        "Score breakdown: "
        f"job fit {item['breakdown']['Job fit']} out of 3, "
        f"price fit {item['breakdown']['Price fit']} out of 2, "
        f"effort fit {item['breakdown']['Effort fit']} out of 1."
    )
    st.write(f"**Price:** {md(tool['price_note'])}")
    st.write(f"**Setup:** {tool['effort']}")
    st.write("**Why it matched:** " + md("; ".join(item["reasons"])))
    if employees:
        note = team_size_note(tool, employees)
        if note:
            st.write(f"**Team size check:** {note}")
    st.write(f"**What it does well:** {md(tool['pros'])}")
    st.write(f"**What to check:** {md(tool['cons'])}")
    show_tool_link(tool["website"])
    st.caption(
        f"Price checked {date_label(tool['checked_date'])}. "
        "Verify the current plan with the provider before purchase."
    )
    st.markdown(f"[Source details](https://{tool['source']})")

def show_first_week_plan(tool, systems=""):
    st.subheader(f"Your first week with {tool['tool_name']}")
    st.write(
        "1. Visit the provider website and review the current plan and terms. "
        f"Catalog price note: {md(tool['price_note'])}."
    )
    if systems.strip():
        st.write(
            "2. Ask the provider whether it works with the systems you already "
            f"use: {md(systems.strip())}."
        )
    else:
        st.write("2. Check whether the tool works with the systems your business already uses.")
    st.write("3. Test one real task with a small amount of work before relying on it.")
    st.write(
        "4. After seven days, compare the time saved and the actual cost. "
        f"Keep this in mind: {md(tool['cons'])}."
    )
    st.caption(
        "This is an adoption checklist. VipervisionAI does not create accounts, "
        "connect systems, or purchase tools for you."
    )

def build_starter_stack(pool, cap, comfort, first_job):
    """Top budget fit tool for up to three different jobs, same rules as the quiz.
    The job you picked comes first, then the other jobs in catalog order.
    """
    order = [first_job] + [job for job in JOBS if job != first_job]
    stack = []
    for job in order:
        ranked = score_tools(pool, job, cap, comfort)
        if ranked and ranked[0]["breakdown"]["Price fit"] == 2:
            stack.append({"job": job, "item": ranked[0]})
        if len(stack) == 3:
            break
    return stack

def show_starter_stack(pool, budget_label, comfort, first_job):
    cap = BUDGET_CAP[budget_label]
    stack = build_starter_stack(pool, cap, comfort, first_job)
    st.subheader("Your starter stack")
    if not stack:
        st.info("No catalog tools fit your budget for a starter stack. Try a higher budget.")
        return
    st.write(
        "One top tool for each of up to three different jobs, starting with your "
        "job and scored with the same rules as your quiz answers. Only tools that "
        "fit your budget are included."
    )
    start_total = 0.0
    paid_total = 0.0
    rows = []
    for entry in stack:
        tool = entry["item"]["tool"]
        st.write(
            f"**{tool['tool_name']}** for {entry['job']}: "
            f"{md(tool['price_note'])}. Score "
            f"{entry['item']['score']} out of 6."
        )
        show_tool_link(tool["website"], tool["tool_name"])
        start_cost = starting_cost(tool)
        start_total += start_cost or 0.0
        if pd.notna(tool["monthly_cost"]):
            paid_total += float(tool["monthly_cost"])
        rows.append(
            {
                "Tool": tool["tool_name"],
                "Job": entry["job"],
                "Price": tool["price_note"],
                "Starting monthly cost": start_cost,
                "Website": tool["website"],
            }
        )
    st.write(f"Estimated monthly total to start: ${start_total:,.0f}.")
    if paid_total > start_total:
        st.write(
            "If you later move every tool in the stack to its published paid plan: "
            + md(f"${paid_total:,.0f}")
            + " a month."
        )
    if cap != float("inf"):
        if start_total <= cap:
            st.success(f"The starting total is within your budget of {md(budget_label)}.")
        else:
            st.warning(f"The starting total is above your budget of {md(budget_label)}.")
    st.caption(
        "Totals use published starting prices only. Card processing fees and per "
        "use fees are not included."
    )
    stack_csv = pd.DataFrame(rows).to_csv(index=False)
    st.download_button(
        "Download your starter stack",
        data=stack_csv,
        file_name="vipervision_starter_stack.csv",
        mime="text/csv",
    )

def shortlist_csv(ranked):
    rows = []
    for number, item in enumerate(ranked[:3], start=1):
        tool = item["tool"]
        rows.append(
            {
                "Rank": number,
                "Tool": tool["tool_name"],
                "Fit score out of 6": item["score"],
                "Price": tool["price_note"],
                "Setup": tool["effort"],
                "Why it matched": "; ".join(item["reasons"]),
                "Website": tool["website"],
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)
# This is the AI integration 
def ai_job_from_text(text):
    """Use OpenRouter to interpret a business problem."""
    if not text.strip():
        return None

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Classify this small-business problem into "
                            "exactly one of these categories: "
                            + ", ".join(JOBS)
                            + ". Respond with only the category name."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
            },
            timeout=20,
        )

        response.raise_for_status()
        category = response.json()["choices"][0]["message"]["content"].strip()

        return category if category in JOBS else None

    except Exception:
        return None
def job_from_text(text):
    """Pick the job whose plain words appear most often in the typed text."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    best_job = None
    best_hits = 0
    for job in JOBS:
        hits = len(words & set(JOB_WORDS[job]))
        if hits > best_hits:
            best_job = job
            best_hits = hits
    return best_job

def show_chat_answer(answer):
    job = answer["job"]
    with st.chat_message("user"):
        if answer["typed"]:
            st.write(md(answer["text"]))
        st.write(
            f"I run a {answer['business']}. My job is {job}. "
            f"My budget is {md(answer['budget'])}. My comfort level is "
            f"{answer['comfort'].lower()}."
        )
    with st.chat_message("assistant", avatar=vee_avatar()):
        if answer["typed"] and answer["understood"]:
            st.write(f"I read your message as the job {job}.")
        elif answer["typed"]:
            st.write(
                "I could not match your words to a catalog job, so I used the "
                f"job you chose in the list: {job}."
            )
        pool = business_tools(tools, answer["business"])
        ranked = score_tools(pool, job, BUDGET_CAP[answer["budget"]], answer["comfort"])
        if ranked:
            st.write("Here are the strongest catalog matches for your answers.")
            for index, item in enumerate(ranked[:3], start=1):
                with st.container(border=True):
                    show_tool_result(item, index)
        else:
            st.info(
                "I found no catalog match for that job and business type. "
                "Try another job to explore the current tools."
            )

def clear_results():
    st.session_state.pop("recommendation_results", None)

def fill_roi_price():
    choice = st.session_state.get("roi_tool", "My own number")
    if choice == "My own number":
        return
    row = tools[tools["tool_name"] == choice].iloc[0]
    if pd.notna(row["monthly_cost"]):
        st.session_state["roi_price"] = int(round(float(row["monthly_cost"])))

# The three explore cards. Tool names and counts come straight from the catalog.
EXPLORE_CARDS = [
    {
        "key": "customer",
        "category": "Support and reputation",
        "title": "Customer questions",
        "description": "Reply to reviews and answer customer questions in one place.",
        "jobs": ["Reviews", "Customer Questions"],
        "button": "Explore customer questions",
        "quiz_job": "Customer Questions",
    },
    {
        "key": "marketing",
        "category": "Content and campaigns",
        "title": "Marketing",
        "description": "Make social posts, email campaigns, and simple promotions.",
        "jobs": ["Marketing", "Social media"],
        "button": "Explore marketing",
        "quiz_job": "Marketing",
    },
    {
        "key": "bookings",
        "category": "Scheduling and workflow",
        "title": "Bookings and operations",
        "description": "Handle online bookings, staff schedules, and daily tasks.",
        "jobs": ["Bookings", "Daily operations", "Scheduling"],
        "button": "Explore bookings",
        "quiz_job": "Bookings",
    },
]

# The three free help cards
FREE_HELP_CARDS = [
    {
        "key": "sba",
        "category": "Local support",
        "title": "Find local SBA help",
        "description": "Find Small Business Administration offices and advisors near you.",
        "button": "Find local SBA help",
        "url": "https://www.sba.gov/local-assistance",
    },
    {
        "key": "score",
        "category": "Mentoring",
        "title": "SCORE business mentoring",
        "description": "Get free advice from volunteer mentors with business experience.",
        "button": "Visit SCORE",
        "url": "https://www.score.org",
    },
    {
        "key": "partners",
        "category": "Partner network",
        "title": "SBA resource partners",
        "description": "Small Business Development Centers and other SBA partner programs.",
        "button": "See SBA partners",
        "url": "https://www.sba.gov/partners",
    },
]

def show_explore_card(card):
    """One explore card: label, title, description, catalog tools, button, and real counts."""
    matches = tools[tools["challenge"].isin(card["jobs"])]
    free_count = int((matches["free_price"] == 0).sum())
    easy_count = int((matches["effort"] == "Easy").sum())
    chips = "".join(
        f'<span class="card-chip">{escape(str(name))}</span>' for name in matches["tool_name"]
    ) or '<span class="card-empty">No catalog tools yet</span>'
    badge = '<span class="card-badge">Free options</span>' if free_count else ""
    st.markdown(
        f'<div class="card-top"><span class="card-category">{card["category"]}</span>{badge}</div>'
        f'<div class="card-title">{card["title"]}</div>'
        f'<p class="card-description">{card["description"]}</p>'
        f'<div class="card-label">Tools in our catalog</div>'
        f'<div class="card-chips">{chips}</div>',
        unsafe_allow_html=True,
    )
    if st.button(card["button"] + " \u2192", key=f"explore_{card['key']}", use_container_width=True):
        st.session_state["vee_explore_suggestion"] = card["title"]
    st.markdown(
        f'<div class="card-metrics">'
        f'<div><div class="metric-label">Tools</div><div class="metric-value">{len(matches)}</div></div>'
        f'<div><div class="metric-label">Free start</div><div class="metric-value">{free_count}</div></div>'
        f'<div><div class="metric-label">Easy setup</div><div class="metric-value">{easy_count}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Navigation and hero
# All colors and layout live in style.css
def load_css(file_name):
    with open(BASE_DIR / file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.markdown("<div id='top'></div>", unsafe_allow_html=True)

# Nav bar locked to the top of the page
st.markdown("""
<div class="navbar">
    <div class="nav-logo">VipervisionAI</div>
    <div class="nav-links">
        <a href="./" target="_self">Home</a>
        <a href="./catalog" target="_self">Catalog</a>
        <a href="./about" target="_self">About the Website</a>
        <a href="./#about-the-creators" target="_self">About the Creators</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Home page: everything except the catalog
def home_page():
    # Header box with a glowing pulse line on each side
    PULSE_LEFT = (
        '<svg class="pulse-wave pulse-wave-left" width="140" height="38" viewBox="0 0 300 80" style="width: 140px; height: 38px; flex: 0 0 140px;" aria-hidden="true"><defs><filter id="wave-glow-left" x="-10%" y="-60%" width="120%" height="220%"><feGaussianBlur stdDeviation="3"></feGaussianBlur></filter></defs><path d="M 0 40 H 70 C 78 40 80 28 86 28 C 93 28 94 56 101 56 C 108 56 109 22 117 22 C 124 22 124 48 131 48 C 136 48 137 36 142 36 C 147 36 148 46 153 46 C 158 46 159 30 165 30 C 172 30 172 58 180 58 C 187 58 189 18 197 18 C 205 18 206 52 214 52 C 220 52 222 40 230 40 H 300" fill="none" stroke="#39FF88" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" opacity="0.6" filter="url(#wave-glow-left)"></path><path d="M 0 40 H 70 C 78 40 80 28 86 28 C 93 28 94 56 101 56 C 108 56 109 22 117 22 C 124 22 124 48 131 48 C 136 48 137 36 142 36 C 147 36 148 46 153 46 C 158 46 159 30 165 30 C 172 30 172 58 180 58 C 187 58 189 18 197 18 C 205 18 206 52 214 52 C 220 52 222 40 230 40 H 300" fill="none" stroke="#39FF88" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"></path></svg>'
    )
    PULSE_RIGHT = PULSE_LEFT.replace("pulse-wave-left", "pulse-wave-right").replace("wave-glow-left", "wave-glow-right")
    st.markdown(
        '<div class="hero-row" style="display: flex; align-items: center; justify-content: center; gap: 20px;">'
        + PULSE_LEFT
        + '<div class="top-text"><h1>VipervisionAI</h1>'
        '<p>Find the right AI tools for your small business.</p></div>'
        + PULSE_RIGHT
        + '</div>',
        unsafe_allow_html=True,
    )
    # Division between header and the rest of the page
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown('<div id="chat-with-vee"></div>', unsafe_allow_html=True)
    vee_col, greet_col = st.columns([1, 2])
    with vee_col:
        # Container key lets style.css add Vee's border
        with st.container(key="vee_image"):
            if VEE_FILE.exists():
                try:
                    st.image(str(VEE_FILE), width="stretch")
                except Exception:
                    st.image(str(VEE_FILE), use_container_width=True)
            else:
                st.markdown("#### Vee, your guide")
    with greet_col:
        # Yellow ombre box around the Chat with Vee area
        with st.container(key="vee_chat_box"):
            render_vee_chat(
                tools=tools,
                business_types=BUSINESS_TYPES,
                budget_cap=BUDGET_CAP,
                comfort_options=COMFORT_OPTIONS,
                business_tools=business_tools,
                score_tools=score_tools,
                evaluate_plan_cost=evaluate_plan_cost,
                show_tool_result=show_tool_result,
                ai_job_from_text=ai_job_from_text,
                job_from_text=job_from_text,
                vee_avatar=vee_avatar,
            )

        # Once the first chat with Vee is done, the ROI calculator shows right under it
        vee_state = st.session_state.get("vee_guided", {})
        if vee_state.get("stage") == "done":
            with st.container(border=True, key="vee_roi"):
                st.subheader("Is it worth it? Estimate your ROI")
                st.caption("Illustrative estimate, not a promised saving. Use your own numbers.")

                recommended = st.session_state.get("vee_roi_candidates", [])
                price_choices = ["Enter my own cost"] + [item["name"] for item in recommended]
                default_choice = 1 if recommended else 0
                chosen = st.selectbox("Tool to evaluate", price_choices,
                                      index=default_choice, key="roi_tool_choice")
                selected = next((item for item in recommended if item["name"] == chosen), None)
                cost_known = bool(selected and selected["cost"] is not None)
                if cost_known:
                    tool_price = float(selected["cost"])
                    st.write("**Catalog plan cost for known requirements:** " + md(f"${tool_price:,.2f}/month"))
                    if selected.get("plan"):
                        st.caption("Assessed plan: " + selected["plan"])
                    st.caption(selected["note"])
                    st.caption("Choose 'Enter my own cost' to model a different total.")
                else:
                    # Unknown is not zero. Explicit owner confirmation is necessary
                    # before calculating a net estimate, even if the entered value is $0.
                    profile = vee_state.get("answers", {})
                    custom_key = "roi_custom_" + repr((chosen, profile.get("budget_amount"),
                        profile.get("employees"), profile.get("challenge"), profile.get("goal")))
                    tool_price = float(st.number_input(
                        "Estimated TOTAL monthly tool cost ($)", min_value=0.0,
                        max_value=100000.0, value=0.0, step=5.0,
                        key=custom_key))
                    cost_known = st.checkbox(
                        "Use this amount as my own scenario assumption",
                        value=False, key=custom_key + "_confirmed")
                    if selected:
                        st.warning(selected.get("eligibility_warning") or
                                   "Total price isn't confirmed. Include seats, usage and required subscriptions.")
                    if not cost_known:
                        st.info("Net savings and break-even will appear once you confirm an estimated total cost.")

                c1, c2 = st.columns(2)
                with c1:
                    hours = st.number_input("Hours spent on this task each week",
                                            min_value=0.0, max_value=168.0,
                                            value=5.0, step=0.5, key="roi_hours")
                    value = st.number_input("Hourly value of that work ($)",
                                            min_value=0.0, max_value=10000.0,
                                            value=25.0, step=5.0, key="roi_hourly")
                with c2:
                    reduction = st.slider("Estimated time saved (%)", 0, 100, 30,
                                          key="roi_reduction")
                    st.caption("An assumption to test—not a measured AI performance claim.")

                monthly_time_value = hours * value * 4.33
                estimated_savings = monthly_time_value * reduction / 100
                estimated_net = estimated_savings - tool_price if cost_known else None
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Time value/month", money(monthly_time_value))
                with m2:
                    st.metric("Potential savings/month", money(estimated_savings))
                with m3:
                    st.metric("Estimated net/month", money(estimated_net) if cost_known else "Not yet calculated")
                if cost_known and tool_price > 0 and monthly_time_value > 0:
                    needed = tool_price / monthly_time_value * 100
                    if needed <= 100:
                        st.caption(f"Break-even requires about {needed:.0f}% time savings at these inputs.")
                    else:
                        st.warning("This tool's modeled cost exceeds the full monthly value of the task.")
                if cost_known and estimated_net < 0:
                    st.warning("At these assumptions, costs exceed estimated time savings.")
                if not cost_known:
                    st.caption("Potential savings exclude software costs; this is NOT a net ROI estimate.")
                st.caption("Assumes 4.33 weeks/month. Excludes onboarding, taxes, usage fees, "
                           "additional seats and other costs. Confirm actual pricing and measure savings before purchase.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### Browse the complete catalog")
    st.write("Want to see every researched AI tool, not just Vee's top recommendations?")
    st.link_button("Show All Tools", "./catalog", type="primary")

    # Optional exploration cards after Vee, ROI, and the full-catalog action
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Dark background around the whole explore section
    with st.container(key="explore_section"):
        # Heading with a small pill label, a big title, and a short intro line
        with st.container(key="explore_heading"):
            st.markdown(
                '<div class="explore-pill">AI tools for small business</div>'
                '<div class="explore-title">What <span class="vee-gradient">Vee</span> can help you explore</div>'
                '<p class="explore-subtitle">Browse the AI tools in our catalog by the job you need done, '
                'then compare them to find a good fit for your business.</p>',
                unsafe_allow_html=True,
            )

        card_columns = st.columns(3)
        for column, card in zip(card_columns, EXPLORE_CARDS):
            with column:
                with st.container(key=f"explore_card_{card['key']}"):
                    show_explore_card(card)
        if st.session_state.get("vee_explore_suggestion"):
            st.info(
                "Ask Vee about " + st.session_state["vee_explore_suggestion"].lower()
                + ". The conversation is at the top of this page."
            )
        st.caption(
            f"Counts come from our tool catalog. Prices checked {PRICE_CHECK_DATE}."
        )

    # Divider before Free help
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Free help beyond software, laid out like the explore section
    with st.container(key="explore_section_help"):
        with st.container(key="explore_heading_help"):
            st.markdown(
                '<div class="explore-pill">Free for small business owners</div>'
                '<div class="explore-title">Free help <span class="vee-gradient">beyond</span> software</div>'
                '<p class="explore-subtitle">These free resources can help a small business owner beyond '
                'the tools above. VipervisionAI does not run them and does not receive anything from them.</p>',
                unsafe_allow_html=True,
            )
        help_columns = st.columns(3)
        for column, card in zip(help_columns, FREE_HELP_CARDS):
            with column:
                with st.container(key=f"explore_card_{card['key']}"):
                    st.markdown(
                        f'<div class="card-top"><span class="card-category">{card["category"]}</span>'
                        f'<span class="card-badge">Free</span></div>'
                        f'<div class="card-title">{card["title"]}</div>'
                        f'<p class="card-description">{card["description"]}</p>',
                        unsafe_allow_html=True,
                    )
                    st.link_button(card["button"] + " \u2192", card["url"], use_container_width=True)

    # Divider before the About sections
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # About sections, side by side in two blocks
    about_website_col, about_creators_col = st.columns(2)
    with about_website_col:
        with st.container(key="about_website"):
            st.header("About the Website")
            st.write(
                "VipervisionAI helps small businesses discuss problems with Vee, discover "
                "catalog-grounded AI tool candidates, compare options, and estimate ROI. "
                "We distinguish researched catalog details from unverified suitability. "
                "The site does not purchase or deploy tools."
            )
    with about_creators_col:
        with st.container(key="about_creators"):
            st.header("About the Creators")
            st.write(
                "Built by The Vipers, Team 62, for the AI for Small Business Hackathon: "
                "Usman, Natalie, and Will, team lead."
            )
            if TEAM_CONTRIBUTIONS:
                st.write("**Who did what**")
                for member, contribution in TEAM_CONTRIBUTIONS.items():
                    st.write(f"{member}: {contribution}")
            if OUTSIDE_COACHING:
                st.write("**Coaching and outside help we acknowledge**")
                for line in OUTSIDE_COACHING:
                    st.write(line)
            st.caption(
                "The Hackathon is run for the Smith Master Student Association at the "
                "University of Maryland. As the Hackathon rules state, rights to submissions "
                "belong to the University of Maryland."
            )

    st.caption(
        f"VipervisionAI, by The Vipers. Prices checked {PRICE_CHECK_DATE}. "
        "The prototype can continue improving after the video submission."
    )


# About page: answers common questions and holds the extra details about the site
def about_page():
    # Page heading, same style as the explore section heading
    with st.container(key="explore_heading_about"):
        st.markdown(
            '<div class="explore-pill">About this site</div>'
            '<div class="explore-title">Questions about <span class="vee-gradient">VipervisionAI</span></div>'
            '<p class="explore-subtitle">Quick answers about what the site does, how tools are scored, '
            'and who built it.</p>',
            unsafe_allow_html=True,
        )

    # Common questions
    with st.container(key="about_faq"):
        with st.expander("What does VipervisionAI do?"):
            st.write(
                "It helps small businesses discover, compare, and plan how to "
                "adopt AI tools. Chat with Vee to discuss a problem and see catalog-grounded "
                "candidates with reasons and limitations."
            )
        with st.expander("How does the fit score work?"):
            st.write(
                "Every score uses the same three rules. Job fit is worth 3 points, price fit "
                "is worth 2 points, and effort fit is worth 1 point, for a total out of 6. "
                "The score is a fit guide, not a customer review or a guarantee of results."
            )
        with st.expander("Does an AI pick my tools?"):
            st.write(
                "No. The recommendations come from clear scoring rules, so the same answers "
                "always give the same results, and you can see why each tool matched."
            )
        with st.expander("How current are the prices?"):
            st.write(
                f"Prices in the catalog were checked {PRICE_CHECK_DATE}. Providers can change "
                "their prices and plans, so check the current plan with the provider before "
                "you buy."
            )
        with st.expander("Will VipervisionAI set up the tools for me?"):
            st.write(
                "No. The site does not create accounts, connect systems, or buy tools for you. "
                "It gives you a first week checklist so you can set up a tool with the provider."
            )
        with st.expander("Which businesses does it cover right now?"):
            st.write(
                "The current catalog covers restaurants, coffee shops and landscaping. "
                "We plan to add more verified business categories over time."
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Title for the competition details below
    st.markdown(
        '<div class="about-section-title">Information about the competition</div>',
        unsafe_allow_html=True,
    )

    # Your original dropdowns, styled the same as the questions above
    with st.container(key="about_faq_more"):
        # Research and methodology
        with st.expander("Research and scoring method"):
            st.write(
                "The current catalog contains the tools listed above for restaurants, "
                "coffee shops and landscaping. Each entry includes a job category, business type, "
                "setup effort, price note, provider website, source, and price check date."
            )
            st.write(
                "The fit score uses three visible rules. Job fit contributes 3 points "
                "when the catalog job matches. Price fit contributes 2 points when a "
                "published free starting option or monthly cost fits the selected budget. "
                "Effort fit contributes 1 point when the setup level matches the selected "
                "technology comfort."
            )
            st.write(
                "A quote only price is not treated as a confirmed budget fit. The score "
                "is a platform fit estimate, not an independent performance rating, "
                "customer review, or guarantee."
            )
            st.write(
                "**Beyond reviews:** the score uses facts we can check, which are the job "
                "match, the published price, and the setup effort. It does not use customer "
                "reviews or star ratings. In the sixteen week plan we would add more "
                "checkable facts, such as which systems a tool works with."
            )
            st.write(
                "**What each question tests:** business type and job decide which catalog "
                "tools can match. Budget tests whether a published price fits. Technology "
                "comfort tests whether the setup effort fits. Team size is used for a "
                "free plan limit check when the catalog states one, and it does not change "
                "the score."
            )
            st.write(
                "Sources and dates are shown with each catalog entry. Provider pricing "
                "should be checked again before purchase."
            )
            free_count = int((tools["free_price"] == 0).sum())
            quote_count = int(tools["monthly_cost"].isna().sum())
            one_tool_jobs = sorted(
                job_name
                for job_name, count in tools["challenge"].value_counts().items()
                if count == 1
            )
            st.write(
                f"How we researched: the catalog covers {len(tools)} tools across "
                f"{tools['challenge'].nunique()} jobs, from {tools['source'].nunique()} "
                f"sources, all checked {PRICE_CHECK_DATE}. {free_count} tools have a "
                f"free starting option and {quote_count} "
                f"{'is' if quote_count == 1 else 'are'} quote only. "
                f"{len(one_tool_jobs)} {'job has' if len(one_tool_jobs) == 1 else 'jobs have'} "
                f"one catalog tool each: {', '.join(one_tool_jobs)}."
            )
            st.write("**Catalog coverage by job**")
            flags = tools.assign(
                free_start=tools["free_price"].eq(0),
                quote_only=tools["monthly_cost"].isna(),
                easy_setup=tools["effort"].eq("Easy"),
            )
            coverage = flags.groupby("challenge")[["free_start", "quote_only", "easy_setup"]].sum()
            coverage.insert(0, "tools", flags.groupby("challenge").size())
            coverage = coverage.astype(int).reset_index()
            coverage.columns = [
                "Job",
                "Tools",
                "Free starting option",
                "Quote only",
                "Easy setup",
            ]
            show_table(coverage)

        # Architecture and future work
        with st.expander("How the prototype works and what comes next"):
            st.write(
                "**Current prototype:** the CSV catalog supplies tool details and "
                "published prices. The questionnaire filters by business type and job. "
                "A deterministic Python function calculates the fit score and reasons. "
                "The catalog, comparison view, ROI calculator, and chat helper use "
                "the same catalog data."
            )
            st.graphviz_chart(
                r"""
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", fillcolor="#e8eefc",
                color="#0b1e4b", fontname="Helvetica"];
          catalog [label="ai_tools.csv\nTool catalog"];
          loader [label="load_tools\nChecks the data"];
          scorer [label="score_tools\nJob 3, price 2, effort 1"];
          quiz [label="Quiz results\nTop 3 with reasons"];
          stack [label="Starter stack"];
          chat [label="Chat helper"];
          browse [label="Catalog and compare"];
          roi [label="ROI calculator"];
          catalog -> loader;
          loader -> scorer;
          loader -> browse;
          loader -> roi;
          scorer -> quiz;
          scorer -> stack;
          scorer -> chat;
        }
        """
            )
            st.caption(
                "In words: the catalog file is loaded and checked once. One scoring "
                "function feeds the quiz results, the starter stack, and the chat helper. "
                "The catalog, compare view, and ROI calculator read the same loaded data."
            )
            st.write(
                "**Scalability path:** keep catalog records separate from scoring logic, "
                "add validated tool records and business categories, and test scoring "
                "rules as the catalog grows. The current prototype does not claim "
                "production scale or live provider integrations."
            )
            st.write("**Sixteen week plan**")
            plan = pd.DataFrame(
                [
                    {
                        "Weeks": "1 to 4",
                        "Phase": "Research",
                        "Work": "Interview owners of restaurants, coffee shops, HVAC "
                        "companies, and landscaping companies. Rank their biggest jobs "
                        "and set the metrics we will use to judge tools.",
                    },
                    {
                        "Weeks": "5 to 8",
                        "Phase": "Catalog",
                        "Work": "Grow the catalog beyond twelve tools with a repeatable "
                        "tool discovery process. Record a source and check date for every "
                        "price. Add fields for the point of sale, accounting, and "
                        "calendar systems each tool works with.",
                    },
                    {
                        "Weeks": "9 to 12",
                        "Phase": "Platform",
                        "Work": "Move the catalog to a managed database, add automated "
                        "scoring tests, build multi tool workflows, and match owners to "
                        "Small Business Administration resources.",
                    },
                    {
                        "Weeks": "13 to 16",
                        "Phase": "Validate",
                        "Work": "Run usability tests with first time owners, review "
                        "accessibility, tune the scoring rules with their feedback, and "
                        "prepare the handoff.",
                    },
                ]
            )
            show_table(plan)
            st.write(
                "**Not in the prototype yet:** star ratings, live provider integrations, "
                "and tool deployment. The checklist shows owners how to adopt a tool, and "
                "the owner completes setup with the provider."
            )

        # Catalog health check and scorer test
        with st.expander("Catalog health check and scorer test"):
            st.write(
                "These checks run on every page load against the catalog file and the scorer."
            )
            effort_ok = bool(tools["effort"].isin(["Easy", "Medium"]).all())
            https_ok = bool(tools["website"].astype(str).str.startswith("https://").all())
            dates_ok = bool(pd.to_datetime(tools["checked_date"], errors="coerce").notna().all())
            sources_ok = bool(tools["source"].notna().all())
            jobs_ok = bool(tools["challenge"].isin(JOBS).all())
            passed, total = run_scorer_checks(tools)
            checks = [
                (f"The catalog has {EXPECTED_TOOL_COUNT} tools", len(tools) == EXPECTED_TOOL_COUNT),
                ("Every tool name is unique", not tools["tool_name"].duplicated().any()),
                ("Every tool has a price check date", dates_ok),
                ("Every tool has a source", sources_ok),
                ("Every website link uses https", https_ok),
                ("Every setup level is Easy or Medium", effort_ok),
                ("Every tool job is one the quiz can ask about", jobs_ok),
                ("Vee image file is found", VEE_FILE.exists()),
                (f"Scorer checks passed: {passed} of {total}", passed == total),
            ]
            check_rows = [
                {"Check": label, "Result": "Pass" if ok else "Needs attention"}
                for label, ok in checks
            ]
            show_table(pd.DataFrame(check_rows))
            st.caption(
                "The scorer checks confirm that scores stay between 3 and 6, results are "
                "sorted from best to worst, every result matches the chosen job, and the "
                "breakdown adds up to the score. They run for all listed business types, all "
                "jobs, all budgets, and both comfort levels."
            )


# Catalog page: the full list of tools on its own page
def catalog_page():
    # Catalog
    st.header("Catalog: all tools")
    st.write(
        f"The catalog contains {len(tools)} tools. Prices were checked "
        f"{PRICE_CHECK_DATE}. Provider prices and terms can change."
    )
    search_text = st.text_input(
        "Search the database",
    )
    catalog_col1, catalog_col2, catalog_col3 = st.columns([1, 1, 1])
    with catalog_col1:
        job_filter = st.selectbox("Filter by job", ["All"] + sorted(tools["challenge"].unique()))
    with catalog_col2:
        business_filter = st.selectbox("Filter by business", ["All"] + BUSINESS_TYPES)
    with catalog_col3:
        free_only = st.checkbox("Show only tools with a free starting option")
    shown = tools.copy()
    if job_filter != "All":
        shown = shown[shown["challenge"] == job_filter]
    if business_filter != "All":
        shown = business_tools(shown, business_filter)
    if free_only:
        shown = shown[shown["free_price"] == 0]
    if search_text.strip():
        words = search_text.strip().lower()
        haystack = (
            shown["tool_name"].astype(str)
            + " "
            + shown["challenge"].astype(str)
            + " "
            + shown["pros"].astype(str)
            + " "
            + shown["cons"].astype(str)
            + " "
            + shown["price_note"].astype(str)
        ).str.lower()
        shown = shown[haystack.str.contains(words, regex=False)]
    st.caption(f"Showing {len(shown)} catalog tools.")
    if shown.empty:
        st.info("No catalog tools match those filters. Clear the search or a filter to see more.")
    # Cards go in rows of three
    shown_rows = list(shown.iterrows())
    for row_start in range(0, len(shown_rows), 3):
        card_columns = st.columns(3)
        for column, (card_number, (_, tool)) in zip(
            card_columns, enumerate(shown_rows[row_start:row_start + 3], start=row_start)
        ):
            with column:
                with st.container(key=f"catalog_card_{card_number}"):
                    st.markdown(f"#### {tool['tool_name']}")
                    st.write(f"**Job:** {tool['challenge']}")
                    st.write(f"**Business:** {tool['business_type']}")
                    st.write(f"**Price:** {md(tool['price_note'])}")
                    st.write(f"**Setup:** {tool['effort']}")
                    st.write(f"**What it does well:** {md(tool['pros'])}")
                    st.write(f"**What to check:** {md(tool['cons'])}")
                    show_tool_link(tool["website"])
                    st.caption(f"Price checked {date_label(tool['checked_date'])}.")
                    st.markdown(f"[Source details](https://{tool['source']})")

    # Comparison
    st.header("Compare up to three tools")
    picked = st.multiselect(
        "Choose tools to compare",
        tools["tool_name"].tolist(),
        max_selections=3,
    )
    if picked:
        compare = tools.set_index("tool_name").loc[picked].reset_index()
        compare["free_start"] = compare["free_price"].apply(
            lambda value: "Yes" if pd.notna(value) and value == 0 else "No"
        )
        compare = compare[
            ["tool_name", "challenge", "price_note", "free_start", "effort", "pros", "cons"]
        ].rename(
            columns={
                "tool_name": "Tool",
                "challenge": "Job",
                "price_note": "Price",
                "free_start": "Free starting option",
                "effort": "Setup",
                "pros": "What it does well",
                "cons": "What to check",
            }
        )
        show_table(compare)
        if len(picked) == 1:
            st.info("Pick one or two more tools to compare them side by side.")
        for name in picked:
            link_row = tools[tools["tool_name"] == name].iloc[0]
            show_tool_link(link_row["website"], name)


# Page routing. The nav bar at the top links to these pages,
# so Streamlit's own page menu is hidden.
page = st.navigation(
    [
        st.Page(home_page, title="Home", url_path="home", default=True),
        st.Page(catalog_page, title="Catalog", url_path="catalog"),
        st.Page(about_page, title="About the Website", url_path="about"),
    ],
    position="hidden",
)
page.run()