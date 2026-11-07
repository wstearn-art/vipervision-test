import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import requests
st.set_page_config(
    page_title="VipervisionAI | Find the right AI tool",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ai_tools.csv"
VEE_FILE = BASE_DIR / "vee.png"
EXPECTED_TOOL_COUNT = 12
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
]

# Plain words that map typed text in the chat helper to one catalog job.
JOB_WORDS = {
    "Reviews": ["review", "reviews", "rating", "ratings", "reputation", "stars"],
    "Customer Questions": [
        "question", "questions", "inbox", "message", "messages", "texts",
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
}

BUSINESS_TYPES = ["Restaurant / Cafe", "Coffee Shop"]
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
def load_tools(path):
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
    tools = load_tools(DATA_FILE)
except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
    st.error("The tool catalog could not be loaded. Check ai_tools.csv and try again.")
    st.caption(str(error))
    st.stop()

newest_check = pd.to_datetime(tools["checked_date"], errors="coerce").max()
if pd.notna(newest_check):
    PRICE_CHECK_DATE = date_label(newest_check)

if len(tools) != EXPECTED_TOOL_COUNT:
    st.warning(
        f"The catalog currently contains {len(tools)} tools. "
        f"The contest demonstration expects {EXPECTED_TOOL_COUNT}."
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

def score_tools(data, job, cap, comfort):
    """Explainable rule based score: job fit 3, price fit 2, effort fit 1."""
    scored = []
    for _, row in data.iterrows():
        if row["challenge"] != job:
            continue
        score = 3
        breakdown = {
            "Job fit": 3,
            "Price fit": 0,
            "Effort fit": 0,
        }
        reasons = [f"Matches your job: {job}"]
        if price_fits(row, cap):
            score += 2
            breakdown["Price fit"] = 2
            if pd.notna(row["free_price"]) and row["free_price"] == 0:
                reasons.append("Has a free starting option")
            else:
                reasons.append(
                    f"Published monthly price is ${float(row['monthly_cost']):,.2f}"
                )
        elif pd.isna(row["monthly_cost"]):
            reasons.append("Price is quote only, so budget fit cannot be confirmed")
        else:
            reasons.append("Published starting price is above your selected budget")
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
        scored.append(
            {
                "score": score,
                "tool": row,
                "reasons": reasons,
                "breakdown": breakdown,
            }
        )
    # Stable alphabetical order makes equal scores reproducible.
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
        st.session_state["explore_job"] = card["quiz_job"]
        st.session_state["quiz_job_select"] = card["quiz_job"]
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
        <a href="./#find-my-tools" target="_self">Find my tools</a>
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
        # Dark blue box with a green border around Vee's introduction
        with st.container(key="vee_intro"):
            with st.chat_message("assistant", avatar=vee_avatar()):
                st.write(
                    "Hi, I am Vee! Tell me your top business goal, "
                    "and I will help you compare up to three real AI tools."
                )
            # Gold section for the intro sentence
            with st.container(key="vee_gold_answer"):
                st.write(
                    "Answer five questions to see a ranked shortlist with a clear score "
                    "breakdown, real catalog prices, and practical next steps."
                )
            # Gold section for How it works
            with st.container(key="vee_gold_how"):
                st.markdown(
                    "**How it works:**\n\n"
                    "1. Answer five questions.\n"
                    "2. Read your top three tools and why they matched.\n"
                    "3. Use the first week checklist, compare tools, and test the ROI calculator."
                )
            st.link_button("Start the five questions", "#find-my-tools", type="primary")

    # Divider between Vee's introduction and the explore section
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
        st.caption(
            f"Counts come from our tool catalog. Prices checked {PRICE_CHECK_DATE}."
        )

    # Divider between the explore section and Find my tools
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Recommendation questionnaire
    # Brown box around the whole Find my tools section
    with st.container(key="find_tools_section"):
        st.header("Find my tools")
        st.write(
            "Choose the answers that best describe your business. "
            "The score is a rule based fit guide, not a customer rating."
        )

        with st.form("recommendation_form"):
            with st.container(key="question_card_business"):
                st.write("**1. What type of business do you operate?**")
                business_type = st.selectbox(
                    "Business type",
                    BUSINESS_TYPES,
                    label_visibility="collapsed",
                )
            with st.container(key="question_card_challenge"):
                st.write("**2. What is your biggest business challenge?**")
                job = st.selectbox("Business challenge", JOBS, key="quiz_job_select", label_visibility="collapsed")
            with st.container(key="question_card_budget"):
                st.write("**3. What is your budget for AI tools?**")
                budget_label = st.selectbox(
                    "Monthly budget",
                    BUDGET_OPTIONS,
                    label_visibility="collapsed",
                )
            with st.container(key="question_card_employees"):
                st.write("**4. How many employees do you have?**")
                employees = st.slider(
                    "Number of employees",
                    1,
                    50,
                    1,
                    label_visibility="collapsed",
                )
            with st.container(key="question_card_tech"):
                st.write("**5. How comfortable are you with technology?**")
                comfort = st.selectbox(
                    "Technology comfort",
                    COMFORT_OPTIONS,
                    label_visibility="collapsed",
                )
            systems = st.text_input(
                "Optional: systems you already use, such as a point of sale or a spreadsheet",
                help="This is added to your first week checklist. It does not change the score.",
            )
            submitted = st.form_submit_button("Find my tools", type="primary")

    if submitted:
        st.session_state["recommendation_results"] = {
            "business_type": business_type,
            "job": job,
            "budget_label": budget_label,
            "employees": employees,
            "comfort": comfort,
            "systems": systems,
        }

    if "recommendation_results" in st.session_state:
        result = st.session_state["recommendation_results"]
        pool = business_tools(tools, result["business_type"])
        ranked = score_tools(
            pool,
            result["job"],
            BUDGET_CAP[result["budget_label"]],
            result["comfort"],
        )
        st.subheader("Your best matching tools")
        st.write(
            f"Matched to {result['business_type']} and {result['job']}, "
            f"for a team of {result['employees']}."
        )
        st.info(
            "How the score works: job fit is worth 3 points, price fit is worth "
            "2 points, and effort fit is worth 1 point. Equal scores are listed "
            "alphabetically. A score is not a review or a guarantee of results."
        )
        if not ranked:
            st.warning(
                "No catalog tools are listed for this job and business type. "
                "Try another job to explore the current catalog."
            )
        else:
            if len(ranked) < 3:
                st.caption(
                    f"The catalog has {len(ranked)} tool"
                    f"{'' if len(ranked) == 1 else 's'} for this job and business type. "
                    "We do not fill the list with tools for other jobs."
                )
            for index, item in enumerate(ranked[:3], start=1):
                with st.container(border=True):
                    show_tool_result(item, index, result["employees"])
            st.download_button(
                "Download your shortlist",
                data=shortlist_csv(ranked),
                file_name="vipervision_shortlist.csv",
                mime="text/csv",
            )
            show_first_week_plan(ranked[0]["tool"], result.get("systems", ""))
            show_starter_stack(
                pool, result["budget_label"], result["comfort"], result["job"]
            )
        st.button("Clear my results", on_click=clear_results)

    # The ROI calculator only shows up after someone clicks Find my tools
    if "recommendation_results" in st.session_state:
        # Return on investment calculator
        st.header("Is it worth it? ROI calculator")
        st.write(
            "Use your own estimates to explore a possible return. "
            "The time reduction is an assumption, not a measured result."
        )
        st.selectbox(
            "Optional: fill the tool price from a catalog tool",
            ["My own number"] + tools["tool_name"].tolist(),
            key="roi_tool",
            on_change=fill_roi_price,
        )
        roi_choice = st.session_state.get("roi_tool", "My own number")
        if roi_choice != "My own number":
            roi_row = tools[tools["tool_name"] == roi_choice].iloc[0]
            if pd.isna(roi_row["monthly_cost"]):
                st.caption(f"{roi_choice} is quote only, so enter your own price below.")
            elif pd.notna(roi_row["free_price"]) and roi_row["free_price"] == 0:
                st.caption(
                    f"The slider uses the paid plan price for {roi_choice}. It also has a "
                    "free starting option, so set the slider to 0 to test the free plan."
                )
        hours = st.slider("Hours per week spent on this task", 0, 40, 5)
        value = st.slider("Value of your time, dollars per hour", 10, 100, 25)
        reduction = st.slider("Assumed time reduction, percent", 0, 80, 30)
        if "roi_price" not in st.session_state:
            st.session_state["roi_price"] = 0
        tool_price = st.slider("Tool price, dollars per month", 0, 500, key="roi_price")
        monthly_time_value = hours * value * 4.33
        estimated_savings = monthly_time_value * reduction / 100
        estimated_net = estimated_savings - tool_price
        roi1, roi2, roi3 = st.columns(3)
        with roi1:
            st.metric("Monthly time value", money(monthly_time_value))
        with roi2:
            st.metric("Estimated monthly savings", money(estimated_savings))
        with roi3:
            st.metric("Estimated net monthly benefit", money(estimated_net))
        if tool_price > 0 and monthly_time_value > 0:
            needed = tool_price / monthly_time_value * 100
            if needed <= 100:
                st.write(
                    f"To cover a {md(f'${tool_price}')} monthly price, the tool would need "
                    f"to cut about {needed:.0f} percent of the time spent on this task."
                )
            else:
                st.write(
                    "At these inputs the tool costs more than all of the time spent on "
                    "this task is worth."
                )
        if estimated_net < 0:
            st.warning("With these inputs the tool costs more than the time it saves.")
        st.caption(
            "This estimate uses 4.33 weeks in a month. It excludes setup time, taxes, fees, "
            "and changes in demand. Use the provider price and your own measured results "
            "before making a decision."
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
                "VipervisionAI helps restaurants and coffee shops discover, compare, "
                "and explore adoption steps for AI tools. It includes a searchable catalog, "
                "a transparent rule based fit recommender, comparison, and an ROI estimate. "
                "It is not an AI picker and does not claim to deploy tools."
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
                "It helps restaurants and coffee shops discover, compare, and plan how to "
                "adopt AI tools. You answer five questions, and the site shows your best "
                "matching tools from our catalog with the reasons they matched."
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
                "The current catalog covers restaurants and coffee shops. Our sixteen week "
                "plan would add more business types, such as HVAC and landscaping companies."
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Title for the competition details below
    st.markdown(
        '<div class="about-section-title">Information about the competition</div>',
        unsafe_allow_html=True,
    )

    # Your original dropdowns, styled the same as the questions above
    with st.container(key="about_faq_more"):
        # Chat helper uses the same deterministic scoring rules
        with st.expander("Ask Vee in chat"):
            st.write(
                "I use the same catalog and scoring rules as the questionnaire. "
                "I do not invent tools, prices, or reviews."
            )
            chat_business = st.selectbox(
                "What type of business do you run?",
                BUSINESS_TYPES,
                key="chat_business",
            )
            chat_job = st.selectbox("What is your top job?", JOBS, key="chat_job")
            chat_text = st.text_input(
                "Or type what you need help with (optional)",
                key="chat_text",
                placeholder="For example: I need help with staff schedules",
            )
            chat_budget = st.selectbox(
                "What is your budget?",
                BUDGET_OPTIONS,
                key="chat_budget",
            )
            chat_comfort = st.selectbox(
                "How comfortable are you with technology?",
                COMFORT_OPTIONS,
                key="chat_comfort",
            )
            if st.button("Ask Vee", key="ask_vee"):
                typed = bool(chat_text.strip())
                typed_job = ((ai_job_from_text(chat_text) or job_from_text(chat_text))if typed else None)
                st.session_state["chat_answer"] = {
                    "business": chat_business,
                    "job": typed_job or chat_job,
                    "budget": chat_budget,
                    "comfort": chat_comfort,
                    "text": chat_text.strip(),
                    "typed": typed,
                    "understood": typed_job is not None,
                }
            if "chat_answer" in st.session_state:
                show_chat_answer(st.session_state["chat_answer"])

        # Research and methodology
        with st.expander("Research and scoring method"):
            st.write(
                "The current catalog contains the tools listed above for restaurants "
                "and coffee shops. Each entry includes a job category, business type, "
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
                "breakdown adds up to the score. They run for both business types, all "
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
        "Search the catalog",
        placeholder="For example: schedule, reviews, Square",
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
    for _, tool in shown.iterrows():
        with st.container(border=True):
            st.markdown(f"#### {tool['tool_name']}")
            st.write(f"**Job:** {tool['challenge']}  |  **Business:** {tool['business_type']}")
            st.write(f"**Price:** {md(tool['price_note'])}  |  **Setup:** {tool['effort']}")
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
