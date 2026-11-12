import re
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from vee_chat import render_vee_chat, _current_evaluation
from advisor_context import (evidence_conflicts, display_text, roi_state, roi_calculation,
                             restore_roi_widgets, save_roi_widget, confirm_roi,
                             roi_review_key, monthly_cost_label)
from recommendation_engine import (
    load_catalog, CATALOG_CODE_REVISION,
    score_tools as engine_score_tools,
    plan_price_summary, evaluate_catalog as engine_evaluate_catalog,
)
st.set_page_config(
    page_title="VipervisionAI | Find the right AI tool",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ai_tools.csv"
PLAN_FILE = BASE_DIR / "tool_plans.csv"
VEE_FILE = BASE_DIR / "vee.png"
EXPECTED_TOOL_COUNT = 22
PRICE_CHECK_DATE = "Nov 6, 2026"  # replaced by the newest checked_date in the catalog.

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
    return load_catalog(path, PLAN_FILE)

try:
    tools, tool_plans = load_tools(DATA_FILE, (DATA_FILE.read_bytes(), PLAN_FILE.read_bytes(), date.today().isoformat(), CATALOG_CODE_REVISION))
except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
    st.error("The catalog could not be loaded. Check ai_tools.csv and tool_plans.csv and try again.")
    st.caption(str(error))
    st.stop()

newest_check = pd.to_datetime(tools["checked_date"], errors="coerce").max()
BUSINESS_TYPES = list(dict.fromkeys(tag.strip() for tags in tools['industry_tags']
                                  for tag in tags.split(';') if tag.strip()))
if pd.notna(newest_check):
    PRICE_CHECK_DATE = date_label(newest_check)

if len(tools) < EXPECTED_TOOL_COUNT:
    st.warning(
        f"The catalog currently contains {len(tools)} tools; "
        f"the original baseline contained {EXPECTED_TOOL_COUNT}."
    )

def business_tools(data, business_type):
    """Return tools whose catalog business type includes the selected type."""
    field = 'industry_tags' if 'industry_tags' in data else 'business_type'
    separator = ';' if field == 'industry_tags' else ','
    return data[data[field].apply(
        lambda value: business_type in [part.strip() for part in str(value).split(separator)])].copy()



def evaluate_catalog(data, requirements):
    return engine_evaluate_catalog(data, requirements, tool_plans)


def score_tools(data, job, cap, comfort, requirements=None):
    """Return verified affordable Strong Fit and Partial Fit candidates."""
    req = dict(requirements or {})
    req.setdefault("challenge", "")
    req.setdefault("goal", "")
    req.setdefault("employees", None)
    return engine_score_tools(data, job, cap, comfort, req, tool_plans)




@st.cache_data
def catalog_summaries(catalog_revision, _plans):
    plans = _plans
    conflicts=evidence_conflicts(plans)
    summaries={}
    for row in plans:
        summaries.setdefault(row['tool_name'],[]).append(row['plan_name']+': '+plan_price_summary(row))
    return {name:'; '.join(values)+('. '+conflicts[name] if name in conflicts else '') for name,values in summaries.items()}


_catalog_summaries = catalog_summaries(tools.attrs.get('catalog_revision'), tool_plans)


def plan_summary(tool_name):
    return _catalog_summaries.get(tool_name, '')


@st.cache_data
def run_scorer_checks(data, catalog_revision=None):
    """Check fit classes, affordability gates, ordering, and explanations."""
    total = 0
    passed = 0
    for job in JOBS:
        for cap in BUDGET_CAP.values():
            # Comfort affects conversational presentation, not deterministic fit.
            ranked = score_tools(data, job, cap, COMFORT_OPTIONS[0])
            classes = [item["fit_classification"] for item in ranked]
            ok = (
                all(label in {"Strong Fit", "Partial Fit"} for label in classes)
                and classes == sorted(classes, key=lambda label: 0 if label == "Strong Fit" else 1)
                and all(item["pricing"]["status"] == "confirmed" for item in ranked)
                and all(item["reasons"] for item in ranked)
            )
            total += 1
            passed += int(ok)
    return passed, total

def vee_avatar():
    return str(VEE_FILE) if VEE_FILE.exists() else "assistant"

def show_tool_link(url, name=None):
    """Display a readable link without exposing a raw URL as page text."""
    label = f"Visit {name} website" if name else "Visit provider website"
    st.markdown(f"[{label}]({url})")

# This is the AI integration 


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

# Keep card buttons level, even when descriptions or tool lists differ.
CARD_LAYOUT_FIX = """
<style>
/* Center Vee and the chat panel as one desktop composition. */
.st-key-vee_home_layout {
    max-width: 1120px;
    margin: 0 auto;
}
.st-key-vee_home_layout [data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
    gap: clamp(1rem, 3vw, 2.25rem);
}
.st-key-vee_home_layout .st-key-vee_image {
    margin-top: 0 !important;
    padding-top: 0 !important;
    align-self: flex-start !important;
}
.st-key-vee_home_layout .st-key-vee_image img {
    display: block;
    width: 100%;
    max-width: 320px;
    height: auto;
    margin: 0 auto;
}
.st-key-vee_home_layout .st-key-vee_chat_box {
    width: 100%;
}
@media (max-width: 768px) {
    .st-key-vee_home_layout [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    .st-key-vee_home_layout [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        min-width: 0 !important;
        flex: 1 1 100% !important;
    }
    .st-key-vee_home_layout .st-key-vee_image img {
        max-width: 220px;
    }
}
/* Align buttons and bottom stats on the three tool cards. */
:is(.st-key-explore_card_customer, .st-key-explore_card_marketing, .st-key-explore_card_bookings)
  [data-testid="stVerticalBlock"] {
    display: flex;
    flex-direction: column;
    min-height: 570px;
}
:is(.st-key-explore_card_customer, .st-key-explore_card_marketing, .st-key-explore_card_bookings)
  [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
    margin-top: auto;
}
/* Give free-resource cards the same aligned button row. */
:is(.st-key-explore_card_sba, .st-key-explore_card_score, .st-key-explore_card_partners)
  [data-testid="stVerticalBlock"] {
    display: flex;
    flex-direction: column;
    min-height: 350px;
}
:is(.st-key-explore_card_sba, .st-key-explore_card_score, .st-key-explore_card_partners)
  [data-testid="stElementContainer"]:has([data-testid="stLinkButton"]) {
    margin-top: auto;
}
/* Make the entire button label readable and consistent. */
:is(.st-key-explore_card_customer, .st-key-explore_card_marketing, .st-key-explore_card_bookings,
    .st-key-explore_card_sba, .st-key-explore_card_score, .st-key-explore_card_partners)
  :is([data-testid="stButton"], [data-testid="stLinkButton"]) :is(button, a) {
    width: 100%;
    min-height: 54px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px 14px;
    text-align: center;
    line-height: 1.3;
    white-space: normal;
    overflow-wrap: break-word;
}
/* Stacked cards do not need fixed desktop heights. */
@media (max-width: 767px) {
  [class*="st-key-explore_card_"] [data-testid="stVerticalBlock"] {
    min-height: unset;
  }
}
</style>
"""
st.markdown(CARD_LAYOUT_FIX, unsafe_allow_html=True)

st.markdown("<div id='top'></div>", unsafe_allow_html=True)

# The navigation controls are rendered below, after the pages are defined.
# Native Streamlit links keep the chat session alive between pages.

# Home page: everything except the catalog
def home_page():
    st.session_state["vee_plan_catalog"] = tool_plans
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
    with st.container(key="vee_home_layout"):
        vee_col, greet_col = st.columns([0.9, 2.1], vertical_alignment="top")
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
                evaluate_catalog=evaluate_catalog,
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
                snapshot = roi_calculation(st.session_state, recommended, vee_state["answers"])
                model = roi_state(st.session_state)
                restore_roi_widgets(st.session_state)
                chosen = st.selectbox("Tool to evaluate", price_choices,
                                      key="roi_tool_choice", on_change=save_roi_widget,
                                      args=(st.session_state, "choice"))
                selected = next((item for item in recommended if item["name"] == chosen), None)
                cost_known = bool(selected and selected["cost"] is not None)
                if cost_known:
                    tool_price = float(selected["cost"])
                    st.write("**Catalog plan cost for known requirements:** " + md(monthly_cost_label(tool_price, selected.get("billing_term", ""))))
                    if selected.get("plan"):
                        st.caption("Assessed plan: " + selected["plan"])
                    st.caption(selected["note"])
                    st.caption("Choose 'Enter my own cost' to model a different total.")
                else:
                    # Unknown is not zero. Explicit owner confirmation is necessary
                    # before calculating a net estimate, even if the entered value is $0.
                    tool_price = float(st.number_input(
                        "Estimated TOTAL monthly tool cost ($)", min_value=0.0,
                        max_value=100000.0, step=5.0,
                        key="roi_custom_cost", on_change=save_roi_widget,
                        args=(st.session_state, "custom_cost")))
                    cost_known = st.checkbox(
                        "Use this amount as my own scenario assumption",
                        key="roi_custom_confirmed", on_change=save_roi_widget,
                        args=(st.session_state, "custom_cost_confirmed"))
                    if selected:
                        st.warning(selected.get("eligibility_warning") or
                                   "Total price isn't confirmed. Include seats, usage and required subscriptions.")
                    if not cost_known:
                        st.info("Net savings and break-even will appear once you confirm an estimated total cost.")

                if snapshot["needs_assumption_review"]:
                    st.warning("Review the calculator inputs for the current plan and covered work, "
                               "then confirm these assumptions. Exclude unsupported work.")
                    review_key = roi_review_key(model)
                    st.checkbox("I reviewed these assumptions for the current plan's covered work",
                                key=review_key, on_change=confirm_roi,
                                args=(st.session_state, model["signature"], review_key))

                c1, c2 = st.columns(2)
                with c1:
                    hours = st.number_input("Hours spent on this task each week",
                                            min_value=0.0, max_value=168.0,
                                            step=0.5, key="roi_hours", on_change=save_roi_widget,
                                            args=(st.session_state, "weekly_hours"))
                    value = st.number_input("Hourly value of that work ($)",
                                            min_value=0.0, max_value=10000.0,
                                            step=5.0, key="roi_hourly", on_change=save_roi_widget,
                                            args=(st.session_state, "hourly_value"))
                with c2:
                    reduction = st.slider("Estimated time saved (%)", 0, 100,
                                          key="roi_reduction", on_change=save_roi_widget,
                                          args=(st.session_state, "time_reduction_percent"))
                    st.caption("An assumption to test—not a measured AI performance claim.")

                model["rendered"] = True
                st.session_state["vee_roi_snapshot"] = snapshot
                monthly_time_value = snapshot["monthly_time_value"]
                estimated_savings = snapshot["monthly_savings"]
                estimated_net = snapshot["monthly_net"]
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Time value/month", money(monthly_time_value))
                with m2:
                    st.metric("Potential savings/month", money(estimated_savings) if estimated_savings is not None else "Review assumptions")
                with m3:
                    st.metric("Estimated net/month", money(estimated_net) if estimated_net is not None else "Not yet calculated")
                if estimated_net is not None and tool_price > 0 and monthly_time_value > 0:
                    needed = tool_price / monthly_time_value * 100
                    if needed <= 100:
                        st.caption(f"Break-even requires about {needed:.0f}% time savings at these inputs.")
                    else:
                        st.warning("This tool's modeled cost exceeds the full monthly value of the task.")
                if estimated_net is not None and estimated_net < 0:
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
            '<p class="explore-subtitle">Quick answers about what the site does, how tool fit is classified, '
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
        with st.expander("How are tool fits classified?"):
            st.write(
                "Strong Fit means the plan has documented required capabilities, supports the "
                "workflow, and has a verified total cost within budget. Partial Fit means an "
                "affordable plan addresses a documented part of the problem, and we explain "
                "what is missing. Research Lead means important cost or capability evidence "
                "still needs checking. Known over-budget options are excluded."
            )
        with st.expander("Does an AI pick my tools?"):
            st.write(
                "No. The recommendations come from clear eligibility and fit rules, so the same answers "
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
                "Strong Fit requires documented support for all stated essential capabilities "
                "and a verified total plan cost within budget. Partial Fit requires a verified "
                "affordable plan that documents meaningful support for only part of the task; "
                "the missing capability is shown. Unknown costs or essential capabilities "
                "are Research Leads, not verified matches. Known over-budget plans are excluded. "
                "Industry labels do not affect classification; workflow capabilities do."
            )
            st.write(
                "The labels describe what the available catalog evidence supports; they are "
                "not a promise of business results. A product can appear in the catalog "
                "without qualifying as a recommendation for a particular owner."
            )
            st.write(
                "**Beyond labels:** business type helps Vee understand context and classify "
                "the workflow, but industry labels do not award points. The catalog maps "
                "capabilities to plans and evidence. It does not use customer reviews or "
                "star ratings as proof of fit."
            )
            st.write(
                "**What each question tests:** business type informs workflow context; "
                "capability evidence determines whether a plan solves the stated need. "
                "Budget is a hard eligibility limit. Technology comfort informs implementation "
                "guidance but does not change the fit classification. Team size is used when a plan "
                "has a verified employee limit."
            )
            st.write(
                "Sources and dates are shown with each catalog entry. Provider pricing "
                "should be checked again before purchase."
            )
            free_count = int(tools.apply(has_free_starting_plan, axis=1).sum())
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
                free_start=tools.apply(has_free_starting_plan, axis=1),
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
                "**Current prototype:** ai_tools.csv preserves product descriptions and sources; "
                "tool_plans.csv records plan-specific capabilities and costs. The questionnaire "
                "uses business context to identify workflow needs. Eligibility checks required "
                "features and plan cost before assigning Strong Fit, Partial Fit, or Research Lead. "
                "Known over-budget plans are excluded from recommendations."
            )
            st.graphviz_chart(
                r"""
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", fillcolor="#e8eefc",
                color="#0b1e4b", fontname="Helvetica"];
          catalog [label="ai_tools.csv\nProduct catalog"];
          plans [label="tool_plans.csv\nPlan-level evidence"];
          gate [label="Eligibility\nRequired features + total cost"];
          scorer [label="Fit classification\nStrong · Partial · Research"];
          results [label="Affordable fits\nSeparate research leads"];
          roi [label="Plan-specific pricing\nComparison and ROI"];
          catalog -> gate;
          plans -> gate;
          gate -> scorer;
          scorer -> results;
          plans -> roi;
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
            passed, total = run_scorer_checks(tools, tools.attrs.get('catalog_revision'))
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
                "These checks confirm fit classifications, affordability and capability eligibility, "
                "ranking order, and explanations across catalog jobs, budgets, and comfort levels. "
                "Business labels do not determine fit."
            )


# Catalog page: the full list of tools on its own page
def has_free_starting_plan(tool):
    """A free entry plan does not imply the required AI feature is free."""
    value = tool.get("free_price")
    return pd.notna(value) and float(value) == 0


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
        job_filter = st.selectbox("Filter by job", ["All"] + sorted({tag.strip() for tags in tools['challenge'] for tag in tags.split(';') if tag.strip()}))
    with catalog_col2:
        business_filter = st.selectbox("Filter by business", ["All"] + BUSINESS_TYPES)
    with catalog_col3:
        free_only = st.checkbox("Show only tools with a free starting option")
    shown = tools.copy()
    if job_filter != "All":
        shown = shown[shown['challenge'].apply(lambda value: job_filter in [tag.strip() for tag in value.split(';')])]
    if business_filter != "All":
        shown = business_tools(shown, business_filter)
    if free_only:
        shown = shown[shown.apply(has_free_starting_plan, axis=1)]
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
                    st.write(f"**Structured plan records:** {md(plan_summary(tool['tool_name']))}")
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
        compare["plan_tiers"] = compare["tool_name"].apply(plan_summary)
        compare["free_start"] = compare.apply(
            lambda row: "Yes (AI tier may be paid)" if has_free_starting_plan(row) else "No", axis=1
        )
        compare = compare[
            ["tool_name", "challenge", "price_note", "plan_tiers", "free_start", "effort", "pros", "cons"]
        ].rename(
            columns={
                "tool_name": "Tool",
                "challenge": "Job",
                "price_note": "Price",
                "plan_tiers": "Structured plans and known prices",
                "free_start": "Free starting option",
                "effort": "Setup",
                "pros": "What it does well",
                "cons": "What to check",
            }
        )
        state = st.session_state.get("vee_guided", {})
        if state.get("stage") == "done":
            evaluation = _current_evaluation(state, tools, evaluate_catalog)
            decisions = {str(item["tool"]["tool_name"]): item for item in evaluation["all"]}
            compare["Fit classification"] = compare["Tool"].map(lambda name: decisions[name]["fit_classification"])
            compare["Assessed plan"] = compare["Tool"].map(lambda name: decisions[name]["plan"])
            compare["Profile cost assessment"] = compare["Tool"].map(lambda name: display_text(decisions[name]["reason"]))
        show_table(compare)
        if len(picked) == 1:
            st.info("Pick one or two more tools to compare them side by side.")
        for name in picked:
            link_row = tools[tools["tool_name"] == name].iloc[0]
            show_tool_link(link_row["website"], name)


# Give the Creators link a real page instead of sending visitors to Home.
def creators_page():
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


# Build the pages once so the links and the router use the same page objects.
home_ref = st.Page(home_page, title="Home", url_path="home", default=True)
catalog_ref = st.Page(catalog_page, title="Catalog", url_path="catalog")
about_ref = st.Page(about_page, title="About the Website", url_path="about")
creators_ref = st.Page(creators_page, title="About the Creators", url_path="creators")
# Register pages before rendering links to them.
page = st.navigation([home_ref, catalog_ref, about_ref, creators_ref], position="hidden")

# Native page links navigate without a full browser reload. This preserves Vee's chat.
st.markdown("""
<style>
/* Keep all navigation links readable and styled the same way. */
[data-testid="stPageLink"] { margin-top: 0.15rem; }
[data-testid="stPageLink"] a {
    display: inline-flex;
    align-items: center;
    border: 1px solid transparent;
    border-radius: 0.55rem;
    padding: 0.35rem 0.65rem;
    color: #e6f3fa !important;
    background: transparent !important;
    text-decoration: none !important;
    font-weight: 650;
}
[data-testid="stPageLink"] a:hover,
[data-testid="stPageLink"] a:focus-visible {
    background: #263d4a;
    color: #ffffff !important;
}
/* Mark the current page without changing the chat session. */
[data-testid="stPageLink"] a[aria-current="page"] {
    background: #263d4a;
    color: #ffffff !important;
}
[data-testid="stPageLink"] a p,
[data-testid="stPageLink"] a span {
    color: inherit !important;
    background: transparent !important;
}
/* Remove the unintended muted/gray link background. */
[data-testid="stPageLink"] [data-testid="stMarkdownContainer"] {
    background: transparent !important;
}
/* Keep the header usable on narrow mobile screens. */
@media (max-width: 780px) {
    [data-testid="stPageLink"] a { padding: 0.3rem 0.4rem; font-size: 0.85rem; }
}
</style>
""", unsafe_allow_html=True)
logo_col, home_col, catalog_col, about_col, creators_col = st.columns([3, 1, 1, 1.8, 1.8], vertical_alignment="center")
with logo_col:
    st.markdown("**VipervisionAI**")
with home_col:
    st.page_link(home_ref, label="Home")
with catalog_col:
    st.page_link(catalog_ref, label="Catalog")
with about_col:
    st.page_link(about_ref, label="About the Website")
with creators_col:
    st.page_link(creators_ref, label="About the Creators")

# Display the selected page after the navigation bar.
page.run()
