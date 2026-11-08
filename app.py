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
