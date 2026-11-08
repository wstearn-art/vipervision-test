
import re
import streamlit as st

PROMPTS = {
    "name": "Hi! I'm Vee, your AI business assistant. I'm here to help you find tools that make running your business easier. What's your name?",
    "business": "Nice to meet you, {name}! What type of business do you own?",
    "challenge": "What's the biggest challenge you're facing with your business?",
    "budget": "What's your monthly budget for an AI solution? (For example, free only or around $100.)",
    "comfort": "How comfortable are you with technology? Would you prefer something simple, or are you comfortable learning a new system?",
    "employees": "How many employees do you have, including yourself?",
}
STEPS = ["name", "business", "challenge", "budget", "comfort", "employees", "done"]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}


def _budget_label(message, budget_cap):
    text = message.lower().strip()
    if re.search(r"\b(free|nothing|zero|no budget)\b", text) or text == "$0":
        return "Free only" if "Free only" in budget_cap else None
    match = re.search(r"\$?\s*(\d+(?:\.\d+)?)", text.replace(",", ""))
    if not match:
        return None
    amount = float(match.group(1))
    # Choose a tier with a ceiling at least as high as the stated budget.
    finite = sorted(((float(cap), label) for label, cap in budget_cap.items()
                     if float(cap) != float("inf") and float(cap) > 0))
    for cap, label in finite:
        if amount <= cap:
            return label
    return next((label for label, cap in budget_cap.items()
                 if float(cap) == float("inf")), None)


def _employees(message):
    found = re.search(r"\b(\d+)\b", message)
    if found:
        n = int(found.group(1))
        return n if 1 <= n <= 100000 else None
    for word, n in NUMBER_WORDS.items():
        if re.search(r"\b" + word + r"\b", message.lower()):
            return n
    if re.search(r"\b(just me|only me|myself|solo)\b", message.lower()):
        return 1
    return None


def _business_type(message, available):
    """Match known catalog industries; retain other businesses without inventing matches."""
    lower = message.casefold()
    for option in available:
        if option.casefold() in lower:
            return option
    if "coffee" in lower and any("coffee" in o.lower() for o in available):
        return next(o for o in available if "coffee" in o.lower())
    if any(term in lower for term in ("restaurant", "cafe", "café", "diner", "bistro", "food truck")):
        return next((o for o in available if "restaurant" in o.lower() or "cafe" in o.lower()), None)
    return None


def _tech_comfort(message, options):
    lower = message.lower()
    comfortable = any(word in lower for word in (
        "comfortable", "can learn", "willing to learn", "tech savvy", "experienced", "pretty good", "intermediate"))
    simple = any(word in lower for word in (
        "simple", "not good", "beginner", "not comfortable", "not tech", "easy", "basic", "struggle"))
    if "not comfortable" in lower or "not tech" in lower:
        comfortable = False
    if simple and not comfortable:
        return "Keep it simple" if "Keep it simple" in options else options[0]
    if comfortable:
        return "I can learn a bit" if "I can learn a bit" in options else options[-1]
    return None


def _initial_state():
    return {"stage": "name", "answers": {},
            "messages": [{"role": "assistant", "content": PROMPTS["name"]}],
            "classification_source": None}


def render_vee_chat(
    *, tools, business_types, budget_cap, comfort_options,
    business_tools, score_tools, show_tool_result,
    ai_job_from_text, job_from_text, vee_avatar,
):
    """Render a one-question-at-a-time Vee chat in an existing Streamlit column."""
    if "vee_guided" not in st.session_state:
        st.session_state.vee_guided = _initial_state()
    state = st.session_state.vee_guided

    st.subheader("🐍 Chat with Vee")
    st.caption("Tell me about your business. I'll recommend researched AI tools—not invent new ones.")

    for message in state["messages"]:
        with st.chat_message(message["role"], avatar=vee_avatar() if message["role"] == "assistant" else None):
            st.write(message["content"])

    if state["stage"] == "done":
        a = state["answers"]
        st.divider()
        st.markdown(f"### Recommendations for {a['name']}")
        if a.get("catalog_business") is None:
            st.info(
                f"You told me you operate a {a['business']}. "
                "This industry isn't in our verified catalog yet, so I won't invent matches. "
                "We're expanding industry coverage."
            )
        elif a.get("job") is None:
            st.info("I couldn't confidently map your challenge to a catalog category yet. "
                    "Try rephrasing it and restarting the chat.")
        else:
            pool = business_tools(tools, a["catalog_business"])
            ranked = score_tools(pool, a["job"], budget_cap[a["budget"]], a["comfort"])
            st.caption(f"{a['catalog_business']} · {a['job']} · {a['budget']} · {a['employees']} employees")
            st.caption("Challenge interpreted using " + (
                "OpenRouter AI." if state["classification_source"] == "OpenRouter" else "keyword matching."))
            if not ranked:
                st.info("No researched tools meet both your challenge and budget. "
                        "You can adjust your budget or try another challenge. No paid tools were substituted.")
            else:
                st.write("These are the strongest catalog matches, ranked by fit:")
                for index, item in enumerate(ranked[:3], start=1):
                    with st.container(border=True):
                        show_tool_result(item, index, a["employees"])
        if st.button("Start a new conversation", key="vee_restart"):
            st.session_state.vee_guided = _initial_state()
            st.rerun()
        return

    reply = st.chat_input("Type your reply to Vee...", key="vee_guided_input")
    if not reply:
        return
    reply = reply.strip()
    if not reply:
        return
    stage = state["stage"]
    state["messages"].append({"role": "user", "content": reply})
    a = state["answers"]
    error = None

    if stage == "name":
        if len(reply) > 60:
            error = "What should I call you? A first name is fine."
        else:
            a["name"] = reply
    elif stage == "business":
        a["business"] = reply
        a["catalog_business"] = _business_type(reply, business_types)
    elif stage == "challenge":
        a["challenge"] = reply
        with st.spinner("Vee is understanding your challenge..."):
            category = ai_job_from_text(reply)
        state["classification_source"] = "OpenRouter" if category else "Keyword fallback"
        a["job"] = category or job_from_text(reply)
    elif stage == "budget":
        a["budget"] = _budget_label(reply, budget_cap)
        if a["budget"] is None:
            error = "Could you give me a monthly amount, like $50 or $100, or say 'free only'?"
    elif stage == "comfort":
        a["comfort"] = _tech_comfort(reply, comfort_options)
        if a["comfort"] is None:
            error = "Would you prefer something simple, or are you comfortable learning a new system?"
    elif stage == "employees":
        a["employees"] = _employees(reply)
        if a["employees"] is None:
            error = "About how many employees do you have? You can answer with a number, like 5."

    if error:
        state["messages"].append({"role": "assistant", "content": error})
    else:
        next_stage = STEPS[STEPS.index(stage) + 1]
        state["stage"] = next_stage
        if next_stage == "done":
            state["messages"].append({"role": "assistant", "content":
                f"Thanks, {a['name']}! I've got what I need. Here are the results from our researched catalog."})
        else:
            prompt = PROMPTS[next_stage].format(name=a.get("name", "there"))
            state["messages"].append({"role": "assistant", "content": prompt})
    st.rerun()
