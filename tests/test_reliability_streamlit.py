import json
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import vee_chat as chat
from customer_profile import ensure_profile


# Routing tests use no products; pricing and real catalog decisions have separate tests.
HARNESS = '''
import pandas as pd
from vee_chat import render_vee_chat
from recommendation_engine import evaluate_catalog, load_plan_catalog
plans = load_plan_catalog("tool_plans.csv")
render_vee_chat(tools=pd.read_csv("ai_tools.csv").iloc[0:0],
    business_types=["Landscaping"], comfort_options=["Keep it simple", "I can learn a bit"],
    budget_cap={"Free only":0, "Under $100 a month":100, "Over $100 a month":float("inf")},
    score_tools=None, ai_job_from_text=None, job_from_text=None, vee_avatar=lambda:"assistant",
    evaluate_catalog=lambda data, req:evaluate_catalog(data, req, plans))
'''


class ReliabilityStreamlitTests(unittest.TestCase):
    def test_mixed_question_is_answered_after_successful_edit(self):
        seed = chat._state()
        seed["stage"] = "done"
        seed["answers"].update(challenge="crew scheduling", business="Landscaping", employees=7,
                               comfort="Keep it simple", budget_amount=100)
        ensure_profile(seed["answers"])
        actions = [{"op": "remove", "id": "r1"},
                   {"op": "add", "description": "Answer missed customer calls",
                    "capabilities": ["inbound_call_answering"], "importance": "essential"}]
        question = {"kind": "pricing", "text": "Does Jobber cost less than $100?"}
        proposed = {"base_revision": 0, "actions": actions, "clarification": None, "questions": [question]}
        with patch.object(chat, "_call_ai", return_value=json.dumps(proposed)), patch.object(
                chat, "_answer_questions", return_value="The eligible Jobber total is unverified.") as answer:
            app = AppTest.from_string(HARNESS, default_timeout=20)
            app.session_state["vee_guided"] = seed
            app.run()
            app.chat_input[0].set_value("Does Jobber cost less than $100 a month? Don't change my budget, but remove crew scheduling and add answering missed customer calls instead.").run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(answer.call_args.args[1], [question])
            state = app.session_state["vee_guided"]
            self.assertEqual(state["answers"]["budget_amount"], 100)
            self.assertEqual(state["answers"]["essential_capabilities"], ["inbound_call_answering"])
            self.assertIn("unverified", state["messages"][-1]["content"])
            self.assertIn("updated", state["messages"][-1]["content"])

    def test_gift_shop_budget_and_combined_personal_details_advance_intake(self):
        turns = {
            "I run a gift shop and need inventory management": [
                {"op": "set", "field": "business", "value": "gift shop"},
                {"op": "add", "description": "Manage inventory", "capabilities": ["inventory_management"], "importance": "essential"}],
            "$20 per month maximum": [{"op": "set", "field": "budget_amount", "value": 20}],
            "I'm a beginner and the only employee": [
                {"op": "set", "field": "employees", "value": 1},
                {"op": "set", "field": "comfort", "value": "Keep it simple"}],
        }
        def ai(system, messages, **kwargs):
            if not system.startswith("Interpret the latest"):
                return None
            context, _ = json.JSONDecoder().raw_decode(system.split(" PROFILE=", 1)[1])
            return json.dumps({"base_revision": context["revision"], "actions": turns[messages[-1]["content"]],
                               "clarification": None, "questions": []})
        with patch.object(chat, "_call_ai", side_effect=ai):
            app = AppTest.from_string(HARNESS, default_timeout=20).run()
            for message in turns:
                app.chat_input[0].set_value(message).run()
                self.assertEqual(list(app.exception), [])
            state = app.session_state["vee_guided"]
            self.assertEqual(state["stage"], "done")
            self.assertEqual(state["answers"]["business"], "gift shop")
            self.assertEqual(state["answers"]["budget_amount"], 20)
            self.assertEqual(state["answers"]["employees"], 1)
            self.assertEqual(state["answers"]["comfort"], "Keep it simple")


if __name__ == "__main__":
    unittest.main()
