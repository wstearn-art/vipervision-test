import json
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import vee_chat
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitFlowTests(unittest.TestCase):
    @unittest.skipUnless("key" in inspect.signature(vee_chat.st.link_button).parameters,
                         "Baseline research-link controls require a newer Streamlit than 1.50")
    def test_verified_recommendations_share_comparison_and_roi_costs(self):
        def ai(system, messages, **kwargs):
            if system.startswith("Interpret the latest"):
                return json.dumps({"base_revision": 0, "clarification": None, "actions": [
                    {"op": "add", "description": "Create social media content", "capabilities": ["social_content_creation"], "importance": "essential"},
                    {"op": "set", "field": "business", "value": "Coffee Shop"},
                    {"op": "set", "field": "employees", "value": 3},
                    {"op": "set", "field": "comfort", "value": "Keep it simple"},
                    {"op": "set", "field": "budget_amount", "value": 50}]})
            return None

        with patch.object(vee_chat, "_call_ai", side_effect=ai):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
            app.chat_input[0].set_value("I run a coffee shop with three employees, need social media content, prefer simple tools and can spend $50 monthly.").run()
            self.assertEqual(list(app.exception), [])
            evaluation = app.session_state["vee_guided"]["evaluation"]
            candidates = app.session_state["vee_roi_candidates"]
            self.assertTrue(candidates)
            self.assertTrue(evaluation["ranked"])
            for candidate in candidates:
                item = next(i for i in evaluation["ranked"] if i["tool"]["tool_name"] == candidate["name"])
                self.assertEqual(candidate["cost"], item["pricing"]["cost"])
                self.assertEqual(candidate["plan"], item["pricing"]["plan"])
            self.assertTrue(app.metric)
            next(b for b in app.button if b.key == "vee_ask_compare").click().run()
            self.assertEqual(list(app.exception), [])
            comparison = app.session_state["vee_inline_followup"]["response"]
            for item in evaluation["ranked"][:3]:
                self.assertIn(item["tool"]["tool_name"], comparison)
                self.assertIn(item["fit_classification"], comparison)

    def test_multiturn_intake_correction_comparison_and_roi(self):
        turns = {
            "I run a landscaping business. I need crew coordination and more customers.": [
                {"op": "set", "field": "business", "value": "Landscaping"},
                {"op": "add", "description": "Coordinate crews", "capabilities": [], "importance": "essential"},
                {"op": "add", "description": "Acquire customers", "capabilities": ["local_customer_acquisition"], "importance": "essential"}],
            "By crew coordination I mean scheduling crews and assigning jobs.": [
                {"op": "update", "id": "r1", "description": "Schedule crews and assign jobs", "capabilities": ["crew_dispatch"], "importance": "essential"}],
            "$100 per month": [{"op": "set", "field": "budget_amount", "value": 100}],
            "Seven employees": [{"op": "set", "field": "employees", "value": 7}],
            "Keep it simple": [{"op": "set", "field": "comfort", "value": "Keep it simple"}],
            "Forget attracting customers. Keep scheduling crews and assigning jobs.": [{"op": "remove", "id": "r2"}],
            "I mean scheduling crews and assigning jobs.": [],
        }

        def ai(system, messages, **kwargs):
            if system.startswith("Interpret the latest"):
                context, _ = json.JSONDecoder().raw_decode(system.split(" PROFILE=", 1)[1])
                return json.dumps({"base_revision": context["revision"],
                                   "actions": turns[messages[-1]["content"]], "clarification": None})
            return None

        with patch.object(vee_chat, "_call_ai", side_effect=ai):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
            self.assertEqual(list(app.exception), [])
            for message in turns:
                app.chat_input[0].set_value(message).run()
                self.assertEqual(list(app.exception), [], message)
            state = app.session_state["vee_guided"]
            self.assertEqual(state["stage"], "done")
            profile = state["answers"]
            self.assertEqual(profile["employees"], 7)
            self.assertEqual(profile["budget_amount"], 100)
            self.assertEqual(len(profile["requirements"]), 1)
            self.assertEqual(profile["unresolved_requirements"], [])
            self.assertEqual(profile["revision"], 6)
            evaluation = state["evaluation"]
            self.assertEqual(evaluation["revision"], profile["revision"])
            for candidate in app.session_state["vee_roi_candidates"]:
                assessed = next(i for i in evaluation["ranked"] if i["tool"]["tool_name"] == candidate["name"])
                self.assertEqual(candidate["cost"], assessed["pricing"]["cost"])
                self.assertEqual(candidate["profile_revision"], profile["revision"])
            comparison = vee_chat._catalog_comparison(evaluation["ranked"])
            for item in evaluation["ranked"][:3]:
                self.assertIn(item["fit_classification"], comparison)
            budget_button = next(b for b in app.button if b.key == "vee_budget_apply")
            budget_input = next(n for n in app.number_input if str(n.key).startswith("vee_budget_edit_"))
            budget_input.set_value(80)
            budget_button.click().run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.session_state["vee_guided"]["answers"]["budget_amount"], 80)


if __name__ == "__main__":
    unittest.main()
