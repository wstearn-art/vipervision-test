"""Reconstructed live failures; model responses here are controlled fixtures."""
import copy
import json
import unittest
from unittest.mock import patch

import pandas as pd
import vee_chat as chat
from customer_profile import InvalidProposal, apply_proposal, ensure_profile
from recommendation_engine import evaluate_catalog, load_plan_catalog
from test_conversation import OPTIONS


def proposal(actions=(), revision=0, questions=()):
    return {"base_revision": revision, "actions": list(actions), "clarification": None,
            "questions": list(questions)}


def add(description, capability):
    return {"op": "add", "description": description, "capabilities": [capability] if capability else [],
            "importance": "essential"}


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.state = chat._state()
        ensure_profile(self.state["answers"])
        self.session = patch.object(chat.st, "session_state", {})
        self.session.start()
        self.addCleanup(self.session.stop)

    def interpret(self, message, initial, reviewed):
        with patch.object(chat, "_call_ai", side_effect=[json.dumps(initial), json.dumps(reviewed)]):
            return chat._update_profile(self.state, message, *OPTIONS)

    def test_semantic_review_removes_filler_from_valid_proposal(self):
        for text in ("Wait", "Keep the other two needs"):
            with self.subTest(text=text):
                wrong = {"base_revision": 0, "actions": [add(text, None)], "clarification": None}
                clean = {"base_revision": 0, "actions": [], "clarification": None}
                self.interpret(text, wrong, clean)
                self.assertEqual(self.state["answers"]["requirements"], [])
                self.assertEqual(self.state["answers"]["revision"], 0)

    def test_semantic_review_repairs_missed_removal(self):
        seed = {"base_revision": 0, "actions": [add("Schedule crews", "crew_dispatch"),
                add("Acquire customers", "local_customer_acquisition")], "clarification": None}
        chat._apply_profile_actions(self.state, seed, "Initial needs", *OPTIONS[:3])
        wrong = {"base_revision": 1, "actions": [], "clarification": None}
        fixed = {"base_revision": 1, "actions": [{"op": "remove", "id": "r2"}], "clarification": None}
        self.interpret("Remove customer acquisition, keep crew scheduling", wrong, fixed)
        self.assertEqual(self.state["answers"]["essential_capabilities"], ["crew_dispatch"])

    def test_failed_semantic_review_never_commits_candidate(self):
        before = copy.deepcopy(self.state["answers"])
        wrong = {"base_revision": 0, "actions": [add("Wait", None)], "clarification": None}
        with patch.object(chat, "_call_ai", side_effect=[json.dumps(wrong), None]):
            result = chat._update_profile(self.state, "Wait", *OPTIONS)
        self.assertTrue(result["clarification"])
        self.assertEqual(self.state["answers"], before)

    def test_duplicate_capability_addition_is_rejected_atomically(self):
        first = {"base_revision": 0, "actions": [add("Schedule crews", "crew_dispatch")], "clarification": None}
        chat._apply_profile_actions(self.state, first, "Crew scheduling", *OPTIONS[:3])
        duplicate = {"base_revision": 1, "actions": [add("Coordinate my field crews", "crew_dispatch")], "clarification": None}
        with self.assertRaises(InvalidProposal):
            apply_proposal(self.state["answers"], duplicate, business_types=OPTIONS[0],
                           comfort_options=OPTIONS[1], budget_cap=OPTIONS[2], message="same need")

    def test_gift_shop_intake_omissions_are_repaired(self):
        cases = [
            ("business", "I run a gift shop", "business", "gift shop"),
            ("budget", "$20 per month maximum", "budget_amount", 20),
            ("employees", "I'm the sole employee", "employees", 1),
            ("comfort", "I'm a beginner with technology", "comfort", "Keep it simple"),
        ]
        for stage, message, field, value in cases:
            self.state["stage"] = stage
            revision = self.state["answers"]["revision"]
            wrong = {"base_revision": revision, "actions": [], "clarification": None}
            right = {"base_revision": revision, "actions": [{"op": "set", "field": field, "value": value}], "clarification": None}
            self.interpret(message, wrong, right)
            self.assertEqual(self.state["answers"][field], value)
        self.assertEqual(self.state["answers"]["business"], "gift shop")

    def test_next_unanswered_revisits_missing_earlier_field(self):
        self.assertEqual(chat._next_unanswered({"employees": 1, "comfort": "Keep it simple"}), "challenge")

    def test_mixed_pricing_question_preserves_budget_and_routes_answer(self):
        self.state["answers"]["budget_amount"] = 100
        q = {"text": "Does Jobber cost less than $100 a month?", "kind": "pricing"}
        p = proposal([add("Answer missed calls", "inbound_call_answering")], questions=[q])
        result = self.interpret("Does Jobber cost less than $100? Don't change my budget, add missed-call answering.", p, p)
        self.assertEqual(self.state["answers"]["budget_amount"], 100)
        self.assertEqual(result["questions"], [q])

    def test_pricing_answer_uses_assessed_total_not_advertised_addon(self):
        tools = pd.read_csv("ai_tools.csv")
        plans = load_plan_catalog("tool_plans.csv")
        self.state["answers"].update(essential_capabilities=["inbound_call_answering"], budget_amount=100)
        items = evaluate_catalog(tools, chat._requirements_from_answers(self.state["answers"]), plans)
        self.state["evaluation"] = {"all": items}
        answer = chat._answer_questions(self.state, [{"kind": "pricing", "text": "Does Jobber cost less than $100?"}], [], tools)
        self.assertIn("not verified", answer.lower())
        self.assertIn("base", answer.lower())

    def test_roi_partial_fit_note_scopes_savings_to_covered_work(self):
        entry = {"fit_classification": "Partial Fit", "covered_capabilities": ["employee_scheduling"],
                 "missing_capabilities": ["labor_cost_management"], "plan": "Comp", "cost": 0}
        note = chat._roi_note(entry)
        self.assertIn("Partial Fit", note)
        self.assertIn("labor cost management", note)
        self.assertIn("assumption", note.lower())

    def test_landscaping_remove_add_reintroduce_and_correction(self):
        def run(actions, message):
            p = proposal(actions, self.state["answers"]["revision"])
            self.interpret(message, p, p)
        run([add("Schedule crews", "crew_dispatch"), add("Acquire customers", "local_customer_acquisition"),
             {"op": "set", "field": "employees", "value": 7},
             {"op": "set", "field": "budget_amount", "value": 100}], "Seven employees, $100 budget, crew scheduling and acquisition")
        run([{"op": "remove", "id": "r2"}], "Drop acquisition, keep scheduling crews")
        run([{"op": "update", "id": "r1", "description": "Dispatch crews and assign jobs",
              "capabilities": ["crew_dispatch"], "importance": "essential"}], "I mean assigning jobs to field crews")
        run([add("Answer missed customer calls", "inbound_call_answering")], "Also answer missed calls")
        run([add("Acquire customers", "local_customer_acquisition")], "Bring back customer acquisition")
        profile = self.state["answers"]
        self.assertEqual(profile["essential_capabilities"], ["crew_dispatch", "inbound_call_answering", "local_customer_acquisition"])
        self.assertEqual(len(profile["requirements"]), 3)
        self.assertEqual(profile["requirements"][0]["id"], "r1")
        self.assertEqual(profile["budget_amount"], 100)

    def test_coffee_inventory_forecasting_and_restaurant_budget_regressions(self):
        tools = pd.read_csv("ai_tools.csv")
        plans = load_plan_catalog("tool_plans.csv")
        def edit(actions):
            p = proposal(actions, self.state["answers"]["revision"])
            chat._apply_profile_actions(self.state, p, "Reconstructed scenario", *OPTIONS[:3])
            current = chat._current_evaluation(self.state, tools, None,
                evaluate_catalog=lambda data, req: evaluate_catalog(data, req, plans))
            self.assertEqual(current["revision"], self.state["answers"]["revision"])
            return {i["tool"]["tool_name"]: i for i in current["all"]}
        items = edit([add("Manage inventory", "inventory_management"),
                      {"op": "set", "field": "employees", "value": 10},
                      {"op": "set", "field": "budget_amount", "value": 50}])
        self.assertEqual(items["Cactus AI"]["plan"], "Starter")
        self.assertEqual(items["Cactus AI"]["fit_classification"], "Strong Fit")
        items = edit([add("Forecast inventory", "inventory_forecasting")])
        self.assertNotEqual(items["Cactus AI"]["fit_classification"], "Strong Fit")
        items = edit([{"op": "remove", "id": "r2"}])
        self.assertEqual(items["Cactus AI"]["fit_classification"], "Strong Fit")
        self.state = chat._state()
        ensure_profile(self.state["answers"])
        items = edit([add("Schedule employee shifts", "employee_scheduling"),
                      add("Manage labor costs", "labor_cost_management"),
                      {"op": "set", "field": "employees", "value": 15},
                      {"op": "set", "field": "budget_amount", "value": 100}])
        self.assertEqual(items["7shifts"]["plan"], "Pro")
        items = edit([{"op": "set", "field": "budget_amount", "value": 50}])
        self.assertEqual(items["7shifts"]["plan"], "Comp")
        self.assertEqual(items["7shifts"]["fit_classification"], "Partial Fit")
        self.assertIn("labor cost management", chat._roi_note(items["7shifts"]))
        self.assertEqual(self.state["answers"]["essential_capabilities"], ["employee_scheduling", "labor_cost_management"])

    def test_questions_are_validated_before_any_profile_write(self):
        p = proposal([add("Schedule crews", "crew_dispatch")], questions=[{"kind": "made_up", "text": "Cost?"}])
        before = copy.deepcopy(self.state["answers"])
        with self.assertRaises(InvalidProposal):
            apply_proposal(before, p, message="Bad question metadata", business_types=OPTIONS[0],
                           comfort_options=OPTIONS[1], budget_cap=OPTIONS[2])
        self.assertEqual(before, self.state["answers"])

    def test_review_can_reconcile_duplicate_addition_to_existing_id(self):
        seed = proposal([add("Schedule crews", "crew_dispatch")])
        chat._apply_profile_actions(self.state, seed, "Initial need", *OPTIONS[:3])
        wrong = proposal([add("Coordinate field crews", "crew_dispatch")], 1)
        right = proposal([], 1)
        result = self.interpret("I mean scheduling my field crews", wrong, right)
        self.assertFalse(result["changed"])
        self.assertEqual(len(self.state["answers"]["requirements"]), 1)

    def test_buffer_pricing_does_not_substitute_advertised_free_plan(self):
        tools = pd.read_csv("ai_tools.csv")
        plans = load_plan_catalog("tool_plans.csv")
        self.state["answers"].update(essential_capabilities=["social_content_creation"],
            budget_amount=100, channels=2, monthly_usage=10, usage_unit="posts_per_channel")
        items = evaluate_catalog(tools, chat._requirements_from_answers(self.state["answers"]), plans)
        self.state["evaluation"] = {"all": items}
        assessed = next(i for i in items if i["tool"]["tool_name"] == "Buffer AI Assistant")
        answer = chat._answer_questions(self.state, [{"kind": "pricing", "text": "What does Buffer cost?"}], [], tools)
        self.assertEqual(assessed["plan"], "Essentials")
        self.assertIn("Essentials", answer)
        self.assertNotIn("Free", answer)
        comparison = chat._catalog_comparison([assessed])
        self.assertIn("Current profile plan: Essentials", comparison)
        self.assertIn("may describe other plans", comparison)
        if assessed["status"] == "confirmed":
            self.assertIn(f"${assessed['cost']:,.2f}/month", answer)

    def test_budget_changes_cannot_leave_old_price_in_mapped_priority_summary(self):
        seed = proposal([add("Schedule crews with my old $100 budget", "crew_dispatch")])
        chat._apply_profile_actions(self.state, seed, "Old requirement narrative", *OPTIONS[:3])
        change = proposal([{"op": "set", "field": "budget_amount", "value": 50}], 1)
        chat._apply_profile_actions(self.state, change, "Change budget only", *OPTIONS[:3])
        self.assertEqual(self.state["answers"]["budget_amount"], 50)
        self.assertNotIn("$100", self.state["answers"]["goal"])
        self.assertEqual(self.state["answers"]["essential_capabilities"], ["crew_dispatch"])

    def test_misfiled_beginner_and_headcount_answer_preserves_business(self):
        seed = proposal([{"op": "set", "field": "business", "value": "gift shop"}])
        chat._apply_profile_actions(self.state, seed, "I own a gift shop", *OPTIONS[:3])
        wrong = proposal([{"op": "set", "field": "business", "value": "I am a beginner and sole employee"}], 1)
        right = proposal([{"op": "set", "field": "employees", "value": 1},
                          {"op": "set", "field": "comfort", "value": "Keep it simple"}], 1)
        self.interpret("I am a beginner and sole employee", wrong, right)
        self.assertEqual(self.state["answers"]["business"], "gift shop")
        self.assertEqual(self.state["answers"]["employees"], 1)


if __name__ == "__main__":
    unittest.main()
