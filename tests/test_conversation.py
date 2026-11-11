import copy
import json
import unittest
from unittest.mock import patch

import vee_chat as chat


OPTIONS = (["Restaurant / Cafe", "Coffee Shop", "Landscaping"],
           ["Keep it simple", "I can learn a bit"],
           {"Free only": 0, "Under $100 a month": 100, "Over $100 a month": float("inf")},
           lambda text: None, lambda text: None)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.state = chat._state()

    def turn(self, message, actions=(), clarification=None):
        proposal = {"base_revision": self.state["answers"].get("revision", 0),
                    "actions": list(actions), "clarification": clarification}
        with patch.object(chat, "_call_ai", return_value=json.dumps(proposal)):
            return chat._update_profile(self.state, message, *OPTIONS)

    def add(self, description, capabilities=()):
        self.turn("I need " + description, [{"op": "add", "description": description,
                  "capabilities": list(capabilities), "importance": "essential"}])
        return self.state["answers"]["requirements"][-1]["id"]

    def test_landscaping_corrections_do_not_accumulate(self):
        self.turn("I run a landscaping business with seven employees", [
            {"op": "set", "field": "business", "value": "landscaping business"},
            {"op": "set", "field": "employees", "value": 7}])
        crew = self.add("coordinate crews")
        acquisition = self.add("attract customers", ["local_customer_acquisition"])
        self.state["stage"] = "done"
        actions = [{"op": "remove", "id": acquisition},
                   {"op": "update", "id": crew, "description": "Schedule crews and assign jobs",
                    "capabilities": ["crew_dispatch"], "importance": "essential"}]
        self.turn("Forget customer acquisition. Crew coordination means scheduling crews and assigning jobs.", actions)
        for message in ("I mean scheduling crews and assigning jobs.",
                        "I need help assigning jobs to crews.",
                        "I need help allocating jobs to my team."):
            self.turn(message, [actions[1]])
        profile = self.state["answers"]
        self.assertEqual(len(profile["requirements"]), 1)
        self.assertEqual(profile["requirements"][0]["id"], crew)
        self.assertEqual(profile["unresolved_requirements"], [])
        self.assertEqual(profile["essential_capabilities"], ["crew_dispatch"])
        self.assertEqual(profile["employees"], 7)
        self.assertEqual(profile["revision"], 4)

    def test_original_fragment_failure(self):
        self.turn("I need help with crew coordination and customer acquisition.", [
            {"op": "add", "description": "Coordinate crews", "capabilities": [], "importance": "essential"},
            {"op": "add", "description": "Acquire customers", "capabilities": ["local_customer_acquisition"], "importance": "essential"}])
        self.state["stage"] = "done"
        self.turn("Forget customer acquisition. Crew coordination means scheduling crews and assigning jobs.", [
            {"op": "remove", "id": "r2"},
            {"op": "update", "id": "r1", "description": "Schedule crews and assign jobs",
             "capabilities": ["crew_dispatch"], "importance": "essential"}])
        self.assertEqual(self.state["answers"]["unresolved_requirements"], [])

    def test_confirmation_and_question_preserve_profile(self):
        self.add("schedule crews", ["crew_dispatch"])
        before = copy.deepcopy(self.state["answers"])
        for message in ("yes", "thanks", "Can I include more needs?", "Does this cost $50?"):
            self.assertFalse(self.turn(message)["changed"])
        self.assertEqual(self.state["answers"], before)

    def test_ambiguous_turn_does_not_partially_apply(self):
        self.add("coordinate crews")
        before = copy.deepcopy(self.state["answers"])
        result = self.turn("Change that and my budget", [], "Which need and monthly limit do you mean?")
        self.assertTrue(result["clarification"])
        self.assertEqual(self.state["answers"], before)

    def test_mixed_question_and_instruction(self):
        self.add("schedule crews", ["crew_dispatch"])
        self.turn("What is a paid seat? Also change my budget to $75", [
            {"op": "set", "field": "budget_amount", "value": 75}])
        self.assertEqual(self.state["answers"]["budget_amount"], 75)
        self.assertEqual(len(self.state["answers"]["requirements"]), 1)

    def test_replacement_preserves_unrelated_need(self):
        crew = self.add("schedule crews", ["crew_dispatch"])
        self.add("reply to reviews", ["review_response_drafting"])
        self.turn("Replace crew scheduling with appointment booking", [
            {"op": "update", "id": crew, "description": "Book appointments",
             "capabilities": ["appointment_booking"], "importance": "essential"}])
        self.assertEqual(self.state["answers"]["essential_capabilities"],
                         ["appointment_booking", "review_response_drafting"])


if __name__ == "__main__":
    unittest.main()
