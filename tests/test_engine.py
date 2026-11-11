import csv
import unittest
from pathlib import Path

from recommendation_engine import evaluate_tool, load_plan_catalog, _plan_cost


ROOT = Path(__file__).resolve().parents[1]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"tool_name": "Test", "plan_name": "Team", "monthly_price": "40",
                     "capabilities": "crew_dispatch", "excluded_capabilities": "appointment_booking",
                     "evidence_status": "official_verified", "price_basis": "fixed"}
        self.tool = {"tool_name": "Test"}

    def evaluate(self, required, **req):
        return evaluate_tool(self.tool, [self.plan], {"budget": 100, **req}, required)

    def test_fit_classes(self):
        self.assertEqual(self.evaluate(["crew_dispatch"])["fit_classification"], "Strong Fit")
        self.assertEqual(self.evaluate(["crew_dispatch", "appointment_booking"])["fit_classification"], "Partial Fit")
        self.assertEqual(self.evaluate(["crew_dispatch", "unknown_workflow"])["fit_classification"], "Research Lead")
        self.assertEqual(self.evaluate(["crew_dispatch"], budget=20)["fit_classification"], "Ineligible")

    def test_unresolved_and_unknown_prices_are_not_confirmed_fits(self):
        self.assertEqual(self.evaluate(["crew_dispatch"], unresolved_requirements=["assign work"])["fit_classification"], "Research Lead")
        self.plan["monthly_price"] = ""
        self.assertEqual(self.evaluate(["crew_dispatch"])["status"], "unknown")

    def test_seats_addons_and_usage(self):
        self.plan.update(price_basis="per_user", per_user_cost="10", add_on_cost="5",
                         base_subscription_required="true", base_subscription_monthly_cost="20",
                         included_usage="100", usage_unit="conversations", usage_overage_cost="0.5")
        req = {"seats": 3, "monthly_usage": 120, "usage_unit": "conversations", "budget": 100}
        self.assertEqual(_plan_cost(self.plan, req)["cost"], 65)
        req.pop("seats")
        self.assertEqual(_plan_cost(self.plan, req)["status"], "unknown")

    def test_billing_and_employee_limits(self):
        self.plan.update(billing_term="annual commitment", included_employees="5")
        self.assertEqual(self.evaluate(["crew_dispatch"], monthly_billing_only=True)["status"], "ineligible")
        self.assertEqual(self.evaluate(["crew_dispatch"], employees=7)["status"], "ineligible")

    def test_real_catalog_remains_complete(self):
        plans = load_plan_catalog(ROOT / "tool_plans.csv")
        with (ROOT / "ai_tools.csv").open(encoding="utf-8-sig", newline="") as handle:
            tools = list(csv.DictReader(handle))
        self.assertEqual({p["tool_name"] for p in plans}, {t["tool_name"] for t in tools})
        for tool in tools:
            decision = evaluate_tool(tool, plans, {"budget": 100}, ["crew_dispatch"])
            if decision["fit_classification"] in {"Strong Fit", "Partial Fit"}:
                self.assertEqual(decision["status"], "confirmed")
                self.assertLessEqual(decision["cost"], 100)


if __name__ == "__main__":
    unittest.main()
