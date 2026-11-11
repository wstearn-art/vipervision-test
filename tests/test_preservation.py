"""Preservation checks pinned to baseline 8e170f9."""
import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BLOBS = {
    "app.py": "a3189123d7a69615ee35c45cbc7a7f69af1f973f",
    "recommendation_engine.py": "872a98b4b8b8985e98e47386dd432e3011cd2aa4",
    "style.css": "3f84b6a26b76bf9907a6f96f7cd7df71f6a62758",
    "ai_tools.csv": "3a50b107fed865f321b60aeb41d1375fe316a78b",
    "tool_plans.csv": "c5d6eea6bc70b3f70f58da8eaf669ed44be93654",
    "requirements.txt": "b1e6601df0f15630a653f783fe7401fd96351cac"
}
UI_FUNCTIONS = {
    "_style_vee": "1a5d842f9a5b4fd266b1f58d5a3b96320d08384fc2d80a870dbbcc427119abf9",
    "_result_card": "e9ec2fee8e28d0e7c0f567c6af5522624bd037ddc03552de0d133d7bec023d9f",
    "_tool_followup_actions": "d91f2b29bae347b234642c0ed716acccf9ae81f0d86eb63c02e5c411ff45c3f4",
    "_compare_followup_actions": "3584bcf08fb9c4d6aae5784fe5cd93933e1774d3a9ce1c8dc83a10d55c77b86b"
}


class PreservationTests(unittest.TestCase):
    def test_engine_catalog_app_and_css_unchanged(self):
        for path, expected in BLOBS.items():
            with self.subTest(path=path):
                data = (ROOT / path).read_bytes()
                actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                self.assertEqual(actual, expected)

    def test_cards_comparison_and_chat_styles_unchanged(self):
        tree = ast.parse((ROOT / "vee_chat.py").read_text(encoding="utf-8"))
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name, expected in UI_FUNCTIONS.items():
            with self.subTest(name=name):
                actual = hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()
                self.assertEqual(actual, expected)

    def test_chat_controls_and_presentation_calls_unchanged(self):
        tree = ast.parse((ROOT / "vee_chat.py").read_text(encoding="utf-8"))
        render = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_vee_chat")
        calls = sorted(ast.dump(n, include_attributes=False) for n in ast.walk(render)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and isinstance(n.func.value, ast.Name) and n.func.value.id == "st")
        actual = hashlib.sha256(json.dumps(calls).encode()).hexdigest()
        self.assertEqual(actual, "4b57f1579b8e46f714a1d53fb1439a56e22d41aff3530f6b7a8432bd6228c7d0")

