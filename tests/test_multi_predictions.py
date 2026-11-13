"""Jalankan: python -m unittest discover -s tests -v"""
import base64
import importlib.util
import io
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.multi_predictions import predict_upload

spec = importlib.util.spec_from_file_location("final_dashboard", ROOT / "sesi8/app_final_dashboard.py")
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def upload(frame):
    encoded = base64.b64encode(frame.to_csv(index=False).encode()).decode()
    return "data:text/csv;base64," + encoded


class MultiPredictionTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(ROOT / "data/new_customers.csv").head(3)

    def test_initial_page_has_no_results(self):
        _, output, records, disabled = dashboard.update_multi(None, "gradient_boosting", None)
        self.assertEqual(output, [])
        self.assertIsNone(records)
        self.assertTrue(disabled)

    def test_only_uploaded_customers_and_download(self):
        _, output, records, disabled = dashboard.update_multi(upload(self.data), "gradient_boosting", "three.csv")
        self.assertFalse(disabled)
        self.assertEqual(len(records), 3)
        self.assertEqual([r["customer_id"] for r in records], self.data.customer_id.tolist())
        self.assertEqual(output[0].children[0].children[1].children, "3")
        pie = output[1].children[0].children[0].figure
        self.assertEqual(sum(pie.data[0].values), 3)
        result = dashboard.download_multi(1, records)
        downloaded = pd.read_csv(io.StringIO(result["content"]))
        self.assertEqual(len(downloaded), 3)
        self.assertTrue({"churn_probability", "predicted_churn", "risk_category"} <= set(downloaded))

    def test_invalid_category_clears_previous_results(self):
        self.data.loc[0, "contract_type"] = "Mingguan"
        message, output, records, disabled = dashboard.update_multi(upload(self.data), "gradient_boosting", "bad.csv")
        self.assertIn("Mingguan", str(message))
        self.assertEqual(output, [])
        self.assertIsNone(records)
        self.assertTrue(disabled)

    def test_missing_column(self):
        _, errors = predict_upload(upload(self.data.drop(columns="region")), "bad.csv", "gradient_boosting")
        self.assertIn("region", errors[0])

    def test_empty_corrupt_and_wrong_file_type(self):
        for contents, name in [(upload(self.data.head(0)), "empty.csv"),
                               ("data:text/csv;base64,!!!", "bad.csv"),
                               (upload(self.data), "bad.xlsx")]:
            result, errors = predict_upload(contents, name, "gradient_boosting")
            self.assertIsNone(result)
            self.assertTrue(errors)

    def test_invalid_numeric_and_missing_category(self):
        self.data["monthly_charges"] = [float("inf"), None, 450000]
        self.data.loc[0, "region"] = None
        _, errors = predict_upload(upload(self.data), "bad.csv", "gradient_boosting")
        self.assertTrue(any("monthly_charges" in e for e in errors))
        self.assertTrue(any("region" in e for e in errors))

    def test_every_registered_model_predicts(self):
        self.assertEqual(list(dashboard.MODEL_REGISTRY), ["logreg", "logreg_ros", "gradient_boosting"])
        self.assertEqual(dashboard.default_model(), "logreg")
        for key in dashboard.MODEL_REGISTRY:
            with self.subTest(model=key):
                result, errors = predict_upload(upload(self.data), "three.csv", key)
                self.assertEqual(errors, [])
                self.assertTrue(result.churn_probability.between(0, 1).all())
                # metrik yang ditampilkan harus milik model ini, bukan dari metadata bersama
                _, meta = dashboard.get_model(key)
                self.assertEqual(meta["holdout_metrics"],
                                 dashboard.MODEL_REGISTRY[key]["holdout_metrics"])

    def test_dash_layout_and_callback_endpoint(self):
        client = dashboard.server.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        layout = client.get("/_dash-layout")
        self.assertEqual(layout.status_code, 200)
        self.assertIn(b"multi-upload", layout.data)
        key = next(k for k in dashboard.app.callback_map if "multi-message" in k)
        response = client.post("/_dash-update-component", json={
            "output": key,
            "outputs": [{"id": "multi-message", "property": "children"},
                        {"id": "multi-output", "property": "children"},
                        {"id": "multi-results", "property": "data"},
                        {"id": "multi-download-button", "property": "disabled"}],
            "inputs": [{"id": "multi-upload", "property": "contents", "value": upload(self.data)},
                       {"id": "multi-model", "property": "value", "value": "gradient_boosting"}],
            "state": [{"id": "multi-upload", "property": "filename", "value": "three.csv"}],
            "changedPropIds": ["multi-upload.contents"],
        })
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(len(response.json["response"]["multi-results"]["data"]), 3)


if __name__ == "__main__":
    unittest.main()
