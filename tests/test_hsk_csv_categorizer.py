import unittest

from hsk_csv_categorizer import (
    Prediction,
    apply_predictions_to_rows,
    format_subcategory_cell,
    repair_category_subcategory_link,
    validate_prediction,
)


class HskCsvCategorizerTests(unittest.TestCase):
    def test_validate_prediction_rejects_unknown_category(self) -> None:
        prediction = Prediction(
            row_id=0,
            category="Unknown Category",
            subcategories=["addition"],
            confidence=0.99,
        )
        ok, reason = validate_prediction(prediction, min_confidence=0.72)
        self.assertFalse(ok)
        self.assertEqual(reason, "invalid_category")

    def test_validate_prediction_rejects_invalid_linked_subcategory(self) -> None:
        prediction = Prediction(
            row_id=1,
            category="Food & Dining",
            subcategories=["contrast"],
            confidence=0.95,
        )
        ok, reason = validate_prediction(prediction, min_confidence=0.72)
        self.assertFalse(ok)
        self.assertEqual(reason, "invalid_subcategory:contrast")

    def test_format_subcategory_cell_enforces_max_three_tags(self) -> None:
        with self.assertRaises(ValueError):
            format_subcategory_cell(
                [
                    "addition",
                    "contrast",
                    "sequence",
                    "condition",
                ]
            )

    def test_format_subcategory_cell_uses_semicolon_delimiter(self) -> None:
        cell = format_subcategory_cell(["addition", "contrast", "sequence"])
        self.assertEqual(cell, "addition;contrast;sequence")

    def test_repair_category_subcategory_link(self) -> None:
        prediction = Prediction(
            row_id=3,
            category="Grammar & Function Words",
            subcategories=["food_item"],
            confidence=0.81,
        )
        repaired = repair_category_subcategory_link(prediction)
        self.assertEqual(repaired.category, "Food & Dining")
        self.assertEqual(repaired.subcategories, ["food_item"])

    def test_apply_predictions_preserves_row_order(self) -> None:
        rows = [
            {
                "No": "2",
                "Chinese": "爱好",
                "Pinyin": "ài hào",
                "English": "hobby",
                "Level": "1",
                "Category": "",
            },
            {
                "No": "1",
                "Chinese": "爱",
                "Pinyin": "ài",
                "English": "love",
                "Level": "1",
                "Category": "",
            },
        ]
        predictions = {
            0: Prediction(
                row_id=0,
                category="Home & Daily Life",
                subcategories=["daily_routine"],
                confidence=0.88,
            ),
            1: Prediction(
                row_id=1,
                category="Qualities & States",
                subcategories=["emotion_attitude"],
                confidence=0.9,
            ),
        }
        enriched = apply_predictions_to_rows(rows, predictions)
        self.assertEqual(enriched[0]["No"], "2")
        self.assertEqual(enriched[1]["No"], "1")
        self.assertEqual(enriched[0]["Subcategory"], "daily_routine")
        self.assertEqual(enriched[1]["Subcategory"], "emotion_attitude")


if __name__ == "__main__":
    unittest.main()
