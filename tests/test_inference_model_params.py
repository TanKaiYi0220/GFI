from __future__ import annotations

import unittest

from src.utils.model_stats import build_model_parameter_summary


class FakeParameter:
    def __init__(self, element_count: int) -> None:
        self.element_count = element_count

    def numel(self) -> int:
        return self.element_count


class FakeModel:
    def parameters(self) -> list[FakeParameter]:
        return [
            FakeParameter(12),
            FakeParameter(3),
            FakeParameter(3),
            FakeParameter(3),
        ]


class InferenceModelParameterTests(unittest.TestCase):
    def test_build_model_parameter_summary_counts_all_model_parameters(self) -> None:
        model = FakeModel()

        summary = build_model_parameter_summary(model)

        self.assertEqual(summary["model_param_count"], 21)
        self.assertAlmostEqual(summary["model_param_million"], 0.000021)


if __name__ == "__main__":
    unittest.main()
