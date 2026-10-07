"""CPU-only tests for truthful lambda-sweep coordinates, units and uncertainty."""
from __future__ import annotations

import copy
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURE_CODE = ROOT / "paper/figure_code"
if str(FIGURE_CODE) not in sys.path:
    sys.path.insert(0, str(FIGURE_CODE))
import fig_ablation as figure
from figure_protocol import OLD16, V5


def fixture():
    weights = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
    return {
        "protocol": {"lambdas": weights},
        "results": {
            str(weight): {
                "accuracy": {"mean": 86.0 + index / 10, "std": 2.0 + index / 10},
                "eta_ns": {"mean": 0.8 - index / 10, "std": 0.04 + index / 100},
                "eta_div": {"mean": 0.3 - index / 30, "std": 0.03 + index / 100},
            } for index, weight in enumerate(weights)
        },
    }


class TestLambdaData(unittest.TestCase):
    def test_actual_coordinates_include_zero_not_equidistant_indices(self):
        data = figure.extract_lambda_data(fixture())
        np.testing.assert_array_equal(data.weights, [0, .001, .003, .01, .03, .1])
        self.assertFalse(np.allclose(data.weights, np.arange(6)))

    def test_accuracy_and_standard_deviation_stay_in_percent(self):
        data = figure.extract_lambda_data(fixture())
        np.testing.assert_allclose(data.accuracy, [86, 86.1, 86.2, 86.3, 86.4, 86.5])
        np.testing.assert_allclose(data.accuracy_std, [2, 2.1, 2.2, 2.3, 2.4, 2.5])
        np.testing.assert_allclose(data.eta_ns, [.8, .7, .6, .5, .4, .3])
        np.testing.assert_allclose(data.eta_div, [.3 - index / 30 for index in range(6)])

    def test_producer_order_and_zero_key_spelling_do_not_change_values(self):
        artifact = fixture()
        expected = figure.extract_lambda_data(artifact)
        artifact["protocol"]["lambdas"] = list(reversed(artifact["protocol"]["lambdas"]))
        artifact["results"] = dict(reversed(list(artifact["results"].items())))
        artifact["results"]["0"] = artifact["results"].pop("0.0")
        actual = figure.extract_lambda_data(artifact)
        for field in expected.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))

    def test_duplicate_declared_weights_and_numerical_keys_are_rejected(self):
        artifact = fixture()
        artifact["protocol"]["lambdas"][1] = 0.0
        with self.assertRaisesRegex(ValueError, "distinct"):
            figure.extract_lambda_data(artifact)
        artifact = fixture()
        artifact["results"]["0"] = copy.deepcopy(artifact["results"]["0.0"])
        with self.assertRaisesRegex(ValueError, "Duplicate numerical"):
            figure.extract_lambda_data(artifact)

    def test_invalid_weights_and_missing_or_extra_result_keys_are_rejected(self):
        for value in (-.001, float("nan"), float("inf")):
            artifact = fixture()
            artifact["protocol"]["lambdas"][1] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                figure.extract_lambda_data(artifact)
        for mode in ("missing", "extra"):
            artifact = fixture()
            if mode == "missing":
                artifact["results"].pop("0.01")
            else:
                artifact["results"]["0.2"] = artifact["results"]["0.1"]
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "match"):
                figure.extract_lambda_data(artifact)

    def test_nonfinite_or_negative_metric_statistics_are_rejected(self):
        for metric, stat, value in (
                ("accuracy", "mean", 101), ("accuracy", "std", -.1),
                ("eta_ns", "mean", float("nan")), ("eta_div", "std", float("inf"))):
            artifact = fixture()
            artifact["results"]["0.01"][metric][stat] = value
            with self.subTest(metric=metric, stat=stat), self.assertRaises(ValueError):
                figure.extract_lambda_data(artifact)

    def test_full_sd_is_retained_beyond_the_bounds_of_a_metric(self):
        lower, upper = figure.errorbar_limits(np.array([99.]), np.array([5.]),
                                               minimum_top=100.)
        self.assertLessEqual(lower, 0)
        self.assertGreater(upper, 104)
        lower, upper = figure.errorbar_limits(np.array([.02, .8]),
                                               np.array([.05, .2]))
        self.assertLess(lower, -.03)
        self.assertGreater(upper, 1.0)


class TestLambdaRendering(unittest.TestCase):
    def test_only_lambda_source_is_loaded_for_default_and_explicit_legacy(self):
        for protocol in (V5, OLD16):
            plot = MagicMock()
            with self.subTest(protocol=protocol.name), contextlib.ExitStack() as stack:
                load = stack.enter_context(patch.object(
                    figure, "load_aggregate", return_value=fixture()))
                stack.enter_context(patch.object(figure, "make_figure",
                                                  return_value=(plot, None)))
                stack.enter_context(patch.object(Path, "mkdir"))
                stack.enter_context(patch.object(figure.plt, "close"))
                stack.enter_context(patch("builtins.print"))
                figure.plot(protocol, Path("ablation.pdf"))
            load.assert_called_once_with(protocol, "lambda_ablation")
            self.assertEqual(plot.savefig.call_count, 2)
            self.assertEqual(plot.savefig.call_args_list[0].args[0], Path("ablation.pdf"))
            self.assertEqual(plot.savefig.call_args_list[1].args[0],
                             Path("ablation_preview.png"))
            self.assertEqual(plot.savefig.call_args.kwargs["dpi"], 300)

    def test_malformed_sweep_fails_before_creating_a_figure(self):
        artifact = fixture()
        artifact["results"].pop("0.0")
        with patch.object(figure, "load_aggregate", return_value=artifact), \
                patch.object(figure, "make_figure") as create, \
                self.assertRaises(ValueError):
            figure.plot()
        create.assert_not_called()

    def test_artists_use_numerical_weights_percent_accuracy_and_full_sd(self):
        data = figure.extract_lambda_data(fixture())
        fig, axes = figure.make_figure(data)
        try:
            self.assertEqual(len(axes), 2)
            np.testing.assert_allclose(fig.get_size_inches(), [6.9, 3.45])
            for axis in axes:
                self.assertEqual(axis.get_xscale(), "symlog")
                self.assertEqual(axis.xaxis.get_transform().linthresh, .001)
                np.testing.assert_array_equal(axis.get_xticks(), data.weights)
                for spine in axis.spines.values():
                    self.assertTrue(spine.get_visible())
                self.assertIsNotNone(axis.get_legend())
                self.assertTrue(axis.get_legend().get_frame_on())
            self.assertAlmostEqual(axes[0].get_position().width,
                                   axes[1].get_position().width)
            self.assertEqual(axes[0].get_ylabel(), "Test accuracy (%)")
            self.assertGreaterEqual(axes[0].get_ylim()[1], 100)

            series = [(axes[0].containers[0], data.accuracy, data.accuracy_std),
                      (axes[1].containers[0], data.eta_ns, data.eta_ns_std),
                      (axes[1].containers[1], data.eta_div, data.eta_div_std)]
            for container, means, stds in series:
                np.testing.assert_array_equal(container.lines[0].get_xdata(), data.weights)
                np.testing.assert_array_equal(container.lines[0].get_ydata(), means)
                segments = container.lines[2][0].get_segments()
                for index, segment in enumerate(segments):
                    np.testing.assert_allclose(segment, [
                        [data.weights[index], means[index] - stds[index]],
                        [data.weights[index], means[index] + stds[index]]])
        finally:
            plt.close(fig)

    def test_no_chart_text_is_smaller_than_nine_points_at_print_width(self):
        fig, axes = figure.make_figure(figure.extract_lambda_data(fixture()))
        try:
            scale = 5.4 / figure.FIGURE_WIDTH
            for axis in axes:
                text = ([axis.title, axis.xaxis.label, axis.yaxis.label]
                        + axis.get_xticklabels() + axis.get_yticklabels()
                        + axis.get_legend().get_texts())
                for label in text:
                    self.assertGreaterEqual(label.get_fontsize() * scale, 9 - 1e-9)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
