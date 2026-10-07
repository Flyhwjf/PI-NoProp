"""CPU checks for exact chart mapping, provenance and scientific encodings."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


FIGURE_CODE = Path(__file__).resolve().parents[1] / "paper/figure_code"
if str(FIGURE_CODE) not in sys.path:
    sys.path.insert(0, str(FIGURE_CODE))

import fig_latent_metrics as latent
import fig_noise as noise
import fig_spider_noise as spider
from figure_protocol import V5


@pytest.fixture
def noise_artifact():
    levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = {}
    # Insert dictionaries backwards to catch accidental iteration-based labels.
    for region_index, region in reversed(list(enumerate(noise.REGIONS))):
        results[region] = {}
        for method_index, method in reversed(list(enumerate(noise.METHODS))):
            results[region][method] = {}
            for level_index, level in reversed(list(enumerate(levels))):
                results[region][method][str(level)] = {
                    "accuracy": {"mean": 50 + 10 * region_index + 5 * method_index
                                 - level_index, "std": 2.0 + region_index},
                    "eta_ns": {"mean": 0.1 + 0.2 * region_index + 0.1 * method_index
                               + 0.01 * level_index, "std": 0.5},
                }
    return {"protocol": {"levels_in_channel_standard_deviations": levels},
            "results": results}


@pytest.fixture
def latent_artifact():
    results = {}
    for region_index, region in reversed(list(enumerate(latent.REGIONS))):
        results[region] = {}
        for method_index, method in reversed(list(enumerate(latent.METHODS))):
            results[region][method] = {
                "between_class_centroid_distance": 1.0 + 3 * region_index + method_index,
                "silhouette_128d": -0.04 + 0.04 * method_index + 0.01 * region_index,
                "within_class_trace": 70.0 + 10 * region_index + method_index,
                "n_samples": 324 if region_index == 0 else 252,
            }
    # Projections are deliberately unusable: quantitative mapping must ignore them.
    return {"results": results, "points": "not projected statistics"}


@pytest.fixture
def spider_artifact():
    levels = [0.0, 0.001, 0.005, 0.01, 0.1, 1.0]
    records = {}
    for index, level in enumerate(levels):
        terms = list(spider.NS_SUPPORT)
        if index == 3:
            terms.append("kinetic_energy_gradient")
        passed = index < 3
        records[str(level)] = {
            "terms": terms,
            "max_coefficient_relative_error": 0.001 * 10 ** index,
            "bootstrap_support_fraction": [1.0, 1.0, 0.8, 0.95, 0.6, 0.5][index],
            "discovery_eta": 0.0001 * 5 ** index,
            "validation_eta": 0.0002 * 5 ** index,
            "test_eta": 0.0003 * 5 ** index,
            "validation_passed_under_noise_protocol": passed,
            "failure_reasons": [] if passed else ["source-declared rejection"],
        }
    return {
        "levels_in_field_standard_deviations": levels, "levels": records,
        "protocol": {"max_coefficient_relative_error": 2.0,
                     "min_bootstrap_support": 0.5,
                     "max_validation_eta": 0.25, "max_test_eta": 0.25},
    }


def test_v5_sources_and_public_plot_defaults_are_unchanged():
    for module, suffix, constant in (
        (noise, "noise", "AGGREGATE"),
        (latent, "latent_analysis", "ARTIFACT"),
    ):
        assert getattr(module, constant) == V5.aggregate_path(suffix)
        assert inspect.signature(module.plot).parameters["protocol"].default is V5
        assert inspect.signature(module.plot).parameters["output"].default == module.OUTPUT
    assert not inspect.signature(spider.plot).parameters
    assert spider.AGGREGATE.name == "full_ns_spider_noise.json"
    assert "v5" not in spider.AGGREGATE.name


def test_noise_mapping_preserves_numeric_levels_and_complete_sd(noise_artifact):
    levels, data = noise.extract_noise_data(noise_artifact)
    np.testing.assert_array_equal(levels, [0, .01, .05, .1, .2, .5, 1])
    assert tuple(data) == tuple((r, m) for r in noise.REGIONS for m in noise.METHODS)
    np.testing.assert_allclose(data["low_enstrophy", "none"]["eta_ns_mean"],
                               .1 + .01 * np.arange(7))
    np.testing.assert_array_equal(data["low_enstrophy", "none"]["eta_ns_std"], [.5] * 7)
    np.testing.assert_array_equal(data["high_enstrophy", "discovered"]["accuracy_mean"],
                                 65 - np.arange(7))


@pytest.mark.parametrize("levels", [[0, .1, .1], [0, float("nan"), 1], [0, -.1, 1]])
def test_invalid_noise_levels_fail_closed(noise_artifact, levels):
    noise_artifact["protocol"]["levels_in_channel_standard_deviations"] = levels
    with pytest.raises(ValueError, match="Noise levels"):
        noise.extract_noise_data(noise_artifact)


def test_noise_error_bars_are_not_clipped_at_zero(noise_artifact):
    figure = noise.build_figure(noise_artifact)
    try:
        assert len(figure.axes) == 2
        axis = figure.axes[1]
        assert axis.get_ylim()[0] < -.4
        first_bars = axis.containers[0].lines[2][0].get_segments()
        np.testing.assert_allclose(first_bars[0][:, 1], [-.4, .6])
        np.testing.assert_array_equal(
            figure.axes[0].containers[0].lines[0].get_xdata(),
            noise_artifact["protocol"]["levels_in_channel_standard_deviations"])
        assert all(a.get_xscale() == "symlog" for a in figure.axes)
    finally:
        plt.close(figure)


def test_latent_matrix_uses_metric_region_method_order_and_exact_counts(latent_artifact):
    values, counts = latent.extract_metric_matrix(latent_artifact)
    assert values.shape == (3, 2, 3)
    np.testing.assert_array_equal(values[0], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(values[1], [[-.04, 0, .04], [-.03, .01, .05]])
    np.testing.assert_array_equal(values[2], [[70, 71, 72], [80, 81, 82]])
    np.testing.assert_array_equal(counts, [[324] * 3, [252] * 3])
    assert "324 low-enstrophy and 252 high-enstrophy" in latent.figure_caption(latent_artifact)


@pytest.mark.parametrize("key,value,message", [
    ("within_class_trace", float("nan"), "finite"),
    ("within_class_trace", -1, "nonnegative"),
    ("silhouette_128d", 1.1, r"\[-1, 1\]"),
    ("n_samples", 0, "positive integers"),
    ("n_samples", True, "positive integers"),
    ("n_samples", 200, "same held-out sample count"),
])
def test_invalid_latent_statistics_fail_closed(latent_artifact, key, value, message):
    latent_artifact["results"]["low_enstrophy"]["none"][key] = value
    with pytest.raises(ValueError, match=message):
        latent.extract_metric_matrix(latent_artifact)


def test_latent_panels_show_unconnected_pooled_values_without_error_bars(latent_artifact):
    figure = latent.build_figure(latent_artifact)
    values, _ = latent.extract_metric_matrix(latent_artifact)
    try:
        assert len(figure.axes) == 3
        for metric_index, axis in enumerate(figure.axes):
            assert not axis.containers
            assert len(axis.collections) == 2
            for region_index, collection in enumerate(axis.collections):
                np.testing.assert_array_equal(collection.get_offsets()[:, 0],
                                              values[metric_index, region_index])
            assert all(line.get_linestyle() == ":" for line in axis.lines)
        assert figure.axes[0].get_xlim()[0] == 0
        assert figure.axes[2].get_xlim()[0] == 0
    finally:
        plt.close(figure)


def test_spider_mapping_separates_bootstrap_support_and_gate_flags(spider_artifact):
    data = spider.extract_spider_data(spider_artifact)
    np.testing.assert_array_equal(data["coefficient_error_percent"],
                                 [0.1, 1, 10, 100, 1000, 10000])
    np.testing.assert_allclose(data["bootstrap_percent"], [100, 100, 80, 95, 60, 50])
    np.testing.assert_array_equal(data["passed"], [True, True, True, False, False, False])
    np.testing.assert_array_equal(data["full_support"], [True, True, True, False, True, True])
    assert data["failure_reasons"][4] == ("source-declared rejection",)
    assert data["coefficient_gate_percent"] == 200
    assert data["bootstrap_gate_percent"] == 50
    assert data["validation_gate"] == data["test_gate"] == .25


def test_spider_high_bootstrap_does_not_turn_wrong_support_into_a_pass(spider_artifact):
    data = spider.extract_spider_data(spider_artifact)
    assert data["bootstrap_percent"][3] == 95
    assert not data["passed"][3]
    wrong = copy.deepcopy(spider_artifact)
    wrong["levels"]["0.01"]["validation_passed_under_noise_protocol"] = True
    with pytest.raises(ValueError, match="full NS support"):
        spider.extract_spider_data(wrong)


def test_spider_boolean_decisions_cannot_be_inferred_from_strings(spider_artifact):
    spider_artifact["levels"]["0.01"]["validation_passed_under_noise_protocol"] = "false"
    with pytest.raises(ValueError, match="Boolean flags"):
        spider.extract_spider_data(spider_artifact)


def test_spider_missing_source_never_reads_another_file():
    with patch.object(Path, "is_file", return_value=False), \
            patch.object(Path, "read_text") as read:
        with pytest.raises(FileNotFoundError, match="SPIDER DNS-noise source"):
            spider.load_spider_artifact(Path("missing-noise.json"))
    read.assert_not_called()


def test_spider_status_markers_use_actual_bootstrap_percentages(spider_artifact):
    figure = spider.build_figure(spider_artifact)
    data = spider.extract_spider_data(spider_artifact)
    try:
        axis = figure.axes[1]
        assert axis.lines[0].get_marker() in (None, "None", "")
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), data["bootstrap_percent"])
        assert len(axis.collections) == 2
        for collection, passed in zip(axis.collections, (True, False)):
            selected = data["passed"] == passed
            np.testing.assert_array_equal(collection.get_offsets(), np.column_stack((
                data["positions"][selected], data["bootstrap_percent"][selected],
            )))
            assert collection.get_offset_transform() is axis.transData
        assert axis.get_legend().get_title().get_text() == "Protocol status"
        assert "actual bootstrap percentages" in spider.figure_caption()
        assert "status row" not in spider.figure_caption()
    finally:
        plt.close(figure)


def test_spider_uses_three_axes_and_no_dual_scales(spider_artifact):
    figure = spider.build_figure(spider_artifact)
    try:
        assert len(figure.axes) == 3
        assert [axis.get_yscale() for axis in figure.axes] == ["log", "linear", "log"]
        assert figure.axes[1].get_ylim() == (0, 108)
        assert all("DNS noise / field SD (%)" == a.get_xlabel() for a in figure.axes)
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), np.arange(6))
        assert "categorical positions" in spider.figure_caption()
    finally:
        plt.close(figure)


def test_all_panels_use_matching_physical_size_frames_and_inside_legends(
        noise_artifact, latent_artifact, spider_artifact):
    figures = [noise.build_figure(noise_artifact),
               latent.build_figure(latent_artifact), spider.build_figure(spider_artifact)]
    try:
        for figure in figures:
            assert figure.get_size_inches()[0] == 6.9
            assert figure.dpi == 300
            for axis in figure.axes:
                assert all(spine.get_visible() for spine in axis.spines.values())
                assert axis.get_title(loc="left").startswith("(")
                legend = axis.get_legend()
                if legend is not None:
                    assert legend.get_frame_on()
                    assert legend.get_frame().get_alpha() == .85
    finally:
        for figure in figures:
            plt.close(figure)
