"""CPU-only figure-source and entrypoint tests; no rendering or training."""

import contextlib
import importlib
import inspect
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts import plot_results
import figure_protocol


class TestFigureProtocol(unittest.TestCase):
    def test_entrypoint_defaults_to_v5_and_requires_explicit_old16(self):
        parser = plot_results.build_parser()
        self.assertEqual(parser.parse_args([]).protocol, 'v5')
        self.assertEqual(parser.parse_args(['--protocol', 'old16']).protocol, 'old16')
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--protocol', 'v4'])

    def test_shared_figure_cli_defaults_and_explicit_legacy_match(self):
        output = Path('figure.pdf')
        for argv, expected in (([], figure_protocol.V5),
                               (['--protocol', 'old16'], figure_protocol.OLD16)):
            with self.subTest(argv=argv), patch.object(sys, 'argv', ['figure.py', *argv]):
                protocol, destination = figure_protocol.parse_cli(output, 'test')
            self.assertIs(protocol, expected)
            self.assertEqual(destination, output)

    def test_geometry_lambda_and_metadata_paths_are_isolated(self):
        current, legacy = figure_protocol.V5, figure_protocol.OLD16
        root = figure_protocol.ROOT
        self.assertEqual((current.input_size, current.target_size, current.lambda_phys,
                          current.context_mode), (32, 16, 0.1, 'residual_warmstart'))
        self.assertEqual((legacy.input_size, legacy.target_size, legacy.lambda_phys,
                          legacy.context_mode), (16, 16, 0.01, 'single'))
        self.assertEqual(current.cache_path/'metadata.json',
                         root/'data/cache_hit_ns_input32_target16/metadata.json')
        self.assertEqual(legacy.cache_path/'metadata.json',
                         root/'data/cache_hit_ns/metadata.json')
        for suffix in ('results', *figure_protocol.EXTENSIONS):
            with self.subTest(suffix=suffix):
                self.assertEqual(current.aggregate_path(suffix).name,
                                 f'{current.aggregate_prefix}_{suffix}.json')
                self.assertEqual(legacy.aggregate_path(suffix).name, f'full_ns_{suffix}.json')
                self.assertNotEqual(current.aggregate_path(suffix), legacy.aggregate_path(suffix))

    def test_run_names_use_selected_protocol_and_physics_weight(self):
        self.assertEqual(figure_protocol.V5.run_path('low_enstrophy', 42).name,
                         'full_ns_v5_input32_target16_residual_warmstart_'
                         'discovered_low_enstrophy_lambda0p1_seed42')
        self.assertEqual(figure_protocol.OLD16.run_path('low_enstrophy', 42).name,
                         'full_ns_v4_discovered_low_enstrophy_lambda0p01_seed42')
        self.assertTrue(figure_protocol.V5.run_path('high_enstrophy', 123, 'none').name
                        .endswith('_none_high_enstrophy_lambda0_seed123'))

    def test_metadata_rejects_wrong_geometry_context_weight_and_v4_revision(self):
        source = Path('selected-source.json')
        matched = {'input_spatial_size': 32, 'target_spatial_size': 16,
                   'spatial_context_mode': 'residual_warmstart', 'lambda_phys': 0.1}
        figure_protocol.validate_metadata(figure_protocol.V5, matched, source)
        for key, value in (('input_spatial_size', 16), ('target_spatial_size', 32),
                           ('spatial_context_mode', 'dual_scale'), ('lambda_phys', 0.01),
                           ('model_revision', 'trainable-physics-condition-fusion-prototype-readout')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                figure_protocol.validate_metadata(figure_protocol.V5,
                                                   {**matched, key: value}, source)

    def test_protocol_dependent_figures_default_to_v5(self):
        constants = {'fig_main_results': ('AGGREGATE', 'results'),
                     'fig_noise': ('AGGREGATE', 'noise'),
                     'fig_latent_analysis': ('ARTIFACT', 'latent_analysis')}
        for name, uses_protocol in plot_results.PAPER_FIGURES:
            if not uses_protocol:
                continue
            with self.subTest(figure=name):
                module = importlib.import_module(name)
                self.assertIs(inspect.signature(module.plot).parameters['protocol'].default,
                              figure_protocol.V5)
                if name in constants:
                    attribute, suffix = constants[name]
                    self.assertEqual(getattr(module, attribute),
                                     figure_protocol.V5.aggregate_path(suffix))
        data = importlib.import_module('fig_data_samples')
        ablation = importlib.import_module('fig_ablation')
        self.assertEqual(data.DATA, figure_protocol.V5.cache_path)
        self.assertEqual(ablation.LAMBDA_ARTIFACT,
                         figure_protocol.V5.aggregate_path('lambda_ablation'))
        self.assertEqual(ablation.DECODER_ARTIFACT,
                         figure_protocol.V5.aggregate_path('decoder_ablation'))

    def test_missing_v5_aggregate_does_not_read_a_legacy_file(self):
        with patch.object(Path, 'is_file', return_value=False), \
                patch.object(Path, 'read_text') as read, self.assertRaisesRegex(
                    FileNotFoundError, 'full_ns_v5_input32_target16_residual_warmstart_noise.json'):
            figure_protocol.load_aggregate(figure_protocol.V5, 'noise')
        read.assert_not_called()


class TestFigureDataMapping(unittest.TestCase):
    def test_lambda_source_order_preserves_numeric_coordinates_and_metric_units(self):
        module = importlib.import_module('fig_ablation')
        results = {
            str(weight): {
                'accuracy': {'mean': 80 + weight, 'std': 2.0},
                'eta_ns': {'mean': 0.4 - weight, 'std': 0.01},
                'eta_div': {'mean': 0.1, 'std': 0.02},
            } for weight in (0.1, 0.0, 0.01)
        }
        artifact = {'protocol': {'lambdas': [0.1, 0, 0.01]}, 'results': results}
        sweep = module.extract_lambda_data(artifact)
        plot_results.np.testing.assert_array_equal(sweep.weights, [0, 0.01, 0.1])
        plot_results.np.testing.assert_allclose(sweep.accuracy, [80, 80.01, 80.1])
        plot_results.np.testing.assert_allclose(sweep.accuracy_std, [2, 2, 2])
        plot_results.np.testing.assert_allclose(sweep.eta_ns, [0.4, 0.39, 0.3])

    def test_incomplete_lambda_set_fails_before_creating_a_figure(self):
        module = importlib.import_module('fig_ablation')
        artifact = {'protocol': {'lambdas': [0, 0.01, 0.1]}, 'results': {}}
        with patch.object(module, 'configure_style'), \
                patch.object(module, 'load_aggregate', return_value=artifact), \
                patch.object(module.plt, 'subplots') as create, \
                self.assertRaisesRegex(ValueError, 'must match'):
            module.plot()
        create.assert_not_called()


class TestFigureEntrypoint(unittest.TestCase):
    def test_render_plan_contains_exactly_the_main_and_appendix_figures(self):
        self.assertEqual(plot_results.PAPER_FIGURES, (
            ('fig_framework', False), ('fig_data_samples', True),
            ('fig_ablation', True), ('fig_noise', True), ('fig_latent_metrics', True),
            ('fig_dns_quality', False), ('fig_training_convergence', True),
            ('fig_spider_noise', False), ('fig_latent_analysis', True),
        ))
        for name, _ in plot_results.PAPER_FIGURES:
            self.assertTrue((plot_results.FIGURE_CODE/f'{name}.py').is_file())

    def test_default_and_legacy_dispatch_only_tuned_scripts(self):
        historical = ('style', 'framework', 'local_update', 'dns_quality',
                      'generate_data_samples', 'spider_figure', 'result_figures',
                      'training_convergence_figure', 'noise_figure',
                      'spider_noise_figure', 'ablation_figure', 'latent_figure',
                      'efficiency_figure')
        for argv, protocol in (([], figure_protocol.V5),
                               (['--protocol', 'v5'], figure_protocol.V5),
                               (['--protocol', 'old16'], figure_protocol.OLD16)):
            with self.subTest(argv=argv), contextlib.ExitStack() as stack:
                guards = [stack.enter_context(patch.object(
                    plot_results, name, side_effect=AssertionError(f'Legacy call: {name}')))
                          for name in historical]
                preflight = stack.enter_context(patch.object(plot_results, 'preflight_sources'))
                run = stack.enter_context(patch.object(plot_results.subprocess, 'run'))
                stack.enter_context(patch('builtins.print'))
                plot_results.main(argv)
                preflight.assert_called_once_with(protocol)
                self.assertEqual(run.call_count, len(plot_results.PAPER_FIGURES))
                for call, (name, uses_protocol) in zip(run.call_args_list,
                                                     plot_results.PAPER_FIGURES):
                    expected = [sys.executable, '-B',
                                str(plot_results.FIGURE_CODE/f'{name}.py')]
                    if uses_protocol:
                        expected += ['--protocol', protocol.name]
                    self.assertEqual(call.args, (expected,))
                    self.assertEqual(call.kwargs, {'cwd': plot_results.ROOT, 'check': True})
                for guard in guards:
                    guard.assert_not_called()

    def test_required_sources_use_v5_cache_and_six_selected_runs(self):
        for protocol in (figure_protocol.V5, figure_protocol.OLD16):
            paths = plot_results.required_sources(protocol)
            self.assertIn(protocol.cache_path/'metadata.json', paths)
            for region in plot_results.REGIONS:
                for seed in plot_results.SEEDS:
                    for name in ('history.npz', 'metrics.json', 'config.json'):
                        self.assertIn(protocol.run_path(region, seed)/name, paths)
            self.assertIn(plot_results.ROOT/'outputs/aggregate/full_ns_spider_noise.json', paths)
            self.assertIn(plot_results.ROOT/'data/generated_hit_ns/manifest.json', paths)
            self.assertNotIn(protocol.aggregate_path('baselines'), paths)
            self.assertNotIn(protocol.aggregate_path('relation_ablation'), paths)
            self.assertNotIn(plot_results.ROOT/'outputs/spider/full_ns_equation.json', paths)
        current_paths = plot_results.required_sources()
        self.assertNotIn(figure_protocol.OLD16.cache_path/'metadata.json', current_paths)
        self.assertNotIn(figure_protocol.OLD16.aggregate_path('results'), current_paths)

    def test_missing_new_aggregate_fails_before_any_pdf_is_rendered(self):
        missing = figure_protocol.V5.aggregate_path('noise')
        with patch.object(Path, 'is_file', autospec=True,
                          side_effect=lambda path: path != missing), \
                patch.object(plot_results, 'load_aggregate') as load, \
                patch.object(plot_results.subprocess, 'run') as run, \
                self.assertRaisesRegex(FileNotFoundError, 'no legacy fallback'):
            plot_results.main([])
        load.assert_not_called()
        run.assert_not_called()

    def test_failed_child_is_not_skipped_or_retried_with_legacy_renderer(self):
        failure = subprocess.CalledProcessError(1, ['figure.py'])
        with patch.object(plot_results, 'preflight_sources'), \
                patch.object(plot_results.subprocess, 'run', side_effect=failure) as run, \
                patch('builtins.print'), self.assertRaises(subprocess.CalledProcessError):
            plot_results.main([])
        run.assert_called_once()

    def test_cache_validation_reads_selected_metadata_and_numpy_headers(self):
        for protocol in (figure_protocol.V5, figure_protocol.OLD16):
            metadata = {'input_spatial_size': protocol.input_size,
                        'target_spatial_size': protocol.target_size,
                        'spatial_size': protocol.input_size, 'n_samples': 2}
            arrays = [SimpleNamespace(shape=(2, 4) + (protocol.input_size,)*3),
                      SimpleNamespace(shape=(2,))]
            with self.subTest(protocol=protocol.name), \
                    patch.object(plot_results, 'read_json', return_value=metadata) as read, \
                    patch.object(plot_results.np, 'load', side_effect=arrays) as load:
                plot_results.validate_cache_source(protocol)
            read.assert_called_once_with(protocol.cache_path/'metadata.json')
            self.assertEqual([call.args[0] for call in load.call_args_list],
                             [protocol.cache_path/'fields.npy', protocol.cache_path/'regions.npy'])
            for call in load.call_args_list:
                self.assertEqual(call.kwargs, {'mmap_mode': 'r', 'allow_pickle': False})

    def test_v5_rejects_legacy_cache_metadata_before_reading_fields(self):
        metadata = {'spatial_size': 16}
        with patch.object(plot_results, 'read_json', return_value=metadata), \
                patch.object(plot_results.np, 'load') as load, \
                self.assertRaisesRegex(ValueError, 'cache geometry'):
            plot_results.validate_cache_source()
        load.assert_not_called()

    def test_v5_rejects_wrong_input_array_size_even_with_matching_metadata(self):
        metadata = {'input_spatial_size': 32, 'target_spatial_size': 16}
        arrays = [SimpleNamespace(shape=(2, 4, 16, 16, 16)),
                  SimpleNamespace(shape=(2,))]
        with patch.object(plot_results, 'read_json', return_value=metadata), \
                patch.object(plot_results.np, 'load', side_effect=arrays), \
                self.assertRaisesRegex(ValueError, 'input fields shape'):
            plot_results.validate_cache_source()


if __name__ == '__main__':
    unittest.main()
