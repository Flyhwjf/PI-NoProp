"""CPU-only checks of experiment geometry, naming and checkpoint isolation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts import (analyze_latents, run_baselines, run_decoder_ablation,
                     run_lambda_ablation, run_noise, run_relation_ablation)
from scripts.experiment_protocol import (ExperimentProtocol,
                                        assert_matched_condition, load_run,
                                        make_decoder, protocol_from_args,
                                        protect_legacy_output, validate_checkpoint,
                                        validate_metadata)
from scripts.run_baselines import GlobalPhysicsClassifier, build_parser, make_config
from src.baselines.cnn import SimpleCNN
from src.data.hit_dataset import _ns_energy_terms


class TestExperimentProtocol(unittest.TestCase):
    def test_default_and_legacy_protocols_are_isolated(self):
        current = ExperimentProtocol()
        legacy = ExperimentProtocol(16, 16)
        self.assertEqual(current.prefix,
                         'full_ns_v5_input32_target16_residual_warmstart')
        self.assertEqual(current.cache_dir, 'data/cache_hit_ns_input32_target16')
        self.assertEqual(current.lambda_phys, 0.1)
        self.assertEqual(legacy.prefix, 'full_ns_v4')
        self.assertEqual(legacy.cache_dir, 'data/cache_hit_ns')
        self.assertEqual(legacy.context_encoder, 'single')
        self.assertEqual(legacy.lambda_phys, 0.01)
        self.assertNotEqual(current.aggregate_path('.', 'baselines'),
                            legacy.aggregate_path('.', 'baselines'))

    def test_run_ids_match_main_script(self):
        protocol = ExperimentProtocol()
        self.assertEqual(protocol.run_id('discovered', 'low_enstrophy', 42),
                         f'{protocol.prefix}_discovered_low_enstrophy_lambda0p1_seed42')
        self.assertIn('_lambda0_seed42', protocol.run_id('none', 'low_enstrophy', 42))
        self.assertTrue(protocol.run_id('discovered', 'low_enstrophy', 42,
                                       relation='ns_pp', tag='eqablation')
                        .endswith('_seed42_ns_pp_eqablation'))

    def test_nondefault_lambda_has_separate_baseline_and_aggregate_paths(self):
        default = ExperimentProtocol()
        custom = ExperimentProtocol(lambda_phys=0.01)
        self.assertNotEqual(default.baseline_run_id('cnn_bp', 'low_enstrophy', 42),
                            custom.baseline_run_id('cnn_bp', 'low_enstrophy', 42))
        self.assertNotEqual(default.aggregate_path('.', 'noise'),
                            custom.aggregate_path('.', 'noise'))

    def test_cli_is_explicit_including_legacy(self):
        parser = build_parser()
        protocol = protocol_from_args(parser.parse_args([]))
        args = parser.parse_args(protocol.cli_args())
        self.assertEqual((args.input_size, args.target_size, args.context_encoder,
                          args.lambda_phys, args.cache_dir),
                         (32, 16, 'residual_warmstart', 0.1,
                          'data/cache_hit_ns_input32_target16'))
        args = parser.parse_args(['--input-size', '16', '--target-size', '16'])
        self.assertEqual(protocol_from_args(args).lambda_phys, 0.01)
        with self.assertRaises(ValueError):
            ExperimentProtocol(16, 32)
        for value in (-1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ExperimentProtocol(lambda_phys=value)

    def test_all_baselines_use_same_context_target_and_physics_config(self):
        protocol = ExperimentProtocol(device='cpu')
        config = make_config('low_enstrophy', [1.0, 1.0, -0.005], protocol)
        self.assertEqual(config.data.subdomain_size, 32)
        self.assertEqual(config.data.target_subdomain_size, 16)
        self.assertEqual(config.physics.physics_grid_size, 16)
        self.assertEqual(config.physics.lambda_weight, 0.1)
        self.assertEqual(config.data.n_subdomains, 64)
        self.assertEqual(config.data.n_classes, 5)
        self.assertTrue(config.data.trajectory_disjoint)
        self.assertEqual(config.noprop.spatial_context_mode, 'residual_warmstart')
        cnn = SimpleCNN(4, 5, config.data.subdomain_size).eval()
        global_model = GlobalPhysicsClassifier(config).eval()
        decoder = make_decoder(config).eval()
        with torch.no_grad():
            fields = torch.zeros(1, 4, 32, 32, 32)
            self.assertEqual(tuple(cnn(fields).shape), (1, 5))
            condition = global_model.encode(fields, torch.zeros(1, 3))
            self.assertEqual(tuple(global_model(fields, torch.zeros(1, 3)).shape), (1, 5))
            self.assertEqual(tuple(decoder(condition).shape), (1, 9, 4, 16, 16, 16))

    def test_checkpoint_rejects_old_dimensions_context_lambda_and_decoder(self):
        protocol = ExperimentProtocol(device='cpu')
        config = make_config('low_enstrophy', [1.0, 1.0, -0.005], protocol)
        validate_checkpoint(config, protocol, source='discovered',
                            region='low_enstrophy', seed=42)
        for section, attribute, value in (
                ('data', 'subdomain_size', 16),
                ('data', 'target_subdomain_size', 32),
                ('noprop', 'spatial_context_mode', 'dual_scale'),
                ('physics', 'lambda_weight', 0.01),
                ('physics', 'physics_grid_size', 32),
                ('data', 'n_subdomains', 256),
                ('data', 'trajectory_disjoint', False),
                ('physics', 'n_time', 4),
                ('noprop', 'normalize_condition', False),
                ('decoder', 'architecture', 'linear')):
            changed = copy.deepcopy(config)
            setattr(getattr(changed, section), attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                validate_checkpoint(changed, protocol, source='discovered',
                                    region='low_enstrophy', seed=42)

    def test_load_run_preserves_checkpoint_paths_and_overrides_only_device(self):
        protocol = ExperimentProtocol(device='cpu')
        config = make_config('low_enstrophy', [1.0, 1.0, -0.005], protocol)
        config.data.cache_dir = 'data/custom_checkpoint_cache'
        config.data.data_dir = 'data/custom_checkpoint_source'
        config.decoder.base_channels = 8
        config.device = 'cuda:7'
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            torch.save({'config': config}, run/'checkpoint.pt')
            (run/'metrics.json').write_text(json.dumps({
                'physics_source': 'discovered', 'region': 'low_enstrophy', 'seed': 42,
                'relation_set': 'ns', 'decoder_architecture': 'conv',
                'input_spatial_size': 32, 'target_spatial_size': 16,
                'spatial_context_mode': 'residual_warmstart'}), encoding='utf-8')
            _, loaded, _ = load_run(run, protocol, source='discovered',
                                    region='low_enstrophy', seed=42)
            self.assertEqual(loaded.device, 'cpu')
            self.assertEqual(loaded.data.cache_dir, config.data.cache_dir)
            self.assertEqual(loaded.data.data_dir, config.data.data_dir)
            self.assertEqual(loaded.decoder.base_channels, 8)
            self.assertEqual(config.device, 'cuda:7')

    def test_cached_v4_metadata_cannot_be_used_as_v5(self):
        with self.assertRaises(ValueError):
            validate_metadata({}, ExperimentProtocol(), legacy_ok=True)
        validate_metadata({}, ExperimentProtocol(16, 16), legacy_ok=True)

    def test_existing_v4_outputs_are_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'metrics.json'
            path.write_text('old output', encoding='utf-8')
            with self.assertRaises(ValueError):
                protect_legacy_output(path, ExperimentProtocol(16, 16))
            self.assertEqual(path.read_text(encoding='utf-8'), 'old output')
            protect_legacy_output(path, ExperimentProtocol())

    def test_decoder_condition_comparison_includes_context_and_prototypes(self):
        state = {key: torch.ones(2) for key in (
            'encoder.context_gate', 'encoder.local_backbone.0.weight',
            'encoder.context_backbone.0.weight', 'physics_encoder.0.weight',
            'condition_fusion.weight', 'label_embed.embed.weight',
            'physics_coefficients', 'physics_mean', 'physics_std', 'blocks.0.weight')}
        reference = {'model_state_dict': state}
        candidate = copy.deepcopy(reference)
        candidate['model_state_dict']['blocks.0.weight'].zero_()
        assert_matched_condition(reference, candidate)
        for key in ('encoder.context_gate', 'label_embed.embed.weight', 'physics_std'):
            changed = copy.deepcopy(candidate)
            changed['model_state_dict'][key].zero_()
            with self.subTest(key=key), self.assertRaises(ValueError):
                assert_matched_condition(reference, changed)
        incomplete = {'model_state_dict': {'encoder.context_gate': torch.ones(2)}}
        with self.assertRaises(ValueError):
            assert_matched_condition(incomplete, incomplete)


class TestCrossScriptProtocol(unittest.TestCase):
    SCRIPTS = (run_baselines, run_noise, run_lambda_ablation,
               run_decoder_ablation, run_relation_ablation, analyze_latents)

    def test_all_cli_defaults_and_legacy_options_match(self):
        for script in self.SCRIPTS:
            parser = script.build_parser()
            for cli, expected in (([], ExperimentProtocol()),
                                  (['--input-size', '16', '--target-size', '16'],
                                   ExperimentProtocol(16, 16))):
                with self.subTest(script=script.__name__, cli=cli):
                    actual = protocol_from_args(parser.parse_args(cli))
                    self.assertEqual(actual.metadata(), expected.metadata())

    def test_ablation_commands_are_explicit_and_paths_match(self):
        for protocol in (ExperimentProtocol(), ExperimentProtocol(16, 16),
                         ExperimentProtocol(context_encoder='dual_scale', lambda_phys=0.03)):
            commands = (
                (run_lambda_ablation.experiment_command(0.003, 123, protocol),
                 run_lambda_ablation.run_id(0.003, 123, protocol)),
                (run_lambda_ablation.experiment_command(protocol.lambda_phys, 123, protocol),
                 run_lambda_ablation.run_id(protocol.lambda_phys, 123, protocol)),
                (run_decoder_ablation.experiment_command('linear', 123, protocol),
                 run_decoder_ablation.run_id('linear', 123, protocol)),
                (run_decoder_ablation.experiment_command('conv', 123, protocol),
                 run_decoder_ablation.run_id('conv', 123, protocol)),
                (run_relation_ablation.experiment_command('full', 123, protocol),
                 run_relation_ablation.run_id('full', 123, protocol)))
            for command, expected_path in commands:
                options = dict(zip(command[2::2], command[3::2]))
                with self.subTest(command=command):
                    self.assertEqual(options['--input-size'], str(protocol.input_size))
                    self.assertEqual(options['--target-size'], str(protocol.target_size))
                    self.assertEqual(options['--context-encoder'], protocol.context_encoder)
                    self.assertEqual(options['--cache-dir'], protocol.cache_dir)
                    actual_path = protocol.run_id(
                        options['--physics-source'], options['--region'],
                        int(options['--seed']), weight=float(options['--lambda-phys']),
                        relation=options['--relation-set'], tag=options.get('--run-tag', ''))
                    self.assertEqual(actual_path, expected_path)

    def test_v5_and_v4_lambda_sweeps_reuse_the_correct_main_weight(self):
        current, legacy = ExperimentProtocol(), ExperimentProtocol(16, 16)
        self.assertNotIn('lambdaablation', run_lambda_ablation.run_id(0.1, 42, current))
        self.assertIn('lambdaablation', run_lambda_ablation.run_id(0.01, 42, current))
        self.assertNotIn('lambdaablation', run_lambda_ablation.run_id(0.01, 42, legacy))
        self.assertIn('lambdaablation', run_lambda_ablation.run_id(0.1, 42, legacy))
        self.assertLess(run_decoder_ablation.ARCHITECTURES.index('conv'),
                        run_decoder_ablation.ARCHITECTURES.index('linear'))

    def test_noise_and_latents_select_identical_new_runs(self):
        for protocol in (ExperimentProtocol(), ExperimentProtocol(16, 16)):
            for source in ('none', 'discovered'):
                self.assertEqual(run_noise.run_dir(source, 'high_enstrophy', 456, protocol),
                                 analyze_latents.run_dir(source, 'high_enstrophy', 456, protocol))
                self.assertIn(protocol.prefix,
                              str(analyze_latents.run_dir(source, 'high_enstrophy', 456, protocol)))

    def test_original_seeds_noise_and_sweep_settings_are_preserved(self):
        for script in self.SCRIPTS:
            self.assertEqual(script.SEEDS, (42, 123, 456))
        self.assertEqual(run_noise.LEVELS, (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0))
        self.assertEqual(run_noise.REPETITIONS, 5)
        self.assertEqual(run_lambda_ablation.LAMBDAS, (0.0, 0.001, 0.003, 0.01, 0.03, 0.1))

    def test_baseline_only_reuses_only_matching_cached_baselines(self):
        protocol = ExperimentProtocol(device='cpu')
        methods = ('noprop_reference', 'noprop_ct', 'cnn_bp',
                   'global_physics_bp', 'spider_rate_classifier')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root/'outputs/spider/full_ns_equation.json'
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_text(json.dumps({'equation': {
                'coefficients': [1.0, 1.0, 1.0, -0.005]}}), encoding='utf-8')
            for region in run_baselines.REGIONS:
                for seed in run_baselines.SEEDS:
                    for method in methods:
                        path = (root/'outputs/runs'/
                                protocol.baseline_run_id(method, region, seed)/'metrics.json')
                        path.parent.mkdir(parents=True)
                        path.write_text(json.dumps({
                            'region': region, 'seed': seed, 'accuracy': 20.0,
                            'eta_ns': None, 'eta_div': None, 'parameters': 10,
                            'train_seconds': 0.0, 'peak_memory_mb': 0.0,
                            'experiment_protocol': protocol.metadata()}), encoding='utf-8')
            def fake_loaders(config):
                self.assertEqual((config.data.subdomain_size,
                                  config.data.target_subdomain_size), (32, 16))
                return {config.data.regions[0]: {}}
            with patch.object(run_baselines, 'ROOT', root), \
                    patch.object(run_baselines, 'create_hit_dataloaders', side_effect=fake_loaders), \
                    patch('sys.argv', ['run_baselines.py', '--device', 'cpu', '--baselines-only']), \
                    patch('builtins.print'):
                run_baselines.main()
            result = json.loads(protocol.aggregate_path(root, 'baselines').read_text())
            self.assertEqual(result['protocol']['model_revision'], protocol.model_revision)
            self.assertEqual(set(result['results']['low_enstrophy']), set(methods))
            self.assertFalse((root/'outputs/aggregate/full_ns_baselines.json').exists())

    def test_noise_validates_checkpoints_even_when_all_levels_are_cached(self):
        protocol = ExperimentProtocol(device='cpu')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = protocol.aggregate_path(root, 'noise')
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'protocol': protocol.metadata(), 'results': {
                region: {method: {str(level): {} for level in run_noise.LEVELS}
                         for method in run_noise.METHODS} for region in run_noise.REGIONS}}))
            with patch.object(run_noise, 'ROOT', root), \
                    patch.object(run_noise, 'load_run', side_effect=ValueError('checkpoint mismatch')) as load, \
                    patch('sys.argv', ['run_noise.py', '--device', 'cpu']), \
                    patch('builtins.print'), self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                run_noise.main()
            load.assert_called_once()

    def test_latent_collection_passes_checkpoint_configuration_through(self):
        protocol = ExperimentProtocol(device='cpu')
        config = make_config('low_enstrophy', [1.0, 1.0, -0.005], protocol)
        config.data.cache_dir = 'checkpoint-specific-cache'
        fake_batch = {'field': torch.zeros(1, 4, 32, 32, 32),
                      'ns_terms': torch.zeros(1, 3), 'label': torch.tensor([0])}
        class FakeModel:
            def to(self, device): return self
            def eval(self): pass
            def load_state_dict(self, state): pass
            def __call__(self, fields, **kwargs):
                return torch.zeros(1, 5), [torch.zeros(1, 128)]
        with patch.object(analyze_latents, 'load_run',
                          return_value=({'model_state_dict': {}}, config, {})), \
                patch.object(analyze_latents, 'NoPropModel', return_value=FakeModel()), \
                patch.object(analyze_latents, 'create_hit_dataloaders',
                             return_value={'low_enstrophy': {'test': [fake_batch]}}) as loaders:
            z, y = analyze_latents.collect('discovered', 'low_enstrophy', 42, protocol)
        self.assertIs(loaders.call_args.args[0], config)
        self.assertEqual(config.data.cache_dir, 'checkpoint-specific-cache')
        self.assertEqual(z.shape, (1, 128))
        self.assertEqual(y.tolist(), [0])


class TestRelationAblationProvenance(unittest.TestCase):
    @staticmethod
    def checkpoint():
        return {
            'model_state_dict': {key: torch.ones(2) for key in (
                'encoder.context_gate', 'encoder.local_backbone.0.weight',
                'encoder.context_backbone.0.weight', 'physics_encoder.0.weight',
                'condition_fusion.weight', 'label_embed.embed.weight',
                'physics_coefficients', 'physics_mean', 'physics_std', 'blocks.0.weight')},
            'decoder_state_dict': {'projection.weight': torch.ones(2, 2),
                                   'bn.running_mean': torch.zeros(2),
                                   'bn.num_batches_tracked': torch.tensor(0)},
        }

    def fake_load(self, run, protocol, *, source, region, seed, relation):
        self.assertEqual(source, 'discovered')
        self.assertEqual(region, 'low_enstrophy')
        self.assertEqual(run.name, run_relation_ablation.run_id(relation, seed, protocol))
        checkpoint = self.checkpoint()
        checkpoint['model_state_dict']['blocks.0.weight'].fill_(
            run_relation_ablation.VARIANTS.index(relation))
        return checkpoint, None, {'test': {
            key: float(seed+run_relation_ablation.VARIANTS.index(relation))
            for key in ('accuracy', 'eta_ns', 'eta_div', 'eta_pp', 'eta_energy')}}

    def test_shared_state_requires_condition_normalization_and_decoder_not_local_blocks(self):
        reference = self.checkpoint()
        candidate = copy.deepcopy(reference)
        candidate['model_state_dict']['blocks.0.weight'].zero_()
        run_relation_ablation.assert_matched_shared_state(reference, candidate)
        for group, key in (
                ('model_state_dict', 'encoder.context_gate'),
                ('model_state_dict', 'encoder.local_backbone.0.weight'),
                ('model_state_dict', 'encoder.context_backbone.0.weight'),
                ('model_state_dict', 'physics_encoder.0.weight'),
                ('model_state_dict', 'condition_fusion.weight'),
                ('model_state_dict', 'label_embed.embed.weight'),
                ('model_state_dict', 'physics_coefficients'),
                ('model_state_dict', 'physics_mean'),
                ('model_state_dict', 'physics_std'),
                ('decoder_state_dict', 'projection.weight'),
                ('decoder_state_dict', 'bn.running_mean'),
                ('decoder_state_dict', 'bn.num_batches_tracked')):
            changed = copy.deepcopy(candidate)
            changed[group][key].add_(1)
            with self.subTest(key=key), self.assertRaises(ValueError):
                run_relation_ablation.assert_matched_shared_state(reference, changed)
        changed = copy.deepcopy(candidate)
        del changed['decoder_state_dict']['bn.running_mean']
        with self.assertRaisesRegex(ValueError, 'decoder state keys'):
            run_relation_ablation.assert_matched_shared_state(reference, changed)

    def test_bitwise_matching_rejects_equal_values_with_different_dtype_or_signed_zero(self):
        reference = self.checkpoint()
        changed = copy.deepcopy(reference)
        changed['model_state_dict']['physics_std'] = changed['model_state_dict']['physics_std'].double()
        with self.assertRaisesRegex(ValueError, 'bitwise identical'):
            run_relation_ablation.assert_matched_shared_state(reference, changed)
        changed = copy.deepcopy(reference)
        changed['decoder_state_dict']['bn.running_mean'].fill_(-0.0)
        with self.assertRaisesRegex(ValueError, 'decoder.*bitwise identical'):
            run_relation_ablation.assert_matched_shared_state(reference, changed)

    def test_provenance_lists_actual_run_ids_in_declared_seed_order(self):
        for protocol in (ExperimentProtocol(device='cpu'),
                         ExperimentProtocol(16, 16, device='cpu')):
            with self.subTest(prefix=protocol.prefix), \
                    patch.object(run_relation_ablation, 'load_run', side_effect=self.fake_load) as load:
                artifact = run_relation_ablation.aggregate_runs(protocol)
            self.assertEqual(load.call_count, 9)
            metadata = artifact['protocol']
            self.assertIs(metadata['shared_condition_and_decoder_verified'], True)
            self.assertEqual(metadata['ns_control'],
                             'separately executed, same seeds/settings as main')
            self.assertIn('bitwise checkpoint verification', metadata['condition_matching'])
            self.assertIn('normalization buffers', metadata['condition_matching'])
            self.assertIn('decoder parameters/buffers', metadata['condition_matching'])
            for variant in run_relation_ablation.VARIANTS:
                relation = '' if variant == 'ns' else '_'+variant
                weight = '0p01' if protocol.prefix == 'full_ns_v4' else '0p1'
                expected = [f'{protocol.prefix}_discovered_low_enstrophy_lambda{weight}'
                            f'_seed{seed}{relation}_eqablation'
                            for seed in run_relation_ablation.SEEDS]
                self.assertEqual(metadata['run_ids'][variant], expected)
                actual_calls = [call.args[0].name for call in load.call_args_list
                                if call.kwargs['relation'] == variant]
                self.assertEqual(actual_calls, expected)
                self.assertEqual(artifact['results'][variant]['accuracy']['values'],
                                 [float(seed+run_relation_ablation.VARIANTS.index(variant))
                                  for seed in run_relation_ablation.SEEDS])

    def test_aggregate_only_never_launches_training_even_if_runs_are_missing(self):
        protocol = ExperimentProtocol(device='cpu')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(run_relation_ablation, 'ROOT', root), \
                    patch.object(run_relation_ablation, 'load_run', side_effect=self.fake_load), \
                    patch.object(run_relation_ablation.subprocess, 'run') as train, \
                    patch('sys.argv', ['run_relation_ablation.py', '--aggregate-only', '--device', 'cpu']), \
                    patch('builtins.print'):
                run_relation_ablation.main()
            train.assert_not_called()
            path = protocol.aggregate_path(root, 'relation_ablation')
            original = path.read_bytes()
            with patch.object(run_relation_ablation, 'ROOT', root), \
                    patch.object(run_relation_ablation, 'load_run', side_effect=FileNotFoundError('missing checkpoint')), \
                    patch.object(run_relation_ablation.subprocess, 'run') as train, \
                    patch('sys.argv', ['run_relation_ablation.py', '--aggregate-only', '--device', 'cpu']), \
                    self.assertRaises(FileNotFoundError):
                run_relation_ablation.main()
            train.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_mismatched_decoder_does_not_overwrite_aggregate(self):
        protocol = ExperimentProtocol(device='cpu')
        def mismatched_load(*args, **kwargs):
            result = self.fake_load(*args, **kwargs)
            if kwargs['relation'] == 'ns_pp':
                result[0]['decoder_state_dict']['projection.weight'].add_(1)
            return result
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = protocol.aggregate_path(root, 'relation_ablation')
            path.parent.mkdir(parents=True)
            path.write_text('previous aggregate', encoding='utf-8')
            with patch.object(run_relation_ablation, 'ROOT', root), \
                    patch.object(run_relation_ablation, 'load_run', side_effect=mismatched_load), \
                    patch.object(run_relation_ablation.subprocess, 'run') as train, \
                    patch('sys.argv', ['run_relation_ablation.py', '--aggregate-only', '--device', 'cpu']), \
                    self.assertRaisesRegex(ValueError, 'decoder.*bitwise identical'):
                run_relation_ablation.main()
            train.assert_not_called()
            self.assertEqual(path.read_text(encoding='utf-8'), 'previous aggregate')


class TestCentreTargetNoise(unittest.TestCase):
    def test_32_context_recomputes_only_centre16_physics(self):
        rng = np.random.default_rng(20260718)
        physical = rng.standard_normal((1, 4, 32, 32, 32)).astype(np.float32)
        means = np.asarray([0.2, -0.1, 0.3, 0.05], dtype=np.float32)
        stds = np.asarray([1.2, 0.8, 1.5, 0.6], dtype=np.float32)
        standardized = ((physical-means[None, :, None, None, None])
                        / stds[None, :, None, None, None])
        fields = torch.from_numpy(standardized)
        mean = torch.from_numpy(means).view(1, 4, 1, 1, 1)
        std = torch.from_numpy(stds).view(1, 4, 1, 1, 1)
        dx = 2*np.pi/64
        actual = run_noise.ns_terms_from_standardized(fields, mean, std, dx, 16)
        expected = _ns_energy_terms(physical[0, :, 8:24, 8:24, 8:24], dx)
        np.testing.assert_allclose(actual[0].numpy(), expected, rtol=2e-5, atol=2e-5)
        altered = fields.clone()
        altered[..., :8, :, :] += 1000
        torch.testing.assert_close(
            run_noise.ns_terms_from_standardized(altered, mean, std, dx, 16), actual,
            rtol=0, atol=0)

    def test_observation_noise_changes_but_inference_rng_stays_fixed(self):
        class FakeModel:
            def __init__(self): self.records = []
            def eval(self): pass
            def __call__(self, fields, ns_terms, **kwargs):
                latent = torch.randn(1, 2)
                self.records.append((fields.clone(), ns_terms.clone(), latent.clone()))
                return torch.tensor([[1.0, 0.0]]), [latent]
        model = FakeModel()
        class FakeDecoder:
            def eval(self): pass
            def __call__(self, value): return value
        physics = SimpleNamespace(evaluate_metrics=lambda value: {'eta_ns': 0.0})
        batch = {'field': torch.zeros(1, 4, 32, 32, 32), 'label': torch.tensor([0])}
        means, stds = torch.zeros(1, 4, 1, 1, 1), torch.ones(1, 4, 1, 1, 1)
        for level, repetition in ((0.1, 0), (0.2, 0), (0.1, 1)):
            self.assertEqual(run_noise.evaluate(
                model, FakeDecoder(), physics, [batch], level, repetition,
                means, stds, 0.1, 42, 16), (100.0, 0.0))
        a, b, c = model.records
        torch.testing.assert_close(b[0], 2*a[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[0], c[0]))
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        torch.testing.assert_close(a[2], c[2], rtol=0, atol=0)
        torch.testing.assert_close(a[1], run_noise.ns_terms_from_standardized(
            a[0], means, stds, 0.1, 16))


if __name__ == '__main__':
    unittest.main()
