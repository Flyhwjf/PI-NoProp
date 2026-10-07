import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.hit_dataset import build_learning_cache


class TestContextInputCache(unittest.TestCase):
    @staticmethod
    def _write_dns(root: Path):
        records = []
        splits = ('discovery', 'discovery', 'validation', 'test')
        grid = np.indices((8, 8, 8), dtype=np.float64)
        x, y, z = grid
        for trajectory_id, split in enumerate(splits):
            directory = root/f'trajectory_{trajectory_id:03d}'
            directory.mkdir(parents=True)
            scale = 1.0+0.4*trajectory_id
            for frame in range(3):
                velocity = scale*np.stack([
                    np.sin((x+frame)/3)+0.1*y,
                    np.cos((y+frame)/3)+0.1*z,
                    np.sin((z+frame)/3)+0.1*x,
                ]).astype(np.float32)
                pressure = (0.1*x+0.2*y-0.05*z+frame).astype(np.float32)
                np.savez(directory/f'frame_{frame:04d}.npz',
                         velocity=velocity, pressure=pressure)
            records.append({
                'trajectory_id': trajectory_id,
                'split': split,
                'config': {'box_length': float(2*np.pi)},
            })
        (root/'manifest.json').write_text(json.dumps({
            'trajectories': records}), encoding='utf-8')

    def test_larger_context_is_centred_on_the_smaller_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, cache = root/'dns', root/'cache'
            source.mkdir()
            self._write_dns(source)
            metadata = build_learning_cache(
                source, cache, samples_per_trajectory=2,
                spatial_size=6, target_spatial_size=4,
                time_window=3, n_classes=2)

            fields = np.load(cache/'fields.npy')
            sequences = np.load(cache/'sequences.npy')
            self.assertEqual(fields.shape, (8, 4, 6, 6, 6))
            self.assertEqual(sequences.shape, (8, 3, 4, 4, 4, 4))
            np.testing.assert_allclose(
                fields[:, :, 1:5, 1:5, 1:5], sequences[:, 0])
            self.assertEqual(metadata['input_spatial_size'], 6)
            self.assertEqual(metadata['target_spatial_size'], 4)
            self.assertEqual(metadata['context_boundary'],
                             'periodic wrap on the generated HIT box')

            with self.assertRaises(ValueError):
                build_learning_cache(
                    source, cache, samples_per_trajectory=2,
                    spatial_size=6, target_spatial_size=6,
                    time_window=3, n_classes=2)

    def test_formal_32_context_preserves_the_16_target_task(self):
        legacy = Path('data/cache_hit_ns')
        context = Path('data/cache_hit_ns_input32_target16')
        if not (legacy/'fields.npy').exists() or not (context/'fields.npy').exists():
            self.skipTest('formal paired caches have not both been built')
        for name in ('sequences', 'labels', 'splits', 'regions',
                     'trajectory_ids', 'ns_terms'):
            np.testing.assert_array_equal(
                np.load(legacy/f'{name}.npy', mmap_mode='r'),
                np.load(context/f'{name}.npy', mmap_mode='r'))
        legacy_fields = np.load(legacy/'fields.npy', mmap_mode='r')
        context_fields = np.load(context/'fields.npy', mmap_mode='r')
        self.assertEqual(context_fields.shape[-3:], (32, 32, 32))
        np.testing.assert_array_equal(
            legacy_fields, context_fields[:, :, 8:24, 8:24, 8:24])
        legacy_stats = np.load(legacy/'stats.npz')
        context_stats = np.load(context/'stats.npz')
        for key in legacy_stats.files:
            np.testing.assert_array_equal(legacy_stats[key], context_stats[key])


if __name__ == '__main__':
    unittest.main()
