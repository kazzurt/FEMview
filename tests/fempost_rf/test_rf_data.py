"""Read-only regression checks against the two validated local RF studies.

Run with any Python containing numpy; native checks launch FEMPOST_RF_PYTHON.
These tests use saved solutions only and never invoke a solver.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from fempost.rf_data import RFStudy, RF_RUNTIME


RESULTS = Path(os.environ.get("FEMVIEW_RF_FIXTURES", str(Path(__file__).resolve().parents[1] / "external/rf")))
CAVITY = RESULTS / "emerge-cylindrical-cavity"
GUIDE = RESULTS / "emerge-wr340-waveguide"


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@unittest.skipUnless(CAVITY.exists() and GUIDE.exists(), "Local benchmark studies unavailable")
class SavedRFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = {root: snapshot(root) for root in (CAVITY, GUIDE)}

    @classmethod
    def tearDownClass(cls):
        for root, before in cls.before.items():
            if snapshot(root) != before:
                raise AssertionError(f"A source result file changed: {root}")

    def test_cavity_all_modes_preserve_complex_fields_and_meter_coordinates(self):
        with RFStudy(CAVITY) as study:
            self.assertEqual(study.kind, "eigenmode")
            self.assertEqual(len(study.frequencies_hz), 10)
            self.assertEqual(study.axes[0].shape, (35,))
            self.assertAlmostEqual(study.axes[2][-1], .099)
            self.assertIn("Arbitrary", study.normalization)
            self.assertIn("TE112", study.labels[6])
            with np.load(study.path / "sampled_fields.npz") as saved:
                expected_e, expected_h = saved["E"], saved["H"]
            for index in range(10):
                frame = study.sample(index)
                np.testing.assert_equal(frame.E, expected_e[index])
                np.testing.assert_equal(frame.H, expected_h[index])
                self.assertFalse(frame.E.flags.writeable)
            for bad in (-1, 10):
                with self.assertRaises(IndexError):
                    study.sample(bad)

    def test_guide_sweep_and_saved_1w_samples(self):
        with RFStudy(GUIDE) as study:
            self.assertEqual(study.kind, "driven")
            np.testing.assert_allclose(study.frequencies_hz, np.linspace(2e9, 4e9, 41))
            self.assertEqual(study.sparameters.shape, (41, 2, 2))
            frame = study.sample(9)
            self.assertEqual(frame.frequency_hz, 2.45e9)
            self.assertIn("1 W", frame.normalization)
            power = .5 * np.real(np.cross(frame.E, frame.H.conj()))
            with np.load(study.path / "fields_2p45GHz.npz") as saved:
                np.testing.assert_allclose(power, saved["Savg"], rtol=1e-14, atol=1e-13)
            net = np.trapezoid(np.trapezoid(power[..., 2], x=study.axes[0], axis=0), x=study.axes[1], axis=0)
            self.assertAlmostEqual(float(np.mean(net)), 1.0, delta=.001)

    def test_cache_rejects_original_tree(self):
        for source in (CAVITY, GUIDE):
            with patch.dict(os.environ, {"FEMPOST_RF_CACHE": str(source / "unsafe-cache")}):
                with self.assertRaisesRegex(ValueError, "outside the original"):
                    RFStudy(source)
            self.assertFalse((source / "unsafe-cache").exists())

    @unittest.skipUnless(RF_RUNTIME.exists(), "Existing EMerge runtime unavailable")
    def test_native_duplicate_tabs_publish_same_cache_entry(self):
        cache_parent = Path(__file__).resolve().parents[1] / "structural-fea-environment" / "temporary" / "fempost-rf-verification"
        cache_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="parallel-native-data-", dir=cache_parent) as cache:
            self.assertTrue(Path(cache).resolve().is_relative_to(cache_parent.resolve()))
            with patch.dict(os.environ, {"FEMPOST_RF_CACHE": cache}):
                with RFStudy(GUIDE) as first, RFStudy(GUIDE) as second:
                    self.assertEqual(first.cache_path, second.cache_path)
                    barrier = threading.Barrier(2)
                    def read(study):
                        barrier.wait(timeout=10)
                        return study.sample(3, 2)
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        a, b = list(pool.map(read, (first, second)))
                    np.testing.assert_array_equal(a.E, b.E)
                    np.testing.assert_array_equal(a.H, b.H)
                    self.assertFalse(list(first.cache_path.glob("*.pending.npz")))
                    with np.load(first.cache_path / "field-0003-port-2.npz") as published:
                        np.testing.assert_array_equal(published["E"], a.E)
                    # A dead reader is replaceable without retaining its old handles.
                    old_process = first._bridge.process
                    old_log = first._bridge._log
                    old_process.terminate()
                    old_process.wait(timeout=5)
                    first.probe(3, [.04318, .02159, .2286], 2)
                    self.assertTrue(old_process.stdin.closed)
                    self.assertTrue(old_process.stdout.closed)
                    self.assertTrue(old_log.closed)
                    self.assertIsNot(old_process, first._bridge.process)

    @unittest.skipUnless(RF_RUNTIME.exists(), "Existing EMerge runtime unavailable")
    def test_native_every_frequency_both_excitations_and_exact_probe(self):
        cache_parent = Path(__file__).resolve().parents[1] / "structural-fea-environment" / "temporary" / "fempost-rf-verification"
        cache_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native-data-", dir=cache_parent) as cache:
            self.assertTrue(Path(cache).resolve().is_relative_to(cache_parent.resolve()))
            with patch.dict(os.environ, {"FEMPOST_RF_CACHE": cache}):
                with RFStudy(GUIDE) as study:
                    point = [axis[len(axis) // 2] for axis in study.axes]
                    original = study.sample(9)
                    inspected = study._bridge.request(command="inspect")
                    native_s = np.asarray(inspected["S"])
                    np.testing.assert_array_equal(native_s[..., 0] + 1j * native_s[..., 1], study.sparameters)
                    exact = study.probe(9, point, 1)
                    np.testing.assert_allclose(exact["E"], original.E[16, 10, 100], atol=1e-12)
                    np.testing.assert_allclose(exact["H"], original.H[16, 10, 100], atol=1e-12)
                    outside = study.probe(9, [-1., -1., -1.])
                    self.assertFalse(outside["valid"])
                    self.assertIsNone(outside["E"])
                    for index, frequency in enumerate(study.frequencies_hz):
                        for port, direction in ((1, 1), (2, -1)):
                            result = study.probe(index, point, port)
                            self.assertTrue(result["valid"])
                            self.assertEqual(result["frequency_hz"], frequency)
                            self.assertGreater(direction * result["Savg"][2], 0)
                    first = study.sample(0, 1)
                    last = study.sample(40, 2)
                    self.assertIn("port 2", last.normalization)
                    np.testing.assert_allclose(study.probe(0, point, 1)["E"], first.E[16, 10, 100], atol=1e-12)
                    np.testing.assert_allclose(study.probe(40, point, 2)["H"], last.H[16, 10, 100], atol=1e-12)
                    self.assertFalse(np.allclose(first.E, last.E))
                    bridge_process = study._bridge.process
                self.assertIsNotNone(bridge_process.poll())
                with RFStudy(CAVITY) as study:
                    ijk = (10, 17, 12)
                    point = [axis[i] for axis, i in zip(study.axes, ijk)]
                    for index in range(10):
                        native = study.probe(index, point)
                        saved = study.sample(index)
                        self.assertTrue(native["valid"])
                        np.testing.assert_allclose(native["E"], saved.E[ijk], rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(native["H"], saved.H[ijk], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
