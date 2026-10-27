"""Mesh statistics on the tet10 and hex20 fixtures."""
import json

import numpy as np
import pytest

from fempost import mesh_stats
from fempost.bundle import RunBundle, _json_default


def test_tet10_quality_and_statistics(friction, tmp_path):
    bundle = RunBundle(friction)
    q = mesh_stats.quality(bundle.mesh)
    assert q.scaled_jacobian.shape == (48,)
    assert np.all(q.scaled_jacobian > 0) and np.all(q.scaled_jacobian <= 1.0)
    assert np.all(q.aspect_ratio >= 1.0)
    assert np.nansum(q.volume) == pytest.approx(10.0 * 10.0 * 2.0, rel=1e-6)   # two 10 x 10 x 1 blocks
    assert np.all(q.min_edge > 0) and np.all(q.max_edge >= q.min_edge)
    assert any("corner nodes" in n for n in q.notes)
    stats = mesh_stats.statistics(bundle, q)
    json.dumps(stats, default=_json_default)
    assert stats["counts"]["elements"] == 48 and stats["counts"]["solid_elements"] == 48
    assert stats["bodies"][0]["body"] == "eall" and stats["bodies"][0]["volume_mm3"] == pytest.approx(200.0, rel=1e-6)
    assert stats["sets"]["surfaces"] == {"MASTER": 8, "SLAVE": 8}
    assert stats["sets"]["node_sets"]["NALL"] == 150
    assert stats["quality"]["scaled_jacobian"]["count"] == 48 and stats["quality"]["inverted_elements"] == 0
    assert sum(stats["histograms"]["aspect_ratio"]["counts"]) == 48
    png = tmp_path / "hist.png"
    mesh_stats.histogram_png(stats["histograms"]["scaled_jacobian"], "scaled Jacobian", png)
    assert png.stat().st_size > 1000


def test_hex20_corner_quality(short_screw):
    bundle = RunBundle(short_screw)
    q = mesh_stats.quality(bundle.mesh)
    assert np.allclose(q.scaled_jacobian, 1.0, atol=1e-9)        # 5 mm cubes
    assert np.allclose(q.aspect_ratio, 1.0, atol=1e-9)
    assert np.nansum(q.volume) == pytest.approx(20.0 * 10.0 * 10.0)
    assert np.allclose(q.min_edge, 5.0) and np.allclose(q.max_edge, 5.0)
    stats = mesh_stats.statistics(bundle, q)
    assert stats["counts"]["unused_nodes"] == 0
    assert stats["bodies"][0]["nodes"] == 162


def test_codex_mesher_numbers(codex_run):
    bundle = RunBundle(codex_run)
    stats = mesh_stats.statistics(bundle)
    assert stats["mesher"]["tet10"] == 68963 and stats["mesher"]["nodes"] == 119941
    assert stats["counts"]["elements"] == 68963
    assert stats["regions"]["HUB_BODY"] == 10317
    q = mesh_stats.quality(bundle.mesh)
    inverted = int(np.sum(q.scaled_jacobian < 0))
    assert stats["quality"]["inverted_elements"] == inverted          # corner-inverted quadratic tets are reported
    assert stats["quality"]["scaled_jacobian"]["min"] == pytest.approx(float(np.nanmin(q.scaled_jacobian)))
    assert stats["mesher"]["minDetJac_mm3"] > 0
