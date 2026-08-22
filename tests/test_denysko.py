import numpy as np

from src import denysko as d


def test_u_coefficients_printed_as_x_regression():
    cand = d.Candidate(1, np.array([100.0, 100.0]), -5.0, 105.0)
    line = d.serialize(cand)
    assert line == "y=2x\\ \\left\\{-5\\le x\\le 105\\right\\}"


def test_serialized_polynomial_matches_internal_values():
    rng = np.random.default_rng(3)
    xs = np.array([-5.0, 0.0, 12.5, 50.0, 77.7, 105.0])
    for _ in range(20):
        degree = int(rng.integers(0, d.MAX_DEGREE + 1))
        coef = rng.normal(0.0, 30.0, size=degree + 1)
        a, b = float(rng.uniform(-5, 50)), float(rng.uniform(50, 105))
        cand = d.Candidate(degree, coef, a, b)
        curve = d.parse_line(d.serialize(cand))
        assert curve is not None
        expected = d._poly_u(coef, xs)
        got = curve.poly(xs)
        assert np.allclose(got, expected, atol=0.05)


def test_round_trip_exact_text():
    lines = [
        "y=2x\\ \\left\\{-5\\le x\\le 105\\right\\}",
        "y=0\\ \\left\\{0\\le x\\le 100\\right\\}",
        "y=x^5-3x^2+0.5x-7\\ \\left\\{0\\le x\\le 100\\right\\}",
        "y=-x^3\\ \\left\\{12.25\\le x\\le 88.5\\right\\}",
    ]
    for line in lines:
        curve = d.parse_line(line)
        assert curve is not None
        assert d.format_expression(curve) == line


def test_round_trip_generated_candidates():
    rng = np.random.default_rng(11)
    for _ in range(30):
        degree = int(rng.integers(0, d.MAX_DEGREE + 1))
        coef = rng.normal(0.0, 200.0, size=degree + 1)
        a = float(rng.uniform(-5.0, 60.0))
        b = float(rng.uniform(a + 0.5, 105.0))
        line = d.serialize(d.Candidate(degree, coef, a, b))
        curve = d.parse_line(line)
        assert curve is not None
        assert d.format_expression(curve) == line


def test_normalization_bbox_corner_at_origin():
    for letter in "AOTM":
        p = d.glyph_boundary(letter)
        assert p[:, 0].min() < 0.5
        assert p[:, 1].min() < 0.5
        assert p[:, 0].max() <= 100.0 + 1e-9
        assert p[:, 1].max() <= 100.0 + 1e-9


def test_sample_graph_preserves_whole_domain_under_cap():
    coef = np.array([50.0, 5000.0])
    samples = d.sample_graph(d.Candidate(1, coef, -5.0, 105.0), 1.0, 400)
    assert len(samples) <= 400
    assert len(samples) > 129
    assert samples[0, 0] == -5.0
    assert samples[-1, 0] == 105.0
    assert np.all(np.diff(samples[:, 0]) > 0)
    expected_first = d._poly_u(coef, np.array([-5.0]))[0]
    expected_last = d._poly_u(coef, np.array([105.0]))[0]
    assert samples[0, 1] == expected_first
    assert samples[-1, 1] == expected_last


def test_malformed_lines_do_not_parse():
    for line in [
        "y=x^^2\\ \\left\\{0\\le x\\le 10\\right\\}",
        "y=x^2 \\left\\{0\\le x\\le 10\\right\\}",
        "y=x^2\\ \\left\\{0\\leq x\\le 10\\right\\}",
        "",
        "y=..x\\ \\left\\{0\\le x\\le 10\\right\\}",
    ]:
        assert d.parse_line(line) is None
