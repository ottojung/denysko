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


def _bar_points():
    xs = np.arange(0.0, 100.01, 0.5)
    return np.column_stack([xs, np.full(len(xs), 50.0)])


def test_v1_coverage_pass_and_fail():
    p = _bar_points()
    assert d.validate(["y=50\\ \\left\\{0\\le x\\le 100\\right\\}"], p) == []
    problems = d.validate(["y=50\\ \\left\\{10\\le x\\le 20\\right\\}"], p)
    assert any(m.startswith("V1") for m in problems)


def test_v2_boundary_following_curve_passes():
    p = _bar_points()
    assert d.validate(["y=50\\ \\left\\{0\\le x\\le 100\\right\\}"], p) == []


def test_v2_excursion_fails_inside_expanded_bbox():
    p = _bar_points()
    a = 4.9 / 900.0
    coef = np.array([50.0 - 1600.0 * a, 100.0 * a, -a])
    curve = d.XCurve(np.polynomial.Polynomial(coef), 20.0, 80.0)
    line = d.format_expression(curve)
    problems = d.validate([line], p)
    assert any(m.startswith("V2") for m in problems)
    assert not any(m.startswith("V3") for m in problems)


def test_v3_confinement_checked_independently():
    curve = d.XCurve(np.polynomial.Polynomial([-30.0, 0.0, -1.0]), 0.0, 10.0)
    problems = d.validate([d.format_expression(curve)], np.zeros((0, 2)))
    assert any(m.startswith("V3") for m in problems)

    ok = d.XCurve(np.polynomial.Polynomial([50.0]), 0.0, 10.0)
    assert d.validate([d.format_expression(ok)], np.zeros((0, 2))) == []


def test_v4_round_trip_gate():
    p = _bar_points()
    tampered = "y=50.0\\ \\left\\{10\\le x\\le 90\\right\\}"
    problems = d.validate([tampered], p)
    assert any(m.startswith("V4") for m in problems)


def test_malformed_lines_do_not_parse():
    for line in [
        "y=x^^2\\ \\left\\{0\\le x\\le 10\\right\\}",
        "y=x^2 \\left\\{0\\le x\\le 10\\right\\}",
        "y=x^2\\ \\left\\{0\\leq x\\le 10\\right\\}",
        "",
        "y=..x\\ \\left\\{0\\le x\\le 10\\right\\}",
    ]:
        assert d.parse_line(line) is None
