# SPEC: Single-Letter → Desmos Polynomial Converter

Version 1.0 · Status: draft · Scope frozen: **one letter per run, nothing else**

---

## 1. Mission

Given exactly one letter, emit a small set of `y = f(x)` polynomial equations that,
pasted into Desmos, render that letter so it is recognizable at a glance.

One letter in → one letter drawn. This tool will not become a text renderer.

## 2. Non-goals (explicit)

- Multi-character input, words, spaces, kerning.
- Parametric curves (`x = f(t)`), implicit relations, inequalities, trig/exotics.
- Font selection, styling, bold/italic variants.
- Interactivity or GUI.

If a feature requires more than one glyph, it is out of scope.

## 3. Interface

```
desmos-letter <letter> [--seed N] [--max-curves K]
```

- `<letter>`: exactly one character from `A`–`Z`. Anything else → exit code 2, no output.
- Output: **stdout only**, one Desmos expression per line. The tool writes no files, ever.

### Expression contract (every emitted line)

1. Form `y=<poly>\ \left\{a\le x\le b\right\}` — a domain restriction is **mandatory**; an unrestricted line is invalid output.
2. Standard power basis only (`x^k`, decimal coefficients). No scientific notation, no functions, no variables other than `x`.
3. **Minimum sufficient degree**: each curve uses the smallest degree that is sufficient,
   where *sufficient* means the curve's residuals on its assigned points all satisfy
   `|c(x) − y| ≤ τ` (the coverage tolerance from §6). The fitter must try degrees in
   ascending order and stop at the first sufficient one — never pad degrees "just in case."
4. At most `K` expressions total (default **12**) — keeps results usable in Desmos free tier.
5. Domain endpoints within `[−5, 105]`; coefficients finite and `|c| < 10^9`.

## 4. Geometry contract

- The glyph is rendered in a fixed frame: `font_size = 100`, then normalized so the
  filled shape fits `[0, 100] × [0, 100]`, baseline at `y = 0`, **y-axis pointing up**.
- Normalization happens once, before fitting. All downstream math sees these coordinates only.

## 5. Algorithm requirements

- Point source: rasterize the glyph outline (holes fall out naturally) → point cloud `P`.
- Search may be stochastic but **must be reproducible**: same `(letter, seed)` ⇒ identical output. Seed has a fixed default.
- Fitness must reward *coverage*: minimize over curves c of vertical distance `|c(x) − y|`
  for points in `P`, with outlier tolerance so stray pixels don't dominate.
  (Monotonicity caveat: fitness must decrease as points get *closer* to curves.)

## 6. Built-in validation gate (the tool grades itself)

After fitting, before emitting anything:

| # | Check | Threshold (defaults) |
|---|-------|----------------------|
| V1 | Coverage: fraction of `P` within vertical distance `τ` of the nearest curve | `τ = 2` units, ≥ **95 %** |
| V2 | Confinement: every curve, evaluated on its own domain, stays inside bbox grown by margin | margin = **5 units** |
| V3 | Round-trip: each emitted line parses back to a polynomial via the project's parser | 100 % parse |

Any failure ⇒ exit code 1, print which check failed, print nothing else.
No silent bad output, ever.

## 7. Acceptance criteria

- All 26 letters pass V1–V3 with defaults.
- Three reference letters pinned in CI with exact-output regression (seeded):
  `A` (diagonals + crossbar), `O` (closed hole), `R` (mixed strokes + hole).
- End-to-end runtime ≤ **30 s** per letter on a laptop-class machine.

## 8. Dependencies

Python ≥ 3.11, numpy, matplotlib. Nothing else. (pygad is removed, not optional.)
