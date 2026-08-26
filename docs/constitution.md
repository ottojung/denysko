# Denysko maintenance constitution

This document is the **project-specific task definition** for an unattended ChatGPT orchestrator maintaining Denysko.

It deliberately does **not** duplicate Lubko's orchestration, scheduling, recovery, ownership, polling, branch, review, or release-branch procedures. Read and follow those documents directly:

- Lubko operating skill: <https://github.com/ottojung/lubko/blob/main/docs/SKILL.md>
- Scheduled-orchestrator guide: <https://github.com/ottojung/lubko/blob/main/docs/skills/scheduled.md>

Target repository: <https://github.com/ottojung/denysko>

## Maintenance scope freeze

<u><strong>Scheduled maintenance must not add new features to Denysko.</strong></u>

The scheduled orchestrator may only:

- fix reproducible bugs; and
- solve work that was already represented by an existing Denysko GitHub issue before the orchestrator selected it.

The scheduled orchestrator must **not** invent new product features, broaden the product scope on its own, or open new issues for enhancements, feature ideas, speculative refactors, or other non-bug work.

If scheduled work discovers a new reproducible bug and no issue already tracks it, the orchestrator may create a focused bug issue containing the reproduction and observed incorrect behavior. **New issues created by scheduled maintenance must be bug reports.**

An existing issue remains valid scheduled work even when resolving it changes behavior or adds capability: the human-created issue is the authorization boundary. The orchestrator itself does not create that new product scope.

## Mission

Maintain and improve Denysko by working through its existing GitHub issues and by fixing reproducible correctness, geometry, rendering, numerical-stability, text-layout, and usability bugs discovered while doing that work, subject to the maintenance scope freeze above.

The orchestrator should make real progress, not merely triage or report. Select actionable work according to the Lubko scheduled-orchestrator guide, delegate implementation through Lubko, review the resulting PR itself, verify it independently, and iterate until the issue is actually solved.

If a reproducible Denysko bug is discovered and no issue describes it, create a focused GitHub bug issue with the reproduction and observed behavior before losing the discovery. If it blocks the current task, fix it as part of the current work or as an explicit prerequisite; otherwise leave it as actionable follow-up work. Do not create follow-up issues for features, enhancements, speculative improvements, or non-bug refactors.

Current known regressions must not be normalized as expected behavior. If still reproducible and not already tracked, they are bugs to investigate and fix generically.

## Denysko-specific engineering rules

Preserve the program's central architecture unless an existing issue explicitly establishes that it must change:

- topology and geometry are decided before polynomial optimization;
- the optimizer must not discover or silently repair topology;
- emitted curves remain ordinary globally unbounded polynomials `y=f(x)` with no domain restrictions;
- single-character generation remains the primitive from which text generation is composed;
- fixes must be generic geometric/numerical rules, not character-name special cases;
- do not weaken correctness checks merely to make a troublesome glyph pass;
- deterministic behavior for a fixed seed is part of the contract;
- tests are required evidence, but visual output is also part of correctness for this project.

Before changing behavior that is already documented by an open issue, read the issue carefully and implement its intended rule rather than merely matching one example glyph.

## Work scope

Open GitHub issues are the primary backlog. The orchestrator may also fix bugs discovered through tests, the required visual smoke test below, code review, or manual investigation. The orchestrator may open a new issue only for a reproducible bug; all non-bug work must already have an issue created outside scheduled maintenance.

Prefer work that improves user-visible correctness or removes blockers to the main contract. In particular, failures to generate supported ASCII letters, obviously wrong stroke routing, wrong escape direction, broken relative glyph sizing, malformed text composition, or numerically incorrect emitted equations are correctness bugs rather than cosmetic cleanup.

Keep PRs coherent. One PR may fix more than one issue only when the fixes are inseparable or one is a necessary prerequisite for the other. Otherwise prefer a focused PR with a clear issue relationship.

## Required verification for every task PR

A task PR is not ready merely because its targeted unit tests pass.

Before considering a PR complete:

1. run the repository's normal test suite and any issue-specific regressions;
2. exercise the affected real glyphs/text through the public Denysko CLI;
3. independently review the actual PR diff as required by the Lubko operating skill;
4. generate and inspect the mandatory `Hello, World!` visual artifact described below;
5. post that artifact as an image in a PR comment so a human can inspect the rendered result without checking out the branch.

If the exact `Hello, World!` smoke test is currently impossible because of another genuine Denysko bug (for example unsupported punctuation or a glyph that fails generation), treat that as a blocker to the visual contract. Prefer fixing the blocker first or as a prerequisite rather than weakening or silently changing the smoke-test phrase.

## Mandatory `Hello, World!` PR image

Every task PR must contain a comment with a freshly generated PNG rendering of exactly:

```text
Hello, World!
```

using the PR branch's current Denysko implementation and the default deterministic seed (`42` unless the project contract changes).

The image is a **smoke-test artifact**, not decoration. It must be rendered by Matplotlib from the **actual equations emitted by the public Denysko CLI**. Do not render the font outline, raster mask, skeleton, route corridor, or internal `PathFit` objects instead of the emitted equations.

The intended pipeline is:

```text
public Denysko CLI
    -> emitted y=f(x) equation lines
    -> parse/evaluate those emitted equations
    -> sample them over the text viewport
    -> Matplotlib PNG
    -> image posted in a PR comment
```

Use a stable repository helper for this if one exists. If it does not exist yet, add a small deterministic preview helper rather than reimplementing ad-hoc plotting logic in every task.

The preview helper should:

- invoke or exercise the same public serialization path users receive;
- evaluate the emitted equations themselves;
- use dense enough x sampling that ordinary glyph strokes are visibly smooth;
- use a fixed y viewport that includes the glyph band and clips the intentionally unbounded escape tails after they leave the visible region;
- choose an x viewport covering the laid-out text;
- produce a reasonably wide PNG suitable for reading in a GitHub PR comment;
- avoid adding domain restrictions to the equations merely for plotting.

The exact Matplotlib aesthetics are not part of the contract. The important property is that a reviewer can see whether `Hello, World!` is recognizably and correctly produced from the emitted curves.

### Posting the image

Post the PNG **under the PR as a comment**, not merely as a local file path or a statement that it was generated.

Use the ordinary GitHub image-attachment mechanism when available. If the execution environment cannot directly upload a comment attachment, use a reproducible GitHub-hosted image URL tied to the PR/commit (for example a temporary preview artifact committed on the task branch and referenced by exact commit SHA), and avoid merging disposable preview files into the product tree when they are not intended project assets.

The PR comment should also state the exact command used to generate the equations/preview and the seed.

If preview generation fails, the PR is not ready. Investigate the failure rather than posting an older image.

## Visual review expectations

The orchestrator must actually inspect the generated image before approving/merging the task PR.

Look for obvious regressions such as:

- missing letters or missing required strokes;
- lowercase and uppercase proportions that contradict the selected font;
- routes escaping at visually wrong junctions;
- tails going in an obviously wrong direction;
- strokes that should join but visibly break apart;
- disconnected components whose tails run toward each other when the intended rule is to separate them;
- translated letters that deform relative to the same glyph generated alone;
- numerical blow-ups, nearly vertical serialization artifacts, or curves re-entering the text unexpectedly;
- spacing/layout regressions that make the phrase unreadable.

A bad preview is evidence of a bug even when tests are green. Investigate it, add a regression where practical, and fix the underlying generic rule.

## Regression discipline

When fixing a real glyph bug, add a regression that captures the **general mechanism** whenever possible, plus a real-glyph regression for the reported example when that is stable and useful.

Do not rewrite expectations to bless visibly wrong output. Do not remove difficult letters from the supported regression set. The systematic guaranteed test set is ASCII `A-Z`, `a-z`, and space; other text may be best-effort according to the relevant issues/specification.

For changes to fonts, normalization, routing, escape selection, or text layout, re-check representative mixed text through the mandatory preview in addition to targeted tests.

## Completion standard

A task is complete only when:

- the issue's intended behavior is actually implemented;
- relevant tests pass;
- the public CLI behavior has been exercised;
- the PR diff has been independently reviewed by the orchestrator;
- the fresh `Hello, World!` Matplotlib-from-emitted-curves image has been posted under the PR and visually inspected;
- discovered regressions are fixed or durably tracked as GitHub bug issues;
- documentation is updated when the project contract changed.

Use Lubko's linked operating and scheduled-orchestrator documents for all orchestration mechanics. This constitution defines only what successful Denysko maintenance must accomplish.
