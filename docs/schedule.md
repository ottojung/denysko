# Denysko scheduled maintenance

This is the Denysko-specific entry point for a recurring scheduled orchestrator.

Do **not** duplicate Lubko's scheduling protocol here. Use the canonical Lubko documents directly:

- Scheduled orchestrator guide: <https://github.com/ottojung/lubko/blob/main/docs/skills/scheduled.md>
- Lubko operating skill: <https://github.com/ottojung/lubko/blob/main/docs/SKILL.md>

Target repository: <https://github.com/ottojung/denysko>

## Required Denysko documents

At the start of every scheduled run, after reading the Lubko scheduling/operating guides, read and obey:

- Denysko agent instructions: <https://github.com/ottojung/denysko/blob/master/AGENTS.md>
- Denysko maintenance constitution: <https://github.com/ottojung/denysko/blob/master/docs/constitution.md>

The constitution is especially important: it defines the allowed maintenance scope, Denysko-specific correctness rules, PR verification requirements, and the mandatory `Hello, World!` visual artifact.

## Scheduled-task scope

Focus only on `ottojung/denysko` as the target repository. Lubko is the execution/orchestration platform, not the target project.

Follow Lubko's scheduled guide for ownership, recovery, issue selection, worktrees, managed agents, polling, review, release branches, and merge boundaries. Do not restate or replace those rules here.

For Denysko-specific work:

- prefer inheriting/continuing existing scheduled work exactly as the Lubko scheduled guide requires;
- otherwise work on actionable existing Denysko issues or reproducible Denysko bugs;
- obey the maintenance scope freeze in `docs/constitution.md`;
- scheduled maintenance must not invent new product features;
- new issues created by the scheduled orchestrator may only be focused reproducible bug reports;
- review implementation PRs independently rather than treating tests as proof;
- do not consider a task complete until the constitution's required public-CLI checks and `Hello, World!` Matplotlib-from-emitted-equations PR image are satisfied.

## Minimal scheduled-task prompt

A scheduled task may simply say:

> Target repository: <https://github.com/ottojung/denysko>
>
> Follow: <https://github.com/ottojung/denysko/blob/master/docs/schedule.md>

All detailed scheduling/orchestration mechanics belong to Lubko's canonical scheduled guide. All Denysko-specific maintenance policy belongs to `AGENTS.md` and especially `docs/constitution.md`.