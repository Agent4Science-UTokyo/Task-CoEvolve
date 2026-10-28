# harbor reference (read-only)

Vendored copy of the `harbor` package's Terminus2 implementation, for the
scaffold-evolution proposer to read when it copies `agents/baseline_terminus2.py`
(a 1-line alias, `AgentHarness = Terminus2`) as a starting point instead of
`agents/baseline_kira.py`.

- **Source:** `harbor==0.18.0`, `harbor/agents/terminus_2/` and `harbor/trial/`
- **Not used at runtime.** Execution always imports the real `harbor` package
  from the project's `.venv`; these files exist only so the proposer has a
  stable, version-controlled path to read instead of guessing a
  `site-packages` path or running `find`.
- **Not kept in sync automatically.** If the pinned `harbor` version changes,
  re-copy these three files from the new version's install.

| File | Original path |
|---|---|
| `agents/terminus_2/terminus_2.py` | `harbor/agents/terminus_2/terminus_2.py` (main class, tool schema, agent loop) |
| `agents/terminus_2/tmux_session.py` | `harbor/agents/terminus_2/tmux_session.py` (command execution) |
| `trial/trial.py` | `harbor/trial/trial.py` (timeout/lifecycle) |
