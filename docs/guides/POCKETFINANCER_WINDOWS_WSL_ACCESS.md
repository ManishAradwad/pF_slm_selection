# PocketFinancer access from Windows and WSL

Verified on 2026-10-05 in the Windows Codex desktop app.

## Repository locations

| Repository | Windows path | Linux path |
| --- | --- | --- |
| Shared SMS/data/model research | `\\wsl.localhost\Ubuntu\home\tojinotzenin\pF_slm_selection` | `/home/tojinotzenin/pF_slm_selection` |
| Android | `D:\Personal_Projects\pocket-financer\pocket-financer-android` | `/mnt/d/Personal_Projects/pocket-financer/pocket-financer-android` |
| iOS | `D:\Personal_Projects\pocket-financer\pocket-financer-ios` | `/mnt/d/Personal_Projects/pocket-financer/pocket-financer-ios` |

The locations describe this workstation. Other machines can use their existing
checkouts. Keep private data in its original local checkout. Paths in a guide do
not expand the active session's workspace permissions or authorize unrelated work.

## Diagnose the actual boundary

The Windows sandbox denied distribution enumeration with
`Wsl/EnumerateDistros/Service/E_ACCESSDENIED` and denied reads of the shared UNC
path. The same probes succeeded through Codex's approved outside-sandbox
execution, under the same Windows account. Ubuntu was already running.

A temporary create/read/delete probe passed in all three repositories.
The existing shared environment activated and reported Python 3.11.14.
The test files were removed. These are local access checks, not model, app-build,
or device verification.

Starting the execution tool directly in a denied UNC working directory can fail
with `helper_unknown_error` before PowerShell starts. Use an accessible Windows
working directory to separate that startup failure from the command's result.

## Procedure for agents

1. Read global guidance and each target repository's `AGENTS.md`.
   Check task scope and active workspace roots. The repositories are separate Git
   projects; preserve their branches and unrelated changes.
2. Run a harmless probe from a Windows working directory such as
   `C:\Users\manis`. Useful probes are `wsl.exe --list --verbose` and a read of
   the shared `AGENTS.md`.
3. If the sandbox denies WSL, repeat the probe through the execution tool's
   supported approval mechanism. In Codex use
   `sandbox_permissions="require_escalated"` with a precise justification.
   Automatic review can handle eligible requests when configured.
4. If approved execution works, use it for the authorized WSL operations.
   Do not require the user to repeat earlier access history. Do not treat
   restarting Codex as the default fix for this established failure.
5. If approved execution also fails, diagnose that specific WSL/service error.
   If approval is rejected, report the action and stated reason, respect the
   rejection, and continue unaffected work. Never use another tool to evade it.

Shared invocation from PowerShell:

```powershell
wsl.exe -d Ubuntu --cd /home/tojinotzenin/pF_slm_selection --exec bash -lc 'source scripts/activate_wsl.sh && python --version'
```

The command text alone does not select outside-sandbox execution; the tool's
approval request must do that. Keep real task commands scoped and reviewable.

Use Linux Git and Python inside the shared WSL checkout. Use Windows Git,
PowerShell, and `gradlew.bat` for Android. Windows can edit iOS sources; iOS
builds, XCTest, simulator, and iPhone evidence still require macOS/Xcode.
Using Linux Git on Windows app checkouts can report file-mode changes that
Windows Git does not report. Do not reset, chmod, or normalize the checkout in
response to that noise.

## Why new sessions can find this

The root `AGENTS.md` in each repository includes the path map and recovery
procedure. Android's `copilot-instructions.md` points to that entry guidance.

This workstation also has general Windows/WSL recovery guidance in
`C:\Users\manis\.codex\AGENTS.md`. This global file is outside WSL, so Codex can
load the procedure even when a sandboxed read of the shared repo fails.
Existing communication preferences are preserved.

Codex reads global instructions from its active `CODEX_HOME`, which defaults
to `~/.codex`, then project instructions. A nonempty `AGENTS.override.md`
takes precedence in the same directory. Guidance is discovered at session start;
the current session does not demonstrate a future session's instruction load.
If using a different Windows account, Codex home, or machine, install the same
general procedure in that account's active global guidance. This guide does not
change security settings or promise that all future approvals will succeed.

Official references:

- [Instruction discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
- [Windows sandbox](https://learn.chatgpt.com/docs/windows/windows-sandbox).

## Current source and verification boundaries

The shared successor routing and Python evaluation workflows exist.
Android current operations support successor automatic routing under the opt-in
preference; stored v4 operations retain their review-only behavior.
iOS current v4 operations remain review-only.
These source facts do not establish successful model-driven automatic persistence
or physical-device acceptance. Read the canonical architecture, evaluation
strategy, cross-platform roadmap, and each native next-step guide before making
a runtime claim.
