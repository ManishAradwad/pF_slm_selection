import os
import pathlib
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ACTIVATE = ROOT / "scripts" / "activate_wsl.sh"
SETUP = ROOT / "scripts" / "setup_wsl.sh"


def _is_native_wsl2() -> bool:
    release_file = pathlib.Path("/proc/sys/kernel/osrelease")
    if not release_file.is_file() or pathlib.Path("/.dockerenv").exists():
        return False
    return "microsoft" in release_file.read_text(encoding="utf-8").lower() and "wsl2" in (
        release_file.read_text(encoding="utf-8").lower()
    )


def test_wsl_entry_points_are_valid_bash() -> None:
    for script in (ACTIVATE, SETUP):
        subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", [ACTIVATE, SETUP])
def test_entry_points_require_native_wsl2_ext4_and_reject_containers(
    script: pathlib.Path,
) -> None:
    source = script.read_text(encoding="utf-8")

    assert "/proc/sys/kernel/osrelease" in source
    assert "WSL2" in source
    assert "/.dockerenv" in source
    assert "/run/.containerenv" in source
    assert "/mnt/*" in source
    assert "findmnt" in source
    assert 'repo_fstype" != "ext4"' in source or 'repo_fstype" == "ext4"' in source

    guard_position = source.index("require_native_wsl2_ext4")
    if script == ACTIVATE:
        guard_position = source.index("pf_activate_require_native_wsl2_ext4")
        first_mutation = source.index('mkdir -p "$HF_HOME"')
    else:
        first_mutation = source.index('"$UV_BIN" venv')
    assert guard_position < first_mutation


@pytest.mark.parametrize("script", [ACTIVATE, SETUP])
def test_non_linux_host_fails_before_environment_mutation(
    script: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text("#!/bin/sh\nprintf 'Darwin\\n'\n", encoding="utf-8")
    fake_uname.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    if script == ACTIVATE:
        command = [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            (
                'if source "$1"; then rc=0; else rc=$?; fi; '
                'printf "source_rc=%s\\nshell_alive=yes\\n" "$rc"'
            ),
            "bash",
            str(script),
        ]
    else:
        command = ["bash", str(script)]

    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)

    assert result.returncode != 0 or "source_rc=1" in result.stdout
    assert "Native WSL2 is required" in result.stderr
    if script == ACTIVATE:
        assert "source_rc=1" in result.stdout
        assert "shell_alive=yes" in result.stdout


@pytest.mark.skipif(not _is_native_wsl2(), reason="requires a native WSL2 host")
def test_activation_can_be_sourced_from_native_ext4_checkout() -> None:
    if not (ROOT / ".venv" / "bin" / "python").is_file():
        pytest.skip("native repository virtual environment is not installed")

    env = os.environ.copy()
    env["PF_VENV"] = str(ROOT / ".venv")
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'source "$1" && printf "activated=%s\\n" "$VIRTUAL_ENV"',
            "bash",
            str(ACTIVATE),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"activated={ROOT / '.venv'}" in result.stdout
