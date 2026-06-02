import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def pytest_sessionfinish(session, exitstatus):
    cleanup_targets = [
        PROJECT_DIR / "__pycache__",
        PROJECT_DIR / ".pytest_cache",
        PROJECT_DIR / ".test_tmp",
        PROJECT_DIR / "tests" / "__pycache__",
    ]
    cleanup_targets.extend(PROJECT_DIR.glob("pytest-cache-files-*"))
    cleanup_targets.extend((PROJECT_DIR / "tests").glob("_tmp_pareto_artifacts*"))

    for target in cleanup_targets:
        _remove_inside_project(target)


def _remove_inside_project(target: Path) -> None:
    try:
        resolved = target.resolve()
    except FileNotFoundError:
        return
    if not _is_inside(resolved, PROJECT_DIR):
        raise RuntimeError(f"Refusing to clean path outside project: {resolved}")
    try:
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
    except (FileNotFoundError, PermissionError):
        return


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent.resolve())
        return True
    except ValueError:
        return False
