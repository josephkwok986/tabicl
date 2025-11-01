#!/usr/bin/env python3
"""项目清理脚本：删除常见临时/中间文件，并支持自定义模式。

python my_tools/cleanup.py --dry-run
python my_tools/cleanup.py 

默认会在仓库根目录执行，递归匹配 `DEFAULT_PATTERNS` 中的文件或目录。
如果需要扩展清理范围，可直接修改 `DEFAULT_PATTERNS`，或运行时使用
`--extra` 参数附加新的 glob 模式。传入 `--dry-run` 可预览将被删除的路径。
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Set


# 在这里添加/调整默认清理目标，支持 glob 模式。
DEFAULT_PATTERNS: Sequence[str] = (
    "__pycache__",
    "*.py[cod]",
    "*.so",
    "*.dylib",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".nox",
    ".coverage",
    "coverage.xml",
    "*.tmp",
    "*.log",
    "*.egg-info",
    "build",
    "dist",
    ".hatch",
)


def _iter_matches(root: Path, patterns: Sequence[str]) -> Iterator[Path]:
    """根据 glob 模式寻找匹配路径。"""
    for pattern in patterns:
        if os.sep in pattern or "/" in pattern:
            # 显式路径模式，仅在根目录下匹配
            yield from root.glob(pattern)
        else:
            # 普通文件/目录名，递归匹配
            yield from root.rglob(pattern)


def _collect_targets(root: Path, patterns: Sequence[str]) -> list[Path]:
    seen: Set[Path] = set()
    targets: list[Path] = []
    for path in _iter_matches(root, patterns):
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            continue
        if not resolved.exists():
            continue
        if resolved == root:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        targets.append(resolved)
    # 删除时优先处理深层路径，避免父目录先删导致子路径不存在
    targets.sort(key=lambda p: (-len(p.parents), p.name))
    return targets


def _remove_path(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] {path}")
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
        print(f"目录已删除: {path}")
    else:
        try:
            path.unlink()
            print(f"文件已删除: {path}")
        except FileNotFoundError:
            pass


def clean(root: Path, patterns: Sequence[str], *, dry_run: bool) -> None:
    targets = _collect_targets(root, patterns)
    if not targets:
        print("未发现匹配的临时文件。")
        return
    for path in targets:
        _remove_path(path, dry_run=dry_run)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清理 TabICL 项目中的临时文件/目录。")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录（默认：脚本所在目录的上一层）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要删除的路径，不执行删除操作。",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=(),
        help="额外的 glob 模式，可用于临时扩展清理目标。",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not root.exists():
        print(f"指定的根目录不存在: {root}")
        return 1
    all_patterns = tuple(DEFAULT_PATTERNS) + tuple(args.extra or ())
    clean(root, all_patterns, dry_run=args.dry_run)

    _remove_path(Path("/workspace/Gjj Doc/Code/tabicl/mycmd/server/infer_server/runtime"), dry_run=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
