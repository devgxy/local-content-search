"""`python -m app` 入口。

Step: STEP-00
Doc: docs/steps/STEP-00-项目骨架与环境校验.md §4.5
"""

from __future__ import annotations

import argparse
import sys

from app.config import load_settings
from app.logging_setup import setup_logging
from app.preflight import print_report, run_preflight


def main(argv: list[str] | None = None) -> int:
    """启动 FastAPI 应用的命令行入口。"""
    parser = argparse.ArgumentParser(prog="python -m app", description="本地内容检索")
    parser.add_argument("--check", action="store_true", help="只执行自检并退出")
    args = parser.parse_args(argv)

    settings = load_settings()
    settings.ensure_dirs()
    setup_logging(settings)

    report = run_preflight(settings)
    if args.check:
        print_report(report)
        return 0 if report.ok else 1

    if not report.ok:
        print_report(report)
        return 1

    try:
        import uvicorn
    except ImportError:
        print("未安装 uvicorn。")
        return 1

    try:
        from app.api.app import create_app
    except ImportError:
        print("Web 服务尚未实现（STEP-08）。当前请使用 `python -m app --check`。")
        return 0

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
