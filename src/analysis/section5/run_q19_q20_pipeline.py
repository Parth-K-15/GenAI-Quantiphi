"""Run Section 5 Q19/Q20/Q21 analytics and optionally generate Gemini insights."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[3]


def _check_dependencies(with_llm: bool) -> None:
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
    ]
    if with_llm:
        required.extend(
            [
                ("google.genai", "google-genai"),
                ("dotenv", "python-dotenv"),
            ]
        )

    missing = []
    for module_name, package_name in required:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        unique_missing = sorted(set(missing))
        lines = [
            "Missing required Python packages: " + ", ".join(unique_missing),
            "Install dependencies with:",
            "  pip install -r requirements.txt",
        ]
        raise RuntimeError("\n".join(lines))


def _run_step(cmd: list[str], step_name: str) -> None:
    print(f"[RUN] {step_name}")
    proc = subprocess.run(cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {proc.returncode}")
    print(f"[OK]  {step_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 5 pipeline runner for Q19, Q20, and Q21.")
    parser.add_argument("--with-llm", action="store_true", help="Also run Gemini insights generation for Q19/Q20/Q21.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay between files for Gemini generation.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred model for Section 5 LLM insights.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation progress logs.")
    args = parser.parse_args()

    _check_dependencies(with_llm=args.with_llm)

    steps = [
        (
            [sys.executable, "src/analysis/section5/q19_resignation_drivers_multivariable_reasoning.py"],
            "Q19 analysis",
        ),
        (
            [sys.executable, "src/analysis/section5/q20_attrition_risk_profile_behavior_compensation.py"],
            "Q20 analysis",
        ),
        (
            [sys.executable, "src/analysis/section5/q21_resigned_vs_retained_wlb_overtime_engagement.py"],
            "Q21 analysis",
        ),
    ]

    if args.with_llm:
        llm_cmd = [
            sys.executable,
            "src/llm_section4_insights_generator.py",
            "reports/section5/q19_resignation_drivers_multivariable_reasoning.json",
            "reports/section5/q20_attrition_risk_profile_behavior_compensation.json",
            "reports/section5/q21_resigned_vs_retained_wlb_overtime_engagement.json",
            "--delay",
            str(args.llm_delay),
        ]
        if args.llm_model:
            llm_cmd.extend(["--model", args.llm_model])
        if args.llm_strict_model:
            llm_cmd.append("--strict-model")
        if args.llm_quiet:
            llm_cmd.append("--quiet")

        steps.append(
            (
                llm_cmd,
                "Section 5 Gemini insights generation",
            )
        )

    for cmd, name in steps:
        _run_step(cmd, name)

    print("[DONE] Section 5 pipeline completed.")


if __name__ == "__main__":
    main()
