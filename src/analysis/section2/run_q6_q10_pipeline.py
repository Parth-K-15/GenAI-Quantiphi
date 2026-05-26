"""Run Section 2 Q6/Q7/Q8/Q9/Q10 analytics and optionally generate Gemini insights."""

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
    parser = argparse.ArgumentParser(description="Section 2 pipeline runner for Q6-Q10.")
    parser.add_argument("--with-llm", action="store_true", help="Also run Gemini insights generation for Section 2.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay between files for Gemini generation.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred model for Section 2 LLM insights.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation progress logs.")
    args = parser.parse_args()

    _check_dependencies(with_llm=args.with_llm)

    steps = [
        ([sys.executable, "src/analysis/section2/q6_training_development_analysis.py"], "Q6 analysis"),
        ([sys.executable, "src/analysis/section2/q7_mentorship_impact_analysis.py"], "Q7 analysis"),
        ([sys.executable, "src/analysis/section2/q8_training_roi_analysis.py"], "Q8 analysis"),
        ([sys.executable, "src/analysis/section2/q9_training_program_comparison.py"], "Q9 analysis"),
        ([sys.executable, "src/analysis/section2/q10_advanced_training_readiness.py"], "Q10 analysis"),
    ]

    if args.with_llm:
        llm_cmd = [
            sys.executable,
            "src/llm_section4_insights_generator.py",
            "reports/section2/q6_training_development_analysis.json",
            "reports/section2/q7_mentorship_impact_analysis.json",
            "reports/section2/q8_training_roi_analysis.json",
            "reports/section2/q9_training_program_comparison.json",
            "reports/section2/q10_advanced_training_readiness.json",
            "--delay",
            str(args.llm_delay),
        ]
        if args.llm_model:
            llm_cmd.extend(["--model", args.llm_model])
        if args.llm_strict_model:
            llm_cmd.append("--strict-model")
        if args.llm_quiet:
            llm_cmd.append("--quiet")

        steps.append((llm_cmd, "Section 2 Gemini insights generation"))

    for cmd, name in steps:
        _run_step(cmd, name)

    print("[DONE] Section 2 pipeline completed.")


if __name__ == "__main__":
    main()
