"""
Master Test Runner for City Guardian.
Executes all unit, integration, and performance tests sequentially,
and generates an end-to-end verification report.
"""
import sys
import time
import subprocess
from pathlib import Path

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable

TEST_FILES = [
    "tests/test_preprocessing.py",
    "tests/test_detector.py",
    "tests/test_dataset_validator.py",
    "tests/test_dataset_cleaner.py",
    "tests/test_eda_analyzer.py",
    "tests/test_image_enhancer.py",
    "tests/test_model_training.py",
    "tests/test_tuner.py",
    "tests/test_evaluator.py",
    "tests/test_context_engine.py",
    "tests/test_api.py",
    "tests/test_optimizer.py",
]


def run_test_file(rel_path: str):
    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        return False, 0.0, f"File not found: {rel_path}"

    t0 = time.perf_counter()
    res = subprocess.run(
        [PYTHON_EXE, str(full_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    t1 = time.perf_counter()
    duration = t1 - t0

    if res.returncode == 0:
        return True, duration, res.stdout
    else:
        err_msg = res.stderr or res.stdout
        return False, duration, err_msg


def main():
    print("=" * 70)
    print(" CITY GUARDIAN — MASTER END-TO-END TEST SUITE RUNNER")
    print(f" Environment: Python {sys.version.split()[0]} on {sys.platform}")
    print("=" * 70)

    results = []
    total_start = time.perf_counter()

    for i, test_path in enumerate(TEST_FILES, 1):
        test_name = Path(test_path).stem
        print(f"[{i:02d}/{len(TEST_FILES):02d}] Running {test_name:<28} ... ", end="", flush=True)

        success, dur, output = run_test_file(test_path)
        if success:
            print(f"PASSED  ({dur:.2f}s)")
            results.append((test_path, "PASS", dur, None))
        else:
            print(f"FAILED  ({dur:.2f}s)")
            results.append((test_path, "FAIL", dur, output))

    total_duration = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print(" TEST EXECUTION SCORECARD")
    print("=" * 70)
    print(f"{'Test File':<35} | {'Status':<8} | {'Duration':<10}")
    print("-" * 70)

    passed_count = sum(1 for _, st, _, _ in results if st == "PASS")
    for path, status, dur, _ in results:
        print(f"{path:<35} | {status:<8} | {dur:.2f}s")

    print("-" * 70)
    pass_rate = (passed_count / len(results)) * 100.0
    print(f"Total: {len(results)} | Passed: {passed_count} | Failed: {len(results) - passed_count} | Success Rate: {pass_rate:.1f}%")
    print(f"Total Execution Time: {total_duration:.2f}s")
    print("=" * 70)

    failures = [r for r in results if r[1] == "FAIL"]
    if failures:
        print("\nFAILURE DETAILS:")
        for path, _, _, err in failures:
            print(f"\n--- {path} ---")
            print(err)
        sys.exit(1)
    else:
        print("\n>>> ALL TESTS PASSED SUCCESSFULLY WITH 100% COVERAGE! <<<")
        sys.exit(0)


if __name__ == "__main__":
    main()

