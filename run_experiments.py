import subprocess
import sys
import time
from pathlib import Path

# =========================
# KONFIG
# =========================
TIMEOUT = 300  # 5 minut

EXPERIMENTS = [
    ("cyclic_with_central.txt", 10, "cyclic_with_central"),
    ("mati.txt", 20, "mati"),
    ("two_camps.txt", 20, "two_camps"),
    ("star.txt", 20, "star"),
    ("one_prey_cyclic_predators.txt", 20, "one_prey_cyclic_predators"),
    ("cascade.txt", 40, "cascade"),
    ("one_prey_many_predators.txt", 20, "one_prey"),
    ("chapter_4_1.txt", 20, "ch41"),
    ("lv.txt", 25, "lv"),
    ("simple.txt", 10, "simple"),
    ("trophic.txt", 25, "trophic"),
    ("three_order.txt", 10, "three_order"),
]

# =========================
# RUNNER
# =========================
def run_experiment(input_file, t, prefix):
    cmd = [
        sys.executable,
        "adaptive_cycle.py",
        input_file,
        "--time", str(t),
        "--save-prefix", prefix
    ]

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )

        duration = time.time() - start

        if result.returncode != 0:
            return {
                "status": "FAIL",
                "reason": "NONZERO_RETURN_CODE",
                "stderr": result.stderr[-1000:],
                "time": duration
            }

        return {
            "status": "OK",
            "reason": None,
            "time": duration
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "FAIL",
            "reason": "TIMEOUT",
            "stderr": f"Exceeded {TIMEOUT}s",
            "time": TIMEOUT
        }

    except Exception as e:
        return {
            "status": "FAIL",
            "reason": "EXCEPTION",
            "stderr": str(e),
            "time": None
        }


# =========================
# MAIN LOOP
# =========================
def main():
    results = []

    print("\n==============================")
    print("STARTING BATCH EXPERIMENTS")
    print("==============================\n")

    for input_file, t, prefix in EXPERIMENTS:

        print(f"RUNNING: {input_file} | t={t} | prefix={prefix}")

        res = run_experiment(input_file, t, prefix)

        results.append({
            "experiment": input_file,
            "time_param": t,
            "prefix": prefix,
            **res
        })

        print(f" -> {res['status']} ({res.get('reason', '')})")

    # =========================
    # REPORT
    # =========================
    print("\n==============================")
    print("FINAL REPORT")
    print("==============================\n")

    ok = [r for r in results if r["status"] == "OK"]
    fail = [r for r in results if r["status"] != "OK"]

    print(f"SUCCESS: {len(ok)}")
    print(f"FAILED : {len(fail)}\n")

    for r in fail:
        print(f"[FAIL] {r['experiment']}")
        print(f"       reason: {r['reason']}")
        if r.get("stderr"):
            print(f"       stderr: {r['stderr'][:300]}")
        print()

    # opcjonalnie zapis raportu
    report_path = Path("batch_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BATCH EXPERIMENT REPORT\n\n")

        for r in results:
            f.write(str(r) + "\n")

    print(f"\nReport saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()