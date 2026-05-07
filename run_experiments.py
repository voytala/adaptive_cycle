import subprocess
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

TIMEOUT = 300

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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_experiment(exp):
    input_file, t, prefix = exp

    cmd = [
        sys.executable,
        "adaptive_cycle.py",
        input_file,
        "--time", str(t),
        "--save-prefix", prefix
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,           # 🔥 KLUCZOWE
            timeout=TIMEOUT
        )

        return {
            "exp": input_file,
            "ok": True
        }

    except subprocess.TimeoutExpired:
        return {
            "exp": input_file,
            "ok": False,
            "reason": "TIMEOUT"
        }

    except Exception as e:
        return {
            "exp": input_file,
            "ok": False,
            "reason": str(e)
        }


def main():
    print("START PARALLEL (SAFE THREAD MODE)")

    # 🔥 THREADS zamiast PROCESS (ważne!)
    # bo subprocess i matplotlib = nie lubią ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(run_experiment, e) for e in EXPERIMENTS]

        results = []
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            print(r)

    ok = sum(1 for r in results if r["ok"])
    print("\nSUCCESS:", ok, "/", len(results))


if __name__ == "__main__":
    main()