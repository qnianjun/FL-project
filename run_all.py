import json
import subprocess

choice = input(
"""
==============================
1. Alpha Experiment
2. Poisoning Experiment
3. Run All
==============================

Choose: """
)

def run_exp(cfg):

    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    subprocess.run(
        ["bash", "run_all.sh"],
        check=True
    )


if choice == "1":

    alphas = [0.01, 0.1, 1, 10, 100]

    for alpha in alphas:

        print(f"\nRunning alpha={alpha}")

        run_exp({
            "alpha": alpha,
            "enable_poison": False,
            "poison_client": 0,
            "poison_scale": 10
        })


elif choice == "2":

    scales = [1, 5, 10, 20, 50, 100]

    for scale in scales:

        print(f"\nRunning poison scale={scale}")

        run_exp({
            "alpha": 0.1,
            "enable_poison": True,
            "poison_client": 0,
            "poison_scale": scale
        })


elif choice == "3":

    alphas = [0.01, 0.1, 1, 10, 100]

    for alpha in alphas:

        run_exp({
            "alpha": alpha,
            "enable_poison": False,
            "poison_client": 0,
            "poison_scale": 10
        })

    scales = [1, 5, 10, 20, 50, 100]

    for scale in scales:

        run_exp({
            "alpha": 0.1,
            "enable_poison": True,
            "poison_client": 0,
            "poison_scale": scale
        })

else:
    print("Invalid choice")
