import pandas as pd
import matplotlib.pyplot as plt

iid = pd.read_csv("baseline_iid.csv")
noniid = pd.read_csv("noniid.csv")

plt.plot(
    iid["round"],
    iid["loss"],
    marker="o",
    label="IID"
)

plt.plot(
    noniid["round"],
    noniid["loss"],
    marker="o",
    label="Non-IID"
)

plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("IID vs Non-IID Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("iid_vs_noniid_loss.png", dpi=300)
plt.show()