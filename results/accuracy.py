import matplotlib.pyplot as plt   
import pandas as pd 


#read file
iid=pd.read_csv("baseline_iid.csv")
noniid=pd.read_csv("noniid.csv")

#draw
plt.plot(
    iid["round"],
    iid["accuracy"] *100,
    marker="o",
    label="IID"
)

plt.plot(
    noniid["round"],
    noniid["accuracy"] *100,
    marker="o",
    label="Non-IID"
)

plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.title("IID vs Non-IID Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("iid_vs_noniid.png",dpi=300)

plt.show()