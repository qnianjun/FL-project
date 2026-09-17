import matplotlib.pyplot as plt   
import pandas as pd 


#read file
alpha_poisoning_1=pd.read_csv("alpha_poisoning_1.csv")
alpha_poisoning_01=pd.read_csv("alpha_poisoning_0.1.csv")
alpha_poisoning_001=pd.read_csv("alpha_poisoning_0.01.csv")
alpha_poisoning_10=pd.read_csv("alpha_poisoning_10.csv")
alpha_poisoning_100=pd.read_csv("alpha_poisoning_100.csv")

#draw


plt.plot(
    alpha_poisoning_001["round"],
    alpha_poisoning_001["accuracy"] *100,
    marker="o",
    label="α=0.01"
)

plt.plot(
    alpha_poisoning_01["round"],
    alpha_poisoning_01["accuracy"] *100,
    marker="o",
    label="α=0.1"
)


plt.plot(
    alpha_poisoning_1["round"],
    alpha_poisoning_1["accuracy"] *100,
    marker="o",
    label="α=1"
)

plt.plot(
    alpha_poisoning_10["round"],
    alpha_poisoning_10["accuracy"] *100,
    marker="o",
    label="α=10"
)

plt.plot(
    alpha_poisoning_100["round"],
    alpha_poisoning_100["accuracy"] *100,
    marker="o",
    label="α=100"
)


plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.title("Dirichlet Non-IID with Poisoning Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Dirichlet_Non_IID_with_Poisoning_Poisoning.png",dpi=300)

plt.show()