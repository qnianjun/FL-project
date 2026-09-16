import matplotlib.pyplot as plt   
import pandas as pd 


#read file
alpha_001=pd.read_csv("alpha_0.01.csv")
alpha_01=pd.read_csv("alpha_0.1.csv")
alpha_1=pd.read_csv("alpha_1.csv")
alpha_10=pd.read_csv("alpha_10.csv")
alpha_100=pd.read_csv("alpha_100.csv")

#draw
plt.plot(
    alpha_001["round"],
    alpha_001["accuracy"] *100,
    marker="o",
    label="α = 0.01"
)



plt.plot(
    alpha_01["round"],
    alpha_01["accuracy"] *100,
    marker="o",
    label="α = 0.1"
)

plt.plot(
    alpha_1["round"],
    alpha_1["accuracy"] *100,
    marker="o",
    label="α = 1"
)

plt.plot(
    alpha_10["round"],
    alpha_10["accuracy"] *100,
    marker="o",
    label="α = 10"
)

plt.plot(
    alpha_100["round"],
    alpha_100["accuracy"] *100,
    marker="o",
    label="α = 100"
)

plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.title("Dirichlet Non-IID Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Dirichlet_Non_IID.png",dpi=300)

plt.show()