import matplotlib.pyplot as plt   
import pandas as pd 


#read file
poisoning_1=pd.read_csv("1_poisoning.csv")
poisoning_2=pd.read_csv("2_poisoning.csv")
poisoning_5=pd.read_csv("5_poisoning.csv")
poisoning_10=pd.read_csv("10_poisoning.csv")
poisoning_20=pd.read_csv("20_poisoning.csv")
poisoning_50=pd.read_csv("50_poisoning.csv")

#draw
plt.plot(
    poisoning_1["round"],
    poisoning_1["accuracy"] *100,
    marker="o",
    label="1"
)

plt.plot(
    poisoning_2["round"],
    poisoning_2["accuracy"] *100,
    marker="o",
    label="2"
)

plt.plot(
    poisoning_5["round"],
    poisoning_5["accuracy"] *100,
    marker="o",
    label="5"
)

plt.plot(
    poisoning_10["round"],
    poisoning_10["accuracy"] *100,
    marker="o",
    label="10"
)

plt.plot(
    poisoning_20["round"],
    poisoning_20["accuracy"] *100,
    marker="o",
    label="20"
)

plt.plot(
    poisoning_50["round"],
    poisoning_50["accuracy"] *100,
    marker="o",
    label="50"
)


plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.title("Accuracy by Poisoning Scale")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("poisoning.png",dpi=300)

plt.show()