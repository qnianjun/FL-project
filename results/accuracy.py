import matplotlib.pyplot as plt   
import pandas as pd 


#read file
clients_1=pd.read_csv("1_poisoning_clients.csv")
clients_2=pd.read_csv("2_poisoning_clients.csv")
clients_3=pd.read_csv("3_poisoning_clients.csv")
clients_4=pd.read_csv("4_poisoning_clients.csv")
clients_5=pd.read_csv("5_poisoning_clients.csv")

#draw
plt.plot(
    clients_1["round"],
    clients_1["accuracy"] *100,
    marker="o",
    label="1"
)

plt.plot(
    clients_2["round"],
    clients_2["accuracy"] *100,
    marker="o",
    label="2"
)

plt.plot(
    clients_3["round"],
    clients_3["accuracy"] *100,
    marker="o",
    label="3"
)

plt.plot(
    clients_4["round"],
    clients_4["accuracy"] *100,
    marker="o",
    label="4"
)

plt.plot(
    clients_5["round"],
    clients_5["accuracy"] *100,
    marker="o",
    label="5"
)


plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.title("Accuracy by Poisoning client's numbers")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("poisoning.png",dpi=300)

plt.show()