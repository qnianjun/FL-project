import matplotlib.pyplot as plt   
import pandas as pd 


#read file
light=pd.read_csv("light.csv")
mid=pd.read_csv("mid.csv")
serious=pd.read_csv("serious.csv")
#draw
plt.plot(
    light["round"],
    light["accuracy"] *100,
    marker="o",
    label="light"
)

plt.plot(
    mid["round"],
    mid["accuracy"] *100,
    marker="o",
    label="middle"
)

plt.plot(
    serious["round"],
    serious["accuracy"] *100,
    marker="o",
    label="serious"
)

plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.title("Dirichlet Non-IID Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Dirichlet_Non_IID.png",dpi=300)

plt.show()