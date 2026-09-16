import matplotlib.pyplot as plt   
import pandas as pd 


#read file
non_iid=pd.read_csv("NON_IID.csv")
iid=pd.read_csv("IID.csv")


#draw
plt.plot(
    non_iid["round"],
    non_iid["accuracy"] *100,
    marker="o",
    label="NON_IID"
)



plt.plot(
    iid["round"],
    iid["accuracy"] *100,
    marker="o",
    label="IID"
)


plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.title("IID vs Non-IID Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("IID_vs_Non_IID.png",dpi=300)

plt.show()