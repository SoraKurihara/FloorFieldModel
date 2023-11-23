import matplotlib.pyplot as plt
import numpy as np

from FloorFieldModel import FloorFieldModel

cmap = plt.cm.colors.ListedColormap(["white", "red", "black", "green", "blue"])

Map = np.load(r"map/Takatsuki_SimpleWall.npy")
print(Map)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(Map, cmap=cmap, vmin=0, vmax=4)
plt.tight_layout()
plt.show()

model = FloorFieldModel(
    r"map/Takatsuki.xlsx",
    num=3,
    inflow=True,
    pedestrian_count=3000,
)
model.run(steps=1000)
model.plot()
