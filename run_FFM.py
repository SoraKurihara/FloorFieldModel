import matplotlib.pyplot as plt
import numpy as np

from FloorFieldModel import FloorFieldModel

cmap = plt.cm.colors.ListedColormap(["white", "red", "black", "green"])

Map = np.load("Takatsuki_SimpleWall.npy")
print(Map)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(Map, cmap=cmap, vmin=0, vmax=1)
plt.tight_layout()
plt.show()

model = FloorFieldModel(
    r"Takatsuki_SimpleWall.npy",
    num=0,
    pedestrian_count=100,
)
model.run(steps=300)
model.plot()
