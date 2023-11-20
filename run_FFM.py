import matplotlib.pyplot as plt
import numpy as np

from FloorFieldModel import FloorFieldModel

cmap = plt.cm.colors.ListedColormap(["white", "red", "black", "green"])

Map = np.load(r"map/Takatsuki_SimpleWall.npy")
print(Map)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(Map, cmap=cmap, vmin=0, vmax=3)
plt.tight_layout()
plt.show()

SFF = np.load(r"SFF/Takatsuki.npy")
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(SFF)
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

model = FloorFieldModel(
    r"map/Takatsuki_SimpleWall.npy",
    num=0,
    pedestrian_count=10000,
)
model.run(steps=10000)
model.plot()
