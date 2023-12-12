import matplotlib.pyplot as plt
import numpy as np

SFF = np.load(r"SFF/Takatsuki_Linf.npy")
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(SFF)
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
