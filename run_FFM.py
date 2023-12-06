from src.FloorFieldModel import FloorFieldModel

model = FloorFieldModel(r"map/Takatsuki.xlsx", SFF=r"SFF/Takatsuki_L2.npy")
model.params(
    N=0,
    inflow=300,
    k_S=3,
    k_D=1,
    k_Dir=None,
    k_Str=None,
)
model.run(steps=100)
model.plot()
