from src.FloorFieldModel import FloorFieldModel

model = FloorFieldModel(r"map/Takatsuki.xlsx", method="L1")
model.params(
    N=0,
    inflow=3000,
    k_S=3,
    k_D=1,
    k_Dir=None,
    k_Str=None,
)
model.run(steps=10000)
model.plot()
