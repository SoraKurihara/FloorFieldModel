from src.FloorFieldModel import FloorFieldModel

model = FloorFieldModel(r"map/Takatsuki.xlsx", method="Linf")
model.params(
    N=0, inflow=300, k_S=3, k_D=1, k_Dir=None, k_Str=None, d="Neumann"
)
model.run(steps=1000)
model.plot()
