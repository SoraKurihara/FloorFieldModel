from src.FloorFieldModel import FloorFieldModel

model = FloorFieldModel(
    r"map/Takatsuki.xlsx",
    num=4,
    inflow=True,
    pedestrian_count=300,
)
model.run(steps=100)
model.plot()
