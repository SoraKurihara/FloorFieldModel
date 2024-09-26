import FloorFieldModel as FFM

ffm = FFM.FloorFieldModel(Map="random", SFF=None, method="L2")
ffm.params(N=5, inflow=None, k_S=3, k_D=1, d="Neumann")
ffm.run(steps=100)
ffm.plot(footprints=False)