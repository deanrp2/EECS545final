import numpy as np
from transient import Transient
from pathlib import Path

qdot_boil = [10000, 50000]
Pout_frac = [0.05, .5]
Pout_time = [0, 60]
mdot = [4, 30]
T0 = [300, 340]
P0 = [5, 20]

outlog = Path("accident_classification/operation_accidents.csv")

if not outlog.exists():
    with open(outlog, "w") as f:
        f.write("qdot_boil,Pout_frac,Pout_time,mdot,T0,P0,accident,accident_time\n")
else:
    pass

for i in range(100):
    p1 = np.random.uniform(*qdot_boil)
    p2 = np.random.uniform(*Pout_frac)
    p3 = np.random.uniform(*mdot)
    p4 = np.random.uniform(*T0)
    p5 = np.random.uniform(*P0)
    p6 = np.random.uniform(*Pout_time)
    t = Transient()
    t.simulate(10*60, qdot_boil = p1, Pout_frac = p2,
            mdot = p3, T0 = p4, P0 = p5, Pout_time = p6)
    with open(outlog, "a") as f:
        f.write(f"{p1},{p2},{p3},{p4},{p5},{p6},{t.accident},{t.accident_time}\n")


