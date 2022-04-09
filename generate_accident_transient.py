import numpy as np
import matplotlib.pyplot as plt
from transient import Transient
from pathlib import Path
import time

start = time.time()

qdot_boil = [10000, 50000]
Pout_frac = [0.05, .5]
Pout_time = [0, 60]
mdot = [4, 30]
T0 = [300, 340]
P0 = [5, 20]

slog = Path("transient_prediction/scalar_res.csv")

t_eval = np.linspace(0, 10*60, 501)

outparams = ["Tboil", "Eturb", "Pout", "omega", "x1", "h1", "T1", "P1", "rho1",
        "x2", "h2", "T2", "P2", "rho2"]

logfs = [Path("transient_prediction/" + a + "_log.csv") for a in outparams]

if True:#not slog.exists():
    with open(slog, "w") as f:
        f.write("qdot_boil,Pout_frac,Pout_time,mdot,T0,P0,accident,accident_time\n")

for l in logfs:
    if True:#not l.exists():
        with open(l, "w") as f:
            for t in t_eval:
                f.write("%E,"%t)
            f.write("\n")

for i in range(1000):
    p1 = np.random.uniform(*qdot_boil)
    p2 = np.random.uniform(*Pout_frac)
    p3 = np.random.uniform(*mdot)
    p4 = np.random.uniform(*T0)
    p5 = np.random.uniform(*P0)
    p6 = np.random.uniform(*Pout_time)
    t = Transient()
    t.simulate(10*60, t_eval = t_eval, qdot_boil = p1, Pout_frac = p2,
            mdot = p3, T0 = p4, P0 = p5, Pout_time = p6)

    if (t.accident == "No accident"):
        with open(slog, "a") as f:
            f.write(f"{p1},{p2},{p3},{p4},{p5},{p6},{t.accident},{t.accident_time}\n")
        for i, o in enumerate(outparams):
            with open(logfs[i], "a") as f:
                for tv in getattr(t, o):
                    f.write("%E,"%tv)
                f.write("\n")

print("Runtime:", time.time() - start, "s")
