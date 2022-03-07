# EECS545final

Repo to house shared components of EECS545 final project

# Transient class

All simulations are performed with the ``Transient`` class. Two sets of
parameters need to be specified for a simulation.
- Design parameters: Parameters which define the system. This mostly includes
component dimensions.
- Operational parameters: Parameters which operators can select for a set system.

Design parameters are specified on initialization of an instance of the ``Transient`` class.
Operational parameters are specified when calling ``Transient.simulate``.
Defaults for all parameters will be considered the "nominal" configuration.

# Operational parameter ranges
Reasonable ranges for the operational ranges (with the nominal design parameters) are:
- ``qdot_boil``: [10000-40000]
- ``Pout_frac``: [.05-.4]
- ``mdot``: [5-30]
- ``T0``: [300-340]
- ``P0``: [5-20]
- ``Pout_time``: [0-60]

# Example simulation
```
t = Transient()
tfinal = 5*60 #s
t.simulate(tfinal)
plt.plot(t.t, t.Tboil)
plt.show()
```

