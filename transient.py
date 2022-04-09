import numpy as np
import matplotlib.pyplot as plt
from iapws import IAPWS97 as iap
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

class Transient:
    """
    Simulation for Boiler, Pipe and Turbine components in
    Rankine cycle
    """
    def __init__(self, 
            Rboil=.5, Lboil=1., Aturb=np.pi*1**2, Mboil=800*420/1000, 
            Mturb=300, sim_parms = None):
        """
        Initialize transient simulation. Specify design parameters here.
        Defaults are from nominal design.

        Rboil: radius of pipe through boiler, m
        Lboil: length of pipe through boiler, m
        Mboil: thermal inertia of boiler, kJ/K
        Mturb: mass of turbine, kg

        sim_parms: dict of simulation parameters,
        only use if needed
        """
        self.Rboil = Rboil #m
        self.Lboil = Lboil #m
        self.Aboil = 2*np.pi*Rboil*Lboil #m^2
        self.Aturb = Aturb #m^2
        self.Mboil = Mboil #kJ/K
        self.Mturb = Mturb #kg

        #simulation parameters
        self.a = .1 #s/radian
        self.xboil_u = .98
        self.Tcrit = 450 #K
        self.omega_upper = 9 #rad/s
        if sim_parms is None:
            pass
        else:
            k = list(sim_parms.keys())
            if "a" in k:
                self.a = sim_parms["a"]
            if "xboil_u" in k:
                self.xboil_u = sim_parms["xboil_u"]
            if "Tcrit" in k:
                self.Tcrit = sim_parms["Tcrit"]

    def simulate(self, tfinal, max_step = None, qdot_boil=25000, Pout_frac=.1, Pout_time=60,
            mdot=13, T0=300, P0=15.5, t_eval = None):
        """
        Perform simulation until time tend with given operational conditions.

        tfinal: time to run the simulation until, s
        max_step: maximum step size used by the solver, try to avoid using when running
            large numbers of simulations to generate training data. Only use if you want
            a nice looking figure of a transient. Default stepping yields very accurate
            solutions, s
        qdot_boil: heat generation rate in the boiler, kW
        Pout_frac: fraction of total energy to remove from turbine per second, [-]
        Pout_time: time to start removing energy from turbine, some time is given for
            the system to start up, s
        mdot: mass flow rate through system, kg/s
        T0: inlet temperature into boiler, K
        P0: inlet pressure into boiler, MPa

        After running simulation, many new attributes are assigned
        self.t: np.array of simulation time evaluations, s
        self.Tboil: np.array of boiler temperature at each time, K
        self.Eturb: np.array of turbine energy at each time, kJ
        self.Pout: np.array of energy taken from turbine, kJ
        self.omega: np.array of turbine rotation speed, rad/s
        self.x1: np.array of water quality at boiler outlet
        self.h1: np.array of water enthalpy at boiler outlet, kJ/kg
        self.T1: np.array of water temperature at boiler outlet, K
        self.P1: np.aray of water pressure at boiler outlet, MPa
        self.rho1: np.array of water density at boiler outlet, kg/m^3
        self.x2: np.array of water quality at turbine outlet
        self.h2: np.array of water enthalpy at turbine outlet, kJ/kg
        self.T2: np.array of water temperature at turbine outlet, K
        self.P2: np.aray of water pressure at turbine outlet, MPa
        self.rho2: np.array of water density at turbine outlet, kg/m^3

        self.accident: str describing type of accident encountered, if None
            no accident occurred
        self.accident_time: float of time accident occurred
        """
        #calculate h using dittus-boelter and liu-winterton
        def hfind(mdot, T): #replace with expl parameters when using ML
            wat = iap(T = T, P = P0)
            x = wat.x
            wf = iap(P = wat.P, x = 0)
            visc = wf.mu
            Pr = wf.Prandt
            kwater = wf.k
            vg = 1/iap(P = wat.P, x = 1).rho
            vf = 1/wf.rho
            Tf = wf.T
            P = wat.P
            T = wat.T


            Re = mdot*2*self.Rboil/(visc*np.pi*self.Rboil**2)
            hc = kwater/self.Aboil*0.023*Re**0.8*Pr**0.4
            if x < 1e-5:
                return hc
            else:
                Pcrit = 2.12 #MPa
                Mwater = 18.02 #kg/mol

                F = (1 + x*Pr*(vg/vf - 1))**0.35
                S = (1 + 0.055*F**0.1*Re**0.16)**-1
                hnb = 55*(P/Pcrit)**.12*(qdot_boil/self.Aboil)**(2/3) - np.log10(P/Pcrit)**-0.55*Mwater**-0.5
                hnb = hnb / 1000
                return np.sqrt(F**2*hc**2 + S**2*hnb**2)


        #function to calculate power loss in turbine
        def Ploss(Eturb, omega):
            return Eturb*np.tanh(self.a*omega)/10

        #various quantity calculations
        def calc_omega(Eturb):
            #abs in Eturb for numerics, if Eturb is negative, simulation dies anyways
            return np.sqrt(4*np.pi*np.abs(Eturb*1000)/self.Mturb/self.Aturb)

        def calc_Ecool(rho):
            return mdot**3*self.Aturb/2/rho**2

        def calc_w1(Tboil):
            h = hfind(mdot, (Tboil + T0)/2)
            h1 = self.Aboil*h/mdot*(Tboil - T0) + w0.h
            return iap(h = h1, P = P0)

        #Functions to detect occurance of ATE
        def boiler_xupper(t, x):
            w1 = calc_w1(x[0])
            upper = self.xboil_u - w1.x
            return upper

        def boiler_Tcrit(t, x):
            return x[0] - self.Tcrit

        def turbine_E(t, x):
            return x[1]

        def turbine_omega(t, x):
            omega = calc_omega(x[1])
            return omega - self.omega_upper

        ATE_events = [boiler_xupper, boiler_Tcrit, turbine_E, turbine_omega]
        ATE_names = ["boiler dryout", "boiler meltdown", "turbine reversal", "turbine rotation limit"]
        for a in ATE_events: #make all events terminate simulatin
            a.terminal = True

        #time stepper function
        w0 = iap(T = T0, P = P0)
        def f(t, x):
            Tboil = x[0]
            h = hfind(mdot, (Tboil + T0)/2)
            dTboildt = 1/self.Mboil*(qdot_boil - self.Aboil*h*(Tboil - T0))

            w1 = calc_w1(Tboil)

            Eturb = x[1]
            omega = calc_omega(Eturb)
            Ecoolant = calc_Ecool(w1.rho)

            if t > Pout_time:
                Poutn = Pout_frac*Eturb
            else:
                Poutn = 0
            dEturbdt =  Ecoolant - Poutn - Ploss(Eturb, omega)

            return [dTboildt, dEturbdt]

        #perform stepping calculation
        E0 = 1e-5 #kJ, energy to help with numerics
        if max_step is None:
            p = solve_ivp(f, (0, tfinal), [T0, E0], method = "Radau", events = ATE_events, t_eval = t_eval)
        else:
            p = solve_ivp(f, (0, tfinal), [T0, E0], method = "Radau", events = ATE_events, max_step = max_step, t_eval = t_eval)

        #calculate simulation quantities
        self.t = p.t
        self.Tboil = p.y[0]
        self.Eturb = p.y[1]

        #logging accident status
        if p.status == 0:
            self.accident = "No accident"
            self.accident_time = None
        elif p.status == -1:
            self.accident = "Unphysical"
            self.accident_time = self.t[-1]
        else:
            accident_index = np.where(np.array([len(a) for a in p.t_events]) > 0)[0][0]
            self.accident = ATE_names[accident_index]
            self.accident_time = p.t_events[accident_index][0]


        #getting rotation speed
        self.omega = np.array([calc_omega(a) for a in self.Eturb])

        #making list of water properties
        self.wat1 = []
        self.wat2 = []
        for i in range(len(self.t)):
            w1 = calc_w1(self.Tboil[i])
            self.wat1.append(w1)

            h2 = w1.h - calc_Ecool(w1.rho)
            self.wat2.append(iap(s = w1.s, h = h2))

        #making lists to hold values of water properties
        self.x1 = []
        self.h1 = []
        self.T1 = []
        self.P1 = []
        self.rho1 = []

        self.x2 = []
        self.h2 = []
        self.T2 = []
        self.P2 = []
        self.rho2 = []

        for i in range(len(self.t)):
            self.x1.append(self.wat1[i].x)
            self.x2.append(self.wat2[i].x)

            self.h1.append(self.wat1[i].h)
            self.h2.append(self.wat2[i].h)

            self.T1.append(self.wat1[i].T)
            self.T2.append(self.wat2[i].T)

            self.P1.append(self.wat1[i].P)
            self.P2.append(self.wat2[i].P)

            self.rho1.append(self.wat1[i].rho)
            self.rho2.append(self.wat2[i].rho)

        #recalculating Pout
        self.Pout = []
        for i in range(len(self.t)):
            if self.t[i] > Pout_time:
                self.Pout.append(Pout_frac*self.Eturb[i])
            else:
                self.Pout.append(0)
