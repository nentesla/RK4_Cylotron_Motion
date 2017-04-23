import matplotlib.pyplot as plt
import numpy as np


class Particle:
    def __init__(self, pos, vel, mass, charge):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.charge = charge


class Geometry:
    def __init__(self, B, gapTop, gapBottom, EinGap, maxRadius):
        self.B = B
        self.gapTop = gapTop
        self.gapBottom = gapBottom
        self.EinGap = EinGap
        self.maxRadius = maxRadius


proton = Particle([0.0, 0.0, 0.0], [32000, 0.0, 0.0], 1.67E-27, 1.60E-19)
volts = 100000
deeGap = 0.009525
BField = [0, 0, -1.2]
EField = [ 0,-volts / deeGap, 0]
RMax = 0.0508
cyclotron = Geometry(BField, deeGap / 2, -deeGap / 2, EField, RMax)
max_orbits = 10
T = 2.73253111e-8
dT = 1E-10
Ticks_T = T/dT
Max_iterations = 100000


#
# dR = 0
# sumR = []

def Eval(pos, geo):

    if geo.gapTop > pos[1] > geo.gapBottom:
        E_ = np.array(geo.EinGap)
    else:
        E_ = np.array([0,0,0])
    if pos[0] < 0:
        E = E_
    else:
        E = -E_
    return np.array(E)


def EMacceleration(vel, E, B, q_m):
    q_mE = np.array(q_m * E)
    velxB = np.cross(vel, B)
    q_mvelxB = np.array(q_m * velxB)
    acceleration = q_mE + q_mvelxB
    return acceleration


def RungeKutta(particle, max_orbits, geo, Max_iterations, dT):
    q_m = particle.charge / particle.mass
    P0 = np.array(particle.pos)
    V0 = np.array(particle.vel)
    Presults = []
    Vresults = []
    T_vs_Orb = []
    orbits = 0
    i = 0
    while i < Max_iterations and orbits < max_orbits:
        i += 1
        P_last = P0
        P1 = P0
        V1 = V0
        A1 = dT * EMacceleration(V1, Eval(P1, geo), geo.B, q_m)
        V1 = dT * V1

        P2 = P0 + (V1 * 0.5)
        V2 = V0 + (A1 * 0.5)
        A2 = dT * EMacceleration(V2, Eval(P2, geo), geo.B, q_m)
        V2 = dT * V2

        P3 = P0 + (V2 * 0.5)
        V3 = V0 + (A2 * 0.5)
        A3 = dT * EMacceleration(V3, Eval(P3, geo), geo.B, q_m)
        V3 = dT * V3

        P4 = P0 + V3
        V4 = V0 + A3
        A4 = dT * EMacceleration(V4, Eval(P4, geo), geo.B, q_m)
        V4 = dT * V4

        dV = (A1 + 2 * (A2 + A3) + A4)
        V0 = V0 + dV / 6
        dP = (V1 + 2 * (V2 + V3) + V4)
        P0 = P0 + dP / 6

        Vresults.append(V0)
        Presults.append(P0)
        if P_last[1] > 0 > P0[1]:
            orbits += 1
            T_vs_Orb.append([i, orbits])
            print('current orbits =', orbits)

    return (Presults, Vresults, i, T_vs_Orb)

x = []
y = []
R = RungeKutta(proton, max_orbits, cyclotron, Max_iterations, dT)
P = R[0]
V = R[1]
Iterations = R[2]
print('total iterations =', Iterations)
T_vs_Orbitals = R[3]
T = []
Orbitals = []

for i in range(0, Iterations):
    x.append(P[i][0])
    y.append(P[i][1])

for n in range(len(T_vs_Orbitals)):
    T.append(T_vs_Orbitals[n][0])
    Orbitals.append(T_vs_Orbitals[n][1])

RadFinal = np.linalg.norm(P[Iterations-1])
RadFinalDeePlot = RadFinal + 0.05*RadFinal
vFinal = np.linalg.norm(V[Iterations-1])

print(x[Iterations-1])
print(y[Iterations-1])
print('final velocity =', vFinal)
print('final radius =',RadFinal)
#plt.subplot(221)
plt.plot(x, y, 'black',linewidth=0.5)
plt.plot([-RadFinalDeePlot,RadFinalDeePlot],[cyclotron.gapTop,cyclotron.gapTop], 'red')
plt.plot([-RadFinalDeePlot,RadFinalDeePlot],[cyclotron.gapBottom,cyclotron.gapBottom], 'red')
spacing = 91
for t in range(0, np.int(Iterations/spacing)):
    plt.plot(x[t*spacing],y[t*spacing], 'ro', markersize=1.5)
plt.axis('equal')
plt.xlabel('X-displacement (meters)')
plt.ylabel('Y-displacement (meters)')
plt.title('Motion in a Cyclotron')
plt.grid(True)
plt.figure()

#plt.subplot(222)
plt.plot(Orbitals, T)
plt.xlabel('Orbital (number of revolutions)')
plt.ylabel('Time to Voltage Change (RK4 dT)')
plt.title('Time to Voltage Change vs Orbital')
print(T)
print(Orbitals)

plt.show()
