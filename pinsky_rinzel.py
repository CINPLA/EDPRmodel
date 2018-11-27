from scipy.integrate import solve_ivp, odeint
import numpy as np
from math import exp
import matplotlib.pyplot as plt

I_s = 2.5    # somatic injection current [uA * cm**-2]
g_c = 10.5   # [mS cm**-2]

C_m = 3.     # membrane capacitance [uF cm**-2]
p = 0.5      # proportion of the membrane area taken up by the soma

g_L = 0.1    # [mS cm**-2]
g_Na = 30.   # [mS cm**-2]
g_DR = 15.   # [mS cm**-2]
g_Ca = 10.   # [mS cm**-2]
g_AHP = 0.8  # [mS cm**-2]
g_C = 15.    # [mS cm**-2]

E_L = -60.   # [mV]
E_Na = 60.   # [mV]
E_K = -75.   # [mV]
E_Ca = 80.   # [mV]

def alpha_m(Vs):
    V1 = Vs + 46.9
    alpha = - 0.32 * V1 / (exp(-V1 / 4.) - 1.)
    return alpha

def beta_m(Vs):
    V2 = Vs + 19.9
    beta = 0.28 * V2 / (exp(V2 / 5.) - 1.)
    return beta

def alpha_h(Vs):
    alpha = 0.128 * exp((-43. - Vs) / 18.)
    return alpha

def beta_h(Vs):
    V5 = Vs + 20.
    beta = 4. / (1 + exp(-V5 / 5.))
    return beta

def alpha_n(Vs):
    V3 = Vs + 24.9
    alpha = - 0.016 * V3 / (exp(-V3 / 5.) - 1)
    return alpha

def beta_n(Vs):
    V4 = Vs + 40.
    beta = 0.25 * exp(-V4 / 40.)
    return beta

def alpha_s(Vd):
    alpha = 1.6 / (1 + exp(-0.072 * (Vd-5.)))
    return alpha

def beta_s(Vd):
    V6 = Vd + 8.9
    beta = 0.02 * V6 / (exp(V6 / 5.) - 1.)
    return beta

def alpha_c(Vd):
    V7 = Vd + 53.5
    V8 = Vd + 50.
    if Vd <= -10:
        alpha = 0.0527 * exp(V8/11.- V7/27.)
    else:
        alpha = 2 * exp(-V7 / 27.)
    return alpha

def beta_c(Vd):
    V7 = Vd + 53.5
    if Vd <= -10:
        beta = 2. * exp(-V7 / 27.) - alpha_c(Vd)
    else:
        beta = 0.
    return beta

def alpha_q(Ca):
    alpha = min(0.00002*Ca, 0.01)
    return alpha

def beta_q(Ca):
    return 0.001

def chi(Ca):
    return min(Ca/250., 1.)

def m_inf(Vs):
    return alpha_m(Vs) / (alpha_m(Vs) + beta_m(Vs))

def dVdt(t, V):
   
    Vs, Vd, n, h, s, c, q, Ca = V

    I_syn = 0.
    I_leak_s = g_L*(Vs - E_L)
    I_leak_d = g_L*(Vd - E_L)
    I_Na = g_Na * m_inf(Vs)**2 * h * (Vs - E_Na)
    I_DR = g_DR * n * (Vs - E_K)
    I_ds = g_c * (Vd - Vs)
    
    I_Ca = g_Ca * s**2 * (Vd - E_Ca)
    I_AHP = g_AHP * q * (Vd - E_K)
    I_C = g_C * c * chi(Ca) * (Vd - E_K)    
    I_sd = -I_ds 

    dVsdt = (1./C_m)*( -I_leak_s - I_Na - I_DR + I_ds/p + I_s/p )
    dVddt = (1./C_m)*( -I_leak_d - I_Ca - I_AHP - I_C + I_sd/(1-p) + I_syn/(1-p))
    dhdt = alpha_h(Vs)*(1-h) - beta_h(Vs)*h 
    dndt = alpha_n(Vs)*(1-n) - beta_n(Vs)*n
    dsdt = alpha_s(Vd)*(1-s) - beta_s(Vd)*s
    dcdt = alpha_c(Vd)*(1-c) - beta_c(Vd)*c
    dqdt = alpha_q(Ca)*(1-q) - beta_q(Ca)*q
    dCadt = -0.13*I_Ca - 0.075*Ca

    return dVsdt, dVddt, dndt, dhdt, dsdt, dcdt, dqdt, dCadt


t_span = (0, 800)

Vs0 = -64.6
Vd0 = -64.5
n0 = 0.001
h0 = 0.999
s0 = 0.009
c0 = 0.007
q0 = 0.01
Ca0 = 0.2
V0 = [Vs0, Vd0, n0, h0, s0, c0, q0, Ca0]

sol = solve_ivp(dVdt, t_span, V0, max_step=0.05)

Vs, Vd, n, h, s, c, q, Ca = sol.y
t = sol.t

plt.plot(t, Vs, label='Vs')
plt.plot(t, Vd, label='Vd')
plt.legend()
plt.show()

plt.plot(t,q, label='q')
plt.legend()
plt.show()

plt.plot(t, Ca, label='Ca')
plt.legend()
plt.show()
