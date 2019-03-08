from pinsky_rinzel_pump import pinskyrinzel as pr
import matplotlib.pyplot as plt
import numpy as np

my_cell = pr.PinskyRinzel(309, 18, 144, 18, 144, 140, 4, 140, 4, 6, 130, 6, 130, 0, 0, 0, 0, -152, -18, -152, -18, 0, 0, \
    0.001, 0.999, 0.009, 0.007, 0.01)

V = np.linspace(-0.12, 0.04, 101)
Ca = np.linspace(1e-3, 1e-3+300e-9, 101)

n_inf = my_cell.alpha_n(V) / (my_cell.alpha_n(V) + my_cell.beta_n(V))
tau_n = 1.0 / (my_cell.alpha_n(V) + my_cell.beta_n(V))

h_inf = my_cell.alpha_h(V) / (my_cell.alpha_h(V) + my_cell.beta_h(V))
tau_h = 1.0 / (my_cell.alpha_h(V) + my_cell.beta_h(V))

s_inf = my_cell.alpha_s(V) / (my_cell.alpha_s(V) + my_cell.beta_s(V))
tau_s = 1.0 / (my_cell.alpha_s(V) + my_cell.beta_s(V))

c_inf = np.zeros(len(V))
tau_c = np.zeros(len(V))
for i in range(len(V)):
    c_inf[i] = my_cell.alpha_c(V[i]) / (my_cell.alpha_c(V[i]) + my_cell.beta_c(V[i]))
    tau_c[i] = 1.0 / (my_cell.alpha_c(V[i]) + my_cell.beta_c(V[i]))

#q_inf = np.zeros(len(Ca))
#tau_q = np.zeros(len(Ca))
#for i in range(len(Ca)):
#    q_inf[i] = min(2e4*Ca[i],10) / (min(2e4*Ca[i],10) + 1)
#    tau_q[i] = 1.0 / (min(2e4*Ca[i],10) + 1)

f1 = plt.figure(1)
plt.subplot(121)
plt.plot(V*1000, n_inf)
plt.title(r'$n_{\infty}$')
plt.xlabel('V [mV]')
plt.ylabel(r'$n_{\infty}$')
plt.subplot(122)
plt.plot(V*1000, tau_n*1000)
plt.title(r'$\tau_n$')
plt.xlabel('V [mV]')
plt.ylabel(r'$\tau_n$ [ms]')
plt.tight_layout()

f2 = plt.figure(2)
plt.subplot(121)
plt.plot(V*1000, h_inf)
plt.title(r'$h_{\infty}$')
plt.xlabel('V [mV]')
plt.ylabel(r'$h_{\infty}$')
plt.subplot(122)
plt.plot(V*1000, tau_h*1000)
plt.title(r'$\tau_h$')
plt.xlabel('V [mV]')
plt.ylabel(r'$\tau_h$ [ms]')
plt.tight_layout()

f3 = plt.figure(3)
plt.subplot(121)
plt.plot(V*1000, s_inf)
plt.title(r'$s_{\infty}$')
plt.xlabel('V [mV]')
plt.ylabel(r'$s_{\infty}$')
plt.subplot(122)
plt.plot(V*1000, tau_s*1000)
plt.title(r'$\tau_s$')
plt.xlabel('V [mV]')
plt.ylabel(r'$\tau_s$ [ms]')
plt.tight_layout()

f4 = plt.figure(4)
plt.subplot(121)
plt.plot(V*1000, c_inf)
plt.title(r'$c_{\infty}$')
plt.xlabel('V [mV]')
plt.ylabel(r'$c_{\infty}$')
plt.subplot(122)
plt.plot(V*1000, tau_c*1000)
plt.title(r'$\tau_c$')
plt.xlabel('V [mV]')
plt.ylabel(r'$\tau_c$ [ms]')
plt.tight_layout()

#f5 = plt.figure(5)
#plt.subplot(121)
#plt.plot(V*1000, q_inf)
#plt.title(r'$q_{\infty}$')
#plt.xlabel('V [mV]')
#plt.ylabel(r'$q_{\infty}$')
#plt.subplot(122)
#plt.plot(V*1000, tau_q*1000)
#plt.title(r'$\tau_q$')
#plt.xlabel('V [mV]')
#plt.ylabel(r'$\tau_q$ [ms]')
#plt.tight_layout()

plt.show()
