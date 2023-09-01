import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from brian2 import *
import h5py
defaultclock.dt = 0.04*ms

def myhist(arr):
    dens, _ = np.histogram(arr, range = [-90, 50], bins = 500, density=True)
    dens = softmax(dens)
    return dens

Cm = 1.3*uF # /cm**2
gL = 0.05*mS
EL = -70*mV #  -54.4*mV
ENa = 90*mV
EK = -100*mV #  -77*mV
EH = -32*mV # -20*mV
Eexc = 0 *mV
Einh = -75*mV

gNa = 30*mS
gK = 23*mS
gKA = 16.0*mS
gH = 8.0 * mS
sigma = 0.4*mV

omega_1_e = 2 * Hz # [0.2 2]
omega_2_e = 5 * Hz # [4  12]
omega_3_e = 35 * Hz # [25  45]
omega_4_e = 75 * Hz # [50  90]


ampl_1_e = 0.2 * mS # [0.2 5]
ampl_2_e = 0.1 * mS # [0.2 5]
ampl_3_e = 0.1 * mS # [0.2 5]
ampl_4_e = 0.05 * mS # [0.2 5]


omega_1_i = 0.5 * Hz # [0.2 2]
omega_2_i = 6 * Hz # [4  12]
omega_3_i = 40 * Hz # [25  45]
omega_4_i = 70 * Hz # [50  90]


ampl_1_i = 2 * mS # [0.2 25]
ampl_2_i = 1 * mS # [0.2 25]
ampl_3_i = 0.5 * mS # [0.2 25]
ampl_4_i = 0.2 * mS # [0.2 25]




N = 1000

# OLM Model
eqs = '''
dV/dt = (INa + IKdr + IL + IKA + IH + Iexc + Iinh + Iext)/Cm + sigma*xi/ms**0.5 : volt
IL = gL*(EL - V)           : ampere
INa = gNa*m**3*h*(ENa - V) : ampere
IKdr = gK*n**4*(EK - V) : ampere
IKA = gKA * a*b * (EK - V) :  ampere
IH = gH * r * (EH - V) :  ampere
#dm/dt = (alpha_m*(1-m)-beta_m*m) : 1
m = alpha_m / (alpha_m + beta_m) : 1
alpha_m = 1.0 / exprel(-(V+38*mV)/(10*mV))/ms : Hz
beta_m = 4*exp(-(V+65*mV)/(18*mV))/ms : Hz
dh/dt = (alpha_h*(1-h)-beta_h*h) : 1
alpha_h = 0.07*exp(-(V+63*mV)/(20*mV))/ms : Hz
beta_h = 1./(exp(-0.1/mV*(V+33*mV))+1)/ms : Hz
dn/dt = (alpha_n*(1-n)-beta_n*n) : 1
alpha_n = 0.018/mV * (V - 25*mV) / (1 - exp( (V - 25*mV)/(-25*mV)) )/ms : Hz
beta_n = 0.0036*((V-35*mV)/mV) / (exp( (V-35*mV)/(12*mV) ) - 1) /ms : Hz
da/dt = (a_inf - a) / tau_a : 1
a_inf = 1 / (1 + exp( (V + 14*mV)/(-16*mV) ) ) : 1
tau_a = 5*ms : second
db/dt = (b_inf - b) / tau_b : 1
b_inf = 1 / (1 + exp( (V + 71*mV)/(7.3*mV) ) ) : 1
tau_b = 1 / (0.000009/(exp((V - 26*mV)/(18.5*mV)) + 0.014/(0.2 + exp((V+70*mV)/(-11*mV)))))*ms : second
dr/dt = (r_inf - r) / tau_r : 1
r_inf =  1 / (1 + exp( (V + 84*mV) / (10.2*mV) ) ) : 1
tau_r = 1 / (exp(-14.59 - 0.086*V/mV) + exp(-1.87 + 0.0701*V/mV) ) * ms: second

Iexc = gexc*(Eexc - V)           : ampere
Iinh = ginh*(Einh - V)           : ampere
gexc = ampl_1_e*0.5*(cos(2*pi*t*omega_1_e) + 1 ) + ampl_2_e*0.5*(cos(2*pi*t*omega_2_e) + 1 ) + ampl_3_e*0.5*(cos(2*pi*t*omega_3_e) + 1 ) + ampl_4_e*0.5*(cos(2*pi*t*omega_4_e) + 1 ) : siemens
ginh = ampl_1_i*0.5*(cos(2*pi*t*omega_1_i) + 1 ) + ampl_2_i*0.5*(cos(2*pi*t*omega_2_i) + 1 ) + ampl_3_i*0.5*(cos(2*pi*t*omega_3_i) + 1 ) + ampl_4_i*0.5*(cos(2*pi*t*omega_4_i) + 1 ) : siemens
'''





neuron = NeuronGroup(N, eqs, method='heun', namespace={"Iext" : 0.0*uA})
neuron.V = -90*mV
neuron.n = 0.09
neuron.h = 1.0


M_full_V = StateMonitor(neuron, 'V', record=np.arange(N))
gexc_monitor = StateMonitor(neuron, 'gexc', record=0)
ginh_monitor = StateMonitor(neuron, 'ginh', record=0)


run(2000*ms, report='text')

Varr = np.asarray(M_full_V.V/mV)

plt.plot(Varr[400:500, :].T)
plt.show()

hists = np.apply_along_axis(myhist, 0, Varr)
hists = hists.astype(np.float32)
print(hists.shape)

file = h5py.File('data.hdf5', mode='w')
file.create_dataset('density', data=hists)
file.create_dataset('gexc', data=np.asarray(gexc_monitor.gexc/mS).astype(np.float32))
file.create_dataset('ginh', data=np.asarray(ginh_monitor.ginh/mS).astype(np.float32))
file.close()

# #plotting for visual control
# path = '/home/ivan/Data/lstm_dens/'
# Vbins = np.linspace(-90, 50, 500)
# for idx in range(0, hists.shape[1], 20):
#     fig, axes = plt.subplots(nrows=1, sharex=True)
#     axes.plot(Vbins, hists[:, idx])
#     fig.savefig(path + str(idx) + '.png', dpi=50)
#     plt.close(fig)
#
#
# plt.show()
