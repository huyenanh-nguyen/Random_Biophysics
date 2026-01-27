from neuron import h, gui

# create a simple soma
soma = h.Section(name='soma')
soma.L = soma.diam = 20  # microns

# insert Hodgkin-Huxley channels
soma.insert('hh')

# current clamp
stim = h.IClamp(soma(0.5))
stim.delay = 5
stim.dur = 1
stim.amp = 0.1

# record membrane potential
v = h.Vector().record(soma(0.5)._ref_v)
t = h.Vector().record(h._ref_t)

# run simulation
h.finitialize(-65)
h.continuerun(40)

# print result
print("Final membrane potential:", v[-1])
