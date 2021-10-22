#import pyNN.spiNNaker as sim
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
from neo.core import AnalogSignal

#from spynnaker.pyNN.models.neuron.builds.meanfield_base import MeanfieldBase

runtime = 50

time_step = 1.0

n_neurons = 10

p.setup(time_step)
#sim.set_numbre_of_neurons_per_core(sim.MeanfieldBase, 1)
pop = list()

pop.append(p.Population(1, p.extra_models.Meanfield()))

pop[0].record(['Ve', 'Vi', 'Fout_th', 'gsyn_exc', 'gsyn_inh'])#, to_file='test.dat')
#pop.record(['Vi'])
#pop.record('Fout_th')

p.run(runtime)

data = pop[0].get_data(['Ve','Vi', 'Fout_th', 'gsyn_exc', 'gsyn_inh'])# Block
#Ve_data = pop[0].get_gata('Ve')
#data = pop[0].get_data(['Vi'])

data_Ve_nparray = pop[0].spinnaker_get_data(['Ve'])
data_Vi_nparray = pop[0].spinnaker_get_data(['Vi'])

data_Fout_th_nparray = pop[0].spinnaker_get_data(['Fout_th'])

#Vi_pop = pop.get_data(['Vi'])
#Fout = pop.get_data('Fout_th')

#print(data)
print(data.segments[0])

for seg in data.segments:
    for i in ['Ve','Vi', 'Fout_th', 'gsyn_exc', 'gsyn_inh']:
        print(seg)
        print(seg.filter(name=i))

#test = data.segments[0].filter(name='Vi')[0]
print(data_Ve_nparray)
print(data_Vi_nparray)
print(data_Fout_th_nparray)

#Figure(
#    Panel(data.segments[0].filter(name='Vi')[0],
#          ylabel="Membrane potential (mV)",
#          data_labels=[pop[0].label], yticks=True, xlim=(0, runtime))
#)
#plt.show()
#print(data.segments[0].filter(name='Ve')[0])
#print(data.segments[0].filter(name='Vi')[0])
#print(Fout.segments[0].filter(name='Fout_th')[0])
#print(np.linspace(0, runtime))

###############################################################
#v = neo.segments[0].filter(name='Ve')[0]
#print(v)

#runtime =500
#p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
#nNeurons=1
#p.set_number_of_neurons_per_core(p.Meanfield, nNeurons)

#x = Ve_pop.segments[0].filter(name='Ve')[0]
#y = err_pop.segments[0].filter(name='err_func')[0]

#Figure(
#    Panel(err_pop.segments[0].filter(name='err_pop')[0],
#          ylabel="MF (mV)",
#          yticks=True,
#          xlim=(0, runtime),
#          xticks=True),
#    title="test"
#)
#plt.plot(y)
#plt.show()

p.end()
