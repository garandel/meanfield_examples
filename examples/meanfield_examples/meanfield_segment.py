from spynnaker.pyNN.extra_algorithms.splitter_components import (
    SplitterAbstractPopulationVertexNeuronsSynapses)


#import pyNN.spiNNaker as sim
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
from pyNN.random import NumpyRNG, RandomDistribution
import matplotlib.pyplot as plt
import numpy as np
#from neo.core import AnalogSignal



runtime = 500

timestep = 1 #0.25

n_MF = 2
epsilon = 1.0
J_E = 0.1
delay = 2.0



p.setup(timestep)

pop = list()

rng = NumpyRNG(seed=1)


#-----------------------vvvv--------------------------------------#
##############---data with spinnaker_get_data([''])---#############
#-----------------------------------------------------------------#

MF_splitter = SplitterAbstractPopulationVertexNeuronsSynapses(1)

#p.set_number_of_neurons_per_core(p.extra_models.Meanfield(), 1)
MF_pop = p.Population(2,
                      p.extra_models.Meanfield(),
                      label="MF",
                      additional_parameters={"splitter": MF_splitter})

E_conn = p.FixedProbabilityConnector(epsilon, rng=rng)

#Projections

MF_proj = p.Projection(
    MF_pop, MF_pop,
    E_conn, receptor_type="excitatory",
    synapse_type=p.StaticSynapse(weight=J_E, delay=delay))

MF_pop.record(['Ve', 'Vi','w', 'gsyn_exc', 'gsyn_inh'])#, to_file='test.dat')
p.run(runtime)

data_Ve_nparray = MF_pop.spinnaker_get_data(['Ve'])
data_Vi_nparray = MF_pop.spinnaker_get_data(['Vi'])
data_W_nparray = MF_pop.spinnaker_get_data(['w'])

print(data_Ve_nparray)
print(data_Vi_nparray)
print(data_W_nparray)

fig = plt.figure()

xve=data_Ve_nparray[:,1]
yve=data_Ve_nparray[:,2]
figVe = fig.add_subplot(3,1,1)
figVe.title.set_text("Ve")
figVe.plot(xve, yve)

xvi=data_Vi_nparray[:,1]
yvi=data_Vi_nparray[:,2]
figVi = fig.add_subplot(3,1,2)
figVi.title.set_text('Vi')
figVi.plot(xvi, yvi)

xw=data_W_nparray[:,1]
yw=data_W_nparray[:,2]
figW = fig.add_subplot(3,1,3)
figW.title.set_text("W")
figW.plot(xw, yw)

fig.savefig('test_totaly.png')
p.end()

#-----------------------^^^^---------------------------------------


#data_mf = p.Population(1,p.extra_models.Meanfield())
#spikes =  p.Population(1, p.

#---------------------------------------------------------
#other_firing_rates = {'a': 0,
#                      'b': 0,
#                      'tauw': 1.,
#                      'Trefrac': 5.0,
#                      'Vreset': -65.,
#                      'delta_v': -0.5,
#                      'ampnoise': 0.0,
#                      'Timescale_inv': 0.5,
#                      'Ve': 6.,
#                      'Vi': 30.
#                     }

#other_config = {'p0':0.,
#                'p1':0.,
#                'p2':0.,
#                'p3':0.,
#                'p4':0.,
#                'p5':0.,
#                'p6':0.,
#                'p7':0.,
#                'p8':0.,
#                'p9':0.,
#                'p10':0.,
#                }
#P1 = np.load('FS-cell_CONFIG1_fit.npy')
#params = {}

#for i in range(0,11):
#    params['p'+str(i)] = P1[i]

#-----------------------------------------------------------
#pop.append(p.Population(1, p.extra_models.Meanfield()))

#pop[0].record(['Ve', 'Vi','w'])#, 'gsyn_exc', 'gsyn_inh'])#, to_file='test.dat')
#p.run(runtime)

#data = pop[0].get_data(['Ve','Vi','w'])#, 'gsyn_exc', 'gsyn_inh'])# Block
#Ve_data = pop[0].get_gata('Ve')
#print(data)
#print(data.segments[0].filter(name='Ve'))
#fig = plt.figure()

#for seg in data.segments:
#    for i in ['Ve','Vi', 'w']:#, 'gsyn_exc', 'gsyn_inh']:
#        print(seg)
#        print(seg.filter(name=i))
#        times = 
#        plt.plot()
#        fig.savefig(i'_test.png')

        
#xlim = (0, runtime)
#y = data.segments[0].filter(name='Ve')[0]
#plt.plot(xlim, y)


#test = data.segments[0].filter(name='Vi')[0]

#Figure(
#    Panel(data.segments[0].filter(name='Ve'),
#          ylabel="MF (mV)",
#          yticks=True,
#          xlim=(0, runtime)),
#    title="test",
#    annotation="Simulated with {}".format(p.name())
#)



#plt.show()




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

#p.end()
