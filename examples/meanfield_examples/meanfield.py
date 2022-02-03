from spynnaker.pyNN.extra_algorithms.splitter_components import (
    SplitterAbstractPopulationVertexNeuronsSynapses)


#import pyNN.spiNNaker as sim
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
#from neo.core import AnalogSignal



runtime = 500

time_step = 1 #0.25

n_neurons = 10

p.setup(time_step)

pop = list()

#-----------------------vvvv--------------------------------------#
##############---data with spinnaker_get_data([''])---#############
#-----------------------------------------------------------------#

#MF_splitter = SplitterAbstractPopulationVertexNeuronsSynapses(1)

#p.set_number_of_neurons_per_core(p.extra_models.Meanfield(), 1)
pop.append(p.Population(1,
                        p.extra_models.Meanfield()))#,additional_parameters={"splitter": MF_splitter}))

pop[0].record(['Ve', 'Vi','w'])#, 'gsyn_exc', 'gsyn_inh'])#, to_file='test.dat')
p.run(runtime)

data_Ve_nparray = pop[0].spinnaker_get_data(['Ve'])
data_Vi_nparray = pop[0].spinnaker_get_data(['Vi'])
data_W_nparray = pop[0].spinnaker_get_data(['w'])

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

fig.savefig('test_totality.png')

p.end()
