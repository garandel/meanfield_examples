# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Standard PyNN imports
import spynnaker8 as p

# pynn plotting stuff
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# Define a synfire chain as usual
p.setup(timestep=1.0, min_delay=1.0)
nNeurons = 200  # number of neurons in each population
p.set_number_of_neurons_per_core(p.IF_curr_exp, nNeurons / 2)

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

populations = list()
projections = list()

weight_to_spike = 2.0
delay = 17

loopConnections = list()
for i in range(0, nNeurons):
    singleConnection = (i, ((i + 1) % nNeurons), weight_to_spike, delay)
    loopConnections.append(singleConnection)

injectionConnection = [(0, 0, weight_to_spike, 1)]
spikeArray = {'spike_times': [[0]]}
populations.append(p.Population(
    nNeurons, p.IF_curr_exp(**cell_params_lif),  label='pop_1'))
populations.append(
    p.Population(1, p.SpikeSourceArray(**spikeArray), label='inputSpikes_1'))

projections.append(p.Projection(populations[0], populations[0],
                   p.FromListConnector(loopConnections)))
projections.append(p.Projection(populations[1], populations[0],
                   p.FromListConnector(injectionConnection)))

populations[0].record('spikes')

# Activate live output for the population
p.external_devices.activate_live_output_for(
    populations[0], database_notify_host="localhost",
    database_notify_port_num=19999)

# Start the simulation
p.run(5000)

spikes = populations[0].get_data("spikes")

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, 5000)),
    title="Simple synfire chain example with injected spikes",
    annotations="Simulated with {}".format(p.name())
)
plt.show()

p.end()
