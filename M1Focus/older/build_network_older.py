from bmtk.builder import NetworkBuilder
import numpy as np
from bmtk.utils.sim_setup import build_env_bionet
import synapses
import os
from connectors import *

randseed = 123412
np.random.seed(randseed)

network_dir = 'network'
t_sim = 11500.0
dt = 0.05

num_cells = 50 # 10000
column_width, column_height = 1000., 1000.
x_start, x_end = -column_width/2, column_width/2
y_start, y_end = -column_width/2, column_width/2
z_start, z_end = 0., column_height
z_5A = 500.

min_conn_dist = 0.0 # Distance constraint for all cells
max_conn_dist = 300.0 # 300.0 #9999.

# When enabled, a shell of virtual cells will be created around the core network.
edge_effects = True

###################################################################################
####################### Cell Proportions and Positions ############################
def num_prop(ratio, N):
    """Calculate numbers of total N in proportion to ratio"""
    ratio = np.asarray(ratio)
    p = np.cumsum(np.insert(ratio.ravel(), 0, 0)) # cumulative proportion
    return np.diff(np.round(N / p[-1] * p).astype(int)).reshape(ratio.shape)

# Number of cells in each population. Following 80/20 E/I equal on CP-CS and 60% FSI to 40% LTS for Interneurons
# Densities by cell proportion unless otherwise specified: CP: 20%  CS: 20% CTH: 20% CC: 20% FSI: 12% LTS: 8%
# Corticopontine, Corticostriatal, Fast Spiking Interneuron, Low Threshold Spiker
num_CP, num_CS, num_FSI, num_LTS = num_prop([40, 40, 12, 8], num_cells)
# num_CTH = int(num_cells * 0.2)  # Corticothalamic
# num_CC = int(num_cells * 0.2)   # Corticocortical

# amount of cells per layer
numCP_in5A, numCP_in5B = num_prop([5, 95], num_CP) # CP cells are basically only in layer 5B and nowhere else.
numCS_in5A, numCS_in5B = num_prop([95, 5], num_CS) # CS cells span top of 5B to middle of 2/3

numFSI_in5A, numFSI_in5B = num_prop([1, 1], num_FSI) # Even distribution of FSI cells between Layers 5A and 5B
numLTS_in5A, numLTS_in5B = num_prop([1, 1], num_LTS) # Even distribution of LTS cells between Layers 5A and 5B

# total 400x400x1820 (ignoring layer 1)
# Order from top to bottom is 2/3, 4, 5A, 5B, 6
# Layer 2/3 (420 um thick) 23.1%
# Layer 5A (250 um thick) 13.7% (z is 250 to 499)
# Layer 5B (250 um thick) 13.7%  (z is 0 to 249)
num_cells_5A = numCP_in5A + numCS_in5A + numFSI_in5A + numLTS_in5A
num_cells_5B = numCP_in5B + numCS_in5B + numFSI_in5B + numLTS_in5B

pos_list_5A = np.random.rand(num_cells_5A, 3)
pos_list_5A[:,0] = pos_list_5A[:,0] * (x_end - x_start) + x_start
pos_list_5A[:,1] = pos_list_5A[:,1] * (y_end - y_start) + y_start
pos_list_5A[:,2] = pos_list_5A[:,2] * (z_end - z_5A) + z_5A

pos_list_5B = np.random.rand(num_cells_5B,3)
pos_list_5B[:,0] = pos_list_5B[:,0] * (x_end - x_start) + x_start
pos_list_5B[:,1] = pos_list_5B[:,1] * (y_end - y_start) + y_start
pos_list_5B[:,2] = pos_list_5B[:,2] * (z_5A - z_start) + z_start

## TODO: generate random orientations


def build_networks(network_definitions: list) -> dict:
    # network_definitions should be a list of dictionaries, e.g. [{}]
    # Keys should include an arbitrary 'network_name', a positions_list (if any),
    # And 'cells'. 'cells' should contain a list of dictionaries, and the dictionary
    # should corrospond with any valid input for BMTK's NetworkBuilder.add_nodes method
    # A dictionary of NetworkBuilder BMTK objects will be returned, reference by individual network_name
    for net_def in network_definitions:
        network_name = net_def['network_name']
        if networks.get(network_name) is None:
            networks[network_name] = NetworkBuilder(network_name)  # This is changed
        pos_list = net_def.get('positions_list', None)

        # Add cells to the network
        num = 0
        for cell in net_def['cells']:
            num_cells = cell['N']
            extra_kwargs = {}
            if pos_list is not None:
                extra_kwargs['positions'] = pos_list[num:num + num_cells]
                num += num_cells

            cell = {k: v for k, v in cell.items() if v is not None}
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            networks[network_name].add_nodes(**cell, **extra_kwargs)

    return networks

def build_edges(networks, edge_definitions, edge_params, edge_add_properties, syn=None):
    # Builds the edges for each network given a set of 'edge_definitions'
    # edge_definitions examples shown later in the code
    for edge in edge_definitions:
        network_name = edge['network']
        edge_src_trg = edge['edge']
        edge_params_val = edge_params[edge['param']]
        dynamics_file = edge_params_val['dynamics_params']
        model_template = syn[dynamics_file]['level_of_detail']

        model_template_kwarg = {'model_template': model_template}

        net = networks[network_name]

        conn = net.add_edges(**edge_src_trg, **edge_params_val, **model_template_kwarg)

        if edge.get('add_properties'):
            edge_add_properties_val = edge_add_properties[edge['add_properties']]
            conn.add_properties(**edge_add_properties_val)

def save_networks(networks,network_dir):
    # Remove the existing network_dir directory
    for f in os.listdir(network_dir):
        os.remove(os.path.join(network_dir, f))

    # Run through each network and save their nodes/edges
    for network_name, network in networks.items():
        print('Building ' + network_name)
        network.build()
        network.save_nodes(output_dir=network_dir)
        network.save_edges(output_dir=network_dir)


networks = {}   # Dictionary to store NetworkBuilder objects referenced by name
network_definitions = [
    {   # Start Layer 5A
        'network_name': 'cortex',
        'positions_list': pos_list_5A,
        'cells': [
            {   # CP
                'N': numCP_in5A,
                'pop_name': 'CP',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CP_Cell',
                'morphology': 'blank.swc'
            },
            {   # CS
                'N': numCS_in5A,
                'pop_name': 'CS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CS_Cell',
                'morphology': 'blank.swc'
            },
            {   # FSI
                'N': numFSI_in5A,
                'pop_name': 'FSI',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:FSI_Cell',
                'morphology': 'blank.swc'
            },
            {   # LTS
                'N': numLTS_in5A,
                'pop_name': 'LTS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:LTS_Cell',
                'morphology': 'blank.swc'
            }
        ]
    },  # End Layer 5A
    {   # Start Layer 5B
        'network_name': 'cortex',
        'positions_list': pos_list_5B,
        'cells': [
            {   # CP
                'N': numCP_in5B,
                'pop_name': 'CP',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CP_Cell',
                'morphology': 'blank.swc'
            },
            {   # CS
                'N': numCS_in5B,
                'pop_name': 'CS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CS_Cell',
                'morphology': 'blank.swc'
            },
            {   # FSI
                'N': numFSI_in5B,
                'pop_name': 'FSI',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:FSI_Cell',
                'morphology': 'blank.swc'
            },
            {   # LTS
                'N': numLTS_in5B,
                'pop_name': 'LTS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:LTS_Cell',
                'morphology': 'blank.swc'
            }
        ]
    },  # End Layer 5B
    {   # Extrinsic Thalamic Inputs
        'network_name': 'thalamus',
        'positions_list': None,
        'cells': [
            {   # Virtual Cells
                'N': num_CP + num_CS,
                'pop_name': 'thal',
                'potential': 'exc',
                'model_type': 'virtual'
            }
        ]
    },
    {   # Extrinsic Intbase Inputs
        'network_name': 'Intbase',
        'positions_list': None,
        'cells': [
            {   # Virtual Cells
                'N': num_FSI + num_LTS,
                'pop_name': 'Int',
                'potential': 'exc',
                'model_type': 'virtual'
            }
        ]
    }
]

##########################################################################
############################  EDGE EFFECTS  ##############################

if edge_effects: # When enabled, a shell of virtual cells will be created around the core network.

    # compute the outer shell range. The absolute max_conn_dist will extend each dimension of the core by 2*max_conn_dist
    shell_x_start, shell_y_start, shell_z_start = np.array((x_start, y_start, z_start)) - max_conn_dist
    shell_x_end, shell_y_end, shell_z_end = np.array((x_end, y_end, z_end)) + max_conn_dist

    # compute the core and shell volume
    core_volume_5A = (x_end - x_start) * (y_end - y_start) * (z_end - z_5A)
    core_volume_5B = (x_end - x_start) * (y_end - y_start) * (z_5A - z_start)
    shell_volume_5A = (shell_x_end - shell_x_start) * (shell_y_end - shell_y_start) * (shell_z_end - z_5A)
    shell_volume_5B = (shell_x_end - shell_x_start) * (shell_y_end - shell_y_start) * (z_5A - shell_z_start)

    # Increase the number of original cells based on the volume difference between core and shell
    #Layer 5A
    virt_num_cells_5A = int(round(num_cells_5A * shell_volume_5A / core_volume_5A))
    #Layer 5B
    virt_num_cells_5B = int(round(num_cells_5B * shell_volume_5B / core_volume_5B))

    # Create a positions list for cells in the shell
    virt_pos_list_5A = np.random.rand(virt_num_cells_5A, 3)
    virt_pos_list_5A[:,0] = virt_pos_list_5A[:,0] * (shell_x_end - shell_x_start) + shell_x_start
    virt_pos_list_5A[:,1] = virt_pos_list_5A[:,1] * (shell_y_end - shell_y_start) + shell_y_start
    virt_pos_list_5A[:,2] = virt_pos_list_5A[:,2] * (shell_z_end - z_5A) + z_5A
    i_shell = (virt_pos_list_5A[:,0] < x_start) | (virt_pos_list_5A[:,0] > x_end) | \
              (virt_pos_list_5A[:,1] < y_start) | (virt_pos_list_5A[:,1] > y_end) | \
              (virt_pos_list_5A[:,2] > z_end)
    virt_pos_list_5A = virt_pos_list_5A[i_shell]

    virt_pos_list_5B = np.random.rand(virt_num_cells_5B, 3)
    virt_pos_list_5B[:,0] = virt_pos_list_5B[:,0] * (shell_x_end - shell_x_start) + shell_x_start
    virt_pos_list_5B[:,1] = virt_pos_list_5B[:,1] * (shell_y_end - shell_y_start) + shell_y_start
    virt_pos_list_5B[:,2] = virt_pos_list_5B[:,2] * (z_5A - shell_z_start) + shell_z_start
    i_shell = (virt_pos_list_5B[:,0] < x_start) | (virt_pos_list_5B[:,0] > x_end) | \
              (virt_pos_list_5B[:,1] < y_start) | (virt_pos_list_5B[:,1] > y_end) | \
              (virt_pos_list_5B[:,2] < z_start)
    virt_pos_list_5B = virt_pos_list_5B[i_shell]

    # Recalculate number of cells in each layer
    virt_num_cells_5A = len(virt_pos_list_5A)
    virt_numCP_in5A, virt_numCS_in5A, virt_numFSI_in5A, virt_numLTS_in5A = \
        num_prop([numCP_in5A, numCS_in5A, numFSI_in5A, numLTS_in5A], virt_num_cells_5A)

    virt_num_cells_5B = len(virt_pos_list_5B)
    virt_numCP_in5B, virt_numCS_in5B, virt_numFSI_in5B, virt_numLTS_in5B = \
        num_prop([numCP_in5B, numCS_in5B, numFSI_in5B, numLTS_in5B], virt_num_cells_5B)

    virt_num_cells = virt_num_cells_5A + virt_num_cells_5B

    # This network should contain all the same properties as the original network, except
    # the cell should be virtual. For connectivity, you should name the cells the same as
    # the original network because connection rules defined later will require it
    shell_network = [
    {   # Start Layer 5A
        'network_name': 'shell',
        'positions_list': virt_pos_list_5A,
        'cells': [
            {   # CP
                'N': virt_numCP_in5A,
                'pop_name': 'CP',
                'model_type': 'virtual'
            },
            {   # CS
                'N': virt_numCS_in5A,
                'pop_name': 'CS',
                'model_type': 'virtual'
            },
            {   # FSI
                'N': virt_numFSI_in5A,
                'pop_name': 'FSI',
                'model_type': 'virtual'
            },
            {   # LTS
                'N': virt_numLTS_in5A,
                'pop_name': 'LTS',
                'model_type': 'virtual'
            }
        ]
    }, # End Layer 5A
    {   # Start Layer 5B
        'network_name': 'shell',
        'positions_list': virt_pos_list_5B,
        'cells': [
            {   # CP
                'N': virt_numCP_in5B,
                'pop_name': 'CP',
                'model_type': 'virtual'
            },
            {   # CS
                'N': virt_numCS_in5B,
                'pop_name': 'CS',
                'model_type': 'virtual'
            },
            {   # FSI
                'N': virt_numFSI_in5B,
                'pop_name': 'FSI',
                'model_type': 'virtual'
            },
            {   # LTS
                'N': virt_numLTS_in5B,
                'pop_name': 'LTS',
                'model_type': 'virtual'
            }
        ]
    } # End Layer 5B
]
# Add the shell to our network definitions
network_definitions.extend(shell_network)
########################## END EDGE EFFECTS ##############################
##########################################################################

# Build and save our NetworkBuilder dictionary
networks = build_networks(network_definitions)


##########################################################################
#############################  BUILD EDGES  ##############################

# Whole reason for restructuring network building lies here, by separating out the
# source and target params from the remaining parameters in NetworkBuilder's
# add_edges function we can reuse connectivity rules for the virtual shell
# or elsewhere
# [
#    {
#       'network': 'network_name', # => The name of the network that these edges should be added to (networks['network_name'])
#       'edge': {
#                    'source': {},
#                    'target': {}
#               }, # should contain source and target only, any valid add_edges param works
#       'param': 'name_of_edge_parameter' # to be coupled with when add_edges is called
#       'add_properties': 'prop_name' # name of edge_add_properties for adding additional connection props, like delay
#    }
# ]

# Will be called by conn.add_properties for the associated connection
edge_add_properties = {
    'syn_dist_delay_feng_section_default': {
        'names': ['delay', 'sec_id', 'sec_x'],
        'rule': syn_dist_delay_feng_section,
        'rule_params': {'sec_x': 0.9},
        'dtypes': [np.float, np.int32, np.float]
    },
    'syn_uniform_delay_section_default': {
        'names': ['delay', 'sec_id', 'sec_x'],
        'rule': syn_uniform_delay_section,
        'rule_params': {'sec_x': 0.9},
        'dtypes': [np.float, np.int32, np.float]
    },
    'section_id_placement': {
        'names': ['sec_id', 'sec_x'],
        'rule': section_id_placement,
        'dtypes': [np.int32, np.float]
    }
}

edge_definitions = [
    {   # FSI -> FSI Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> FSI Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'FSI2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'FSI2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'FSI2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'FSI2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> LTS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'LTS2LTS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> LTS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'LTS2LTS_rec',
        'add_properties': ''
    },
    {   # LTS -> FSI Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> FSI Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'LTS2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'LTS2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'LTS2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'LTS2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> FSI
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'CP2FSI',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> FSI
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'CS2FSI',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> LTS
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'CP2LTS',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> LTS
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'CS2LTS',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CS
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CP2CS',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CP
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CS2CP',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CP Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CP2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CP Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CP2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CS2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CS2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },

        ################### THALAMIC INPUT ################

    {   # Thalamus Excitation to CP
        'network': 'cortex',
        'edge': {
            'source': networks['thalamus'].nodes(),
            'target': networks['cortex'].nodes(pop_name=['CP'])
        },
        'param': 'Thal2CP'
    },
    {   # Thalamus Excitation to CS
        'network': 'cortex',
        'edge': {
            'source': networks['thalamus'].nodes(),
            'target': networks['cortex'].nodes(pop_name=['CS'])
        },
        'param': 'Thal2CS'
    },

        ##################### Interneuron baseline INPUT #####################

    {    # To all FSI
        'network': 'cortex',
        'edge': {
            'source': networks['Intbase'].nodes(),
            'target': networks['cortex'].nodes(pop_name=['FSI'])
        },
        'param': 'Intbase2FSI'
    },
    {    # To all LTS
        'network': 'cortex',
        'edge': {
            'source': networks['Intbase'].nodes(),
            'target': networks['cortex'].nodes(pop_name=['LTS'])
        },
        'param': 'Intbase2LTS'
    },
]

# A few connectors require a list for tracking synapses that are recurrent, declare them here
FSI_FSI_list = []
FSI_CP_list = []
FSI_CS_list = []
CP_CP_list = []
CS_CS_list = []
LTS_LTS_list = []
LTS_CP_list = []
CP_LTS_list = []
CS_LTS_list = []
FSI_LTS_list = []
LTS_FSI_list = []
# CS_CP_list = []
# CP_CS_list = []

# edge_params should contain additional parameters to be added to add_edges calls
# The following parameters for random synapse placement are not necessary
# if conn.add_properties specifies sec_id and sec_x.
# distance_range: place synapse within distance range [dmin, dmax] from soma
# target_sections: place synapse within the given sections in a list
edge_params = {
    'FSI2FSI_uni': {
        'iterator': 'one_to_all',
        'connection_rule': sphere_dropoff,
        'connection_params': {'p': 0.34, 'stdev': 131.48, 'track_list': FSI_FSI_list},
        'syn_weight': 1,
        'dynamics_params': 'FSI2FSI.json'
    },
    'FSI2FSI_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.43, 'all_edges': FSI_FSI_list},
        'syn_weight': 1,
        'dynamics_params': 'FSI2FSI.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CP_uni': {
        'iterator': 'one_to_all',
        'connection_rule': sphere_dropoff,
        'connection_params': {'p': 0.20, 'stdev': 95.98, 'track_list': FSI_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'FSI2CP.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CP_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.28, 'all_edges': FSI_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'FSI2CP.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CS_uni': {
        'iterator': 'one_to_all',
        'connection_rule': sphere_dropoff,
        'connection_params': {'p': 0.17, 'stdev': 95.98, 'track_list': FSI_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'FSI2CS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CS_rec': {
        'iterator': 'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params': {'p': 0.20, 'all_edges': FSI_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'FSI2CS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2LTS_uni': {
        'iterator': 'one_to_all',
        'connection_rule': sphere_dropoff,
        'connection_params': {'p': 0.34, 'stdev': 131.48, 'track_list': FSI_LTS_list},
        'syn_weight': 1,
        'dynamics_params': 'FSI2LTS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2LTS_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.21, 'all_edges': FSI_LTS_list},
        'syn_weight': 1,
        'dynamics_params': 'FSI2LTS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CP2FSI': {
        'iterator': 'one_to_all',
        'connection_rule': cs2cp_gaussian_cylinder,
        'connection_params': {'p': 0.32, 'stdev': 99.25},
        'syn_weight': 1,
        'dynamics_params': 'CP2FSI.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CS2FSI': {
        'iterator': 'one_to_all',
        'connection_rule': cs2cp_gaussian_cylinder,
        'connection_params': {'p': 0.22, 'stdev': 99.25},
        'syn_weight': 1,
        'dynamics_params': 'CS2FSI.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2LTS_uni': {
        'iterator': 'one_to_all',
        'connection_rule': sphere_dropoff,
        'connection_params': {'p': 0.034, 'stdev': 131.48, 'track_list': LTS_LTS_list},
        'syn_weight': 1,
        'dynamics_params': 'INT2INT.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2LTS_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.043, 'all_edges': LTS_LTS_list},
        'syn_weight': 1,
        'dynamics_params': 'INT2INT.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CP_uni': {
        'iterator': 'one_to_all',
        'connection_rule': sphere_dropoff,
        'connection_params': {'p': 0.35, 'stdev': 95.98, 'track_list': LTS_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'LTS2PN.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CP_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.17, 'all_edges': LTS_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'LTS2PN.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CS_uni': {
        'iterator': 'one_to_all',
        'connection_rule': sphere_dropoff,
        'connection_params': {'p': 0.35, 'stdev': 95.98, 'track_list': LTS_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'LTS2PN.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CS_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.17, 'all_edges': LTS_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'LTS2PN.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2FSI_uni': {
        'iterator': 'one_to_all',
        'connection_rule': sphere_dropoff,
        'connection_params': {'p': 0.53, 'stdev': 131.48, 'track_list': LTS_FSI_list},
        'syn_weight': 1,
        'dynamics_params': 'LTS2FSI.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2FSI_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.21, 'all_edges': LTS_FSI_list},
        'syn_weight': 1,
        'dynamics_params': 'LTS2FSI.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CP2LTS': {
        'iterator': 'one_to_all',
        'connection_rule': cs2cp_gaussian_cylinder,
        'connection_params': {'p': 0.57, 'stdev': 99.25, 'track_list': CP_LTS_list},
        'syn_weight': 1,
        'dynamics_params': 'PN2LTS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CP2LTS_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.2, 'all_edges': CP_LTS_list},
        'syn_weight': 1,
        'dynamics_params': 'PN2LTS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CS2LTS': {
        'iterator': 'one_to_all',
        'connection_rule': cs2cp_gaussian_cylinder,
        'connection_params': {'p': 0.57, 'stdev': 99.25, 'track_list': CS_LTS_list},
        'syn_weight': 1,
        'dynamics_params': 'PN2LTS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CS2LTS_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.2, 'all_edges': CS_LTS_list},
        'syn_weight': 1,
        'dynamics_params': 'PN2LTS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CP2CS': {
        'iterator': 'one_to_all',
        'connection_rule': cs2cp_gaussian_cylinder,
        'connection_params': {'p': 0.01, 'stdev': 124.62},
        'syn_weight': 1,
        'dynamics_params': 'CP2CS.json', # same as 'CS2CS.json'
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CS2CP': {
        'iterator': 'one_to_all',
        'connection_rule': cs2cp_gaussian_cylinder,
        'connection_params': {'p': 0.1, 'stdev': 124.62},
        'syn_weight': 1,
        'dynamics_params': 'CS2CS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CP2CP_uni': {
        'iterator': 'one_to_all',
        'connection_rule': cs2cp_gaussian_cylinder,
        'connection_params': {'p': 0.10, 'stdev': 124.62, 'track_list': CP_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'CP2CP.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CP2CP_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.3, 'all_edges': CP_CP_list},
        'syn_weight': 1,
        'dynamics_params': 'CP2CP.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CS2CS_uni': {
        'iterator': 'one_to_all',
        'connection_rule': cs2cp_gaussian_cylinder,
        'connection_params': {'p': 0.1, 'stdev': 124.62, 'track_list': CS_CS_list},
        'syn_weight': 1,
        'dynamics_params': 'CS2CS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'CS2CS_rec': {
        'iterator': 'one_to_all',
        'connection_rule': recurrent_connector_o2a,
        'connection_params': {'p': 0.3, 'all_edges': CS_CS_list},
        'syn_weight': 1,
        'dynamics_params': 'CS2CS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'target_sections': ['dend']
    },
    'Thal2CP': {
        'connection_rule': one_to_one_thal,
        'connection_params': {'offset': 1000},
        'syn_weight': 1,
        'dynamics_params': 'Thal2CP.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'delay': 0.0,
        'target_sections': ['dend']
    },
    'Thal2CS': {
        'connection_rule': one_to_one_thal,
        'connection_params': {'offset': 1000},
        'syn_weight': 1,
        'dynamics_params': 'Thal2CS.json', # same as 'Thal2CP.json'
        'distance_range': [min_conn_dist, max_conn_dist],
        'delay': 0.0,
        'target_sections': ['dend']
    },
    'Intbase2FSI': {
        'connection_rule': one_to_one_intbase,
        'connection_params': {'offset1': 4000, 'offset2': 8000},
        'syn_weight': 1,
        'dynamics_params': 'Intbase2FSI.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'delay': 0.0,
        'target_sections': ['dend']
    },
    'Intbase2LTS': {
        'connection_rule': one_to_one_intbase,
        'connection_params': {'offset1': 4000, 'offset2': 8000},
        'syn_weight': 1,
        'dynamics_params': 'Intbase2LTS.json',
        'distance_range': [min_conn_dist, max_conn_dist],
        'delay': 0.0,
        'target_sections': ['dend']
    }
} # edges referenced by name

################################################################################
############################  EDGE EFFECTS EDGES  ##############################

if edge_effects:
    # These rules are for edge effect edges. They should directly mimic the connections
    # created previously, re-use the params set above. This keeps our code DRY
    virt_edges = [
    {   # FSI -> FSI Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> FSI Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'FSI2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'FSI2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'FSI2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'FSI2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> LTS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'LTS2LTS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> LTS Reciprocal
        'network': 'cortex',
        'edge': {
            'source':  {'pop_name': ['LTS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'LTS2LTS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> LTS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'LTS2LTS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> FSI Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> FSI Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'LTS2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'LTS2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'LTS2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'LTS2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> FSI
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'CP2FSI',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> FSI
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'CS2FSI',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> LTS
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'CP2LTS',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> LTS
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'CS2LTS',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CS
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CP2CS',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CP
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CS2CP',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CP Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CP2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CP Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CP2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CS2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CS2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    }
]
edge_definitions = edge_definitions + virt_edges
########################## END EDGE EFFECTS ##############################
##########################################################################

##########################################################################
############################ GAP JUNCTIONS ###############################
net = NetworkBuilder("cortex")
conn = net.add_gap_junctions(source={'pop_name': 'FSI'}, target={'pop_name': 'FSI'},
            resistance=1500, target_sections=['somatic'],
            connection_rule=perc_conn,
            connection_params={'p': 0.4})
conn._edge_type_properties['sec_id'] = 0
conn._edge_type_properties['sec_x'] = 0.9

net = NetworkBuilder("cortex")
conn = net.add_gap_junctions(source={'pop_name': 'LTS'}, target={'pop_name': 'LTS'},
            resistance=1500, target_sections=['somatic'],
            connection_rule=perc_conn,
            connection_params={'p': 0.3})
conn._edge_type_properties['sec_id'] = 0
conn._edge_type_properties['sec_x'] = 0.9


##########################################################################
###############################  BUILD  ##################################

# Load synapse dictionaries
# see synapses.py - loads each json's in components/synaptic_models into a
# dictionary so the properties can be referenced in the files eg: syn['file.json'].get('property')
synapses.load()
syn = synapses.syn_params_dicts()

# Build your edges into the networks
build_edges(networks, edge_definitions, edge_params, edge_add_properties, syn)

# Save the network into the appropriate network dir
save_networks(networks, network_dir)

# Usually not necessary if you've already built your simulation config
build_env_bionet(
    base_dir = './',
    network_dir = network_dir,
    tstop = t_sim,
    dt = dt,
    report_vars = ['v'],
    celsius = 31.0,
    spikes_inputs=[
        ('thalamus', './input/thalamus_base.h5'),
        ('thalamus', './input/thalamus_short.h5'),
        ('thalamus', './input/thalamus_long.h5'),  # Name of population which spikes will be generated for, file
        ('Intbase', './input/Intbase.h5')
    ],
    components_dir='components',
    config_file='config.json',
    compile_mechanisms=False
)
