import numpy as np
import os
from functools import partial
from bmtk.builder import NetworkBuilder
from bmtk.utils.sim_setup import build_env_bionet
import synapses
import connectors
from connectors import (
    spherical_dist, cylindrical_dist_z, GaussianDropoff, UniformInRange,
    pr_2_rho, ReciprocalConnector, UnidirectionConnector,
    OneToOneSequentialConnector, CorrelatedGapJunction,
    syn_dist_delay_feng_section_PN, syn_section_PN, syn_dist_delay_feng
)

##############################################################################
############################## General Settings ##############################

randseed = 1234
rng = np.random.default_rng(randseed)
connectors.rng = rng

network_dir = 'network'
t_sim = 31000.0  # ms
dt = 0.1  # ms

# Network size and dimensions
num_cells = 1000  # 10000
column_width, column_height = 1000., 1000.
x_start, x_end = - column_width / 2, column_width / 2
y_start, y_end = - column_width / 2, column_width / 2
z_start, z_end = - column_height / 2, column_height / 2
z_5A = 0.

# Distance constraint for all cells
min_conn_dist = 20.0  # um. PN soma diameter
max_conn_dist = 300.0  # or np.inf
# Distance range for total probability in estimation of Gaussian drop function
# ptotal_dist_range = (0., 150.)

# When enabled, a shell of virtual cells will be created around the core cells.
edge_effects = False

##############################################################################
####################### Cell Proportions and Positions #######################

def num_prop(ratio, N):
    """Calculate numbers of total N in proportion to ratio"""
    ratio = np.asarray(ratio)
    p = np.cumsum(np.insert(ratio.ravel(), 0, 0))  # cumulative proportion
    return np.diff(np.round(N / p[-1] * p).astype(int)).reshape(ratio.shape)


# Number of cells in each population.
# Following 80/20 E/I equal on CP-CS and 60% FSI to 40% LTS for Interneurons
# Densities by cell proportion unless otherwise specified:
# CP: 20%  CS: 20% CTH: 20% CC: 20% FSI: 12% LTS: 8%
# Corticopontine, Corticostriatal,
# Fast Spiking Interneuron, Low Threshold Spiker
num_CP, num_CS, num_FSI, num_LTS = num_prop([40, 40, 12, 8], num_cells)
# num_CTH = int(num_cells * 0.2)  # Corticothalamic
# num_CC = int(num_cells * 0.2)   # Corticocortical

# Amount of cells per layer
# CP cells are basically only in layer 5B and nowhere else.
numCP_5A, numCP_5B = num_prop([5, 95], num_CP)
# CS cells span top of 5B to middle of 2/3
numCS_5A, numCS_5B = num_prop([95, 5], num_CS)
# Even distribution of FSI cells between Layers 5A and 5B
numFSI_5A, numFSI_5B = num_prop([1, 1], num_FSI)
# Even distribution of LTS cells between Layers 5A and 5B
numLTS_5A, numLTS_5B = num_prop([1, 1], num_LTS)

# Total 400x400x1820 (ignoring layer 1)
# Order from top to bottom is 2/3, 4, 5A, 5B, 6
# Layer 2/3 (420 um thick) 23.1%
# Layer 5A (250 um thick) 13.7% (z is 250 to 499)
# Layer 5B (250 um thick) 13.7%  (z is 0 to 249)
num_cells_5A = numCP_5A + numCS_5A + numFSI_5A + numLTS_5A
num_cells_5B = numCP_5B + numCS_5B + numFSI_5B + numLTS_5B


# Generate random cell positions
# Use poisson-disc sampling to generate positions with minimum distance limit.
use_poiss_disc = False

# Get positions for cells in the core
def samples_in_core(samples):
    core_idx = (samples[:, 0] >= x_start) & (samples[:, 0] <= x_end) & \
        (samples[:, 1] >= y_start) & (samples[:, 1] <= y_end) & \
        (samples[:, 2] >= z_start) & (samples[:, 2] <= z_end)
    pos_list_5 = samples[core_idx]  # layer 5 volume
    idx_5A = pos_list_5[:, 2] >= z_5A  # index in 5A, others in 5B
    return core_idx, pos_list_5[idx_5A], pos_list_5[~idx_5A]

# Generate samples in cube with side_length
side_length = max(column_width, column_height)
if edge_effects:
    side_length += 2 * max_conn_dist  # Extend by 2 * max_conn_dist

    # Compute the outer shell range. Extend the edge by max_conn_dist.
    shell_x_start, shell_y_start, shell_z_start = \
        np.array((x_start, y_start, z_start)) - max_conn_dist
    shell_x_end, shell_y_end, shell_z_end = \
        np.array((x_end, y_end, z_end)) + max_conn_dist

    # Compute the core and shell volume
    core_volume_5A = (x_end - x_start) * (y_end - y_start) * (z_end - z_5A)
    core_volume_5B = (x_end - x_start) * (y_end - y_start) * (z_5A - z_start)
    shell_volume_5A = (shell_x_end - shell_x_start) * \
        (shell_y_end - shell_y_start) * (shell_z_end - z_5A) - core_volume_5A
    shell_volume_5B = (shell_x_end - shell_x_start) * \
        (shell_y_end - shell_y_start) * (z_5A - shell_z_start) - core_volume_5B

    # Determine the number of shell cells with the same density
    virt_num_cells_5A = int(round(num_cells_5A *
                                  shell_volume_5A / core_volume_5A))
    virt_num_cells_5B = int(round(num_cells_5B *
                                  shell_volume_5B / core_volume_5B))

    # Get positions for cells in the shell
    def samples_in_shell(samples):
        shell_idx = (samples[:, 0] >= shell_x_start) &\
            (samples[:, 0] <= shell_x_end) & \
            (samples[:, 1] >= shell_y_start) & \
            (samples[:, 1] <= shell_y_end) & \
            (samples[:, 2] >= shell_z_start) &\
            (samples[:, 2] <= shell_z_end)
        pos_list_5 = samples[shell_idx]
        idx_5A = pos_list_5[:, 2] >= z_5A  # index in 5A, others in 5B
        return pos_list_5[idx_5A], pos_list_5[~idx_5A]

# Generate samples in cube [0, 1]^3, then scale it to side_length and center it
def scale_cube(samples):
    return side_length * (samples - 0.5)


if use_poiss_disc:
    from scipy.stats import qmc  # qmc.PoissonDisk new in scipy 1.10.0

    ncand = 30  # number of candidates (related to density of points)
    radius = min_conn_dist / side_length
    engine = qmc.PoissonDisk(d=3, radius=radius, ncandidates=ncand, seed=rng)
    samples = scale_cube(engine.fill_space())

    core_idx, pos_list_5A, pos_list_5B = samples_in_core(samples)
    print("Number of positions in 5A, 5B: (%d, %d)"
          % (len(pos_list_5A), len(pos_list_5B)))
    print("Number of cells in 5A, 5B: (%d, %d)"
          % (num_cells_5A, num_cells_5B))
    if len(pos_list_5A) < num_cells_5A or len(pos_list_5B) < num_cells_5B:
        raise ValueError("There are not enough position samples generated.")
    if edge_effects:
        shell_pos_list_5A, shell_pos_list_5B = \
            samples_in_shell(samples[~core_idx])
        print("Number of positions in 5A, 5B: (%d, %d)"
              % (len(shell_pos_list_5A), len(shell_pos_list_5B)))
        print("Number of cells in 5A, 5B: (%d, %d)"
              % (virt_num_cells_5A, virt_num_cells_5B))
        if len(shell_pos_list_5A) < virt_num_cells_5A or \
                len(shell_pos_list_5B) < virt_num_cells_5B:
            raise ValueError("There are not enough position samples "
                             "generated in shell.")
else:
    cell_dens = num_cells / (column_width * column_width * column_height)
    num_pos = int(cell_dens * side_length ** 3)
    samples = scale_cube(rng.random((num_pos, 3)))
    num_pos = int(0.1 * num_pos)
    while True:
        core_idx, pos_list_5A, pos_list_5B = samples_in_core(samples)
        add_samples = len(pos_list_5A) < num_cells_5A \
            or len(pos_list_5B) < num_cells_5B
        if edge_effects:
            shell_pos_list_5A, shell_pos_list_5B = \
                samples_in_shell(samples[~core_idx])
            add_samples = add_samples \
                or len(shell_pos_list_5A) < virt_num_cells_5A \
                or len(shell_pos_list_5B) < virt_num_cells_5B
        if add_samples:
            new_samples = scale_cube(rng.random((num_pos, 3)))
            samples = np.concatenate((samples, new_samples), axis=0)
        else:
            break

# Draw desired number of samples from the position list
pos_list_5A = rng.choice(pos_list_5A, num_cells_5A, replace=False)
pos_list_5B = rng.choice(pos_list_5B, num_cells_5B, replace=False)

if edge_effects:
    shell_pos_list_5A = rng.choice(shell_pos_list_5A,
                                   virt_num_cells_5A, replace=False)
    shell_pos_list_5B = rng.choice(shell_pos_list_5B,
                                   virt_num_cells_5B, replace=False)

    # Keep only the PN cells in the lateral shell around the core
    def shell_PN_5A(pos_list):
        return pos_list[pos_list[:, 2] <= z_end]

    def shell_PN_5B(pos_list):
        return pos_list[pos_list[:, 2] >= z_start]

    virt_numPN_5A, virt_numITN_5A = num_prop(
        [numCP_5A + numCS_5A, numFSI_5A + numLTS_5A], virt_num_cells_5A)
    virt_numPN_5B, virt_numITN_5B = num_prop(
        [numCP_5B + numCS_5B, numFSI_5B + numLTS_5B], virt_num_cells_5B)

    PN_list_5A = shell_PN_5A(shell_pos_list_5A[:virt_numPN_5A])
    ITN_list_5A = shell_pos_list_5A[virt_numPN_5A:]
    shell_pos_list_5A = np.concatenate((PN_list_5A, ITN_list_5A))
    PN_list_5B = shell_PN_5B(shell_pos_list_5B[:virt_numPN_5B])
    ITN_list_5B = shell_pos_list_5B[virt_numPN_5B:]
    shell_pos_list_5B = np.concatenate((PN_list_5B, ITN_list_5B))

    virt_numCP_5A, virt_numCS_5A = num_prop(
        [numCP_5A, numCS_5A], len(PN_list_5A))
    virt_numFSI_5A, virt_numLTS_5A = num_prop(
        [numFSI_5A, numLTS_5A], virt_numITN_5A)
    virt_numCP_5B, virt_numCS_5B = num_prop(
        [numCP_5B, numCS_5B], len(PN_list_5B))
    virt_numFSI_5B, virt_numLTS_5B = num_prop(
        [numFSI_5B, numLTS_5B], virt_numITN_5B)

# TODO: generate random orientations


##############################################################################
####################### Functions for Building Network #######################

def build_networks(network_definitions: list) -> dict:
    """
    `network_definitions` should be a list of dictionaries, e.g. [{}, {}, ...]
    Keys should include an arbitrary `network_name`, a positions_list (if any),
    and `cells`. `cells` should contain a list of dictionaries, and each
    dictionary should corrospond with any valid input for BMTK
    NetworkBuilder.add_nodes() method. A dictionary of BMTK NetworkBuilder
    objects will be returned, reference by individual network_name."""
    for net_def in network_definitions:
        network_name = net_def['network_name']
        if networks.get(network_name) is None:
            networks[network_name] = NetworkBuilder(network_name)
        pos_list = net_def.get('positions_list')

        # Add cells to the network
        num = 0
        for cell in net_def['cells']:
            num_cells = cell['N']
            extra_kwargs = {}
            if pos_list is not None:
                extra_kwargs['positions'] = pos_list[num:num + num_cells]
                num += num_cells

            cell = {k: v for k, v in cell.items() if v is not None}
            extra_kwargs = {k: v for k, v in extra_kwargs.items()
                            if v is not None}
            networks[network_name].add_nodes(**cell, **extra_kwargs)

    return networks


def build_edges(networks, edge_definitions, edge_params,
                edge_add_properties, syn):
    """
    Builds the edges for each network given a set of 'edge_definitions',
    examples shown later in the code
    """
    for edge in edge_definitions:
        network_name = edge['network']
        net = networks[network_name]
        # edge arguments
        print("Adding edge: " + edge['param'])
        edge_params_val = edge_params[edge['param']].copy()
        # get synapse template file
        dynamics_file = edge_params_val.get('dynamics_params')
        model_template = syn[dynamics_file]['level_of_detail']
        # get source and target nodes
        edge_src_trg = edge.get('edge')
        if edge_src_trg:
            edge_src_trg = edge_src_trg.copy()
            src_net = edge_src_trg.pop('source_network', network_name)
            trg_net = edge_src_trg.pop('target_network', network_name)
            source = networks[src_net].nodes(**edge_src_trg.get('source', {}))
            target = networks[trg_net].nodes(**edge_src_trg.get('target', {}))
            edge_params_val.update({'source': source, 'target': target})
        # use connector class
        connector_class = edge_params_val.pop('connector_class', None)
        if connector_class is not None:
            # create a connector object
            connector_params = edge_params_val.pop('connector_params', {})
            connector = connector_class(**connector_params)
            # keep object reference in the dictionary
            edge_params[edge['param']]['connector_object'] = connector
            if edge_src_trg:
                connector.setup_nodes(source=source, target=target)
            edge_params_val.update(connector.edge_params())
        conn = net.add_edges(model_template=model_template, **edge_params_val)

        edge_properties = edge.get('add_properties')
        if edge_properties:
            edge_properties_val = edge_add_properties[edge_properties].copy()
            if connector_class is not None:
                # pass connector object to the rule for edge properties
                edge_properties_val['rule'] = partial(
                    edge_properties_val['rule'], connector=connector)
            conn.add_properties(**edge_properties_val)


def get_connector(param):
    """Get connector object stored in edge_params"""
    edge_params_val = edge_params[param]
    if 'connector_object' in edge_params_val:
        return edge_params_val['connector_object']
    else:
        raise ValueError("No connector used in '%s'" % param)


def save_networks(networks, network_dir):
    """Build and save network"""
    # Remove the existing network_dir directory
    for f in os.listdir(network_dir):
        os.remove(os.path.join(network_dir, f))

    # Run through each network and save their nodes/edges
    for network_name, network in networks.items():
        print('Building ' + network_name)
        network.build()
        network.save_nodes(output_dir=network_dir)
        network.save_edges(output_dir=network_dir)


##############################################################################
############################ Network Definitions #############################

# Dictionary to store NetworkBuilder objects referenced by name
networks = {}
network_definitions = [
    {   # Start Layer 5A
        'network_name': 'cortex',
        'positions_list': pos_list_5A,
        'cells': [
            {   # CP
                'N': numCP_5A,
                'pop_name': 'CP',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CP_Cell',
                'morphology': 'blank.swc'
            },
            {   # CS
                'N': numCS_5A,
                'pop_name': 'CS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CS_Cell',
                'morphology': 'blank.swc'
            },
            {   # FSI
                'N': numFSI_5A,
                'pop_name': 'FSI',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:FSI_Cell',
                'morphology': 'blank.swc'
            },
            {   # LTS
                'N': numLTS_5A,
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
                'N': numCP_5B,
                'pop_name': 'CP',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CP_Cell',
                'morphology': 'blank.swc'
            },
            {   # CS
                'N': numCS_5B,
                'pop_name': 'CS',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:CS_Cell',
                'morphology': 'blank.swc'
            },
            {   # FSI
                'N': numFSI_5B,
                'pop_name': 'FSI',
                'rotation_angle_zaxis': None,
                'rotation_angle_yaxis': None,
                'model_type': 'biophysical',
                'model_template': 'hoc:FSI_Cell',
                'morphology': 'blank.swc'
            },
            {   # LTS
                'N': numLTS_5B,
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
    {   # Extrinsic Baseline Inputs
        'network_name': 'baseline',
        'positions_list': None,
        'cells': [
            {   # Virtual Cells
                'N': num_cells,
                'pop_name': 'base',
                'potential': 'exc',
                'model_type': 'virtual'
            }
        ]
    }
]


##############################################################################
################################ EDGE EFFECTS ################################

if edge_effects:
    # This network should contain all the same properties as the original
    # network, except the cell should be virtual. For connectivity, you should
    # name the cells the same as the original network because connection rules
    # defined later will require it
    shell_network = [
    {   # Start Layer 5A
        'network_name': 'shell',
        'positions_list': shell_pos_list_5A,
        'cells': [
            {   # CP
                'N': virt_numCP_5A,
                'pop_name': 'CP',
                'model_type': 'virtual'
            },
            {   # CS
                'N': virt_numCS_5A,
                'pop_name': 'CS',
                'model_type': 'virtual'
            },
            {   # FSI
                'N': virt_numFSI_5A,
                'pop_name': 'FSI',
                'model_type': 'virtual'
            },
            {   # LTS
                'N': virt_numLTS_5A,
                'pop_name': 'LTS',
                'model_type': 'virtual'
            }
        ]
    },  # End Layer 5A
    {   # Start Layer 5B
        'network_name': 'shell',
        'positions_list': shell_pos_list_5B,
        'cells': [
            {   # CP
                'N': virt_numCP_5B,
                'pop_name': 'CP',
                'model_type': 'virtual'
            },
            {   # CS
                'N': virt_numCS_5B,
                'pop_name': 'CS',
                'model_type': 'virtual'
            },
            {   # FSI
                'N': virt_numFSI_5B,
                'pop_name': 'FSI',
                'model_type': 'virtual'
            },
            {   # LTS
                'N': virt_numLTS_5B,
                'pop_name': 'LTS',
                'model_type': 'virtual'
            }
        ]
    }  # End Layer 5B
    ]

    # Add the shell to our network definitions
    network_definitions.extend(shell_network)

############################## END EDGE EFFECTS ##############################
##############################################################################

# Build and save our NetworkBuilder dictionary
networks = build_networks(network_definitions)


##########################################################################
#############################  BUILD EDGES  ##############################

# Whole reason for restructuring network building lies here, by separating out
# the source and target params from the remaining parameters in
# NetworkBuilder.add_edges() function we can reuse connectivity rules for the
# virtual shell or elsewhere
# [
#  {
#   'network': 'network_name',  # Name of the network to which edges are added
#   'edge': {
#       'source': {},  # dictionary of properties of desired population
#       'target': {},
#       'source_network': 'network_name1'  # network name for the population
#       'target_network': 'network_name2'  # if different from network_name
#       },  # source and target are required
#       # source_network and target_network are optional
#   'param': 'name_of_edge_parameters',  # to be coupled with when add_edges()
#   'add_properties': 'prop_name'  # name of edge_add_properties for additional
#       # connection properties, like delay
#   }
# ]


edge_definitions = [
    {   # CP -> CP Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CP2CP',
        'add_properties': 'syn_dist_delay_feng_section_PN'
    },
    {   # CS -> CS Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CS2CS',
        'add_properties': 'syn_dist_delay_feng_section_PN'
    },
    {   # CP -> CS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'CP2CS',
        'add_properties': 'syn_dist_delay_feng_section_PN'
    },
    {   # CS -> CP Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'CS2CP',
        'add_properties': 'syn_dist_delay_feng_section_PN'
    },
    {   # FSI -> FSI Reciprocal
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # LTS -> LTS Unidirectional
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'LTS2LTS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # FSI -> LTS forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # FSI <- LTS backward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CP -> FSI forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'CP2FSI',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CP <- FSI backward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'FSI2CP',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CS -> FSI forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'CS2FSI',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CS <- FSI backward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['FSI']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'FSI2CS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CP -> LTS forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CP']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'CP2LTS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CP <- LTS backward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CP']}
        },
        'param': 'LTS2CP',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CS -> LTS forward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['CS']},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'CS2LTS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
    {   # CS <- LTS backward
        'network': 'cortex',
        'edge': {
            'source': {'pop_name': ['LTS']},
            'target': {'pop_name': ['CS']}
        },
        'param': 'LTS2CS',
        'add_properties': 'syn_dist_delay_feng_default'
    },
        ################### THALAMIC INPUT ###################
    {   # Thalamus Excitation to CP
        'network': 'cortex',
        'edge': {
            'source_network': 'thalamus',
            'source': {},
            'target': {'pop_name': ['CP']}
        },
        'param': 'Thal2CP'
    },
    {   # Thalamus Excitation to CS
        'network': 'cortex',
        'edge': {
            'source_network': 'thalamus',
            'source': {},
            'target': {'pop_name': ['CS']}
        },
        'param': 'Thal2CS'
    },
        ################### Baseline INPUT ###################
    {   # Excitation to CP
        'network': 'cortex',
        'edge': {
            'source_network': 'baseline',
            'source': {},
            'target': {'pop_name': ['CP']}
        },
        'param': 'Base2CP'
    },
    {   # Excitation to CS
        'network': 'cortex',
        'edge': {
            'source_network': 'baseline',
            'source': {},
            'target': {'pop_name': ['CS']}
        },
        'param': 'Base2CS'
    },
    {   # Excitation to FSI
        'network': 'cortex',
        'edge': {
            'source_network': 'baseline',
            'source': {},
            'target': {'pop_name': ['FSI']}
        },
        'param': 'Base2FSI'
    },
    {   # Excitation to LTS
        'network': 'cortex',
        'edge': {
            'source_network': 'baseline',
            'source': {},
            'target': {'pop_name': ['LTS']}
        },
        'param': 'Base2LTS'
    }
]

# edge_params should contain additional parameters to be added to add_edges().
# The following parameters for random synapse placement are not necessary in
# edge_params if afferent_section_id and afferent_section_pos are specified.
# distance_range: place synapse within distance range [dmin, dmax] from soma.
# target_sections: place synapse within the given sections in a list.
# afferent_section_id must be specified here even though it will be overwritten
# by add_properties(), since there could be potential error due to the dtype
# being forced to be converted to float if values are not specified in the
# corresponding column in the edge csv file.
edge_params = {
    'CP2CP': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': 0.267,
            'pr': 0.042,
            'estimate_rho': False,
            #estimates rho based off of old values from distance dependent model
            'rho': pr_2_rho(0.0866, 0.0866, 0.042),
            
            
            # 'p0': GaussianDropoff(
            #     stdev=127.0, min_dist=0., max_dist=max_conn_dist,
            #     ptotal=0.0866, ptotal_dist_range=(0., 100.),
            #     dist_type='cylindrical'),
            # 'p0_arg': cylindrical_dist_z,
            # 'pr': 0.042,
            # 'estimate_rho': True,
            # 'dist_range_forward': (0., 100.)
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 1,
        'afferent_section_pos': 0.4,
        'dynamics_params': 'CP2CP.json'
    },
    'CS2CS': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0': 0.23675,
            'pr': 0.015,
            'estimate_rho': False,
            #estimates rho based off old values from distance dependent model
            'rho': pr_2_rho(0.077, 0.077, 0.015),
            # 'p0': GaussianDropoff(
            #     stdev=127.0, min_dist=0., max_dist=max_conn_dist,
            #     ptotal=0.077, ptotal_dist_range=(0., 100.),
            #     dist_type='cylindrical'),
            # 'p0_arg': cylindrical_dist_z,
            # 'pr': 0.015,
            # 'estimate_rho': True,
            # 'dist_range_forward': (0., 100.)
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 1,
        'afferent_section_pos': 0.4,
        'dynamics_params': 'CS2CS.json'
    },
    'CP2CS': {
        'connector_class': UnidirectionConnector,
        'connector_params': {
            'p': 0.03225,
            # 'p': GaussianDropoff(
            #     stdev=127.0, min_dist=0., max_dist=max_conn_dist,
            #     ptotal=0.01, ptotal_dist_range=(0., 100.),
            #     dist_type='cylindrical'),
            #'p_arg': cylindrical_dist_z,
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 1,
        'afferent_section_pos': 0.4,
        'dynamics_params': 'CP2CS.json'
    },
    'CS2CP': {
        'connector_class': UnidirectionConnector,
        'connector_params': {
            'p': 0.343,
            # 'p': GaussianDropoff(
            #     stdev=127.0, min_dist=0., max_dist=max_conn_dist,
            #     ptotal=0.112, ptotal_dist_range=(0., 100.),
            #     dist_type='cylindrical'),
            #'p_arg': cylindrical_dist_z,
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 1,
        'afferent_section_pos': 0.4,
        'dynamics_params': 'CS2CP.json'
    },
    
 
    # 'FSI2FSI': {
    #     'connector_class': UnidirectionConnector,
    #     'connector_params': {
    #         'p': 0.8716666,
    #         'p_arg': spherical_dist,
    #         },
    #     'weight_function': 'lognormal_weight',
    #     'syn_weight': 1.,
    #     'weight_sigma': 0.8,
    #     'sigma_upper_bound': 3.,
    #     'afferent_section_id': 0,  # soma
    #     'afferent_section_pos': 0.5,
    #     'dynamics_params': 'FSI2FSI.json'
    # },
    
    #CHECK WHT THIS DOESNT WORK WITHOUT P0_ARG
    #ALSO CHECK why p0 needs to be halved here, Should it not just be 104.6/120???
    'FSI2FSI': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
             'p0': 0.8716666,
             #'p0': 0.8716666/2,
             'pr': 0.04,
             'estimate_rho': False,
            # #estimates rho based off old values from distance dependent model
             'rho': pr_2_rho(0.103, 0.103, 0.04),
            #'p0_arg': spherical_dist,
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 0,  # soma
        'afferent_section_pos': 0.5,
        'dynamics_params': 'FSI2FSI.json'
    },
    # ASK WHY TIHS NEEDS PARG
    'LTS2LTS': {
        'connector_class': UnidirectionConnector,
        'connector_params': {
            'p': 0.495,
            #'p_arg': spherical_dist,
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 0,  # soma
        'afferent_section_pos': 0.5,
        'dynamics_params': 'LTS2LTS.json'
    },
    'FSI2LTS': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0':0.54666,
            'p1':0.8525,
            'pr': 0.22,
            'estimate_rho': False,
            #estimates rho based off old values from distance dependent model
            'rho': pr_2_rho(0.34, 0.53, 0.22),
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 0,  # soma
        'afferent_section_pos': 0.5,
        'dynamics_params': 'FSI2LTS.json'
    },
    'LTS2FSI': {
        'connector_class': get_connector,
        'connector_params': {'param': 'FSI2LTS'},
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 0,  # soma
        'afferent_section_pos': 0.5,
        'dynamics_params': 'LTS2FSI.json'
    },
    'CP2FSI': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0':0.7005,
            'p1':0.5025,
            'pr': 0.26,
            'estimate_rho': False,
            #estimates rho based off old values from distance dependent model
            'rho': pr_2_rho(0.32, 0.48, 0.26),
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 1,  # dend
        'afferent_section_pos': 0.5,
        'dynamics_params': 'CP2FSI.json'
    },
    'FSI2CP': {
        'connector_class': get_connector,
        'connector_params': {'param': 'CP2FSI'},
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 0,  # soma
        'afferent_section_pos': 0.5,
        'dynamics_params': 'FSI2CP.json'
    },
    'CS2FSI': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0':0.483,
            'p1':0.384166666,
            'pr': 0.17,
            'estimate_rho': False,
            #estimates rho based off old values from distance dependent model
            'rho': pr_2_rho(0.34, 0.53, 0.22),
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 1,  # dend
        'afferent_section_pos': 0.5,
        'dynamics_params': 'CS2FSI.json'
    },
    'FSI2CS': {
        'connector_class': get_connector,
        'connector_params': {'param': 'CS2FSI'},
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 0,  # soma
        'afferent_section_pos': 0.5,
        'dynamics_params': 'FSI2CS.json'
    },
    'CP2LTS': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0':0.61725,
            'p1':0.41125,
            'pr': 0.16,
            'estimate_rho': False,
            #estimates rho based off old values from distance dependent model
            'rho': pr_2_rho(0.28, 0.39, 0.16),
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 1,  # dend
        'afferent_section_pos': 0.5,
        'dynamics_params': 'CP2LTS.json'
    },
    'LTS2CP': {
        'connector_class': get_connector,
        'connector_params': {'param': 'CP2LTS'},
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 2,
        'afferent_section_pos': 0.8,  # end of apic
        'dynamics_params': 'LTS2CP.json'
    },
    'CS2LTS': {
        'connector_class': ReciprocalConnector,
        'connector_params': {
            'p0':0.2845,
            'p1': 0.1025,
            'pr': 0.057,
            'estimate_rho': False,
            #estimates rho based off old values from distance dependent model
            'rho': pr_2_rho(0.13, 0.086, 0.057),
            },
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 1,  # dend
        'afferent_section_pos': 0.5,
        'dynamics_params': 'CS2LTS.json'
    },
    'LTS2CS': {
        'connector_class': get_connector,
        'connector_params': {'param': 'CS2LTS'},
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.8,
        'sigma_upper_bound': 3.,
        'afferent_section_id': 2,
        'afferent_section_pos': 0.8,  # end of apic
        'dynamics_params': 'LTS2CS.json'
    },
    'Thal2CP': {
        'connector_class': OneToOneSequentialConnector,
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.3,
        'sigma_upper_bound': 3.,
        'delay': 0.0,
        'afferent_section_id': 2,
        'afferent_section_pos': 0.8,  # end of apic
        'dynamics_params': 'Thal2CP.json'
    },
    'Thal2CS': {
        'connector_class': get_connector,
        'connector_params': {'param': 'Thal2CP'},
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.3,
        'sigma_upper_bound': 3.,
        'delay': 0.0,
        'afferent_section_id': 2,
        'afferent_section_pos': 0.8,  # end of apic
        'dynamics_params': 'Thal2CS.json'
    },
    'Base2CP': {
        'connector_class': OneToOneSequentialConnector,
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.3,
        'sigma_upper_bound': 3.,
        'delay': 0.0,
        'afferent_section_id': 2,
        'afferent_section_pos': 0.8,  # end of apic
        'dynamics_params': 'Thal2CP.json'
    },
    'Base2CS': {
        'connector_class': get_connector,
        'connector_params': {'param': 'Base2CP'},
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.3,
        'sigma_upper_bound': 3.,
        'delay': 0.0,
        'afferent_section_id': 2,
        'afferent_section_pos': 0.8,  # end of apic
        'dynamics_params': 'Thal2CS.json'
    },
    'Base2FSI': {
        'connector_class': get_connector,
        'connector_params': {'param': 'Base2CP'},
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.6,
        'sigma_upper_bound': 3.,
        'delay': 0.0,
        'afferent_section_id': 1,  # dend
        'afferent_section_pos': 0.5,
        'dynamics_params': 'Base2FSI.json'
    },
    'Base2LTS': {
        'connector_class': get_connector,
        'connector_params': {'param': 'Base2CP'},
        'weight_function': 'lognormal_weight',
        'syn_weight': 1.,
        'weight_sigma': 0.6,
        'sigma_upper_bound': 3.,
        'delay': 0.0,
        'afferent_section_id': 1,  # dend
        'afferent_section_pos': 0.5,
        'dynamics_params': 'Base2LTS.json'
    }
}  # edges referenced by name

# Will be called by conn.add_properties() for the associated connection
edge_add_properties = {
    'syn_dist_delay_feng_section_PN': {
        'names': ['delay', 'afferent_section_id', 'afferent_section_pos'],
        'rule': syn_dist_delay_feng_section_PN,
        'rule_params': {
            'p': 0.9, 'sec_id': (1, 2), 'sec_x': (0.4, 0.6), 'min_delay': 1.0
        },
        'dtypes': [np.float64, np.uint16, np.float64]
    },
    'syn_section_PN': {
        'names': ['afferent_section_id', 'afferent_section_pos'],
        'rule': syn_section_PN,
        'rule_params': {'p': 0.9, 'sec_id': (1, 2), 'sec_x': (0.4, 0.6)},
        'dtypes': [np.uint16, np.float64]
    },
    'syn_dist_delay_feng_default': {
        'names': 'delay',
        'rule': syn_dist_delay_feng,
        'dtypes': np.float64
    }
}


# Load synapse dictionaries
# See synapses.py - loads each json's in components/synaptic_models into a
# dictionary so the properties can be referenced in the files,
# e.g., syn['file.json'].get('property')
synapses.load(rng_obj=rng)
syn = synapses.syn_params_dicts()

# Build your edges into the networks
build_edges(networks, edge_definitions, edge_params, edge_add_properties, syn)


##############################################################################
############################  EDGE EFFECTS EDGES  ############################

if edge_effects:
    # These rules are for edge effect edges. They should mimic the connections
    # created previously but using unidirectional connector.
    # Re-use the connector params set above.

    # Find core network edge types that need shell connections
    core_network_name = 'cortex'
    core_edge_def = []
    for edge in edge_definitions:
        network_name = edge['network']
        if network_name != core_network_name:
            continue
        is_core = True
        edge_src_trg = edge.get('edge')
        if edge_src_trg:
            for net_type in ('source_network', 'target_network'):
                net_name = edge_src_trg.get(net_type)
                if net_name is not None and net_name != core_network_name:
                    is_core = False
        if is_core:
            core_edge_def.append(edge)

    # Automatically set up network edges and parameters for shell network
    # Only connections from shell to core is needed, so UnidirectionConnector
    # is used, and parameters are extracted from connectors used in core edges.
    shell_network_name = 'shell'
    shell_edges = []
    shell_edge_params = {}
    for edge in core_edge_def:
        shell_edge = edge.copy()
        edge_src_trg = shell_edge.get('edge')
        if edge_src_trg:
            edge_src_trg['source_network'] = shell_network_name
        shell_edge['param'] = shell_network_name + shell_edge['param']
        shell_edges.append(shell_edge)

        edge_params_val = edge_params[edge['param']].copy()
        connector = edge_params_val.pop('connector_object', None)
        connector_class = edge_params_val.get('connector_class')
        if (connector_class is not None and
                connector_class is not UnidirectionConnector):
            replace = True
            var_list = ('p', 'p_arg', 'n_syn')
            if connector_class is ReciprocalConnector:
                var_map = ('p0', 'p0_arg', 'n_syn0')
            elif connector_class is get_connector:
                var_map = ('p1', 'p1_arg', 'n_syn1')
            else:
                replace = False
                print("Warning: Connector method not identified. "
                      "Use the same connector class for shell edges.")
            if replace:
                edge_params_val['connector_class'] = UnidirectionConnector
                connector_params = {
                    k: connector.vars[k0] for k, k0 in zip(var_list, var_map)
                    }
                connector_params['verbose'] = connector.verbose
                edge_params_val['connector_params'] = connector_params
        shell_edge_params[shell_edge['param']] = edge_params_val
        # edge_params_val['delay'] = 0.0 # Set delay to 0
        # add_properties = shell_edge.pop('add_properties')
        # if add_properties == 'syn_dist_delay_feng_section_PN':
            # shell_edge['add_properties'] = 'syn_section_PN'

    # Check parameters
    print("\nShell edges:")
    for shell_edge in shell_edges:
        print(shell_edge)
    print("\nShell edge parameters:")
    for param, edge_params_val in shell_edge_params.items():
        print(param + ':')
        print(edge_params_val)
    print("")

    # Build your shell edges into the networks
    build_edges(networks, shell_edges, shell_edge_params,
                edge_add_properties, syn)

########################## END EDGE EFFECTS ##############################
##########################################################################


##########################################################################
############################ GAP JUNCTIONS ###############################
# Currently not working due to some errors in BMTK
# FSI
net = networks['cortex']
population = net.nodes(pop_name='FSI')

# gap junction probability correlated with chemical synapse
gap_junc_FSI = CorrelatedGapJunction(
    p_non=0.1228,
    p_uni=0.56, p_rec=1.,
    connector=edge_params['FSI2FSI']['connector_object']
)
gap_junc_FSI.setup_nodes(source=population, target=population)

g_gap = 0.000033 # microsiemens
conn = net.add_edges(
    is_gap_junction=True, syn_weight=g_gap, target_sections=None,
    afferent_section_id=0, afferent_section_pos=0.5,
    **gap_junc_FSI.edge_params()
)

# LTS
net = networks['cortex']
population = net.nodes(pop_name='LTS')

# gap junction probability uncorrelated with chemical synapse
LTS_uncorr_p = 0.0783
gap_junc_LTS = CorrelatedGapJunction(
    p_non=LTS_uncorr_p, p_uni=LTS_uncorr_p, p_rec=LTS_uncorr_p,
    connector=edge_params['LTS2LTS']['connector_object']
)
gap_junc_LTS.setup_nodes(source=population, target=population)

g_gap = 0.00038 # microsiemens
conn = net.add_edges(
    is_gap_junction=True, syn_weight=g_gap, target_sections=None,
    afferent_section_id=0, afferent_section_pos=0.5,
    **gap_junc_LTS.edge_params()
)

##########################################################################
###############################  BUILD  ##################################


# Save the network into the appropriate network dir
save_networks(networks, network_dir)

# Usually not necessary if you've already built your simulation config
if False:
    build_env_bionet(
        base_dir='./',
        network_dir=network_dir,
        tstop=t_sim,
        dt=dt,
        report_vars=['v'],
        celsius=31.0,
        spikes_inputs=[
            ('baseline', './input/baseline.h5'),
            ('shell', './input/shell.h5')
        ],  # (Population for which spikes will be generated, file name)
        components_dir='components',
        config_file='config.json',
        compile_mechanisms=False
    )
