import sys
import os,pathlib,json
import warnings
import synapses
from bmtk.simulator import bionet
from bmtk.simulator.bionet.pyfunction_cache import add_weight_function
from neuron import h
pc = h.ParallelContext()

CONFIG = 'config.json'
USE_CORENEURON = False


def run(config_file=CONFIG, use_coreneuron=USE_CORENEURON):

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # register synaptic weight function
    synapses.load(randseed=1111)
    add_weight_function(synapses.lognormal_weight, name='lognormal_weight')
        
    
    with open(config_file, 'r') as json_file:
        conf_dict = json.load(json_file)
        if os.environ.get("OUTPUT_DIR"):
            output_dir = os.path.abspath(os.environ.get('OUTPUT_DIR'))
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            print(f"Output directory updated to {output_dir}")
            conf_dict['manifest']['$OUTPUT_DIR'] = output_dir
        if use_coreneuron:
            import corebmtk
            conf = corebmtk.Config.from_json(conf_dict, validate=True)
        else:
            conf = bionet.Config.from_json(conf_dict, validate=True)
    pc.barrier()

    conf.build_env()
    graph = bionet.BioNetwork.from_config(conf)

    if use_coreneuron:
        sim = corebmtk.CoreBioSimulator.from_config(
            conf, network=graph, gpu=False)
    else:
        sim = bionet.BioSimulator.from_config(conf, network=graph)

    pc.barrier()
    
    cells = graph.get_local_cells()
    for cell in cells:
        cells[cell].hobj.insert_mechs(cells[cell].gid)
        pass
    
    
    sim.run()

    bionet.nrn.quit_execution()


if __name__ == '__main__':
    run(sys.argv[-1])
