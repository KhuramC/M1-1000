from bmtool import bmplot
from bmtool.util import util
import matplotlib.pyplot as plt

# bmplot.plot_inspikes('config.json')
# bmplot.plot_basic_cell_info('config.json')
bmplot.plot_3d_positions(config='config.json',populations='cortex',group_by='pop_name',title='Cell Positions',save_file=False)
#bmplot.plot_3d_positions(config='config.json',populations='shell',group_by='pop_name',title='Cell Positions',save_file=False)
plt.show()