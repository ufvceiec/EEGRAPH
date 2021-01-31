from .importData import *
from .modelateData import *
from .tools import *

class Graph:
    
    def __init__(self):
        pass
    
        
    def load_data(self, path, exclude = [None]):
        input_data = InputData(path, exclude)
        self.data = input_data.load()
        input_data.display_info()
        

    def modelate(self, window_size, connectivity, bands = [None], threshold = None):
        
        print('\033[1m' + 'Model Data.' + '\033[0m')
        print(search(connectivity_measures, connectivity))
        
        model_data = ModelData(self.data, eval(search(connectivity_measures, connectivity)))  
        connectivity_matrix, G = model_data.connectivity_workflow(bands, window_size, threshold)
        
        return connectivity_matrix, G
        

    def visualize(self, graph):
        fig = draw_graph(graph, False, False)
        fig.update_layout(title='', plot_bgcolor='white' ) 
        fig.write_html('graph_plot.html', auto_open=True, default_height='100%', default_width='100%')