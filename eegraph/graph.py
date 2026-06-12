from .importData import *
from .modelateData import *
from .tools import *

class Graph:
    
    def __init__(self):
        pass
        
    def load_data(self, path, exclude = [None],  electrode_montage_path = None):
        input_data = InputData(path, exclude)
        self.data = input_data.load()
        
        self.ch_names=self.data.ch_names
        if(electrode_montage_path):
            self.ch_names=input_data.set_montage(electrode_montage_path)
        
        input_data.display_info(self.ch_names)
        

    def modelate(self, window_size, connectivity, bands = [None], threshold = None):
        print('\033[1m' + 'Model Data.' + '\033[0m')
        cls_name = search(connectivity_measures, connectivity)
        print(cls_name)

        model_data = ModelData(self.data, self.ch_names, globals()[cls_name]())  
        connectivity_matrix, G = model_data.connectivity_workflow(bands, window_size, threshold)
        
        return connectivity_matrix, G
        

    def compute_metrics(self, graphs):
        """Compute graph-theoretic metrics for one or all graphs.

        Parameters
        ----------
        graphs : dict or NetworkX Graph/DiGraph
            Either the full dictionary returned by ``modelate()`` or a single
            NetworkX graph (e.g. ``graphs[0]``).

        Returns
        -------
        metrics : dict
            If a dict of graphs is passed, returns ``{graph_index: metrics_dict}``.
            If a single graph is passed, returns its ``metrics_dict`` directly.

        Examples
        --------
        >>> graphs, matrix = G.modelate(window_size=5, connectivity='pearson_correlation')
        >>> metrics = G.compute_metrics(graphs)          # all windows
        >>> metrics = G.compute_metrics(graphs[0])       # single window
        """
        if isinstance(graphs, dict):
            return compute_metrics_all(graphs)
        else:
            return compute_graph_metrics(graphs)

    def visualize_html(self, graph, name, auto_open = True):
        fig = draw_graph(graph)
        fig.update_layout(title='', plot_bgcolor='white' ) 
        fig.write_html(str(name) + '_plot.html', auto_open=auto_open, default_height='100%', default_width='100%')
        
        
    def visualize_png(self, graph, name):
        fig = draw_graph(graph)
        fig.update_layout(title='', plot_bgcolor='white' ) 
        fig.write_image(str(name) + '.png', format='png',height=1000,width=1800)
