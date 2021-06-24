from .strategy import *

#Class that uses the Strategy Abstract class
class ModelData: 
    def __init__(self, data, ch_names, strategy: Strategy):
        self.raw_data = data.get_data()
        self.ch_names = ch_names
        self.num_channels = data.info['nchan']
        self.sample_rate = data.info['sfreq']
        self.sample_duration = data.times.max()
        self.sample_length = self.sample_rate * self.sample_duration
        self._strategy = strategy
        self.threshold = self._strategy.threshold
        
    def connectivity_workflow(self, bands, window_size, threshold):
        #If the user assigns a new threshold
        if(threshold):
            self.threshold = threshold
            
        self.connectivity_matrix = self._strategy.calculate_connectivity_workflow(self, bands, window_size)
        print('\nThreshold:', self.threshold)
        
        out = self._strategy.make_graph_workflow(self)
        if(type(out) is tuple):
            self.connectivity_graphs = out[0]
            self.connectivity_matrix = out[1]
        else:
            self.connectivity_graphs = out

        return self.connectivity_graphs, self.connectivity_matrix
        
    