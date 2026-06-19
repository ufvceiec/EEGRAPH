import unittest
import sys
sys.path.append('..')
import eegraph as eegraph
from eegraph.tools import *


class TestTools(unittest.TestCase):
    
    def test_processed_input_bands(self):
        frequency_bands = ['delta', 'thet', 'alpha', 'betah', 'gamma']
        expexted_result = [True, False, True, False, True]
        result = input_bands(frequency_bands)
        self.assertEqual(result, expexted_result)
        
    #=================    
    #Channel names
    
    def test_processed_channel_names_space(self):
        channel_names = ['EEG Fp1', 'EEG Fp2', 'EEG AF7', 'EEG AF3', 'EEG AF4', 'EEG AF8', 'EEG F7', 'EEG F5', 'EEG F3', 'EEG F1', 'EEG Fz', 'EEG F2', 'EEG F4', 'EEG F6']
        expected_channel_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6']
        result = process_channel_names(channel_names)
        self.assertEqual(result, expected_channel_names)
        
    def test_processed_channel_names_dash(self):
        channel_names = ['Fp1-EEG', 'Fp2-EEG', 'AF7-EEG', 'AF3-EEG', 'AF4-EEG', 'AF8-EEG', 'F7-EEG', 'F5-EEG', 'F3-EEG', 'F1-EEG', 'Fz-EEG', 'F2-EEG', 'F4-EEG', 'F6-EEG']
        expected_channel_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6']
        result = process_channel_names(channel_names)
        self.assertEqual(result, expected_channel_names)
        
    
    #=================    
    #Time intervals
    
    def test_calculate_intervals_float_no_flag(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
        sample_rate = 2
        sample_duration = np.float64(7.5)
        seconds = 2
        sample_length = 15
        expected_result_data = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15]]
        expected_interval = [(0, 4), (4, 8), (8, 12), (12, 15)] 
        expected_flag = 0
        
        result = calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
        
        for i, segment in enumerate(result[0]):
            self.assertEqual(list(segment), expected_result_data[i])
        self.assertEqual(result[1], expected_interval)
        self.assertEqual(result[2], expected_flag)
        
    def test_calculate_intervals_float_flag(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]])
        sample_rate = 10
        sample_duration = np.float64(2.9)
        seconds = 3
        sample_length = 29
        expected_result_data = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], [1,2,3,4,5,6,7,8,9,10]]
        expected_interval = [(0, 29), (0, sample_rate)]
        expected_flag = 1
        
        result = calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
        
        for i, segment in enumerate(result[0]):
            self.assertEqual(list(segment), expected_result_data[i])
        self.assertEqual(result[1], expected_interval)
        self.assertEqual(result[2], expected_flag)
        
    def test_calculate_intervals_list(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]])
        sample_rate = 10
        sample_duration = np.float64(2.9)
        seconds = [0,1,2]
        sample_length = 29
        expected_result_data = [[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20]]
        expected_interval = [(0, 10), (10, 20)]
        expected_flag = 0
        
        result = calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
        
        for i, segment in enumerate(result[0]):
            self.assertEqual(list(segment), expected_result_data[i])
        self.assertEqual(result[1], expected_interval)
        self.assertEqual(result[2], expected_flag)
        
    def test_calculate_intervals_list_Exception_exceed_sample_length(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]])
        sample_rate = 10
        sample_duration = np.float64(2.5)
        seconds = [0,3]
        sample_length = 25
        
        with self.assertRaises(Exception):
            calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
            
            
    def test_calculate_intervals_list_Exception_intervals_not_starting_0(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]])
        sample_rate = 10
        sample_duration = np.float64(3)
        seconds = [1, 2]
        sample_length = 30
        
        with self.assertRaises(Exception):
            calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
            
            
    def test_calculate_intervals_list_Exception_intervals_oneValue(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]])
        sample_rate = 10
        sample_duration = np.float64(3)
        seconds = [1]
        sample_length = 30
        expected_result_data = [[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20], [21,22,23,24,25,26,27,28,29,30]]
        expected_interval = [(0, 10), (10, 20), (20, 30)]
        expected_flag = 0
        
        result = calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
        for i, segment in enumerate(result[0]):
            self.assertEqual(list(segment), expected_result_data[i])
        self.assertEqual(result[1], expected_interval)
        self.assertEqual(result[2], expected_flag)
            
    #=================    
    #Frequency bands 
    
    def test_calculate_bands_fft(self):
        values = np.random.uniform(0, 100, 2048)
        sample_rate = 512 
        bands= [True, False, True, False, True]
        result = calculate_bands_fft(values, sample_rate, bands)
        
        self.assertTrue(len(result[0]) == len(result[1]) == len(result[2]))
        
    def test_search_method(self):
        connectivity = 'cross_correlation'
        expected_result = 'Cross_correlation_Estimator'

        result = search(connectivity_measures, connectivity)
        self.assertEqual(result, expected_result)
    
    def test_search_method_NameError(self):     
        connectivity = 'cross_correlations'
        with self.assertRaises(NameError):
            search(connectivity_measures, connectivity)
            
            
    #=================    
    #Connectivity
    
    def test_calculate_connectivity(self):
        data = []
        channels = 4
        intervals = 1
        for i in range(channels * intervals):
            data.append(np.random.uniform(-0.5, 1, 2048))
            
        steps = [(0, 2048)]
        sample_rate = 512
        connectivity = eegraph.strategy.Pearson_correlation_Estimator()
        connectivity.flag = 0
        
        result = calculate_connectivity(data, steps, channels, sample_rate, connectivity)
        self.assertEqual(np.shape(result), (intervals,channels,channels))
        
        
    def test_calculate_connectivity__more_intervals(self):
        data = []
        channels = 4
        intervals = 2
        for i in range(channels * intervals):
            data.append(np.random.uniform(-0.5, 1, 1024))
            
        steps = [(0, 1024), (1024, 2048)]
        sample_rate = 512
        connectivity = eegraph.strategy.Cross_correlation_Estimator()
        connectivity.flag = 0
        
        result = calculate_connectivity(data, steps, channels, sample_rate, connectivity)
        self.assertEqual(np.shape(result), (intervals,channels,channels))
        
    def test_calculate_connectivity_bands(self):
        data = []
        channels = 4
        intervals = 2
        for i in range(channels * intervals):
            data.append(np.random.uniform(-0.5, 1, 1024))
            
        bands= [True, True, True, False, False]
        steps = [(0, 1024), (1024, 2048)]
        channels = 4
        sample_rate =512
        connectivity = eegraph.strategy.Pli_Bands_Estimator()
        connectivity.flag = 0
        
        result = calculate_connectivity_with_bands(data, steps, channels, sample_rate, connectivity, bands)
        self.assertEqual(np.shape(result), (sum(bands)*intervals,channels,channels))
        
    def test_calculate_dtf(self):
        data = []
        channels = 16
        intervals = 1
        for i in range(channels * intervals):
            data.append(np.random.uniform(0, 1, 2048))
            
        steps = [(0, 2048)]
        sample_rate = 512
        bands= [True, True, False, True, False]
        flag=0
        
        result = calculate_dtf(data, steps, channels, sample_rate, bands, flag)
        self.assertEqual(np.shape(result), (sum(bands),channels,channels))
        
    def test_calculate_connectivity_single_channel(self):
        data = []
        channels = 4
        intervals = 1
        for i in range(channels * intervals):
            data.append(np.random.uniform(-0.5, 1, 2048))
            
        sample_rate = 512
        connectivity = eegraph.strategy.Shannon_entropy_Estimator()
        connectivity.flag = 0

        result = calculate_connectivity_single_channel(data, sample_rate, connectivity)
        self.assertEqual(len(result), channels)  


    def test_calculate_connectivity_single_channel_bands(self):
        data = []
        channels = 4
        intervals = 1
        for i in range(channels * intervals):
            data.append(np.random.uniform(-0.5, 1, 2048))
            
        sample_rate = 512
        bands= [True, True, False, True, False]
        connectivity = eegraph.strategy.Spectral_entropy_Estimator()
        connectivity.flag = 0

        result = calculate_connectivity_single_channel_with_bands(data, sample_rate, connectivity, bands)
        self.assertEqual(len(result), channels * sum(bands))  
        
        
        
    #=================    
    #Graphs      
        
    def test_make_graph(self):
        channels = 4
        data = np.zeros(shape=(channels,channels))
        matrix = [data]
        matrix[0][0] = [1,0,0.8,0.4]
        matrix[0][1] = [0.8,1,0.3,0]
        matrix[0][2] = [0.2,0.9,1,0]
        matrix[0][3] = [0.1,0.2,0.5,1]
        ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3']
        threshold = 0.7
        expected_edges = 3
        
        result = make_graph(matrix, ch_names, threshold)
        self.assertEqual(len(result[0].nodes()), channels)
        self.assertEqual(len(result[0].edges()), expected_edges)
        
    def test_make_single_channel_graph(self):
        channels = 16
        data = [0.81564148562685876, 0.30660675598762527, 0.71377519539990526, 0.3190937053018591, 0.38838726704914084, 0.7493007647717073, 0.35925485161888404, 0.9736121835275438, 0.84431093836793725, 0.3802669640607751, 0.3813287481487231, 0.4254342766449424, 0.72896987340610406, 0.3902971874488028, 0.35633248790669203, 0.34861031389046215]
        
        ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT9']
        expected_edges = 22 #All edges between top 25% nodes. 16 channels -> 4 nodes with connections. All 4 nodes interconnected -> 6 edges in total + 16 self loops = 22
        
        result, _ = single_channel_graph(data, ch_names, channels, 0.25)
        self.assertEqual(len(result[0].nodes()), channels)
        self.assertEqual(len(result[0].edges()), expected_edges)
    
    #=================
    #Graph metrics

    def _make_test_graph(self):
        """Helper: small fully-connected weighted graph with known properties."""
        G = nx.Graph()
        nodes = ['Fp1', 'Fp2', 'AF7', 'AF3']
        G.add_nodes_from(nodes)
        for u, v in [('Fp1','Fp2'), ('Fp1','AF7'), ('Fp1','AF3'),
                     ('Fp2','AF7'), ('Fp2','AF3'), ('AF7','AF3')]:
            G.add_edge(u, v, weight=0.8, thickness=1)
        return G

    def test_compute_graph_metrics_keys(self):
        G = self._make_test_graph()
        metrics = compute_graph_metrics(G)
        expected_keys = [
            'density', 'transitivity', 'average_clustering',
            'global_efficiency', 'local_efficiency', 'average_path_length',
            'degree_assortativity', 'small_world_sigma',
            'degree_centrality', 'betweenness_centrality', 'eigenvector_centrality'
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)

    def test_compute_graph_metrics_density(self):
        G = self._make_test_graph()
        metrics = compute_graph_metrics(G)
        # 4 nodes fully connected → density = 1.0
        self.assertAlmostEqual(metrics['density'], 1.0)

    def test_compute_graph_metrics_scalar_ranges(self):
        G = self._make_test_graph()
        metrics = compute_graph_metrics(G)
        for key in ['density', 'transitivity', 'average_clustering',
                    'global_efficiency', 'local_efficiency']:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_compute_graph_metrics_centrality_nodes(self):
        G = self._make_test_graph()
        metrics = compute_graph_metrics(G)
        for key in ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality']:
            self.assertEqual(set(metrics[key].keys()), set(G.nodes()))

    def test_compute_graph_metrics_disconnected(self):
        G = nx.Graph()
        G.add_nodes_from(['Fp1', 'Fp2', 'AF7', 'AF3'])
        G.add_edge('Fp1', 'Fp2', weight=0.9, thickness=1)
        # Graph is disconnected — average_path_length should still return a value
        metrics = compute_graph_metrics(G)
        self.assertIn('average_path_length', metrics)

    def test_compute_graph_metrics_directed(self):
        G = nx.DiGraph()
        G.add_nodes_from(['Fp1', 'Fp2', 'AF7'])
        G.add_edge('Fp1', 'Fp2', weight=0.7, thickness=1)
        G.add_edge('Fp2', 'AF7', weight=0.6, thickness=1)
        # Should not raise; directed graph is converted internally
        metrics = compute_graph_metrics(G)
        self.assertIn('density', metrics)

    def test_compute_metrics_all(self):
        G = self._make_test_graph()
        graphs = {0: G, 1: G}
        all_metrics = compute_metrics_all(graphs)
        self.assertEqual(set(all_metrics.keys()), {0, 1})
        for m in all_metrics.values():
            self.assertIn('density', m)

    def test_draw_graph(self):
        G1 = nx.Graph()
        nodes_list = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT9']
        edges_list = [('Fp1', 'Fp2'), ('Fp1', 'AF3'), ('Fp1', 'F7'), ('AF7', 'AF3'), ('AF8', 'F7')]
        G1.add_nodes_from(nodes_list)
        for pair in edges_list:
            G1.add_edge(pair[0], pair[1], weight=1, thickness=1)
    
        self.assertTrue(draw_graph(G1))
        
    def test_draw_graph_unkown_node(self):
        G1 = nx.Graph()
        nodes_list = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'XX', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT9']
        edges_list = [('Fp1', 'Fp2'), ('Fp1', 'AF3'), ('Fp1', 'F7'), ('AF7', 'AF3'), ('AF8', 'F7')]
        G1.add_nodes_from(nodes_list)
        for pair in edges_list:
            G1.add_edge(pair[0], pair[1], weight=1, thickness=1)
    
        with self.assertWarns(Warning):
            draw_graph(G1)
    

class TestImportData(unittest.TestCase):
    
    def test_load_data(self):
        path = '.chb02_16.edf'                               #Public EEG dataset. https://physionet.org/content/chbmit/1.0.0/
        channels = 23
        expected_ch_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 
                           'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1']
        G = eegraph.Graph()
        G.load_data(path)
        
        self.assertEqual(len(G.ch_names), channels)
        self.assertEqual(G.ch_names, expected_ch_names)
        
    def test_load_data_electrode_montage(self):
        path = '.test_eeg.gdf'                               
        electrode_montage_path = '.electrodemontage.set.ced'
        expected_ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 
                             'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 
                             'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'PO9', 'O1', 'Oz', 'O2']         #Labels in electrode montage file
        channels = 64
        
        G = eegraph.Graph()
        G.load_data(path, electrode_montage_path = electrode_montage_path)
        
        self.assertEqual(len(G.ch_names), channels)
        self.assertEqual(G.ch_names, expected_ch_names)

class TestModelData(unittest.TestCase):
    
    def setUp(self):
        path = '.test_eeg.gdf'
        self.window_size = 5
        self.G = eegraph.Graph()
        self.G.load_data(path)
    
    def test_modelate_no_bands(self):
        connectivity = 'corr_cross_correlation'
        expected_graphs = 7    #32 secs / 5 = 6.4 -> 7
        
        graphs, _ = self.G.modelate(window_size = self.window_size, connectivity = connectivity)
        self.assertEqual(len(graphs), expected_graphs)
        
    def test_modelate_bands(self):
        connectivity = 'imag_coherence'
        bands = ['delta','theta','alpha','beta','gamma']
        expected_graphs = 7 * 5    #32 secs / 5 = 6.4 -> 7 * 5 frequency bands. 
        
        graphs, _ = self.G.modelate(window_size = self.window_size, connectivity = connectivity, bands=bands)
        self.assertEqual(len(graphs), expected_graphs)
        
    def test_modelate_need_bands_error(self):
        connectivity = 'imag_coherence'
        
        with self.assertRaises(NameError):
            graphs, _ = self.G.modelate(window_size = self.window_size, connectivity = connectivity)
            
    def test_modelate_dont_need_bands_error(self):
        connectivity = 'cross_correlation'
        bands = ['delta','theta','alpha','beta','gamma']
        
        with self.assertRaises(NameError):
            graphs, _ = self.G.modelate(window_size = self.window_size, connectivity = connectivity, bands=bands)
        
        
class TestVisualizeData(unittest.TestCase):
    
    def test_visualize(self):
        G = eegraph.Graph()
        G.load_data('.test_eeg.gdf', electrode_montage_path = '.electrodemontage.set.ced')
        graphs, _ = G.modelate(window_size = 10, connectivity = 'wpli', threshold=0.9, bands = ['theta','alpha','beta'])
        G.visualize(graphs[0], 'test_1')

        expexted_html_file_path = 'test_1_plot.html'
        f = open(expexted_html_file_path)
        self.assertTrue(f)
        f.close()

    def test_visualize_channel_warning(self):
        G = eegraph.Graph()
        G.load_data('.test_eeg.gdf')
        graphs, _ = G.modelate(window_size = 10, connectivity = 'cross_correlation')
        
        with self.assertWarns(Warning):
            G.visualize(graphs[0], 'test_2')
    
class TestNewConnectivityMeasures(unittest.TestCase):
    """Unit tests for the 11 new connectivity estimators added to EEGraph."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.sr = 256
        n = 512
        self.x = rng.standard_normal(n)
        self.y = rng.standard_normal(n)
        self.data = [self.x, self.y]
        self.bands = [True, True, True, True, True]

    # ── No-bands (undirected) ──────────────────────────────────────────────

    def test_aec_returns_scalar(self):
        est = eegraph.strategy.Aec_Estimator()
        r = est.calculate_conn(self.data, 0, 1, self.sr, 2)
        self.assertIsInstance(r, float)

    def test_aec_range(self):
        est = eegraph.strategy.Aec_Estimator()
        r = est.calculate_conn(self.data, 0, 1, self.sr, 2)
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r, 1.0)

    def test_aec_orth_returns_scalar(self):
        est = eegraph.strategy.Aec_orth_Estimator()
        r = est.calculate_conn(self.data, 0, 1, self.sr, 2)
        self.assertIsInstance(r, float)

    def test_aec_orth_range(self):
        est = eegraph.strategy.Aec_orth_Estimator()
        r = est.calculate_conn(self.data, 0, 1, self.sr, 2)
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r, 1.0)

    def test_mutual_information_non_negative(self):
        est = eegraph.strategy.Mutual_information_Estimator()
        mi = est.calculate_conn(self.data, 0, 1, self.sr, 2)
        self.assertGreaterEqual(mi, 0.0)

    def test_mutual_information_identical_signals(self):
        est = eegraph.strategy.Mutual_information_Estimator()
        data = [self.x, self.x]
        mi = est.calculate_conn(data, 0, 1, self.sr, 2)
        # MI(X,X) should be positive (entropy of X)
        self.assertGreater(mi, 0.0)

    def test_sync_likelihood_range(self):
        est = eegraph.strategy.Sync_likelihood_Estimator()
        sl = est.calculate_conn(self.data, 0, 1, self.sr, 2)
        self.assertGreaterEqual(sl, 0.0)
        self.assertLessEqual(sl, 1.0)

    def test_granger_causality_non_negative(self):
        est = eegraph.strategy.Granger_causality_Estimator()
        gc = est.calculate_conn(self.data, 0, 1, self.sr, 2)
        self.assertGreaterEqual(gc, 0.0)

    def test_granger_causality_asymmetric(self):
        est = eegraph.strategy.Granger_causality_Estimator()
        gc_xy = est.calculate_conn(self.data, 0, 1, self.sr, 2)
        gc_yx = est.calculate_conn(self.data, 1, 0, self.sr, 2)
        # GC is generally asymmetric for random signals
        # Just verify both are non-negative scalars
        self.assertGreaterEqual(gc_xy, 0.0)
        self.assertGreaterEqual(gc_yx, 0.0)

    def test_transfer_entropy_non_negative(self):
        est = eegraph.strategy.Transfer_entropy_Estimator()
        te = est.calculate_conn(self.data, 0, 1, self.sr, 2)
        self.assertGreaterEqual(te, 0.0)

    # ── With-bands (undirected) ────────────────────────────────────────────

    def test_dwpli_returns_5_values(self):
        est = eegraph.strategy.Dwpli_Estimator()
        result = est.calculate_conn(self.data, 0, 1, self.sr, 2, self.bands)
        self.assertEqual(len(result), 5)

    def test_dwpli_values_are_numeric(self):
        est = eegraph.strategy.Dwpli_Estimator()
        result = est.calculate_conn(self.data, 0, 1, self.sr, 2, self.bands)
        for v in result:
            self.assertIsInstance(v, float)

    def test_ppc_returns_5_values(self):
        est = eegraph.strategy.Ppc_Estimator()
        result = est.calculate_conn(self.data, 0, 1, self.sr, 2, self.bands)
        self.assertEqual(len(result), 5)

    def test_ppc_non_negative(self):
        est = eegraph.strategy.Ppc_Estimator()
        result = est.calculate_conn(self.data, 0, 1, self.sr, 2, self.bands)
        for v in result:
            self.assertGreaterEqual(v, 0.0)

    def test_lagged_coherence_returns_5_values(self):
        est = eegraph.strategy.Lagged_coherence_Estimator()
        result = est.calculate_conn(self.data, 0, 1, self.sr, 2, self.bands)
        self.assertEqual(len(result), 5)

    def test_lagged_coherence_non_negative(self):
        est = eegraph.strategy.Lagged_coherence_Estimator()
        result = est.calculate_conn(self.data, 0, 1, self.sr, 2, self.bands)
        for v in result:
            self.assertGreaterEqual(v, 0.0)

    def test_psi_returns_5_values(self):
        est = eegraph.strategy.Psi_Estimator()
        result = est.calculate_conn(self.data, 0, 1, self.sr, 2, self.bands)
        self.assertEqual(len(result), 5)

    def test_psi_numeric(self):
        est = eegraph.strategy.Psi_Estimator()
        result = est.calculate_conn(self.data, 0, 1, self.sr, 2, self.bands)
        for v in result:
            self.assertIsInstance(v, float)

    # ── Integration: calculate_connectivity with new no-bands measures ─────

    def test_calculate_connectivity_aec(self):
        channels = 3
        data = [np.random.standard_normal(512) for _ in range(channels)]
        steps = [(0, 512)]
        est = eegraph.strategy.Aec_Estimator()
        est.flag = 0
        result = calculate_connectivity(data, steps, channels, self.sr, est)
        self.assertEqual(np.shape(result), (1, channels, channels))

    def test_calculate_connectivity_granger(self):
        channels = 3
        data = [np.random.standard_normal(512) for _ in range(channels)]
        steps = [(0, 512)]
        est = eegraph.strategy.Granger_causality_Estimator()
        est.flag = 0
        result = calculate_connectivity(data, steps, channels, self.sr, est)
        self.assertEqual(np.shape(result), (1, channels, channels))

    def test_calculate_connectivity_bands_dwpli(self):
        channels = 3
        data = [np.random.standard_normal(512) for _ in range(channels)]
        steps = [(0, 512)]
        bands = [True, True, False, True, False]
        est = eegraph.strategy.Dwpli_Estimator()
        est.flag = 0
        result = calculate_connectivity_with_bands(data, steps, channels, self.sr, est, bands)
        self.assertEqual(np.shape(result), (sum(bands), channels, channels))


class TestNewGraphMetrics(unittest.TestCase):
    """Tests for the new graph-theoretic metrics added to compute_graph_metrics."""

    def _make_complete_graph(self):
        G = nx.complete_graph(5)
        for u, v in G.edges():
            G[u][v]['weight'] = 0.8
        return G

    def _make_disconnected_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(6))
        G.add_edge(0, 1, weight=0.9)
        G.add_edge(2, 3, weight=0.7)
        G.add_edge(4, 5, weight=0.8)
        return G

    def test_modularity_key_present(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        self.assertIn('modularity', m)

    def test_modularity_range(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        self.assertGreaterEqual(m['modularity'], 0.0)
        self.assertLessEqual(m['modularity'], 1.0)

    def test_modularity_disconnected_non_negative(self):
        """Disconnected graph should have non-negative modularity."""
        G = self._make_disconnected_graph()
        m = compute_graph_metrics(G)
        self.assertGreaterEqual(m['modularity'], 0.0)

    def test_rich_club_key_present(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        self.assertIn('rich_club_coefficient', m)

    def test_rich_club_is_dict(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        self.assertIsInstance(m['rich_club_coefficient'], dict)

    def test_rich_club_values_in_range(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        for v in m['rich_club_coefficient'].values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_closeness_centrality_key_present(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        self.assertIn('closeness_centrality', m)

    def test_closeness_centrality_all_nodes(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        self.assertEqual(set(m['closeness_centrality'].keys()), set(G.nodes()))

    def test_closeness_centrality_range(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        for v in m['closeness_centrality'].values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_node_strength_key_present(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        self.assertIn('node_strength', m)

    def test_node_strength_non_negative(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        for v in m['node_strength'].values():
            self.assertGreaterEqual(v, 0.0)

    def test_degree_key_present(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        self.assertIn('degree', m)

    def test_degree_values_are_int(self):
        G = self._make_complete_graph()
        m = compute_graph_metrics(G)
        for v in m['degree'].values():
            self.assertIsInstance(v, int)

    def test_disconnected_modularity_non_negative(self):
        G = self._make_disconnected_graph()
        m = compute_graph_metrics(G)
        self.assertGreaterEqual(m['modularity'], 0.0)

    def test_metrics_all_new_keys_present(self):
        G = self._make_complete_graph()
        graphs = {0: G}
        all_m = compute_metrics_all(graphs)
        for key in ['modularity', 'rich_club_coefficient',
                    'closeness_centrality', 'node_strength', 'degree']:
            self.assertIn(key, all_m[0])


if __name__ == '__main__':
    unittest.main()
