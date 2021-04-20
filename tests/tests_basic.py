import unittest
import sys
sys.path.append('..')
import eegraph as eegraph
from eegraph.tools import *

class TestTools(unittest.TestCase):
    connectivity_measures = {'cross_correlation': 'Cross_correlation_Estimator', 'pearson_correlation': 'Pearson_correlation_Estimator', 'squared_coherence': 'Squared_coherence_Estimator',
                         'imag_coherence': 'Imag_coherence_Estimator', 'corr_cross_correlation': 'Corr_cross_correlation_Estimator', 'wpli': 'Wpli_Estimator', 
                         'plv': 'Plv_Estimator', 'pli': 'Pli_No_Bands_Estimator', 'pli_bands': 'Pli_Bands_Estimator', 'dtf': 'Dtf_Estimator', 'power_spectrum': 'Power_spectrum_Estimator',
                         'spectral_entropy': 'Spectral_entropy_Estimator', 'shannon_entropy': 'Shannon_entropy_Estimator'}
    
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
        sample_rate = 256 
        bands= [True, False, True, False, True]
        result = calculate_bands_fft(values, sample_rate, bands)
        
        self.assertTrue(len(result[0]) < len(result[1] < len(result[2])))
        
    def test_search_method(self):
        connectivity = 'cross_correlation'
        expected_result = 'Cross_correlation_Estimator()'
        
        result = search(connectivity_measures, connectivity)
        self.assertEqual(result, expected_result)
    
    def test_search_method_NameError(self):     
        connectivity = 'cross_correlations'
        with self.assertRaises(NameError):
            search(connectivity_measures, connectivity)
            
            
    #=================    
    #Connectivity
    
    def test_calculate_connectivity(self):
        data = [np.random.uniform(0, 100, 5),np.random.uniform(0, 100, 5),np.random.uniform(0, 100, 5),np.random.uniform(0, 100, 5)]
        steps = [(0, 5)]
        channels = 4
        sample_rate = 1
        connectivity = eegraph.strategy.Pearson_correlation_Estimator()
        connectivity.flag = 0
        
        result = calculate_connectivity(data, steps, channels, sample_rate, connectivity)
        self.assertEqual(np.shape(result), (1,4,4))
        
        
    def test_calculate_connectivity__more_intervals(self):
        data = [np.random.uniform(0, 100, 4),np.random.uniform(0, 100, 4),np.random.uniform(0, 100, 4),np.random.uniform(0, 100, 4),np.random.uniform(0, 100, 6), np.random.uniform(0, 100, 6), 
               np.random.uniform(0, 100, 6), np.random.uniform(0, 100, 6)]
        steps = [(0, 4), (4,10)]
        channels = 4
        sample_rate = 2
        connectivity = eegraph.strategy.Pearson_correlation_Estimator()
        connectivity.flag = 0
        
        result = calculate_connectivity(data, steps, channels, sample_rate, connectivity)
        self.assertEqual(np.shape(result), (2,4,4))
        
    def test_calculate_connectivity_bands(self):
        data = [np.random.uniform(0, 100, 10),np.random.uniform(0, 100, 10),np.random.uniform(0, 100, 10),np.random.uniform(0, 100, 10), np.random.uniform(0, 100, 10),np.random.uniform(0, 100, 10),
                np.random.uniform(0, 100, 10),np.random.uniform(0, 100, 10), np.random.uniform(0, 100, 10),np.random.uniform(0, 100, 10),np.random.uniform(0, 100, 10),np.random.uniform(0, 100, 10)]
        bands= [True, True, True, False, False]
        steps = [(0, 10)]
        channels = 4
        sample_rate = 1
        connectivity = eegraph.strategy.Squared_coherence_Estimator()
        connectivity.flag = 0
        
        result = calculate_connectivity_with_bands(data, steps, channels, sample_rate, connectivity, bands)
        self.assertEqual(np.shape(result), (3,4,4))
        
    def test_calculate_connectivity_single_channel(self):
        data = [np.random.uniform(0, 100, 5),np.random.uniform(0, 100, 5),np.random.uniform(0, 100, 5),np.random.uniform(0, 100, 5)]
        steps = [(0, 5)]
        channels = 4
        sample_rate = 1
        connectivity = eegraph.strategy.Shannon_entropy_Estimator()
        connectivity.flag = 0
        
        result = calculate_connectivity_single_channel(data, sample_rate, connectivity)
        self.assertEqual(len(result), channels)  

        
        
if __name__ == '__main__':
    unittest.main()
