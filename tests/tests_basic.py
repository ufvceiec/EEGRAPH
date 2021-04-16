import unittest
from eegraph.graph import Graph
from eegraph.importData import *
from eegraph.modelateData import *
from eegraph.strategy import *
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
        
    def test_calculate_intervals_no_flag(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
        sample_rate = 2
        sample_duration = 7.5
        seconds = 2
        sample_length = 15
        expected_result_data = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15]]
        expected_interval = [(0, 4), (4, 8), (8, 12), (12, 15)] 
        expected_flag = 0
        
        result = calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
        
        for i, segment in enumerate(result[0]):
            print('seg1', segment)
            self.assertEqual(segment, expected_result_data[i])
        self.assertEqual(result[1], expected_interval)
        self.assertEqual(result[2], expected_flag)
        
    def test_calculate_intervals_flag(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]])
        sample_rate = 10
        sample_duration = 2.9
        seconds = 3
        sample_length = 29
        expected_result_data = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], [1,2,3,4,5,6,7,8,9,10]]
        expected_interval = [(0, 29), (0, sample_rate)]
        expected_flag = 1
        
        result = calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
        
        for i, segment in enumerate(result[0]):
            self.assertEqual(segment, expected_result_data[i])
        self.assertEqual(result[1], expected_interval)
        self.assertEqual(result[2], expected_flag)
        
    def test_calculate_intervals_list_Exception_exceed_sample_length(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]])
        sample_rate = 10
        sample_duration = 2.5
        seconds = [0,3]
        sample_length = 29
        
        with self.assertRaises(Exception):
            calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
            
            
    def test_calculate_intervals_list_Exception_intervals_not_starting_0(self):
        data = ([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]])
        sample_rate = 10
        sample_duration = 3
        seconds = [1, 2]
        sample_length = 29
        
        with self.assertRaises(Exception):
            calculate_time_intervals(data, sample_rate, sample_duration, seconds, sample_length)
        
    def test_calculate_bands_fft(self):
        values = np.random.uniform(0, 100, 2048)
        sample_rate = 512 
        bands= [True, False, True, False, True]
        result = calculate_bands_fft(values, sample_rate, bands)
        
        self.assertTrue((result[0].mean() >= result[1].mean()) and (result[0].mean() >=  result[2].mean()))
        
    def test_search_method(self):
        connectivity = 'cross_correlation'
        expected_result = 'Cross_correlation_Estimator()'
        
        result = search(connectivity_measures, connectivity)
        self.assertEqual(result, expected_result)
    
    def test_search_method_NameError(self):     
        connectivity = 'cross_correlations'
        with self.assertRaises(NameError):
            search(connectivity_measures, connectivity)
        
    
        
        
if __name__ == '__main__':
    unittest.main()
