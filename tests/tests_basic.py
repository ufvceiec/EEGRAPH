import unittest
from eegraph.graph import *
from eegraph.importData import *
from eegraph.modelateData import *
from eegraph.strategy import *
from eegraph.tools import *

class TestTools(unittest.TestCase):
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
        
    def test_calculate_bands_fft(self):
        values = np.random.uniform(0, 100, 8192)
        sample_rate = 512 
        bands= [True, False, True, False, True]
        result = calculate_bands_fft(values, sample_rate, bands)
        
        self.assertTrue(result[0].mean() > result[1].mean() > result[2].mean())
    
if __name__ == '__main__':
    unittest.main()
