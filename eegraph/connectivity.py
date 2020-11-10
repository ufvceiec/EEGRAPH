from .tools import *


def cross_correlation(path, window_size, exclude = [None], threshold = 0.4):
    conn = 'cc'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    scaled_data = re_scaling(raw_data)
    
    data_intervals, steps = time_intervals(scaled_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity(data_intervals, steps, num_channels, sample_rate, conn)
        
    make_graph(connectivity_matrix, ch_names,  threshold)
    
    
def pearson_correlation(path, window_size, exclude = [None], threshold = 0.7):
    conn = 'pearson'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity(data_intervals, steps, num_channels, sample_rate, conn)

    make_graph(connectivity_matrix, ch_names,  threshold)
    
    
def squared_coherence(path, window_size, bands, exclude = [None], threshold = 0.7):
    conn = 'coh'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    wanted_bands = input_bands(bands)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity_with_bands(data_intervals, steps, num_channels, sample_rate, conn, wanted_bands)

    make_graph(connectivity_matrix, ch_names,  threshold)
    

def imag_coherence(path, window_size, bands, exclude = [None], threshold = 0.4):
    conn = 'icoh'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    wanted_bands = input_bands(bands)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity_with_bands(data_intervals, steps, num_channels, sample_rate, conn, wanted_bands)

    make_graph(connectivity_matrix, ch_names,  threshold)
    
    
