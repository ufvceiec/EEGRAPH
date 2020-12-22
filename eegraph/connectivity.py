from .tools import *


def cross_correlation(path, window_size, exclude = [None], threshold = 0.50):
    conn = 'cc'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    scaled_data = re_scaling(raw_data)
    
    data_intervals, steps = time_intervals(scaled_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity(data_intervals, steps, num_channels, sample_rate, conn)
        
    make_graph(connectivity_matrix, ch_names,  threshold)
    
    
def pearson_correlation(path, window_size, exclude = [None], threshold = 0.70):
    conn = 'pearson'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity(data_intervals, steps, num_channels, sample_rate, conn)

    make_graph(connectivity_matrix, ch_names,  threshold)
    
    
def squared_coherence(path, window_size, bands, exclude = [None], threshold = 0.65):
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
    
    
    
def corr_cross_correlation(path, window_size, exclude = [None], threshold = 0.3):
    conn = 'corcc'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    scaled_data = re_scaling(raw_data)
    
    data_intervals, steps = time_intervals(scaled_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity(data_intervals, steps, num_channels, sample_rate, conn)
        
    make_graph(connectivity_matrix, ch_names,  threshold)
    
    
def wpli(path, window_size, bands, exclude = [None], threshold = 0.45):
    conn = 'wpli'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    wanted_bands = input_bands(bands)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity_with_bands(data_intervals, steps, num_channels, sample_rate, conn, wanted_bands)

    make_graph(connectivity_matrix, ch_names,  threshold)
    
    
def plv(path, window_size, bands, exclude = [None], threshold = 0.83):
    conn = 'plv'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    wanted_bands = input_bands(bands)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity_with_bands(data_intervals, steps, num_channels, sample_rate, conn, wanted_bands)
        
    make_graph(connectivity_matrix, ch_names,  threshold)
    
    
    
def pli(path, window_size, bands, exclude = [None], threshold = 0.1):
    conn = 'pli'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    wanted_bands = input_bands(bands)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    if sum(wanted_bands):
        print("\nPLI with bands.")
        connectivity_matrix = calculate_connectivity_with_bands(data_intervals, steps, num_channels, sample_rate, conn, wanted_bands)
    else:
        print("\nPLI without bands.")
        conn = 'pli_no_bands'
        connectivity_matrix = calculate_connectivity(data_intervals, steps, num_channels, sample_rate, conn)
        
    make_graph(connectivity_matrix, ch_names,  threshold)
    
    
    
def dtf(path, window_size, bands, exclude = [None], threshold = 0.3):
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    wanted_bands = input_bands(bands)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_dtf(data_intervals, steps, num_channels, sample_rate, wanted_bands)
        
    make_directed_graph(connectivity_matrix, ch_names,  threshold)
    
    
#def vg(path, window_size, exclude = [None], threshold = 0.3, kernel= 'binary'):
    
    #data = input_data_type(path, exclude)
    
    #raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    #data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    #visibility_grahps = calculate_visibility_graphs(data_intervals, kernel)
    
    

def power_spectrum(path, window_size, bands, exclude = [None]):
    
    conn = 'ps'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    wanted_bands = input_bands(bands)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity_single_channel_with_bands(data_intervals, sample_rate, conn, wanted_bands)
    
    single_channel_graph(connectivity_matrix, ch_names, num_channels, wanted_bands)
    
    
def spectral_entropy(path, window_size, bands, exclude = [None]):
    
    conn = 'se'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    wanted_bands = input_bands(bands)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity_single_channel_with_bands(data_intervals, sample_rate, conn, wanted_bands)
    
    single_channel_graph(connectivity_matrix, ch_names, num_channels, wanted_bands)
    
    
def shannon_entropy(path, window_size, exclude = [None]):
    
    conn = 'she'
    
    data = input_data_type(path, exclude)
    
    raw_data, num_channels, sample_rate, sample_duration, ch_names = get_display_info(data)
    
    data_intervals, steps = time_intervals(raw_data, sample_rate, sample_duration, window_size)
    
    connectivity_matrix = calculate_connectivity_single_channel(data_intervals, sample_rate, conn)
    
    single_channel_graph(connectivity_matrix, ch_names, num_channels)