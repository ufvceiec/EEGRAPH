import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.stats import entropy
import antropy as ant
from .tools import *
from abc import ABC, abstractmethod

    
class Strategy(ABC):
    
    @abstractmethod
    def calculate_connectivity_workflow(self):
        pass  


    def make_graph_workflow(self, data):
        pass


#Concrete Strategies, Workflows. 
class Connectivity_No_Bands(Strategy):
    def calculate_connectivity_workflow(self, data, bands, window_size):
        dont_need_bands(bands)
        data_intervals, steps, self.flag = calculate_time_intervals(data.raw_data, data.sample_rate, data.sample_duration, window_size, data.sample_length)
        self.connectivity_matrix = calculate_connectivity(data_intervals, steps, data.num_channels, data.sample_rate, self)
        
        return self.connectivity_matrix
    
    def make_graph_workflow(self, data):
        G = make_graph(self.connectivity_matrix, data.ch_names, data.threshold)
    
        return G
    
class Connectivity_With_Bands(Strategy):
    def calculate_connectivity_workflow(self, data , bands, window_size):
        self.bands = input_bands(bands)
        data_intervals, steps, self.flag = calculate_time_intervals(data.raw_data, data.sample_rate, data.sample_duration, window_size, data.sample_length)
        self.connectivity_matrix = calculate_connectivity_with_bands(data_intervals, steps, data.num_channels, data.sample_rate, self, self.bands)
        
        return self.connectivity_matrix

    def make_graph_workflow(self, data):
        G = make_graph(self.connectivity_matrix, data.ch_names, data.threshold)
    
        return G
    
class Connectivity_single_channel_With_Bands(Strategy):
    def calculate_connectivity_workflow(self, data, bands, window_size):       
        self.bands = input_bands(bands)
        data_intervals, steps, self.flag = calculate_time_intervals(data.raw_data, data.sample_rate, data.sample_duration, window_size, data.sample_length)
        self.connectivity_matrix = calculate_connectivity_single_channel_with_bands(data_intervals, data.sample_rate, self, self.bands)
        
        return self.connectivity_matrix

    def make_graph_workflow(self, data):
        G, c_m = single_channel_graph(self.connectivity_matrix, data.ch_names, data.num_channels, data.threshold, self.bands)
    
        return (G, c_m)
    
class Connectivity_single_channel_No_Bands(Strategy):
    def calculate_connectivity_workflow(self, data, bands, window_size):
        dont_need_bands(bands)
        data_intervals, steps, self.flag = calculate_time_intervals(data.raw_data, data.sample_rate, data.sample_duration, window_size, data.sample_length)
        self.connectivity_matrix = calculate_connectivity_single_channel(data_intervals, data.sample_rate, self)
        
        return self.connectivity_matrix

    def make_graph_workflow(self, data):
        G, c_m = single_channel_graph(self.connectivity_matrix, data.ch_names, data.num_channels, data.threshold)
    
        return (G, c_m)
    
class Cross_correlation_rescaled(Strategy):
    def calculate_connectivity_workflow(self, data, bands, window_size):
        dont_need_bands(bands)
        scaled_data = re_scaling(data.raw_data)
        data_intervals, steps, self.flag = calculate_time_intervals(scaled_data, data.sample_rate, data.sample_duration, window_size, data.sample_length)
        self.connectivity_matrix = calculate_connectivity(data_intervals, steps, data.num_channels, data.sample_rate, self)
        
        return self.connectivity_matrix
        
    def make_graph_workflow(self, data):
        G = make_graph(self.connectivity_matrix, data.ch_names, data.threshold)
    
        return G
    
class Dtf_With_Bands(Strategy):
    def calculate_connectivity_workflow(self, data, bands, window_size):
        self.bands = input_bands(bands)
        data_intervals, steps, self.flag = calculate_time_intervals(data.raw_data, data.sample_rate, data.sample_duration, window_size, data.sample_length)
        self.connectivity_matrix = calculate_dtf(data_intervals, steps, data.num_channels, data.sample_rate, self.bands, self.flag)
        
        return self.connectivity_matrix

    def make_graph_workflow(self, data):
        G = make_graph(self.connectivity_matrix, data.ch_names, data.threshold, True)
    
        return G
        
        
#Connectivity measures            
class Cross_correlation_Estimator(Cross_correlation_rescaled):
    def __init__(self):
        self.threshold = 0.5
        
    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        x = data_intervals[i]
        y = data_intervals[j]
        
        Rxy = signal.correlate(x,y, 'full')
        Rxx = signal.correlate(x,x, 'full')
        Ryy = signal.correlate(y,y, 'full')
        
        lags = np.arange(-len(data_intervals[i]) + 1, len(data_intervals[i]))
        lag_0 = int((np.where(lags==0))[0])

        Rxx_0 = Rxx[lag_0]
        Ryy_0 = Ryy[lag_0]
        
        Rxy_norm = (1/(np.sqrt(Rxx_0*Ryy_0)))* Rxy
        
        #We use the mean from lag 0 to a 10% displacement. 
        disp = round((len(data_intervals[i])) * 0.10)

        cc_coef = Rxy_norm[lag_0: lag_0 + disp].mean()
        
        return cc_coef
        

class Pearson_correlation_Estimator(Connectivity_No_Bands):
    def __init__(self):
        self.threshold = 0.7
        
    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        r, p_value = (stats.pearsonr(data_intervals[i],data_intervals[j]))
        
        return r

class Squared_coherence_Estimator(Connectivity_With_Bands):
    def __init__(self):
        self.threshold = 0.65
        
    def calculate_conn(self, data_intervals, i, j, sample_rate, channels, bands):
        f, Cxy = (signal.coherence(data_intervals[i], data_intervals[j], sample_rate))
        
        delta, theta, alpha, beta, gamma = frequency_bands(f, Cxy)
        
        return delta.mean(), theta.mean(), alpha.mean(), beta.mean(), gamma.mean()
    
class Imag_coherence_Estimator(Connectivity_With_Bands):
    def __init__(self):
        self.threshold = 0.4
        
    def calculate_conn(self, data_intervals, i, j, sample_rate, channels, bands):
        _, Pxx = signal.welch(data_intervals[i], fs=sample_rate)
        _, Pyy = signal.welch(data_intervals[j], fs=sample_rate)
        f, Pxy = signal.csd(data_intervals[i],data_intervals[j],fs=sample_rate)
        icoh = np.imag(Pxy)/(np.sqrt(Pxx*Pyy))
        
        delta, theta, alpha, beta, gamma = frequency_bands(f, icoh)
        
        return delta.mean(), theta.mean(), alpha.mean(), beta.mean(), gamma.mean()
    
class Corr_cross_correlation_Estimator(Cross_correlation_rescaled):
    def __init__(self):
        self.threshold = 0.1
        
    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        x = data_intervals[i]
        y = data_intervals[j]
        
        Rxy = signal.correlate(x,y, 'full')
        Rxx = signal.correlate(x,x, 'full')
        Ryy = signal.correlate(y,y, 'full')
        
        lags = np.arange(-len(data_intervals[i]) + 1, len(data_intervals[i]))
        lag_0 = int((np.where(lags==0))[0])

        Rxx_0 = Rxx[lag_0]
        Ryy_0 = Ryy[lag_0]
        
        Rxy_norm = (1/(np.sqrt(Rxx_0*Ryy_0)))* Rxy
        negative_lag = Rxy_norm[:lag_0]
        positive_lag = Rxy_norm[lag_0 + 1:]
        
        corCC = positive_lag - negative_lag
        
        #We use the mean from lag 0 to a 10% displacement. 
        disp = round((len(data_intervals[i])) * 0.10)
        
        corCC_coef = corCC[:disp].mean()
        
        return corCC_coef    
    
class Wpli_Estimator(Connectivity_With_Bands):
    def __init__(self):
        self.threshold = 0.45
        
    def calculate_conn(self, data_intervals, i, j, sample_rate, channels, bands):
        f, Pxy = signal.csd(data_intervals[i],data_intervals[j],fs=sample_rate)
        
        delta, theta, alpha, beta, gamma = frequency_bands(f, Pxy)
        
        delta_denominator = np.mean(abs(np.imag(delta)))
        theta_denominator = np.mean(abs(np.imag(theta)))
        alpha_denominator = np.mean(abs(np.imag(alpha)))
        beta_denominator = np.mean(abs(np.imag(beta)))
        gamma_denominator = np.mean(abs(np.imag(gamma)))
        
        
        if(delta_denominator):
            wpli_delta = abs(np.mean(abs(np.imag(delta)) * np.sign(np.imag(delta)))) / (np.mean(abs(np.imag(delta))))
        else:
            wpli_delta = 0
            
        if(theta_denominator):
            wpli_theta = abs(np.mean(abs(np.imag(theta)) * np.sign(np.imag(theta)))) / (np.mean(abs(np.imag(theta))))
        else:
            wpli_theta = 0
           
        if(alpha_denominator):           
            wpli_alpha = abs(np.mean(abs(np.imag(alpha)) * np.sign(np.imag(alpha)))) / (np.mean(abs(np.imag(alpha)))) 
        else:
            wpli_alpha = 0
           
        if(beta_denominator): 
            wpli_beta = abs(np.mean(abs(np.imag(beta)) * np.sign(np.imag(beta)))) / (np.mean(abs(np.imag(beta))))
        else:
            wpli_beta = 0
           
        if(gamma_denominator):
            wpli_gamma = abs(np.mean(abs(np.imag(gamma)) * np.sign(np.imag(gamma)))) / (np.mean(abs(np.imag(gamma))))
        else:
            wpli_gamma = 0
           
        return wpli_delta, wpli_theta, wpli_alpha, wpli_beta, wpli_gamma
    
class Plv_Estimator(Connectivity_With_Bands):
    def __init__(self):
        self.threshold = 0.8
        
    def calculate_conn(self, data_intervals, i, j, sample_rate, channels, bands):
        sig1_delta, sig1_theta, sig1_alpha, sig1_beta, sig1_gamma = calculate_bands_fft(data_intervals[i], sample_rate, bands)
        sig2_delta, sig2_theta, sig2_alpha, sig2_beta, sig2_gamma = calculate_bands_fft(data_intervals[j], sample_rate, bands)
        
        sig1_bands = instantaneous_phase([sig1_delta, sig1_theta, sig1_alpha, sig1_beta, sig1_gamma])
        sig2_bands = instantaneous_phase([sig2_delta, sig2_theta, sig2_alpha, sig2_beta, sig2_gamma])
        
        complex_phase_diff_delta = np.exp(complex(0,1)*(sig1_bands[0] - sig2_bands[0]))
        complex_phase_diff_theta = np.exp(complex(0,1)*(sig1_bands[1] - sig2_bands[1]))
        complex_phase_diff_alpha = np.exp(complex(0,1)*(sig1_bands[2] - sig2_bands[2]))
        complex_phase_diff_beta = np.exp(complex(0,1)*(sig1_bands[3] - sig2_bands[3]))
        complex_phase_diff_gamma = np.exp(complex(0,1)*(sig1_bands[4] - sig2_bands[4]))
        
        plv_delta = np.abs(np.sum(complex_phase_diff_delta))/len(sig1_bands[0])
        plv_theta = np.abs(np.sum(complex_phase_diff_theta))/len(sig1_bands[1])
        plv_alpha = np.abs(np.sum(complex_phase_diff_alpha))/len(sig1_bands[2])
        plv_beta = np.abs(np.sum(complex_phase_diff_beta))/len(sig1_bands[3])
        plv_gamma = np.abs(np.sum(complex_phase_diff_gamma))/len(sig1_bands[4])
        
        return plv_delta, plv_theta, plv_alpha, plv_beta, plv_gamma
        
class Pli_Bands_Estimator(Connectivity_With_Bands):
    def __init__(self):
        self.threshold = 0.1
        
    def calculate_conn(self, data_intervals, i, j, sample_rate, channels, bands):
        sig1_delta, sig1_theta, sig1_alpha, sig1_beta, sig1_gamma = calculate_bands_fft(data_intervals[i], sample_rate, bands)
        sig2_delta, sig2_theta, sig2_alpha, sig2_beta, sig2_gamma = calculate_bands_fft(data_intervals[j], sample_rate, bands)
        
        sig1_bands = instantaneous_phase([sig1_delta, sig1_theta, sig1_alpha, sig1_beta, sig1_gamma])
        sig2_bands = instantaneous_phase([sig2_delta, sig2_theta, sig2_alpha, sig2_beta, sig2_gamma])
        
        phase_diff_delta = sig1_bands[0] - sig2_bands[0]
        phase_diff_delta = (phase_diff_delta + np.pi) % (2 * np.pi) - np.pi
        
        phase_diff_theta = sig1_bands[1] - sig2_bands[1]
        phase_diff_theta = (phase_diff_theta + np.pi) % (2 * np.pi) - np.pi
        
        phase_diff_alpha = sig1_bands[2] - sig2_bands[2]
        phase_diff_alpha = (phase_diff_alpha + np.pi) % (2 * np.pi) - np.pi
        
        phase_diff_beta = sig1_bands[3] - sig2_bands[3]
        phase_diff_beta  = (phase_diff_beta  + np.pi) % (2 * np.pi) - np.pi
        
        phase_diff_gamma = sig1_bands[4] - sig2_bands[4]
        phase_diff_gamma  = (phase_diff_gamma  + np.pi) % (2 * np.pi) - np.pi
        
        pli_delta = abs(np.mean(np.sign(phase_diff_delta)))
        pli_theta = abs(np.mean(np.sign(phase_diff_theta)))
        pli_alpha = abs(np.mean(np.sign(phase_diff_alpha)))
        pli_beta = abs(np.mean(np.sign(phase_diff_beta)))
        pli_gamma = abs(np.mean(np.sign(phase_diff_gamma)))
        
        return pli_delta, pli_theta, pli_alpha, pli_beta, pli_gamma
    
class Pli_No_Bands_Estimator(Connectivity_No_Bands):
    def __init__(self):
        self.threshold = 0.1
        
    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        sig1_phase = instantaneous_phase([data_intervals[i]])
        sig2_phase = instantaneous_phase([data_intervals[j]])
        phase_diff = sig1_phase[0] - sig2_phase[0]
        phase_diff = (phase_diff  + np.pi) % (2 * np.pi) - np.pi
        pli = abs(np.mean(np.sign(phase_diff)))
        
        return pli
    
class Power_spectrum_Estimator(Connectivity_single_channel_With_Bands):
    def __init__(self):
        self.threshold = 0.25            #<----- 25%
        
    #https://www.kite.com/python/answers/how-to-plot-a-power-spectrum-in-python
    def single_channel_conn(self, data, sample_rate):
        fourier_transform = np.fft.rfft(data)
        abs_fourier_transform = np.abs(fourier_transform)
        power_spectrum = np.square(abs_fourier_transform)
        return power_spectrum.mean()    
    
class Spectral_entropy_Estimator(Connectivity_single_channel_With_Bands):
    def __init__(self):
        self.threshold = 0.25            #<----- 25%
        
    #https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html#antropy.spectral_entropy
    def single_channel_conn(self, data, sample_rate):
        nperseg = len(data)
        se = ant.spectral_entropy(data, sample_rate, method='welch', nperseg = nperseg, normalize=True)
        return se
    
class Shannon_entropy_Estimator(Connectivity_single_channel_No_Bands):
    def __init__(self):
        self.threshold = 0.25            #<----- 25%
        
    #https://www.kite.com/python/answers/how-to-calculate-shannon-entropy-in-python
    def single_channel_conn(self, data, sample_rate):
        pd_series = pd.Series(data)
        counts = pd_series.value_counts()
        she = entropy(counts)
        return she

class Dtf_Estimator(Dtf_With_Bands):
    def __init__(self):
        self.threshold = 0.3