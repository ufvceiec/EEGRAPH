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
        lag_0 = int(np.where(lags==0)[0][0])

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
        lag_0 = int(np.where(lags==0)[0][0])

        Rxx_0 = Rxx[lag_0]
        Ryy_0 = Ryy[lag_0]
        
        Rxy_norm = (1/(np.sqrt(Rxx_0*Ryy_0)))* Rxy
        negative_lag = Rxy_norm[:lag_0]
        positive_lag = Rxy_norm[lag_0 + 1:]

        min_len = min(len(positive_lag), len(negative_lag))
        if min_len == 0:
            return 0.0
        corCC = positive_lag[:min_len] - negative_lag[:min_len]
        
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


# ─────────────────────────────────────────────────────────────────────────────
# New directed base classes
# ─────────────────────────────────────────────────────────────────────────────

class Connectivity_No_Bands_Directed(Strategy):
    """Directed, no-band workflow — produces a DiGraph."""
    def calculate_connectivity_workflow(self, data, bands, window_size):
        dont_need_bands(bands)
        data_intervals, steps, self.flag = calculate_time_intervals(
            data.raw_data, data.sample_rate, data.sample_duration,
            window_size, data.sample_length
        )
        self.connectivity_matrix = calculate_connectivity(
            data_intervals, steps, data.num_channels, data.sample_rate, self
        )
        return self.connectivity_matrix

    def make_graph_workflow(self, data):
        G = make_graph(self.connectivity_matrix, data.ch_names, data.threshold, True)
        return G


class Connectivity_With_Bands_Directed(Strategy):
    """Directed, with-bands workflow — produces a DiGraph per band."""
    def calculate_connectivity_workflow(self, data, bands, window_size):
        self.bands = input_bands(bands)
        data_intervals, steps, self.flag = calculate_time_intervals(
            data.raw_data, data.sample_rate, data.sample_duration,
            window_size, data.sample_length
        )
        self.connectivity_matrix = calculate_connectivity_with_bands(
            data_intervals, steps, data.num_channels, data.sample_rate, self, self.bands
        )
        return self.connectivity_matrix

    def make_graph_workflow(self, data):
        G = make_graph(self.connectivity_matrix, data.ch_names, data.threshold, True)
        return G


# ─────────────────────────────────────────────────────────────────────────────
# New undirected, no-bands estimators
# ─────────────────────────────────────────────────────────────────────────────

class Aec_Estimator(Connectivity_No_Bands):
    """Amplitude Envelope Correlation (Pearson correlation of Hilbert envelopes)."""
    def __init__(self):
        self.threshold = 0.5

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        from scipy.signal import hilbert
        env_i = np.abs(hilbert(data_intervals[i]))
        env_j = np.abs(hilbert(data_intervals[j]))
        r, _ = stats.pearsonr(env_i, env_j)
        return float(r)


class Aec_orth_Estimator(Connectivity_No_Bands):
    """Orthogonalized AEC (Hipp et al. 2012) — leakage-corrected envelope correlation."""
    def __init__(self):
        self.threshold = 0.3

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        from scipy.signal import hilbert
        x = hilbert(data_intervals[i]).astype(complex)
        y = hilbert(data_intervals[j]).astype(complex)
        # Orthogonalise y w.r.t. x
        y_orth = np.imag(y * np.conj(x) / np.abs(x))
        r1, _ = stats.pearsonr(np.abs(x), np.abs(y_orth))
        # Symmetrise
        x_orth = np.imag(x * np.conj(y) / np.abs(y))
        r2, _ = stats.pearsonr(np.abs(y), np.abs(x_orth))
        return float((r1 + r2) / 2.0)


class Mutual_information_Estimator(Connectivity_No_Bands):
    """Mutual Information estimated via equal-width histogram binning."""
    def __init__(self):
        self.threshold = 0.2

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        n_bins = max(10, int(np.sqrt(len(data_intervals[i]))))
        x = data_intervals[i]
        y = data_intervals[j]
        # Joint histogram
        c_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
        c_x = c_xy.sum(axis=1)
        c_y = c_xy.sum(axis=0)
        n = c_xy.sum()
        if n == 0:
            return 0.0
        p_xy = c_xy / n
        p_x = c_x / n
        p_y = c_y / n
        # MI = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
        mask = p_xy > 0
        mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / np.outer(p_x, p_y)[mask]))
        return float(max(0.0, mi))


class Sync_likelihood_Estimator(Connectivity_No_Bands):
    """Synchronisation Likelihood (Stam & van Dijk 2002) — recurrence-based measure."""
    def __init__(self):
        self.threshold = 0.05

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        x = data_intervals[i]
        y = data_intervals[j]
        N = len(x)
        if N < 10:
            return 0.0
        # Embedding parameters (simplified: m=3, tau=1)
        m, tau = 3, 1
        # Build embedded matrices
        max_t = N - (m - 1) * tau
        if max_t <= 0:
            return 0.0
        X = np.array([x[t:t + m * tau:tau] for t in range(max_t)])
        Y = np.array([y[t:t + m * tau:tau] for t in range(max_t)])
        # Use p_ref = 0.05 (probability of recurrence)
        p_ref = 0.05
        # Threshold: distance below which points are "recurrent"
        dists_x = np.sqrt(((X[:, None] - X[None, :]) ** 2).sum(axis=2))
        dists_y = np.sqrt(((Y[:, None] - Y[None, :]) ** 2).sum(axis=2))
        eps_x = np.percentile(dists_x, p_ref * 100)
        eps_y = np.percentile(dists_y, p_ref * 100)
        R_x = (dists_x < eps_x).astype(float)
        R_y = (dists_y < eps_y).astype(float)
        np.fill_diagonal(R_x, 0)
        np.fill_diagonal(R_y, 0)
        # SL = mean over i of (sum_j R_x[i,j] * R_y[i,j]) / sum_j R_x[i,j]
        num = (R_x * R_y).sum(axis=1)
        den = R_x.sum(axis=1)
        valid = den > 0
        if not valid.any():
            return 0.0
        sl = float(np.mean(num[valid] / den[valid]))
        return sl


# ─────────────────────────────────────────────────────────────────────────────
# New undirected, with-bands estimators
# ─────────────────────────────────────────────────────────────────────────────

class Dwpli_Estimator(Connectivity_With_Bands):
    """Debiased Weighted Phase-Lag Index (Vinck et al. 2011)."""
    def __init__(self):
        self.threshold = 0.3

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels, bands):
        f, Pxy = signal.csd(data_intervals[i], data_intervals[j], fs=sample_rate)
        delta, theta, alpha, beta, gamma = frequency_bands(f, Pxy)

        def _dwpli(band):
            if len(band) == 0:
                return 0.0
            im = np.imag(band)
            n = len(im)
            num = np.mean(im) ** 2 - np.var(im) / n
            den = np.mean(im ** 2)
            return float(num / den) if den != 0 else 0.0

        return _dwpli(delta), _dwpli(theta), _dwpli(alpha), _dwpli(beta), _dwpli(gamma)


class Ppc_Estimator(Connectivity_With_Bands):
    """Pairwise Phase Consistency (Vinck et al. 2010) — unbiased PLV²."""
    def __init__(self):
        self.threshold = 0.3

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels, bands):
        sig1_delta, sig1_theta, sig1_alpha, sig1_beta, sig1_gamma = calculate_bands_fft(data_intervals[i], sample_rate, bands)
        sig2_delta, sig2_theta, sig2_alpha, sig2_beta, sig2_gamma = calculate_bands_fft(data_intervals[j], sample_rate, bands)
        sig1_bands = instantaneous_phase([sig1_delta, sig1_theta, sig1_alpha, sig1_beta, sig1_gamma])
        sig2_bands = instantaneous_phase([sig2_delta, sig2_theta, sig2_alpha, sig2_beta, sig2_gamma])

        def _ppc(phi1, phi2):
            diffs = phi1 - phi2
            n = len(diffs)
            if n < 2:
                return 0.0
            z = np.exp(1j * diffs)
            plv2 = (np.abs(z.mean())) ** 2
            # Unbiased: PPC = (n * PLV² - 1) / (n - 1)
            ppc = (n * plv2 - 1) / (n - 1)
            return float(max(0.0, ppc))

        return (
            _ppc(sig1_bands[0], sig2_bands[0]),
            _ppc(sig1_bands[1], sig2_bands[1]),
            _ppc(sig1_bands[2], sig2_bands[2]),
            _ppc(sig1_bands[3], sig2_bands[3]),
            _ppc(sig1_bands[4], sig2_bands[4]),
        )


class Lagged_coherence_Estimator(Connectivity_With_Bands):
    """Lagged Coherence — squared imaginary coherence (Pascual-Marqui 2007)."""
    def __init__(self):
        self.threshold = 0.2

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels, bands):
        _, Pxx = signal.welch(data_intervals[i], fs=sample_rate)
        _, Pyy = signal.welch(data_intervals[j], fs=sample_rate)
        f, Pxy = signal.csd(data_intervals[i], data_intervals[j], fs=sample_rate)
        lc = (np.imag(Pxy) ** 2) / (Pxx * Pyy + 1e-12)
        delta, theta, alpha, beta, gamma = frequency_bands(f, lc)

        def _safe_mean(arr):
            return float(arr.mean()) if len(arr) > 0 else 0.0

        return _safe_mean(delta), _safe_mean(theta), _safe_mean(alpha), _safe_mean(beta), _safe_mean(gamma)


# ─────────────────────────────────────────────────────────────────────────────
# New directed, no-bands estimators
# ─────────────────────────────────────────────────────────────────────────────

class Granger_causality_Estimator(Connectivity_No_Bands_Directed):
    """Bivariate Granger Causality (F-test based, AR order = 5)."""
    def __init__(self):
        self.threshold = 0.1

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        x = np.array(data_intervals[i], dtype=float)
        y = np.array(data_intervals[j], dtype=float)
        p = 5  # AR model order
        n = len(x)
        if n <= p + 1:
            return 0.0

        def _design_matrix(series, order):
            """Build [y(t-1), ..., y(t-p)] regressor matrix."""
            rows = []
            for t in range(order, len(series)):
                rows.append(series[t - order:t][::-1])
            return np.array(rows)

        # Restricted model: x(t) ~ x(t-1..p)
        X_r = _design_matrix(x, p)
        y_r = x[p:]
        beta_r, _, _, _ = np.linalg.lstsq(X_r, y_r, rcond=None)
        rss_r = np.sum((y_r - X_r @ beta_r) ** 2)

        # Unrestricted model: x(t) ~ x(t-1..p) + y(t-1..p)
        Y_u = _design_matrix(y, p)
        X_u = np.hstack([X_r, Y_u])
        beta_u, _, _, _ = np.linalg.lstsq(X_u, y_r, rcond=None)
        rss_u = np.sum((y_r - X_u @ beta_u) ** 2)

        if rss_r <= 0 or rss_u <= 0:
            return 0.0
        # GC = log(RSS_restricted / RSS_unrestricted)
        gc = float(np.log(rss_r / (rss_u + 1e-12)))
        return max(0.0, gc)


class Transfer_entropy_Estimator(Connectivity_No_Bands_Directed):
    """Transfer Entropy from j→i via histogram estimation (k=1 lag)."""
    def __init__(self):
        self.threshold = 0.05

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels):
        x = np.array(data_intervals[i], dtype=float)
        y = np.array(data_intervals[j], dtype=float)
        n_bins = max(5, int(np.sqrt(len(x)) // 2))
        # Discretise
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if x_max == x_min or y_max == y_min:
            return 0.0

        def _digitize(arr, lo, hi, n):
            bins = np.linspace(lo, hi, n + 1)
            return np.clip(np.digitize(arr, bins) - 1, 0, n - 1)

        xd = _digitize(x, x_min, x_max, n_bins)
        yd = _digitize(y, y_min, y_max, n_bins)

        # x(t), x(t-1), y(t-1) — use lag k=1
        xt  = xd[1:]
        xt1 = xd[:-1]
        yt1 = yd[:-1]

        # Joint counts
        def _prob(*arrays):
            keys = list(zip(*arrays))
            from collections import Counter
            c = Counter(keys)
            total = len(keys)
            return {k: v / total for k, v in c.items()}

        p_xt_xt1_yt1 = _prob(xt, xt1, yt1)
        p_xt1_yt1    = _prob(xt1, yt1)
        p_xt_xt1     = _prob(xt, xt1)
        p_xt1        = _prob(xt1,)

        te = 0.0
        for (a, b, c_), p_joint in p_xt_xt1_yt1.items():
            p_cond_full    = p_joint / (p_xt1_yt1.get((b, c_), 1e-12))
            p_cond_reduced = p_xt_xt1.get((a, b), 1e-12) / (p_xt1.get((b,), 1e-12))
            if p_cond_full > 0 and p_cond_reduced > 0:
                te += p_joint * np.log2(p_cond_full / p_cond_reduced)

        return float(max(0.0, te))


# ─────────────────────────────────────────────────────────────────────────────
# New directed, with-bands estimators
# ─────────────────────────────────────────────────────────────────────────────

class Pdc_Estimator(Dtf_With_Bands):
    """Partial Directed Coherence via MVAR model (scot library)."""
    def __init__(self):
        self.threshold = 0.3

    def calculate_connectivity_workflow(self, data, bands, window_size):
        self.bands = input_bands(bands)
        data_intervals, steps, self.flag = calculate_time_intervals(
            data.raw_data, data.sample_rate, data.sample_duration,
            window_size, data.sample_length
        )
        self.connectivity_matrix = calculate_pdc(
            data_intervals, steps, data.num_channels, data.sample_rate, self.bands, self.flag
        )
        return self.connectivity_matrix


class Psi_Estimator(Connectivity_With_Bands_Directed):
    """Phase Slope Index (Nolte et al. 2008) — directed spectral measure."""
    def __init__(self):
        self.threshold = 0.0

    def calculate_conn(self, data_intervals, i, j, sample_rate, channels, bands):
        f, Pxy = signal.csd(data_intervals[i], data_intervals[j], fs=sample_rate,
                            nperseg=min(256, len(data_intervals[i])))
        _, Pxx = signal.welch(data_intervals[i], fs=sample_rate,
                              nperseg=min(256, len(data_intervals[i])))
        _, Pyy = signal.welch(data_intervals[j], fs=sample_rate,
                              nperseg=min(256, len(data_intervals[i])))

        # Coherency C(f) = Pxy / sqrt(Pxx * Pyy)
        C = Pxy / (np.sqrt(Pxx * Pyy) + 1e-12)
        # PSI = Im( sum_f  conj(C(f)) * C(f+df) )
        # computed per frequency band
        def _psi_band(idx):
            if len(idx) < 2:
                return 0.0
            C_band = C[idx]
            psi = float(np.imag(np.sum(np.conj(C_band[:-1]) * C_band[1:])))
            return psi

        delta_idx = np.where((f >= 0.5) & (f < 4))[0]
        theta_idx = np.where((f >= 4)  & (f < 8))[0]
        alpha_idx = np.where((f >= 8)  & (f < 13))[0]
        beta_idx  = np.where((f >= 13) & (f < 30))[0]
        gamma_idx = np.where((f >= 30) & (f < 100))[0]

        return (
            _psi_band(delta_idx),
            _psi_band(theta_idx),
            _psi_band(alpha_idx),
            _psi_band(beta_idx),
            _psi_band(gamma_idx),
        )