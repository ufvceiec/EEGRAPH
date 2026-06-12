# Changelog

All notable changes to EEGraph are documented here.

---

```
Change Log
==========

0.1.17 (2026-06-12)
------------------
- Fixed re_scaling(): result of df.sub() was discarded; now correctly assigned back.
- Fixed off-by-one in single_channel_graph(): percentile slice excluded last channel.
- Fixed delta band lower boundary in frequency_bands(): changed > to >= for consistency with obtain_frequency_bands().
- Fixed isinstance() checks in time_stamps() replacing type() == comparisons (fixes numpy scalar types).
- Fixed zero-length interval guard in time_stamps() when EEG length equals exact window multiple.
- Fixed deprecated logging.warn() → logging.warning().
- Fixed eval() in importData.load(): replaced with _readers dispatch dict of lambdas.
- Fixed eval() in Graph.modelate(): replaced with globals() lookup; search() now returns class name without parentheses.
- Fixed if(threshold) is not None: parenthesis error in ModelData.connectivity_workflow().
- Fixed type(out) is tuple → isinstance(out, tuple) in ModelData.connectivity_workflow().
- Fixed empty-array guard in Corr_cross_correlation_Estimator.calculate_conn().
- Fixed typo classifers → classifiers in setup.py.
- Fixed duplicate test method test_processed_channel_names_dash in tests.py.
- Fixed NumPy ≥ 1.25 DeprecationWarning: Y[f==0] → Y[f==0][0] in obtain_frequency_bands().
- Fixed NumPy ≥ 1.25 DeprecationWarning: lag_0 extraction uses np.where()[0][0].
- Fixed invalid escape sequence \s → r"\s" in set_montage() CSV delimiter.
- Added CLAUDE.md with architecture reference for contributors.
- Added compute_graph_metrics() and compute_metrics_all() to tools.py for graph-theoretic analysis.
- Added Graph.compute_metrics() convenience method.
- Added MkDocs HTML documentation suite (docs/ directory, mkdocs.yml).

0.1.12 (31/03/2021)
------------------
- Removed exclude option for VHDR files

0.1.11 (06/09/2021)
------------------
- Fixing returned connectivity matrix for single channel connectivity measures (Power Spectrum, Spectral Entropy, Shannon Entropy).
- Graph visualization with single channel connectivity measures now include the value for each node.


0.1.10 (06/09/2021)
------------------
- Fixing issue involving frequency bands.

0.1.9 (05/05/2021)
------------------
- Adding custom input percentage threshold for spectral entropy, power spectrum and shannon entropy.
- Updating DTF to generate directed graphs.

0.1.8 (05/04/2021)
------------------
- Updating Corrected cross-correlation default threshold.
- Rescaling data for Corrected cross-correlation.

0.1.7 (04/22/2021)
------------------
- Updating Entropy depedency to Antropy, all dependencies are now installed with pip.
- Updating np.complex to complex, deprecation warning numpy 1.20.

0.1.6 (03/25/2021)
------------------
- Adding the option to include a electrode montage file for channel labels, in method load_data().
- Adding warning if a channel label is not a recognize electrode position, it will be ignored for visualization.

0.1.5 (02/26/2021)
------------------
- Fixing issue with spectral entropy that returned empty graph, when the top 25% top values are the same.

0.1.4 (02/24/2021)
------------------
- Fixing some issues with dtf connectivity.
- Rounding printed intervals.
- Adding warning when a frequency band is empty.

0.1.3 (02/13/2021)
------------------
- Fixing bug that causes an increase in runtime when a single interval is used in 'modelate' method.

0.1.2 (02/11/2021)
------------------
- Fixing bug that causes an increase in runtime when a single interval is used in 'modelate' method.

0.1.1 (02/10/2021)
------------------
- Fixing error 'Cant convert complex to float' in Squared Coherence connectivity measure.

0.1.0 (02/02/2021)
------------------
- First Release
```
