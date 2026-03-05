"""
Shared test runner for pyaldata performance comparison.
Used both as a standalone script (for baseline) and imported from the notebook.
"""
import sys
import time
import warnings

import numpy as np

MAT_PATH = r"C:\data\raw\M079\M079_2025_09_26_10_30\M079_2025_09_26_10_30_pyaldata.mat"


def run_all_tests(df):
    """
    Run all testable pyaldata functions.
    Returns (results, timings) dicts.
    """
    import pyaldata
    from pyaldata.smoothing import smooth_data, norm_gauss_window
    from pyaldata.integrity_checks import all_integer
    from collections import Counter

    results = {}
    timings = {}

    def bench(name, func, n):
        t0 = time.perf_counter()
        for _ in range(n):
            r = func()
        timings[name] = (time.perf_counter() - t0) / n
        return r

    # ── Prepare subsets ────────────────────────────────────────────────
    lengths = pyaldata.get_trial_lengths(df)
    mc_len = Counter(lengths.tolist()).most_common(1)[0][0]
    df_same = pyaldata.select_trials(df, lengths == mc_len)
    df_sub = df_same.head(min(50, len(df_same))).copy().reset_index(drop=True)
    df_50 = df.head(50).copy().reset_index(drop=True)

    # ── Utils ──────────────────────────────────────────────────────────
    results['determine_ref_field'] = bench(
        'determine_ref_field', lambda: pyaldata.determine_ref_field(df), 100)
    results['get_time_varying_fields'] = bench(
        'get_time_varying_fields', lambda: pyaldata.get_time_varying_fields(df), 50)
    results['get_array_fields'] = bench(
        'get_array_fields', lambda: pyaldata.get_array_fields(df), 100)
    results['get_string_fields'] = bench(
        'get_string_fields', lambda: pyaldata.get_string_fields(df), 100)
    results['get_trial_lengths'] = bench(
        'get_trial_lengths', lambda: pyaldata.get_trial_lengths(df), 100)
    results['get_trial_length_single'] = bench(
        'get_trial_length_single', lambda: pyaldata.get_trial_length(df.iloc[0]), 100)
    results['trials_are_same_length'] = bench(
        'trials_are_same_length', lambda: pyaldata.trials_are_same_length(df), 20)
    results['trials_are_same_length_true'] = bench(
        'trials_are_same_length_true', lambda: pyaldata.trials_are_same_length(df_sub), 20)
    results['remove_suffix'] = bench(
        'remove_suffix', lambda: pyaldata.remove_suffix('MOp_spikes', '_spikes'), 1000)
    results['all_integer'] = bench(
        'all_integer', lambda: all_integer(np.array([1.0, 2.0, 3.0])), 1000)

    # ── Extract signals ────────────────────────────────────────────────
    results['concat_trials'] = bench(
        'concat_trials', lambda: pyaldata.concat_trials(df, 'MOp_spikes'), 20)
    ct = results['concat_trials']
    results['reverse_concat_first'] = bench(
        'reverse_concat', lambda: pyaldata.reverse_concat(ct, df), 10)[0]
    results['get_signals'] = bench(
        'get_signals', lambda: pyaldata.get_signals(df, ['MOp_spikes', 'all_spikes'], [0, 1, 2]), 50)
    results['get_sig_by_trial'] = bench(
        'get_sig_by_trial', lambda: pyaldata.get_sig_by_trial(df_sub, 'MOp_spikes'), 50)
    results['stack_time_average'] = bench(
        'stack_time_average', lambda: pyaldata.stack_time_average(df, 'MOp_spikes'), 20)
    results['split_array'] = [
        a.tolist() for a in bench(
            'split_array',
            lambda: pyaldata.split_array(np.arange(100).reshape(10, 10), [3, 3, 4]), 1000)]

    # ── Signals ────────────────────────────────────────────────────────
    results['signal_dimensionality'] = bench(
        'signal_dimensionality', lambda: pyaldata.signal_dimensionality(df, 'MOp_spikes'), 100)
    results['add_gradient'] = bench(
        'add_gradient', lambda: pyaldata.add_gradient(df_50, 'MOp_spikes'), 5)['dMOp_spikes'].iloc[0]
    results['add_norm'] = bench(
        'add_norm', lambda: pyaldata.add_norm(df_50, 'MOp_spikes'), 5)['MOp_spikes_norm'].iloc[0]

    # ── Firing rates ───────────────────────────────────────────────────
    results['add_firing_rates_bin'] = bench(
        'add_firing_rates_bin', lambda: pyaldata.add_firing_rates(df, method='bin'), 3)['MOp_rates'].iloc[0]
    results['add_firing_rates_smooth'] = bench(
        'add_firing_rates_smooth', lambda: pyaldata.add_firing_rates(df, method='smooth'), 3)['MOp_rates'].iloc[0]
    results['get_average_firing_rates'] = bench(
        'get_average_firing_rates', lambda: pyaldata.get_average_firing_rates(df, 'MOp_spikes'), 20)
    results['remove_low_firing_neurons_shape'] = bench(
        'remove_low_firing_neurons',
        lambda: pyaldata.remove_low_firing_neurons(df, 'MOp_spikes', 1.0), 3)['MOp_spikes'].iloc[0].shape

    # ── Smoothing ──────────────────────────────────────────────────────
    win = norm_gauss_window(0.01, 0.05)
    test_arr = df['MOp_spikes'].iloc[0].astype(float)
    results['smooth_data'] = bench(
        'smooth_data', lambda: smooth_data(test_arr, win=win), 100)
    results['hw_to_std'] = bench(
        'hw_to_std', lambda: pyaldata.hw_to_std(0.05), 1000)
    df_small = df.head(20).copy().reset_index(drop=True)
    df_small_r = pyaldata.add_firing_rates(df_small, method='bin')
    results['smooth_signals'] = bench(
        'smooth_signals', lambda: pyaldata.smooth_signals(df_small_r, ['MOp_rates']), 5)['MOp_rates'].iloc[0]

    # ── Tools ──────────────────────────────────────────────────────────
    mask = (df.trial_id < 10).values
    results['select_trials_str'] = bench(
        'select_trials_str', lambda: pyaldata.select_trials(df, 'trial_id < 10'), 20).shape
    results['select_trials_mask'] = bench(
        'select_trials_mask', lambda: pyaldata.select_trials(df, mask), 20).shape
    results['select_trials_callable'] = bench(
        'select_trials_callable', lambda: pyaldata.select_trials(df, lambda t: t.trial_id < 10), 10).shape
    results['merge_signals'] = bench(
        'merge_signals',
        lambda: pyaldata.merge_signals(df, ['MOp_spikes', 'all_spikes'], 'merged'), 3)['merged'].iloc[0]
    results['combine_time_bins'] = bench(
        'combine_time_bins', lambda: pyaldata.combine_time_bins(df_sub, 2), 3)['MOp_spikes'].iloc[0]
    # Drop object-dtype array columns (e.g. kslabel with string values) that
    # would cause trial_average to fail on .mean() — pre-existing limitation.
    obj_arr_cols = [c for c in df_sub.columns
                    if isinstance(df_sub[c].iloc[0], np.ndarray) and df_sub[c].iloc[0].dtype == object]
    df_sub_clean = df_sub.drop(columns=obj_arr_cols)
    results['trial_average'] = bench(
        'trial_average', lambda: pyaldata.trial_average(df_sub_clean.copy(), None), 3)['MOp_spikes'].iloc[0]
    results['subtract_cross_condition_mean'] = bench(
        'subtract_cross_condition_mean',
        lambda: pyaldata.subtract_cross_condition_mean(df_sub_clean), 3)['MOp_spikes'].iloc[0]
    df_a = df.iloc[:100].copy().reset_index(drop=True)
    df_b = df.iloc[50:150].copy().reset_index(drop=True)
    ka, kb = bench(
        'keep_common_trials', lambda: pyaldata.keep_common_trials(df_a, df_b), 3)
    results['keep_common_trials_shapes'] = (ka.shape, kb.shape)

    # ── Signal transformations ─────────────────────────────────────────
    results['get_range'] = bench(
        'get_range', lambda: pyaldata.get_range(test_arr), 1000)
    results['center_arr'] = bench(
        'center', lambda: pyaldata.center(test_arr), 1000)
    results['z_score_arr'] = bench(
        'z_score', lambda: pyaldata.z_score(test_arr.astype(float)), 1000)
    results['center_signal'] = bench(
        'center_signal', lambda: pyaldata.center_signal(df_50, 'MOp_spikes'), 5)['MOp_spikes'].iloc[0]
    results['z_score_signal'] = bench(
        'z_score_signal', lambda: pyaldata.z_score_signal(df_50, 'MOp_spikes'), 5)['MOp_spikes'].iloc[0]
    results['sqrt_transform_signal'] = bench(
        'sqrt_transform_signal', lambda: pyaldata.sqrt_transform_signal(df_50, 'MOp_spikes'), 5)['MOp_spikes'].iloc[0]
    results['zero_normalize_signal'] = bench(
        'zero_normalize_signal', lambda: pyaldata.zero_normalize_signal(df_50, 'MOp_spikes'), 5)['MOp_spikes'].iloc[0]
    results['center_normalize_signal'] = bench(
        'center_normalize_signal', lambda: pyaldata.center_normalize_signal(df_50, 'MOp_spikes'), 5)['MOp_spikes'].iloc[0]
    results['soft_normalize_signal'] = bench(
        'soft_normalize_signal', lambda: pyaldata.soft_normalize_signal(df_50, 'MOp_spikes'), 5)['MOp_spikes'].iloc[0]
    results['transform_signal'] = bench(
        'transform_signal',
        lambda: pyaldata.transform_signal(df_50, 'MOp_spikes', ['center', 'soft_normalize']), 5)['MOp_spikes'].iloc[0]

    # ── Interval ───────────────────────────────────────────────────────
    results['slice_around_index'] = bench(
        'slice_around_index', lambda: pyaldata.slice_around_index(10, 3, 5), 1000)
    trial0 = df.iloc[0]
    results['slice_around_point'] = bench(
        'slice_around_point',
        lambda: pyaldata.slice_around_point(trial0, 'idx_trial_start', 0, 4), 1000)
    results['slice_between_points'] = bench(
        'slice_between_points',
        lambda: pyaldata.slice_between_points(trial0, 'idx_trial_start', 'idx_trial_end', 0, 0), 1000)
    results['slice_in_trial'] = bench(
        'slice_in_trial', lambda: pyaldata.slice_in_trial(trial0, slice(0, 5)), 1000)
    df_int = df[df.trial_length >= 10].head(20).copy().reset_index(drop=True)
    results['restrict_to_interval'] = bench(
        'restrict_to_interval',
        lambda: pyaldata.restrict_to_interval(
            df_int, 'idx_trial_start', rel_end=4, ignore_warnings=True), 3)['MOp_spikes'].iloc[0]
    epoch_fun = pyaldata.generate_epoch_fun('idx_trial_start', rel_end=4)
    results['extract_interval_shapes'] = [
        a.shape for a in bench(
            'extract_interval_from_signal',
            lambda: pyaldata.extract_interval_from_signal(df_int, 'MOp_spikes', epoch_fun), 10)]

    # ── Regression / dim_reduce ────────────────────────────────────────
    results['expand_field_in_time'] = bench(
        'expand_field_in_time',
        lambda: pyaldata.expand_field_in_time(df_50, 'trial_id'), 5)['trial_id_ext'].iloc[0]
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression
        results['dim_reduce_shape'] = bench(
            'dim_reduce',
            lambda: pyaldata.dim_reduce(
                df_sub, PCA(n_components=5), 'MOp_spikes', 'MOp_pca'), 3)['MOp_pca'].iloc[0].shape
        results['regress_shape'] = bench(
            'regress',
            lambda: pyaldata.regress(
                df_sub, LinearRegression(), 'MOp_spikes', 'all_spikes', 'predicted'), 3)['predicted'].iloc[0].shape
    except ImportError:
        pass

    # ── DF utils ───────────────────────────────────────────────────────
    results['concat_TDs_shape'] = bench(
        'concat_TDs', lambda: pyaldata.concat_TDs([df.head(10), df.iloc[10:20]]), 50).shape
    results['rename_fields_cols'] = bench(
        'rename_fields',
        lambda: pyaldata.rename_fields(df, {'MOp_spikes': 'MOp_spikes_renamed'}), 20).columns.tolist()
    results['copy_fields_cols'] = bench(
        'copy_fields',
        lambda: pyaldata.copy_fields(df, {'MOp_spikes': 'MOp_spikes_copy'}), 20).columns.tolist()

    return results, timings


# ── Standalone baseline runner ─────────────────────────────────────────
if __name__ == '__main__':
    import pickle

    sys.path.insert(0, r"c:\repos\pyaldata")
    warnings.filterwarnings('ignore')

    import pyaldata

    print("Loading data with ORIGINAL code...")
    t0 = time.perf_counter()
    for _ in range(3):
        df = pyaldata.mat2dataframe(MAT_PATH, shift_idx_fields=False)
    load_time = (time.perf_counter() - t0) / 3
    print(f"  Loaded: {df.shape} in {load_time*1000:.1f} ms")

    print("Running tests...")
    results, timings = run_all_tests(df)
    timings['mat2dataframe'] = load_time

    output_path = r'c:\repos\pyaldata\baseline_results.pkl'
    pickle.dump({'results': results, 'timings': timings}, open(output_path, 'wb'))
    print(f"\nBaseline saved to {output_path}")
    print(f"Timed {len(timings)} functions:")
    for k, v in sorted(timings.items()):
        print(f"  {k:42s} {v*1000:10.3f} ms")
