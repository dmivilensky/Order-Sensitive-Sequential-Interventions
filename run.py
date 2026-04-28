#!/usr/bin/env python3
import os
import math
import random
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA_PATH = '/mnt/data/reviewing.csv'
OUT_DIR = '/mnt/data/closing_empirics_reviewing_final_outputs'
SEED = 7
BASE_ACCEPT_BONUS = 1.0
BASE_DURATION_PENALTY = 0.02
BOOTSTRAP_B = 120
TRAIN_FRAC = 0.7
N_SPLITS = 30
LAMBDA_GRID = [0.005, 0.01, 0.02, 0.05]
TARGETS = ['decide', 'collect reviews']
REVIEWS = ['get review 1', 'get review 2', 'get review 3']
PATH_ORDER = ['none', 'u', 'w', 'uw_forward', 'uw_reverse']
ENDPOINT_ORDER = ['none', 'u', 'w', 'uw']
POLICY_ORDER = [
    'sequence_sensitive',
    'reference_path',
    'greedy_one_step',
    'fixed_forward',
    'fixed_reverse',
    'endpoint_pooled',
    'frequency',
]
COLORS = {
    'none': '#B0B8C1',
    'u': '#4C78A8',
    'w': '#F58518',
    'uw': '#54A24B',
    'forward': '#4C78A8',
    'reverse': '#E45756',
    'sequence_sensitive': '#54A24B',
    'reference_path': '#4C78A8',
    'greedy_one_step': '#F58518',
    'fixed_forward': '#72B7B2',
    'fixed_reverse': '#E45756',
    'endpoint_pooled': '#B279A2',
    'frequency': '#9D755D',
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def short_name(s: str) -> str:
    return s.replace('get review ', 'r').replace('collect reviews', 'collect').replace('decide', 'decide')


def family_label(target: str, u: str, w: str) -> str:
    return f'{target} <- {{{u}, {w}}}'


def family_slug(target: str, u: str, w: str) -> str:
    return family_label(target, u, w).replace(' <- ', '__').replace('{', '').replace('}', '').replace(', ', '__').replace(' ', '_')


def load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'lifecycle:transition' in df.columns and (df['lifecycle:transition'] == 'complete').any():
        df = df[df['lifecycle:transition'] == 'complete'].copy()
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['case:concept:name', 'concept:name', 'time:timestamp']).copy()
    df = df.sort_values(['case:concept:name', 'time:timestamp', 'concept:name']).reset_index(drop=True)
    return df


def build_case_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in df.groupby('case:concept:name', sort=False):
        g = g.reset_index(drop=True)
        names = g['concept:name'].tolist()
        rows.append({
            'case_id': cid,
            'events': names,
            'times': g['time:timestamp'].tolist(),
            'case_end': g['time:timestamp'].max(),
            'final_accept': 1 if 'accept' in set(names) else 0,
        })
    return pd.DataFrame(rows)


def compute_reward(final_accept: float, remaining_days: float, duration_penalty: float) -> float:
    return float(final_accept) - duration_penalty * float(remaining_days)


def build_family_episodes(case_df: pd.DataFrame, target: str, u: str, w: str, duration_penalty: float = BASE_DURATION_PENALTY) -> pd.DataFrame:
    rows = []
    for _, row in case_df.iterrows():
        names = row['events']
        if target not in names:
            continue
        target_idx = names.index(target)
        before = names[:target_idx]
        iu = before.index(u) if u in before else None
        iw = before.index(w) if w in before else None

        endpoint = 'none'
        path = 'none'
        order = None
        if iu is not None and iw is None:
            endpoint = 'u'; path = 'u'
        elif iu is None and iw is not None:
            endpoint = 'w'; path = 'w'
        elif iu is not None and iw is not None:
            endpoint = 'uw'
            if iu < iw:
                order = 'forward'; path = 'uw_forward'
            else:
                order = 'reverse'; path = 'uw_reverse'

        target_time = row['times'][target_idx]
        remaining_days = (row['case_end'] - target_time).total_seconds() / 86400.0
        rows.append({
            'case_id': row['case_id'],
            'target': target,
            'u': u,
            'w': w,
            'endpoint': endpoint,
            'path': path,
            'order': order,
            'final_accept': float(row['final_accept']),
            'remaining_days': remaining_days,
            'reward': compute_reward(row['final_accept'], remaining_days, duration_penalty),
            'duration_penalty': duration_penalty,
            'family_label': family_label(target, u, w),
        })
    return pd.DataFrame(rows)


def bootstrap_mean_ci(x: np.ndarray, b: int = BOOTSTRAP_B, seed: int = SEED) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(b):
        sample = rng.choice(x, size=len(x), replace=True)
        boots.append(float(np.mean(sample)))
    return float(np.mean(x)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def bootstrap_gap_ci(x: np.ndarray, y: np.ndarray, b: int = BOOTSTRAP_B, seed: int = SEED) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(b):
        xs = rng.choice(x, size=len(x), replace=True)
        ys = rng.choice(y, size=len(y), replace=True)
        boots.append(float(np.mean(ys) - np.mean(xs)))
    return float(np.mean(y) - np.mean(x)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def summarize_endpoints(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINT_ORDER:
        sub = ep[ep['endpoint'] == endpoint]
        rows.append({
            'endpoint': endpoint,
            'count': int(len(sub)),
            'reward_mean': float(sub['reward'].mean()) if len(sub) else math.nan,
            'accept_rate': float(sub['final_accept'].mean()) if len(sub) else math.nan,
            'remaining_days_mean': float(sub['remaining_days'].mean()) if len(sub) else math.nan,
        })
    return pd.DataFrame(rows)


def summarize_paths(ep: pd.DataFrame) -> pd.DataFrame:
    none_mean = float(ep.loc[ep['path'] == 'none', 'reward'].mean())
    rows = []
    for path in PATH_ORDER:
        sub = ep[ep['path'] == path]
        rows.append({
            'path': path,
            'count': int(len(sub)),
            'reward_mean': float(sub['reward'].mean()) if len(sub) else math.nan,
            'reward_gain_over_none': float(sub['reward'].mean() - none_mean) if len(sub) and not math.isnan(none_mean) else math.nan,
            'accept_rate': float(sub['final_accept'].mean()) if len(sub) else math.nan,
            'remaining_days_mean': float(sub['remaining_days'].mean()) if len(sub) else math.nan,
        })
    return pd.DataFrame(rows)


def summarize_orders(ep: pd.DataFrame) -> pd.DataFrame:
    fw = ep[ep['path'] == 'uw_forward']
    rv = ep[ep['path'] == 'uw_reverse']
    kappa, lo, hi = bootstrap_gap_ci(fw['reward'].to_numpy(), rv['reward'].to_numpy())
    acc_gap, acc_lo, acc_hi = bootstrap_gap_ci(fw['final_accept'].to_numpy(), rv['final_accept'].to_numpy())
    dur_gap, dur_lo, dur_hi = bootstrap_gap_ci(fw['remaining_days'].to_numpy(), rv['remaining_days'].to_numpy())
    return pd.DataFrame([
        {
            'metric': 'reward',
            'forward_count': int(len(fw)),
            'reverse_count': int(len(rv)),
            'forward_mean': float(fw['reward'].mean()) if len(fw) else math.nan,
            'reverse_mean': float(rv['reward'].mean()) if len(rv) else math.nan,
            'reverse_minus_forward': kappa,
            'ci_lo': lo,
            'ci_hi': hi,
        },
        {
            'metric': 'accept_rate',
            'forward_count': int(len(fw)),
            'reverse_count': int(len(rv)),
            'forward_mean': float(fw['final_accept'].mean()) if len(fw) else math.nan,
            'reverse_mean': float(rv['final_accept'].mean()) if len(rv) else math.nan,
            'reverse_minus_forward': acc_gap,
            'ci_lo': acc_lo,
            'ci_hi': acc_hi,
        },
        {
            'metric': 'remaining_days',
            'forward_count': int(len(fw)),
            'reverse_count': int(len(rv)),
            'forward_mean': float(fw['remaining_days'].mean()) if len(fw) else math.nan,
            'reverse_mean': float(rv['remaining_days'].mean()) if len(rv) else math.nan,
            'reverse_minus_forward': dur_gap,
            'ci_lo': dur_lo,
            'ci_hi': dur_hi,
        },
    ])


def path_means(ep: pd.DataFrame) -> Dict[str, float]:
    return {p: float(ep.loc[ep['path'] == p, 'reward'].mean()) for p in PATH_ORDER}


def endpoint_means(ep: pd.DataFrame) -> Dict[str, float]:
    return {e: float(ep.loc[ep['endpoint'] == e, 'reward'].mean()) for e in ENDPOINT_ORDER}


def reconstruction_table(ep: pd.DataFrame) -> pd.DataFrame:
    pm = path_means(ep)
    kappa = pm['uw_reverse'] - pm['uw_forward']
    rows = []
    for p in ['none', 'u', 'w', 'uw_forward']:
        rows.append({'path': p, 'observed_value': pm[p], 'reconstructed_value': pm[p], 'abs_diff': 0.0})
    rows.append({'path': 'uw_reverse', 'observed_value': pm['uw_reverse'], 'reconstructed_value': pm['uw_forward'] + kappa, 'abs_diff': abs(pm['uw_reverse'] - (pm['uw_forward'] + kappa))})
    return pd.DataFrame(rows)


def mobius_table(ep: pd.DataFrame) -> pd.DataFrame:
    em = endpoint_means(ep)
    phi_none = em['none']; phi_u = em['u']; phi_w = em['w']; phi_uw = em['uw']
    return pd.DataFrame({
        'ideal': ['empty', '{u}', '{w}', '{u,w}'],
        'phi_endpoint': [phi_none, phi_u, phi_w, phi_uw],
        'theta_mobius': [phi_none, phi_u - phi_none, phi_w - phi_none, phi_uw - phi_u - phi_w + phi_none],
    })


def dynamic_programming_check(ep: pd.DataFrame) -> pd.DataFrame:
    pm = path_means(ep)
    baseline = pm['none']
    gain_u = pm['u'] - baseline
    gain_w = pm['w'] - baseline
    gain_fw = pm['uw_forward'] - baseline
    gain_rv = pm['uw_reverse'] - baseline
    g_empty_u = gain_u
    g_empty_w = gain_w
    g_u_w = gain_fw - gain_u
    g_w_u = gain_rv - gain_w
    enum = {'none': 0.0, 'u': g_empty_u, 'w': g_empty_w, 'uw_forward': g_empty_u + g_u_w, 'uw_reverse': g_empty_w + g_w_u}
    u_val = max(0.0, g_u_w)
    w_val = max(0.0, g_w_u)
    dp_gain = max(0.0, g_empty_u + u_val, g_empty_w + w_val)
    exhaustive_path = max(enum, key=enum.get)
    if dp_gain == 0.0:
        dp_path = 'none'
    elif g_empty_u + u_val >= g_empty_w + w_val:
        dp_path = 'uw_forward' if u_val > 0 else 'u'
    else:
        dp_path = 'uw_reverse' if w_val > 0 else 'w'
    return pd.DataFrame([{
        'dp_argmax': dp_path,
        'exhaustive_argmax': exhaustive_path,
        'equal': dp_path == exhaustive_path,
        'best_value': baseline + dp_gain,
        'g(empty,u)': g_empty_u,
        'g(empty,w)': g_empty_w,
        'g({u},w)': g_u_w,
        'g({w},u)': g_w_u,
    }])


def stratified_split(ep: pd.DataFrame, train_frac: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    split = pd.Series(index=ep.index, dtype=object)
    for path, g in ep.groupby('path', sort=False):
        idx = list(g.index)
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            n_train = 1
        else:
            n_train = max(1, min(n - 1, int(round(train_frac * n))))
        train_idx = set(idx[:n_train])
        for i in idx:
            split.loc[i] = 'train' if i in train_idx else 'test'
    out = ep.copy()
    out['split'] = split.values
    return out


def choose_policies(train_ep: pd.DataFrame) -> pd.DataFrame:
    pm = path_means(train_ep)
    em = endpoint_means(train_ep)
    counts = {p: int((train_ep['path'] == p).sum()) for p in PATH_ORDER}

    def best_of(cands: List[str]) -> str:
        cands = [c for c in cands if not math.isnan(pm[c])]
        return max(cands, key=lambda c: pm[c])

    endpoint_to_path = {'none': 'none', 'u': 'u', 'w': 'w', 'uw': 'uw_forward'}
    endpoint_best = max([e for e in ENDPOINT_ORDER if not math.isnan(em[e])], key=lambda e: em[e])
    rows = [
        {'policy': 'sequence_sensitive', 'chosen_path': best_of(PATH_ORDER), 'train_estimate': max(v for v in pm.values() if not math.isnan(v))},
        {'policy': 'reference_path', 'chosen_path': best_of(['none', 'u', 'w', 'uw_forward']), 'train_estimate': pm[best_of(['none', 'u', 'w', 'uw_forward'])]},
        {'policy': 'greedy_one_step', 'chosen_path': best_of(['none', 'u', 'w']), 'train_estimate': pm[best_of(['none', 'u', 'w'])]},
        {'policy': 'fixed_forward', 'chosen_path': 'uw_forward', 'train_estimate': pm['uw_forward']},
        {'policy': 'fixed_reverse', 'chosen_path': 'uw_reverse', 'train_estimate': pm['uw_reverse']},
        {'policy': 'endpoint_pooled', 'chosen_path': endpoint_to_path[endpoint_best], 'train_estimate': pm[endpoint_to_path[endpoint_best]]},
        {'policy': 'frequency', 'chosen_path': max(PATH_ORDER, key=lambda p: counts[p]), 'train_estimate': pm[max(PATH_ORDER, key=lambda p: counts[p])]},
    ]
    return pd.DataFrame(rows)


def evaluate_policies(policy_df: pd.DataFrame, test_ep: pd.DataFrame) -> pd.DataFrame:
    pm_test = test_ep.groupby('path', sort=False)['reward'].mean().to_dict()
    counts_test = test_ep.groupby('path', sort=False)['reward'].size().to_dict()
    out = policy_df.copy()
    out['test_value'] = [float(pm_test.get(p, math.nan)) for p in out['chosen_path']]
    out['test_path_count'] = [int(counts_test.get(p, 0)) for p in out['chosen_path']]
    return out


def repeated_policy_eval(ep: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    for split_seed in range(SEED, SEED + N_SPLITS):
        split_ep = stratified_split(ep, TRAIN_FRAC, split_seed)
        train_ep = split_ep[split_ep['split'] == 'train'].copy()
        test_ep = split_ep[split_ep['split'] == 'test'].copy()
        pol_train = choose_policies(train_ep)
        pol_eval = evaluate_policies(pol_train, test_ep)
        pol_eval['split_seed'] = split_seed
        all_rows.append(pol_eval)
    split_df = pd.concat(all_rows, ignore_index=True)
    ref = split_df[split_df['policy'] == 'reference_path'][['split_seed', 'test_value']].rename(columns={'test_value': 'ref_val'})
    greedy = split_df[split_df['policy'] == 'greedy_one_step'][['split_seed', 'test_value']].rename(columns={'test_value': 'greedy_val'})
    merged = split_df.merge(ref, on='split_seed').merge(greedy, on='split_seed')
    rows = []
    for pol, g in merged.groupby('policy', sort=False):
        test_mean, lo, hi = bootstrap_mean_ci(g['test_value'].to_numpy())
        gain_ref, gain_ref_lo, gain_ref_hi = bootstrap_mean_ci((g['test_value'] - g['ref_val']).to_numpy())
        gain_g, gain_g_lo, gain_g_hi = bootstrap_mean_ci((g['test_value'] - g['greedy_val']).to_numpy())
        mode = g['chosen_path'].mode().iloc[0]
        rows.append({
            'policy': pol,
            'chosen_path_mode': mode,
            'chosen_path_mode_freq': float((g['chosen_path'] == mode).mean()),
            'mean_test_value': test_mean,
            'test_value_ci_lo': lo,
            'test_value_ci_hi': hi,
            'mean_gain_vs_reference': gain_ref,
            'gain_vs_reference_ci_lo': gain_ref_lo,
            'gain_vs_reference_ci_hi': gain_ref_hi,
            'mean_gain_vs_greedy': gain_g,
            'gain_vs_greedy_ci_lo': gain_g_lo,
            'gain_vs_greedy_ci_hi': gain_g_hi,
            'win_rate_vs_reference': float((g['test_value'] > g['ref_val']).mean()),
            'win_rate_vs_greedy': float((g['test_value'] > g['greedy_val']).mean()),
        })
    return split_df, pd.DataFrame(rows)


def family_support_row(ep: pd.DataFrame) -> Dict[str, object]:
    fw = ep[ep['path'] == 'uw_forward']
    rv = ep[ep['path'] == 'uw_reverse']
    kappa, lo, hi = bootstrap_gap_ci(fw['reward'].to_numpy(), rv['reward'].to_numpy())
    return {
        'family_label': ep['family_label'].iloc[0],
        'target': ep['target'].iloc[0],
        'u': ep['u'].iloc[0],
        'w': ep['w'].iloc[0],
        'n_cases': int(len(ep)),
        'endpoint_none': int((ep['endpoint'] == 'none').sum()),
        'endpoint_u': int((ep['endpoint'] == 'u').sum()),
        'endpoint_w': int((ep['endpoint'] == 'w').sum()),
        'endpoint_uw': int((ep['endpoint'] == 'uw').sum()),
        'forward_count': int(len(fw)),
        'reverse_count': int(len(rv)),
        'min_order_support': int(min(len(fw), len(rv))),
        'reference_path_score': float(fw['reward'].mean()) if len(fw) else math.nan,
        'kappa_reverse_minus_forward': kappa,
        'kappa_ci_lo': lo,
        'kappa_ci_hi': hi,
    }


def lambda_sweep(case_df: pd.DataFrame, families: List[Tuple[str, str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_kappa, rows_best = [], []
    for lam in LAMBDA_GRID:
        for target, u, w in families:
            ep = build_family_episodes(case_df, target, u, w, duration_penalty=lam)
            label = family_label(target, u, w)
            pm = summarize_paths(ep).sort_values('reward_mean', ascending=False)
            best_path = pm['path'].iloc[0]
            rows_kappa.append({'family_label': label, 'lambda': lam, 'kappa': float(ep.loc[ep['path'] == 'uw_reverse', 'reward'].mean() - ep.loc[ep['path'] == 'uw_forward', 'reward'].mean())})
            rows_best.append({'family_label': label, 'lambda': lam, 'best_path': best_path})
    return pd.DataFrame(rows_kappa), pd.DataFrame(rows_best)


def save_df(df: pd.DataFrame, csv_path: str, tex_path: str = None, index: bool = False):
    df.to_csv(csv_path, index=index)
    if tex_path is not None:
        df.to_latex(tex_path, index=index, escape=False, float_format=lambda x: f'{x:.3f}')


def plot_structure_overview(support_df: pd.DataFrame, path: str):
    x = np.arange(len(support_df))
    labels = [f"{short_name(t)}\n({short_name(u)},{short_name(w)})" for t, u, w in support_df[['target', 'u', 'w']].itertuples(index=False, name=None)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1.25, 1.0, 1.15]})
    bottoms = np.zeros(len(support_df))
    for col, lab in [('endpoint_none', 'none'), ('endpoint_u', 'u'), ('endpoint_w', 'w'), ('endpoint_uw', 'uw')]:
        vals = support_df[col].to_numpy()
        axes[0].bar(x, vals, bottom=bottoms, color=COLORS[lab], label=lab)
        bottoms += vals
    axes[0].set_title('Endpoint support')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels); axes[0].set_ylabel('Cases')
    axes[0].legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18))
    width = 0.35
    axes[1].bar(x - width/2, support_df['forward_count'], width=width, color=COLORS['forward'], label='forward')
    axes[1].bar(x + width/2, support_df['reverse_count'], width=width, color=COLORS['reverse'], label='reverse')
    axes[1].set_title('Two-sided diamond support')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels); axes[1].set_ylabel('Cases')
    axes[1].legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.12))
    y = support_df['kappa_reverse_minus_forward'].to_numpy()
    lo = y - support_df['kappa_ci_lo'].to_numpy(); hi = support_df['kappa_ci_hi'].to_numpy() - y
    axes[2].errorbar(x, y, yerr=[lo, hi], fmt='o', ecolor='#444', capsize=4, color='#222')
    axes[2].scatter(x, y, s=55, c=[COLORS['reverse'] if v > 0 else COLORS['forward'] for v in y])
    axes[2].axhline(0.0, color='#666', linewidth=1)
    axes[2].set_title(r'Local order effect $\hat\kappa$')
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels); axes[2].set_ylabel('reverse - forward')
    fig.suptitle('Support separation and order heterogeneity across selected families', y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_decomposition(ep_pos: pd.DataFrame, ep_neg: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, ep in [(axes[0], ep_pos), (axes[1], ep_neg)]:
        title = family_label(ep['target'].iloc[0], ep['u'].iloc[0], ep['w'].iloc[0])
        path_tab = summarize_paths(ep)
        recon = reconstruction_table(ep)
        merged = path_tab.merge(recon[['path', 'reconstructed_value']], on='path')
        baseline = float(merged.loc[merged['path'] == 'none', 'reward_mean'].iloc[0])
        merged['observed_gain'] = merged['reward_mean'] - baseline
        merged['reconstructed_gain'] = merged['reconstructed_value'] - baseline
        x = np.arange(len(PATH_ORDER)); width = 0.36
        ax.bar(x - width/2, merged['observed_gain'], width=width, color='#4C78A8', label='Observed')
        ax.bar(x + width/2, merged['reconstructed_gain'], width=width, color='#F2CF5B', edgecolor='#333', hatch='//', label='Reconstructed')
        ax.axhline(0.0, color='#666', linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(PATH_ORDER, rotation=15)
        ax.set_ylabel('Gain over none')
        ax.set_title(title + '\n' + rf'$\hat{{\kappa}}={float(recon.iloc[-1]["reconstructed_value"] - recon.iloc[3]["reconstructed_value"]):.2f}$')
        max_err = float(recon['abs_diff'].max())
        ax.text(0.98, 0.04, f'max abs. error = {max_err:.3g}', transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc'))
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle('Exact reference-path decomposition on representative positive and negative families', y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_policy_heatmap(policy_summary_df: pd.DataFrame, path: str):
    fams = policy_summary_df['family_label'].drop_duplicates().tolist()
    labels = [f"{short_name(t)} ({short_name(u)},{short_name(w)})" for t, u, w in policy_summary_df[['target', 'u', 'w']].drop_duplicates().itertuples(index=False, name=None)]
    mat = []
    for fam in fams:
        sub = policy_summary_df[policy_summary_df['family_label'] == fam].set_index('policy')
        mat.append([sub.loc[p, 'mean_test_value'] for p in POLICY_ORDER])
    arr = np.array(mat)
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    im = ax.imshow(arr, aspect='auto', cmap='viridis')
    ax.set_yticks(np.arange(len(fams))); ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(POLICY_ORDER))); ax.set_xticklabels([p.replace('_', '\n') for p in POLICY_ORDER])
    threshold = float(np.nanmean(arr))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f'{arr[i,j]:.1f}', ha='center', va='center', color='white' if arr[i,j] < threshold else 'black', fontsize=8)
    ax.set_title('Held-out path-conditional value across policies and families')
    cbar = fig.colorbar(im, ax=ax); cbar.set_label('Mean held-out value')
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_policy_gains(policy_summary_df: pd.DataFrame, path: str):
    seq = policy_summary_df[policy_summary_df['policy'] == 'sequence_sensitive'].copy()
    labels = [f"{short_name(t)}\n({short_name(u)},{short_name(w)})" for t, u, w in seq[['target', 'u', 'w']].itertuples(index=False, name=None)]
    x = np.arange(len(seq)); width = 0.35
    lo_ref = seq['mean_gain_vs_reference'] - seq['gain_vs_reference_ci_lo']
    hi_ref = seq['gain_vs_reference_ci_hi'] - seq['mean_gain_vs_reference']
    lo_g = seq['mean_gain_vs_greedy'] - seq['gain_vs_greedy_ci_lo']
    hi_g = seq['gain_vs_greedy_ci_hi'] - seq['mean_gain_vs_greedy']
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width/2, seq['mean_gain_vs_reference'], width=width, color=COLORS['sequence_sensitive'], yerr=[lo_ref, hi_ref], capsize=4, label='vs reference')
    ax.bar(x + width/2, seq['mean_gain_vs_greedy'], width=width, color=COLORS['greedy_one_step'], yerr=[lo_g, hi_g], capsize=4, label='vs greedy')
    ax.axhline(0.0, color='#666', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Mean held-out value gain')
    ax.set_title('Sequence-sensitive planning gains are selective, not universal')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_lambda_sweep(kappa_df: pd.DataFrame, best_df: pd.DataFrame, path: str):
    fams = kappa_df['family_label'].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [1.0, 1.2]})
    piv = kappa_df.pivot(index='family_label', columns='lambda', values='kappa').loc[fams]
    arr = piv.to_numpy()
    im = axes[0].imshow(arr, aspect='auto', cmap='coolwarm')
    axes[0].set_yticks(np.arange(len(fams))); axes[0].set_yticklabels([f.replace(' <- ', '\n').replace('{', '').replace('}', '') for f in fams])
    axes[0].set_xticks(np.arange(len(LAMBDA_GRID))); axes[0].set_xticklabels([str(x) for x in LAMBDA_GRID])
    axes[0].set_xlabel(r'Duration penalty $\lambda$'); axes[0].set_title(r'Order effect $\hat\kappa(\lambda)$')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            axes[0].text(j, i, f'{arr[i,j]:.1f}', ha='center', va='center', fontsize=8)
    cbar = fig.colorbar(im, ax=axes[0]); cbar.set_label('reverse - forward')
    cat_map = {'none': 0, 'u': 1, 'w': 2, 'uw_forward': 3, 'uw_reverse': 4}; inv = {v: k for k, v in cat_map.items()}
    pivb = best_df.pivot(index='family_label', columns='lambda', values='best_path').loc[fams]
    arrb = pivb.replace(cat_map).to_numpy(dtype=float)
    axes[1].imshow(arrb, aspect='auto', cmap='Set3', vmin=-0.5, vmax=4.5)
    axes[1].set_yticks(np.arange(len(fams))); axes[1].set_yticklabels([f.replace(' <- ', '\n').replace('{', '').replace('}', '') for f in fams])
    axes[1].set_xticks(np.arange(len(LAMBDA_GRID))); axes[1].set_xticklabels([str(x) for x in LAMBDA_GRID])
    axes[1].set_xlabel(r'Duration penalty $\lambda$'); axes[1].set_title('Best pooled path under reward sweep')
    for i in range(arrb.shape[0]):
        for j in range(arrb.shape[1]):
            axes[1].text(j, i, inv[int(arrb[i,j])].replace('_', '\n'), ha='center', va='center', fontsize=7)
    legend_handles = [Patch(facecolor=plt.get_cmap('Set3')(cat_map[p] / 4.0), label=p) for p in cat_map]
    axes[1].legend(handles=legend_handles, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=5)
    fig.suptitle('Reward-sensitivity diagnostics', y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_log(DATA_PATH)
    case_df = build_case_table(df)
    dataset_df = pd.DataFrame([{
        'dataset': 'reviewing',
        'n_cases': int(df['case:concept:name'].nunique()),
        'n_events': int(len(df)),
        'n_activities': int(df['concept:name'].nunique()),
    }])
    save_df(dataset_df, os.path.join(OUT_DIR, 'table_dataset_summary.csv'), os.path.join(OUT_DIR, 'table_dataset_summary.tex'))

    families = [(t, u, w) for t in TARGETS for u, w in combinations(REVIEWS, 2)]
    support_rows = []
    path_rows = []
    policy_rows = []
    policy_split_rows = []
    dp_rows = []

    ep_pos = None
    ep_neg = None
    for fam in families:
        target, u, w = fam
        label = family_label(target, u, w)
        slug = family_slug(target, u, w)
        fam_dir = os.path.join(OUT_DIR, slug)
        os.makedirs(fam_dir, exist_ok=True)
        ep = build_family_episodes(case_df, target, u, w)
        ep.to_csv(os.path.join(fam_dir, 'episodes.csv'), index=False)
        endpoint_df = summarize_endpoints(ep)
        path_df = summarize_paths(ep)
        order_df = summarize_orders(ep)
        recon_df = reconstruction_table(ep)
        mobius_df = mobius_table(ep)
        dp_df = dynamic_programming_check(ep)
        split_df, policy_df = repeated_policy_eval(ep)
        split_df['family_label'] = label
        split_df['target'] = target
        split_df['u'] = u
        split_df['w'] = w
        policy_df['family_label'] = label
        policy_df['target'] = target
        policy_df['u'] = u
        policy_df['w'] = w
        save_df(endpoint_df, os.path.join(fam_dir, 'table_endpoint_summary.csv'), os.path.join(fam_dir, 'table_endpoint_summary.tex'))
        save_df(path_df, os.path.join(fam_dir, 'table_path_values.csv'), os.path.join(fam_dir, 'table_path_values.tex'))
        save_df(order_df, os.path.join(fam_dir, 'table_order_summary.csv'), os.path.join(fam_dir, 'table_order_summary.tex'))
        save_df(recon_df, os.path.join(fam_dir, 'table_reconstruction_check.csv'), os.path.join(fam_dir, 'table_reconstruction_check.tex'))
        save_df(mobius_df, os.path.join(fam_dir, 'table_mobius_endpoint.csv'), os.path.join(fam_dir, 'table_mobius_endpoint.tex'))
        save_df(dp_df, os.path.join(fam_dir, 'table_dp_check.csv'), os.path.join(fam_dir, 'table_dp_check.tex'))
        save_df(policy_df, os.path.join(fam_dir, 'table_policy_comparison.csv'), os.path.join(fam_dir, 'table_policy_comparison.tex'))
        split_df.to_csv(os.path.join(fam_dir, 'policy_split_results.csv'), index=False)

        support_rows.append(family_support_row(ep))
        path_df = path_df.copy(); path_df['family_label'] = label; path_rows.append(path_df)
        policy_rows.append(policy_df)
        policy_split_rows.append(split_df)
        dp_rows.append({
            'family_label': label,
            'target': target,
            'u': u,
            'w': w,
            'dp_argmax': dp_df.loc[0, 'dp_argmax'],
            'exhaustive_argmax': dp_df.loc[0, 'exhaustive_argmax'],
            'equal': bool(dp_df.loc[0, 'equal']),
            'best_value': float(dp_df.loc[0, 'best_value']),
        })
        if fam == ('decide', 'get review 1', 'get review 2'):
            ep_pos = ep.copy()
        if fam == ('decide', 'get review 2', 'get review 3'):
            ep_neg = ep.copy()

    support_df = pd.DataFrame(support_rows)
    path_all_df = pd.concat(path_rows, ignore_index=True)
    policy_summary_df = pd.concat(policy_rows, ignore_index=True)
    policy_splits_df = pd.concat(policy_split_rows, ignore_index=True)
    dp_summary_df = pd.DataFrame(dp_rows)
    lambda_kappa_df, lambda_best_df = lambda_sweep(case_df, families)

    save_df(support_df, os.path.join(OUT_DIR, 'table_family_support_summary.csv'), os.path.join(OUT_DIR, 'table_family_support_summary.tex'))
    save_df(path_all_df, os.path.join(OUT_DIR, 'table_path_values_across_families.csv'), os.path.join(OUT_DIR, 'table_path_values_across_families.tex'))
    save_df(policy_summary_df, os.path.join(OUT_DIR, 'table_policy_summary_across_families.csv'), os.path.join(OUT_DIR, 'table_policy_summary_across_families.tex'))
    policy_splits_df.to_csv(os.path.join(OUT_DIR, 'table_policy_splits_long.csv'), index=False)
    save_df(dp_summary_df, os.path.join(OUT_DIR, 'table_dp_exhaustive_summary.csv'), os.path.join(OUT_DIR, 'table_dp_exhaustive_summary.tex'))
    save_df(lambda_kappa_df, os.path.join(OUT_DIR, 'table_lambda_sweep_kappa.csv'), os.path.join(OUT_DIR, 'table_lambda_sweep_kappa.tex'))
    save_df(lambda_best_df, os.path.join(OUT_DIR, 'table_lambda_sweep_bestpath.csv'), os.path.join(OUT_DIR, 'table_lambda_sweep_bestpath.tex'))

    compact_support = support_df[['family_label', 'endpoint_none', 'endpoint_u', 'endpoint_w', 'endpoint_uw', 'forward_count', 'reverse_count', 'kappa_reverse_minus_forward', 'kappa_ci_lo', 'kappa_ci_hi']].copy()
    compact_support['endpoint_counts'] = compact_support.apply(lambda r: f"({int(r['endpoint_none'])},{int(r['endpoint_u'])},{int(r['endpoint_w'])},{int(r['endpoint_uw'])})", axis=1)
    compact_support['order_counts'] = compact_support.apply(lambda r: f"{int(r['forward_count'])}/{int(r['reverse_count'])}", axis=1)
    compact_support['kappa_ci'] = compact_support.apply(lambda r: f"{r['kappa_reverse_minus_forward']:.2f} [{r['kappa_ci_lo']:.2f}, {r['kappa_ci_hi']:.2f}]", axis=1)
    compact_support = compact_support[['family_label', 'endpoint_counts', 'order_counts', 'kappa_ci']]
    compact_support.to_csv(os.path.join(OUT_DIR, 'table_main_family_summary.csv'), index=False)
    compact_support.to_latex(os.path.join(OUT_DIR, 'table_main_family_summary.tex'), index=False, escape=False)

    seq_rows = policy_summary_df[policy_summary_df['policy'] == 'sequence_sensitive'][['family_label', 'chosen_path_mode', 'chosen_path_mode_freq', 'mean_test_value', 'mean_gain_vs_reference', 'mean_gain_vs_greedy', 'win_rate_vs_reference']].copy()
    seq_rows['chosen_path'] = seq_rows.apply(lambda r: f"{r['chosen_path_mode']} ({r['chosen_path_mode_freq']:.2f})", axis=1)
    seq_rows['gain_vs_reference'] = seq_rows['mean_gain_vs_reference'].map(lambda x: f"{x:.2f}")
    seq_rows['gain_vs_greedy'] = seq_rows['mean_gain_vs_greedy'].map(lambda x: f"{x:.2f}")
    seq_rows['win_rate'] = seq_rows['win_rate_vs_reference'].map(lambda x: f"{x:.2f}")
    seq_rows = seq_rows[['family_label', 'chosen_path', 'mean_test_value', 'gain_vs_reference', 'gain_vs_greedy', 'win_rate']]
    seq_rows.to_csv(os.path.join(OUT_DIR, 'table_main_policy_summary.csv'), index=False)
    seq_rows.to_latex(os.path.join(OUT_DIR, 'table_main_policy_summary.tex'), index=False, escape=False)

    plot_structure_overview(support_df, os.path.join(OUT_DIR, 'fig_structure_overview.png'))
    plot_decomposition(ep_pos, ep_neg, os.path.join(OUT_DIR, 'fig_decomposition_examples.png'))
    plot_policy_heatmap(policy_summary_df, os.path.join(OUT_DIR, 'fig_policy_heatmap.png'))
    plot_policy_gains(policy_summary_df, os.path.join(OUT_DIR, 'fig_policy_gains.png'))
    plot_lambda_sweep(lambda_kappa_df, lambda_best_df, os.path.join(OUT_DIR, 'fig_lambda_sweep.png'))

    with open(os.path.join(OUT_DIR, 'empirical_report.md'), 'w', encoding='utf-8') as f:
        f.write('# Closing empirical suite on reviewing.csv\n\n')
        f.write('## Family support summary\n\n')
        f.write(compact_support.to_markdown(index=False))
        f.write('\n\n## Sequence-sensitive planning summary\n\n')
        f.write(seq_rows.to_markdown(index=False))
        f.write('\n\n## DP vs exhaustive\n\n')
        f.write(dp_summary_df.to_markdown(index=False))

    print('done', OUT_DIR)
    print(compact_support.to_string(index=False))
    print(seq_rows.to_string(index=False))
    print(dp_summary_df.to_string(index=False))

if __name__ == '__main__':
    main()
