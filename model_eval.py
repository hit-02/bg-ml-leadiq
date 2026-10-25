"""
Lead IQ model evaluation -- reusable functions for scoring model V1.

Mirrors the recommendation system's model_eval.py: evaluation logic lives
here as importable functions, not duplicated inline in every notebook that
needs it. Called from model_train_deploy.ipynb, right after training --
never from model_inference.py, since every function here needs a labeled,
multi-row test set to compute anything (lift and AUC are meaningless for
the single, unlabeled lead predict() scores).

3 functions, each pulled directly from what model_train_deploy.ipynb's
segment_breakdown cell and model_eval.ipynb already computed inline:
  - lift_at_k                      : the single-lift building block
  - lift_auc_metrics               : lift@1/5/10/20%, ROC-AUC, PR-AUC
  - eval_plots_and_decile_lift     : the 4-panel eval figure + decile chart
  - hot_threshold_confusion_matrix : top-10% "HOT" threshold precision/recall
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)


def lift_at_k(y, p, k):
    """Conversion rate in the top k% of scored leads, divided by the overall
    base rate. lift_at_k(y, p, 0.10) == 2.48 means the top 10% of leads by
    score convert at 2.48x the base rate."""
    n = max(int(len(y) * k), 1)
    return y[np.argsort(-p)[:n]].mean() / y.mean()


def lift_auc_metrics(y, p, ks=(0.01, 0.05, 0.10, 0.20)):
    """Prints lift@k for each k in ks, plus ROC-AUC and PR-AUC. Returns the
    same numbers as a dict so callers can log/assert on them instead of
    only reading the printout."""
    metrics = {f'lift@{int(k*100)}%': lift_at_k(y, p, k) for k in ks}
    metrics['roc_auc'] = roc_auc_score(y, p)
    metrics['pr_auc'] = average_precision_score(y, p)
    metrics['base_rate'] = y.mean()

    for k in ks:
        print(f"lift@{int(k*100)}%: {metrics[f'lift@{int(k*100)}%']:.2f}")
    print(f"ROC-AUC {metrics['roc_auc']:.4f}   PR-AUC {metrics['pr_auc']:.4f}   (base {metrics['base_rate']:.4f})")
    return metrics


def eval_plots_and_decile_lift(y, p, curves_path='eval_curves.png', decile_path='decile_lift.png'):
    """4-panel figure (cumulative gains, ROC, PR curve, score distribution
    by class) plus the decile lift bar chart. Saves both figures to disk
    and returns the decile lift table (dl) so callers can inspect or log
    the numbers, not just eyeball the plot."""
    order = np.argsort(-p); y_sorted = y[order]
    frac = np.arange(1, len(y)+1) / len(y)
    gains = np.cumsum(y_sorted) / y.sum()

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax[0,0].plot(frac, gains); ax[0,0].plot([0,1],[0,1],'--k')
    ax[0,0].set(title='Cumulative gains (lift chart)', xlabel='% leads contacted (by score)', ylabel='% converters captured')
    fpr, tpr, _ = roc_curve(y, p); ax[0,1].plot(fpr, tpr); ax[0,1].plot([0,1],[0,1],'--k')
    ax[0,1].set(title=f'ROC (AUC={roc_auc_score(y,p):.3f})')
    prec, rec, _ = precision_recall_curve(y, p); ax[1,0].plot(rec, prec)
    ax[1,0].axhline(y.mean(), ls='--', c='k'); ax[1,0].set(title='PR curve (dashed = base rate)')
    ax[1,1].hist(p[y==0], bins=50, alpha=.5, density=True, label='neg')
    ax[1,1].hist(p[y==1], bins=50, alpha=.5, density=True, label='pos'); ax[1,1].legend()
    ax[1,1].set(title='Score distribution by class')
    plt.tight_layout(); plt.savefig(curves_path, dpi=150); plt.show()

    te_d = pd.DataFrame({'y': y, 'p': p})
    te_d['decile'] = 10 - pd.qcut(te_d['p'].rank(method='first'), 10, labels=False)

    dl = te_d.groupby('decile')['y'].agg(pos_rate='mean', n='count')
    dl['lift'] = dl['pos_rate'] / te_d['y'].mean()
    dl['cum_lift'] = (te_d.sort_values('p', ascending=False)['y']
                      .expanding().mean()
                      .iloc[[int(len(te_d)*k/10)-1 for k in range(1,11)]]
                      .values / te_d['y'].mean())

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(dl.index, dl['lift'], color=['#c0392b' if d==1 else '#e67e22' if d<=2 else '#3498db' for d in dl.index])
    ax.plot(dl.index, dl['cum_lift'], 'ko--', label='cumulative lift', ms=5)
    ax.axhline(1.0, ls=':', c='gray', label='random (1.0x)')
    for d, v in dl['lift'].items():
        ax.text(d, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)
    ax.set(xlabel='Score decile (1 = highest 10%)', ylabel='Lift vs base rate',
           title=f'Decile lift \u2014 person-level target (test n={len(te_d):,}, base {te_d.y.mean():.2%})')
    ax.set_xticks(range(1, 11)); ax.legend()
    plt.tight_layout(); plt.savefig(decile_path, dpi=150); plt.show()

    print(dl.round(3).to_string())
    return dl


def hot_threshold_confusion_matrix(y, p, quantile=0.90):
    """Precision/recall of the top (1-quantile) share of leads by score --
    e.g. quantile=0.90 evaluates the top 10% as if they were flagged HOT.
    Returns the confusion-matrix counts and the two headline rates as a
    dict, in addition to printing them."""
    thr = np.quantile(p, quantile)
    pred_hot = (p >= thr).astype(int)
    tp = ((pred_hot==1)&(y==1)).sum(); fp = ((pred_hot==1)&(y==0)).sum()
    fn = ((pred_hot==0)&(y==1)).sum(); tn = ((pred_hot==0)&(y==0)).sum()

    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')

    print(f'operational threshold = top {int((1-quantile)*100)}% (score >= {thr:.4f})')
    print(pd.DataFrame([[tp, fn], [fp, tn]], index=['actual_pos', 'actual_neg'], columns=['pred_HOT', 'pred_not']))
    print(f'HOT precision {precision:.3%}  |  recall of converters in HOT {recall:.3%}')

    return {'threshold': thr, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall}

# add to model_eval.py, after hot_threshold_confusion_matrix

# PRD 3.2 tier boundaries -- must stay in sync with TIER_BINS/TIER_LABELS
# in model_inference.py. Duplicated here (not imported) so this file has
# no dependency on mlflow/snowflake.ml being importable just to eval a
# trained model.
_TIER_BINS = [0, 39, 74, 100]
_TIER_LABELS = ['Cold', 'Warm', 'Hot']


def build_scored_output(te, p_te, id_cols=('LEAD_ID', 'LEAD_TS', 'LABEL')):
    """Map this run's test-set raw scores -> 1-100 percentile -> PRD tier.
    Returns a DataFrame with id_cols plus raw_score, score_1_100, tier."""
    scored = te[list(id_cols)].copy()
    scored['raw_score'] = p_te
    scored['score_1_100'] = (
        scored['raw_score'].rank(pct=True) * 99 + 1
    ).round(0).astype(int)
    scored['tier'] = pd.cut(scored['score_1_100'], bins=_TIER_BINS, labels=_TIER_LABELS)
    return scored


def write_scored_table(session, scored, ts, database='AWS_DP_PROD', schema='ML_OPS', role='DATA_SCIENTIST'):
    """Write this run's scored output to a permanent, timestamped table --
    no overwritten 'latest' alias, so every run's scored output stays
    queryable by its own table name and nothing here is ever destroyed by
    a later run. ts must be the same timestamp used for this run's model
    snapshot (from train_model_and_snapshot), not recomputed here."""
    session.sql(f'USE ROLE {role}').collect()
    scored_ts_table = f'LEADIQ_SCORED_V1_{ts}'
    session.write_pandas(
        scored,
        table_name=scored_ts_table,
        database=database,
        schema=schema,
        auto_create_table=True,
        overwrite=False,
    )
    print(f'permanent: {database}.{schema}.{scored_ts_table}')
    return scored_ts_table

def feature_importance_plot(model, features, top_n=15, plot_path='importance.png'):
    """Top-N feature importances (gain) as a horizontal bar chart. Returns
    the full importance Series (all features, sorted desc) so callers can
    inspect or log the numbers beyond just the plot."""
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(imp.round(4).to_string())
    imp.head(top_n).iloc[::-1].plot.barh(figsize=(8, 6), title=f'Top {top_n} feature importance (gain)')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.show()
    return imp


def segment_lift_table(te, p_te, src_cols, min_n=5000):
    """Lift/AUC broken out by pipeline segment (inferred from the one-hot
    SRC_* column that's set for each row). Segments with fewer than min_n
    rows are dropped -- too small to trust a per-segment AUC/lift on."""
    te_seg = te.copy()
    te_seg['score'] = p_te
    te_seg['pipeline'] = te_seg[src_cols].idxmax(axis=1).str.replace('SRC_', '')

    rows = []
    for seg, g in te_seg.groupby('pipeline'):
        if len(g) < min_n:
            continue
        r = {'pipeline': seg, 'n': len(g), 'base_rate': g['LABEL'].mean()}
        r['lift@10%_within'] = lift_at_k(g['LABEL'].values, g['score'].values, 0.10)
        r['auc'] = roc_auc_score(g['LABEL'], g['score']) if g['LABEL'].nunique() > 1 else np.nan
        rows.append(r)
    seg_table = pd.DataFrame(rows).sort_values('n', ascending=False)
    print(seg_table.round(3).to_string(index=False))
    return seg_table