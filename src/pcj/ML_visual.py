# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")


# %%
def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


# %%
def plot_pred_vs_actual(y_true, y_pred, title=None, figsize=(6,6)):
    plt.figure(figsize=figsize)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
    plt.xlabel('Actual'); plt.ylabel('Predicted')
    if title: plt.title(title)
    plt.show()


# %%
def plot_residuals(y_true, y_pred, title=None, figsize=(6,4)):
    residuals = y_true - y_pred
    plt.figure(figsize=figsize)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Predicted'); plt.ylabel('Residual (Actual - Pred)')
    if title: plt.title(title)
    plt.show()


# %%
def plot_residual_dist(y_true, y_pred, title=None, figsize=(6,4)):
    residuals = y_true - y_pred
    plt.figure(figsize=figsize)
    sns.histplot(residuals, kde=True)
    if title: plt.title(title)
    plt.show()


# %%
def permutation_importance_plot(model, X, y, feature_names=None, n_repeats=30, top_n=10, figsize=(8,4), random_state=0):
    perm = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    names = feature_names if feature_names is not None else X.columns
    imp = pd.Series(perm.importances_mean, index=names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=figsize)
    sns.barplot(x=imp.values, y=imp.index, palette='viridis')
    plt.xlabel('Mean permutation importance'); plt.title('Permutation importance (top {})'.format(len(imp)))
    plt.show()
    return imp


# %%
def pca_outlier_plot(X, labels, title='PCA (2D) + labels', scaler=None, figsize=(7,5)):
    # X: DataFrame or ndarray; labels: array-like (e.g. pred from IsolationForest)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    labels = np.asarray(labels)
    plt.figure(figsize=figsize)
    mask_pos = labels > 0
    plt.scatter(Z[mask_pos,0], Z[mask_pos,1], s=10, label='normal', alpha=0.6)
    plt.scatter(Z[~mask_pos,0], Z[~mask_pos,1], s=25, color='r', label='outlier', alpha=0.9)
    plt.legend(); plt.title(title); plt.show()


# %%
def evaluate_regressor(model, X_test, y_test, feature_names=None, target_name=None, plot=True):
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    result = {
        'model': model, 'x_test': X_test, 'y_test': y_test, 'y_pred': y_pred,
        'metrics': metrics, 'feature_names': feature_names, 'target': target_name
    }
    if plot:
        title = f"{target_name} — R²={metrics['r2']:.3f}" if target_name else None
        plot_pred_vs_actual(y_test, y_pred, title=title)
        plot_residuals(y_test, y_pred, title=title)
        plot_residual_dist(y_test, y_pred, title=title)
        if feature_names is not None:
            try:
                permutation_importance_plot(model, X_test, y_test, feature_names=feature_names)
            except Exception:
                pass
    return result
