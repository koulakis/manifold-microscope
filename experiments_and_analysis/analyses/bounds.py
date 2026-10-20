"""Paper power-law aggregation, independent of result storage."""
from collections import defaultdict

def _curve(rows, dimension):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    grouped = defaultdict(list)
    for row in rows:
        grouped[row['number_of_train_points']].append(row['hausdorff_distance'])
    x = np.array(sorted(grouped), dtype=float)
    if len(x) < 2:
        raise ValueError('At least two completed training sizes are required for each bound curve.')
    repetitions = [grouped[int(n)] for n in x]
    empirical = np.array([np.mean(values) for values in repetitions])
    if not np.isfinite(empirical).all() or np.any(empirical <= 0) or np.any(x <= 1):
        raise ValueError('Log-log bounds require finite positive errors and training counts > 1.')
    regression = LinearRegression().fit(np.log(x).reshape(-1, 1), np.log(empirical))
    fitted = np.exp(regression.intercept_) * x ** regression.coef_[0]
    lower = (1 / x) ** (2 / (2 + dimension))
    upper = (np.log(x) / x) ** (2 / (2 + dimension))
    fefferman = (np.log(x) / x) ** (1 / dimension)
    # These are the active notebook formulas; its unused reach loads are omitted.
    exported = dict(x=x, empirical=empirical, empirical_all_repetitions=repetitions,
                    lower_genovese=np.quantile(fitted / lower, 0.01) * lower,
                    upper_genovese=np.quantile(fitted / upper, 0.99) * upper,
                    upper_fefferman=np.quantile(fitted / fefferman, 0.99) * fefferman)
    summary = dict(dimension=dimension, training_counts=x.tolist(),
                   repetition_counts=list(map(len, repetitions)),
                   alpha=float(-regression.coef_[0]), coefficient=float(np.exp(regression.intercept_)))
    return exported, summary

