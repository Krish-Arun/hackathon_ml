"""
Subsampling for the model comparison.

WHY SUBSAMPLE AT ALL:
The full table is 2.1M rows. KNN and Random Forest on that, five times over
for cross-validation, would take far longer than the time we have. So every
model is compared on the SAME subsample - identical budget, identical rows -
which keeps the comparison fair even though it is not the full dataset. The
selected model is refit on everything in section 9.

WHY WE SAMPLE OBSERVATIONS, NOT ROWS:
Top-1 accuracy asks "of this observation's candidates, did we rank the right
one first?". That question only makes sense if the observation's WHOLE
candidate group is present. Sampling random rows would break groups apart and
make Top-1 meaningless, so we sample whole observations and keep all their
candidates.
"""
import numpy as np


def subsample_observations(df, max_rows, seed=42):
    """
    Keep whole candidate groups, up to roughly `max_rows` rows.

    Groups are chosen uniformly at random across trips, so the trip mix is
    preserved in expectation.
    """
    if len(df) <= max_rows:
        return df

    keys = df["trip_id"].to_numpy().astype(np.int64) * 1_000_000 + df["obs_idx"].to_numpy()
    unique_obs = np.unique(keys)

    avg_group = len(df) / len(unique_obs)
    n_keep = max(1, int(max_rows / avg_group))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(unique_obs, size=min(n_keep, len(unique_obs)), replace=False)
    return df[np.isin(keys, chosen)]
