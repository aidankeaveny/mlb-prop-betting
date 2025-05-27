import numpy as np
import pandas as pd
import torch
from pybaseball import batting_stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

# Global feature list
feature_cols = ['G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SB', 'CS',
    'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'BB%', 'K%', 'BB/K', 'wOBA', 'wRC', 'wRC+', 'WAR', 'HR/FB',
    'LD%', 'GB%', 'FB%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'EV', 'LA', 'Barrel%', 'HardHit%',
    'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'SwStr%', 'Zone%', 'F-Strike%']


def load_batting_years(start=1990, end=2023):
    years = {}
    for year in range(start, end + 1):
        try:
            df = batting_stats(year)
            df['Season'] = year
            df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['TotalBases'] = (
                df['1B'] +
                2 * df['2B'] +
                3 * df['3B'] +
                4 * df['HR']
            )
            years[year] = df
        except Exception:
            continue
    return years



def build_feature_dataset(years, target_var='OPS'):
    print("✅ Using build_feature_dataset with extended feature engineering")
    player_data = []

    for target_year in range(2000, 2024):
        try:
            print(f"Processing year {target_year}...")

            df_target = years[target_year].copy()
            if 'player_name' in df_target.columns:
                df_target.rename(columns={'player_name': 'Name'}, inplace=True)

            # Compute TotalBases if requested
            if target_var == 'TotalBases' and 'TotalBases' not in df_target.columns:
                df_target['1B'] = df_target['H'] - df_target['2B'] - df_target['3B'] - df_target['HR']
                df_target['TotalBases'] = (
                    df_target['1B'] +
                    2 * df_target['2B'] +
                    3 * df_target['3B'] +
                    4 * df_target['HR']
                )

            if target_var not in df_target.columns:
                print(f"❌ Skipping {target_year}: target_var '{target_var}' not found.")
                continue

            df_target = df_target[['Name', target_var]].rename(columns={target_var: f"{target_var}_target"})
            df_target['Season'] = target_year

            # Load previous season (y1)
            df_y1 = years.get(target_year - 1)
            if df_y1 is None:
                continue
            df_y1 = df_y1.copy()
            if 'player_name' in df_y1.columns:
                df_y1.rename(columns={'player_name': 'Name'}, inplace=True)

            y1_features = df_y1[['Name'] + [col for col in feature_cols if col in df_y1.columns]].copy()
            y1_features.columns = [f"{col}_y1" if col != 'Name' else 'Name' for col in y1_features.columns]

            # Aggregate history
            history_years = [years[y] for y in range(1900, target_year - 1) if y in years]
            if not history_years:
                continue
            df_history = pd.concat(history_years)
            if 'player_name' in df_history.columns:
                df_history.rename(columns={'player_name': 'Name'}, inplace=True)
            valid_cols = [col for col in feature_cols if col in df_history.columns]

            agg = (
                df_history
                .groupby('Name')[valid_cols]
                .agg(['mean', 'std', 'max', 'sum'])
            )
            agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
            agg = agg.reset_index()

            merged = pd.merge(df_target, y1_features, on='Name', how='inner')
            merged = pd.merge(merged, agg, on='Name', how='inner')

            # === Feature Engineering ===

            # Target encodings
            if 'Team' in df_y1.columns:
                team_means = df_history.groupby('Team')[target_var].mean().to_dict()
                df_y1['Team_encoded'] = df_y1['Team'].map(team_means).fillna(df_history[target_var].mean())
                merged['Team_encoded'] = merged['Name'].map(df_y1.set_index('Name')['Team_encoded'])

            if 'Pos' in df_y1.columns:
                pos_means = df_history.groupby('Pos')[target_var].mean().to_dict()
                df_y1['Pos_encoded'] = df_y1['Pos'].map(pos_means).fillna(df_history[target_var].mean())
                merged['Pos_encoded'] = merged['Name'].map(df_y1.set_index('Name')['Pos_encoded'])

            # 3-year rolling average
            rolling_years = [years[y] for y in range(target_year - 3, target_year) if y in years]
            if len(rolling_years) == 3:
                df_rolling = pd.concat(rolling_years)
                if 'player_name' in df_rolling.columns:
                    df_rolling.rename(columns={'player_name': 'Name'}, inplace=True)
                df_rolling_avg = df_rolling.groupby("Name")[feature_cols].mean().reset_index()
                df_rolling_avg.columns = ['Name'] + [f"{col}_3yr_avg" for col in df_rolling_avg.columns[1:]]
                merged = pd.merge(merged, df_rolling_avg, on="Name", how="left")

            # Recency-weighted stats
            weights = [0.6, 0.3, 0.1]
            recency_years = [years.get(y) for y in range(target_year - 3, target_year) if years.get(y) is not None]
            if len(recency_years) == 3:
                for df in recency_years:
                    if 'player_name' in df.columns:
                        df.rename(columns={'player_name': 'Name'}, inplace=True)
                df_weighted = recency_years[0].copy()
                for col in feature_cols:
                    if col in df_weighted.columns:
                        df_weighted[col] = weights[0] * df_weighted[col]
                for i in range(1, 3):
                    df_temp = recency_years[i][['Name'] + [col for col in feature_cols if col in recency_years[i].columns]].copy()
                    for col in feature_cols:
                        if col in df_temp.columns:
                            df_temp[col] = weights[i] * df_temp[col]
                    df_weighted = pd.merge(df_weighted, df_temp, on="Name", how="outer", suffixes=('', f'_w{i}'))
                
                # Drop duplicate columns before aggregation
                df_weighted = df_weighted.fillna(0)
                df_weighted = df_weighted.loc[:, pd.Index(df_weighted.columns).duplicated(keep=False) == False]

                # Aggregate and rename cleanly
                dupes = df_weighted.columns[df_weighted.columns.duplicated()].tolist()
                if dupes:
                    print(f"‼️ Still has duplicates before groupby on {target_year}: {dupes}")
                weighted_sums = df_weighted.groupby("Name", as_index=False).sum()
                new_cols = ['Name'] + [f"{col}_recency" for col in weighted_sums.columns if col != 'Name']
                if len(new_cols) != len(weighted_sums.columns):
                    raise ValueError("❌ Cannot assign new column names due to duplicate original columns.")
                weighted_sums.columns = new_cols
                if df_weighted.columns.duplicated().any():
                    print("‼️ Duplicate columns before groupby:", df_weighted.columns[df_weighted.columns.duplicated()].tolist())
                merged = pd.merge(merged, weighted_sums, on="Name", how="left")

            # Rate-based features
            for base in ['HR', 'WAR', 'BB', 'SO', 'SB', 'CS', 'RBI', 'H']:
                if f"{base}_y1" in merged.columns and "PA_y1" in merged.columns:
                    merged[f"{base}_per_PA_y1"] = merged[f"{base}_y1"] / (merged["PA_y1"] + 1e-5)
                if f"{base}_y1" in merged.columns and "AB_y1" in merged.columns:
                    merged[f"{base}_per_AB_y1"] = merged[f"{base}_y1"] / (merged["AB_y1"] + 1e-5)

            # Position one-hot
            if 'Pos' in df_y1.columns:
                pos_df = df_y1[['Name', 'Pos']].copy()
                pos_df['Pos'] = pos_df['Pos'].astype(str).fillna('UNK').str.split('-').str[0]
                pos_dummies = pd.get_dummies(pos_df['Pos'], prefix='Pos').astype(float)
                pos_encoded = pd.concat([pos_df[['Name']], pos_dummies], axis=1)
                merged = pd.merge(merged, pos_encoded, on="Name", how="left")

            player_data.append(merged)

        except Exception as e:
            
            print(f"❌ Skipping {target_year} due to error: {e}")

    # === Final Assembly ===
    df = pd.concat(player_data).reset_index(drop=True)

    # Feature cleanup
    if 'Age Rng_recency' in df.columns:
        df['Age Rng_recency'] = df['Age Rng_recency'].astype(str).str.extract(r'(\d+)').astype(float)
    if 'Dol_recency' in df.columns:
        df['Dol_recency'] = df['Dol_recency'].astype(str).str.replace(r'[^\d.\-]', '', regex=True).replace('', np.nan).astype(float)
    if 'Team_recency' in df.columns:
        team_dummies = pd.get_dummies(df['Team_recency'], prefix='Team')
        df = pd.concat([df.drop(columns='Team_recency'), team_dummies], axis=1)

    features = df.columns.difference(['Name', f"{target_var}_target", 'Season']).tolist()
    non_numeric = df[features].select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric features found: {non_numeric}")

    X = df[features].fillna(0).values
    y = df[f"{target_var}_target"].values

    selector = VarianceThreshold(threshold=1e-4)
    X = selector.fit_transform(X)
    features = [features[i] for i in selector.get_support(indices=True)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return {
        "df": df,
        "features": features,
        "scaler": scaler,
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    }
