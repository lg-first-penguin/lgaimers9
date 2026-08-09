# code/train.py
import sys
import os


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

def process_trackman_features_safe(df_main, df_trm, is_train_split=True):
    """타임 리크가 차단된 10-Key 상황 지문 기반 트랙맨 전처리 결합 엔진"""
    df_main_copy = df_main.copy()
    df_trm_copy = df_trm.copy()
    
    if is_train_split:
        max_season = df_main_copy['season'].max()
        max_month = df_main_copy[df_main_copy['season'] == max_season]['game_month'].max()
        future_mask = (df_trm_copy['season'] > max_season) | \
                      ((df_trm_copy['season'] == max_season) & (df_trm_copy['game_month'] > max_month))
        df_trm_copy = df_trm_copy[~future_mask].reset_index(drop=True)
        print(f"⏳ [Time Filter] {max_season}년 {max_month}월 이전의 트랙맨 데이터만 잘라내어 피처를 산출합니다.")
    
    match_cols = [dfc for dfc in df_main_copy.columns if (dfc in df_trm_copy.columns) and dfc != 'row_id']
    
    grouped = df_trm_copy.groupby(match_cols + ['pitch_type_group', 'auto_pitch_type'])
    grouped_phase1 = grouped[['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break', 'extension', 'rel_height', 'rel_side', 'zone_speed']].agg(['mean', 'std'])
    grouped_phase2 = grouped_phase1.reset_index()
    
    std_cols = [col for col in grouped_phase2.columns if 'std' in col]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ['_'.join(col).strip('_') for col in grouped_phase2.columns]
    
    grouped_phase3 = grouped_phase2.drop(columns='auto_pitch_type')
    grouped_phase3 = grouped_phase3.groupby(match_cols + ['pitch_type_group']).agg(['mean'])
    
    pivoted = grouped_phase3.unstack(level='pitch_type_group')
    pivoted.columns = [f"{col[0]}_{col[1]}_{col[2]}" for col in pivoted.columns]
    tm_final = pivoted.reset_index()
    tm_final = tm_final.fillna(0)
    
    df_main_copy['top_bottom'] = df_main_copy['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    tm_final['top_bottom'] = tm_final['top_bottom'].map({'Top': 0, 'Bottom': 1}).astype(np.int64)
    
    for col in ['batter_hand', 'pitcher_hand']:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({'Left': 1, 'Right': 2}).astype(np.int64)
            
    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how='left')
    new_feature_cols = [col for col in tm_final.columns if col not in match_cols]
    tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(tr_final[new_feature_cols].mean())
    
    return tr_final, match_cols

def add_engineered_features(df, league_success_mean):
    """asof_* 및 카운트 정보를 조합한 파생 피처를 추가합니다."""
    df = df.copy()

    df['pitcher_recent1_gap'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent3_gap'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent5_gap'] = df['asof_pitcher_prev5_game_success_rate'] - df['asof_pitcher_success_rate']

    df['pitcher_relative_success'] = df['asof_pitcher_success_rate'] - league_success_mean

    df['count_diff'] = df['strikes_before'] - df['balls_before']
    df['is_full_count'] = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(np.int64)

    df['pitcher_count_advantage_raw'] = df['asof_pitcher_success_rate'] * df['count_diff']
    df['pitcher_count_advantage_rel'] = df['pitcher_relative_success'] * df['count_diff']

    df['pitcher_trend'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate']
    df['pitcher_consistency'] = df[[
        'asof_pitcher_prev1_game_success_rate',
        'asof_pitcher_prev3_game_success_rate',
        'asof_pitcher_prev5_game_success_rate',
    ]].std(axis=1)

    df['matchup'] = df['asof_pitcher_success_rate'] - df['asof_batter_success_rate']

    pressure_signal = df['li'] * ((df['strikes_before'] >= 2) | (df['balls_before'] >= 3)).astype(np.int64)
    df['count_pressure'] = df['pitcher_relative_success'] * pressure_signal

    return df

def main():
    DATA_DIR = "./open/data"
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    
    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    
    train_df = tr_final.dropna(subset=['control_success']).reset_index(drop=True)
    target_col = 'control_success'

    train_mask = train_df['season'] < 2024
    val_mask = train_df['season'] == 2024

    league_success_mean = train_df.loc[train_mask, target_col].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ['row_id', target_col]
    features = [col for col in train_df.columns if col not in drop_cols]

    categorical_features = [col_cat for col_cat in tr_final.select_dtypes(include='object').columns if col_cat != 'row_id' and col_cat in features]

    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna('missing')

    X_train, y_train = train_df.loc[train_mask, features], train_df.loc[train_mask, target_col]
    X_val, y_val = train_df.loc[val_mask, features], train_df.loc[val_mask, target_col]
    
    train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=categorical_features)
    
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05040411253232039,
        depth=7,
        l2_leaf_reg=3.14659036827521,
        random_strength=3.715568024268865,
        bagging_temperature=0.4609270457436248,
        border_count=106,
        min_data_in_leaf=81,
        bootstrap_type='Bayesian',
        loss_function='Logloss',
        eval_metric='BrierScore',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    os.makedirs("./open/temp", exist_ok=True)
    with open("./open/temp/latest_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved to ./open/temp/latest_model.pkl")

if __name__ == "__main__":
    main()