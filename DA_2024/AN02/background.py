import logging
import sys
import os

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Starting model training...")

# 기존 코드 시작
# 기본 데이터 처리 라이브러리
import pandas as pd
import numpy as np
from tqdm import tqdm
pd.set_option('display.max_columns', None)

# 로그 및 경고 처리
import warnings
warnings.filterwarnings('ignore')  # 경고 무시
sys.path.append('../../../jupyter_WorkingDirectory/UDM/')

import myLib

# 머신러닝 및 데이터 전처리 라이브러리
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from category_encoders import TargetEncoder
from factor_analyzer import FactorAnalyzer

# 머신러닝 알고리즘
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import Lasso, Ridge

# 데이터 불균형 처리
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# 지리 정보 처리 라이브러리
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, box

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# DuckDB 데이터베이스 연결
import duckdb

# 데이터 준비 코드 시작
gdf = gpd.read_file('../../../KoRoad/TAAS/AN02_2/merged_grid_data.shp')
grid = gpd.read_file('../../../jupyter_WorkingDirectory/GIS/result/Korea_Grid_drop_duplicate_200m_5179.shp')
gdf.drop(columns=['발생건수', '다발지_사', '다발지_사', '다발지_중', '다발지_경',
       '다발지_부', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일', '요일별발생',  '심야', '오전', '오후', '저녁',
       '점심', '출근', '퇴근', '시간대발생', '심야_보정', '기타/불명', '눈', '맑음', '비', '안개', '흐림', '날씨당발생', ],inplace=True)
gdf.rename(columns=
    {
        'road_lengt' : '도로길이',  
       '기타/불명_': '날씨_기타',
        '눈_보정비' : '눈_보정',
       '비_보정비' : '비_보정',
    }
,inplace=True)
df = gdf[gdf['gid'].isin(gdf[(gdf['사고건수']>0)&(gdf['도로길이']>0)]['gid'])]
df.rename(columns=
    {
        'road_lengt' : '도로길이',  
       '기타/불명_': '날씨_기타',
        '눈_보정비' : '눈_보정',
       '비_보정비' : '비_보정',
    }
,inplace=True)

bi_df = gdf.copy()
bi_df[bi_df['도로길이'] > 0].reset_index(drop=True,inplace=True)
bi_df.drop(columns=['geometry','사망자수','중상자수','경상자수','부상신고자'],inplace=True)
bi_df['bi_1st_tg'] = bi_df['사고건수'].apply(lambda x : 1 if x >= 1 else 0)
bi_df[~(bi_df.drop(columns=['월','사고건수','bi_1st_tg']) == 0).all(axis=1)].reset_index(drop=True,inplace=True)


# 'new_gid' 생성
bi_df['new_gid'] = bi_df['gid'].astype('str') + '_' + bi_df['월'].astype('str')

# Step 1: 학습에 사용할 데이터 준비 (gid 및 new_gid 제거)
y = bi_df[bi_df['도로길이'] > 0]['bi_1st_tg']
X = bi_df[bi_df['도로길이'] > 0].drop(columns=[
    '사고건수', 'bi_1st_tg', 'gid', 'new_gid',  
    '월_보정비', '화_보정비', '수_보정비', '목_보정비', '금_보정비', '토_보정비', '일_보정비', 
    '오전_보정', '오후_보정', '저녁_보정', '점심_보정', '출근_보정', '퇴근_보정',
    '날씨_기타', '눈_보정', '맑음_보정', '비_보정', '안개_보정', '흐림_보정'
])

# Step 2: 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
train_indices, test_indices = train_test_split(bi_df[bi_df['도로길이'] > 0].index, test_size=0.3, random_state=42)

# Step 3: SMOTE와 언더샘플링 적용
logging.info("Applying SMOTE and undersampling...")
smote = SMOTE(sampling_strategy=1, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

target_size = 1000000
majority_class_size = target_size // 2
minority_class_size = target_size // 2
under_sampler = RandomUnderSampler(sampling_strategy={0: majority_class_size, 1: minority_class_size}, random_state=42)
X_train_combined, y_train_combined = under_sampler.fit_resample(X_train_smote, y_train_smote)

# Step 4: Grid Search를 통해 최적 하이퍼파라미터 찾기
logging.info("Performing Grid Search for hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [1,2, 10],
    'min_samples_leaf': [1, 5,10],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_combined, y_train_combined)
best_model = grid_search.best_estimator_
logging.info(f"Best Model Parameters: {grid_search.best_params_}")

# Step 5: 최적 모델로 테스트 세트 평가
logging.info("Evaluating best model on test set...")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 모델 성능 평가
logging.info("Model performance on test set:")
print(classification_report(y_test, y_pred))
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# 예측 결과를 final_df에 추가
final_df = bi_df.copy()
final_df.loc[test_indices, 'pred'] = y_pred
final_df.loc[test_indices, 'pred_proba'] = y_pred_proba
logging.info("Predictions and probabilities added to final_df.")

# 모든 테스트 샘플에 대해 피처별 중요도 계산
logging.info("Calculating feature impact for each test sample...")
results = []

for idx, row in tqdm(X_test.iterrows(), total=len(X_test), desc="Calculating feature impact"):
    original_proba = best_model.predict_proba([row])[0][1]
    feature_impact = {"index": idx}
    
    for feature in X.columns:
        temp_row = row.copy()
        temp_row[feature] = X_train[feature].mean()
        new_proba = best_model.predict_proba([temp_row])[0][1]
        feature_impact[feature] = abs(original_proba - new_proba)
    
    results.append(feature_impact)

# 결과를 데이터프레임으로 변환
feature_impact_df = pd.DataFrame(results)

# 피처 영향도 데이터 저장
feature_impact_df.to_csv("./result/feature_impact.csv", index=False)
logging.info("Feature impact data saved to 'feature_impact.csv'.")
logging.info("Script finished successfully.")