# 기본 데이터 처리 라이브러리
import pandas as pd
import numpy as np
from tqdm import tqdm
import geopandas as gpd
pd.set_option('display.max_columns', None)

# 로그 및 경고 처리
import logging
import warnings
warnings.filterwarnings('ignore')  # 경고 무시
import os
import sys
sys.path.append('../../../jupyter_WorkingDirectory/UDM/')

import myLib


# 머신러닝 및 데이터 전처리 라이브러리
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from category_encoders import TargetEncoder
from factor_analyzer import FactorAnalyzer
import shap


# 머신러닝 알고리즘
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import Lasso, Ridge
from lightgbm import LGBMRegressor
# from catboost import CatBoostClassifier  # 사용시 주석 해제

# 데이터 불균형 처리
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def calc_ari(x):
    ksi = x['사망자수']+x['중상자수']
    acc_cnt = x['사고건수']
    ari = np.sqrt(ksi**2+acc_cnt**2) / 12
    return ari
    
# Adjusted R2 score 계산 함수
def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))


# 로깅 설정
logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

gdf = gpd.read_file('../../../KoRoad/TAAS/AN02_2/merged_grid_data.shp')
class_result = pd.read_csv('./result/merged_result.csv', encoding='cp949')
logging.info("Load Data Finish")
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

class_result.rename(columns=lambda x: x.replace('_x', '_갯수').replace('_y', '_영향도').replace('·','_'),inplace=True)
class_result.rename(columns={'월_갯수' : '발생월','다발지선정_갯수':'다발지선정_횟수','도로길이_갯수':'격자내도로길이',}, inplace=True)

wonbon = df.copy()

wonbon['new_gid'] = wonbon['gid'].astype('str')+'_'+ wonbon['월'].astype('str')

reg_df = pd.merge(class_result,wonbon[['new_gid','사망자수','중상자수','경상자수','부상신고자']], how = 'left', left_on = ['new_gid'], right_on=['new_gid'])

reg_df.fillna({
    '사망자수' : 0,
    '중상자수' : 0,
    '경상자수' : 0,
    '부상신고자' : 0
},inplace=True)

reg_df['monthly_ari'] = reg_df.apply(calc_ari,axis=1)

annual = reg_df.groupby(['gid','사고건수','사망자수','중상자수'],as_index=False).agg({
    '사고건수':'sum', '사망자수':'sum','중상자수':'sum'
})

annual['annual_ari'] = annual.apply(lambda x : round(np.sqrt((x['사망자수']+x['중상자수'])**2+(x['사고건수']**2)),4) ,axis=1 )

ari_dict = {key:value for key,value in zip(annual['gid'],annual['annual_ari'])}
reg_df['annual_ari'] = reg_df['gid'].apply(lambda x : ari_dict.get(x))

tmp = reg_df[reg_df['pred']>0]
tmp.drop(columns=[ 
    #'gid', '발생월', '다발지선정_횟수', '격자내도로길이', '사고건수',
    '월_보정비', '화_보정비', '수_보정비','목_보정비', '금_보정비', '토_보정비', '일_보정비', 
    '오전_보정', '오후_보정', '저녁_보정', '점심_보정','출근_보정', '퇴근_보정', 
    '날씨_기타', '눈_보정', '맑음_보정', '비_보정', '안개_보정', '흐림_보정',
    #'버스정류장_갯수', '속도위반_갯수', '신호위반_갯수', '과학_기술_갯수', '교육_갯수', '보건의료_갯수','부동산_갯수', '소매_갯수', '수리_개인_갯수', '숙박_갯수', '시설관리__갯수', '예술_스포_갯수', '음식_갯수',
    'bi_1st_tg', 'new_gid', 'pred', 'pred_proba', 'index', 'final_df_idx',
    '월_영향도','다발지선정_영향도', '도로길이_영향도', '버스정류장_영향도', '속도위반_영향도', '신호위반_영향도', 
    '과학_기술_영향도', '교육_영향도', '보건의료_영향도', '부동산_영향도', '소매_영향도', '수리_개인_영향도',
    '숙박_영향도', '시설관리__영향도', '예술_스포_영향도', '음식_영향도',  
    '사망자수','중상자수', '경상자수', '부상신고자', 
    #'monthly_ari', 'annual_ari'
],inplace=True)
logging.info("Data Preprocessing Complete")

# 데이터 준비 (가정)
X = tmp.drop(columns=['gid', 'monthly_ari', '사고건수'])
y = tmp['monthly_ari']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

# GridSearchCV 설정을 위한 하이퍼파라미터 그리드
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}
xg_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}
lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_samples': [10, 20, 30]
}

# 모델별 그리드 서치 설정
models = [
    ("Random Forest", RandomForestRegressor(random_state=42), rf_param_grid),
    ("XGBoost", XGBRegressor(random_state=42), xg_param_grid),
    ("LightGBM", LGBMRegressor(random_state=42), lgbm_param_grid)
]

# 모델 학습 및 평가
for name, model, param_grid in models:
    print(f"\n{name} Model Training...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', verbose=2, n_jobs=20)
    grid_search.fit(X_train, y_train)
    
    # 최적의 모델로 예측
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 성능 평가
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    k = X_test.shape[1]
    adj_r2 = adjusted_r2(r2, n, k)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{name} Results")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Adjusted R2 Score: {adj_r2:.4f}")