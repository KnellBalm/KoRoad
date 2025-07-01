import logging
import sys
import os

# 기본 데이터 처리 라이브러리
import pandas as pd
import numpy as np
from tqdm import tqdm

# 로그 및 경고 처리
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 및 데이터 전처리 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# 데이터 불균형 처리
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 지리 정보 처리 라이브러리
import geopandas as gpd

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

# Step 1: 학습에 사용할 데이터 준비 (gid 및 new_gid 제거)
y = bi_df[bi_df['도로길이'] > 0]['bi_1st_tg']
X = bi_df[bi_df['도로길이'] > 0].drop(columns=[
    '사고건수', 'bi_1st_tg', 'gid', 'new_gid',  # 학습용 X에서 gid와 new_gid 제거
    '월_보정비', '화_보정비', '수_보정비', '목_보정비', '금_보정비', '토_보정비', '일_보정비', 
    '오전_보정', '오후_보정', '저녁_보정', '점심_보정', '출근_보정', '퇴근_보정',
    '맑음_보정', '날씨_기타', '눈_보정', '비_보정', '흐림_보정', '안개_보정'
])
logging.info("Data preparation completed: X and y ready for training.")

# Step 2: 데이터 분할
logging.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 인덱스 분할
train_indices, test_indices = train_test_split(bi_df[bi_df['도로길이'] > 0].index, test_size=0.3, random_state=42)
logging.info("Data split completed.")

# Step 3: SMOTE와 언더샘플링 적용
logging.info("Applying SMOTE and undersampling...")
smote = SMOTE(sampling_strategy=1, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

target_size = 1500000
majority_class_size = target_size // 2
minority_class_size = target_size // 2
under_sampler = RandomUnderSampler(sampling_strategy={0: majority_class_size, 1: minority_class_size}, random_state=42)
X_train_combined, y_train_combined = under_sampler.fit_resample(X_train_smote, y_train_smote)
logging.info("SMOTE and undersampling completed.")

# 모델 학습
logging.info("Training RandomForest model...")
model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    max_depth=45,
    min_samples_leaf=15,
    min_samples_split=15,
    max_features='sqrt',
    n_estimators=600,
    criterion='entropy',
    verbose=0,
    n_jobs=-1
)
model.fit(X_train_combined, y_train_combined)
logging.info("Model training completed.")

# 테스트 세트 평가
logging.info("Evaluating model on test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Step 5: 모델 성능 평가
logging.info("Generating classification report and feature importance...")
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Feature Importance 추출 및 컬럼명과 함께 정렬하여 출력
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")


# Step 6: 예측 결과를 final_df에 추가
logging.info("Adding predictions and probabilities to final_df...")
# final_df에 예측 결과 추가 및 저장
final_df = bi_df.copy()
final_df.loc[test_indices, 'pred'] = y_pred
final_df.loc[test_indices, 'pred_proba'] = y_pred_proba
final_df.index.name = 'final_df_idx'
final_df.to_csv('./result/acc_occr_classification_final.csv', index=True, encoding='cp949')
logging.info("Predictions and probabilities added to final_df and saved.")

# 모든 테스트 샘플에 대해 피처별 중요도 계산
logging.info("Calculating feature impact for each test sample...")
results = []
for idx, row in tqdm(X_test.iterrows(), total=len(X_test), desc="Calculating feature impact"):
    original_proba = model.predict_proba([row])[0][1]
    feature_impact = {"index": idx}
    for feature in X.columns:
        temp_row = row.copy()
        temp_row[feature] = X_train[feature].mean()  # 원래 값 대신 평균 값으로 대체하여 영향도 측정
        new_proba = model.predict_proba([temp_row])[0][1]
        feature_impact[feature] = abs(original_proba - new_proba)
    results.append(feature_impact)

# feature_impact_df에 test_indices 추가 및 저장
feature_impact_df = pd.DataFrame(results)
feature_impact_df['final_df_idx'] = test_indices
feature_impact_df.to_csv('./result/feature_impact_final.csv', index=False)
logging.info("Feature impact data saved to 'feature_impact_final.csv'.")

# final_df와 feature_impact_df 병합
acc_df = pd.read_csv('./result/acc_occr_classification_final.csv', encoding='cp949', index_col='final_df_idx')
feature_impact_df = pd.read_csv('./result/feature_impact_final.csv')
merged_df = pd.merge(acc_df, feature_impact_df, left_index=True, right_on='final_df_idx', how='inner')

# 최종 병합 결과 저장
merged_df.to_csv('./result/merged_result.csv', index=False, encoding='cp949')
logging.info("Merged result saved to './result/merged_result.csv'")
logging.info("Script finished successfully.")
