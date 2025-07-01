import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
tqdm.pandas()
import shap
import logging

# 로깅 설정
logging.basicConfig(
    filename="reg_model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Script started.")

def calc_ari(x):
    ksi = x['사망자수']+x['중상자수']
    acc_cnt = x['사고건수']
    ari = np.sqrt(ksi**2+acc_cnt**2) / 12
    return ari

# Adjusted R2 score 계산 함수
def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

logging.info("Reading and preparing data.")
gdf = gpd.read_file('../../../KoRoad/TAAS/AN02_2/merged_grid_data.shp')
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


class_result = pd.read_csv('./result/merged_result.csv', encoding='cp949')
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
    '날씨_기타', '맑음_보정', '눈_보정','흐림_보정', #'안개_보정', '비_보정',
    #'버스정류장_갯수', '속도위반_갯수', '신호위반_갯수', '과학_기술_갯수', '교육_갯수', '보건의료_갯수','부동산_갯수', '소매_갯수', '수리_개인_갯수', '숙박_갯수', '시설관리__갯수', '예술_스포_갯수', '음식_갯수',
    'bi_1st_tg', 'new_gid', 'pred', 
    #'pred_proba', 
    'index', 'final_df_idx',
    '월_영향도','다발지선정_영향도', '도로길이_영향도', '버스정류장_영향도', '속도위반_영향도', '신호위반_영향도', 
    '과학_기술_영향도', '교육_영향도', '보건의료_영향도', '부동산_영향도', '소매_영향도', '수리_개인_영향도',
    '숙박_영향도', '시설관리__영향도', '예술_스포_영향도', '음식_영향도',  
    '사망자수','중상자수', '경상자수', '부상신고자', 
    #'monthly_ari', 'annual_ari'
],inplace=True)
tmp.head()
tmp['new_gid'] = tmp['gid'].astype('str')+'_' + tmp['발생월'].astype('str')
tmp.set_index('new_gid',inplace=True)
# 데이터 준비
X = tmp.drop(columns = ['사고건수','gid','monthly_ari','pred_proba','안개_보정','비_보정','annual_ari','숙박_갯수','과학_기술_갯수'])
y = tmp['monthly_ari']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 설정 및 평가 함수
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    # 평가 메트릭 계산
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    adj_r2 = adjusted_r2(r2, len(y_test), X_test.shape[1])
    
    print(f"\n{model_name} Results")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Adjusted R2 Score: {adj_r2:.4f}")
    
    # Feature Importance
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    logging.info(f"{model_name} Feature Importance calculated.")
    # X_test에 예측값 추가 (복원 로직)
    results_df = pd.DataFrame({
        'pred': pred,
    }, index=X_test.index)  # gid가 인덱스로 설정된 상태
    
    X_test_with_pred = X_test.join(results_df)
    X_test_with_pred['actual'] = y_test
    
    # tmp 복원 (외부에서 사용 가능하도록 반환)
    tmp_restored = tmp.join(X_test_with_pred[['pred', 'actual']], how='left')
    
    return tmp_restored, feature_importance_df, model

# 모델 학습 및 평가
rf_model = RandomForestRegressor(random_state=42, n_estimators=300, max_depth=30, min_samples_leaf=2, min_samples_split=5) #300 30 2 5
rf_restored, rf_feature_importance, rf_model = train_and_evaluate(rf_model, "Random Forest")
rf_restored.drop(columns=['actual'])
logging.info("Model training and evaluation completed.")

# 1. pred가 있는 행들만 선택
valid_pred_index = rf_restored[rf_restored['pred'].notna()].index
X_valid = rf_restored.loc[valid_pred_index, X.columns]  # feature만 추출

# 1. explainer 생성
logging.info("Calculating SHAP values.")
explainer = shap.TreeExplainer(rf_model)

# 2. tqdm을 사용하여 SHAP value 계산
shap_values_list = []

# tqdm을 사용해 각 샘플에 대해 SHAP 계산
for i in tqdm(range(X_valid.shape[0]), desc="Calculating SHAP values"):
    shap_values_list.append(explainer.shap_values(X_valid.iloc[[i]]))  # 각 행에 대해 SHAP 계산

# 3. 리스트를 배열로 변환
shap_values = np.vstack(shap_values_list)

# 4. SHAP values를 DataFrame으로 변환
shap_df = pd.DataFrame(shap_values, columns=[f"shap_{col}" for col in X.columns], index=valid_pred_index)

# 5. SHAP values를 rf_restored에 병합
rf_restored = rf_restored.join(shap_df, how='left')

logging.info("Script completed successfully.")
rf_restored.to_csv("./result/rf_restored_with_shap.csv", encoding='cp949',index=False)

