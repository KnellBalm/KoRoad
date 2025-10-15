import os
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import logging
import numpy as np

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np

import geopandas as gpd
from shapely import Point, LineString, Polygon, MultiPoint
from shapely.ops import nearest_points
from sklearn.cluster import DBSCAN

from tqdm import tqdm
tqdm.pandas()

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm, rc

# 한글 폰트 설정
font_path = '/usr/local/src/python3.10/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/KoPub Dotum Medium.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# UDM
def find_nearest_line_within_radius(point, lines, column_name, radius=1000):
    """
    특정 반경 내에서 가장 가까운 라인을 찾고 특정 열의 값을 반환하는 함수.
    """
    buffer = point.buffer(radius)
    nearby_lines = lines[lines.intersects(buffer)]

    if not nearby_lines.empty:
        distances = nearby_lines.geometry.apply(lambda line: point.distance(line))
        nearest_idx = distances.idxmin()
        nearest_value = nearby_lines.loc[nearest_idx, column_name]
        nearest_distance = distances[nearest_idx]
        return nearest_value, nearest_distance
    else:
        return None, None

def mk_flag(x):
    '''make mapping flag'''
    if pd.isnull(x['SPLT_LINK_x']) and pd.isnull(x['nearest_line_info']):
        return 'failed'
    elif pd.isnull(x['SPLT_LINK_x']) and pd.notnull(x['nearest_line_info']):
        return 'second_match'
    else: 
        return 'first_match'

#Data Load
sgg = gpd.read_file('../../GIS/sgg/sig.shp',encoding='cp949')
sido = gpd.read_file('../../GIS/sido/ctprvn.shp',encoding='cp949')
splt_link = gpd.read_file('../../../KoRoad/TAAS/AN04/세분화링크/세분화링크_20240822.shp')
splt_link.set_crs(5179,inplace=True)

mts = pd.read_csv('../../../KoRoad/TAAS/AN04/merge_target/교통과학장비_설치정보_241007.csv')
mts = gpd.GeoDataFrame(
    mts[(~mts['제어기 경도'].isna()) & (~mts['장비번호'].isin(['J1539', 'J1558', 'J1559', 'J2782']))],
    geometry=mts.apply(lambda x: Point(x['제어기 경도'], x['제어기 위도']), axis=1),
    crs=4326
).to_crs(5179)

# 유효한 지오메트리만 필터링
mts = mts[mts.geometry.is_valid & mts.geometry.notnull()].reset_index(drop=True)

mless = mts[['장비번호', 'geometry']]
link = splt_link[['SPLT_LINK', 'geometry']]

first_merged = gpd.sjoin_nearest(mless, link, how='left', distance_col='distance', max_distance=30)
print("First Mapping Finish")

unmatched_camera = first_merged[first_merged['distance'].isna()]
results = unmatched_camera.geometry.progress_apply(
    lambda point: find_nearest_line_within_radius(point, splt_link, 'SPLT_LINK',radius=1000)
)
print("Second Mapping Finish")

results_df = pd.DataFrame(results.tolist(), index=unmatched_camera.index, columns=['nearest_line_info', 'distance_to_nearest'])
unmatched_camera = unmatched_camera.join(results_df)

#
merged_mts = mts.copy()
#매핑
merged_mts = pd.merge(merged_mts,first_merged,how='left',left_on = '장비번호', right_on = '장비번호').drop(columns=['geometry_y','index_right'])
full_mts = pd.merge(merged_mts,unmatched_camera,how='left',left_on = '장비번호', right_on = '장비번호').drop(columns=['geometry','index_right','SPLT_LINK_y','distance_y'])
#컬럼정비
full_mts['match_flag'] = full_mts.apply(mk_flag, axis=1)
full_mts['SPLT_LINK_x'].fillna(full_mts['nearest_line_info'],inplace=True)
full_mts['distance_x'].fillna(full_mts['distance_to_nearest'],inplace=True)
full_mts.drop(columns=['nearest_line_info','distance_to_nearest'],inplace=True)

#컬럼 재정비
full_mts.rename(columns={
    'geometry_x' : 'geometry',
    'SPLT_LINK_x' : 'SPLT_LINK',
    'distance_x' : 'distance'
},inplace=True)

full_mts.columns = [x.replace(' ','').replace('·','_') for x in full_mts.columns]
full_mts.to_file('./result/MTS_SPLT_LINK_MAPPING_FINISH.shp',encoding='cp949',index=False,driver='ESRI Shapefile')