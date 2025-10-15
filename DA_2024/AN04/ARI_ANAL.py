#UDM
def mk_ari(x):
    '''
    make_ari_index
    '''
    ksi = x.DTH_DNV_CNT + x.SE_DNV_CNT
    acc_cnt = x.ACC_NO
    return np.sqrt(ksi**2+acc_cnt**2) / 1

def chg_first_installed_label(x):
    '''
    this use only first installed manless violation equipment
    '''
    if x == -1: return 'before_installed'
    elif x == 0 : return 'installed'
    else : return 'after_installed'

def calc_result(x):
    '''
    final calculation for ARI UP or Down
    '''
    if x.after_installed < x.installed or x.after_installed < x.before_installed:
        return 'down'
    else: return 'up'

def label_installed_years(group):
    '''
    make installed year label
    '''
    labels = []
    seen_zero = 0
    for index, row in group.iterrows():
        if row['inst_time'] < 0:
            labels.append(f"before_{-int(row['inst_time'])}years")
        elif row['inst_time'] == 0:
            if seen_zero == 0:
                labels.append("first_installed")
                seen_zero += 1
            elif seen_zero == 1:
                labels.append("second_installed")
                seen_zero += 1
            elif seen_zero == 2:
                labels.append("third_installed")
                seen_zero += 1
            else:
                labels.append(f"after_{seen_zero + 1}years_second_installed")
                seen_zero += 1
        else:
            labels.append(f"after_{int(row['inst_time'])}years")
    return labels
    
def make_double_flag(x):
    if x in double_cam_link_list: return 1
    else : return 2


def make_double_flag(x):
    if x in double_cam_link_list: return 1
    else : return 2

#Data Load
mts_link = gpd.read_file('./result/MTS_SPLT_LINK_MAPPING_FINISH.shp',encoding='cp949')
ari = pd.read_csv('/DATA/KoRoad/TAAS/AN04/링크_사고연동_240822.csv')

#ARI_Grouping&Calculate
ari_gb = ari.groupby(['SPLIT_LINK_ID','ACC_YEAR'],as_index=False).agg({'ACC_NO' : 'count','DTH_DNV_CNT' : 'sum','SE_DNV_CNT': 'sum'})
ari_gb['ARI'] = ari_gb.apply(mk_ari,axis=1)

#Make ARI_DataFrame by Full Year
years = [2018,2019, 2020, 2021, 2022,2023]
ari_df_year = ari_gb.groupby('SPLIT_LINK_ID').progress_apply(
    lambda group: group.set_index('ACC_YEAR').reindex(years, fill_value=0).assign(SPLIT_LINK_ID=group.name).reset_index()
).reset_index(drop=True)
ari_df_year.rename(columns={'index': 'ACC_YEAR'}, inplace=True)

#preprocessing MTS DataFrame
mts_link['installed_year'] = mts_link['관리_설치'].apply(lambda x: x[:4])
mts_gb = mts_link.groupby(['SPLT_LINK','installed_year'],as_index=False).agg({'장비번호':'count'}).rename(columns={'장비번호':'cam_cnt'})
mts_gb['installed_year'] = mts_gb['installed_year'].astype('int')

####################################################################################
############## calculate Only Installed before&After difference  ###################
####################################################################################

#merge with ARI agg DF and MTS agg DF
merged = pd.merge(ari_df_year,mts_gb, how='left',left_on=['SPLIT_LINK_ID'], right_on = ['SPLT_LINK'])
## make installed time
merged['inst_time'] = merged['ACC_YEAR'] - merged['installed_year']    
##extract need columns
target = merged[~merged['installed_year'].isna()][['ACC_YEAR','SPLIT_LINK_ID','ARI','installed_year','cam_cnt','inst_time']]
target = target.astype({'installed_year' : 'int', 'cam_cnt':'int', 'inst_time':'int'})
##calculate before & after only 1 years
target_gb = target[target['inst_time'].isin([-1,0,1])].groupby(['SPLIT_LINK_ID','inst_time'],as_index=False).agg({'ARI':'sum'})
## making label
target_gb['inst_time'] = target_gb['inst_time'].map(chg_first_installed_label)

## make result DataFrame
target_gb = target_gb.pivot(index='SPLIT_LINK_ID', columns = 'inst_time', values='ARI').dropna().reset_index()
target_gb = target_gb[['SPLIT_LINK_ID', 'before_installed', 'installed','after_installed', ]]
target_gb = target_gb[~(target_gb['before_installed']==0)&~(target_gb['installed']==0)&~(target_gb['after_installed']==0)]
target_gb['result'] = target_gb.apply(lambda x : calc_result(x),axis=1)

####################################################################################
######################### Second+ Installed Equipment ##############################
####################################################################################

filtered = mts_gb.groupby(['SPLT_LINK'])['installed_year'].nunique().reset_index()
double_installed = mts_gb[mts_gb['SPLT_LINK'].isin(filtered[filtered['installed_year']> 1]['SPLT_LINK'].to_list())]
double_ari_df = ari_df_year[ari_df_year['SPLIT_LINK_ID'].isin(double_installed['SPLT_LINK'].unique())].reset_index(drop=True)
double_merge = pd.merge(double_ari_df,double_installed,how='left',left_on=['SPLIT_LINK_ID','ACC_YEAR'],right_on=['SPLT_LINK','installed_year']).drop(columns=['SPLT_LINK'])

result_df = pd.DataFrame()
for i in tqdm(double_merge.SPLIT_LINK_ID.unique()):
    target_df = double_merge[double_merge['SPLIT_LINK_ID'] == i]
    target_df.sort_values(['SPLIT_LINK_ID','ACC_YEAR'],ascending=[True,True],inplace=True)
    target_df['installed_year'] = target_df['installed_year'].ffill().bfill()
    
    if target_df['installed_year'].sum() == 0:
        continue  # skip if no valid installed_year data
    target_df['inst_time'] = target_df['ACC_YEAR'] - target_df['installed_year']
    target_df = target_df.sort_values(by='ACC_YEAR')  # sort for consistency
    target_df['installed_label'] = label_installed_years(target_df)
    result_df = pd.concat([result_df, target_df])


####################################################################################
############################### Full Merged Equipment ##############################
####################################################################################

mts_link = gpd.read_file('./result/MTS_SPLT_LINK_MAPPING_FINISH.shp',encoding='cp949')
ari = pd.read_csv('/DATA/KoRoad/TAAS/AN04/링크_사고연동_240822.csv')
# ARI 집계 DF(MTS와 연결된 LINK만) 생성 및 ARI 계산
## ari_df_year = 연도별 링크별 ARI 계산 DF

# ARI 집계 DF(MTS와 연결된 LINK만) 생성 및 ARI 계산
full_ari = ari.copy()
full_ari = full_ari[full_ari['SPLIT_LINK_ID'].isin(mts_link['SPLT_LINK'].unique())]
ari_gb = full_ari.groupby(['SPLIT_LINK_ID','ACC_YEAR'],as_index=False).agg({'ACC_NO' : 'count','DTH_DNV_CNT' : 'sum','SE_DNV_CNT': 'sum'})
ari_gb['ARI'] = ari_gb.apply(mk_ari,axis=1)

# 연도별 ARI LINK DF 생성
years = [2018,2019, 2020, 2021, 2022,2023]
ari_df_year = ari_gb.groupby('SPLIT_LINK_ID').progress_apply(
    lambda group: group.set_index('ACC_YEAR').reindex(years, fill_value=0).assign(SPLIT_LINK_ID=group.name).reset_index()
).reset_index(drop=True)
ari_df_year.rename(columns={'index': 'ACC_YEAR'}, inplace=True)

## MTS 설치 2회 이상 Flag 생성

full_mts = mts_link.copy()
full_mts['installed_year'] = full_mts['관리_설치'].apply(lambda x : x[:4])
full_mts_grouped = full_mts.groupby(['SPLT_LINK','installed_year'],as_index=False).size().rename(columns={'size':'cam_cnt'})
full_mts_grouped['installed_year'] = full_mts_grouped['installed_year'].astype('int')
counting_double_year = full_mts_grouped.groupby('SPLT_LINK').size()
#카메라가 2개 이상 있는 링크 리스트
double_cam_link_list = counting_double_year[counting_double_year > 1].index
ari_df_year['install_doubled_flag'] = ari_df_year['SPLIT_LINK_ID'].map(make_double_flag)

full_merged = pd.merge(ari_df_year, full_mts_grouped,how='left', left_on=['SPLIT_LINK_ID','ACC_YEAR'], right_on=['SPLT_LINK','installed_year']).drop(columns=['SPLT_LINK'])
result_df = pd.DataFrame()
for i in tqdm(full_merged.SPLIT_LINK_ID.unique()):
    target_df = full_merged[full_merged['SPLIT_LINK_ID'] == i]
    target_df.sort_values(['SPLIT_LINK_ID','ACC_YEAR'],ascending=[True,True],inplace=True)
    target_df['installed_year'] = target_df['installed_year'].ffill().bfill()
    
    if target_df['installed_year'].sum() == 0:
        continue  # skip if no valid installed_year data
    target_df['inst_time'] = target_df['ACC_YEAR'] - target_df['installed_year']
    target_df = target_df.sort_values(by='ACC_YEAR')  # sort for consistency
    target_df['installed_label'] = label_installed_years(target_df)
    result_df = pd.concat([result_df, target_df])
    
result_df.drop(columns=['cam_cnt'],inplace=True)