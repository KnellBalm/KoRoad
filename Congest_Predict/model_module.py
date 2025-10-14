import warnings
warnings.filterwarnings('ignore')
import logging

from sqlalchemy import create_engine
import pandas as pd
import pickle
from xgboost import XGBRegressor

import holidays
from datetime import timedelta
import itertools
import sys
import traceback

# ---------------------------------------------------------------------
# Logging 설정
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("forecast_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ---------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------
try:
    _db_connection = create_engine('mysql+pymysql://koai:1234@192.168.106.66/koroad')
    logging.info("Database engine created successfully")
except Exception as e:
    logging.exception("Failed to create DB connection")
    raise

# ---------------------------------------------------------------------
# Utility 함수
# ---------------------------------------------------------------------
def load_data(query, db_connection=_db_connection):
    logging.info("Loading data from DB...")
    try:
        df = pd.read_sql(query, db_connection)
        df.dropna(inplace=True)
        logging.info(f"Data loaded successfully: shape={df.shape}")
        return df
    except Exception:
        logging.exception("Error occurred while loading data")
        raise

def check_adjacent_holiday(dt):
    """입력: datetime.date, 출력: 전날/다음날 중 공휴일 존재 여부"""
    try:
        kr_holidays = holidays.SouthKorea()
        target_date = dt if not isinstance(dt, (pd.Timestamp,)) else dt.date()
        prev_day = target_date - timedelta(days=1)
        next_day = target_date + timedelta(days=1)
        return int(prev_day in kr_holidays or next_day in kr_holidays)
    except Exception:
        logging.exception("Error in check_adjacent_holiday")
        raise

def preprocessing(df, future=False, peak_season_months=None):
    logging.info(f"Preprocessing started. future={future}")
    try:
        df.dropna(inplace=True)
        if not future:  # 학습 데이터일 때
            df['dela_secnd'] = df['dela_secnd'].dt.total_seconds().astype(int)

        # datetime 처리
        df['tkt_evt_dt'] = pd.to_datetime(df['tkt_evt_dt'])
        df['evt_dt'] = df['tkt_evt_dt'].dt.date
        df['evt_yr'] = df['tkt_evt_dt'].dt.year
        df['evt_mm'] = df['tkt_evt_dt'].dt.month
        df['evt_dotw'] = df['tkt_evt_dt'].dt.day_name()
        df['evt_hour'] = df['tkt_evt_dt'].dt.hour

        # 공휴일 전후 flag
        df['is_near_holidays'] = df['evt_dt'].apply(lambda x: check_adjacent_holiday(x))
        # 피크 시즌 flag (실행부에서 전달받은 months 사용)
        if peak_season_months is None:
            peak_season_months = [1, 12]
        df['is_peak_season'] = df['evt_mm'].apply(lambda x: 1 if x in peak_season_months else 0)
        # 평일만 포함
        df = df[df['evt_dotw'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]

        if not future:
            df.drop(columns=['tkt_evt_dt', 'evt_dt'], inplace=True)
        else:
            df.drop(columns=['tkt_evt_dt'], inplace=True)
            df.set_index('evt_dt', inplace=True)

        # 카테고리형 변환
        for col in ["branch_cd", "group_no", 'evt_mm', 'evt_dotw', 'evt_hour']:
            df[col] = df[col].astype("category")

        logging.info(f"Preprocessing complete. shape={df.shape}")
        return df
    except Exception:
        logging.exception("Error in preprocessing")
        raise

def train_and_save_models(df, target_col="dela_secnd", peak_season_months=None):
    logging.info("Model training started")

    def train_and_save(subset_df, filename):
        if target_col not in subset_df.columns:
            logging.warning(f"{filename}: Target column '{target_col}' not found. Skipped.")
            return

        subset_df = subset_df[subset_df[target_col].notna()]
        subset_df = subset_df[subset_df[target_col] >= 0]

        if subset_df.empty:
            logging.warning(f"{filename}: No data available after cleaning. Skipped.")
            return

        X = subset_df.drop(columns=[target_col])
        y = subset_df[target_col]

        if y.nunique() <= 1:
            logging.warning(f"{filename}: Target has only one unique value ({y.iloc[0]}). Skipped.")
            return

        logging.info(
            f"Training {filename} - shape={subset_df.shape}, "
            f"target_min={y.min()}, target_max={y.max()}, target_mean={y.mean()}, target_unique={y.nunique()}"
        )

        try:
            model = XGBRegressor(
                objective="reg:squarederror",   # 안정적 회귀용
                base_score=float(y.mean()),     # 직접 지정
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                enable_categorical=True,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X, y)

            with open(filename, "wb") as f:
                pickle.dump(model, f)

            logging.info(f"Model saved to {filename}")

        except Exception:
            logging.exception(f"Error while training {filename}")

    # 성수기 / 비성수기 분리
    if peak_season_months is None:
        peak_season_months = [1, 12]
    seasonal_df = df[df["evt_mm"].isin(peak_season_months)]
    regular_df = df[~df["evt_mm"].isin(peak_season_months)]

    train_and_save(seasonal_df, "seasonal_model.pkl")
    train_and_save(regular_df, "regular_model.pkl")

    logging.info("Model training process finished")


def make_future_dataframe(start, end, df, peak_season_months=None):
    logging.info("Making future dataframe...")
    try:
        kr_holidays = holidays.SouthKorea(
            years=range(pd.to_datetime(start).year, pd.to_datetime(end).year + 1)
        )

        # 평일 + 9~17시 + 공휴일 제외
        dates = [
            x for x in pd.date_range(start, end, freq="H")
            if (x.hour in range(9, 18)) and (x.weekday() < 5) and (x.date() not in kr_holidays)
        ]

        branches = df["branch_cd"].unique()
        tasks = df["group_no"].unique()

        combinations = list(itertools.product(branches, tasks, dates))
        future_df = pd.DataFrame(combinations, columns=["branch_cd", "group_no", "tkt_evt_dt"])
        future_df = preprocessing(future_df, future=True, peak_season_months=peak_season_months)

        logging.info(f"Future dataframe complete. shape={future_df.shape}")
        return future_df
    except Exception:
        logging.exception("Error in make_future_dataframe")
        raise

def predict_delay(df, target_col="dela_secnd", peak_season_months=None):
    logging.info("Prediction started")

    preds = pd.Series(0, index=df.index, dtype=float)  # 기본값 0으로 초기화

    try:
        # 모델 불러오기 (없으면 None 처리)
        try:
            with open("seasonal_model.pkl", "rb") as f:
                seasonal_model = pickle.load(f)
        except FileNotFoundError:
            logging.warning("seasonal_model.pkl not found, default 0 used")
            seasonal_model = None

        try:
            with open("regular_model.pkl", "rb") as f:
                regular_model = pickle.load(f)
        except FileNotFoundError:
            logging.warning("regular_model.pkl not found, default 0 used")
            regular_model = None

        if peak_season_months is None:
            peak_season_months = [1, 12]

        seasonal_mask = df["evt_mm"].isin(peak_season_months)
        regular_mask = ~seasonal_mask

        if seasonal_model is not None:
            X_seasonal = df.loc[seasonal_mask].drop(columns=[target_col], errors="ignore")
            preds.loc[seasonal_mask] = seasonal_model.predict(X_seasonal)

        if regular_model is not None:
            X_regular = df.loc[regular_mask].drop(columns=[target_col], errors="ignore")
            preds.loc[regular_mask] = regular_model.predict(X_regular)

        preds = preds.clip(lower=0)  # 안전장치: 음수 방지

        df["pred_delay"] = preds
        logging.info("Prediction complete")
        return df

    except Exception:
        logging.exception("Error in predict_delay")
        df["pred_delay"] = preds  # 에러가 나더라도 0 반환
        return df

# ---------------------------------------------------------------------
# Main 실행부
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        logging.info("Pipeline started")

        # 실행 시 peak season 월 지정
        peak_season_months = [1,4,12]   # 👉 여기서 원하는 월 리스트를 수정하면 전체 반영됨

        query = """
            SELECT branch_cd, group_no, tkt_evt_dt, delay_time as dela_secnd 
            FROM ka_ticket_stats
        """
        rawdf = load_data(query)
        rawdf = preprocessing(rawdf, peak_season_months=peak_season_months)

        train_and_save_models(rawdf, target_col="dela_secnd", peak_season_months=peak_season_months)

        forecast_df = make_future_dataframe('2025-09-01', '2026-12-31', rawdf, peak_season_months=peak_season_months)
        forecast_df = predict_delay(forecast_df, target_col="dela_secnd", peak_season_months=peak_season_months)
        forecast_df.reset_index(drop=False, inplace=True)
        forecast_df.to_csv('forecast_result.csv', index=False)

        logging.info("Forecasting pipeline complete. Result saved to forecast_result.csv")

    except Exception as e:
        logging.error("Pipeline failed")
        logging.error(traceback.format_exc())
        sys.exit(1)
