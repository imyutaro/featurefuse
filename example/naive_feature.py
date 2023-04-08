import pandas as pd

from featurefuse.base import Feature


class NaiveFeature1(Feature):
    def create_feature(
        self,
        air_passengers_data: pd.DataFrame,
        shift_b_from: int,
        shift_b_to: int,
        **kwargs,
    ) -> pd.DataFrame:

        # 利用するカラムのみコピー
        fe = air_passengers_data[["Month", "#Passengers"]].copy(deep=True)

        # lag特徴量（過去{shift_b_from}ヶ月~過去{shift_b_to}ヶ月までの乗客数）
        self.create_description(
            f"#Passengers_b{shift_b_from} ~ #Passengers_b{shift_b_to}",
            f"過去{shift_b_from}ヶ月~過去{shift_b_to}ヶ月までの乗客数",
        )
        for i in range(shift_b_from, shift_b_to + 1):
            fe[f"#Passengers_b{i}"] = fe["#Passengers"].shift(i)

        fe = fe.drop("#Passengers", axis="columns")

        return fe


class NaiveFeature2(Feature):
    def create_feature(
        self,
        air_passengers_data_1000: pd.DataFrame,
        rolling_min: int,
        rolling_max: int,
        rolling_mean: int,
        rolling_median: int,
        **kwargs,
    ) -> pd.DataFrame:

        # 利用するカラムのみコピー
        fe = air_passengers_data_1000[["Month", "#Passengers"]].copy(deep=True)

        # roll特徴量（過去1ヶ月~過去4ヶ月までの乗客数についての特徴量）
        self.create_description(
            f"#Passengers_min_b{rolling_min}", f"過去{rolling_min}ヶ月までの乗客数の最小値"
        )
        self.create_description(
            f"#Passengers_max_b{rolling_max}", f"過去{rolling_max}ヶ月までの乗客数の最大値"
        )
        self.create_description(
            f"#Passengers_mean_b{rolling_mean}", f"過去{rolling_mean}ヶ月までの乗客数の平均値"
        )
        self.create_description(
            f"#Passengers_median_b{rolling_median}", f"過去{rolling_median}ヶ月までの乗客数の中央値"
        )

        # 過去iヶ月までの乗客数の最小値
        fe[f"#Passengers_min{rolling_min}"] = (
            fe[f"#Passengers"].rolling(rolling_min).min()
        )
        # 過去iヶ月までの乗客数の最大値
        fe[f"#Passengers_max{rolling_max}"] = (
            fe[f"#Passengers"].rolling(rolling_max).max()
        )
        # 過去iヶ月までの乗客数の平均値
        fe[f"#Passengers_mean{rolling_mean}"] = (
            fe[f"#Passengers"].rolling(rolling_mean).mean()
        )
        # 過去iヶ月までの乗客数の中央値
        fe[f"#Passengers_median{rolling_median}"] = (
            fe[f"#Passengers"].rolling(rolling_median).median()
        )

        fe = fe.drop("#Passengers", axis="columns")

        return fe


class NaiveFeature3(Feature):
    def create_feature(
        self, air_passengers_data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:

        # 利用するカラムのみコピー
        fe = air_passengers_data[["Month", "#Passengers"]].copy(deep=True)

        tmp = fe.copy(deep=True)
        for i in range(25):
            tmp[f"#Passengers_b{i}"] = fe["#Passengers"].shift(i)

        # 過去の同じ月の乗客の平均
        self.create_description(
            "#Passengers_ma",
            "過去の同じ月の乗客数の平均値",
        )
        fe[f"#Passengers_ma"] = (
            tmp[f"#Passengers"] + tmp[f"#Passengers_b12"] + tmp[f"#Passengers_b24"]
        ) / 3

        fe = fe.drop("#Passengers", axis="columns")

        return fe


class NaiveFeature4(Feature):
    def create_feature(
        self, air_passengers_data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:

        # 利用するカラムのみコピー
        fe = air_passengers_data[["Month", "#Passengers"]].copy(deep=True)

        # lag特徴量（過去1ヶ月~過去24ヶ月までの乗客数）
        self.create_description("aaa", "bbb")
        fe[f"aaa"] = 1000

        fe = fe.drop("#Passengers", axis="columns")

        return fe


class NaiveFeature5(Feature):
    def create_feature(
        self, air_passengers_data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:

        # 利用するカラムのみコピー
        fe = air_passengers_data[["Month", "#Passengers"]].copy(deep=True)

        # lag特徴量（過去1ヶ月~過去24ヶ月までの乗客数）
        self.create_description("aaa", "同じカラムがあったときのテスト用")
        fe[f"aaa"] = 1000

        fe = fe.drop("#Passengers", axis="columns")

        return fe
