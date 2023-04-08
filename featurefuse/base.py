import logging
import time
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Union

import pandas as pd

log = logging.getLogger(__name__)


@contextmanager
def timer(name):
    t0 = time.time()
    log.info(f"[{name}] start")
    yield
    log.info(f"[{name}] done in {time.time() - t0:.0f} s")


class Feature(metaclass=ABCMeta):
    """特徴量作成クラスの基底クラス。
    特徴量作成クラスはこのクラスを継承して、create_feature()に特徴量作成のコードを記述する。
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.descriptions = {"特徴量作成クラス名": [], "特徴量カラム名": [], "説明": []}

    @abstractmethod
    def create_feature(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, **kwargs) -> Union[pd.DataFrame, dict]:
        """特徴量作成の際に実行する関数。
        create_feature()の返り値のDataFrameにはshiji_dateを含んでないといけない。

        Raises:
            ValueError: create_feature()関数が返すDataFrameにshiji_dateカラムが無い場合にエラー

        Returns:
            Union[pd.DataFrame, dict]: 特徴量DataFrameと作成した特徴量に関する説明を記述した辞書
        """
        with timer(self.name):
            fe = self.create_feature(**kwargs)
        return fe, self.descriptions

    def create_description(self, col_name: str, description: str) -> None:
        """作成した特徴量の説明をインスタンス変数に追記する関数

        Args:
            col_name (str): descriptionに関連するカラム名
            description (str): col_nameの説明
        """
        existed_cols = self.descriptions["特徴量カラム名"]
        if col_name in existed_cols:
            return

        self.descriptions["特徴量作成クラス名"].append(self.name)
        self.descriptions["特徴量カラム名"].append(col_name)
        self.descriptions["説明"].append(description)
