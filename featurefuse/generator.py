import logging
import textwrap
from dataclasses import dataclass, field
from typing import Union

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run(
    feature_config: dict,
    fe_dict: dict,
    join_key: str,
    **kwargs,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """feature_configで指定された特徴量を作成する関数

    Args:
        feature_config (DictConfig): 特徴量作成に利用するクラスを記述したconfig(辞書形式)
        fe_dict (dict): 特徴量作成クラスを表す文字列とそれに対応するクラス

    Returns:
        Union[pd.DataFrame, pd.DataFrame]:
        作成した特徴量をMonthで結合したDataFrame, 作成した特徴量の説明を記述したDataFrame
    """

    use_feature = feature_config["use_feature"]

    if "feature_params" in feature_config:
        feature_params = feature_config["feature_params"]
    else:
        feature_params = dict()

    joined_df = pd.DataFrame([])
    descriptions = []
    for fe in use_feature:
        if fe in feature_params and feature_params[fe] is not None:
            df, description = fe_dict[fe].run(**kwargs, **feature_params[fe])
        else:
            df, description = fe_dict[fe].run(**kwargs)

        description = pd.DataFrame.from_dict(description)
        descriptions.append(description)
        if joined_df.empty:
            joined_df = df
        else:
            joined_df = pd.merge(
                joined_df,
                df,
                how="left",
                on=join_key,
                suffixes=["", "_duplicated_columns"],
            )
            joined_df = joined_df.filter(
                regex="^(?!.*_duplicated_columns$)", axis="columns"
            )
            dropped_cols = joined_df.filter(
                regex="(?!.*_duplicated_columns$)", axis="columns"
            ).columns
            message = f"同名のカラムが存在するためにdropしたカラム : {dropped_cols}"
            log.info(message)

        if len(joined_df) != len(df):
            error_msg = textwrap.indent(
                textwrap.dedent(
                    f"""
            作成した特徴量の行数が不正です。
            作成した特徴量の行数：{len(df)}
            期待される行数：{len(joined_df)}
            """
                ),
                " " * 3,
            )
            raise ValueError(error_msg)

    descriptions = pd.concat(descriptions, axis="rows", ignore_index=True)

    return joined_df, descriptions
