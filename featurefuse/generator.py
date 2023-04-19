import logging
import textwrap
from typing import Union

import pandas as pd

log = logging.getLogger(__name__)


def run(
    feature_config: dict,
    fe_dict: dict,
    join_key: str,
    **kwargs,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Make features which are specified in feature_config.

    Args:
        feature_config (dict): feature info (making feature list and feature params) which you want to make
        # TODO: Can I use just feature_config?
        fe_dict (dict): List of implemented features. Here you must instantiate each feature class.
                        You can only specify feature from this listed feature in config.

    Returns:
        Union[pd.DataFrame, pd.DataFrame]: Made feature DataFrame and made feature description.
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
            message = (
                f"dropped columns because of duplicated column name : {dropped_cols}"
            )
            log.info(message)

        if len(joined_df) != len(df):
            error_msg = textwrap.indent(
                textwrap.dedent(
                    f"""
            Inavalid numbers of rows.
            Output: {len(df)}
            Expected: {len(joined_df)}
            """
                ),
                " " * 3,
            )
            raise ValueError(error_msg)

    descriptions = pd.concat(descriptions, axis="rows", ignore_index=True)

    return joined_df, descriptions
