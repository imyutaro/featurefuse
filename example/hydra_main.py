import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from feature.generate import SelectedFeature

log = logging.getLogger(__name__)


@dataclass
class MyFeatures:
    from naive_feature import NaiveFeature1, NaiveFeature2, NaiveFeature3, NaiveFeature4

    NaiveFeature1: NaiveFeature1 = NaiveFeature1()
    NaiveFeature2: NaiveFeature2 = NaiveFeature2()
    NaiveFeature3: NaiveFeature3 = NaiveFeature3()
    NaiveFeature4: NaiveFeature4 = NaiveFeature4()


@hydra.main(config_path="hydra_config", config_name="default")
def main(feature_config: DictConfig) -> None:

    data_url = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    air_passengers_data = pd.read_csv(data_url)
    log.info(air_passengers_data)

    air_passengers_data_1000 = air_passengers_data
    air_passengers_data_1000["#Passengers"] = (
        air_passengers_data_1000["#Passengers"] + 1000
    )
    log.info(air_passengers_data_1000)

    # 実装した特徴量クラスの辞書
    fe_dict = asdict(MyFeatures())

    # 特徴量の作成
    feature, description = SelectedFeature.generate(
        feature_config,
        fe_dict,
        join_key="Month",
        air_passengers_data=air_passengers_data,
        air_passengers_data_1000=air_passengers_data_1000,
    )

    log.info(feature)
    log.info(description)

    output_dir = Path(os.getcwd())
    output_feature = output_dir / "feature.csv"
    feature.to_csv(output_feature, index=False)

    output_description = output_dir / "description.csv"
    description.to_csv(output_description, index=False)


if __name__ == "__main__":
    main()
