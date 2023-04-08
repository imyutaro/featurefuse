import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml
from featurefuse.generator import run


@dataclass
class MyFeatures:
    # List of implemented features. Here you must instantiate each feature class.
    # You can only specify feature from this listed feature in config.
    from naive_feature import (
        NaiveFeature1,
        NaiveFeature2,
        NaiveFeature3,
        NaiveFeature4,
        NaiveFeature5,
    )

    NaiveFeature1: NaiveFeature1 = NaiveFeature1()
    NaiveFeature2: NaiveFeature2 = NaiveFeature2()
    NaiveFeature3: NaiveFeature3 = NaiveFeature3()
    NaiveFeature4: NaiveFeature4 = NaiveFeature4()
    NaiveFeature5: NaiveFeature5 = NaiveFeature5()


def _parse_args():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--feature_config",
        dest="feature_config",
        help="path to config which specify use feature.",
        default="./config/feature.yaml",
        type=str,
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    with open(args.feature_config, mode="r") as f:
        feature_config = yaml.safe_load(f)

    data_url = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    air_passengers_data = pd.read_csv(data_url)
    print(air_passengers_data)

    air_passengers_data_1000 = air_passengers_data
    air_passengers_data_1000["#Passengers"] = (
        air_passengers_data_1000["#Passengers"] + 1000
    )
    print(air_passengers_data_1000)

    # Implemented Feature Class
    fe_dict = asdict(MyFeatures())

    # Make feature
    feature, description = run(
        feature_config,
        fe_dict,
        join_key="Month", # key column to join each feature DataFrame
        air_passengers_data=air_passengers_data,
        air_passengers_data_1000=air_passengers_data_1000,
    )
    print(feature)
    print(description)

    output_dir = Path("./outputs/")
    output_dir.mkdir(exist_ok=True)

    output_feature = output_dir / "feature.csv"
    feature.to_csv(output_feature, index=False, compression="bz2")

    output_description = output_dir / "description.csv"
    description.to_csv(output_description, index=False)

    # Input to model or something
    # model.train(feature)


if __name__ == "__main__":
    main()
