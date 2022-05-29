import pandas as pd
from shapely.geometry import shape
import json
import numpy as np
from datetime import datetime


def extract_data():
    # Extract demand data
    trip_stats = pd.read_csv("raw_data/trip_stats_taz.csv")
    taz_boundaries = pd.read_json("raw_data/taz_boundaries.json")
    taz_boundaries["geometry"] = taz_boundaries["geometry"].apply(lambda x: shape(json.loads(x)))
    df = pd.merge(trip_stats, taz_boundaries, how='inner', on='taz')

    # Extract those neighborhoods that are higher than 2% of pickups
    pickups_df = df[["nhood", "pickups"]].groupby(["nhood"]).sum(["pickups"]) \
        .sort_values(by=['pickups'], ascending=False)

    total_pickups = sum(pickups_df["pickups"])
    pickups_df["percent"] = round(pickups_df["pickups"] * 100 / total_pickups, 2)
    selected_nhoods = list(pickups_df[pickups_df["percent"] >= 2].index) + ["Chinatown"]
    selected_df = df[df["nhood"].isin(selected_nhoods)]

    demand_df = selected_df[["nhood", "day_of_week", "hour", "pickups"]] \
        .groupby(["nhood", "day_of_week", "hour"]) \
        .agg(avg_pickups=('pickups', np.mean),
             max_pickups=('pickups', np.max),
             min_pickups=('pickups', np.min)).round(2) \
        .reset_index()

    # Extract Uber movement data
    taz_sf = pd.read_json("raw_data/san_francisco_taz.json")

    tazs = []
    counties = []
    movement_ids = []

    def extract_info(row):
        try:
            taz = int(row["properties"].get("TAZ"))
        except:
            taz = None

        try:
            movement_id = int(row["properties"].get("MOVEMENT_ID"))
        except:
            movement_id = None
        tazs.append(taz)
        counties.append(row["properties"].get("COUNTY"))
        movement_ids.append(movement_id)

    for row in taz_sf["features"]:
        extract_info(row)

    taz_sf["taz"] = tazs
    taz_sf["county"] = counties
    taz_sf["movement_id"] = movement_ids

    taz_sf.drop(labels=["type", "features"], axis=1, inplace=True)
    taz_sf = taz_sf.dropna(subset=['taz', 'movement_id'])

    # Create a map between movement_id and nhood
    taz_nhood = pd.merge(taz_sf, taz_boundaries, how="left", on="taz")
    taz_nhood_mapping = pd.Series(taz_nhood["nhood"].values, index=taz_nhood["movement_id"]).to_dict()

    uber_df = pd.read_csv("raw_data/san_francisco-censustracts-2020-1-All-DatesByHourBucketsAggregate.csv")
    # Only consider intra-neighborhood trips (aka trips within neighborhood or trips where not
    # over 30 mins away from the neighborhood
    uber_df["source_nhood"] = uber_df["sourceid"].apply(lambda x: taz_nhood_mapping.get(x))
    uber_df["dest_nhood"] = uber_df["dstid"].apply(lambda x: taz_nhood_mapping.get(x))
    uber_df = uber_df[((uber_df["source_nhood"] == uber_df["dest_nhood"]) |
                       (uber_df["geometric_mean_travel_time"] <= 30 * 60))
                      & (uber_df["source_nhood"] is not None)
                      & (uber_df["source_nhood"].isin(selected_nhoods))]

    # Create a day_of_week column
    uber_df["day_of_week"] = uber_df.apply(lambda row: datetime(2020, row["month"], row["day"]).weekday(), axis=1)

    uber_df_grouped = uber_df \
        .groupby(["source_nhood", "day_of_week", "start_hour", "end_hour"]) \
        .agg(
        total_travel_time=('mean_travel_time', np.sum),
        avg_travel_time=('mean_travel_time', np.mean),
        min_travel_time=('mean_travel_time', np.min),
        max_travel_time=('mean_travel_time', np.max)
    ).round(2) \
        .reset_index()

    uber_df_grouped.rename(columns={'source_nhood': 'nhood'}, inplace=True)

    # Note: Since we don't have specific hourly time, the travel_time within start and end hour will be
    # used for each hour within it

    uber_df = []
    for index, row in uber_df_grouped.iterrows():
        start_hour = row["start_hour"]
        end_hour = row["end_hour"]
        for hour in range(start_hour, end_hour):
            uber_df.append([row["nhood"], row["day_of_week"], hour, row["avg_travel_time"],
                            row["min_travel_time"], row["max_travel_time"]])

    travel_times_df = pd.DataFrame(uber_df, columns=["nhood", "day_of_week", "hour", "avg_travel_time",
                                                      "min_travel_time", "max_travel_time"])

    # Combined the demand data and travel times into one
    combined = demand_df.merge(travel_times_df, how='outer', on=['nhood', 'day_of_week', 'hour'])

    """
    Fill avg_travel_time, min_travel_time, max_travel_time of the missing time slot as the following order:
    1. Values of the time slot before it
    2. If not, values of the average, min and max of the whole neighborhood, min and max
    """
    nhood_stats = combined[["nhood", "avg_travel_time", "min_travel_time", "max_travel_time"]]
    nhood_stats = nhood_stats.dropna()
    nhood_stats = nhood_stats.groupby(["nhood"]) \
        .agg(avg_travel_time=('avg_travel_time', np.mean),
             min_travel_time=('min_travel_time', np.max),
             max_travel_time=('max_travel_time', np.min)).round(2) \
        .reset_index()

    missing_null_idx = combined[combined.isna().any(axis=1)].index

    for idx in missing_null_idx:
        current_row_data = combined.loc[idx, :]
        current_row_data_hour = current_row_data["hour"]
        current_row_data_nhood = current_row_data["nhood"]

        previous_hour = current_row_data_hour - 1 if current_row_data_hour > 0 else 24
        next_row_data = combined[(combined["hour"] == current_row_data_hour - 1) &
                                 (combined["day_of_week"] == current_row_data["day_of_week"]) &
                                 (combined["nhood"] == current_row_data_nhood)]

        avg_travel_time = next_row_data['avg_travel_time'].values

        if avg_travel_time:
            avg_travel_time = avg_travel_time[0]
            min_travel_time = next_row_data['min_travel_time'].values[0]
            max_travel_time = next_row_data['max_travel_time'].values[0]
        else:
            nhood_stat = nhood_stats[nhood_stats["nhood"] == current_row_data_nhood]
            avg_travel_time = nhood_stat['avg_travel_time'].values
            if avg_travel_time:
                avg_travel_time = avg_travel_time[0]
                min_travel_time = nhood_stat['min_travel_time'].values[0]
                max_travel_time = nhood_stat['max_travel_time'].values[0]
            else:
                avg_travel_time = None
                min_travel_time = None
                max_travel_time = None

        combined.loc[idx, 'avg_travel_time'] = avg_travel_time
        combined.loc[idx, 'min_travel_time'] = min_travel_time
        combined.loc[idx, 'max_travel_time'] = max_travel_time

    # So we drop the nhood Mission Bay since it doesn't have corresponding uber data
    combined.dropna(inplace=True)

    # Make a numeric columns from nhood and hour_bucket so we can plug in our model easier
    combined["nhood_id"] = pd.Categorical(combined["nhood"], categories=combined["nhood"].unique()).codes

    # Save final data
    combined.to_csv("cleaned_data/uber_demand_travel_times.csv", index=False, header=True)


if __name__ == "__main__":
    extract_data()
