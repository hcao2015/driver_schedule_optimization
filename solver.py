from argparse import ArgumentParser
import pandas as pd
import numpy as np
from constants import average_hourly_wage, day_of_week_map, neighborhood_map
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary


def get_probablity_new_trip(d_max, d_min, d_avg):
    """
    :param d_max maximum pickup counts for current region and time block.
    :param d_min minimum pickup counts for current region and time block.
    :param d_avg average pickup counts for current region and time block.
    :return: probability new trip happens, range between 0.5 and 1
    """
    p_min = 0.5
    p_max = 1
    return p_min + ((p_max-p_min)/(d_max - d_min)) * (d_avg - d_min)


# TODO: Prob can switch to matrix multiplication for faster?
def get_expected_revenue(day_available, time_available, location_available, df_filtered):
    """
    Return expected revenue from probability of new trip and minimum of trips can be taken within 1 hour
    :param df_filtered:
    :param location_available:
    :param time_available:
    :param day_available:
    :return:
    For example: {
        0: [28, ....],
        3: [21, ....],
        4: [18, ....],
    }
    """
    expected_revenue = np.zeros(shape=(7, 24, 19))
    for day in day_available:
        for hour in time_available:
            for location in location_available:
                df_filter_row = df_filtered[(df_filtered["day_of_week"] == day) &
                                          (df_filtered["hour"] == hour) &
                                          (df_filtered["nhood_id"] == location)]

                # This is min number of trips per hour
                min_num_trips = 3600 / df_filter_row["max_travel_time"]
                probability_new_trip = get_probablity_new_trip(df_filter_row["max_pickups"],
                                                               df_filter_row["min_pickups"],
                                                               df_filter_row["avg_pickups"], )
                expected_revenue[day][hour][location] = probability_new_trip * min_num_trips * average_hourly_wage
    return expected_revenue


def solver(data):
    time_available = data["time_available"]
    day_available = data["day_available"]
    location_available = data["location_available"]
    max_total_hours = data["max_total_hours"]

    df = pd.read_csv("cleaned_data/uber_demand_travel_times.csv")
    df_filtered = df[df["hour"].isin(time_available) &
                        df["day_of_week"].isin(day_available) &
                        df["nhood_id"].isin(location_available)].reset_index()

    # Calculate the expected revenue
    expected_revenue = get_expected_revenue(day_available, time_available, location_available,
                                            df_filtered)

    # Initialize
    # There's 19 neighborhoods for j and 24 time blocks for i.
    # There's 7 days of a week
    dim_j = range(0, 19)
    dim_i = range(0, 24)
    dim_k = range(0, 7)

    # time_available_{i}=1 indicates that the driver is available to drive at time block i, and 0 otherwise.
    time_available = [0] * 24
    for time_slot in data["time_available"]:
        time_available[time_slot] = 1

    # location_available_{j} = 1 indicate the driver is willing to drive in nhood j, and 0 otherwise.
    location_available = [0] * 19
    for nhood in data["location_available"]:
        location_available[nhood] = 1

    # day_available_{k} = 1 indicates the driver is available on day k, and 0 otherwise
    day_available = [0] * 7
    for day in data["day_available"]:
        day_available[day] = 1

    # Create the model
    model = LpProblem(name="uber-scheduling", sense=LpMaximize)

    # The decision variables x is a matrix of size 7x19x24
    x = LpVariable.matrix('x', (dim_k, dim_i, dim_j), 0, 1, LpBinary)

    # Add the constraints
    # Constraint 1: Time Availability of driver
    # Constrain 2: Location Availability of the driver
    # Constrain 3: Day Available of the driver
    for k in dim_k:
        for i in dim_i:
            for j in dim_j:
                model += (x[k][i][j] <= time_available[i], f"Time Availability Constraint: {k} {i} {j}")
                model += (x[k][i][j] <= location_available[j], f"Location Availability Constraint: {k} {i} {j}")
                model += (x[k][i][j] <= day_available[k], f"Day Availability Constrain: {k} {i} {j}")

    # Constraint 4: Maximum Time Per Week Constraint
    model += (lpSum([x[k][i][j] for k in dim_k for i in dim_i for j in dim_j]) <= data["max_total_hours"],
              "Maximum Time Per Week Constraint")

    # Constraint 5: Only work at one neighborhood at a time
    for k in dim_k:
        for i in dim_i:
            model += (lpSum([x[k][i][j] for j in dim_j]) <= 1, f"Only work at one neighborhood at a time: {k} {i}")

    # Objective function: Maximize revenue
    model += lpSum([expected_revenue[k] * x[k] for k in dim_k])

    # Solve
    status = model.solve()

    print(f"Objective Value: {model.objective.value()}")

    print('Intepret schedule as: \n')
    for var in model.variables():
        if var.value() == 1:
            var_nums = [int(num) for num in var.name.split('_')[1:]]
            print(
                f"On {day_of_week_map[var_nums[0]]}, at time {var_nums[1]}, "
                f"works at neighborhood {neighborhood_map[var_nums[2]]} ")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Build optimized schedule for Uber driver in one week work"
    )

    parser.add_argument(
        "--max-hours-weekly",
        help="Maximum number of hours drivers can work per week. For example, 12",
        type=int,
    )

    parser.add_argument(
        "--time-available",
        help="Time slots that driver is available to work. Values need to be from 0-23 and in a list. "
             "For example, 16 17 18 19 represents from 4pm-8pm",
        nargs='+', type=int
    )

    parser.add_argument(
        "--day-available",
        help="Days of the week that driver is available to work. Values need to be from 0-6 and in a list. "
             "For example, 0 1 3 4 represents Monday, Tuesday, Thursday, Friday",
        nargs='+', type=int
    )

    parser.add_argument(
        "--location-available",
        help="SanFrancisco locations that driver is available to work. Values need to be from 0-18 and in a list. "
             "For example, 0 9 18 represents Castro/Upper Market, Mission, Western SoMa",
        nargs='+', type=int
    )

    args = parser.parse_args()

    data = {
        "max_total_hours": args.max_hours_weekly,
        "day_available": args.day_available,
        "time_available": args.time_available,
        "location_available": args.location_available
    }

    if args.max_hours_weekly and args.day_available and args.time_available and args.location_available:
        print("Input data is: \n", data)
        solver(data)
    else:
        # If no data, we run the example case
        from test_cases import example_one
        solver(example_one)

