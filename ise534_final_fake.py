import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm
import random
import multiprocessing
import pickle

def save_point(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_point(filename):
    with open(filename, "rb") as f:
        res = pickle.load(f)
    return res

def filter_each(cord, raid=1, criterion=0.95):
    for i in range(cord.shape[0]):
        dists = np.sqrt(np.sum(np.square(cord - cord[i]), axis=1))
        in_sample = np.count_nonzero(dists < raid)
        ratio = (in_sample - 1) / (cord.shape[0] - 1)
        if ratio > criterion:
            return True
    return False

def distance(x1, x2):
    dist = np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
    return dist

def find_move_fixed(data, unique):
    move = set()
    fixed = set()
    for i in tqdm(range(len(unique))):
        count_bool = []
        curMac = unique[i]
        curMac_data = data[data['ClientMacAddr'] == curMac]
        unique_date = curMac_data['date'].unique()
        for unique_dt in range(len(unique_date)):
            single_day_data = curMac_data[curMac_data['date'] == unique_date[unique_dt]]
            coords_m = single_day_data.iloc[:, [9,10]].values
            ratio = filter_each(coords_m, raid=2, criterion=0.95)
            count_bool.append(ratio)
        if np.count_nonzero(count_bool)/len(count_bool) >= 0.9:
            fixed.add(curMac)
        else:
            move.add(curMac)
    return [move, fixed]

def daily_moveable(data_clean,date,move):
    daily_move = set()
    daily = data_clean[data_clean['date'] == date]
    unique_mac = np.unique(daily['ClientMacAddr'])
    print(unique_mac)
    for each in unique_mac:
        if each in move:
            daily_move.add(each)
    ground = list(daily[daily["Level"] == "1st Floor"]["ClientMacAddr"])
    ground_move = []
    for ip in daily_move:
        if ip in ground:
            ground_move.append(ip)
    return daily_move, ground_move

if __name__ == "__main__":
    data = load_point("./data/data_datetime.pkl")

    m = Basemap(projection='lcc', resolution=None,
                width=8E6, height=8E6,
                lat_0=51, lon_0=-0.93)

    true_x, true_y = m(data["lng"], data["lat"])
    data["true_x"] = true_x
    data["true_y"] = true_y
    offset_x, offset_y = m(-0.9326021, 51.4605103)
    x_cor, y_cor = data['true_x'] - offset_x, data['true_y'] - offset_y
    data['x_cor'] = x_cor
    data['y_cor'] = y_cor

    datacp = data.copy()
    include = datacp['x_cor'].apply(lambda x: x < 0) & datacp['y_cor'].apply(lambda x: x > 0)
    data_clean = datacp.drop(datacp[include].index)

    data_clean['add_count'] = data_clean.groupby(['ClientMacAddr', 'date'])['date'].transform('count')
    filterclient = data_clean['add_count'].map(lambda x: x > 20)
    data_clean = data_clean[filterclient]

    test_data = pd.read_excel('./data/ise534 fake datanew_REVISED.xlsx')
    test_data = test_data.iloc[:, 1:]
    test_data["date"] = test_data["localtime"].apply(lambda x: x[:10])
    test_data["datetime"] = test_data["localtime"].apply(
        lambda x: datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S')).apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

    m = Basemap(projection='lcc', resolution=None,
                width=8E6, height=8E6,
                lat_0=51, lon_0=-0.93)

    true_x, true_y = m(test_data["lng"], test_data["lat"])
    test_data["true_x"] = true_x
    test_data["true_y"] = true_y
    offset_x, offset_y = m(-0.9326021, 51.4605103)
    x_cor, y_cor = test_data['true_x'] - offset_x, test_data['true_y'] - offset_y
    test_data['x_cor'] = x_cor
    test_data['y_cor'] = y_cor

    include = test_data['x_cor'].apply(lambda x: x < 0) & test_data['y_cor'].apply(lambda x: x > 0)
    test_data = test_data.drop(test_data[include].index)

    move, fixed = load_point("./data/move_fixed_tmp.pkl")

    ########
    random.seed(534)
    # 判断test case的日期是否有足够moveable device的出现 大于我们问题总数
    date = list(set(test_data["date"]))
    # for i in range(len(date)):
    #     tmp_datetime = pd.to_datetime(date[i])
    #     date[i] = tmp_datetime
    daily_move_people, ground_move = daily_moveable(data_clean=data_clean, date=date[0],
                                                    move=move)  # 发生问题当天的move devices，ground floor move devices
    daily_move_people = list(daily_move_people)
    security_num = 3  # 对应于第一类问题的安保人数
    tech_num = 3  # 对应于第二类问题的技术人数
    maintain_num = 3  # 对应于第三类问题的后勤人数
    solve_time = 15
    cur_df = data_clean[data_clean['date'] == date[0]]  # 发生问题的当天
    total_move_daily = len(daily_move_people)  # 发生问题当天可用的全部move的个数

    if total_move_daily < security_num + tech_num + maintain_num:
        print(date, "Nothing happened.")
    else:
        useful_date = date
        total_ground_move = len(ground_move)  # 一层全部move的个数
        tech_chosen = random.sample(range(0, total_move_daily), tech_num + maintain_num)[:tech_num]  # 选出的tech人员
        maintain_chosen = random.sample(range(0, total_move_daily), tech_num + maintain_num)[tech_num:]  # 选出的maintain人员
        security_chosen = random.sample(range(0, total_ground_move),
                                        security_num + tech_num + maintain_num)  # 选出的security人员

        problem1 = list(test_data[test_data["Emergency Level"] == "Class I"].sort_values(by="datetime", ascending=True)[
                            "ClientMacAddr"])
        problem2 = list(
            test_data[test_data["Emergency Level"] == "Class II"].sort_values(by="datetime", ascending=True)[
                "ClientMacAddr"])
        problem3 = list(
            test_data[test_data["Emergency Level"] == "Class III"].sort_values(by="datetime", ascending=True)[
                "ClientMacAddr"])

        problem1_time = list(
            test_data[test_data["Emergency Level"] == "Class I"].sort_values(by="datetime", ascending=True)["datetime"])
        problem2_time = list(
            test_data[test_data["Emergency Level"] == "Class II"].sort_values(by="datetime", ascending=True)[
                "datetime"])
        problem3_time = list(
            test_data[test_data["Emergency Level"] == "Class III"].sort_values(by="datetime", ascending=True)[
                "datetime"])

        tech = []
        for j in range(tech_num):
            ip = daily_move_people[tech_chosen[j]]
            tech.append(ip)
        maintain = []
        for j in range(maintain_num):
            ip = daily_move_people[maintain_chosen[j]]
            maintain.append(ip)
        security = []
        for j in range(security_num):
            ip = ground_move[security_chosen[j]]
            if ip not in tech and ip not in maintain:
                security.append(ip)
            if len(security) == security_num:
                break

        print("Current date is " + str(useful_date[0]) + "\n")

        start_time = str(useful_date[0]) + " 09:00:00"
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

        print("Problem1:")
        availablity = [0] * security_num
        for j in range(len(problem1)):
            time = problem1_time[j]
            problem_ip = problem1[j]
            problem_level = str(
                test_data[(test_data["ClientMacAddr"] == problem_ip) & (test_data["datetime"] == time)]["Level"])
            problem_location = list(
                test_data[(test_data["ClientMacAddr"] == problem_ip) & (test_data["datetime"] == time)].iloc[0, 11:13])
            shortest_dist = float('inf')
            shortest_index = -1
            for k in range(security_num):
                if type(availablity[k]) != int:
                    if (availablity[k] + timedelta(minutes=solve_time)).strftime('%Y-%m-%d %H:%M:%S') > time:
                        continue
                solver_ip = security[k]
                solver_level = str(
                    cur_df[(cur_df["ClientMacAddr"] == solver_ip) & (cur_df["datetime"] >= time)].sort_values(
                        by="datetime", ascending=True).iloc[0, 1])
                solver_location = list(
                    cur_df[(cur_df["ClientMacAddr"] == solver_ip) & (cur_df["datetime"] >= time)].sort_values(
                        by="datetime", ascending=True).iloc[0, 10:12])
                if solver_level == problem_level:
                    dist = distance(solver_location, problem_location)
                else:
                    dist = distance(solver_location, [0, 0]) + distance(problem_location, [0, 0])
                if dist < shortest_dist:
                    shortest_dist = dist
                    shortest_index = k

            if shortest_index == -1:
                print("Problem cannot be solved")

            select_solver = security[shortest_index]
            availablity[shortest_index] = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            print("The solver for the " + str(j) + "th problem1 is " + select_solver)

        print("Problem2:")
        availablity = [0] * tech_num
        for j in range(len(problem2)):
            time = problem2_time[j]
            problem_ip = problem2[j]
            problem_level = str(
                test_data[(test_data["ClientMacAddr"] == problem_ip) & (test_data["datetime"] == time)]["Level"])
            problem_location = list(
                test_data[(test_data["ClientMacAddr"] == problem_ip) & (test_data["datetime"] == time)].iloc[0, 11:13])
            shortest_dist = float('inf')
            shortest_index = -1
            for k in range(tech_num):
                if type(availablity[k]) != int:
                    if (availablity[k] + timedelta(minutes=solve_time)).strftime('%Y-%m-%d %H:%M:%S') > time:
                        continue
                solver_ip = tech[k]
                solver_level = str(
                    cur_df[(cur_df["ClientMacAddr"] == solver_ip) & (cur_df["datetime"] >= time)].sort_values(
                        by="datetime", ascending=True).iloc[0, 1])
                solver_location = list(
                    cur_df[(cur_df["ClientMacAddr"] == solver_ip) & (cur_df["datetime"] >= time)].sort_values(
                        by="datetime", ascending=True).iloc[0, 10:12])
                if solver_level == problem_level:
                    dist = distance(solver_location, problem_location)
                else:
                    dist = distance(solver_location, [0, 0]) + distance(problem_location, [0, 0])
                if dist < shortest_dist:
                    shortest_dist = dist
                    shortest_index = k

            if shortest_index == -1:
                print("Problem cannot be solved")

            select_solver = tech[shortest_index]
            availablity[shortest_index] = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            print("The solver for the " + str(j) + "th problem2 is " + select_solver)

        print("Problem3:")
        availablity = [0] * maintain_num
        for j in range(len(problem3)):
            time = problem3_time[j]
            problem_ip = problem3[j]
            problem_level = str(
                test_data[(test_data["ClientMacAddr"] == problem_ip) & (test_data["datetime"] == time)]["Level"])
            problem_location = list(
                test_data[(test_data["ClientMacAddr"] == problem_ip) & (test_data["datetime"] == time)].iloc[0, 11:13])
            shortest_dist = float('inf')
            shortest_index = -1
            for k in range(maintain_num):
                if type(availablity[k]) != int:
                    if (availablity[k] + timedelta(minutes=solve_time)).strftime('%Y-%m-%d %H:%M:%S') > time:
                        continue
                solver_ip = maintain[k]
                solver_level = str(
                    cur_df[(cur_df["ClientMacAddr"] == solver_ip) & (cur_df["datetime"] >= time)].sort_values(
                        by="datetime", ascending=True).iloc[0, 1])
                solver_location = list(
                    cur_df[(cur_df["ClientMacAddr"] == solver_ip) & (cur_df["datetime"] >= time)].sort_values(
                        by="datetime", ascending=True).iloc[0, 10:12])
                if solver_level == problem_level:
                    dist = distance(solver_location, problem_location)
                else:
                    dist = distance(solver_location, [0, 0]) + distance(problem_location, [0, 0])
                if dist < shortest_dist:
                    shortest_dist = dist
                    shortest_index = k

            if shortest_index == -1:
                print("Problem cannot be solved")

            select_solver = maintain[shortest_index]
            availablity[shortest_index] = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            print("The solver for the " + str(j) + "th problem3 is " + select_solver)

        print()
        print("-----------------------------")
        print()