import pandas as pd
import logging
import numpy as np
import collections

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_data(data, threshold):
    filtered_data = {}
    for k, v in data.items():
        if len(v) > threshold:
            filtered_data[k] = v
    return filtered_data


def read_data(data_path, threshold):
    data = pd.read_excel(data_path, sheet_name='Sheet1', header=[0], usecols='A,B,C').fillna(0)
    data_label = {}
    count = 0
    for i in range(len(data)):
        if data.iloc[i]["Abstract"] == 0 or data.iloc[i]["Problem Description"] == 0:
            count += 1
            continue
        text = [data.iloc[i]["Abstract"], data.iloc[i]["Problem Description"], i]
        label = data.iloc[i]["Root Cause"]
        if label not in data_label:
            data_label[label] = [text]
        else:
            data_label[label].append(text)
    filtered_data = filter_data(data_label, threshold=threshold)
    logger.info("drop %d samples" % count)

    logger.info("starting samples eval data")
    train_data = collections.defaultdict(list)
    eval_data = collections.defaultdict(list)

    train_data_nums = 0
    eval_data_nums = 0
    for k, v in filtered_data.items():
        eval_nums = len(v) // 10
        if eval_nums == 0:
            eval_nums = 1
        eval_data_nums += eval_nums
        train_data_nums += len(v) - eval_nums
        indices = np.random.choice(list(range(len(v))), eval_nums)
        for j in range(len(v)):
            if j in indices:
                eval_data[k].append(v[j])
            else:
                train_data[k].append(v[j])
    logger.info('train data nums: %d, eval data nums: %d' % (train_data_nums, eval_data_nums))
    return train_data, eval_data


if __name__ == "__main__":
    a, b = read_data('./data/source_data.xlsx', 5)