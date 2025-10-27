import os

from keras_preprocessing.sequence import pad_sequences

from Parse import parse_mooc_args
import pandas as pd
import numpy as np
from tqdm import tqdm


def trn_val_split(args):
    all_click_df = pd.read_csv(args.data_path + 'all_ratings.csv')
    # 添加原始顺序列以确保排序稳定性
    all_click_df['original_order'] = range(len(all_click_df))

    all_click = all_click_df
    num_student = all_click.student_id.nunique()
    all_user_ids = all_click.student_id.unique()
    # 替换为使用包含原始顺序的排序
    all_click = all_click.sort_values(['student_id', 'date', 'original_order'])

    sample_user_ids = np.random.choice(all_user_ids, size=int(num_student * 0.2), replace=False)

    test_pos = all_click[all_click['student_id'].isin(sample_user_ids)]
    train_pos = all_click[~all_click['student_id'].isin(sample_user_ids)]

    # 确保排序正确，使用原始顺序处理同一日期的情况
    test_pos = test_pos.sort_values(['student_id', 'date', 'original_order'])
    test_ans = test_pos.groupby('student_id').tail(1)
    test_pos = test_pos.groupby('student_id').apply(lambda x: x[:-1]).reset_index(drop=True)

    train_data = pd.concat([train_pos, test_pos])
    train_data = train_data.sort_values(['student_id', 'date', 'original_order'])

    train_data.to_csv(args.data_path + 'train_data.csv', index=False, sep=",")
    test_pos.to_csv(args.data_path + 'test_pos.csv', index=False, sep=",")
    test_ans.to_csv(args.data_path + 'test_ans.csv', index=False, sep=",")
    print("The dataset has been split")
    return all_click.courses.unique()


# 将输入的数据进行padding，使得序列特征的长度都一致
def gen_model_input_test(train_set, seq_max_len=20):
    student_id = np.array([line[0] for line in train_set])
    courses = [line[1] for line in train_set]
    category_id = [line[2] for line in train_set]
    candidate = np.array([line[3] for line in train_set])
    candidate_cate = np.array([line[4] for line in train_set])
    label = np.array([line[5] for line in train_set])
    train_hist_len = np.array([line[6] for line in train_set])

    courses_seq_pad = pad_sequences(courses, maxlen=seq_max_len, padding='post', truncating='pre', value=0)
    courses_str = [",".join([str(v) for v in line]) for line in courses_seq_pad]
    category_id_seq_pad = pad_sequences(category_id, maxlen=seq_max_len, padding='post', truncating='pre', value=0)
    category_id_str = [",".join([str(v) for v in line]) for line in category_id_seq_pad]

    train_model_input = {"student_id": student_id, "courses": courses_str, "category_id": category_id_str
        , "candidate": candidate, "candidate_cate": candidate_cate, "label": label, "hist_len": train_hist_len}
    train_data_transformed = pd.DataFrame(train_model_input)
    # 每个用户只保留最近10次正例
    train_data_transformed = train_data_transformed.groupby(['student_id']).tail(100)
    train_data_transformed = train_data_transformed.reset_index(drop=True)
    return train_data_transformed

def gen_model_input_train(train_set, seq_max_len=20):
    student_id = np.array([line[0] for line in train_set])
    courses = [line[1] for line in train_set]
    category_id = [line[2] for line in train_set]
    candidate = np.array([line[3] for line in train_set])
    candidate_cate = np.array([line[4] for line in train_set])
    label = np.array([line[5] for line in train_set])
    train_hist_len = np.array([line[6] for line in train_set])

    courses_seq_pad = pad_sequences(courses, maxlen=seq_max_len, padding='post', truncating='pre', value=0)
    courses_str = [",".join([str(v) for v in line]) for line in courses_seq_pad]
    category_id_seq_pad = pad_sequences(category_id, maxlen=seq_max_len, padding='post', truncating='pre', value=0)
    category_id_str = [",".join([str(v) for v in line]) for line in category_id_seq_pad]

    train_model_input = {"student_id": student_id, "courses": courses_str, "category_id": category_id_str
        , "candidate": candidate, "candidate_cate": candidate_cate, "label": label, "hist_len": train_hist_len}
    train_data_transformed = pd.DataFrame(train_model_input)
    # 每个用户只保留最近10次正例
    train_data_transformed = train_data_transformed.groupby(['student_id']).tail(3000)
    train_data_transformed = train_data_transformed.reset_index(drop=True)
    return train_data_transformed

def gen_train_data_set(data, all_item_ids, cid2cateid_dict, negsample=0):
    # 此部分代码暂未公开
    # 此部分代码暂未公开
    # 此部分代码暂未公开

    return train_set



def gen_test_data_set(data, all_item_ids, cid2cateid_dict, negsample=0):
    data.sort_values(['student_id', 'date', 'original_order'], inplace=True)
    item_ids = all_item_ids
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('student_id')):
        pos_course_list = hist['courses'].tolist()
        pos_cate_list = hist['category_id'].tolist()
        # 确保每个学生仅处理最后一次点击
        if len(pos_course_list) == 0:
            continue
        # 历史序列为除最后一个之外的所有课程
        course_hist = pos_course_list[:-1]
        cate_hist = pos_cate_list[:-1]
        candidate = pos_course_list[-1]
        candidate_cate = pos_cate_list[-1]

        # 添加正样本
        test_set.append((reviewerID, course_hist, cate_hist, candidate, candidate_cate, 1, len(course_hist)))

        # 生成负样本
        pos_set = set(pos_course_list)
        candidate_set = list(set(item_ids) - pos_set)
        neg_samples = np.random.choice(candidate_set, size=negsample, replace=len(candidate_set) < negsample)

        for neg_course in neg_samples:
            neg_cate = cid2cateid_dict.get(neg_course, 0)
            test_set.append((reviewerID, course_hist, cate_hist, neg_course, neg_cate, 0, len(course_hist)))

    return test_set


def get_cid2cateid_dict(courses_info):
    cid2cateid_dict = {}
    for _, value in courses_info.iterrows():
        cid2cateid_dict[value['id']] = int(value['category_id'])
    return cid2cateid_dict


def merge_training_data(train_data: pd.DataFrame, output_path: str):
    """
    将每组1条正样本 + 20条负样本转换为5条新训练数据（1正4负格式），并保存为CSV。

    参数：
    - train_data: 原始训练数据 DataFrame
    - output_path: 保存合并结果的CSV路径
    """
    merged_rows = []
    group_size = 21  # 每组数据包含1正+20负

    assert len(train_data) % group_size == 0, f"总行数必须是{group_size}的倍数"

    for i in range(0, len(train_data), group_size):
        group = train_data.iloc[i:i + group_size]
        pos_sample = group[group['label'] == 1]
        neg_samples = group[group['label'] == 0]

        # 校验
        if len(pos_sample) != 1 or len(neg_samples) != 20:
            raise ValueError(f"第{i}到{i + group_size}行不满足1正20负的规则")

        pos_row = pos_sample.iloc[0]

        # 拆分为5组，每组4个负样本
        for j in range(5):
            neg_part = neg_samples.iloc[j * 4: (j + 1) * 4]
            neg_candidate = ",".join(map(str, neg_part['candidate'].tolist()))
            neg_category = ",".join(map(str, neg_part['candidate_cate'].tolist()))

            merged_rows.append({
                'student_id': pos_row['student_id'],
                'courses': pos_row['courses'],
                'category_id': pos_row['category_id'],
                'candidate': pos_row['candidate'],
                'candidate_cate': pos_row['candidate_cate'],
                'label': pos_row['label'],
                'hist_len': pos_row['hist_len'],
                'neg_candidate': neg_candidate,
                'neg_category': neg_category
            })

    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_csv(output_path, index=False)
    print(f"合并后的数据已保存到：{output_path}")

def merge_test_data(test_data: pd.DataFrame, output_path: str):
    """
    将每组1条正样本 + 99条负样本整合为1条新测试数据（1正99负），并保存为CSV。

    参数：
    - test_data: 原始测试数据 DataFrame
    - output_path: 保存合并结果的CSV路径
    """
    merged_rows = []
    group_size = 100  # 每组数据包含1正+99负

    assert len(test_data) % group_size == 0, f"总行数必须是{group_size}的倍数"

    for i in range(0, len(test_data), group_size):
        group = test_data.iloc[i:i + group_size]
        pos_sample = group[group['label'] == 1]
        neg_samples = group[group['label'] == 0]

        # 校验
        if len(pos_sample) != 1 or len(neg_samples) != 99:
            raise ValueError(f"第{i}到{i + group_size}行不满足1正99负的规则")

        pos_row = pos_sample.iloc[0]

        # 整合99个负样本
        neg_candidate = ",".join(map(str, neg_samples['candidate'].tolist()))
        neg_category = ",".join(map(str, neg_samples['candidate_cate'].tolist()))

        merged_rows.append({
            'student_id': pos_row['student_id'],
            'courses': pos_row['courses'],
            'category_id': pos_row['category_id'],
            'candidate': pos_row['candidate'],
            'candidate_cate': pos_row['candidate_cate'],
            'label': pos_row['label'],
            'hist_len': pos_row['hist_len'],
            'neg_candidate': neg_candidate,
            'neg_category': neg_category
        })

    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_csv(output_path, index=False)
    print(f"测试数据合并完成，保存到：{output_path}")


if __name__ == '__main__':
    args = parse_mooc_args()
    courses_info = pd.read_csv(args.data_path + 'courses_info_with_pre.csv')
    cid2cateid_dict = get_cid2cateid_dict(courses_info)

    #判断分割文件是否存在，若有直接读取，不再 trn_val_split
    if os.path.exists(args.data_path + 'train_data.csv') and \
       os.path.exists(args.data_path + 'test_pos.csv') and \
       os.path.exists(args.data_path + 'test_ans.csv'):
        print("检测到已有分割数据，直接读取")
        train_data = pd.read_csv(args.data_path + 'train_data.csv')
        all_click_df = pd.read_csv(args.data_path + 'all_ratings.csv')
        all_item_ids = all_click_df['courses'].unique()
    else:
        print("未检测到分割数据，重新进行数据划分")
        all_item_ids = trn_val_split(args)
        train_data = pd.read_csv(args.data_path + 'train_data.csv')

    train_set = gen_train_data_set(train_data, all_item_ids, cid2cateid_dict, negsample=20)
    train_data_transformed = gen_model_input_train(train_set, seq_max_len=20)
    # 保存训练集
    train_data_transformed.to_csv(args.data_path + 'train_data_transformed_01.csv', index=False, sep=",")

    # 测试集生成代码调整
    test_ans = pd.read_csv(args.data_path + 'test_ans.csv')
    test_pos = pd.read_csv(args.data_path + 'test_pos.csv')
    # 合并测试答案和测试历史，确保包含原始顺序
    test_data = pd.concat([test_pos, test_ans]).sort_values(['student_id', 'date', 'original_order'])
    test_set = gen_test_data_set(test_data, all_item_ids, cid2cateid_dict, negsample=99)
    test_data_transformed = gen_model_input_test(test_set, seq_max_len=20)
    test_data_transformed.to_csv(args.data_path + 'test_data_transformed.csv', index=False, sep=",")

    train_data = pd.read_csv(args.data_path + 'train_data_transformed_01.csv')
    merge_training_data(train_data, args.data_path + 'train_data_transformed_02.csv')


