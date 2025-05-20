import torch
import os
import pandas as pd
from mmseg.datasets import (OPTIC_dataset, OPTIC_dataset_FT, Fundus, CXR, ISBI_dataset, Polyp_dataset, Polyp,
                            BTMRI_dataset, BUSI_dataset)
from mmseg.baselines import (SourceOnly, TTCS)


def convert_labeled_list(root, csv_list):
    img_list = list()
    label_list = list()
    for csv_file in csv_list:
        data = pd.read_csv(os.path.join(root, csv_file))
        img_list += data['image'].tolist()
        label_list += data['mask'].tolist()
    return img_list, label_list


def build_dataset(root, cfg, domain=None, baseline='other'):
    if domain == 'cxr':
        dataset = CXR(root)

    elif domain == 'ISBI':
        import pandas as pd
        df_list = []
        for i, file_name in enumerate([cfg.dataset.list_file, cfg.dataset.list_file.replace('Test', 'Training')]):

            df = pd.read_csv(file_name)
            df_list.append(df)

        df_merged = pd.concat(df_list, axis=0, ignore_index=True)
        dataset = ISBI_dataset(root, df_merged)
        # dataset = ISIC_dataset(root, cfg.dataset.list_file)

    elif domain in Fundus:

        data_csv = []
        if domain != 'REFUGE_Valid':
            data_csv.append(domain + '_train.csv')
            data_csv.append(domain + '_test.csv')
        else:
            data_csv.append(domain + '.csv')  # REFUGE_Valid没有分train and test
        ts_img_list, ts_label_list = convert_labeled_list(root, data_csv)
        if baseline == 'SAM_FT':
            dataset = OPTIC_dataset_FT(root, ts_img_list, ts_label_list)  # FT就用这个，否则改回来
        else:
            dataset = OPTIC_dataset(root, ts_img_list, ts_label_list)

    elif domain in Polyp:
        data_csv = [domain + '_train.csv', domain + '_test.csv']
        ts_img_list, ts_label_list = convert_labeled_list(root, data_csv)
        dataset = Polyp_dataset(root, ts_img_list, ts_label_list)

    elif domain == 'BT_MRI':
        dataset = BTMRI_dataset(root)

    elif domain == 'BUSI':
        dataset = BUSI_dataset(root)

    return dataset


def build_solver(baseline):
    solver_dicts = {
        'SourceOnly': SourceOnly,
        'TTCS': TTCS,
        # 'MemorySelection': MemorySelection,
    }
    return solver_dicts[baseline]