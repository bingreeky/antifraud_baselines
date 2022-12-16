import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


def calcu_trading_entropy(
    data_2: pd.DataFrame
) -> float:
    """calculate trading entropy of given data

    Args:
        data (pd.DataFrame): 2 cols, Amount and Type

    Returns:
        float: entropy
    """
    # if empty
    if len(data_2) == 0:
        return 0

    amounts = np.array([data_2[data_2['Type'] == type]['Amount'].sum()
                       for type in data_2['Type'].unique()])
    if amounts.sum() == 0:
        print(f"data: {data_2}")
        print(f"amounts: {amounts}")
        print(f"amounts sum: {amounts.sum()}")
    proportions = amounts / amounts.sum()
    ent = -np.array([proportion*np.log(1e-5 + proportion)
                    for proportion in proportions]).sum()

    return ent


def span_transaction_2d(
    data: pd.DataFrame,
    time_windows: list
) -> np.ndarray:
    """transform transction record to feature matrices \\ 
    not concerning train or test

    Args:
        data (pd.DataFrame): transaction records
        time_windows (list): len of temporal axis

    Returns:
        np.ndarray: (sample_num, |TimeWindows|, feat_num) transaction feature matrices
    """

    data = data[data['Amount'] != 0]  # (117498, 4430)
    # time_windows = [1, 3, 10, 30, 50, 100, 300, 500, 1000, 2000]


    # onehot encoding for Location and MerchantType
    # encoder_loc = LabelEncoder().fit(data['Location'])
    # encoder_mchnt = LabelEncoder().fit(data['Type'])
    # data.loc[:,'Location'] = encoder_loc.transform(data['Location'])
    # data.loc[:,'Type'] = encoder_mchnt.transform(data['Type'])

    # data shape be like: ()

    nume_feature_ret = []
    cate_feature_ret = []
    label_ret = []
    for row_idx in tqdm(range(len(data))):
        record = data.iloc[row_idx]
        record_arr = np.array(record)
        acct_no = record['Source']  # emm

        feature_of_one_record = []
        for time_span in time_windows:
            feature_of_one_timestamp = []
            prev_records = data.iloc[(row_idx - time_span):row_idx, :]
            prev_and_now_records = data.iloc[(
                row_idx - time_span):row_idx + 1, :]
            prev_records = prev_records[prev_records['Source'] == acct_no]

            # AvgAmountT
            feature_of_one_timestamp.append(
                prev_records['Amount'].sum() / time_span)
            # TotalAmountTs
            feature_of_one_timestamp.append(prev_records['Amount'].sum())
            # BiasAmountT
            feature_of_one_timestamp.append(
                record['Amount'] - feature_of_one_timestamp[0])
            # NumberT
            feature_of_one_timestamp.append(len(prev_records))
            # MostCountryT no
            # print(prev_records)
            # print(feature_of_one_timestamp)
            # feature_of_one_timestamp.append(prev_records['Location'].mode()[0] if len(prev_records) != 0 else 0)

            # MostTerminalT -> no data for this item
            # MostMerchantT
            # feature_of_one_timestamp.append(prev_records['Type'].mode()[0] if len(prev_records) != 0 else 0)

            # TradingEntropyT ->  TradingEntropyT = EntT − NewEntT
            old_ent = calcu_trading_entropy(prev_records[['Amount', 'Type']])
            new_ent = calcu_trading_entropy(prev_and_now_records[['Amount', 'Type']])
            feature_of_one_timestamp.append(old_ent - new_ent)
            # 5 features


            # Location Type Target
            # no mentioning here! attribute embedding

            # # rest 4423
            # one_hot_feats = prev_records.iloc[:, 7:].sum().to_list()
            # assert len(one_hot_feats) == 4423, "one hot transforming error"
            # feature_of_one_timestamp.extend(one_hot_feats)
            # assert len(feature_of_one_timestamp) == 4429, "feat dim error"

            feature_of_one_record.append(feature_of_one_timestamp)


        # record attribute info
        cate_feature_ret.append(record[['Location','Type','Target']].to_numpy())

        # one record shape be like: (|TimeWindows|,|feat_num|)
        nume_feature_ret.append(feature_of_one_record)
        label_ret.append(record['Labels'])

    nume_feature_ret = np.array(nume_feature_ret).transpose(0, 2, 1)
    cate_feature_ret = np.stack(cate_feature_ret)
    # ret shape be like: (sample_num, |feat_num|, |TimeWindows|)

    # sanity check
    # print(nume_feature_ret.shape,'\n',cate_feature_ret.shape)
    assert nume_feature_ret.shape == (len(data), 5, len(time_windows)), "output shape invalid."
    assert cate_feature_ret.shape == (len(data), 3)

    # np.expand_dims(nume_feature_ret, axis=1)

    return nume_feature_ret.astype(np.float32),cate_feature_ret.astype(np.int64), np.array(label_ret).astype(np.int64)


def span_transaction_3d(

) -> None:
    """_summary_
    """

    pass


if __name__ == "__main__":
    
    data = pd.read_csv("./data/STRAD.csv")
    data = data[data['Labels'] != 2] 
    data = data[data['Amount'] != 0]
    # data_with_label = pd.get_dummies(data_with_label, columns=["Target","Location","Type"])
    # data_with_label['Target'], data_with_label['Location'], data_with_label['Type'] = data['Target'], data['Location'], data['Type']
    # data_with_label.insert(loc=4,column="Target",value=data['Target'])
    # data_with_label.insert(loc=4,column="Location",value=data['Location'])
    # data_with_label.insert(loc=4,column="Type",value=data['Type'])
    #  shape: (140576, 4427)

    # data_with_label.to_csv("./data/data_with_label_dummies.csv",index=False)

    if sys.argv[1] == '2d':

        tgt_encoder = LabelEncoder()
        data['Target'] = tgt_encoder.fit_transform(data['Target'])
        term_encoder = LabelEncoder()
        data['Location'] = term_encoder.fit_transform(data['Location'])
        type_encoder = LabelEncoder()
        data['Type'] = type_encoder.fit_transform(data['Type'])

        print("transforming to 2d data...")
        nume_features, cate_features, labels = span_transaction_2d(data,[1,2,3,5,10,50,100,500])
        # nume_features.to_csv('./data/STRAD_2d_nume_features.csv',header=0,index=0)
        np.save("./data/STRAD_2d_nume_features.npy", nume_features)
        np.save('./data/STRAD_2d_cate_features.npy', cate_features)
        np.save("./data/STRAD_labels.npy", labels)
        print(f"transforming to 2D data completed.")
        