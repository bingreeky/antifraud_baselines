import sys
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
    # delete transactions with ZERO amount
    data = data[data['Amount'] != 0]

    # onehot encoding for Location and MerchantType
    encoder_loc = LabelEncoder().fit(data['Location'])
    encoder_mchnt = LabelEncoder().fit(data['Type'])
    data.loc[:,'Location'] = encoder_loc.transform(data['Location'])
    data.loc[:,'Type'] = encoder_mchnt.transform(data['Type'])

    feature_ret = []
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
            # TotalAmountT
            feature_of_one_timestamp.append(prev_records['Amount'].sum())
            # BiasAmountT
            feature_of_one_timestamp.append(
                record['Amount'] - feature_of_one_timestamp[0])
            # NumberT
            feature_of_one_timestamp.append(len(prev_records))
            # MostCountryT
            # print(prev_records)
            # print(feature_of_one_timestamp)
            feature_of_one_timestamp.append(prev_records['Location'].mode()[
                                            0] if len(prev_records) != 0 else 0)
            # MostTerminalT -> no data for this item
            # MostMerchantT
            feature_of_one_timestamp.append(prev_records['Type'].mode()[
                                            0] if len(prev_records) != 0 else 0)
            # TradingEntropyT ->  TradingEntropyT = EntT − NewEntT
            old_ent = calcu_trading_entropy(prev_records[['Amount', 'Type']])
            new_ent = calcu_trading_entropy(
                prev_and_now_records[['Amount', 'Type']])
            feature_of_one_timestamp.append(old_ent - new_ent)
            # 7 features

            feature_of_one_record.append(feature_of_one_timestamp)

        # one record shape be like: (|TimeWindows|,|feat_num|)
        feature_ret.append(feature_of_one_record)
        label_ret.append(record['Labels'])

    feature_ret = np.array(feature_ret).transpose(0, 2, 1)
    # ret shape be like: (sample_num, |feat_num|, |TimeWindows|)

    # sanity check
    assert feature_ret.shape == (len(data), 7, len(
        time_windows)), "output shape invalid."

    np.expand_dims(feature_ret, axis=1)

    return feature_ret, np.array(label_ret)

if __name__ == "__main__":
    
    data = pd.read_csv("./data/STRAD.csv")
    data_with_label = data[data['Labels'] != 2]
    
    if sys.argv[1] == '2d':
        features, labels = span_transaction_2d(data_with_label,[1,2,7,30])
        features.to_csv('./data/STRAD_2d_features.csv',header=0,index=0)
        np.save("./data/STRAD_2d_features.npy", features)
        np.save("./data/STRAD_labels.npy", labels)
        print(f"transforming to 2D data completed.")
        