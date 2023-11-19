import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import datetime
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

PATH_DATA = './data'

# Читаем данные транзакций
transactions = pd.read_csv(os.path.join(PATH_DATA, 'transactions.csv'),
                           header=0,
                           index_col=False)
gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'),
                           header=0,
                           index_col=0)
gender_test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'),
                          header=0,
                          index_col=0)

# Читаем данные полов
transactions_train = transactions.merge(gender_train, how='inner', on='client_id')
transactions_test = transactions.merge(gender_test, how='inner', on='client_id')

label = transactions_train['gender']
transactions_train.drop('gender', axis=1, inplace=True)


def preprocessing_data(res: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    # Формируем дату
    day_time = data['trans_time'].str.split(' ', n=1, expand=True)
    day_time.columns = ['day', 'time']
    day_time['day'] = day_time['day'].astype(int)

    # Стратовая дата
    start_date = datetime.datetime(2020, 3, 8, 0, 0, 0) - datetime.timedelta(219)

    # Замена времени в исходном датасете с гендерами
    trans_time = pd.Series(start_date + pd.to_timedelta(np.ceil(day_time['day']), unit="D"), name='trans_time')
    # trans_time.index = res['client_id']

    # trans_time.dt.month
    # trans_time.dt.day
    res['weekday'] = trans_time.dt.weekday
    # trans_time.dt.hour

    cat_mcc = res["mcc_code"]
    cat_mcc.index = res['client_id']
    cat_mcc.name = 'mcc_describe'

    a = cat_mcc.mask((724 <= cat_mcc) & (cat_mcc < 1799), 1) \
                .mask((1799 <= cat_mcc) & (cat_mcc < 2842) | (4900 <= cat_mcc) & (cat_mcc < 5200) | (5714 <= cat_mcc) & (cat_mcc < 5715) | (9702 <= cat_mcc) & (cat_mcc < 9752), 2) \
                .mask((2842 <= cat_mcc) & (cat_mcc < 3299), 3) \
                .mask((3299 <= cat_mcc) & (cat_mcc < 3441) | (7511 <= cat_mcc) & (cat_mcc < 7519), 4) \
                .mask((3441 <= cat_mcc) & (cat_mcc < 3882) | (6760 <= cat_mcc) & (cat_mcc < 7011), 5) \
                .mask((3882 <= cat_mcc) & (cat_mcc < 4789), 6) \
                .mask((4789 <= cat_mcc) & (cat_mcc < 4900), 7) \
                .mask((5200 <= cat_mcc) & (cat_mcc < 5499), 8) \
                .mask((5499 <= cat_mcc) & (cat_mcc < 5599) | (5699 <= cat_mcc) & (cat_mcc < 5714) | (5969 <= cat_mcc) & (cat_mcc < 5999), 9) \
                .mask((5599 <= cat_mcc) & (cat_mcc < 5699), 10) \
                .mask((5715 <= cat_mcc) & (cat_mcc < 5735) | (5811 <= cat_mcc) & (cat_mcc < 5950), 11) \
                .mask((5735 <= cat_mcc) & (cat_mcc < 5811) | (5999 <= cat_mcc) & (cat_mcc < 6760) | (5962 <= cat_mcc) & (cat_mcc < 5963) | (7011 <= cat_mcc) & (cat_mcc < 7033), 12) \
                .mask((5950 <= cat_mcc) & (cat_mcc < 5962) | (5963 <= cat_mcc) & (cat_mcc < 5969), 13) \
                .mask((7033 <= cat_mcc) & (cat_mcc < 7299), 14) \
                .mask((7299 <= cat_mcc) & (cat_mcc < 7511) | (7519 <= cat_mcc) & (cat_mcc < 7523), 15) \
                .mask((7523 <= cat_mcc) & (cat_mcc < 7699), 16) \
                .mask((7699 <= cat_mcc) & (cat_mcc < 7999), 17) \
                .mask((7999 <= cat_mcc) & (cat_mcc < 8351), 18) \
                .mask((8351 <= cat_mcc) & (cat_mcc < 8699), 19) \
                .mask((8699 <= cat_mcc) & (cat_mcc < 8999), 20) \
                .mask((8999 <= cat_mcc) & (cat_mcc < 9702) | (9752 <= cat_mcc) & (cat_mcc < 9754), 21)

    res['mcc_describe'] = a.reset_index(drop=True)
    res['mcc_describe'] = res['mcc_describe'].astype(object)

    res['amount_up'] = res['amount'].where(res['amount'] >= 0)
    a = res['amount_up']
    res['amount_up'] = a.mask(a < a.quantile(0.05), a.quantile(0.05)) \
                        .mask(a > a.quantile(0.95), a.quantile(0.95))
    res['amount_up'] = MinMaxScaler().fit_transform(res[['amount_up']]) * 1000
    
    res['amount_down'] = res['amount'].where(res['amount'] <= 0).abs()
    a = res['amount_down']
    res['amount_down'] = a.mask(a < a.quantile(0.05), a.quantile(0.05)) \
                          .mask(a > a.quantile(0.95), a.quantile(0.95))
    res['amount_down'] = MinMaxScaler().fit_transform(res[['amount_down']]) * 1000

    # Характеристика по клиентам заработок и траты
    tmp = res[['client_id', 'amount_up', 'amount_down']].groupby('client_id').agg({'amount_up': ['mean', 'median', 'std', 'count', 'sum'], \
                                                                                   'amount_down': ['mean', 'median', 'std', 'count', 'sum']})
    tmp.columns = tmp.columns.map('{0[0]}_client_{0[1]}'.format)
    res = res.merge(tmp, how='outer', on='client_id')

    # Характеристика по кол-во трат клиентами в дни недели заработок и траты
    aaa = res[['client_id', 'weekday', 'amount_up', 'amount_down']].groupby(['client_id', 'weekday']).count()
    aaa = aaa.unstack(-1)
    aaa.columns = aaa.columns.map('{0[0]}_weekday_{0[1]}'.format)
    res = res.merge(aaa, how='outer', on='client_id')

    # Заработок - траты
    # res['delta+-'] = res['amount_up_client_sum'] - res['amount_down_client_sum']
    # a = res['delta+-']
    # res['delta+-'] = a.mask(a < a.quantile(0.05), a.quantile(0.05)) \
    #                   .mask(a > a.quantile(0.95), a.quantile(0.95))

    res['mcc_code'] = res.mcc_code.astype(object)
    res['trans_type'] = res.trans_type.astype(object)

    tmp = res.groupby('client_id')['term_id'].nunique()
    tmp.name = 'count_term_id'
    res = res.merge(tmp, how='outer', on='client_id')

    # Характеристика по неделям для всех заработок и траты
    tmp = res.groupby('client_id')['trans_type'].nunique()
    tmp.name = 'count_trans_type'
    res = res.merge(tmp, how='outer', on='client_id')

    # Характеристика по неделям для всех заработок и траты
    tmp = res.groupby('client_id')['mcc_code'].nunique()
    tmp.name = 'count_mcc_code'
    res = res.merge(tmp, how='outer', on='client_id')

    # Частота покупок за время существования
    time_client = pd.concat([trans_time, res['client_id']], axis=1)
    abc = time_client.groupby('client_id').agg({'trans_time': ['min', 'max']}).diff(axis=1)
    abc.columns = ['nan', 'days']
    abcde = pd.DataFrame(res['client_id'].value_counts()).merge(abc['days'].dt.days, on='client_id')
    all_time_freq = abcde['days'] / abcde['count']
    all_time_freq.name = 'all_time_freq'
    res = res.merge(all_time_freq, on='client_id')

    res.drop(['amount', 'amount_up', 'amount_down', 'weekday', 'trans_time'], axis=1, inplace=True)

    return res


train = preprocessing_data(transactions_train, transactions)
test = preprocessing_data(transactions_test, transactions)

cat_features = ['mcc_code', 'trans_type', 'trans_city', 'mcc_describe']

model = CatBoostClassifier(
    iterations=50,
    random_seed=63,
    learning_rate=0.011,
    custom_loss='AUC',
    verbose=10
)
model.fit(
    train.drop(['term_id', 'client_id'], axis=1), label,
    cat_features=cat_features
)

# test['probability'] = model.predict_proba(test.drop(['term_id', 'client_id'], axis=1))[:, 1]
# submission= test[['client_id', 'probability']]
#
# submission.to_csv('result.csv')

