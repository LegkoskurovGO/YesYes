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

razbivka = transactions_train.groupby(['gender', 'client_id'], as_index=False).count()

razbivka2 = pd.concat([razbivka.query('gender == 1').sample(n=756)['client_id'], razbivka.query('gender == 0').sample(n=756)['client_id']], axis=0)

train = transactions_train[~transactions_train['client_id'].isin(razbivka2)].reset_index(drop=True)
test = transactions_train[transactions_train['client_id'].isin(razbivka2)].reset_index(drop=True)


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
    # res['amount_up'] = MinMaxScaler().fit_transform(res[['amount_up']]) * 1000
    
    res['amount_down'] = res['amount'].where(res['amount'] <= 0).abs()
    a = res['amount_down']
    res['amount_down'] = a.mask(a < a.quantile(0.05), a.quantile(0.05)) \
                          .mask(a > a.quantile(0.95), a.quantile(0.95))
    # res['amount_down'] = MinMaxScaler().fit_transform(res[['amount_down']]) * 1000

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
    res['amount_mean_up_weekday'] = res[[x for x in res.columns if "amount_up_weekday" in x]].mean(axis=1)
    res['amount_mean_down_weekday'] = res[[x for x in res.columns if "amount_down_weekday" in x]].mean(axis=1)


    # Заработок - траты
    res['delta+-'] = res['amount_up_client_sum'] - res['amount_down_client_sum']
    a = res['delta+-']
    res['delta+-'] = a.mask(a < a.quantile(0.05), a.quantile(0.05)) \
                      .mask(a > a.quantile(0.95), a.quantile(0.95))

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

    res.drop(['amount',
              'amount_up',
              'amount_down',
              'weekday',
              'trans_time',
              *[x for x in res.columns if "amount_up_weekday" in x],
              *[x for x in res.columns if "amount_down_weekday" in x]], axis=1, inplace=True)

    return res


def construct_features(data):
    # convert time data for the whole dataset first

    splitted = data['trans_time'].str.split(' ', n=1, expand=True)
    data["day"] = pd.DataFrame(splitted[0]).astype("int64")
    data["time"] = pd.DataFrame(splitted[1].str.split(':', expand=True)[0]).astype("int64")

    # amount per transaction

    data["Mean_net_money_per_transaction"] = data.groupby(['client_id'])["amount"].transform("mean")
    data["Std_net_money_per_transaction"] = data.groupby(['client_id'])["amount"].transform("std").fillna(0)

    func = lambda x: x.values[0] if x[x < 0].count() == 1 else x[x < 0].mean()
    data["Mean_spend_money_per_transaction"] = data.groupby(['client_id'])["amount"].transform(func).fillna(0)
    func = lambda x: x.values[0] if x[x > 0].count() == 1 else x[x > 0].mean()
    data["Mean_earn_money_per_transaction"] = data.groupby(['client_id'])["amount"].transform(func).fillna(0)

    func = lambda x: x[x < 0].std()
    data["Money_spend_std_per_transaction"] = data.groupby(['client_id'])["amount"].transform(func).fillna(0)
    func = lambda x: x[x > 0].std()
    data["Mean_earn_std_per_transaction"] = data.groupby(['client_id'])["amount"].transform(func).fillna(0)

    data["Money_earn_spend_ratio_per_transaction"] = (data["Mean_earn_money_per_transaction"].abs() / data["Money_spend_std_per_transaction"].abs()).fillna(0)
    data["Money_earn_spend_ratio_per_transaction"].replace(np.inf, 1000, inplace = True)

    print("amount per transaction is completed")

    # amount all

    data["ALL_money_net"] = data.groupby(['client_id'])["amount"].transform("sum")

    func = lambda x: x[x < 0].sum()
    data["ALL_money_spend"] = data.groupby(['client_id'])["amount"].transform(func).fillna(0)
    func = lambda x: x[x > 0].sum()
    data["ALL_money_earn"] = data.groupby(['client_id'])["amount"].transform(func).fillna(0)

    data["ALL_money_spend_earn_ratio"] = (data["ALL_money_spend"].abs() / data["ALL_money_earn"].abs())
    data["ALL_money_spend_earn_ratio"].replace(np.inf, 1000, inplace = True)

    print("amount all is completed")

    # frequency of transactions

    data["Frequency_of_spending_per_day"] =  data.groupby(['client_id', 'day'])['day'].transform("count")
    data["Frequency_of_spending_all"] =  data.groupby(['client_id'])['day'].transform("count")
    func = lambda x: (x.count()/7)
    data["Frequency_of_spending_per_week"] = data.groupby(['client_id'])['day'].transform(func)

    data["Hours_std_transaction_per_day"] = data.groupby(['client_id','day'])['time'].transform("std").fillna(0)

    print("frequency is completed")

    # habits

    data["Terminal_habit"] = data.groupby(['client_id','term_id'])['term_id'].transform("count").fillna(0)
    data["Terminal_habit_sum_money"] = data.groupby(['client_id','term_id'])['amount'].transform("sum").fillna(0)
    data["Terminal_habit_money"] =data.groupby(['client_id','term_id'])['amount'].transform("mean").fillna(0)

    data["Service_habit"] = data.groupby(['client_id','trans_type'])['trans_type'].transform("count").fillna(0)

    data["Product_habit_frequency"] = data.groupby(['client_id','mcc_code'])['mcc_code'].transform("count").fillna(0)
    data["Product_habit_sum_money"] = data.groupby(['client_id','mcc_code'])['amount'].transform("sum").fillna(0)
    data["Product_habit_mean_money"] = data.groupby(['client_id','mcc_code'])['amount'].transform("mean").fillna(0)

    print("habits is completed")

    # amount per day

    data["Mean_net_money_per_day"] = data.groupby(['client_id','day'])["amount"].transform("mean")
    data["Std_net_money_per_day"] = data.groupby(['client_id','day'])["amount"].transform("std").fillna(0)

    func = lambda x: x.values[0] if x[x < 0].count() == 1 else x[x < 0].mean()
    data["Mean_spend_money_per_day"] = data.groupby(['client_id','day'])["amount"].transform(func).fillna(0)
    func = lambda x: x.values[0] if x[x > 0].count() == 1 else x[x > 0].mean()
    data["Mean_earn_money_per_day"] = data.groupby(['client_id','day'])["amount"].transform(func).fillna(0)

    print("amount per day is completed")

    print("Feature construction is completed")
    return data



train = preprocessing_data(train, transactions)
test = preprocessing_data(test, transactions)

# train = construct_features(train)
# test = construct_features(test)

print(train.head(20))
train.info()
print(test.head(20))
test.info()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

train_x = train.drop(['term_id', 'client_id', 'gender', 'mcc_code'], axis=1)
train_x['trans_type'] = train_x['trans_type'].astype(int)
train_y = train['gender']

train_x = pd.concat([train_x, pd.get_dummies(train_x['trans_city'].astype(str))], axis=1)
train_x = pd.concat([train_x, pd.get_dummies(train_x['mcc_describe'].astype(str))], axis=1)
train_x.drop(['mcc_describe', 'trans_city'], axis=1, inplace=True)
#
test_x = train.drop(['term_id', 'client_id', 'gender', 'mcc_code'], axis=1)
test_x['trans_type'] = test_x['trans_type'].astype(int)
test_y = train['gender']

test_x = pd.concat([test_x, pd.get_dummies(test_x['trans_city'].astype(str))], axis=1)
test_x = pd.concat([test_x, pd.get_dummies(test_x['mcc_describe'].astype(str))], axis=1)
test_x.drop(['mcc_describe', 'trans_city'], axis=1, inplace=True)

train_x.fillna(0, inplace=True)
train_y.fillna(0, inplace=True)

test_x.fillna(0, inplace=True)
test_y.fillna(0, inplace=True)


model = RandomForestClassifier(max_depth=10)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
auc = roc_auc_score(test_y, y_pred)
print(f'\n\nНаш результат: {auc}\n\n')
a = input('всё')
quit()

cat_features = ['mcc_code', 'trans_type', 'trans_city', 'mcc_describe']

# model = CatBoostClassifier(
#     iterations=100,
#     random_seed=63,
#     learning_rate=0.025,
#     custom_loss='AUC',
#     verbose=10
# )
# model.fit(
#     train.drop(['term_id', 'client_id', 'gender'], axis=1), train['gender'],
#     cat_features=cat_features
# )
#
from sklearn.metrics import roc_auc_score
# y_pred = model.predict_proba(test.drop(['term_id', 'client_id', 'gender'], axis=1))[:, 1]
# auc = roc_auc_score (test['gender'], y_pred)
# print(auc)


#
# test['probability'] = model.predict_proba(test.drop(['term_id', 'client_id'], axis=1))[:, 1]
# submission= test[['client_id', 'probability']]
#
# submission.to_csv('result.csv')
#



model = CatBoostClassifier(
    iterations=150,
    random_seed=63,
    custom_loss='AUC',
    eval_metric='AUC',
    verbose=20,
    od_type='Iter',
    od_wait=70,
    use_best_model=True
)
model.fit(
    train.drop(['term_id', 'client_id', 'gender'], axis=1), train['gender'],
    eval_set=(test.drop(['term_id', 'client_id', 'gender'], axis=1), test['gender']),
    cat_features=cat_features
)
y_pred = model.predict_proba(test.drop(['term_id', 'client_id', 'gender'], axis=1))[:, 1]
auc = roc_auc_score (test['gender'], y_pred)
print(auc)