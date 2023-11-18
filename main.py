import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
# import catboost


def compose_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,seconds=None):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
            '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
if v is not None)


# Читаем данные транзакций
data = pd.read_csv('transactions.csv',
                   sep=',',
                   header=0,
                   index_col=False)
# Читаем данные полов
genders = pd.read_csv('train.csv',
                      sep=',',
                      header=0,
                      index_col=0)

# Мёрджим
res = data.merge(genders, how='inner', on='client_id')
# Перевод из float в bool
res['gender'] = res.gender.astype(bool)

# Формируем дату 
day_time = data['trans_time'].str.split(' ', n=1, expand=True)
day_time.columns = ['day', 'time']
day_time['day'] = day_time['day'].astype(int)

# Стратовая дата
start_date = datetime.datetime(2020, 3, 8, 0, 0, 0) - datetime.timedelta(219)

# # Приведение данных к использованию функции
# year = 2019 + day_time['day'] // 365
# day = day_time["day"] % 365
# res_data = pd.concat([day_time['time'].str.split(':', n=2, expand=True), day, year], axis=1)
# res_data.columns = ['hour', 'minute', 'second', 'day', 'year']

# # Формирование правильной даты в datetime
# abcdefg = compose_date(years=res_data['year'], days=res_data['day'], hours=res_data['hour'], minutes=res_data['minute'], seconds=res_data['second'])

# Замена времени в исходном датасете с гендерами 
trans_time = pd.Series(start_date + pd.to_timedelta(np.ceil(day_time['day']), unit="D"), name='trans_time')

# trans_time.dt.month
# trans_time.dt.day
res['weekday'] = trans_time.dt.weekday
# trans_time.dt.hour

res['amount_up'] = res['amount'].where(res['amount'] >= 0)
res['amount_down'] = res['amount'].where(res['amount'] <= 0).abs()

# Характеристика по клиентам заработок и траты
tmp = res.groupby('client_id').agg({'amount_up': ['mean', 'median', 'std', 'count', 'sum'], \
                                    'amount_down': ['mean', 'median', 'std', 'count', 'sum']})
tmp.columns = tmp.columns.map('{0[0]}_client_{0[1]}'.format)
res = res.merge(tmp, how='outer', on='client_id')

# Характеристика по кол-во трат клиентами в дни недели заработок и траты
aaa = res[['client_id', 'weekday', 'amount_up', 'amount_down']].groupby(['client_id', 'weekday']).count()
aaa = aaa.unstack(-1)
aaa.columns = aaa.columns.map('{0[0]}_weekday_{0[1]}'.format)
res = res.merge(aaa, how='outer', on='client_id')

# Заработок - траты
res['delta+-'] = res['amount_up_client_sum'] - res['amount_down_client_sum']

# Группировка MCC кодов
mcc = res[["mcc_code", 'client_id']]

cat_mcc_dct = {
    "Контрактные услуги": range(724, 1799),
    "Оптовые поставщики и производители": [*range(1799, 2842),*range(4900, 5200), * range(5714, 5715), *range(9702, 9752)],
    "Авиакомпании": range(2842, 3299),
    "Аренда автомобилей": [*range(3299,3441),*range(7511, 7519)],
    "Отели и мотели": [*range(3441,3882),*range(6760, 7011)],
    "Транспорт": range(3882, 4789),
    "Коммунальные и кабельные услуги": range(4789, 4900),
    "Розничные магазины": range(5200, 5499),
    "Автомобили и транспортные средства": [*range(5499, 5599), *range(5699, 5714), * range(5969, 5999)],
    "Магазины одежды": range(5599, 5699),
    "Различные магазины": [*range(5715, 5735), *range(5811, 5950)],
    "Поставщик услуг": [*range(5735, 5811),*range(5999, 6760), * range(5962, 5963), * range(7011, 7033)],
    "Продажи по почте/ телефону": [*range(5950,5962), *range(5963, 5969)],
    "Личные услуги": range(7033, 7299),
    "Бизнес услуги": [*range(7299, 7511), *range(7519, 7523)],
    "Ремонтные услуги": range(7523, 7699),
    "Развлечения": range(7699, 7999),
    "Профессиональные услуги": range(7999, 8351),
    "Членские организации": range(8351, 8699),
    "Бизнес Услуги": range(8699, 8999),
    "Государственные услуги": [*range(8999, 9702), *range(9752, 9754)],
}

cat_mcc = mcc.replace({'mcc_code': cat_mcc_dct.values()}, {'mcc_code': cat_mcc_dct.keys()})
res["mcc_describe"] = cat_mcc["mcc_code"]
res['mcc_code'] = res.mcc_code.astype(object)
res['trans_type'] = res.trans_type.astype(object)

# Характеристика по количетсву mcc_code
tmp = res.groupby('client_id')['mcc_code'].nunique()
tmp.name = 'type_mcc_code'
res = res.merge(tmp, how='outer', on='client_id')

# Характеристика по количетсву trans_type
tmp = res.groupby('client_id')['trans_type'].nunique()
tmp.name = 'type_trans_type'
res = res.merge(tmp, how='outer', on='client_id')

# Характеристика по количетсву term_id
tmp = res.groupby('client_id')['term_id'].nunique()
tmp.name = 'type_term_id'
res = res.merge(tmp, how='outer', on='client_id')

res.drop(['amount', 'amount_up', 'amount_down', 'weekday', 'trans_time'], axis=1, inplace=True)

# Частота покупок за время существования
time_client = pd.concat([trans_time, res['client_id']], axis=1)
abc = time_client.groupby('client_id').agg({'trans_time': ['min', 'max']}).diff(axis=1)
abc.columns = ['nan', 'days']
abcde = pd.DataFrame(res['client_id'].value_counts()).merge(abc['days'].dt.days, on='client_id')
all_time_freq = abcde['days'] / abcde['count']
all_time_freq.name = 'all_time_freq'
res = res.merge(all_time_freq, on='client_id')

res.info()
