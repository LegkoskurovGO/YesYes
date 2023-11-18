import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
# import catboost


def compose_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,
                 seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
            '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
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
res = data.merge(genders, how='outer', on='client_id')
# Перевод из float в bool
res['gender'] = res.gender.astype(bool)

# Формируем дату 
day_time = data['trans_time'].str.split(' ', n=1, expand=True)
day_time.columns = ['day', 'time']
day_time['day'] = day_time['day'].astype(int)

# Стратовая дата
start_date = datetime.datetime(2020, 3, 8, 0, 0, 0) - datetime.timedelta(219)

# Приведение данных к использованию функции
year = 2019 + day_time['day'] // 365
day = day_time["day"] % 365
res_data = pd.concat([day_time['time'].str.split(':', n=2, expand=True), day, year], axis=1)
res_data.columns = ['hour', 'minute', 'second', 'day', 'year']

# Формирование правильной даты в datetime
abcdefg = compose_date(years=res_data['year'], days=res_data['day'], hours=res_data['hour'], minutes=res_data['minute'], seconds=res_data['second'])

# Замена времени в исходном датасете с гендерами 
res['trans_time'] = pd.Series(abcdefg, name='trans_time')

res['month'] = res.trans_time.dt.month
res['day'] = res.trans_time.dt.day
res['weekday'] = res.trans_time.dt.weekday
res['hour'] = res.trans_time.dt.hour

print(res.head(10).to_string())