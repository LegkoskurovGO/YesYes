# Дополнительный файл для презентации

# Расчёт даты
219-67

start_date = datetime.datetime(2020, 3, 8, 0, 0, 0) - datetime.timedelta(219)

day_time.day.max()

start_date + datetime.timedelta(456)



tmp = day_time.groupby('gender').sample(n=1677628)

tmp2 = tmp.groupby(['gender', 'day']).count()

(lsp := int(len(tmp)/2))

sns.lineplot(data=tmp2.iloc[:lsp, :], x='day', y='MCC', hue='gender')
plt.ylabel('Колличество покупок')

sns.lineplot(data=tmp2.iloc[300:370, :], x='day', y='MCC', hue='gender')
plt.ylabel('Колличество покупок')




sns.lineplot(data=day_time.drop('time', axis=1).query('MCC == 5992').groupby(['gender', 'day'], as_index=False)['MCC'].count(), x='day', y='MCC', hue='gender')

day_time.drop('time', axis=1).query('MCC == 5992').groupby(['gender', 'day'], as_index=False)['MCC'].count().sort_values('MCC', ascending=False)

sns.lineplot(data=day_time.drop('time', axis=1).query('MCC == 5813').groupby(['gender', 'day'], as_index=False)['MCC'].count(), x='day', y='MCC', hue='gender')

# Определяем день недели
day_week = day_time.drop('time', axis=1).query('MCC == 5813').groupby(['gender', 'day'], as_index=False)['MCC'].count()
day_week
day_week['day'] = day_week['day'].apply(lambda x: x % 7)
day_week.drop('gender', axis=1).groupby('day').count()

day_week = day_time.drop(columns=['time', 'hour'])
day_week['day'] = day_week['day'].apply(lambda x: x % 7)
day_week
day_week.drop('gender', axis=1).groupby('day').count()

day_week.drop('gender', axis=1).query('MCC == 5813').groupby('day').count().plot.line()
day_week.drop('gender', axis=1).query('MCC == 5611').groupby('day').count().plot.line()

day_time.drop('time', axis=1).query('MCC == 5532').groupby(['gender', 'day'], as_index=False)['MCC'].count()

sns.lineplot(data=day_time.drop('time', axis=1).query('MCC == 5621').groupby(['gender', 'day'], as_index=False)['MCC'].count(), x='day', y='MCC', hue='gender')

day_time.drop('time', axis=1).query('MCC == 5532').groupby(['gender', 'day'], as_index=False)['MCC'].count().sort_values('MCC', ascending=False)

# 8 марта - воскресенье
219 % 7 == 2
year = 2020

# Дизбаланс классов
1885901 - 1677628


# Уникальные коды
def nuniques_df(dataframe: pd.DataFrame, grouper: str, name: str) -> None:
    group_keys_ = dataframe.groupby(grouper).groups.keys()
    tmp = [dataframe[dataframe[grouper] == i].nunique() for i in group_keys_]
    res = pd.concat([dataframe.nunique(), *tmp], axis=1)
    res.columns = ['All', *group_keys_]
    print(f'Группируем {name} по {grouper}: \n{res}')

nuniques_df(day_time, 'gender', 'day_time')

group = day_time.groupby("gender", as_index=False).agg({"MCC": "unique"}).reset_index(drop=True)

all_uniq = set(day_time["MCC"])

x = group[group['gender'] == 0]["MCC"]
a0 = set(x.to_numpy()[0])
x1 = group[group['gender'] == 1]["MCC"]
a1 = set(x1.to_numpy()[0])

tiff0 = all_uniq - a1
tiff1 = all_uniq - a0
print(all_uniq - a0)


dic = {"gender=1" : tiff1, "gender=0" : tiff0}
pprint(dic)



sns.histplot(tmp, hue='gender', x='MCC')

# Распределение по тратам

