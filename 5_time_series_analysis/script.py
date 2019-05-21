import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

years = [2009, 2010, 2011, 2012, 2013, 2014]

file = "./data/era.alegrete.{}.csv"

series_list = []



for year in years:
    df = pd.DataFrame().from_csv(file.format(year))
    df["ano"] = year
    series_list.append(df)

df_complete = pd.concat(series_list, ignore_index=True)

year_aux = df_complete['ano']

scaler = MinMaxScaler(feature_range=(0, 1))

scaler = scaler.fit(df_complete.values)

normalized = scaler.transform(df_complete.values)


df_complete = pd.DataFrame(normalized, columns=['DiasJulianos','VelocidadeU', 'VelocidadeV', 'Temperatura',
                                                'CoberturaNuvens', 'RadiacaoGlobal', 'RadiacaoDireta', 'RadiacaoDifusa', 'Ano'])

df_complete['AnoInteiro'] = year_aux

anos = df_complete['AnoInteiro'].unique()

filtered =  df_complete[df_complete['AnoInteiro'] == 2009]

sns.lineplot(x='DiasJulianos', y='VelocidadeU', data=filtered)
sns.lineplot(x='DiasJulianos', y='Temperatura', data=filtered)
sns.lineplot(x='DiasJulianos', y='RadiacaoGlobal', data=filtered)
sns.lineplot(x='DiasJulianos', y='RadiacaoDireta', data=filtered)
sns.lineplot(x='DiasJulianos', y='RadiacaoDifusa', data=filtered)

plt.show()










