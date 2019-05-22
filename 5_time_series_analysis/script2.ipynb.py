import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main():
    sns.set()

    years = [2009, 2010, 2011, 2012, 2013, 2014]

    file = "./data/era.alegrete.{}.csv"

    df_yearly = []

    for year in years:
        df = pd.DataFrame().from_csv(file.format(year))
        df["ano"] = year
        df_yearly.append(df)


    df_complete = pd.concat(df_yearly, ignore_index=True)

    #year_aux = df_complete['ano']
    #dias_julianos_aux = df_complete['V1']

    #data_scaler = MinMaxScaler(feature_range=(0, 1))

    #data_scaler = data_scaler.fit(df_complete.values)

    #normalized = data_scaler.transform(df_complete.values)


    df_complete.columns=\
        ['DiasJulianos', 'VelocidadeU', 'VelocidadeV', 'Temperatura', 'CoberturaNuvens',
         'RadiacaoGlobal','RadiacaoDireta', 'RadiacaoDifusa', 'Ano']

    #df_complete = df_complete.drop(['DiasJulianos'], axis=1)
    #df_complete['AnoInteiro'] = year_aux
    #df_complete['DiasJulianos'] = dias_julianos_aux

    years_unique = df_complete['Ano'].unique()

    for year in years_unique:
        filtered = df_complete[df_complete['Ano'] == year]

        sns.set(style="darkgrid")

        fig, (velocidadeU, temperatura, radiacao_global, radiacao_direta,radiacao_difusa) = plt.subplots(nrows=5, ncols=1)
        fig.suptitle("Dados Ano {} ".format(year))

        sns.lineplot(x="DiasJulianos", y="VelocidadeU", data=filtered, palette="tab10", linewidth=0.5, ax=velocidadeU)
        velocidadeU.set(xlabel="Dias Julianos", ylabel='Velocidade U')

        sns.lineplot(x="DiasJulianos", y="Temperatura", data=filtered, palette="tab10", linewidth=0.5, ax=temperatura)
        temperatura.set(xlabel="Dias Julianos", ylabel='Tempearatura')

        sns.lineplot(x="DiasJulianos", y="RadiacaoGlobal", data=filtered, palette="tab10", linewidth=0.5, ax=radiacao_global)
        radiacao_global.set(xlabel="Dias Julianos", ylabel='Radiação Global')

        sns.lineplot(x="DiasJulianos", y="RadiacaoDireta", data=filtered, palette="tab10", linewidth=0.5, ax=radiacao_direta)
        radiacao_direta.set(xlabel="Dias Julianos", ylabel='Radiação Direta')

        sns.lineplot(x="DiasJulianos", y="RadiacaoDifusa", data=filtered, palette="tab10", linewidth=0.5, ax=radiacao_difusa)
        radiacao_difusa.set(xlabel="Dias Julianos", ylabel='Radiação Difusa')


        plt.show()




if __name__ == '__main__':
    main()















