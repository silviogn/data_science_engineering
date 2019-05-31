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

    i = 1
    for x, r in df_complete.iterrows():
        df_complete.at[x, 'AnoDiasJulianos'] = i
        i = i + 1



    df_complete.columns = \
        ['DiasJulianos', 'VelocidadeU', 'VelocidadeV', 'Temperatura', 'CoberturaNuvens',
         'RadiacaoGlobal', 'RadiacaoDireta', 'RadiacaoDifusa', 'Ano', 'AnoDiasJulianos']

    #df_complete = df_complete.sort_values(by=['Ano', 'DiasJulianos'])
    #df_complete = df_complete[df_complete['Ano'].isin([2009,2010])]

    #df_complete = df_complete.sort_values(by=['AnoDiasJulianos'])

    #corrdf = df_complete.pct_change()

    corr = df_complete['RadiacaoGlobal'].corr(df_complete['RadiacaoDifusa'])

    print(corr)

    #df_complete.to_csv('all.csv')
    sns.set(style="darkgrid")

    #fig, (velocidadeU, temperatura, radiacao_global, radiacao_direta,radiacao_difusa) = plt.subplots(nrows=5, ncols=1)
    #fig.suptitle("Dados Ano {} ".format(year))

    f = sns.lineplot(x="AnoDiasJulianos", y="VelocidadeU", data=df_complete, palette="tab10", linewidth=0.5) #, ax=velocidadeU)
    f.set(xlabel="Dias Julianos", ylabel='Velocidade U')
    plt.show()

    f = sns.lineplot(x="AnoDiasJulianos", y="Temperatura", data=df_complete, palette="tab10", linewidth=0.5) #, ax=temperatura)
    f.set(xlabel="Dias Julianos", ylabel='Tempearatura')
    plt.show()

    f = sns.lineplot(x="AnoDiasJulianos", y="RadiacaoGlobal", data=df_complete, palette="tab10", linewidth=0.5) #, ax=radiacao_global)
    f.set(xlabel="Dias Julianos", ylabel='Radiação Global')
    plt.show()

    f = sns.lineplot(x="AnoDiasJulianos", y="RadiacaoDireta", data=df_complete, palette="tab10", linewidth=0.5) #, ax=radiacao_direta)
    f.set(xlabel="Dias Julianos", ylabel='Radiação Direta')
    plt.show()

    f = sns.lineplot(x="AnoDiasJulianos", y="RadiacaoDifusa", data=df_complete, palette="tab10", linewidth=0.1) #, ax=radiacao_difusa)
    f.set(xlabel="Dias Julianos", ylabel='Radiação Difusa')
    plt.show()






if __name__ == '__main__':
    main()















