#!/usr/bin/env python3.9
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename : str) -> pd.DataFrame:
    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
                "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
                "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
                "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }
    regions_swap = {value: key for key, value in regions.items()}
    def read_append(csv):
        df = pd.read_csv(csv, sep=';', names=headers, encoding='cp1250', low_memory=False)
        df['region'] = regions_swap[csv.name.split('.')[0]]
        return df
    with zipfile.ZipFile(filename, 'r') as zf:
        df = pd.DataFrame()
        for year_zip in zf.namelist():
            inner_zip = zipfile.ZipFile(zf.open(year_zip))
            csv_list = list(map(inner_zip.open, inner_zip.namelist()))
            csv_list = [x for x in csv_list if x.name.split('.')[0] in regions.values()]
            df_temp = pd.concat(map(read_append, csv_list), ignore_index=True)
            df = pd.concat([df, df_temp], ignore_index=True)
    return df

# Ukol 2: zpracovani dat
def parse_data(df : pd.DataFrame, verbose : bool = False) -> pd.DataFrame:
    df_new = df.copy()
    df_new = df_new.drop_duplicates(subset=['p1'])
    df_new['date'] = pd.to_datetime(df_new["p2a"])
    df_new['p2a'] = pd.to_datetime(df_new["p2a"])
    for x in 'abdefgo':
        df_new[x] = df_new[x].str.replace(',','.')
        df_new[x] = pd.to_numeric(df_new[x], errors='coerce')
    df_new['n'] = pd.to_numeric(df_new['n'], errors='coerce')
    for x in df_new.columns:
        if df_new.dtypes.loc[x] == 'int64' and df_new[x].max() < 127 and df_new[x].min() > -127:
            df_new[x] = df_new[x].astype('int8')
    for x in 'pqhi':
        df_new[x] = df_new[x].astype('category')
    if verbose:
        orig_size = df.memory_usage(deep=True).sum() / 10**6
        new_size = df_new.memory_usage(deep=True).sum() / 10**6
        print(f"orig_size={orig_size:0.1f} MB")
        print(f"new_size={new_size:0.1f} MB")
    return df_new

# Ukol 3: počty nehod v jednotlivých regionech podle viditelnosti
def plot_visibility(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    df = df.copy()
    df.loc[df['p19'] == 1, 'p19'] = 'den-nezhorsena'
    df.loc[(df['p19'] == 2) | (df['p19'] == 3), 'p19'] = 'den-zhorsena'
    df.loc[(df['p19'] == 4) | (df['p19'] == 6), 'p19'] = 'noc-nezhorsena'
    df.loc[(df['p19'] == 5) | (df['p19'] == 7), 'p19'] = 'noc-zhorsena'
    table = pd.pivot_table(df, values='p1', index=['region', 'p19'], aggfunc="count").reset_index()
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    g = sns.catplot(x='region', y='p1', data=table, col='p19', col_wrap=2, kind='bar', order=['HKK', 'VYS', 'PAK', 'OLK'], sharex=False)
    g.set_axis_labels('Kraj', 'Pocet nehod')
    g.set_titles("Vyditelnost: {col_name}")
    g.tight_layout()
    if fig_location:
        g.savefig(fig_location)
    if show_figure:
        plt.show()

# Ukol4: druh srážky jedoucích vozidel
def plot_direction(df: pd.DataFrame, fig_location: str = None,
                   show_figure: bool = False):
    df = df.copy()
    df = df.drop(df[df['p7'] == 0].index) # deleting rows with 0
    df.loc[df['p7'] == 1, 'p7'] = 'celni'
    df.loc[(df['p7'] == 2) | (df['p7'] == 3), 'p7'] = 'bocni'
    df.loc[df['p7'] == 4, 'p7'] = 'zezadu'
    df['months'] = df['p2a'].dt.month
    table = pd.pivot_table(data=df, values='p1', index=['region', 'months', 'p7'], aggfunc='count').reset_index()
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    g = sns.catplot(data=table, x='months', y='p1', kind='bar', hue='p7', col='region', col_order=['HKK', 'VYS', 'PAK', 'OLK'], col_wrap=2, sharex=False, legend_out=True)
    g.set_titles("Kraj: {col_name}")
    g._legend.set_title("Druh srazky")
    g.set_axis_labels('Mesic', 'Pocet nehod')
    g.tight_layout()
    if show_figure:
        plt.show()
    if fig_location:
        g.savefig(fig_location)

# Ukol 5: Následky v čase
def plot_consequences(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    df = df.copy()
    df= df[df['region'].isin(['HKK', 'VYS', 'PAK', 'OLK'])]
    df['p2a'] = pd.to_datetime(df["p2a"].dt.strftime('%Y-%m'))
    table = pd.pivot_table(df, values=['p13a', 'p13b', 'p13c'], index=['p2a','region'], aggfunc=np.sum)
    table = pd.melt(table.reset_index(), id_vars=['p2a', 'region'], value_vars=['p13a', 'p13b', 'p13c'])
    sns.set_style("darkgrid")
    sns.set_palette('deep')
    facet_kws={'sharey': True, 'sharex': False, 'legend_out': True}
    g = sns.relplot(data=table, x='p2a', y='value', kind='line', col='region', col_wrap=2, hue='variable', facet_kws=facet_kws)
    g._legend.set_title("Nasledky")
    sns.move_legend(g, "center right", bbox_to_anchor=(1.05, 0.5))
    for t, l in zip(g._legend.texts, ['Usmrceni', 'Tezke zraneni', 'Lehke zraneni']):
        t.set_text(l)
    g.set_axis_labels('Datum', 'Pocet nehod')
    g.set_titles("Kraj: {col_name}")
    g.set_xticklabels([f'01/{x}' for x in range(15,23)])
    g.set(xlim=(pd.to_datetime("20160101", format='%Y%m%d'), pd.to_datetime("20220101", format='%Y%m%d')))
    g.tight_layout()
    if show_figure:
        plt.show()
    if fig_location:
        g.savefig(fig_location)

if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni 
    # funkce.
    df = load_data("data.zip")
    df2 = parse_data(df, True)
    
    plot_visibility(df2, "01_visibility.png", True)
    plot_direction(df2, "02_direction.png", True)
    plot_consequences(df2, "03_consequences.png")


# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
