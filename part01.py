#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Matous Slonek

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""


from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def integrate(x: np.array, y: np.array) -> float:
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])/2)


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None=None):
    x = np.linspace(-3, 3, 1000)
    u, v, w = [t* x**2 for t in a]
    fig, ax = plt.subplots()
    ax.plot(x, u, label=f'$y_{{{a[0]}}}(x)$')
    ax.plot(x, v, label=f'$y_{{{a[1]}}}$(x)')
    ax.plot(x, w, label=f'$y_{{{a[2]}}}(x)$')
    ax.fill_between(x, v, color='orange', alpha=0.1)
    ax.fill_between(x, w, color='green', alpha=0.1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-21, 21)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$f_a(x)$')
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.13))
    ax.text(3.1, 9, f'$\int f_{{{a[0]}}}(x)dx$')
    ax.text(3.1, 18, f'$\int f_{{{a[1]}}}(x)dx$')
    ax.text(3.1, -18, f'$\int f_{{{a[2]}}}(x)dx$')
    ax.spines["top"].set_bounds(-3, 4.2)
    ax.spines["bottom"].set_bounds(-3, 4.2)
    ax.spines["right"].set_position(('outward', 66))
    fig.tight_layout()
    if show_figure:
        fig.show()
    if save_path:
        fig.savefig(save_path)


def generate_sinus(show_figure: bool=False, save_path: str | None=None):
    t = np.linspace(0, 100, 4000)
    f1 = 0.5 * np.sin((np.pi * t) / 50)
    f2 = 0.25 * np.sin(np.pi * t)
    plt.figure(figsize=(6,8))
    plt.subplot(311)
    plt.plot(t, f1)
    plt.yticks(np.arange(-0.8, 0.81, step=0.4))
    plt.xlim(0, 100)
    plt.xlabel('t')
    plt.ylabel('$f_1(t)$')
    plt.subplot(312)
    plt.plot(t, f2)
    plt.yticks(np.arange(-0.8, 0.81, step=0.4))
    plt.xlim(0, 100)
    plt.xlabel('t')
    plt.ylabel('$f_2(t)$')
    plt.subplot(313)
    plt.ylim(-0.8, 0.8)
    plt.yticks(np.arange(-0.8, 0.81, step=0.4))
    plt.xlim(0, 100)
    f12 = f1 + f2
    upper = np.where(f1+f2>f1, f12, None)
    lower = np.where(f1+f2<f1, f12, None)
    plt.plot(t, upper, color='green')
    plt.plot(t, lower, color='red')
    plt.xlabel('t')
    plt.ylabel('$f_1(t)+f_2(t)$')
    plt.tight_layout(h_pad=2)
    if show_figure:
        plt.show()
    if save_path:
        plt.savefig(save_path)


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html"):
    site = requests.get(url)
    soup = BeautifulSoup(site.text, 'xml')
    data = [{"year": int(row.find_all("p")[0].text),
            "month": int(row.find_all("p")[1].text),
            "temp": np.array([float(x.text.replace(',','.')) for x in row.find_all("p")[2:]])}
            for row in soup.find_all('tr')]
    return data


def get_avg_temp(data, year=None, month=None) -> float:
    if value:= year or month: # one
        key = 'year' if year else 'month'
        data = [x for x in data if x[key] == value]
    if year and month:  # both
        data = [x for x in data if x['year'] == year and x['month'] == month]
    temps = np.concatenate(np.array([x['temp'] for x in data], dtype=object))
    return np.mean(temps)
