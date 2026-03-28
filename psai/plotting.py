
import os
import pandas as pd
import matplotlib.pyplot as plt

def save_line(df: pd.DataFrame, x: str, y: str, out_png: str, out_pdf: str, title: str = ""):
    fig = plt.figure()
    plt.plot(df[x].values, df[y].values)
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)

def save_hist(values, out_png: str, out_pdf: str, xlabel: str, title: str = ""):
    fig = plt.figure()
    plt.hist(values, bins=30)
    plt.xlabel(xlabel)
    if title:
        plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
