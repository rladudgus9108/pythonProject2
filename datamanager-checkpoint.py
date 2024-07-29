from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class DataManager:

    def __init__(self, data_path: Path, taginfo_path: Path):
        self.data_path = data_path
        self._load_taginfo(taginfo_path)

    def _load_taginfo(self, taginfo_path) -> None:
        self.taginfo = pd.read_csv(taginfo_path, encoding='euc-kr')

    def get_tag_desc(self, tagname) -> str:
        desc = self.taginfo[self.taginfo.TagName == tagname].Description.values[0]
        return f"{tagname} | {desc}"
    
    @lru_cache(maxsize=1024*1024*1024)
    def load_data(self, tagname:str) -> None:
        df = pd.read_csv(self.data_path / f"{tagname}.csv")
        df.time = pd.to_datetime(df.time, format="mixed")
        df.set_index("time", inplace=True)
        df.drop(columns=["name", "sourcetime","quality", "tagname"], inplace=True)
        return df
    
    def plot_data(self, tagnames: list[str], start:str=None, end:str=None, figsize:tuple=(20,6)) -> None:
        if isinstance(tagnames, str):
            tagnames = [tagnames]
        df_list = [self.load_data(tagname)[start:end] for tagname in tagnames]
        plt.figure(figsize=figsize)
        for df, tagname in zip(df_list, tagnames):
            plt.step(df.index, df.values, label=self.get_tag_desc(tagname))
        plt.grid()
        plt.legend(loc=1)
        plt.show()

    def plot_data2(self, tagnames: list[str], tagnames2: list[str], start:str=None, end:str=None, figsize:tuple=(20,10)) -> None:
        if isinstance(tagnames, str):
            tagnames = [tagnames]
        df_list1 = [self.load_data(tagname)[start:end] for tagname in tagnames]

        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True, figsize=figsize)
        for df, tagname in zip(df_list1, tagnames):
            ax[0].step(df.index, df.values, label=self.get_tag_desc(tagname))
        ax[0].grid()
        ax[0].legend(loc=1)
        
        df_list2 = [self.load_data(tagname)[start:end] for tagname in tagnames2]
        for df, tagname in zip(df_list2, tagnames2):
            ax[1].step(df.index, df.values, label=self.get_tag_desc(tagname))
        ax[1].grid()
        ax[1].legend(loc=1)

        plt.show()