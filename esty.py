import pandas as pd
import numpy as np


class Main():
    '''
    self.dfを更新していく感じで。

    '''

    def __init__(self):
        self.df = pd.DataFrame()
        self.param = {}

    def basic_features(self):
        '''
        self.dfに基本的な情報追加

        :return:
        '''

        def chg_ndays(series, n):
            return series.fillna(method= "ffill").pct_change(n)
        def ma_ndays(series, n):
            return series.fillna(method= "ffill").rolling(n).mean()

        ma_window = self.param["ma_window"]

        self.df = self.df.sort_values("date")
        self.df["chg_1d"] = self.df.groupby("id")["px_last"].transform(chg_ndays, 1)
        self.df[f"ma_{ma_window}d"] = self.df.groupby("id")["px_last"].transform(ma_ndays, ma_window)
        return

    def longlong_pl(self):
        '''
        株と債券からロングロングの
        日次リターン、累積（和）リターン、最大ドローダウン
        '''

        w_es = self.param["weight_es"]
        w_ty = self.param["weight_ty"]

        pivot = self.df[self.df["id"].isin(self.param["ticker_future"])]\
            .pivot(index= "date", columns= "id", values= "chg_1d")
        pivot[f"longlong_{w_es}_{w_ty}"] = pivot["ES1 Index"]* w_es + pivot["TY1 Comdty"]* w_ty
        stack = pivot[[f"longlong_{w_es}_{w_ty}"]].stack()\
            .reset_index()\
            .rename(columns= {0: "chg_1d"})\
            .sort_values("date")
        stack["cumsum_chg"] = stack["chg_1d"].cumsum()
        stack["cumprod_chg"] = stack["chg_1d"].cumprod()* 100.
        stack["max_dd"] = stack["cumsum_chg"] - stack["cumsum_chg"].expanding().max()

        self.df = pd.concat([self.df, stack], axis= 0, join= "outer", sort= True)
        return stack

    def adjust_df_to_sign(self) -> pd.DataFrame:
        '''
        sign作成用のデータフレーム作成
        # key: date
        # id: vix,move(px_last, ma), longlong(dd)
        '''

        ma_window = self.param["ma_window"]
        longlong_id = [i for i in self.df.id if "longlong" in i]

        df0 = self.df[self.df["id"].isin(self.param["ticker_vol"])]\
            .loc[:, ["date", "id", "px_last", f"ma_{ma_window}d"]]\
            .rename(columns= {"px_last": "px", f"ma_{ma_window}d": "ma"})
        df0["id"] = df0["id"].apply(lambda x: str.lower(x.split(" ")[0]))
        df0 = df0.set_index(["date", "id"]).stack().reset_index() \
            .pivot_table(index="date", columns=["id", "level_2"], values=0)
        df0.columns = ['_'.join(col) for col in df0.columns]

        df1 = self.df[self.df["id"].isin(longlong_id)]\
            .loc[:, ["date", "max_dd"]]
        df = pd.merge(df0, df1, on= ["date"], how= "right")
        return df

    def signs_dd_only(self):
        '''
        entry: DDがadhoc超
        exit: keepdays後

        :return:
        '''

        # 分析用のdf作成
        df = self.adjust_df_to_sign()

        # まずはVIX,MOVEどちらかがMA下抜け かつ DDadhoc超にフラグ作成
        ad = self.param["dd_adhoc"]
        days = self.param["keep_days"]
        df["target"] = df["max_dd"].apply(lambda x: 1 if x <= ad else 0)

        # 各フラグのスタート時点のみ抽出
        df[f"sign_{ad}_{days}"] = df["target"].rolling(days).max()
        df[f"position_{ad}_{days}"] = df[f"sign_{ad}_{days}"].shift()

        # データ統合
        stack = df[["date", f"position_{ad}_{days}"]].copy()
        stack["id"] = "dd_only"
        self.df = pd.concat([self.df, stack]
                            ,axis= 0
                            ,join= "outer"
                            ,sort= True)
        return df
    
    def signs_vixmove(self):
        '''
        entry: vix, moveどちらかがMA下抜け　かつ　DDがadhoc超
        exit: keepdays後

        :return:
        '''

        # 分析用のdf作成
        df = self.adjust_df_to_sign()

        # まずはVIX,MOVEどちらかがMA下抜け かつ DDadhoc超にフラグ作成
        ad = self.param["dd_adhoc"]
        days = self.param["keep_days"]
        f0 = lambda x: 1 if x < 0 else 0
        f1 = lambda x: 1 if x >= 1 else 0
        df["move_lower"] = (df["move_px"] - df["move_ma"]).apply(f0)
        df["vix_lower"] = (df["vix_px"] - df["vix_ma"]).apply(f0)
        df["lower"] = (df["move_lower"] + df["vix_lower"]).apply(f1)
        df["dd_upper"] = df["max_dd"].apply(lambda x: 1 if x <= ad else 0)
        df["target"] = (df["lower"] * df["dd_upper"])

        # 各フラグのスタート時点のみ抽出
        df[f"sign_{ad}_{days}"] = df["target"].rolling(days).max()
        df[f"position_{ad}_{days}"] = df[f"sign_{ad}_{days}"].shift()

        # データ統合
        stack = df[["date", f"position_{ad}_{days}"]].copy()
        stack["id"] = "vixmove"
        self.df = pd.concat([self.df, stack]
                            ,axis= 0
                            ,join= "outer"
                            ,sort= True)
        return df

    def signs_move(self):
        '''
        entry: moveがMA下抜け　かつ　DDがadhoc超
        exit: keepdays後

        :return:
        '''

        # 分析用のdf作成
        df = self.adjust_df_to_sign()

        # まずはVIX,MOVEどちらかがMA下抜け かつ DDadhoc超にフラグ作成
        ad = self.param["dd_adhoc"]
        days = self.param["keep_days"]
        f0 = lambda x: 1 if x < 0 else 0
        df["move_lower"] = (df["move_px"] - df["move_ma"]).apply(f0)
        df["dd_upper"] = df["max_dd"].apply(lambda x: 1 if x <= ad else 0)
        df["target"] = (df["move_lower"] * df["dd_upper"])

        # 各フラグのスタート時点のみ抽出
        df[f"sign_{ad}_{days}"] = df["target"].rolling(days).max()
        df[f"position_{ad}_{days}"] = df[f"sign_{ad}_{days}"].shift()

        # データ統合
        stack = df[["date", f"position_{ad}_{days}"]].copy()
        stack["id"] = "move"
        self.df = pd.concat([self.df, stack]
                            ,axis= 0
                            ,join= "outer"
                            ,sort= True)
        return df


    def portfolio_return(self):
        '''
        ポジションとロングロングリターンから日次リターン算出
        '''

        # ポジション
        ad = self.param["dd_adhoc"]
        days = self.param["keep_days"]
        pos = self.df[["date", "id", f"position_{ad}_{days}"]]\
            .dropna()\
            .pivot(index= "date", columns= "id", values= f"position_{ad}_{days}")

        # ロングロングリターン
        w_es = self.param["weight_es"]
        w_ty = self.param["weight_ty"]
        ll_chg = self.df[self.df.id == f"longlong_{w_es}_{w_ty}"]\
                     .loc[:, ["date", "chg_1d"]].set_index("date")

        # リターン
        calc = pd.merge(pos, ll_chg, right_index= True, left_index= True)
        rtn = calc[list(pos.columns)].apply(lambda x: x * calc["chg_1d"])
        stack = rtn.stack().reset_index()\
            .rename(columns= {"level_1": "id", 0: "ll_chg_1d"})
        self.df = pd.merge(self.df, stack, on= ["date", "id"], how= "left")
        return calc, stack
    