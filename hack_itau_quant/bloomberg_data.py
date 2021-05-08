import pandas as pd
from xbbg import blp, pipeline, const

class BloombergData:

    def get_hack_data():

        df_IBOV = blp.bdh('BOVV11 BZ Equity', 'PX_LAST', '2017-12-29', '2021-04-30')
        df_SP   = blp.bdh('SPXI11 BZ Equity', 'PX_LAST', '2017-12-29', '2021-04-30')
        df_IMAB = blp.bdh('IMAB11 BZ Equity', 'PX_LAST', '2017-12-29', '2021-04-30')
        df_IRFM = blp.bdh('IRFM11 BZ Equity', 'PX_LAST', '2017-12-29', '2021-04-30')

        df_IBOV.index = pd.to_datetime(df_IBOV.index)
        df_SP.index = pd.to_datetime(df_SP.index)
        df_IMAB.index = pd.to_datetime(df_IMAB.index)
        df_IRFM.index = pd.to_datetime(df_IRFM.index)

        df_IMAB_indice = blp.bdh('BZRFIMAB index', 'PX_LAST', '2017-12-30', '2019-05-19')
        df_IMAB_indice.index = pd.to_datetime(df_IMAB_indice.index)
        df_IMAB_indice = df_IMAB_indice.rename(columns ={'BZRFIMAB index' : 'IMAB11 BZ Equity'})
        df_IMAB = pd.concat([df_IMAB_indice/1000, df_IMAB])

        df_IRFM_indice = blp.bdh('BZRFIRFM Index', 'PX_LAST', '2017-12-30', '2019-09-22')
        df_IRFM_indice.index = pd.to_datetime(df_IRFM_indice.index)
        df_IRFM_indice = df_IRFM_indice.rename(columns ={'BZRFIRFM Index' : 'IRFM11 BZ Equity'})
        df_IRFM = pd.concat([df_IRFM_indice/1000, df_IRFM])

        df = pd.concat([df_IBOV, df_SP, df_IMAB, df_IRFM], axis = 1).dropna()

        return df