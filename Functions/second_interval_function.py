import pandas as pd
from datetime import datetime, timedelta



def DF_1_sec(df_):
    # print(df_.head(1))
    first_doy = datetime.strptime(df_.iloc[0].datetime, "%Y-%m-%d %H:%M:%S")
    last_doy = datetime.strptime(df_.iloc[-1].datetime, "%Y-%m-%d %H:%M:%S")
    # print("first: ", first_doy)
    # print("last: ", last_doy)

    def datetime_range(start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta

    timestep = 1
    dts = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in datetime_range(first_doy, last_doy, timedelta(seconds=timestep))]
    # print(dts)
    dts_df = pd.DataFrame(dts, columns=["datetime"])
    # print(dts_df)
    # print(df_.shape)
    # print(dts_df.shape)

    df_1_merge = pd.merge(dts_df, df_, how="outer", on=["datetime"])
    # print("\n\n\n")
    # print(df_1_merge)
    # print(df_1_merge.shape)

    return df_1_merge

def Filter_data(df_w):
    print(f"Pre filter: {df_w.shape}")
    df_w = df_w[df_w.num_unfixed_biases <= 2]
    df_w = df_w[df_w.num_double_differences >= 4]
    df_w = df_w[df_w.SDU < 0.1]
    print(f"Post filter: {df_w.shape}\n")
    return df_w


