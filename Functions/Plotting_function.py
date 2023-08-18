from Functions.get_data import get_df, get_calving_events
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates


def Plot_all_three(Node):
    df, df_og = get_df(Node)
    Quakes_df, calv_pt_df = get_calving_events()
    """one node"""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Node 9 Vs Tide Gauge and Weather ", fontsize=20)
    plt.style.use('bmh')

    ax1.scatter(df["datetime"], df["north_east_speed_mpd"], s=4, color="black")
    ax1.scatter(df["datetime"], df["north_east_speed_mpd_20_min"], s=4, color="red")

    ax2.scatter(df["datetime"], df["kf_height"], s=4, color="black")
    ax2.scatter(df["datetime"], df["kf_height_5min"], s=4, color="red")

    ax1.set_ylabel("Horizontal Flow (m/d)")
    ax2.set_ylabel("Height (m)")

    for calve_event in range(len(calv_pt_df)):
        c_event = calv_pt_df["Datetime_obj"].loc[calve_event]
        if calve_event == 0:
            ax2.axvline(x=c_event, color='grey', zorder=0, label="Camera Calving")
        else:
            ax2.axvline(x=c_event, color='grey', zorder=0)

    for Seismic_event in range(len(Quakes_df)):
        s_event = Quakes_df["datetime"].loc[Seismic_event]
        if Seismic_event == 0:
            ax2.axvline(x=s_event, color='green', zorder=0, label="Seismic Calving")
        else:
            ax2.axvline(x=s_event, color='green', zorder=0)

    fig.legend(ncol=6, loc="lower center", fontsize=15, scatterpoints=1, markerscale=3, )
    ax2.xaxis.set_major_formatter(mpl_dates.DateFormatter("%j"))
    ax2.set_xlabel("DOY")
    plt.show()



