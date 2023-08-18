import numpy as np
import pandas as pd
import time
from Functions.second_interval_function import DF_1_sec, Filter_data


def Run_Kalman_3D(node, sigma_h, noise_v, noise_p):
    start_time = time.time()

    """Open the CSV files converting to Pandas then Removes Known Bad data"""
    df_OG = pd.read_csv(f'Example node/{node}.csv')
    df_f = Filter_data(df_OG)

    """Function Making 1 second Data"""
    df_beginning = DF_1_sec(df_f)

    """Turns df (dn,de,du,datetime) to a list"""
    df = df_beginning[["dn", "de", "du", "datetime"]].values.tolist()

    """Variables"""
    sigma_height = sigma_h
    process_noise_position = noise_p
    process_noise_velocity = noise_v

    Transition_matrix = [[1, 1], [0, 1]]
    Process_Noise_VCM = [[process_noise_position, 0], [0, process_noise_velocity]]
    Design_matrix_A = [1, 0]

    "first run of the filter has to be done separately"
    data_ls = [{"epoch": 0,
                "height": 0,
                "northing": 0,
                "easting": 0,
                "sigma_height": 999999,
                "kf_height": 0,
                "kf_height_velocity": 0,
                "kf_northing": 0,
                "kf_northing_velocity": 0,
                "kf_easting": 0,
                "kf_easting_velocity": 0,
                "corrected_state_H": [150, 0],
                "corrected_state_N": [7361000, -0.00005],
                "corrected_state_E": [536000, 0.00025],
                "corrected_state_VCM": np.array([[1000000000, 0], [0, 0.001]]),
                "datetime": np.nan,
                "5mins_time_gap": 0,
                }]

    """Calculation Functions"""

    def get_measurements(z, data_gp):
        if np.isnan(z[0]):
            n, e, u, sig_h = 0, 0, 0, 999999
            data_gp += 1

        else:
            sig_h = sigma_height
            n = z[0]
            e = z[1]
            u = z[2]
            data_gp = 0

        date = z[3]
        return n, e, u, date, sig_h, data_gp,

    def get_previous_data(g):
        p_c_s_H = data_ls[g]["corrected_state_H"]
        p_c_s_N = data_ls[g]["corrected_state_N"]
        p_c_s_E = data_ls[g]["corrected_state_E"]

        P_C_S_VCM = data_ls[g]["corrected_state_VCM"]
        return p_c_s_H, p_c_s_N, p_c_s_E, P_C_S_VCM

    def predicted_state_calc(ar_h, ar_n, ar_e):
        p_c_H = np.matmul(Transition_matrix, ar_h)
        p_c_N = np.matmul(Transition_matrix, ar_n)
        p_c_E = np.matmul(Transition_matrix, ar_e)
        return [p_c_H, p_c_N, p_c_E]

    def get_innovation(ps_ls):
        if x == 0:
            vi_h = 0
            vi_e = 0
            vi_n = 0
        else:
            if du == 0:
                vi_h, vi_e, vi_n = 0, 0, 0
            else:
                vi_h = du - ps_ls[0][0]
                vi_n = dn - ps_ls[1][0]
                vi_e = de - ps_ls[2][0]

        return [vi_h, vi_n, vi_e]

    def get_Psss(p_s_vcm, obs_ci):
        p_one = np.matmul(p_s_vcm, np.transpose(Design_matrix_A))
        try:
            p_two = np.matmul(Design_matrix_A, p_one)
        except:
            print("Design_matrix_A", Design_matrix_A)
            print("p_one", p_one)
        p_three = p_two + obs_ci
        return p_one, p_two, p_three

    def get_kalman_gain(p_1, p_3):
        inverse_matrix = np.linalg.inv([[p_3]])
        p1_1 = p_1[0] * inverse_matrix
        p1_2 = p_1[1] * inverse_matrix
        ans = np.array([p1_1[0][0], p1_2[0][0]])
        return ans

    def get_gain_x_vi(k_g, v_i):
        g_x_vi_h = v_i[0] * k_g
        g_x_vi_n = v_i[1] * k_g
        g_x_vi_e = v_i[2] * k_g
        return [g_x_vi_h, g_x_vi_n, g_x_vi_e]

    def get_correct_state_now(p_s_ls, g_x_vi_ls):
        cor_stat_h = p_s_ls[0] + g_x_vi_ls[0]
        cor_stat_n = p_s_ls[1] + g_x_vi_ls[1]
        cor_stat_e = p_s_ls[2] + g_x_vi_ls[2]
        return [cor_stat_h, cor_stat_n, cor_stat_e]

    def get_G_A_Cx_i(K_G, D_M, Ps_vcm):
        t1 = np.matmul(D_M, Ps_vcm)
        a = t1[0]
        b = t1[1]
        one = K_G[0] * a
        two = K_G[1] * a
        three = K_G[0] * b
        four = K_G[1] * b

        ans = np.array([[one, two], [three, four]])
        return ans

    """Variable to track Data gaps """
    data_gap = 0

    """Loop through each epoch"""
    for x in range(len(df)):
        if x in np.arange(0, 4000000, 50000):
            print("Epoch:", x)

        if x > 900000:
            break
        epoch = x + 1
        dn, de, du, datetime_obj, sigma_height_, data_gap = get_measurements(df[x], data_gap)

        Prev_C_S_H, Prev_C_S_N, Prev_C_S_E, Previous_Corrected_state_vcm = get_previous_data(x)

        predicted_stat_ls = predicted_state_calc(Prev_C_S_H, Prev_C_S_N, Prev_C_S_E, )
        transition_VCM = np.around(np.matmul(Transition_matrix, (np.matmul(Previous_Corrected_state_vcm,
                                                                           np.transpose(Transition_matrix)))),
                                   decimals=8)

        P_S_VCM = transition_VCM + Process_Noise_VCM

        Vi_ls = get_innovation(predicted_stat_ls)

        ci = sigma_height_ ** 2

        p1, p2, p3 = get_Psss(P_S_VCM, ci)

        Kalman_gain_G = get_kalman_gain(p1, p3)
        G_x_Vi_ls = get_gain_x_vi(Kalman_gain_G, Vi_ls)

        cor_state_h, cor_state_n, cor_state_e = get_correct_state_now(predicted_stat_ls, G_x_Vi_ls)

        Corrected_state_ls = [cor_state_h, cor_state_n, cor_state_e]
        G_A_Cx_i = get_G_A_Cx_i(Kalman_gain_G, np.array(Design_matrix_A), np.array(P_S_VCM))
        Corrected_state_vcm = (P_S_VCM - G_A_Cx_i)

        time_step_large = 0
        if data_gap >= 300:
            time_step_large = 1
            Corrected_state_ls = [[Prev_C_S_H[0], 0], cor_state_n, cor_state_e]
            Corrected_state_vcm = [1000000000, 0], [0, 0.001]

        values = {"epoch": epoch,
                  "height": du,
                  "northing": dn,
                  "easting": de,
                  "sigma_height": sigma_height_,
                  "predicted_state": predicted_stat_ls,
                  "vi": Vi_ls,
                  "ci": ci,
                  "p1": p1,
                  "p2": p2,
                  "p3": p3,
                  "kalman_gain": Kalman_gain_G,
                  "G_x_Vi": G_x_Vi_ls,
                  "corrected_state_H": Corrected_state_ls[0],
                  "corrected_state_N": Corrected_state_ls[1],
                  "corrected_state_E": Corrected_state_ls[2],
                  "corrected_state_VCM": Corrected_state_vcm,
                  "kf_height": Corrected_state_ls[0][0],
                  "kf_height_velocity": Corrected_state_ls[0][1],
                  "kf_northing": Corrected_state_ls[1][0],
                  "kf_northing_velocity": Corrected_state_ls[1][1],
                  "kf_easting": Corrected_state_ls[2][0],
                  "kf_easting_velocity": Corrected_state_ls[2][1],
                  "datetime": datetime_obj,
                  "5mins_time_gap": time_step_large,
                  "Corrected_state_vcm_00": Corrected_state_vcm[0][0],
                  "Corrected_state_vcm_01": Corrected_state_vcm[0][1],
                  "Corrected_state_vcm_10": Corrected_state_vcm[1][0],
                  "Corrected_state_vcm_11": Corrected_state_vcm[1][1],
                  }
        data_ls.append(values)
    print("Converting to Dataframe")
    df_kf = pd.DataFrame(data_ls)

    df_2 = df_kf[["datetime", "height", "northing",
                  "easting", "epoch", "kf_height",
                  "kf_height_velocity", "kf_northing","kf_northing_velocity",
                  "kf_easting", "kf_easting_velocity", "5mins_time_gap",
                  "sigma_height"]]

    df_2["north_east_speed_mpd"] = np.sqrt(
        (df_2["kf_easting_velocity"] ** 2) + (df_2["kf_northing_velocity"] ** 2)) * 86400

    save_location = f"Filtered_Nodes\{node}_{process_noise_position}_{sigma_h}.csv"

    print("Saving.....")
    df_2.to_csv(save_location)

    finish_time = time.time()
    time_taken = finish_time - start_time
    print("\nThis took:", "%.2f" % time_taken, "seconds")
