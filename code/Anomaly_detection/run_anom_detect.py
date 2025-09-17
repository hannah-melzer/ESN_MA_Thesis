import numpy as np
import os
from calc_prediction_error_new import prediction_error

region = "GS_new"

# Load CESM data
CESM_data = np.load(f"/home/hmelzer/work/esn/CESM/Data/ssh_{region}.npy")
CESM_lon = np.load(f"/home/hmelzer/work/esn/CESM/Data/lon_{region}.npy")

def CESM(savedir_base, T_int=73):
    data = CESM_data

    input_shape = (data.shape[1], data.shape[2])
    input_size  = input_shape[0] * input_shape[1]

    Ntrain = 511  # 7 years
    Npred = 730   # 10 years
    Ntrans = 146  # 2 years

    k = 0.1

    specs = [
        {"type": "pixels", "size": (10, 10), "factor": k},
        {"type": "conv", "size": (10, 10), "kernel": "gauss",  "factor": k},
        {"type": "conv", "size": (10, 10), "kernel": "random", "factor": k},
        {"type": "gradient", "factor": 1e-3},
        {"type": "vorticity", "factor": 1e-3},
        {"type": "dct", "size": (10, 10), "factor": k},
        {"type": "random_weights", "input_size": input_size, "hidden_size": 10000, "factor": 10}
    ]

    savedir = savedir_base
    os.makedirs(savedir, exist_ok=True)  # ensure directory exists

    parameter_dict = dict(
        specs=specs,
        Npred=Npred,
        Ntrain=Ntrain,
        Ntrans=Ntrans,
        spectral_radius=0.94,
        neuron_connections=100,
        n_PCs=350,
        sigma=(0, 1, 1),
        eps=1e-2,
        plot_prediction=True,
        dtype='float64',
        lstsq_method='svd',
        lstsq_thresh=1e-3,
        ST_method='DCT',
        cpus_to_use=32,
        scale_min=-1,
        scale_max=1,
        savedir=savedir,
        neuron_dist='normal',
        upper_sr_calc_dim=5000,
        save_condition='always',
        random_seed=np.random.seed(),
    )

    print(f"Running CESM with T_int = {T_int} → {savedir}")
    result = prediction_error(
        data,
        CESM_lon,
        T_int=T_int,
        config=parameter_dict,
        **parameter_dict
    )
    print(f"Finished CESM with T_int = {T_int}")
    return result

# === Main Execution ===
T_int_list = [18, 24, 36, 54, 73] #3 months, 4 mo, 6 mo, 9 mo, 1 yr

for T_int in T_int_list:
    savedir_base = f"CESM/{region}_new_anom_T_int{T_int}"
    CESM(savedir_base, T_int=T_int)
