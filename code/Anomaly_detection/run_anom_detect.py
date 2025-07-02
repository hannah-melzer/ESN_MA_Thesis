import numpy as np

region = "Kuro_new_2"

CESM_data = np.load(f"/home/hmelzer/work/esn/CESM/Data/ssh_{region}_5d.npy")
CESM_lon = np.load(f"/home/hmelzer/work/esn/CESM/Data/lon_{region}.npy")

from Anomaly_Detection_ESN_Function_sigma0 import anomaly_detection_ESN

def CESM(savedir):
    data = CESM_data

    input_shape = (data.shape[1], data.shape[2])
    input_size  = input_shape[0] * input_shape[1]
    
    input_shape = (data.shape[1], data.shape[2])
    input_size  = input_shape[0] * input_shape[1]
    
    Ntrain = 657 #9 years (25-33) 3d - 1095
    Npred = 583 #8 years (34-41) 3d - 972
    Ntrans = 146 #2 years (25-26) 3d - 243
    N = Npred+Ntrain+1

    k = 0.1
    
    specs = [
        {"type": "pixels", "size": (10, 10), "factor": k},
        {"type": "conv", "size": (10, 10), "kernel": "gauss",  "factor": k},
        {"type": "conv", "size": (10, 10), "kernel": "random",  "factor": k},
        {"type": "gradient", "factor": 1e-3},
        {"type": "vorticity", "factor": 1e-3},
        {"type": "dct", "size": (10, 10), "factor": k},
        {"type": "random_weights", "input_size": input_size, "hidden_size": 10000, "factor": 10}
    ]

    
    parameter_dict = dict(
        specs = specs,
        Npred  = Npred,
        Ntrain = Ntrain,
        Ntrans = Ntrans,
        spectral_radius = 0.94,
        neuron_connections = 100,
        n_PCs  = 350,
        sigma  =  (0,1,1),
        eps    =  1e-2,
        plot_prediction = True,
        dtype='float64',
        lstsq_method ='svd',
        lstsq_thresh = 1e-3,
        ST_method = 'DCT',
        cpus_to_use = 32,
        scale_min = -1,
        scale_max =  1,
        savedir = savedir,
        neuron_dist = 'normal',
        upper_sr_calc_dim=5000,
        save_condition = 'always',
        random_seed = np.random.seed(),
        
        
    )

    return anomaly_detection_ESN(data, CESM_lon, T_int = 73, large_window = 365, small_window = 18,config=parameter_dict, **parameter_dict)

CESM(region+"_anom")