from time import time
import gc
import numpy as np
from esn_dev.input_map import InputMap
from esn_dev import hidden, optimize, readout
from esn_dev.utils import preprocess, save_error
from esn_dev.visualize_results_anom import *

def prediction_error(
    data,
    lon,
    specs,
    T_int=60,
    savedir='tmp',
    spectral_radius=1.5,
    neuron_connections=10,
    Ntrans=500,
    Ntrain=2500,
    Npred=500,
    plot_prediction=True,
    sigma=(1, 5, 5),
    eps=1e-5,
    n_PCs=None,
    dtype=np.float64,
    lstsq_method='scipy',
    lstsq_thresh=None,
    ST_method='DCT',
    cpus_to_use=32,
    scale_min=-1,
    scale_max=1,
    neuron_dist='normal',
    upper_sr_calc_dim=5000,
    save_condition='always',
    random_seed=None,
    config=None,
    **kwargs
):

    start_tottime = time()
    np.random.seed(random_seed)
    seed = str(np.random.get_state()[1][0])
    config['random_seed'] = seed

    N = Ntrain + Npred + 1
    assert data.ndim == 3 and data.shape[0] >= N
    img_shape = data.shape[1:]
    print(f"Image shape is {img_shape}")

    # --- Preprocessing ---
    print("Starting pre-processing...")

    train_inputs = data[1:Ntrain]
    train_inputs, training_min, training_max = preprocess(
        data=train_inputs, training_min=None, training_max=None, **kwargs
    )
    config['training_min'] = training_min
    config['training_max'] = training_max

    del train_inputs
    gc.collect()

    data_ST = preprocess(data=data, **config)

    print("Finished pre-processing!")

    # --- Training/Prediction/Error Calculation ---
    print("Starting training/ prediction/ error calculation ... ")

    time_int_spat_error_li = []
    time_space_int_scalar_error_li = []
    predictions_ST_li = []
    pred_targets = data[Ntrain+1:Npred]


    for t in range(Npred - T_int + 1):
        if t % 100 == 0:
            print(f"Time step: {t}")

        train_inputs = data_ST[t:t+Ntrain]
        train_targets = data_ST[t+Ntrans:Ntrain+t]
        pred_targets_ST = data_ST[Ntrain+t:Ntrain+t+T_int]

        # Build ESN
        map_ih = InputMap(specs)
        hidden_size = map_ih.output_size(img_shape)
        esn = hidden.initialize_dynsys(
            map_ih, hidden_size, spectral_radius, neuron_connections,
            dtype=dtype, upper_sr_calc_dim=upper_sr_calc_dim
        )

        h_trans = hidden.evolve_hidden_state(esn, train_inputs[:Ntrans], h=np.zeros(hidden_size), mode='transient')
        H = hidden.evolve_hidden_state(esn, train_inputs[Ntrans:], h=h_trans, mode='train')
        y0, h0 = train_targets[-1], H[-1]

        train_targets = train_targets.reshape(train_targets.shape[0], -1)
        H, pca_object = hidden.dimension_reduce(H, pca_object=None, n_PCs=n_PCs)

        esn = optimize.create_readout_matrix(
            esn, H, train_targets,
            lstsq_method=lstsq_method,
            lstsq_thresh=lstsq_thresh,
            dtype=dtype
        )

        predictions_ST = readout.predict(esn, y0, h0, Npred=T_int, pca_object=pca_object)
        pred_error = np.abs(predictions_ST - pred_targets_ST)
        time_int_spat_error_t = pred_error.sum(axis=0)
        time_space_int_scalar_error_t = time_int_spat_error_t.sum()

        time_int_spat_error_li.append(time_int_spat_error_t)
        time_space_int_scalar_error_li.append(time_space_int_scalar_error_t)

        predictions_ST_li.append(predictions_ST[0])

    # --- Post-loop: Stack + inverse preprocess ---
    predictions_ST_all = np.array(predictions_ST_li)         # shape: (Nt, T_int, M, N)
    predictions_iST_all = preprocess(data=predictions_ST_all, inverse=True, **config)

    time_int_spat_error = np.array(time_int_spat_error_li)
    time_space_int_scalar_error = np.array(time_space_int_scalar_error_li)

    # --- Timing ---
    end_tottime = time()
    elapsed_time = end_tottime - start_tottime
    print(f"Total time: {elapsed_time:.1f}s")
    config['timings_dictionary'] = [f"Total time (no plotting/saving): {elapsed_time:.1f}s"]

    # --- Save ---
    print("Saving ...")
    arrays_to_save = {
        "y0": y0,
        "h0": h0,
        "Whh": esn[1],
        "bh": esn[2],
        "Who": esn[3],
        "time_int_spat_error": time_int_spat_error,
        "time_space_int_scalar_error": time_space_int_scalar_error,
        "predictions": predictions_iST_all,
        "pred_targets": pred_targets,
    }

    folder = save_error(
        dict_of_arrays=arrays_to_save,
        param_dict=config,
        savedir=config['savedir'],
    )

    print("Done!")
    return