#Written by Jacob Ungar Feldling and James Avery
#Reproduced and edited with permission by Hannah Melzer




# --- Tech Preamble ---

import numpy as np
from esn_dev.input_map import InputMap
from esn_dev import hidden, optimize, readout
from esn_dev.utils import *
from esn_dev.visualize_results_anom import *
from time import time
import gc
from esn_dev.anomaly import sliding_score




# --- Define Function = heart of the code --- 

def anomaly_detection_ESN(data, lon,
                             specs,
                             T_int = 60,
                             large_window = 200,
                             small_window = 5,
                             savedir='tmp',
                             spectral_radius=1.5,
                             neuron_connections=10,
                             Ntrans=500,        # Number of transient initial steps before training
                             Ntrain=2500,       # Number of steps to train on
                             Npred=500,         # Number of steps for free-running prediction
                             plot_prediction=True,
                             sigma=(1,5,5),
                             eps=1e-5,
                             n_PCs=None,
                             dtype=np.float64,
                             lstsq_method ='scipy',
                             lstsq_thresh = None,
                             ST_method = 'DCT',
                             cpus_to_use = 32,
                             scale_min = -1,
                             scale_max =  1,
                             neuron_dist = 'normal',
                             upper_sr_calc_dim=5000,
                             save_condition = 'always',   
                             random_seed = None,
                             config = None,
                             **kwargs):
    
    """
    Given data and configurations, use the scalable spatial echo state network
    to train on video-like input and predict future video frames in
    free-running-mode (predictions normally diverge from targets over time).
    
    ---
    What is the scalable spatial echo state network?

    ESN consists of dynamical system and a separate readout/prediction layer:
    The dynamical system is recurrent, and input driven, i.e.:
    
                        h[t+1] = tanh( Win . x+Whh . h[t] + bh ),
                        
    where h is a high-dimensional hidden state with some memory, x is the input,
    Win is a matrix mapping x-->h, Whh is the 'reservoir', a matrix mapping from
    h-->h. bh is an optional bias vector with dimension as h. Image input is flatened
    to vector format x.
    
    The prediction layer then uses h from training steps as regressors,
    gathered in a matrix H:
    
                                y[t+1] = Who.h[t]
                                
    Where Who is an output matrix mapping from h-->y (output, same dimesion
    as input, x) that is optimized for this purpose using the efficient 
    multiple least squares method; only one time. No gradient-based learning here. 
    
    Win and Whh are not optimized, and must be selected for the ESN to give good
    predictions. A number of parameters control this, like 'specs' (for Win) and 
    'spectral radius' (for Whh).
    --- 
    
    if 'save_condition' is True or 'always', save model and prediction
    arrays as '.npy' files along with the configuration dictionary 'config'
    (saved in human-readable .yaml, and machine-readable .pkl formats)
    if plot_prediction is True, also save an mp4-video comparing predictions
    with a target data set, unseen by the ESN, and a plot of MSE over time. 
    
    Params:
        data:        (T,M,N)-ndarray, where T is time, M,N are spatial.
        specs:       [list] Input map specifications, that create Win matrix.
                     Example:
        [{'type':'pixels','size':input_shape,'factor':1.0},
        {'type' :'conv','size':(9, 9),'kernel':'gauss','factor': 1.0},
        {'type' :'gradient','factor': 1.0},
        {'type' :'random_weights','input_size':(input_size),'hidden_size':1000,'factor':1.},
        {"type" :'gradient', 'factor' 1.},
        {"type" :'dct', 'size':(15,15), 'factor': 1}]

        savedir:            project folder to save at least one esn model in
                            if 'save_condition' is also fulfilled. The model is
                            saved in 'savedir/esn001', and if run again
                            'savedir/esn002', with an mse-overview in 'savedir'.
        spectral_radius:    spectral radius of reservoir Whh. Values around 1
                            are a good start. Crucial for network stability,
                            and good predictions.
        neuron_connections: Number of non-zero elements in each
                            row of reservoir matrix Whh.
        Ntrans:             Duration of part of training set _not_ to regress
                            on, but to use for warming up the dynamical system,
                            giving better hidden states, h. trans: Transient.
        Ntrain:             Total duration of training set (including Ntrans).
        Npred:              Number of prediction steps to use. Data set must be
                            large enough to have N=Ntrain+Npred steps so that
                            predictions can be compared to unseen target obs.
        plt_prediction:     (bool) Whether to plot animation and error-plots.
                            Only used if 'save_condition' is fulfilled.
        ST_method:          'FFT' or 'DCT'. Used for preprocessing of video to
                            enhance learning. ST is standardizing transform
                            inherent to the IMED: IMage Euclidean Distance.
                            FFT and DCT are frequency methods.
        sigma:              Parameter of spatial IMED pre-processing of the data
                            to enhance learning. Larger sigma blurs out image
                            more. If scalar, sigma is applied along both time
                            and spatial axes. Otherwise, must be specified as
                            sigma=(1,5,5) for 1 along time axis and 5 along 
                            both spatial axes (normally a better option.)
        eps:                Parameter of IMED-proprocessing of the data. 
                            The larger eps is, the more noise is suppressed
                            to allow post-processing to work (operations are
                            opposite of pre-processing steps). Should be a
                            small positive number, growing with 'sigma'.
                            1e-2 is usually a good choice.
        cpus_to_use:        Number of cpu cores to use for parallel standardizing
                            transform preprocessing and postprocessing.
                            Must be an int. If -1, uses all cores. Equal to
                            scipy.fft.___ 'workers' keyword.
        n_PCs:              Number of trained parameters to fit per pixel
                            in the image (video slice). Equally, it is 
                            the number of principal components used in a
                            PCA dimension reduction to improve scalability
                            and training performance. If None, PCA is not used,
                            and Nhidden parameters will be fitted per pixel.
        dtype:              Data type of the ESN-structures (not necessarily)
                            equal to data-dtype. np.float64, or 'float64' recommended,
                            as low precision propagates due to iterative modelling.
        lstsq_method:       If passed, choice of least squares method.
        lstsq_thresh:       Parameter of the 'lstsq_method' of choice. Allows fitting
                            with rank-deficient matrices, adds robustness.
        scale_min:          Scaled minimum of the training set. -1 is recommended,
                            as it is the minimum of the activation function tanh().
        scale_min:          Scaled maximum of the training set. 1 is recommended,
                            the maximum of activation function tanh()
        neuron_dist:        'uniform' or 'normal'. Distribution of random non-zero
                            values of matrix reservoir Whh.
        upper_sr_calc_dim:  (int) Used to adapt Whh to the chosen 'spectral_radius'.
                            If increased above dimension of h, inefficient but more precise
                            calculations are made. Dimension of h is a result of 'specs'
        save_condition:     Whether to save model output and plot (if 'plot_predictions').
                            Can be True, or False, to enable or disable,
                            or 'always', and 'never' for same functionality.
                            or 'if_better', to only save if lower mse com any previous 
                            models was achieved.
        random_seed:        Seed to inialize (and recover) random generator used for e.g.
                            creation of reservoir matrix Whh. Can be int or None.
        config:             Dictionary of all loaded parameters that is also used as 
                            input to this function using (...,config=config,**config).
                            If not passed, parameters and updated values from run
                            cannot be saved (when 'save_condition' is fulfilled).
    Returns:
        mse:                MSE of predictions versus unseen prediction targets.
    """




    # --- Data Prep ---
    #start timing of entire function, except plotting and output IO
    start_tottime = time()
    
    #Save initial seed
    np.random.seed(random_seed)
    seed = str(np.random.get_state()[1][0])
    
    # update config to save config at end
    config['random_seed'] = seed

    N = Ntrain + Npred + 1
    assert data.ndim == 3
    assert data.shape[0] >= N
    # Shape of video frames
    img_shape = data.shape[1:]
    print(f"Image shape is {img_shape}")




    # time all parts of ESN
    timings = {}
    start_init = time()
    
    


    # --- Data Pre-processing ---
    print("Starting pre-processing...")

    train_inputs = data[1:Ntrain]

    train_inputs, training_min, training_max = preprocess(
        data=train_inputs,
        training_min = None,
        training_max = None,
        **kwargs
        )
    
    config['training_min']=training_min
    config['training_max']=training_max

    del train_inputs
    gc.collect()
    
    data_ST = preprocess(data=data,**config)  

    print("Finished pre-processing!")




    # --- Starting training/ prediction/ error calculation ---
    print("Starting training/ prediction/ error calculation ... ")

    T_total = Npred 

    time_int_spat_error_li = []
    time_space_int_scalar_error_li = []


    for t in range(T_total - T_int + 1): 

        if t % 100 == 0:
            print(f"Time step: {t}")



        # --- Data Prep ---
        train_inputs = data_ST[t:t+Ntrain]
        train_targets = data_ST[t+Ntrans:Ntrain+t]
        pred_targets_ST = data_ST[Ntrain+t:Ntrain+t+T_int]


        

        # --- Building Dyn. Sys ---
        map_ih = InputMap(specs)
        hidden_size = map_ih.output_size(img_shape)
        
        esn = hidden.initialize_dynsys(map_ih, hidden_size,spectral_radius,neuron_connections,dtype=dtype, upper_sr_calc_dim=upper_sr_calc_dim)




        # --- Evolution of dyn. sys. ---   
        h_trans = hidden.evolve_hidden_state(esn, train_inputs[:Ntrans], h=np.zeros(hidden_size),mode='transient')




        # --- Harvest dyn. sys ---
        H = hidden.evolve_hidden_state(esn, train_inputs[Ntrans:], h=h_trans,mode='train')
        y0, h0 = train_targets[-1], H[-1]



        
        # --- PCA dim. reduction ---
        train_targets = train_targets.reshape(train_targets.shape[0], -1)
        H, pca_object = hidden.dimension_reduce(H,pca_object=None, n_PCs = n_PCs)




        # --- Create Readout Matrix using lstsq --- 
        esn = optimize.create_readout_matrix(esn,
                                        H, train_targets,
                                        lstsq_method=lstsq_method,
                                        lstsq_thresh=lstsq_thresh,
                                        dtype=dtype)
        



        # --- Predict ---
        predictions_ST = readout.predict(esn, y0, h0, Npred=T_int, pca_object=pca_object) 




        # --- Calculate Prediction Errors ----
        pred_error = np.abs(predictions_ST - pred_targets_ST)
        # Sum over time
        time_int_spat_error_t = pred_error.sum(axis=0) # shape: (lon, lat)
        #Sum over space
        time_space_int_scalar_error_t = time_int_spat_error_t.sum() #scalar

        #append
        time_int_spat_error_li.append(time_int_spat_error_t)
        time_space_int_scalar_error_li.append(time_space_int_scalar_error_t)


    time_int_spat_error = np.array(time_int_spat_error_li)
    time_space_int_scalar_error = np.array(time_space_int_scalar_error_li)
    



    # --- window scoring ---
    print("Starting Window Scoring ...")

    anom_score, lw_mu, lw_std, sw_mu = sliding_score(
    time_space_int_scalar_error, small_window=small_window, large_window=large_window)


    end_tottime = time()
    print(f'Total time: {end_tottime-start_tottime:.1f}s')

    timings = [
                f'Total time (no plotting/saving): {end_tottime-start_tottime:.1f}s',
            ]
    config['timings_dictionary'] = timings 

    # --- Save & Plot (if appl.) ---
    print("Saving ...")

    arrays_to_save = dict(
        y0=y0,
        h0=h0,
        Whh=esn[1],
        bh=esn[2],
        Who=esn[3],
        time_int_spat_error = time_int_spat_error,
        time_space_int_scalar_error = time_space_int_scalar_error, 
        anom_score = anom_score,
        lw_mu = lw_mu, 
        lw_std = lw_std, 
        sw_mu = sw_mu,
        )
    
    folder = save_error(
        dict_of_arrays= arrays_to_save,
        param_dict=config,
        savedir = config['savedir'],
    )
    


    
    # --- Plot ---
    print("Plotting ...")

    if plot_prediction and folder is not None:

        
        fig_1 = time_space_int_scalar_error_plot(time_space_int_scalar_error, lon)
        fig_1.savefig(f"{folder}/time_space_int_scalar_error.png", dpi=300, bbox_inches="tight")

        fig_2 = time_space_int_scalar_error_lw_sw_plot(time_space_int_scalar_error, lw_mu, sw_mu, lw_std, lon, subplot_kw=None)
        fig_2.savefig(f"{folder}/time_space_int_scalar_error_lw_sw_plot.png", dpi=300, bbox_inches="tight")

        fig_3 = anom_score_plot(anom_score)
        fig_3.savefig(f"{folder}/anom_score.png", dpi=300, bbox_inches="tight")
          
    return 

print("Done!")

