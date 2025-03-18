## Adjusting the SWE Model

After tuning the model to a simple SWE model with symmetric disturbance, I shifted, the disturbance off centre (to the lower left corner). 

As already pointed out by Jacob, the blurring properties of sigma_xy have a major effect and can become overly sensitive to noise (numerical instabilities). 

At sigma_xy = 2, mse started to increase drastically as the prediction time went on (see figure below)

However, it was also Jacob who stressed the point that we should prefer an esn which captures the dynamic evolution over one that minimizies the mse. 

**Plots for run 23**

- optimal hyperparameter settings described [here](./Hyperparameter_Tuning_SWE.md)

  

![MSE](./MSE_plot_23.png)


[Animation](./comparasion_23.mp4)

