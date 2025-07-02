from esn_dev.utils import score_over_time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cm 

def animate_comparison(targets, predictions, error, lon, lat, unit, filepath='comparison.mp4', fps=24, dpi=300, mima=(None, None), mima_err=(None, None)):
    targets = targets.copy()
    predictions = predictions.copy()
    
    # Determine color limits for targets/predictions
    if mima[0] is None:
        v = np.nanmax(np.abs(np.concatenate((targets, predictions))))
        vmin, vmax = -v, v
    else:
        vmin, vmax = mima

    # Determine color limits for error
    if mima_err[0] is None:
        vmin_err = np.nanmin(error)
        vmax_err = np.nanmax(error)
    else:
        vmin_err, vmax_err = mima_err

    print("targets shape:", targets.shape, "min/max:", np.nanmin(targets), np.nanmax(targets))
    print("predictions shape:", predictions.shape, "min/max:", np.nanmin(predictions), np.nanmax(predictions))
    print("error shape:", error.shape, "min/max:", np.nanmin(error), np.nanmax(error))
    
    print("vmin, vmax:", vmin, vmax)
    print("vmin_err, vmax_err:", vmin_err, vmax_err)

    # Initialize figure
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3,
        figsize=(12, 5),
        dpi=dpi,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Initial pcolormesh plots
    im1 = ax1.pcolormesh(lon, lat, targets[0, :, :], cmap=cm.balance, vmin=vmin, vmax=vmax, shading='auto')
    im2 = ax2.pcolormesh(lon, lat, predictions[0, :, :], cmap=cm.balance, vmin=vmin, vmax=vmax, shading='auto')
    im3 = ax3.pcolormesh(lon, lat, error[0, :, :], cmap=cm.thermal, vmin=vmin_err, vmax=vmax_err, shading='auto')

    for idx, ax in enumerate([ax1, ax2, ax3]):
        ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
        ax.coastlines()
        ax.set_extent([lon.min() + 1, lon.max() - 1, lat.min() + 1, lat.max() - 1], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        if idx > 0:  # ax2 and ax3
            gl.left_labels = False

    # Titles
    ax1.set_title('Target')
    ax2.set_title('Prediction')
    ax3.set_title('Prediction Error')


    # Adjust layout to make space below for colorbars
    fig.subplots_adjust(
        left=0.05, 
        right=0.95, 
        top=0.92, 
        bottom=0.2,   # increase bottom space for colorbars
        wspace=0.15
    )

    # Colorbars below plots
    # Shared colorbar for targets and predictions
    cbar_ax1 = fig.add_axes([0.15, 0.07, 0.3, 0.03])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label(unit)

    # Colorbar for error
    cbar_ax2 = fig.add_axes([0.6, 0.07, 0.3, 0.03])
    cbar2 = fig.colorbar(im3, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Error')


    # Define the init function
    def init():
        fig.suptitle(f'Time = 0', y=0.99, fontsize='x-large')
        return im1, im2, im3

    # Define the animate function
    def animate(i):
        fig.suptitle(f'Time = {i:03d}', y=0.99, fontsize='x-large')
        
        # Directly set the 2D array slice without flattening
        im1.set_array(targets[i, :, :])
        im2.set_array(predictions[i, :, :])
        
        # Make sure you access the correct slice for time_int_spat_error
        if i < len(error):
            im3.set_array(error[i, :, :])
        else:
            im3.set_array(error[-1, :, :])  # Hold last frame
        return im1, im2, im3


    # Create the animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=targets.shape[0], interval=1000 / fps, blit=False
    )

    # Save the animation
    anim.save(filepath, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)




def MSE_over_time(targets, predictions, subplot_kw=None):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        subplot_kw=subplot_kw,
        figsize=(5.5, 2.74)
    )
    
    ax.plot(
        np.arange(1, len(targets) + 1),
        score_over_time(predictions, targets),
        color='black',
        linestyle='-'
    )
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('MSE')
    ax.grid(True)

    return fig



def time_space_int_scalar_error_plot(time_space_int_scalar_error, lon, Ntrain, subplot_kw=None):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        subplot_kw=subplot_kw,
        figsize=(5.5, 2.74)
    )
    
    # Normalized error
    normalized_error = time_space_int_scalar_error / lon.size
    ax.plot(normalized_error)

    # Compute all possible labels
    time_steps = np.arange(len(normalized_error))
    full_labels = (Ntrain + time_steps) * 5

    # Let Matplotlib decide the ticks, then relabel them
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Error')
    ax.grid(True)

    fig.canvas.draw()  # Needed to populate the tick locations
    ticks = ax.get_xticks().astype(int)

    # Filter valid ticks (some may be outside data range)
    valid_ticks = ticks[(ticks >= 0) & (ticks < len(full_labels))]
    ax.set_xticks(valid_ticks)
    ax.set_xticklabels(full_labels[valid_ticks])

    return fig


def time_space_int_scalar_error_lw_sw_plot(time_space_int_scalar_error, lw_mu, sw_mu, lw_std, lon, Ntrain, subplot_kw=None):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        subplot_kw=subplot_kw,
        figsize=(5.5, 2.74)
    )
    
    # Normalize by number of longitudes
    ts_error = time_space_int_scalar_error / lon.size
    lw_mu = lw_mu / lon.size
    sw_mu = sw_mu / lon.size
    lw_std = lw_std / lon.size

    # Plot the lines
    ax.plot(ts_error, label='Time Space Int Scalar Error', linestyle='-', linewidth=1.5)
    ax.plot(lw_mu, label='lw_mu', linestyle='-', linewidth=1.5)
    ax.plot(sw_mu, label='sw_mu', linestyle='-', linewidth=1.5)
    ax.plot(lw_std, label='lw_std', linestyle='-', linewidth=1.5)

    # Axis labels
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Error')
    ax.grid(True)
    ax.legend()

    # Create transformed x-axis labels
    time_steps = np.arange(len(ts_error))
    full_labels = (Ntrain + time_steps) * 5

    # Let matplotlib pick the ticks and relabel them
    fig.canvas.draw()  # Needed so that ax.get_xticks() works correctly
    ticks = ax.get_xticks().astype(int)
    valid_ticks = ticks[(ticks >= 0) & (ticks < len(full_labels))]

    ax.set_xticks(valid_ticks)
    ax.set_xticklabels(full_labels[valid_ticks])

    return fig



def anom_score_plot(anom_score, Ntrain, subplot_kw=None):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        subplot_kw=subplot_kw,
        figsize=(5.5, 2.74)
    )

    # Plot anomaly score
    ax.plot(anom_score, color='r')

    # Set semilog scale for the y-axis
    ax.set_yscale('log')

    # Add dashed threshold line
    ax.axhline(y=1e-5, color='k', linestyle='--', label='Threshold')

    # Axis labels and grid
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Anomaly Score')
    ax.grid(True)
    ax.legend()

    # Generate transformed x-axis labels
    time_steps = np.arange(len(anom_score))
    full_labels = (Ntrain + time_steps) * 5

    # Let matplotlib decide tick locations and override labels
    fig.canvas.draw()
    ticks = ax.get_xticks().astype(int)
    valid_ticks = ticks[(ticks >= 0) & (ticks < len(full_labels))]
    ax.set_xticks(valid_ticks)
    ax.set_xticklabels(full_labels[valid_ticks])

    return fig
