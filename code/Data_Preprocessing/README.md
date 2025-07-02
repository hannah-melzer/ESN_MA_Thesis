#### **Data Pre-Processing**

1) performs chunking for the very large O(TB) dataset to the region of interest [see here](chunk_all_files.sh)
2) performs regridding on a regular, structured grid and interpolation of the NaN regions (land) using laplace diffusion [see here](regridding.py) 
3) performs re-sampling to 5 daily data (so the number of days per year is an integer) and detrending via [see here](detrend.py)

An example is outlined [here](Prep_CESM_Data.ipynb)
