# 3rdYR_EPR_Project
 3rd Year Double-Electron-Electron Resonance Project.

The code used for this project was given to us by our supervisor and was used to write his previous paper on a similar topic. My project partner and I heavily modified it for use in analysing the data which we scraped from many different scientific papers, however our supervisor's code is still scattered throughout and required for the scripts to run. The cubic spline script, however, was entirely coded by myself. Due to large amounts of the code not being entirely our own, it was difficult to write the code in the most efficient way.

EPR_trace_cubicspline.py:
This code was used to take the raw data from the software we used to scrape the scientific graphs and format it such that the x-axes contained evenly-spaced points, which was crucial for later analysis.

EPR_pc_analysis.py:
A modified version of our supervisor's code. Used to compare numbers of principle component curves that make up the trace data to work out how many are required to accurately reproduce the full data.

EPR_mpp_weight_calculator.py:
A modified version of our supervisor's code. Runs the Tikhonov regularisation, plots the full kernel against the trace, and plots the final distance distribution as well as calculating the final distance value.

orisel.py:
Our supervisor's functions library.
