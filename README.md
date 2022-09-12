# public_icecube_data_analysis
To get the repository running, first download the 2008-2018 public icecube data analysis: https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/. Also, add a folder in the directory labeled "processed_data"

Then, put the files in the "events" folder of the data release into the "data/events" folder of the repository. Put the effective area csv files into the "data/tabulatedaeff" folder in the repository, and put the smearing files  into just the data folder. The other necessary data files should be in the repository.

For "A03_analyze_all_sky_map.ipynb", if you want to do a quick check, change the step size to 15, and you can run that file locally without it taking an absurd amount of time, just remember to set N-cpu = None when calling main. However, if you want to do a smaller step size, I recommend submitting "A03p1_submit_all_sky.sbatch" to run it. 

For "A05p1_analyze_source_classes_limits.ipynb", this should be run interactively, because it takes a crazy amount of time when run without parallel processing. I'd recommend 10 or 20 n_cpu, but it will still take a bit to run. 

For "MC03_neutrino_to_muon_energy", you have to run it twice -- once with the "background energies" files commented out and oncxe with the "signal energies" files commented it out. It should be labeled in the document
