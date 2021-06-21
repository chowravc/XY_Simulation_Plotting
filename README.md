# XY_Simulation_Plotting
Extra package to make plots of outputs from XY_Simulation.

Install packages in python 3.7.10 as an anaconda environment with:

```
conda create --name XYenv --file requirements.txt
conda activate XYenv
```

Input images, global director angles dat file, and defect location dat files to `/input/<your-directory-name>`. Update whatever values are required in the main.py including extensions, `<your-directory-name>` and IMPORTANT: number of expected plus and minus defects. There is overlay information stored in an inkscape svg file which includes oblique incidence and polariser/analyser. This will be overlaid as a mask on the output file, make any changes wanted.

When done, run:

```
python main.py
```
