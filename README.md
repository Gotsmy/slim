# Sulfur limitation increases specific pDNA yield in *E. coli* fed-batch processes

Creation of models, simulation of processes, and analysis of data as discussed in our research paper available on <mark>TODO</mark>.

## Python Environement
You can recreate the python environment by installing the packages listed in ```requirements.yml```.
The ```snek``` package can be downloaded from <a href = https://github.com/Gotsmy/snek/>GitHub</a> (select release 0.03 for guaranteed compatibility).

## Scripts
Includes python scripts used for the dFBA simulations.

## Results
* Unfortunately, the synthetic raw data results are not included as they take too much storage space. However, you can re-generate them by running the scripts (change ```loc``` variable first). Expect approximately 5 GB files per script.
* ```experimental_data.csv``` contains the experimental results.

## Preprocessing
Preprocessing was done to reduce the file size of simulation results.
* ```prepro_batch.ipynb``` contains code for the batch simulations results preprocessing.
* ```prepro_fed_batch.ipynb``` contains code for the fed-batch simulations results preprocessing.
* Unfortunately, the preprocessed results are not included as they still take too much storage space. However, you can re-generate them by running the preprocessing notebooks. Expect approximately 300 MB files per preprocessing.

## Notebooks
Jupyter Notebooks with code that was used to create the figures is given in the directory ```/notebooks```.

* ```00_model_creation.ipynb``` includes the model creation.
* ```01_analysis_1.ipynb``` includes code for figures 1 and 2.
* ```02_analysis_2.ipynb``` includes code for figures 3 and S1. 
* ```03_analysis_3.ipynb``` includes code for figures 4, 8, S2, S3, and S4.
* ```04_analysis_4.ipynb``` includes code for figures 5, 6 and 7.

