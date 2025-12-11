# Abalone Size Prediction
DSCI 522 Section 2 Group 31 Repository

#### Abalone Abalone, Yummy Yummy in my Tummy!

Welcome to our Abalone Size Prediction Project! 

This is a data analysis project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

## Description
This project aims to predict the age of an abalone from its physical features and sex. The model used in analysis is k-Nearest Neighbhours (k-NN) Regressor. The resulting model estimates the age of new abalone by identifying the k nearest abalones in the training set. 

Our final model results in a Test RMSE = 2.2884 in comparison to the Train RMSE = 1.8626 (with k = 5). 

## Data Source
The dataset in this project was obtained from the UCI Machine Learning Repository by Warwick Nash et al. and can be found at the link below.

https://archive.ics.uci.edu/dataset/1/abalone 

## Instructions on Running the Analysis and Rendering the Quarto Document

#### Prerequisites for running our analysis
1. Please install Docker Desktop and have it Docker Running on the computer. This is required for the Docker image to successfully build and run on your local machine. 

2. Please close all other running Jupyter Lab Terminals on your computer. Or else, it will trigger a Token Conflict Error.

#### Step 1: Clone our repo
First the repository must be cloned to your local computer using `git clone https://github.com/will-chh/522_grp31_Abalone_Age_Prediction.git` 

#### Step 2: Get Docker Started 
Initialize the Docker image by navigating to the project root with `cd 522_grp31_Abalone_Age_Prediction` in the terminal and run the following command: `docker compose up`

Terminal should now start pulling the docker image, and display the link to jupyter lab as below: 
![JupyterLab startup screenshot](img/docker1.png)

#### Step 3: Loading Docker Container: 
Open the JupyterLab URL displayed in the terminal http://127.0.0.1:8888/lab or simply type localhost:8888 in your browser.


#### Step 4: Run the analysis with the following commands
Open the Terminal on the jupyter lab on your brower, which is also the Command Line Interface, and run the following commands.

1. Cleaning up all the output files if there are outdated outputs, otherwise, when you run the script, all outputs will automatically get overwritten as well.

You can clear all outputs by running `make clean` in the terminal CLI.

2. Running the Analysis and rendering the reports: 

You have two options here: 
#### Automated Make Commands (make analysis and make report)
Running make commands `make analysis` will return the following 7 artifacts:
1. Cleaned dataset
data/processed/cleaned_abalone.csv

2. EDA scatter matrix plot
results/eda_scatter_matrix.png

3. Training dataset
data/processed/train.csv

4. Testing dataset
data/processed/test.csv

5. Trained kNN model
results/knn_model.pkl

6. Fitted scaler
results/knn_scaler.pkl

7. Model evaluation plot (Actual vs Predicted)
results/knn_eval_plot.png

Running `make report` will render the quarto html file that can be found in the reports folder. 

#### Alternatively, each individual steps of commands are listed below
Run these commands one by one in the Command Line Interface

```bash
4.1: Running only data_import
python utils/01_data_import.py \
  --input_path https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data \
  --output_path data/processed/cleaned_abalone.csv

4.2: Running only data eda
python utils/02_data_eda.py \
  --input_path data/processed/cleaned_abalone.csv \
  --output_path results/eda_scatter_matrix.png

4.3: Running model preprocess
python utils/03_model_preprocess.py \
  --input_path data/processed/cleaned_abalone.csv \
  --train_output data/processed/train.csv \
  --test_output data/processed/test.csv

4.5: Running model eval
python utils/04_model_fit.py \
  --train_path data/processed/train.csv \
  --model_output results/knn_model.pkl \
  --scaler_output results/knn_scaler.pkl \
  --n_neighbors 5

4.5: Model Evaluation Step with plotting Actual vs Predicted Values
python utils/05_model_eval.py \
  --train_path data/processed/train.csv \
  --test_path data/processed/test.csv \
  --model_path results/knn_model.pkl \
  --scaler_path results/knn_scaler.pkl \
  --plot_output results/knn_eval_plot.png

4.6: Render the Quarto Report
quarto render report/abalone_report.qmd
```

#### Step 5: Shutdown the Container:
After the report is rendered, to stop the container, press `Ctrl + C` to stop running the container in the terminal.

Then fully shut down and remove the container with: `docker compose down`


#### Steps 6: Remove the image that was pulled locally
To remove the image that was pulled locally, note the image name and tag from docker-compose.yml and run the following command:

`docker rmi <image_name:tag>`

such as this latest tag as an example:

`docker rmi will-chh/abalone_age_prediction:latest`

## Contributors:
- Yuting Ji

- Gurveer Madurai

- Seungmyun Park

- William Chong

## License 
MIT License
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license

