# Level Control of Steam Generator from Nuclear Power Plant

This project aims to develop a new control system for a Steam Generator using modern and advanced control techniques as well as sophisticated machine learning algorithms. Evaluating the model dynamics of a Steam Generator is highly complex and the results are often 
not satisfactory. Therefore, we applied machine learning to generate the dynamic equations based on historical data.

## Dynamic Model From Data Using pysindy (Advance Machine learning algorithm based in SVD)

1. Data Collection: Data collected from monitoring system in NP in Excel Historic format. defining states, inputs and disturbances in the system.
2. Data Preprocessing: Cleaning and transforming the data to make it suitable for modeling. This may include handling missing values, normalizing data, and feature engineering.
3. Data Splitting: Dividing the data into training and testing sets to evaluate the model's performance.
4. Create e Trainning the pysindy: Training the machine learning model using the training data to extract the SS dynamics equations from SG.
5. Model Evaluation: Assessing the model's performance using the testing data, using score for Metrics of accuracy.
6. Model Tuning: Fine-tuning the model's hyperparameters to improve its performance (actual phase of project).

## Apply Model Predictive Control in a Model (MPC)

Model Predictive Control (MPC) is an advanced method of process control that uses a dynamic model of the process to predict and optimize the control actions over a finite time horizon.

1. Model Definition: Define the mathematical model of the system to be controlled. This model predicts the future behavior of the system based on current and past states and inputs.
2. Prediction Horizon: Set a prediction horizon N, which is the time window over which future predictions are made.
3. Cost Function: Define a cost function that quantifies the performance of the control actions. This function typically includes terms for tracking error, control effort, and possibly other criteria.
4. Constraints: Specify constraints on inputs, outputs, and states to ensure safe and feasible operation of the system.
5. Optimization Problem: At each time step, solve an optimization problem to find the control actions that minimize the cost function over the prediction horizon while satisfying the constraints.
6. Control Action Implementation: Implement the first control action from the optimized sequence.
7. Receding Horizon: Shift the prediction horizon forward by one time step and repeat the process at the next time step.

## MPC Diagram
![Block-diagram-of-the-MPC](https://github.com/user-attachments/assets/7b794443-6a9e-4750-87ba-04c84f9462ec)

source: https://www.researchgate.net/figure/Block-diagram-of-the-MPC_fig1_327136772
