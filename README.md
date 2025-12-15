# Gym-Workout-Fatigue-Predictor-using-PyTorch

Develop a regression model to predict post-workout fatigue based on multiple exercise-related features: duration, average heart rate, and workout intensity.

Dataset:

8 synthetic samples of workout sessions

Features:

Duration (minutes)

Average heart rate (BPM)

Intensity (1–10 scale)

Target: Fatigue score (0–100 scale)

Model:

PyTorch nn.Sequential linear regression model (nn.Linear(3, 1))


Training:

Loss function: Mean Squared Error (MSE)

Optimizer: Stochastic Gradient Descent (SGD)

Custom training loop for 500 epochs with gradient computation (loss.backward()) and weight updates (optimizer.step())

Inference:

Predict fatigue for unseen workouts using torch.no_grad()

Example input: [25, 110, 4] → predicted fatigue output




