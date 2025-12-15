import torch
import torch.nn as nn
import torch.optim as optim

#Duration Dataset:
#[30, 45, 60, 75, 90, 40, 55, 70]

#HeartRate Dataset:
#[115], [125], [135], [145], [155], [120], [130], [140]

#Intensity Dataset:
#[4], [5], [6], [7], [8], [3], [6], [7]

#Creating tensor for all factors (duration, heartrate, intensity) 
fatigue_factors = torch.tensor([[30, 115, 4], [45, 125, 5],  [60, 135, 6], [75, 145, 7], [90, 155, 8], [40, 120, 3], [55, 130, 6], [70, 140, 7]], dtype=torch.float32)


#Creating tensor for Fatigue Score (0 - 100)
fatigue_level = torch.tensor([[28], [42], [58], [72], [85], [32], [54], [68]], dtype=torch.float32)

#define the model
model = nn.Sequential(nn.Linear(3, 1))

#loss function & optimizer 
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001)

#each pass in training data
for epoch in range(500):
  optimizer.zero_grad()
  outputs = model(fatigue_factors)
  loss = loss_function(outputs, fatigue_level)
  loss.backward()
  optimizer.step()


with torch.no_grad():
  test_factors = torch.tensor([[25, 110, 4]], dtype=torch.float32)
  predicted_fatigue = model(test_factors)
  print(f'The predicted fatigue is {predicted_fatigue.item():.1f} on a scale of 1-100')
