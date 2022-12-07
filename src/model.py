import torch
import torch.nn as nn

# define the number of input and output features
input_size = 1
hidden_size = 32
output_size = 1

# define a simple feedforward neural network with one hidden layer
class SimpleNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)
    
  def forward(self, x):
    x = x.float()
    x = torch.sigmoid(self.fc1(x))
    x = torch.sigmoid(self.fc2(x))
    return x

# load the model from file if it exists otherwise start with a new model
try:
  model = SimpleNet(input_size, hidden_size=hidden_size, output_size=output_size)
  model.load_state_dict(torch.load('model.pth'))
  model.eval()
except:
  model = SimpleNet(input_size, hidden_size=hidden_size, output_size=output_size)



# define a dataset of even and odd numbers
#dataset = [(2, 1), (3, 0), (4, 1), (5, 0), (6, 1), (7, 0), (8, 1)]

# generate a random dataset of even and odd numbers
import random
dataset = []
for x in range(1, 11):
  y = 1 if x % 2 == 0 else 0
  dataset.append((x, y))

print("generating a dataset:")
print(dataset)

# define the loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1):
  for x, y in dataset:
    # make a prediction using the model
    # train with gpu if available

    if torch.cuda.is_available():
      model = model.cuda()
      y_pred = model(torch.tensor([[x]]).cuda())
    else:
      y_pred = model(torch.tensor([[x]]))
    
    # calculate the loss
    loss = loss_fn(y_pred, torch.tensor([[y]]).float())
    
    # backpropagate the loss and update the model parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# save the model
torch.save(model.state_dict(), 'model.pth')

# make some example predictions
print('2', model(torch.tensor([[2]])))
print('3', model(torch.tensor([[3]])))
print('4', model(torch.tensor([[4]])))
print('5', model(torch.tensor([[5]])))
print('6', model(torch.tensor([[6]])))
print('7', model(torch.tensor([[7]])))
print('40', model(torch.tensor([[40]])))
print('50', model(torch.tensor([[50]])))
print('51', model(torch.tensor([[51]])))
print('1001', model(torch.tensor([[1001]])))