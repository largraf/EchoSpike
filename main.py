import matplotlib.pyplot as plt
from utils import load_NMNIST, train, test
from model import CLAPP_SNN

device = 'cpu'
epochs = 1
batch_size = 1

# Load Data Set and show example
train_loader, test_loader = load_NMNIST(20, batch_size=batch_size)

# Training

SNN = CLAPP_SNN(2312, [512, 512, 512], 16, 20).to(device)
acc_hist, loss_hist = train(SNN, train_loader, epochs, device)
plt.plot(acc_hist)
plt.plot(loss_hist)
plt.show()
correct = test(SNN, test_loader, device)
print(correct)
print('Accuracy:', sum(correct)/10000)
