import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc0 = nn.Linear(28*28, 27)
        # [canal input (1 color), canal output, taille fenetre]
        self.fc0 = nn.Conv2d(1, 2, 3)
        self.fc1 = nn.Conv2d(2, 2, 3)
        self.fc2 = nn.Linear(2*11*11, 27)

    def forward(self, x):
        #pooling
        x = F.max_pool2d(self.fc0(x), (2,2))
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x.view(x.size(0), 2*11*11)))

        return F.log_softmax(x)
