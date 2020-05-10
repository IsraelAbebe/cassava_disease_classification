import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

#         self.model = nn.Linear(n_features, n_classes)
        
        self.fc_1 = nn.Linear(n_features, 700)
        self.fc_2 = nn.Linear(700, 200)
        self.fc_out = nn.Linear(200, n_classes)
        self.dropout = nn.Dropout(p=0.2)

#     def forward(self, x):
#         x = self.dropout(self.fc_1(F.relu(x)))
#         x = self.dropout(self.fc_2(F.relu(x)))
#         x = self.fc_out(F.relu(x))
        
#         return x

    def forward(self, x):
        x = self.dropout(self.fc_1(F.relu(x)))
        x = self.dropout(self.fc_2(F.relu(x)))
        x = self.fc_out(F.relu(x))
        
        return x
#         return self.model(x)