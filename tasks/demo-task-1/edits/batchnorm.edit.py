"""BatchNorm baseline — replace Net class with a BatchNorm architecture.

Replaces lines 10-33 (the Net class) with a deeper architecture using
BatchNorm2d/BatchNorm1d and padding to preserve spatial dimensions.

Line numbers reference the post-pre_edit file (pre_edit does not touch lines 10-33).
"""

_FILE = "pytorch-examples/mnist/main.py"

_BATCHNORM_NET = """\
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 10,
        "end_line": 33,
        "content": _BATCHNORM_NET,
    },
]
