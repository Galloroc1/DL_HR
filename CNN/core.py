import torch.nn as nn
import logging

LAYER_PARAMS_DICT = {

    'resnet': {18: [{"BasicBlock": [[64, 64], [64, 64]], 'down_sample': False},
                    {"BasicBlock": [[64, 128], [128, 128]], 'down_sample': True},
                    {"BasicBlock": [[128, 256], [256, 256]], 'down_sample': True},
                    {"BasicBlock": [[256, 512], [512, 512]], 'down_sample': True},
                    ],
               34: [{"BasicBlock": [[64, 64]] + [[64, 64]] * 2, 'down_sample': False},
                    {"BasicBlock": [[64, 128]] + [[128, 128]] * 3, 'down_sample': True},
                    {"BasicBlock": [[128, 256]] + [[256, 256]] * 5, 'down_sample': True},
                    {"BasicBlock": [[256, 512]] + [[512, 512]] * 2, 'down_sample': True},
                    ],
               50: [{"BasicBlock": [[64, 64, 256], [256, 64, 256], [256, 64, 256]], 'down_sample': True,
                     'down_sample_stride': 1},
                    {"BasicBlock": [[256, 128, 512]] + [[512, 128, 512]] * 3, 'down_sample': True,
                     'down_sample_stride': 2},
                    {"BasicBlock": [[512, 256, 1024]] + [[1024, 256, 1024]] * 5, 'down_sample': True,
                     'down_sample_stride': 2},
                    {"BasicBlock": [[1024, 512, 2048]] + [[2048, 512, 2048]] * 2, 'down_sample': True,
                     'down_sample_stride': 2},
                    ],
               101: [{"BasicBlock": [[64, 64, 256], [256, 64, 256], [256, 64, 256]], 'down_sample': True,
                      'down_sample_stride': 1},
                     {"BasicBlock": [[256, 128, 512]] + [[512, 128, 512]] * 3, 'down_sample': True,
                      'down_sample_stride': 2},
                     {"BasicBlock": [[512, 256, 1024]] + [[1024, 256, 1024]] * 22, 'down_sample': True,
                      'down_sample_stride': 2},
                     {"BasicBlock": [[1024, 512, 2048]] + [[2048, 512, 2048]] * 2, 'down_sample': True,
                      'down_sample_stride': 2},
                     ],
               152: [{"BasicBlock": [[64, 64, 256], [256, 64, 256], [256, 64, 256]], 'down_sample': True,
                      'down_sample_stride': 1},
                     {"BasicBlock": [[256, 128, 512]] + [[512, 128, 512]] * 7, 'down_sample': True,
                      'down_sample_stride': 2},
                     {"BasicBlock": [[512, 256, 1024]] + [[1024, 256, 1024]] * 36, 'down_sample': True,
                      'down_sample_stride': 2},
                     {"BasicBlock": [[1024, 512, 2048]] + [[2048, 512, 2048]] * 3, 'down_sample': True,
                      'down_sample_stride': 2},
                     ],
               },
    "vgg": {
        11: {"conv1": [64], "pool1": 'M', "conv2": [128], "pool2": "M", "conv3": [256, 256], "pool3": "M",
             "conv4": [512, 512], "pool4": "M", "conv5": [512, 512], "pool5": "M"},
        13: {"conv1": [64, 64], "pool1": 'M', "conv2": [128, 128], "pool2": "M", "conv3": [256, 256], "pool3": "M",
             "conv4": [512, 512], "pool4": "M", "conv5": [512, 512], "pool5": "M"},
        16: {"conv1": [64, 64], "pool1": 'M', "conv2": [128, 128], "pool2": "M", "conv3": [256, 256, 256], "pool3": "M",
             "conv4": [512, 512, 512], "pool4": "M", "conv5": [512, 512, 512], "pool5": "M"},
        19: {"conv1": [64, 64], "pool1": 'M', "conv2": [128, 128], "pool2": "M", "conv3": [256, 256, 256, 256],
             "pool3": "M", "conv4": [512, 512, 512, 512], "pool4": "M", "conv5": [512, 512, 512, 512],
             "pool5": "M"},
    }

}


class BaseCNN(nn.Module):

    def __init__(self, depth: int) -> None:
        super(BaseCNN, self).__init__()
        self.depth = depth
        self.input_channel = 3
        self.sequential = nn.Sequential()

    def forward(self, x):
        raise

    def build_model(self):
        raise

    def detail(self, is_print=False):
        for i in self.sequential:
            logging.log(1, i)
            print(i) if is_print else None
