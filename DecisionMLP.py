import torch.nn as nn


class DecisionMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DecisionMLP, self).__init__()
        # 主分类器分支
        self.mlp1 = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(input_size * 4, num_classes),
            nn.GELU(),
        )

        # 二次分类器分支
        self.mlp2 = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(input_size * 4, num_classes),
            nn.GELU(),
        )

        # 决策层
        self.decision = nn.Sequential(
            nn.Linear(num_classes, num_classes * 4),
            nn.ReLU(),
            nn.Linear(num_classes * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 主分类器路径
        main_output = self.mlp1(x)

        # 计算是否激活二次分类器
        decision_output = self.decision(main_output).squeeze()
        use_secondary = decision_output > 0.5

        if use_secondary.any():
            # 二次分类器路径
            secondary_output = self.mlp2(x)
            main_output[use_secondary] = secondary_output[use_secondary]

        return main_output
