from mmseg.models.backbones.swin import SwinTransformer
import torch
import torch.nn as nn
# 继承SwinTransformer类，并添加prompt机制
class PromptSwinTransformer(SwinTransformer):
    # 实际调用是这样的:
    # model = PromptSwinTransformer(
    #     4,    # num_prompts，位置参数
    #     96,   # prompt_dim，位置参数
    #     96,   # 第一个 *args 参数
    #     [2, 2, 6, 2],  # 第二个 *args 参数
    #     num_heads=[3, 6, 12, 24]  # kwargs 参数
    # )

    
    # 构造函数
    def __init__(self, 
                 num_prompts=4,  # 每层添加的prompt数量
                 prompt_dim=128,  # prompt的维度，需要和patch_embed的维度一致
                 *args, 
                 **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)

         # 为每个stage创建不同维度的prompt
         # 为每一层创建可学习的prompt tokens
        dims = [prompt_dim * (2**i) for i in range(len(self.stages))]  # [128, 256, 512, 1024]
        self.num_prompts = num_prompts
        
        # 使用nn.ParameterList而不是nn.ModuleList
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(num_prompts, dim))  # 不需要.cuda()
            for dim in dims
        ])
        
        # 初始化prompts
        for prompt in self.prompts:
            nn.init.xavier_uniform_(prompt)
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.drop_after_pos(x)
        
        for i, stage in enumerate(self.stages):
            B, L, C = x.shape
            
            # 使用对应维度的prompt
            prompts = self.prompts[i].unsqueeze(0).expand(B, -1, -1)  # [B, num_prompts, C]
            x_with_prompts = torch.cat([prompts, x], dim=1)  # [B, num_prompts+L, C]
            
            # 通过stage处理
            x_with_prompts = stage(x_with_prompts)
            
            # 移除prompt tokens
            x = x_with_prompts[:, self.num_prompts:, :]
            
            if i < len(self.stages) - 1:
                x = x.reshape(B, *stage.downsample.out_resolution, -1).permute(0, 3, 1, 2)
                
        return x