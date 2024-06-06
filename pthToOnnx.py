import torch
from openpose.src import util
from openpose.src.model import bodypose_model
# 假设你的模型是MyModel，并且它有一个实例化的对象my_model
# 假设你的模型的输入是input_tensor

# 加载模型和权重
# model_dict = torch.load('model/body_pose_model.pth')
model = bodypose_model()
model_dict = util.transfer(model, torch.load('model/body_pose_model.pth'))
model.load_state_dict(model_dict)
# my_model = torch.nn.Module()
# my_model.load_state_dict(model_dict)
model.eval()
#checkpoint = torch.load('model.pth')  # 加载模型的pth文件
model.load_state_dict(model_dict)  # 加载权重

# 设置是否是测试模式
model.eval()

# 模拟输入
input_tensor = torch.randn(1, 3, 224, 224)  # 假设输入尺寸是NCHW，这里的数字需要根据你的实际模型输入调整

# 转换为ONNX格式
torch.onnx.export(
    model,
    input_tensor,
    "model.onnx",
    verbose=True,
    input_names=['input'],
    output_names=['output']
)