from torchvision.models.quantization import mobilenet_v3_large
import torchvision
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset,dataloader
# from torchvision.models.quantization import ResNet50_QuantizedWeights
import torchvision.models.quantization as models
import torch.optim as optim
import time
import copy
from tqdm import tqdm
from torch import nn
from PIL import Image
import glob
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# unquant_path = '/home/vp.shivasan/gait/src/saved_models/mobilenetv2_unquant.pt'
# quant_path = '/home/vp.shivasan/gait/src/saved_models/mobilenetv2_quant.pt'

unquant_path = '/home/vp.shivasan/gait/src/saved_models/mobilenetv2_unquant_5_3000F.pt'
quant_path = '/home/vp.shivasan/gait/src/saved_models/mobilenetv2_quant_5_3000F.pt'

def create_combined_model(model_fe):
  # Step 1. Isolate the feature extractor.
  model_fe_features = nn.Sequential(
    model_fe.quant,  # Quantize the input
    model_fe.features,
    model_fe.dequant,  # Dequantize the output
  )

  # Step 2. Create a new "head"
  new_head = nn.Sequential(
    nn.Dropout(p=0.35),
    nn.Linear(num_ftrs, 6),
  )

  # Step 3. Combine, and don't forget the quant stubs.
  new_model = nn.Sequential(
    model_fe_features,
    nn.AdaptiveAvgPool2d(output_size = (1, 1)),
    nn.Flatten(1),
    new_head,
  )
  return new_model


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

model = models.mobilenet_v2(pretrained=True, progress=True, quantize=False)
num_ftrs = model.classifier[1].in_features

# Step 1
model.train()
model.fuse_model()
# Step 2
model_ft = create_combined_model(model)
model_ft[0].qconfig = torch.quantization.default_qat_qconfig  # Use default QAT configuration
# Step 3
model_ft = torch.quantization.prepare_qat(model_ft, inplace=True)

for param in model_ft.parameters():
  param.requires_grad = False

model_ft.load_state_dict(torch.load(quant_path))
model_ft.to('cpu')  # We can fine-tune on GPU if available
model_ft.eval()

cls_to_idx = {'ASC_STAIRS': 0, 'DESC_STAIRS': 1, 'DOWN_RAMP': 2, 'FLAT_GROUND': 3, 'NO_DETECTION': 4, 'UP_RAMP': 5}

# image_path = '/home/vp.shivasan/gait/IITGN-data_FULL/ASC_STAIRS/color_219871020100.png'
c = 0
n = 0
for image_path in tqdm(glob.glob('/home/vp.shivasan/gait/IITGN-data_FULL/DESC_STAIRS/*.png')):

  image = Image.open(image_path)
  image = data_transforms['val'](image)
  image = image.unsqueeze(0)
  model_out = model_ft(image)
  _, preds = torch.max(model_out, 1)
  if preds.item() == 1:
    c +=1
  else:
    pause = True
  n+=1
print(c,n)
print(c/n)
dummy = 1
