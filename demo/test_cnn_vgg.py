# todo add root path
from CNN.VGG import VGG
from torchvision.models.vgg import vgg16
r = VGG(depth=16)
r.detail(is_print=True)