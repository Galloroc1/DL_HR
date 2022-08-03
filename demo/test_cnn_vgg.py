# todo add root path
from CNN.VGG import VGG
from torchvision.models.vgg import vgg16

r = VGG(depth=16)
# r.load_state_dict_(drop_key_lens=3)

print(r)
