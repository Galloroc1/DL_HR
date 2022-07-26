# todo add root path
from CNN.VGG import VGG
r = VGG(depth=16)
r.build_model()
r.detail(is_print=True)