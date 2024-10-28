from making_dataset import adapter_dataset
from model import *
from train import *

#adapter_dataset()

greg = GoogleNet(9)
print(greg)

train(greg)


