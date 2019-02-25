from scipy import misc
import torch
from torchvision import transforms

from model import StyleTransferModel
from telegram_token import token
import numpy as np
from PIL import Image
from io import BytesIO
from multiprocessing import Queue, Process
from time import sleep
