import torch
import torch.nn as nn

class STNLayer(nn.Module):
    def __init__(self, channel):
        super(STNLayer, self).__init__()
	# Start
	self.conv_1x1 = nn.Conv2d(channel, channel/32, kernel_size=1, stride=1, bias=False)
	
	if channel == 256:
	    kernel1 = 7
	    kernel2 = 7 # 5
	elif channel == 512:
	    kernel1 = 5
	    kernel2 = 3
	else: # 1024
	    kernel1 = 3
	    kernel2 = 1

	# Encoder
        self.en_conv1 = nn.Conv2d(channel/32, channel/8, kernel_size=kernel1)
        self.en_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.en_act1 = nn.ReLU(True)
	self.en_conv2 = nn.Conv2d(channel/8, channel/4, kernel_size=kernel2)

	# Decoder
	self.de_conv1 = nn.ConvTranspose2d(channel/4, channel/8, kernel_size=kernel2)
	self.de_act1 = nn.ReLU(True)
        self.de_pool1 = nn.MaxUnpool2d(kernel_size=2, stride=2) 
	self.de_conv2 = nn.ConvTranspose2d(channel/8, channel/32, kernel_size=kernel1)
        
	# Final
	self.conv_1x1_2 = nn.Conv2d(channel/32, 1, kernel_size=1, stride=1)


    def forward(self, x):
	x_input = x
	x = self.conv_1x1(x)

	x1 = self.en_conv1(x)
	x, indices1 = self.en_pool1(x1)
	x = self.en_act1(x)
	x = self.en_conv2(x)
	
	x = self.de_conv1(x)
	x = self.de_act1(x)
        x = self.de_pool1(x, indices1, output_size=x1.size())
	x = self.de_conv2(x)

	x = self.conv_1x1_2(x)

	out = x_input * x

        return out

