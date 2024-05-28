from pynq import Overlay, allocate
import numpy as np
from PIL import Image
from IPython.display import display
import time

ol = Overlay('xsajup.xsa')
conv_ip = ol.hls_top_function_0 

KERNEL_SIZE = 3
convKernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)


def hls_top_function(inputImg, resultImg):
    imgHeight, imgWidth, imgChannels = inputImg.shape


    input_buffer = allocate(shape=inputImg.shape, dtype=np.uint8)
    output_buffer = allocate(shape=inputImg.shape, dtype=np.uint8)


    np.copyto(input_buffer, inputImg)

    conv_ip.write(0x10, input_buffer.physical_address)
    conv_ip.write(0x14, output_buffer.physical_address)
    conv_ip.write(0x18, imgWidth)
    conv_ip.write(0x1C, imgHeight)
    conv_ip.write(0x20, imgChannels)

    conv_ip.write(0x00, 1)

    while (conv_ip.read(0x00) & 0x4) == 0:
        pass

    np.copyto(resultImg, output_buffer)

input_image = Image.open('hamburgertwee.jpg')
input_array = np.array(input_image)

start_time = time.time()
output_array = np.empty_like(input_array)
hls_top_function(input_array, output_array)
end_time = time.time()

print(f"Processing time: {end_time - start_time} seconds")

output_image = Image.fromarray(output_array)

output_image.save('output_image.png')
display(input_image)
display(output_image)
