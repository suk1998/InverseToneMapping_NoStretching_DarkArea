import colour
import matplotlib.pyplot as plt


img = colour.read_image('./hdr_result/HDR_V1-0007_supremo.jpg')
lut = colour.io.luts.read_LUT_IridasCube('BT_2020_BT2020_LUT127_1.cube')
plt.title('Source Image')
plt.imshow(img)
plt.show()

# apply lut 
lut_arr = lut.apply(img)
plt.figure(figsize=(16,9))
plt.title('LUT_applied_Image')
plt.imshow(lut_arr)
plt.show()

LUT_reproduced_img = colour.write_image(lut_arr,'lut_applied_HDR_V1-0007_supremo.jpg', bit_depth='uint8')