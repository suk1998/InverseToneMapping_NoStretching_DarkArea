Category: AI Upscaling(FHD to 4K) Color Enhancement
Technologies Used: HDR Inverse Tone Mapping, BT2020 Color Enhancement algorithm, AI Super Resolution:SCUNET
Date: July 2023

The characteristic of HDR Inverse Tone Mapping that excessively brightens the shadows can be both an advantage and a disadvantage depending on the content. 
I had a video in my 8K upscaling project featuring fireworks exploding at night. Applying the traditional tone mapping revealed clouds in the darker areas 
that were previously not visible, resulting in what seemed like noise in the output. I modified the tone mapping code to ensure that the night sky remained 
untouched by tone mapping, making it appear darker, while only applying tone mapping to brighten the fireworks. The resulting video is the one displayed above. 
I named the modified code 'No Stretching Dark Area'.

## Original FHD Image
![Original_FHD_Image](https://github.com/suk1998/InverseToneMapping_NoStretching_DarkArea/blob/main/640_360_03.mp4_20221230_163337.985_jpeg70.jpg)

## SCUNET UPscaling to 4K + HDR Inverse Tone Mapping + Color Enhancement Algorithm
![Scunet Upscaling to 4K_InverseToneMapping+ColorEnhancement](https://github.com/suk1998/InverseToneMapping_NoStretching_DarkArea/blob/main/640_360_03_x2_SCU_color.mov_20221230_163430.225_jpeg70.jpg)
