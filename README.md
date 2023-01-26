# NeRF
Implementing NeRF model from scratch

<table>
  <th colspan="3">
  <center>v1_coarse_fine</center>
  </th>
  <tr>
    <td><center>Chair</center></td>
    <td><center>Ship</center></td>
    <td><center>Hotdog</center></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/prajwalsingh/NeRF/blob/main/results/chair.gif" />
    </td>
    <td>
      <img src="https://github.com/prajwalsingh/NeRF/blob/main/results/ship.gif" />
    </td>
    <td>
      <img src="https://github.com/prajwalsingh/NeRF/blob/main/results/hotdog.gif" />
    </td>
  </tr>
</table>

<table>
  <th colspan="3">
  <center>v11</center>
  </th>
  <tr>
<!--     <td><center>Chair</center></td>
    <td><center>Ship</center></td> -->
    <td colspan="3"><center>Hotdog</center></td>
  </tr>
  <tr>
<!--     <td></td>
    <td></td> -->
    <td colspan="3">
      <img src="https://github.com/prajwalsingh/NeRF/blob/main/results/hotdogv11.gif" width="256px" height="256px"/>
    </td>
  </tr>
</table>


<table>
  <th colspan="3">
  <center>v16</center>
  </th>
  <tr>
    <td><center>Lego</center></td>
    <td><center>Hotdog</center></td>
    <td><center>Materials</center></td>
  </tr>
  <tr>
    <td><img src="https://github.com/prajwalsingh/NeRF/blob/main/results/legov16.gif" width="320px" height="320px"/></td>
    <td><img src="https://github.com/prajwalsingh/NeRF/blob/main/results/hotdogv16.gif" width="320px" height="320px"/></td>    
    <td><img src="https://github.com/prajwalsingh/NeRF/blob/main/results/materialsv16.gif" width="320px" height="320px"/></td>
  </tr>
</table>

<table>
  <th colspan="2">
  <center>Final Version (v18) (Special thanks to: [4])</center>
  </th>
  <tr>
    <td><img src="https://github.com/prajwalsingh/NeRF-Pytorch/blob/main/results/nerf_result_1.gif" width="800px" height="800px"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/prajwalsingh/NeRF-Pytorch/blob/main/results/nerf_result_2.gif" width="800px" height="800px"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/prajwalsingh/NeRF-Pytorch/blob/main/results/llff_1.gif" width="800px" height="800px"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/prajwalsingh/NeRF-Pytorch/blob/main/results/llff_2.gif" width="800px" height="800px"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/prajwalsingh/NeRF-Pytorch/blob/main/results/llff_3.gif" width="800px" height="800px"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/prajwalsingh/NeRF-Pytorch/blob/main/results/llff_self_4.gif" width="800px" height="800px"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/prajwalsingh/NeRF-Pytorch/blob/main/results/real_1.gif" width="800px" height="800px"/></td>
  </tr>
</table>

## Custom Dataset

1. Capture a video (forward facing), place it in Image2Pose_NERF/customdataset folder.
2. Open terminal in that folder and extract frames using [![Stackoverflow](https://stackoverflow.com/a/66524095/5224257)]:
   > ffmpeg -i input.mp4 '%04d.png'
4. Place the extracted image in extract folder.
5. Reduce the number of frames using filterimage.py code present in Image2Pose_NERF/customdataset folder.
6. Open COLMAP, click on new project -> select the customdataset for database and customdataset/images for image path -> save
7. Now in COLMAP select Reconstruction option -> Auto Reconstruction -> Select workspace as customdataset -> Select image folder as customdataset/images -> select data type as video frames -> check shared intrinsics -> check sparse -> uncheck dense and now click Run. It will create Sparse name folder in customdataset.
8. Now, in Image2Pose_NERF folder run image2pose.py, it will create poses_bounds.npy file [![LLFF](https://gitlab.scss.tcd.ie/alainm/LLFF/-/tree/master)]:
   > python image2pose.py customdataset/
9. customdataset folder is final dataset.


## References:

[1] Computer Graphics and Deep Learning with NeRF using TensorFlow and Keras [![Link](https://pyimagesearch.com/2021/11/17/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-2/)]

[2] 3D volumetric rendering with NeRF [![Link](https://keras.io/examples/vision/nerf/)]

[3] Nerf Official Colab Notebook [![Link](https://colab.research.google.com/drive/1L6QExI2lw5xhJ-MLlIwpbgf7rxW7fcz3#scrollTo=31sNNVves8C2)]

[4] NeRF PyTorch [![Link](https://github.com/sillsill777/NeRF-PyTorch)] ( Special Thanks :) )
