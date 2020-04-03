import os
from _Tracking.Illustrations import *
from _Tracking.TrackingModel import *
import numpy as np

dir = 'D:\\Наука\\_Статьи\\__в работе\\2020 - Sensors - Tracking\\'

showplane(os.path.join(dir, "model.pdf"), False)
showpaths(os.path.join(dir, "paths2d.pdf"), os.path.join(dir, "paths3d.pdf"), False)
showsdynamicstd_xyz(os.path.join(dir, "dynamic_xyz.pdf"), False)
showsdynamicstd_Vxyz(os.path.join(dir, "dynamic_Vxyz.pdf"), False)
showsdynamicstd_va(os.path.join(dir, "dynamic_va.pdf"), False)
showsdynamicstd_R(os.path.join(dir, "dynamic_R.pdf"), False)
showsdynamicstd_angles(os.path.join(dir, "dynamic_angles.pdf"), False)


