import os
from ControlledModel.Illustrations import *

dir = 'D:\\Наука\\_Статьи\\__в работе\\navigation\\'

showbearing(os.path.join(dir, 'fig_4_bearing_orig.pdf'), False)
showsonarmodel(os.path.join(dir, 'fig_3_sensor_orig.pdf'), False)
showoptimalcontrol(os.path.join(dir, 'fig_2_optcontrol_orig.pdf'), False)
showmodel(os.path.join(dir, 'fig_1_model_orig.pdf'), False)