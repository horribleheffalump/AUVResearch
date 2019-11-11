import os
from ControlledModel.Illustrations import *
from ControlledModel.AUV import *
import numpy as np

dir = 'D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\'

#showbearing(os.path.join(dir, 'fig_4_bearing_orig.pdf'), False)
#showsonarmodel(os.path.join(dir, 'fig_3_sensor_orig.pdf'), False)
#showoptimalcontrol(os.path.join(dir, 'fig_2_optcontrol_orig.pdf'), False)
#showmodel(os.path.join(dir, 'fig_1_model_orig.pdf'), False)

TT = 100.0
T = 300.0
delta = 1.0
N = int(T / delta)

mX0 = np.array([0.001,-0.0002,-10.0003]) 
v = 2.0 # 5.0
U = lambda t: np.array([np.pi / 100.0 * np.cos(1.0 * np.pi * t / TT), np.pi / 3.0 * np.cos(4.0 * np.pi * t / TT), v])
t_history = np.arange(0.0, T + delta, delta)
VNominal_history = np.array(list(map(lambda t: AUV.V(U(t)), t_history[:-1])))
deltaXNominal_history = np.vstack((np.zeros(mX0.shape), delta * VNominal_history))
XNominal_history = mX0 + np.cumsum(deltaXNominal_history, axis = 0) 

maxX = np.max(XNominal_history, axis = 0)
minX = np.min(XNominal_history, axis = 0)
Xb = np.array([[maxX[0] + 100, maxX[1] + 100, 0.0], [maxX[0] + 100, minX[1] - 100, 0.0], [minX[0] - 100, maxX[1] + 100, 0.0], [minX[0] - 100, minX[1] - 100, 0.0]])


#shownominal3d(os.path.join(dir, 'fig_5_sample_nominal_3d_orig.pdf'), XNominal_history, Xb, False)
#shownominal2d(os.path.join(dir, 'fig_6_sample_nominal_2d.pdf'), t_history, XNominal_history, False)


#ce_cmnf = np.loadtxt("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\control\\control_error_cmnf_0.txt")
#ce_pseudo = np.loadtxt("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\control\\control_error_pseudo_0.txt")
#ce_kalman = np.loadtxt("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\control\\control_error_kalman_0.txt")

#showsample(os.path.join(dir, 'fig_7_sample_byvirtue.pdf'), t_history,  XNominal_history, [ce_cmnf, ce_pseudo, ce_kalman], False)

#showsample3d(os.path.join(dir, 'fig_7_sample_byvirtue.pdf'), XNominal_history, [ce_cmnf, ce_pseudo, ce_kalman], Xb, True)

#m_ee_cmnf = np.loadtxt("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\estimate\\estimate_error_cmnf_mean.txt")
#m_ee_pseudo = np.loadtxt("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\estimate\\estimate_error_pseudo_mean.txt")
#m_ee_kalman = np.loadtxt("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\estimate\\estimate_error_kalman_mean.txt")

#s_ee_cmnf = np.loadtxt("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\estimate\\estimate_error_cmnf_std.txt")
#s_ee_pseudo = np.loadtxt("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\estimate\\estimate_error_pseudo_std.txt")
#s_ee_kalman = np.loadtxt("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\estimate\\estimate_error_kalman_std.txt")

#showstats(os.path.join(dir, 'fig_8_stats_byvirtue.pdf'), t_history, [m_ee_cmnf, m_ee_pseudo, m_ee_kalman], [s_ee_cmnf, s_ee_pseudo, s_ee_kalman], False)

path = "Z:\\Наука - Data\\2019 - Sensors - AUV\\data\\acoustic\\control\\"
ce_cmnf = np.loadtxt(path + "control_error_cmnf_0006.txt")
ce_kalman = np.loadtxt(path + "control_error_kalman_0006.txt")
path = "Z:\\Наука - Data\\2019 - Sensors - AUV\\data\\acoustic_new\\control\\"
ce_pseudo = np.loadtxt(path + "control_error_pseudo_0008.txt")
showsample(os.path.join(dir, "fig_9_1_sample_acoustic.pdf"), t_history,  XNominal_history, [ce_cmnf, ce_pseudo, ce_kalman], False)




#path = "Z:\\Наука - Data\\2019 - Sensors - AUV\\data\\byvirt_new\\estimate\\"
#path = "Z:\\Наука - Data\\2019 - Sensors - AUV\\data\\acoustic\\estimate\\"
#m_ee_cmnf = np.loadtxt(path + "estimate_error_cmnf_mean.txt")
#m_ee_pseudo = np.loadtxt(path + "estimate_error_pseudo_mean.txt")
#m_ee_kalman = np.loadtxt(path + "estimate_error_kalman_mean.txt")

#s_ee_cmnf = np.loadtxt(path + "estimate_error_cmnf_std.txt")
#s_ee_pseudo = np.loadtxt(path + "estimate_error_pseudo_std.txt")
#s_ee_kalman = np.loadtxt(path + "estimate_error_kalman_std.txt")

#showstats(os.path.join(dir, 'fig_8_stats_byvirtue.pdf'), t_history, [m_ee_cmnf, m_ee_pseudo, m_ee_kalman], [s_ee_cmnf, s_ee_pseudo, s_ee_kalman], False)
#showstats(os.path.join(dir, 'fig_10_stats_acoustic.pdf'), t_history, [m_ee_cmnf], [s_ee_cmnf], False)
