
from Seabed import *
from AUVControlled import *
from CMNFFilter import *
from testenvironmentControlled import *
from math import *
#np.random.seed(2213123)

seabed = Seabed()
estimateslope = True

pc=28.0
pr=20.0
tr=20.0

accuracy = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PhiBounds =     [[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr]]
ThetaBounds =   [[10.0+tr,10.0-tr],   [100.0+tr,100.0-tr],  [190.0+tr,190.0-tr],  [280.0+tr,280.0-tr], [55.0+tr,55.0-tr],   [145.0+tr,145.0-tr],  [235.0+tr,235.0-tr],  [325.0+tr,325.0-tr]]
NBeams = 10

TT = 360.0
T = 360.0
delta = 1.0
N = int(T / delta)

X0 = np.array([5,5,-10])
DW = np.array([0.5,0.5,0.5])
U = lambda t: np.pi * np.array([1 / 100.0 * np.cos(1.0 * np.pi * t / TT), 1 / 3.0 * np.cos(4.0 * np.pi * t / TT)])
#U = lambda t: np.array([np.pi/2 - np.pi  / 100.0 * np.cos(1.0 * np.pi * t / TT), np.pi / 3.0 * np.cos(4.0 * np.pi * t / TT)])
v = 2.0 # 5.0
t_history = np.arange(0.0, T + delta, delta)
VNominal_history = np.array(list(map(lambda t: AUVControlled.V(v, U(t)), t_history[:-1])))
deltaXNominal_history = np.vstack((np.zeros(X0.shape), delta * VNominal_history))
XNominal_history = X0 + np.cumsum(deltaXNominal_history, axis = 0)

auv = AUVControlled(T, delta, X0, DW, U, v)
test = testenvironmentControlled(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, auv, seabed, estimateslope)


test.showbearing('D:\\Наука\\_Статьи\\__в работе\\navigation\\fig_4_bearing_orig.pdf', False)
test.showsonarmodel('D:\\Наука\\_Статьи\\__в работе\\navigation\\fig_3_sensor_orig.pdf', False)
test.showoptimalcontrol('D:\\Наука\\_Статьи\\__в работе\\navigation\\fig_2_optcontrol_orig.pdf', False)
test.showmodel('D:\\Наука\\_Статьи\\__в работе\\navigation\\fig_1_model_orig.pdf', False)