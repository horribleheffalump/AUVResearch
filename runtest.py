from testenvironment import *
from SlopeApproximator import *

np.random.seed(2213123)

pc=38.0
pr=16.0
tr=18.0

T = 10.0
delta = 0.1
NBeams = 16
accuracy = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PhiBounds =     [[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr]]
ThetaBounds =   [[10.0+tr,10.0-tr],   [100.0+tr,100.0-tr],  [190.0+tr,190.0-tr],  [280.0+tr,280.0-tr], [55.0+tr,55.0-tr],   [145.0+tr,145.0-tr],  [235.0+tr,235.0-tr],  [325.0+tr,325.0-tr]]
X0 = [0.001,-0.0002,-10.0003]
V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])
estimateslope = True
seabed = Seabed()



test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope)

test.showradar()
#test.plottrajectory('D:\\projects.git\\NavigationResearch\\results\\')
test.plotseabedsequence('D:\\projects.git\\NavigationResearch\\results\\', 'both')
#test.crit()

