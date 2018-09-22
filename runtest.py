from testenvironment import *
from SlopeApproximator import *

np.random.seed(2213123)


pc=28.0
pr=20.0
tr=20.0


T = 60.0
delta = 1.0
NBeams = 10
accuracy = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PhiBounds =     [[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr]]
ThetaBounds =   [[10.0+tr,10.0-tr],   [100.0+tr,100.0-tr],  [190.0+tr,190.0-tr],  [280.0+tr,280.0-tr], [55.0+tr,55.0-tr],   [145.0+tr,145.0-tr],  [235.0+tr,235.0-tr],  [325.0+tr,325.0-tr]]

## for X
#accuracy = [0.1]
#PhiBounds =     [[pc-pr,pc+pr]] 8 + i * 4
#ThetaBounds =   [[100.0+tr,100.0-tr]] 80 + k * 4

## for Y
#accuracy = [0.1]
#PhiBounds =     [[pc-pr,pc+pr]]
#ThetaBounds =   [[280.0+tr,280.0-tr]] 260 + k * 4

## for Z
#accuracy = [0.1]
#PhiBounds =     [[pc-pr,pc+pr]]
#ThetaBounds =   [[190.0+tr,190.0-tr]] 170 + k * 4


X0 = [0.001,-0.0002,-10.0003]
#V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])
V = lambda t: np.array([0.5, 0.5 * np.cos(0.2 * t), 0.02 * np.sin(0.5 * t)])
#X0 = [10.0, 10.0, -10.0]
#V = lambda t: np.array([0.3, 0.3, -0.02 * np.cos(0.1 * t)])
estimateslope = False 
seabed = Seabed()


test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope)
#test.showscheme()


#for i in range(0,10):
#   test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope)
#   test.stats([20, 40, 80, 159], [1, 3, 2], 'D:\\projects.git\\NavigationResearch\\results\\')

#for i in range(0,5000):
#    print('');
#    print('trajectory ' + str(i));
#    print('');
#    #np.random.seed(2213123-i*100)
#    estimateslope = False 
#    seabed = Seabed()
#    test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope)
#    test.speedstats([6,6,6], 'D:\\projects.git\\NavigationResearch\\results\\slope_known\\')

#    estimateslope = True 
#    seabed = Seabed()
#    test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope)
#    test.speedstats([6,6,6], 'D:\\projects.git\\NavigationResearch\\results\\slope_unknown\\')


test.showradar([6])
#test.plot3Dtrajectory([6,6,6], 'D:\\projects.git\\NavigationResearch\\results\\')
#test.plotspeed([6,6,6], 'D:\\projects.git\\NavigationResearch\\results\\')
#test.plottrajectory3Dseabed()
#test.plottrajectory('D:\\projects.git\\NavigationResearch\\results\\')
#test.plotseabedsequence('D:\\projects.git\\NavigationResearch\\results\\', 'both')
#test.crit()

