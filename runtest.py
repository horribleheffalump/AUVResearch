from testenvironment import *


# Make data.
#T = 30.0
#delta = 0.05
#NBeams = 15
#PhiBounds = [9.0,11.0]
#ThetaBounds = [-3.0,3.0]
#X0 = [0.001,-0.0002,13.0003]
#V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])

T = 60.0
delta = 0.1
NBeams = 10
accuracy = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PhiBounds =     [[15.0,30.0],   [20.0, 35.0],  [10.0,25.0],  [10.0, 35.0], [15.0,30.0],   [20.0, 35.0],  [10.0,25.0],  [10.0, 35.0]]
ThetaBounds =   [[10.0+15.0,10.0-15.0],   [100.0+15.0,100.0-15.0],  [190.0+15.0,190.0-15.0],  [280.0+15.0,280.0-15.0], [55.0+15.0,55.0-15.0],   [145.0+15.0,145.0-15.0],  [235.0+15.0,235.0-15.0],  [325.0+15.0,325.0-15.0]]
X0 = [0.001,-0.0002,-10.0003]
V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])
#V = lambda t: np.array([0.3, -0.02*np.cos(0.05 * t), 0.0])


test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V)

#test.showradar()
test.plottrajectory('D:\\projects.git\\NavigationResearch\\results\\')
#test.plotseabedsequence('D:\\projects.git\\NavigationResearch\\results\\')