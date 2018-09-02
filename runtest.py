from testenvironment import *
from SlopeApproximator import *



#T = 10.0
#delta = 0.1
#NBeams = 10
#accuracy = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#PhiBounds =     [[15.0,30.0],   [20.0, 35.0],  [10.0,25.0],  [10.0, 35.0], [15.0,30.0],   [20.0, 35.0],  [10.0,25.0],  [10.0, 35.0]]
#ThetaBounds =   [[10.0+15.0,10.0-15.0],   [100.0+15.0,100.0-15.0],  [190.0+15.0,190.0-15.0],  [280.0+15.0,280.0-15.0], [55.0+15.0,55.0-15.0],   [145.0+15.0,145.0-15.0],  [235.0+15.0,235.0-15.0],  [325.0+15.0,325.0-15.0]]
##PhiBounds =     [[5.0,6.0],   [10.0, 11.0],  [10.0,11.0],  [10.0, 11.0], [15.0,16.0],   [10.0, 11.0],  [8.0,9.0],  [10.0, 11.0]]
##ThetaBounds =   [[10.0+2.0,10.0-2.0],   [100.0+2.0,100.0-2.0],  [190.0+2.0,190.0-2.0],  [280.0+2.0,280.0-2.0], [55.0+2.0,55.0-2.0],   [145.0+2.0,145.0-2.0],  [235.0+2.0,235.0-2.0],  [325.0+2.0,325.0-2.0]]
#X0 = [0.001,-0.0002,-10.0003]
#V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])
#estimateslope = False
#seabed = Seabed()



#test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope)

#test.showradar()
#test.plottrajectory('D:\\projects.git\\NavigationResearch\\results\\')
#test.plotseabedsequence('C:\\projects.git\\NavigationResearch\\results\\', 'none')
#test.crit()


phi_c = np.arange(11.0, 40.0, 2.0)
phi_r = np.arange(1.0, 10.0, 1.0)
th_r = np.arange(4.0, 16.0, 1.0)

print(phi_c.size * phi_r.size * th_r.size)
X0 = [0.001,-0.0002,-10.0003]
V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])
estimateslope = False

T = 10.0
delta = 0.1
NBeams = 10
accuracy = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

path = 'D:\\projects.git\\NavigationResearch\\results\\results_crit_11-25.txt'

N = 10

for pc in phi_c:
    for pr in phi_r:
        for tr in th_r:
            start = datetime.datetime.now()
            c = np.zeros(N)
            for i in range(0,N):
                PhiBounds =     [[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr]]
                ThetaBounds =   [[10.0+tr,10.0-tr],   [100.0+tr,100.0-tr],  [190.0+tr,190.0-tr],  [280.0+tr,280.0-tr], [55.0+tr,55.0-tr],   [145.0+tr,145.0-tr],  [235.0+tr,235.0-tr],  [325.0+tr,325.0-tr]]
                seabed = Seabed()
                test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope)
                c[i] = test.crit()
            finish = datetime.datetime.now()
            print(pc, pr, tr, c)
            with open(path, "a") as myfile:
                myfile.write(
                    finish.strftime("%Y-%m-%d %H-%M-%S") + " " + 
                    str(pc)  + " " + 
                    str(pr)  + " " + 
                    str(tr)  + " " + 
                    str(np.average(c))  + " " + 
                    "elapsed seconds: " + str((finish-start).total_seconds()) + " " +
                    "\n"
                    )
