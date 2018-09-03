import concurrent.futures
from testenvironment import *
from SlopeApproximator import *


phi_c = np.arange(30.0, 60.0, 0.5)
phi_r = np.arange(7.0, 15.0, 0.2)
th_r = np.arange(7.0, 16.0, 0.2)

iterable = np.fromfunction(lambda i, j, k: np.array([phi_c[i], phi_r[j], th_r[k]]), (phi_c.size, phi_r.size, th_r.size), dtype=int)
iterable = np.transpose(np.reshape(iterable, (3, phi_c.size * phi_r.size * th_r.size)))


X0 = [0.001,-0.0002,-10.0003]
V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])
estimateslope = False

T = 10.0
delta = 0.1
NBeams = 10
accuracy = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

path = 'results_crit_conc.txt'

N = 10

def exec(e):
    print(e)
    pc, pr, tr = e[0], e[1], e[2]
    start = datetime.datetime.now()
    c = np.zeros(N)
    for i in range(0,N):
        PhiBounds =     [[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr]]
        ThetaBounds =   [[10.0+tr,10.0-tr],   [100.0+tr,100.0-tr],  [190.0+tr,190.0-tr],  [280.0+tr,280.0-tr], [55.0+tr,55.0-tr],   [145.0+tr,145.0-tr],  [235.0+tr,235.0-tr],  [325.0+tr,325.0-tr]]
        seabed = Seabed()
        test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope)
        c[i] = test.crit()
    res = np.average(c)
    finish = datetime.datetime.now()
    with open(path, "a") as myfile:
        myfile.write(
            finish.strftime("%Y-%m-%d %H-%M-%S") + " " + 
            str(pc)  + " " + 
            str(pr)  + " " + 
            str(tr)  + " " + 
            str(res)  + " " + 
            "elapsed seconds: " + str((finish-start).total_seconds()) + " " +
            "\n"
            )
    return res


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
            for e, res in zip(iterable, executor.map(exec, iterable)):
                print(e, res)

if __name__ == '__main__':
    main()


