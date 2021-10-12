import numpy as np
from matplotlib import pyplot as plt
import scipy

def convertToNumpy(inputData,twoOutputs = False):
    numpyArray = np.zeros((len(inputData[:]),len(inputData[0])))
    tupCount = 0
    for tup in inputData[:]:
        eleCount = 0
        for ele in tup:
            numpyArray[tupCount][eleCount] = ele
            eleCount += 1
        tupCount += 1
    if twoOutputs == True:
        return(numpyArray[:,0],numpyArray[:,1])
    else:
        return(numpyArray)


def genCollocationPts(x,y,x_tilde_pts):
    slopeList = []
    interceptList = []
    for i in range(1, len(x)):
        slope = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
        intercept = y[i] - (slope * x[i])
        slopeList.append(slope)
        interceptList.append(intercept)

    y_tilde_pts = []
    for i in range(len(x_tilde_pts)):
        result = np.where(x <= x_tilde_pts[i])
        x_tildeLoc = result[-1][-1]
        if x_tildeLoc >= len(slopeList):
            x_tildeLoc = len(slopeList) - 1
        y_tilde = x_tilde_pts[i] * slopeList[x_tildeLoc] + interceptList[x_tildeLoc]
        y_tilde_pts.append(y_tilde)
    return(np.asarray(y_tilde_pts))

def genCollocationPts_Step(x,y,x_colloc_pts):
    xpts = [x[0]]
    ypts = [y[0]]
    for i in range(1, len(x)):
        xloc = (x[i] - x[i - 1]) / 2 + x[i - 1]
        xpts.append(xloc)
        xpts.append(xloc)
        ypts.append(y[i - 1])
        ypts.append(y[i])

        xpts.append(x[i])
        ypts.append(y[i])
    x = np.asarray(xpts)
    y = np.asarray(ypts)
    y_colloc_pts = []
    for i in range(len(x_colloc_pts)):
        result = np.where(x <= x_colloc_pts[i])
        x_collocLoc = result[-1][-1]
        y_colloc_pts.append(y[x_collocLoc])
    return(np.asarray(y_colloc_pts))

def genStepCollocPts(x,y,x_colloc_pts):
    xpts = [x[0]]
    ypts = [y[0]]
    for i in range(1, len(x)):
        xloc = (x[i] - x[i - 1]) / 2 + x[i - 1]
        xpts.append(xloc)
        
    x = np.asarray(xpts)
    y = np.asarray(ypts)
    y_colloc_pts = []
    for i in range(len(x_colloc_pts)):
        result = np.where(x < x_colloc_pts[i])
        print(result)
        x_collocLoc = result[-1][-1]
        y_colloc_pts.append(y[x_collocLoc])
    return(np.asarray(y_colloc_pts))

    


def build_fourier_linear_system(mu, n_pairs, time_vec, no_mass_cc_vec):
    a_mtrx = np.zeros((len(time_vec), 2 * n_pairs + 1))
    b_vec = np.copy(no_mass_cc_vec)

    for j in range(len(time_vec)):
        for k in range(n_pairs):
            a_mtrx[:, 0] = 1
            a_mtrx[j, (k * 2) + 1] = np.cos((k + 1) * mu * time_vec[j])
            a_mtrx[j, (k + 1) * 2] = np.sin((k + 1) * mu * time_vec[j])

    return (a_mtrx, b_vec)

def genStepFunction(x, y,offset=0.0):
    xpts = [x[0]]
    ypts = [y[0]]
    for i in range(1, len(x)):
        xloc = (x[i] - x[i - 1]) / 2 + x[i - 1]
        xpts.append(xloc)
        xpts.append(xloc+offset)
        ypts.append(y[i - 1])
        ypts.append(y[i])

        xpts.append(x[i])
        ypts.append(y[i])
    return (np.asarray(xpts), np.asarray(ypts))

def gen_A_mtrx(kappa, m, n, colloc_x_vec):
    A_mtrx = np.zeros((m,n))
    A_mtrx[:,0] = 1
    for colNum in range(1,int((n-1)/2)+1):
        A_mtrx[:,2*colNum- 1] = np.cos(colloc_x_vec[:] * kappa * (colNum))
        A_mtrx[:,2*colNum] = np.sin(colloc_x_vec * kappa * (colNum))
    return(A_mtrx)

def fourierSolution(x,kappa,n,c_vec):
    solMtrx = gen_A_mtrx(kappa,m=1,n=n,colloc_x_vec=np.array([x]))
    solValue = solMtrx @ c_vec
    return(solValue[0])

def wavSolution(x,NList,bunch_pts,shift,sigmas,kappa,c_vec):
    solMtrx = genWavMtrx_Updated(NList,bunch_pts,shift,sigmas,kappa,x_colloc_pts=np.array([x]))
    solValue = solMtrx @ c_vec
    return(solValue[0])

def wavSolutionSquared(x,NList,bunch_pts,shift,sigmas,kappa,c_vec):
    result = wavSolution(x,NList,bunch_pts,shift,sigmas,kappa,c_vec)**2
    return(result)

def gen_wav_residual(x,NList,bunch_pts,shift,sigmas,kappa,c_vec,xList,yList):
    f = genCollocationPts(xList,yList,np.array([x]))[0]
    g = wavSolution(x,NList,bunch_pts,shift,sigmas,kappa,c_vec)
    return(f-g)

def gen_f_pt_wav_squared(x,xList,yList):
    result = genCollocationPts(xList,yList,np.array([x]))[0]
    return(result**2)

def fourierSolutionSquared(x,kappa,n,c_vec):
    solMtrx = gen_A_mtrx(kappa,m=1,n=n,colloc_x_vec=np.array([x]))
    solValue = solMtrx @ c_vec
    return(solValue[0]**2)

def genStepChangeList(x,y):
    xpts = [x[0]]
    ypts = [y[0]]
    for i in range(1, len(x)):
        xloc = (x[i] - x[i - 1]) / 2 + x[i - 1]
        xpts.append(xloc)
        xpts.append(xloc)
        ypts.append(y[i - 1])
        ypts.append(y[i])

        xpts.append(x[i])
        ypts.append(y[i])
    x = np.asarray(xpts)
    y = np.asarray(ypts)
    return(x,y)

def gen_f_pt_squared(x,xStepChangeList,yStepChangeList):
    result = np.where(x < xStepChangeList)
    return((yStepChangeList[result[-1][0]]**2))

def genWavMtrx(N,bunch_pts,shift,sigma,kappa,x_colloc_pts):
    #Old function which sums the bounding gaussians for each packet
    #Can only accept 1 sigma, # modes for all packets
    P = len(bunch_pts)
    m = len(x_colloc_pts)
    wav_mtrx = np.zeros((m,int(2*N*P)+1))
    k=0
    x = x_colloc_pts
    for p in bunch_pts:
        wav_mtrx[:,0] += (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-((x-p)**2)/(2*(sigma**2)))

    for p_loc in range(1,len(bunch_pts)+1):
        p = bunch_pts[p_loc-1]
        for k in range(1,N+1):
            loc = (p_loc-1)*N*2 + 2*(k-1) + 1
            wav_mtrx[:,loc] = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.cos(k*kappa*x + (shift*x**2)/2))
            wav_mtrx[:,loc+1] = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.sin(k*kappa*x + (shift*x**2)/2))
    wav_mtrx = wav_mtrx * np.sqrt(2*np.pi) * sigma
    return(wav_mtrx)

def genEquallySpacedBunchPts(x_min,x_max,n):
    #Generate equally spaced means for wavelet packets
    wavelength = x_max-x_min
    mean_dist = wavelength/(n+1)
    bunch_pts = []
    for point_num in range(1,n+1):
        bunch_pt = x_min + point_num * mean_dist
        bunch_pts.append(bunch_pt)
    return(bunch_pts)

def genWavMtrx_Updated(NList,bunch_pts,shift,sigmas,kappa,x_colloc_pts):
    #Initialize wavelet matrix
    P = len(bunch_pts)
    m = len(x_colloc_pts)
    num_eqns = int(sum(NList)*2+len(NList))
    wav_mtrx = np.zeros((m,num_eqns))
    k=0
    x = x_colloc_pts
    for N_index, N in enumerate(NList):
        #Iterates through each packet

        #Location of the column where the packet starts
        baseColNum = int(sum(NList[:N_index]))*2+N_index

        #Initialize sigma and mean for gaussian
        sigma = sigmas[N_index]
        p = bunch_pts[N_index]

        #Create bounding gaussian for the target packet
        wav_mtrx[:,baseColNum] = np.exp(-((x-p)**2)/(2*(sigma**2)))
        for modeNum in range(N):
            #Iterates through the rest of the packet

            #loc is col num where mode begins
            loc = baseColNum+(modeNum)*2 + 1

            k = modeNum+1

            #Fill sin and cos funcs
            wav_mtrx[:,loc] = np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.cos(k*kappa*x + (shift*x**2)/2))
            wav_mtrx[:,loc+1] = np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.sin(k*kappa*x + (shift*x**2)/2))

    return(wav_mtrx)

def plotWavelet(NList,bunch_pts,shift,sigmas,kappa,x_min,x_max,x,y,num_plotting_pts=1000):
    N = len(NList)
    P = len(bunch_pts)

    #num_plotting_pts = 1000
    x_colloc_pts = np.linspace(x_min, x_max, num_plotting_pts)
    y_colloc_pts = genCollocationPts(x, y, x_colloc_pts)
    wav_mtrx = genWavMtrx_Updated(NList, bunch_pts, shift, sigmas, kappa, x_colloc_pts)

    fig = plt.figure(figsize=(18, 5))
    ax = plt.subplot(111)

    plt.grid()
    plt.title(f"Wavelet Basis Function # Modes = {NList}, Packets = {bunch_pts}")
    for N_index, N in enumerate(NList):
        baseColNum = int(sum(NList[:N_index])) * 2 + N_index
        plt.plot(x_colloc_pts, wav_mtrx[:, baseColNum], "k-", label="exp(.)")
        for modeNum in range(N):
            # Iterates through an individual packet
            loc = baseColNum + (modeNum) * 2 + 1

            k = modeNum + 1

            plt.plot(x_colloc_pts, wav_mtrx[:, loc], label=r"exp(.)cos(%i $\kappa x + \phi$ )" % k)
            plt.plot(x_colloc_pts, wav_mtrx[:, loc + 1], label=r"exp(.)sin(%i $\kappa x + \phi$ )" % k)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()
    plt.show()


def genExpanded_g_vec(NList, bunch_pts, shift, sigmas, kappa,c_vec,x_min,x_max,num_pts=10000):
    x_colloc_pts = np.linspace(x_min, x_max, num_pts)

    wav_mtrx = genWavMtrx_Updated(NList, bunch_pts, shift, sigmas, kappa, x_colloc_pts)
    g_vec = wav_mtrx @ c_vec
    return(g_vec)

def genLoadMtrx(NList,bunch_pts,shift,sigmas,kappa,x_min,x_max,xList,yList):
    #Initialize load matrix
    P = len(bunch_pts)

    num_eqns = int(sum(NList)*2+len(NList))
    loadMtrx = np.zeros(num_eqns)

    k=0
    for N_index, N in enumerate(NList):
        #Iterates through each packet

        #Location of the column where the packet starts
        baseColNum = int(sum(NList[:N_index]))*2+N_index

        #Initialize sigma and mean for gaussian
        sigma = sigmas[N_index]
        p = bunch_pts[N_index]

        #Create bounding gaussian for the target packet
        targetFunc = lambda x,p,sigma : np.exp(-((x-p)**2)/(2*(sigma**2))) * genCollocationPts(xList,yList,np.asarray([x]))[0]
        loadMtrx[baseColNum] = scipy.integrate.quad(targetFunc,x_min,x_max,args=(p,sigma),limit=200)[0]
        #print(gramMtrx)
        #(g_integral,g_error) = scipy.integrate.quad(wavSolutionSquared,x_min,x_max,args=(NList,bunch_pts,shift,sigmas,kappa,c_vec),limit=300)

        for modeNum in range(N):
            #Iterates through the rest of the packet

            #loc is col num where mode begins
            loc = baseColNum+(modeNum)*2 + 1

            k = modeNum+1
            #Fill Cosine Func
            targetFunc = lambda x,p,sigma,k,kappa,shift : np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.cos(k*kappa*x + (shift*x**2)/2)) * genCollocationPts(xList,yList,np.asarray([x]))[0]
            
            loadMtrx[loc] = scipy.integrate.quad(targetFunc,x_min,x_max,args=(p,sigma,k,kappa,shift),limit=200)[0]
            
            #Fill Sine Func
            targetFunc = lambda x,p,sigma,k,kappa,shift : np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.sin(k*kappa*x + (shift*x**2)/2)) * genCollocationPts(xList,yList,np.asarray([x]))[0]
            
            loadMtrx[loc+1] = scipy.integrate.quad(targetFunc,x_min,x_max,args=(p,sigma,k,kappa,shift),limit=200)[0]

    return(loadMtrx)

def genGramMtrx(NList,bunch_pts,shift,sigmas,kappa,x_min,x_max,xList,yList):
    #Initialize Gram matrix
    
    #Creates vector with all the basis functions [phi_0, phi_1, ..., phi_n]
    phiVec = wavBasisFunctionVec(bunch_pts,NList,shift,sigmas,kappa,x_min,x_max,xList,yList)

    #Initilize parameters relating to mode and packet number
    P = len(bunch_pts)
    num_eqns = int(sum(NList)*2+len(NList))
    
    #Generate nxn Gram matrix
    gramMtrx = np.zeros((num_eqns,num_eqns))

    for N_index, N in enumerate(NList):
        #Iterates through each packet

        #Location of the column where the packet starts
        baseColNum = int(sum(NList[:N_index]))*2+N_index

        #Initialize sigma and mean for gaussian
        sigma = sigmas[N_index]
        p = bunch_pts[N_index]

        #Create bounding gaussian for the target packet
        for distToDiag in range(baseColNum+1):
            #print("baseColNum",baseColNum,"distToDiag",distToDiag)
            #print(phiVec.func[0](8))
            targetFunc = lambda x : np.exp(-((x-p)**2)/(2*(sigma**2))) * phiVec.func[distToDiag](x)
            gramMtrx[distToDiag,baseColNum] = scipy.integrate.quad(targetFunc,x_min,x_max,limit=100)[0]
        #print(gramMtrx)
        #(g_integral,g_error) = scipy.integrate.quad(wavSolutionSquared,x_min,x_max,args=(NList,bunch_pts,shift,sigmas,kappa,c_vec),limit=300)

        for modeNum in range(N):
            #Iterates through the rest of the packet

            #loc is col num where mode begins
            loc = baseColNum+(modeNum)*2 + 1

            k = modeNum+1
            #Fill Cosine Func
            #targetFunc = lambda x,p,sigma,k,kappa,shift : (np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.cos(k*kappa*x + (shift*x**2)/2)))**2
            
            for distToDiag in range(loc+1):
                #print("baseColNum",baseColNum,"distToDiag",distToDiag)
                #print(phiVec.func[0](8))
                
                targetFunc = lambda x : (np.exp(-((x-p)**2)/(2*(sigma**2)))*np.cos(k*kappa*x + (shift*x**2)/2)) * phiVec.func[distToDiag](x)
                gramMtrx[distToDiag,loc] = scipy.integrate.quad(targetFunc,x_min,x_max,limit=100)[0]
            
            #gramMtrx[loc,loc] = scipy.integrate.quad(targetFunc,x_min,x_max,args=(p,sigma,k,kappa,shift),limit=100)[0]
            
            #Fill Sine Func
            #targetFunc = lambda x,p,sigma,k,kappa,shift : (np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.sin(k*kappa*x + (shift*x**2)/2)))**2
            
            for distToDiag in range(loc+2):
                #print("baseColNum",baseColNum,"distToDiag",distToDiag)
                #print(phiVec.func[0](8))
                targetFunc = lambda x : np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.sin(k*kappa*x + (shift*x**2)/2)) * phiVec.func[distToDiag](x)
                gramMtrx[distToDiag,loc+1] = scipy.integrate.quad(targetFunc,x_min,x_max,limit=100)[0]
            #gramMtrx[loc+1,loc+1] = scipy.integrate.quad(targetFunc,x_min,x_max,args=(p,sigma,k,kappa,shift),limit=100)[0]
        for rowNum in range(num_eqns):
            for i in range(rowNum):
                gramMtrx[rowNum,i] = gramMtrx[i,rowNum]
                #gramMtrx[rowNum,i] = 1
    return(gramMtrx)

class wavBasisFunctionVec:
    """
    Creates a vector of basis functions that can be called as a list with the .func method
    
    """
    
    def __init__(self,bunch_pts,NList,shift,sigmas,kappa,x_min,x_max,xList,yList):
        self.num_eqns = int(sum(NList)*2+len(NList))
        self.func = []
        self.initBasisFuncs(bunch_pts,NList,shift,sigmas,kappa,x_min,x_max,xList,yList)
        
    def initBasisFuncs(self,bunch_pts,NList,shift,sigmas,kappa,x_min,x_max,xList,yList):

        for N_index, N in enumerate(NList):
            #Iterates through each packet

            #Location of the column where the packet starts
            baseColNum = int(sum(NList[:N_index]))*2+N_index

            #Initialize sigma and mean for gaussian
            sigma = sigmas[N_index]
            p = bunch_pts[N_index]
            
            #Fill bounding gaussian function
            def targetFunc(x,p=p,sigma=sigma):
                return(np.exp(-((x-p)**2)/(2*(sigma**2))))
            self.func.append(targetFunc)


            for modeNum in range(N):
                #Iterates through the rest of the packet

                #loc is col num where mode begins
                loc = baseColNum+(modeNum)*2 + 1

                #Calculates k value
                k = modeNum+1
                
                #Fill Cosine Func
                def targetFunc(x,p=p,sigma=sigma,k=k,kappa=kappa,shift=shift):
                    return(np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.cos(k*kappa*x + (shift*x**2)/2)))
                self.func.append(targetFunc)
            

                #Fill Sine Func
                def targetFunc(x,p=p,sigma=sigma,k=k,kappa=kappa,shift=shift):
                    return(np.exp(-((x-p)**2)/(2*(sigma**2)))*(np.sin(k*kappa*x + (shift*x**2)/2)))
                self.func.append(targetFunc)