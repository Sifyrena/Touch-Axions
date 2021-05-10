NS = 4
central_mass = 0
length = 10

resol = 128
# If you own a supercomputer, can try 384 or 512

solitons = [[6,[1,1,0],[0,0,0],0],
            [6,[-1,-1,0],[0,0,0],0],
            [6,[1,-1,0],[0,0,0],1],
            [6,[-1,1,0],[0,0,0],1]]

Plot = 'Phi' # 'Rho'

# Mass, 3-Position, 3-Velocity, Phase (/ pi)

start_time = 0

particles = [] # Not Supported In Interactive Mode
embeds = [] # Not Supported In Interactive Mode

Uniform = False
Density = 0

time_factor = 10

a = 3000
B = 0
UVel = [0,0,0]

t0 = 0
    
Version   = str('PyUL2_Int') # Handle used in console.
D_version = str('Build 2021 May 05') # Detailed Version
S_version = 21.0 # Short Version

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import numexpr as ne
import numba
import pyfftw
import multiprocessing

from IPython.core.display import clear_output

    
num_threads = multiprocessing.cpu_count()

pi = np.pi

print("This is a demo version with almost all scientific capabilities removed.")
####################### Credits Information
def PyULCredits(IsoP = False,UseDispSponge = False,embeds = []):
    print(f"==============================================================================")
    print(f"{Version}.{S_version}: (c) 2020 - 2021 Wang., Y. and collaborators. \nAuckland Cosmology Group\n") 
    print("Original PyUltraLight Team:\nEdwards, F., Kendall, E., Hotchkiss, S. & Easther, R.\narxiv.org/abs/1807.04037")
    
    if IsoP or UseDispSponge or (embeds != []):
        print(f"\n**External Module In Use**")
    
    print(f"==============================================================================")

    
####################### PADDED POTENTIAL FUNCTIONS

PyULCredits()

### Toroid or something
        
def Wrap(TMx, TMy, TMz, lengthC):
    
    if TMx > lengthC/2:
        TMx = TMx - lengthC
        
    if TMx < -lengthC/2:
        TMx = TMx + lengthC
        
        
    if TMy > lengthC/2:
        TMy = TMy - lengthC
    
    if TMy < -lengthC/2:
        TMy = TMy + lengthC
        
        
    if TMz > lengthC/2:
        TMz = TMz - lengthC
    
    if TMz < -lengthC/2:
        TMz = TMz + lengthC
        
    return TMx,TMy,TMz

FWrap = Wrap

### For Immediate Interpolation of Field Energy

def QuickInterpolate(Field,lengthC,resol,position):
        #Code Position
                
        RNum = (position*1/lengthC+1/2)*resol

        RPt = np.floor(RNum)
        RRem = RNum - RPt
                
        RX = RRem[0]
        RY = RRem[1]
        RZ = RRem[2]
        
        Interp = 0
        
        # Need special treatment if any of these is zero or close to resol!
        RPtX = int(RPt[0])
        RPtY = int(RPt[1])
        RPtZ = int(RPt[2])

        if (RPtX >= resol-1) or (RPtY >= resol-1) or (RPtZ >= resol-1):
            #raise RuntimeError (f'Particle #{i} reached boundary on the +ve side. Halting.')
            return Interp
            
        if (RPtX <= 0) or (RPtY <= 0) or (RPtZ <= 0):
            #raise RuntimeError (f'Particle #{i} reached boundary on the +ve side. Halting.')
            return Interp

        else:
        
            SPC = Field[RPtX:RPtX+2,RPtY:RPtY+2,RPtZ:RPtZ+2]
            # This monstrosity is actually faster than tensor algebra...
            Interp += (1-RX)*(1-RY)*(1-RZ)*SPC[0,0,0] # Lower Left Near
            Interp += (1-RX)*(1-RY)*(  RZ)*SPC[0,0,1]
            Interp += (1-RX)*(  RY)*(1-RZ)*SPC[0,1,0]
            Interp += (1-RX)*(  RY)*(  RZ)*SPC[0,1,1]
            Interp += (  RX)*(1-RY)*(1-RZ)*SPC[1,0,0]
            Interp += (  RX)*(1-RY)*(  RZ)*SPC[1,0,1]
            Interp += (  RX)*(  RY)*(1-RZ)*SPC[1,1,0]
            Interp += (  RX)*(  RY)*(  RZ)*SPC[1,1,1] # Upper Right Far

            return Interp
            
### Method 3 Interpolation Algorithm

def InterpolateLocal(RRem,Input):
        
    while len(RRem) > 1:
                    
        Input = Input[1,:]*RRem[0] + Input[0,:]*(1-RRem[0])
        RRem = RRem[1:]
        InterpolateLocal(RRem,Input)
        
    else:
        return Input[1]*RRem + Input[0]*(1-RRem)

def FWNBody(t,TMState,masslist,phiSP,a,lengthC,resol):

    GridDist = lengthC/resol
    
    dTMdt = 0*TMState
    GradientLog = np.zeros(len(masslist)*3)
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
        IndD = int(3*i)
           
        #X,Y,Z
        poslocal = TMState[Ind:Ind+3]

        RNum = (poslocal*1/lengthC+1/2)*resol

        RPt = np.floor(RNum)
        RRem = RNum - RPt
        
        # Need special treatment if any of these is zero or close to resol!

        
        RPtX = int(RPt[0])
        RPtY = int(RPt[1])
        RPtZ = int(RPt[2])

        if (RPtX <= 0) or (RPtY <= 0) or (RPtZ <= 0):
            #raise RuntimeError (f'Particle #{i} reached boundary on the -ve side. Halting.')
            TAr = np.zeros([4,4,4])

            GradientX = 0
            GradientY = 0
            GradientZ = 0        

        elif (RPtX >= resol-4) or (RPtY >= resol-4) or (RPtZ >= resol-4):
            #raise RuntimeError (f'Particle #{i} reached boundary on the +ve side. Halting.')
            TAr = np.zeros([4,4,4])
            
            GradientX = 0
            GradientY = 0
            GradientZ = 0

        else:   
            
            TAr = phiSP[RPtX-1:RPtX+3,RPtY-1:RPtY+3,RPtZ-1:RPtZ+3] # 64 Local Grids

            GArX = (TAr[2:4,1:3,1:3] - TAr[0:2,1:3,1:3])/(2*GridDist) # 8

            GArY = (TAr[1:3,2:4,1:3] - TAr[1:3,0:2,1:3])/(2*GridDist) # 8

            GArZ = (TAr[1:3,1:3,2:4] - TAr[1:3,1:3,2:4])/(2*GridDist) # 8

            GradientX = InterpolateLocal(RRem,GArX)

            GradientY = InterpolateLocal(RRem,GArY)

            GradientZ = InterpolateLocal(RRem,GArZ)

        #XDOT
        dTMdt[Ind]   =  TMState[Ind+3]
        #YDOT
        dTMdt[Ind+1] =  TMState[Ind+4]
        #ZDOT
        dTMdt[Ind+2] =  TMState[Ind+5]
        
        #x,y,z
        
        GradientLocal = -1*np.array([[GradientX],[GradientY],[GradientZ]])

        #Initialized Against ULDM Field
        #XDDOT
        dTMdt[Ind+3] =  GradientLocal[0]
        #YDDOT
        dTMdt[Ind+4] =  GradientLocal[1]
        #ZDDOT
        dTMdt[Ind+5] =  GradientLocal[2]
    
        for ii in range(len(masslist)):
            
            if (ii != i) and (masslist[ii] != 0):
                
                IndX = int(6*ii)
                
                # print(ii)
                
                poslocalX = np.array([TMState[IndX],TMState[IndX+1],TMState[IndX+2]])
                
                rV = poslocalX - poslocal
                
                rVL = np.linalg.norm(rV) # Positive
                
                
                if a == 0:
                    F = 1/(rVL)**3
                else:                    
                    F = -(a**3)/(a**2*rVL**2+1)**(1.5) # The First Plummer
                
                # Differentiated within Note 000.0F
                
                #XDDOT with Gravity
                dTMdt[Ind+3] = dTMdt[Ind+3] - masslist[ii]*F*rV[0]
                #YDDOT
                dTMdt[Ind+4] = dTMdt[Ind+4] - masslist[ii]*F*rV[1]
                #ZDDOT
                dTMdt[Ind+5] = dTMdt[Ind+5] - masslist[ii]*F*rV[2]
        
        GradientLog[IndD  ] = GradientLocal[0]
        GradientLog[IndD+1] = GradientLocal[1]
        GradientLog[IndD+2] = GradientLocal[2]

    return dTMdt, GradientLog


def FWNBody_NI(t,TMState,masslist,phiSP,a,lengthC,resol):

    GridDist = lengthC/resol
    
    dTMdt = 0*TMState
    GradientLog = np.zeros(len(masslist)*3)
    
    for i in range(len(masslist)):
        
        #0,6,12,18,...
        Ind = int(6*i)
        IndD = int(3*i)
           
        #X,Y,Z
        poslocal = TMState[Ind:Ind+3]

        RNum = (poslocal*1/lengthC+1/2)*resol

        RPt = np.floor(RNum)
        RRem = RNum - RPt
        
        # Need special treatment if any of these is zero or close to resol!

        
        RPtX = int(RPt[0])
        RPtY = int(RPt[1])
        RPtZ = int(RPt[2])

        if (RPtX <= 0) or (RPtY <= 0) or (RPtZ <= 0):
            #raise RuntimeError (f'Particle #{i} reached boundary on the -ve side. Halting.')
            TAr = np.zeros([4,4,4])

            GradientX = 0
            GradientY = 0
            GradientZ = 0        

        elif (RPtX >= resol-4) or (RPtY >= resol-4) or (RPtZ >= resol-4):
            #raise RuntimeError (f'Particle #{i} reached boundary on the +ve side. Halting.')
            TAr = np.zeros([4,4,4])
            
            GradientX = 0
            GradientY = 0
            GradientZ = 0

        else:   
            
            TAr = phiSP[RPtX-1:RPtX+3,RPtY-1:RPtY+3,RPtZ-1:RPtZ+3] # 64 Local Grids

            GArX = (TAr[2:4,1:3,1:3] - TAr[0:2,1:3,1:3])/(2*GridDist) # 8

            GArY = (TAr[1:3,2:4,1:3] - TAr[1:3,0:2,1:3])/(2*GridDist) # 8

            GArZ = (TAr[1:3,1:3,2:4] - TAr[1:3,1:3,2:4])/(2*GridDist) # 8

            GradientX = InterpolateLocal(RRem,GArX)

            GradientY = InterpolateLocal(RRem,GArY)

            GradientZ = InterpolateLocal(RRem,GArZ)

        #XDOT
        dTMdt[Ind]   =  TMState[Ind+3]
        #YDOT
        dTMdt[Ind+1] =  TMState[Ind+4]
        #ZDOT
        dTMdt[Ind+2] =  TMState[Ind+5]
        
        #x,y,z
        
        GradientLocal = -1*np.array([[GradientX],[GradientY],[GradientZ]])

        #Initialized Against THE VOID
        #XDDOT
        dTMdt[Ind+3] =  0
        #YDDOT
        dTMdt[Ind+4] =  0
        #ZDDOT
        dTMdt[Ind+5] =  0
    
        for ii in range(len(masslist)):
            
            if (ii != i) and (masslist[ii] != 0):
                
                IndX = int(6*ii)
                
                # print(ii)
                
                poslocalX = np.array([TMState[IndX],TMState[IndX+1],TMState[IndX+2]])
                
                rV = poslocalX - poslocal
                
                rVL = np.linalg.norm(rV) # Positive

                if a == 0:
                    F = 1/(rVL)**3
                else:                    
                    F = -(a**3)/(a**2*rVL**2+1)**(1.5) # The First Plummer
                
                # Differentiated within Note 000.0F
                
                #XDDOT
                dTMdt[Ind+3] = dTMdt[Ind+3] - masslist[ii]*F*rV[0]
                #YDDOT
                dTMdt[Ind+4] = dTMdt[Ind+4] - masslist[ii]*F*rV[1]
                #ZDDOT
                dTMdt[Ind+5] = dTMdt[Ind+5] - masslist[ii]*F*rV[2]
        
        GradientLog[IndD  ] = GradientLocal[0]
        GradientLog[IndD+1] = GradientLocal[1]
        GradientLog[IndD+2] = GradientLocal[2]

    return dTMdt, GradientLog


FWNBody3 = FWNBody
FWNBody3_NI = FWNBody_NI

def NBodyAdvance(TMState,h,masslist,phiSP,a,lengthC,resol,NS):
        #
        if NS == 0: # NBody Dynamics Off
            
            Step, GradientLog = FWNBody3(0,TMState,masslist,phiSP,a,lengthC,resol)
            
            return TMState, GradientLog
        
        if NS == 1:
 
            Step, GradientLog = FWNBody3(0,TMState,masslist,phiSP,a,lengthC,resol)
            TMStateOut = TMState + Step*h
            
            return TMStateOut, GradientLog
        
        elif NS%4 == 0:
            
            NRK = int(NS/4)
            
            H = h/NRK
            
            for RKI in range(NRK):
                TMK1, Trash = FWNBody3(0,TMState,masslist,phiSP,a,lengthC,resol)
                TMK2, Trash = FWNBody3(0,TMState + H/2*TMK1,masslist,phiSP,a,lengthC,resol)
                TMK3, Trash = FWNBody3(0,TMState + H/2*TMK2,masslist,phiSP,a,lengthC,resol)
                TMK4, GradientLog = FWNBody3(0,TMState + H*TMK3,masslist,phiSP,a,lengthC,resol)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)
                
            TMStateOut = TMState

            return TMStateOut, GradientLog


def NBodyAdvance_NI(TMState,h,masslist,phiSP,a,lengthC,resol,NS):
        #
        if NS == 0: # NBody Dynamics Off
            
            Step, GradientLog = FWNBody3_NI(0,TMState,masslist,phiSP,a,lengthC,resol)
            
            return TMState, GradientLog
        
        if NS == 1:
 
            Step, GradientLog = FWNBody3_NI(0,TMState,masslist,phiSP,a,lengthC,resol)
            TMStateOut = TMState + Step*h
            
            return TMStateOut, GradientLog
        
        elif NS%4 == 0:
            
            NRK = int(NS/4)
            
            H = h/NRK
            
            for RKI in range(NRK):
                TMK1, Trash = FWNBody3_NI(0,TMState,masslist,phiSP,a,lengthC,resol)
                TMK2, Trash = FWNBody3_NI(0,TMState + H/2*TMK1,masslist,phiSP,a,lengthC,resol)
                TMK3, Trash = FWNBody3_NI(0,TMState + H/2*TMK2,masslist,phiSP,a,lengthC,resol)
                TMK4, GradientLog = FWNBody3_NI(0,TMState + H*TMK3,masslist,phiSP,a,lengthC,resol)
                TMState = TMState + H/6*(TMK1+2*TMK2+2*TMK3+TMK4)
            
            TMStateOut = TMState

            return TMStateOut, GradientLog



######################### Soliton Init Factory Setting!

def LoadDefaultSoliton():
    
    f = np.load('./Soliton Profile Files/initial_f.npy')
    
    return f

def initsoliton(funct, xarray, yarray, zarray, position, alpha, f, delta_x,Cutoff = 5.6):
    funct*= 0
    
    for index in np.ndindex(funct.shape):
        
        
        # Note also that this distfromcentre is here to calculate the distance of every gridpoint from the centre of the soliton, not to calculate the distance of the soliton from the centre of the grid
        distfromcentre = (
            (xarray[index[0], 0, 0] - position[0]) ** 2 +
            (yarray[0, index[1], 0] - position[1]) ** 2 +
            (zarray[0, 0, index[2]] - position[2]) ** 2
            ) ** 0.5
        # Utilises soliton profile array out to dimensionless radius 5.6.
        if (np.sqrt(alpha) * distfromcentre <= Cutoff):
         
            funct[index] = alpha * f[int(np.sqrt(alpha) * (distfromcentre / delta_x + 1))]

    return funct

    
##########################################################################################
# CREATE THE Just-In-Time Functions (work in progress)

initsoliton_jit = numba.jit(initsoliton)



IsoP = False


# In[54]:


num_threads = multiprocessing.cpu_count()


if a>=1e8:
    a = 0


NumSol = len(solitons)
NumTM = len(particles)

##########################################################################################
#CONVERT INITIAL CONDITIONS TO CODE UNITS

lengthC = length

b = lengthC * B/2

cmass = 0

Vcell = (lengthC / float(resol)) ** 3

ne.set_num_threads(num_threads)

##########################################################################################
# SET UP THE REAL SPACE COORDINATES OF THE GRID - Version 1!

gridvec = np.linspace(-lengthC / 2.0 + lengthC/(resol*2.0), lengthC / 2.0 - lengthC/(resol*2.0), resol, endpoint=False)

xarray, yarray, zarray = np.meshgrid(
    gridvec, gridvec, gridvec,
    sparse=True, indexing='ij')

WN = 2*np.pi*np.fft.fftfreq(resol, lengthC/(resol)) # 2pi Pre-multiplied

Kx,Ky,Kz = np.meshgrid(WN,WN,WN,sparse=True, indexing='ij',)

##########################################################################################
# SET UP K-SPACE COORDINATES FOR COMPLEX DFT

kvec = 2 * np.pi * np.fft.fftfreq(resol, lengthC / float(resol))

kxarray, kyarray, kzarray = np.meshgrid(
    kvec, kvec, kvec,
    sparse=True, indexing='ij',
)

karray2 = ne.evaluate("kxarray**2+kyarray**2+kzarray**2")
##########################################################################################
delta_x = 0.00001 # Needs to match resolution of soliton profile array file. Default = 0.00001

warn = 0 
funct = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')   

# INITIALISE SOLITONS WITH SPECIFIED MASS, POSITION, VELOCITY, PHASE

psi = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')

MassCom = Density*lengthC**3

UVelocity = np.array(UVel)

DensityCom = MassCom / resol**3


psi = ne.evaluate("0*psi + sqrt(Density)")

velx = UVelocity[0]
vely = UVelocity[1]
velz = UVelocity[2]
psi = ne.evaluate("exp(1j*(velx*xarray + vely*yarray + velz*zarray))*psi")
#psi = ne.evaluate("psi + funct")


if solitons != []:
    f = LoadDefaultSoliton()

for s in solitons:
    mass = s[0]
    position = np.array(s[1])
    velocity = np.array(s[2])
    # Note that alpha and beta parameters are computed when the initial_f.npy soliton profile file is generated.
    alpha = (mass / 3.883) ** 2
    beta = 2.454
    phase = s[3] * pi
    funct = initsoliton_jit(funct, xarray, yarray, zarray, position, alpha, f, delta_x)
    ####### Impart velocity to solitons in Galilean invariant way
    velx = velocity[0]
    vely = velocity[1]
    velz = velocity[2]
    funct = ne.evaluate("exp(1j*(alpha*beta*t0 + velx*xarray + vely*yarray + velz*zarray -0.5*(velx*velx+vely*vely+velz*velz)*t0  + phase))*funct")
    psi = ne.evaluate("psi + funct")


fft_psi = pyfftw.builders.fftn(psi, axes=(0, 1, 2), threads=num_threads)

ifft_funct = pyfftw.builders.ifftn(funct, axes=(0, 1, 2), threads=num_threads)       

Density = 0

# Initial Wavefunction Now In Memory

if Uniform:
    rho = ne.evaluate("abs(abs(psi)**2) - Density")
else:
    rho = ne.evaluate("abs(abs(psi)**2)")

rho = rho.real
##########################################################################################
# COMPUTE SIZE OF TIMESTEP (CAN BE INCREASED WITH step_factor)

delta_t = (lengthC/float(resol))**2/np.pi


h = time_factor * delta_t
##########################################################################################
# SETUP PADDED POTENTIAL HERE (From JLZ)

rhopad = pyfftw.zeros_aligned((2*resol, resol, resol), dtype='float64')
bigplane = pyfftw.zeros_aligned((2*resol, 2*resol), dtype='float64')

fft_X = pyfftw.builders.fftn(rhopad, axes=(0, ), threads=num_threads)
ifft_X = pyfftw.builders.ifftn(rhopad, axes=(0, ), threads=num_threads)

fft_plane = pyfftw.builders.fftn(bigplane, axes=(0, 1), threads=num_threads)
ifft_plane = pyfftw.builders.ifftn(bigplane, axes=(0, 1), threads=num_threads)

phiSP = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')
phiTM = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64') # New, separate treatment.

fft_phi = pyfftw.builders.fftn(phiSP, axes=(0, 1, 2), threads=num_threads)
##########################################################################################
# SETUP K-SPACE FOR RHO (REAL)

rkvec = 2 * np.pi * np.fft.fftfreq(resol, lengthC / float(resol))

krealvec = 2 * np.pi * np.fft.rfftfreq(resol, lengthC / float(resol))

rkxarray, rkyarray, rkzarray = np.meshgrid(
    rkvec, rkvec, krealvec,
    sparse=True, indexing='ij'
)

rkarray2 = ne.evaluate("rkxarray**2+rkyarray**2+rkzarray**2")

rfft_rho = pyfftw.builders.rfftn(rho, axes=(0, 1, 2), threads=num_threads)


phik = rfft_rho(rho)  # not actually phik but phik is defined in next line

phik = ne.evaluate("-4*pi*phik/rkarray2")

phik[0, 0, 0] = 0

irfft_phi = pyfftw.builders.irfftn(phik, axes=(0, 1, 2), threads=num_threads)


##########################################################################################
# COMPUTE INTIAL VALUE OF POTENTIAL

if IsoP:
    try:
        green = np.load(f'./Green Functions/G{resol}.npy')
    except FileNotFoundError:
        if not os.path.exists('./Green Functions/'):
            os.mkdir('./Green Functions/')
        green = makeDCTGreen(resol) #make Green's function ONCE
        np.save(f'./Green Functions/G{resol}.npy',green)

    #green = makeEvenArray(green)
    phiSP = IP_jit(rho, green, lengthC, fft_X, ifft_X, fft_plane, ifft_plane)

else:
    phiSP = irfft_phi(phik)

##########################################################################################
phi = phiSP


MI = 0

GridMass = [Vcell*np.sum(rho)] # Mass of ULDM in Grid


#######################################

if np.isnan(rho).any() or np.isnan(psi).any():
    raise RuntimeError("Something is seriously wrong.")
    
    
print('Initialized!')


Success = True
TIntegrate = 0
I = 0

fig = plt.figure(figsize = (16,9))

axA = fig.add_subplot(121)
axB = fig.add_subplot(122)



imA = axA.imshow(np.array(rho[:,:,resol//2]), animated=True, cmap = 'magma', extent = (-length/2,length/2,-length/2,length/2,))

imB = axB.imshow(np.array(phi[:,:,resol//2]), animated=True, cmap = 'ocean_r', extent = (-length/2,length/2,-length/2,length/2,))

fig.canvas.set_window_title('PyUltralight Live')

import time
def updatefig(*args):
    
    time0 = time.time()
    global I, TIntegrate, psi, rho, phi
    TIntegrate += h
    
    psi = ne.evaluate("exp(-1j*h*phi)*psi")

    funct = fft_psi(psi)
    
    funct = ne.evaluate("funct*exp(-1j*0.5*h*karray2)")

    psi = ifft_funct(funct)

    rho = ne.evaluate("(abs(psi)**2)")
    rho = rho.real
    
    phik = rfft_rho(rho)  # not actually phik but phik is defined in next line

    phik = ne.evaluate("-4*pi*phik/rkarray2")

    phik[0, 0, 0] = 0

    # New Green Function Methods
    if not IsoP:
        phiSP = irfft_phi(phik)
    else:
        phiSP = IP_jit(rho, green, lengthC, fft_X, ifft_X, fft_plane, ifft_plane)

    phi = phiSP

    imA.set_array(np.array(rho[:,:,resol//2]))
    imB.set_array(np.array(phi[:,:,resol//2]))
    Gap = time.time() - time0
    
    FPS = int(1 / Gap)
    plt.draw()
    I+= 1

    
    print(f'\r{TIntegrate:.4f} @ {FPS} FPS.', end = '', flush = 'true')
    return imA, imB,
    

ani = animation.FuncAnimation(fig, updatefig, interval=1, blit=True)

plt.show()