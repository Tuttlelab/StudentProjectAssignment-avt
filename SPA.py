# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:15:11 2023

@author: rkb19187
"""
import numpy as np
import pandas, sys, ase, os
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances

projects = choices = pandas.read_csv("Project List by Section.csv", index_col=0)
choices = pandas.read_csv("Project-Data.csv", index_col=0)
choices.columns = [int(x) for x in choices.columns]
Result = pandas.DataFrame(index=choices.index, columns=["Project"])
print(choices)
nStudents = choices.shape[0]

MaxScore = choices.shape[0] #[1/x for x in [1,1,1,1,....]]


indices = np.arange(0, projects.shape[0])
Result["Project"] = projects.index[np.random.choice(indices, choices.shape[0], replace=False)]


#Check that it is logical
def valid():
    return np.unique(Result["Project"].values).shape[0] == Result["Project"].values.shape[0]
if not valid():
    print("Project selection isnt valid")
    sys.exit()

#Calculate a score
score = 0
for student in Result.index:
    #print(student)
    project = Result.at[student, "Project"]
    if project in choices.loc[student].values:
        index = np.where(choices.loc["Student-1"].values == 70)[0][0] +1
        score += 1/index
    else:
        score += 0


print("Score:", score, "/", MaxScore)


### MD
if os.path.exists("Traj.xyz"):
    os.remove("Traj.xyz")
    
def Bond(rmin, bij, k):
    #Vb = 0.5 * k * (rmin - bij) ** 2
    #return Vb
    Fb = k * (rmin - bij) 
    return Fb

def NAMDLJ(Rmin, Emin, rij):
    V = Emin * ((Rmin/rij)**12 - (2 * (Rmin/rij)**6))
    return V

atoms = pandas.DataFrame(dtype=np.int64)
for i,student in enumerate(choices.index):
    atoms.at[student, "i"] = i
    atoms.at[student, "j"] = -1
for j,project in enumerate(projects.index):
    atoms.at[f"Project-{project}", "i"] = i+j
    atoms.at[f"Project-{project}", "j"] = j+1
atoms = atoms.astype(np.int64)
print(atoms)


positions = np.random.random((len(atoms), 3))*200

bonds = pandas.DataFrame(columns=["i", "j", "rmin", "K"])

for student in choices.index:
    i = atoms.at[student, "i"]
    for n,choice in enumerate(choices.loc[student]):
        if choice == -1:
            #this was blank
            continue
        j = atoms[atoms["j"] == choice]["i"].values[0]
        
        rmin = 10
        k = 10/(n+1)
    
        bonds.loc[bonds.shape[0]] = [i,j, rmin, k]
    

bonds["i"] = bonds["i"].astype(np.int64)
bonds["j"] = bonds["j"].astype(np.int64)
print(bonds)
bonds = bonds.values


#plot it
def Plot():
    global positions
    global bonds
    plt.scatter(*positions[:nStudents].T, color="orange", label="Students")
    plt.scatter(*positions[nStudents:].T, color="blue", label="Projects")
    for bond in bonds:
        i = int(bond[0])
        j = int(bond[1])
        pos_i = positions[i]
        pos_j = positions[j]
        line = np.vstack((pos_i, pos_j))    
        plt.plot(line[:,0], line[:,1], c="black", lw=1, alpha=0.4)
    plt.legend()
    plt.show()
    
def writeXYZ():
    global positions
    atoms = ["S"] * 85
    atoms += ["C"] * (positions.shape[0]-85)
    
    mol = ase.Atoms(atoms, positions)
    mol.write("Traj.xyz", append=True)


Forces = np.zeros(positions.shape)
V = np.int64(0)
def CalcForces():
    global Forces
    global V
    global positions
    global bonds
    Forces = np.zeros(positions.shape)
    V = np.int64(0)
    dr = 0.00001
    for bond in bonds:
        #print(bond)
        i = int(bond[0])
        j = int(bond[1])
        rmin = bond[2]
        k = bond[3]
        
        rij = np.linalg.norm(positions[i] - positions[j])
        #print(rij)
        Vb = 0.5 * k * (rij - rmin) ** 2
        V += Vb
        for dim in range(positions.shape[1]):
            dshift = np.zeros(positions.shape[1])
            dFi = []
            #stepsizes = np.array([0.0000001, 0.001, 0.1, 1, 10])
            d = 1
    
            dshift[dim] = d
            drij = np.linalg.norm((positions[i] + (dshift*dr)) - positions[j])
            dY = Vb - (0.5 * k * (drij - rmin) ** 2)
            dX = d*dr # its the absolute change in position, NOT CHANGE IN bond length!!
            if dX != 0.:
                Fi = (dY / dX)
                dFi.append(Fi)

            Forces[i][dim] += np.mean(dFi)
            Forces[j][dim] -= np.mean(dFi)
            
    d = euclidean_distances(positions, positions)    
    #pairlist = np.where(d < cutoff)
    #Same vdW between all particle
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            if j >= i:
                continue

            Rmin = 5
            Emin = 1
            
            #print(atom_i, atom_j, i, j)
            
            V_LJ = NAMDLJ(Rmin, Emin, d[i,j])
            #print(i, j, V_LJ, Rmin, Emin, d[i,j])
            V += V_LJ
            #print(i, j, atom_i, atom_j, sigma, eps, round(V_LJ, 1), d[i,j])
            
            for dim in range(positions.shape[1]):
                dshift = np.zeros((positions.shape[1]))
                dshift[dim] = 1
                #print(dshift)
                drij = np.linalg.norm((positions[i] + (dshift*dr)) - positions[j])
                
                dY = V_LJ - NAMDLJ(Rmin, Emin, drij)
                dX = dshift[dim]*dr # its the absolute change in position, NOT CHANGE IN distance!!
                if dX != 0.:
                    Fi = (dY / dX)
                    #print(Fi, k * (rmin - rij) * dX, k*np.linalg.norm((positions[i] + (dshift*dr))**2 - positions[j]**2) * drij)
                    #Fi = k * (rij**2 - rmin**2) * drij
                    Forces[i][dim] += Fi
                    Forces[j][dim] -= Fi


SGD = 0.1
maxstep = 10
#Plot()
writeXYZ()
#for iteration in range(2):
while maxstep > 0.00001:
    CalcForces()
    
    Fmax = np.abs(Forces).max()
    Vn = V.copy()
    pos_origin = positions.copy()
    
    rn_1 = np.ndarray(positions.shape)
    
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            if np.random.random() < SGD:
                rn_1[i,j] = positions[i,j] + (Forces[i,j] / Fmax)*maxstep
            else:
                rn_1[i,j] = positions[i,j]
    
    positions = rn_1.copy()
    CalcForces()
    print(f"New E: {round(V, 5)}, old E: {round(Vn, 5)} maxstep: {round(maxstep, 5)} Fmax: {round(Fmax, 3)}", end="\t")
    #write_xyz(traj_file)
    if V > Vn:
        maxstep = 0.2 * maxstep
        positions = pos_origin.copy()
        CalcForces()
        print("Reject", round(maxstep, 5))
    else:
        maxstep = 1.2 * maxstep
        print("Accept", round(maxstep, 5))
        #Plot()
        writeXYZ()
        
        