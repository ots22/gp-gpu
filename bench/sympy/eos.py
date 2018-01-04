import numpy as np
from sympy import *
from sympy.utilities.codegen import codegen

def diff_of_matrix(M, x):
    return Matrix(M.rows, M.cols, lambda i,j: diff(M[i,j], x))

def diff_by_matrix(expr, M):
    return Matrix(M.rows, M.cols, lambda i,j: diff(expr, M[i,j]))

G_names_full = 'G11 G12 G13 G21 G22 G23 G31 G32 G33'
for p in G_names_full.split(): vars()[p] = symbols(p)

G_names_symmetric = 'G11 G22 G33 G23 G13 G12'

G = Matrix([[G11, G12, G13],
            [G21, G22, G23],
            [G31, G32, G33]])

F_names_full = 'F11 F12 F13 F21 F22 F23 F31 F32 F33'
for p in F_names_full.split(): vars()[p] = symbols(p)

F = Matrix([[F11, F12, F13],
            [F21, F22, F23],
            [F31, F32, F33]])

def I1():
    return trace(G)

def I2():
    return simplify((trace(G)**2 - trace(G**2))/2)

def I3():
    return det(G)

E, S = symbols('E, S')

params = 'rho_0 K0 B0 alpha beta gamma cv T0'
for p in params.split(): vars()[p] = symbols(p)

e_internal = K0/(2*alpha**2) * (I3()**(alpha/2) - 1)**2      \
    + cv*T0*I3()**(gamma/2) * (exp(S/cv) - 1)       \
    + (B0/2)*I3()**(beta/2) * (I1()**2/3 - I2())

# from axiom (sympy had trouble with the solve)
entropy = cv*log((6*T0*alpha*alpha*cv*I3()**(gamma/2)+(3*B0*I2()+(-B0*I1()*I1()))*alpha*alpha*I3()**(beta/2)+(-3*K0*I3()**alpha)+6*K0*I3()**(alpha/2)+6*E*alpha*alpha+(-3*K0))/(6*T0*alpha*alpha*cv*I3()**(gamma/2)))

# stress
sigma = -2 * rho_0 * sqrt(I3()) * G * diff_by_matrix(e_internal, G)


## now work in terms of F
# G_F = simplify(F.inv() * transpose(F.inv()))

# e_internal_F = e_internal.subs([(G11,G_F[0,0]), (G12,G_F[0,1]), (G13,G_F[0,2]), (G22,G_F[1,1]), (G23, G_F[1,2]), (G33,G_F[2,2])])

# sigma_F = diff_by_matrix(e_internal_F, F)

# numpy array of matrices -- couldn't find a better way in sympy.
dsigma_dG = np.empty((3,3), dtype=object)
# G_F = simplify(F.inv() * transpose(F.inv()))
# dG_dF = np.empty((3,3), dtype=object)
for i in range(3):
    for j in range(3):
        dsigma_dG[i,j] = diff_by_matrix(sigma[i,j], G)
#        dG_dF[i,j] = diff_by_matrix(G_F[i,j], F)


dsigma_dS = diff_of_matrix(sigma, S)

# we want diff_by_matrix to give a partial derivative to the entry
# G[i,j].  This means we cannot set G as a symmetric matrix of symbols
# above and differentiate with respect to them, or we get the wrong
# answer.  To symmetrize it, we may substitute the correct symbols,
# but only after differentiating.

symmetric_pairs = [(G21, G12), (G31, G13), (G32, G23)]

e_internal = e_internal.subs(symmetric_pairs)
sigma = sigma.subs(symmetric_pairs)
entropy = entropy.subs(symmetric_pairs)
dsigma_dS = dsigma_dS.subs(symmetric_pairs)
for i in range(3):
    for j in range(3):
        dsigma_dG[i,j] = dsigma_dG[i,j].subs(symmetric_pairs)

with open('romenskii_energy.inc','w') as f:
    f.write(printing.ccode(e_internal, assign_to='result'))
    
with open('romenskii_entropy.inc','w') as f:
    f.write(printing.ccode(entropy, assign_to='result'))

with open('romenskii_stress.inc','w') as f:
    for i in range(3):
        for j in range(3):
            f.write(printing.ccode(sigma[i,j], assign_to="result(%d,%d)" % (i,j)))
            f.write("\n\n")

for i in range(3):
    for j in range(3):
        with open('romenskii_dsigma%d%d_dG.inc' % (i+1,j+1), 'w') as f:
            for p in range(3):
                for q in range(3):
                    f.write(printing.ccode(dsigma_dG[i,j][p,q], assign_to="result(%d,%d)" % (p,q)))
                    f.write("\n\n")

with open('romenskii_dsigma_dS.inc','w') as f:
    for i in range(3):
        for j in range(3):
            f.write(printing.ccode(dsigma_dS[i,j], assign_to="result(%d,%d)" % (i,j)))
            f.write("\n\n")


