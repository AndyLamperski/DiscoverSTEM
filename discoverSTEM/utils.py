import numpy as np
import sympy as sym
import scipy.linalg as la
import scipy.signal as sp

def set_pweave_variables(filename,kwargs, options = { 'echo' : False },newname=None):
    optKeys = options.keys()
    opt_equal_strings = [key + " = " + str(options[key]) for key in optKeys]
    opt_string = ', '.join(opt_equal_strings) 
    argList = []
    keys = kwargs.keys()
    args = [str(kwargs[key]) for key in keys]
    pairStrings = [key + " = " + arg for key,arg in zip(keys,args)]
    defString = '\n'.join(pairStrings)
    topString = '\n<<PweaveHeadVariables, %s>>=\n%s\n@\n' % (opt_string,defString)
    filestring = open(filename).read()

    if newname is None:
        newname = filename + '.tmp'

    newString = topString + filestring
    with open(newname,"w") as newfile:
        newfile.write(newString)

    return newname



def controllabilityMatrix(A,B):
    n,p = B.shape

    Mat = np.array(B,copy=True)

    Cont = np.zeros((n,n*p))

    for k in range(n):
        Cont[:,k*p:(k+1)*p] = Mat
        Mat = np.dot(A,Mat)

    return Cont

def isControllable(A,B):
    Cont = controllabilityMatrix(A,B)
    Value = (np.linalg.matrix_rank(Cont) == len(A))
    return Value


def observabilityMatrix(A,C):
    return controllabilityMatrix(A.T,C.T).T

def minrealZPK(ZPK,tol=1e-8):
    Z,P,K = ZPK
    if np.abs(K) < tol:
        return np.array([]),np.array([]),0.
    P_list = list(P)
    Z_list = []
    for i,z in enumerate(Z):
        matchFound = False
        for j,p in enumerate(P_list):
            if np.abs(z-p) < tol:
                matchFound = True
                P_list.pop(j)
                break
        if not matchFound:
            Z_list.append(z)

    return np.array(Z_list),np.array(P_list),K
            

def minreal(Sys,tol=1e-8):
    if len(Sys) == 4: 
        A,B,C,D = Sys
    elif len(Sys) == 2:
        ZPK = sp.tf2zpk(*Sys)
        ZPK_min = minrealZPK(ZPK,tol)
        return sp.zpk2tf(*ZPK_min)
        
    Cont = controllabilityMatrix(A,B)
    Obs = observabilityMatrix(A,C)

    H = np.dot(Obs,Cont)
    Hs = np.dot(Obs,np.dot(A,Cont))

    U,S,Vh = la.svd(H)

    if len(np.argwhere(S>tol)) > 0:
        rank = np.argwhere(S>tol)[-1,0] + 1

        rtS = np.diag(np.sqrt(S[:rank]))
        ObsMin = np.dot(U[:,:rank],rtS)
        ContMin = np.dot(rtS,Vh[:rank])
    
        Cmin = ObsMin[:len(C)]
        Bmin = ContMin[:,:B.shape[1]]

        Amin = la.lstsq(ObsMin,la.lstsq(ContMin.T,Hs.T)[0].T)[0]

    else:
        Amin = np.zeros((1,1))
        Bmin = np.zeros((1,1))
        Cmin = np.zeros((1,1))

    SysMin = (Amin,Bmin,Cmin,D)
    # if len(Sys) == 4:
    #     return SysMin

    # elif len(Sys) == 2:
    #     return sp.ss2tf(*SysMin)


def prodStateSpace(P1,P2):
    A1,B1,C1,D1 = P1
    A2,B2,C2,D2 = P2

    Atop = np.hstack((A1,np.dot(B1,C2)))
    Abot = np.hstack((np.zeros((len(A1),len(A2))),A2))
    A = np.vstack((Atop,Abot))

    B = np.vstack((np.dot(B1,D2),
                   B2))

    C = np.hstack((C1,np.dot(D1,C2)))
    D = np.dot(D1,D2)

    return A,B,C,D

def classicFeedback(P,C):
    """
    Create a classical feedback interconnection, T, 
    with P and C as transfer functions.
    """

    nP,dP = P
    nC,dC = C

    n = np.polymul(nP,nC)
    d = np.polyadd(n,np.polymul(dP,dC))

    return n,d

def block_matrix(MBlock):
    Rows = [np.hstack(row) for row in MBlock]
    return np.vstack(Rows)

def dot(*args):
    M = reduce(lambda A,B : np.dot(A,B), args)
    return M

def cast_as_array(x):
    if isinstance(x,np.ndarray):
        return x
    else:
        return np.array(x)

def jacobian(F,x):
    """ 
    Computes: 
    J = dF / dx

    If x is a scalar, then J is just the derivative of each entry of F.

    If F is a scalar and x is a vector, then J is the gradient of F with respect to x. 

    If F and x are 1D vectors, then J is the standard Jacobian.

    More generally, the shape of J will be given by
    
    J.shape = (F.shape[0],...,F.shape[-1], x.shape[0],..., x.shape[-1])
    
    This assumes that F and x are scalars or arrays of symbolic variables.
    
    
    """

    Farr = cast_as_array(F)
    xarr = cast_as_array(x)

    Fflat = Farr.flatten()
    xflat = xarr.flatten()

    nF = len(Fflat)
    nx = len(xflat)
    # a matrix to hold the derivatives
    Mat = np.zeros((nF,nx),dtype=object)

    for i in range(nF):
        for j in range(nx):
            Mat[i,j] = sym.diff(Fflat[i],xflat[j])

    Jac = np.reshape(Mat, Farr.shape + xarr.shape)
    return Jac

def simplify(x):
    if isinstance(x,np.ndarray):
        xflat = x.flatten()
        nX = len(xflat)
        x_simp_list = [sym.simplify(expr) for expr in xflat]
        x_simp = np.reshape(x_simp_list,x.shape)
        return x_simp
    else:
        return sym.simplify(x)

def subs(f,x,xVal):

    if isinstance(x,np.ndarray):
        xflat = x.flatten()
        xValflat = xVal.flatten()
    else:
        xflat = [x]
        xValflat = [xVal]

    nX = len(xflat)
    
    if isinstance(f,np.ndarray):
        fflat = f.flatten()
        nF = len(fflat)

        fSubflat = np.zeros(nF,dtype=object)
        
        for i in range(nF):
            fCur = fflat[i]
            for (xCur,xValCur) in zip(xflat,xValflat):
                fCur = fCur.subs(xCur,xValCur)

            fSubflat[i] = fCur

        return np.reshape(fSubflat,f.shape)
    else:
        for i in range(nX):
            xCur = xflat[i]
            xValCur = xValflat[i]
            if i==0:
                fCur = f.subs(xCur,xValCur)
            else:
                fCur = fCur.subs(xCur,xValCur)

        return fCur
        
def arg_to_flat_tuple(argTup):
    """
    Takes in a tuple of arguments. If any are np.ndarrays, they will 
    be flattended and added to the tuple
    """
    argList = []
    for variable in argTup:
        if isinstance(variable,np.ndarray):
            argList.extend(variable.flatten())
        else:
            argList.append(variable)
    return tuple(argList)
    

def functify(expr, args):
    """
    Usage: 
    name = functify(expr,args)
    This creates a function of the form
    expr = name(args)

    For more information, see
    https://www.youtube.com/watch?v=99JS6ym5FNE
    """
    if isinstance(args,tuple):
        argTup = args
    else:
        argTup = (args,)

    FuncData = {'nvar': len(argTup),
                'shape': (),
                'squeeze': False,
                'flatten': False}


    flatTup = arg_to_flat_tuple(argTup)
    if isinstance(expr,np.ndarray):
        FuncData['shape'] = expr.shape
        if len(expr.shape) < 2:
            FuncData['squeeze'] = True
            exprCast = sym.Matrix(expr)
        elif len(expr.shape) > 2:
            FuncData['flatten'] = True
            exprCast = sym.Matrix(expr.flatten())
        else:
            exprCast = sym.Matrix(expr)

        mods = [{'ImmutableMatrix': np.array}, 'numpy']
        func = sym.lambdify(flatTup,exprCast,modules=mods)
    else:
        func = sym.lambdify(flatTup,expr)

    def foo(*new_arg):
        new_flatTup = arg_to_flat_tuple(new_arg)
        result = func(*new_flatTup)
        if FuncData['squeeze']:
            return result.squeeze()
        elif FuncData['flatten']:
            return np.reshape(result,FuncData['shape'])
        else:
            return result

    return foo
