"""
Classes for calculating QCQP design bounds for photonics problems, 
bridging the QCQP Dual Problem Interface in cvxopt 
and the Maxwell Solvers in maxwell
"""

__all__ = []

import numpy as np
import scipy.sparse as sp
from dolphindes.cvxopt import SparseSharedProjQCQP
from dolphindes.maxwell import TM_FDFD

import warnings

class Photonics_FDFD():
    """
    Mother class for frequency domain photonics problems numerically described using FDFD.
    
    Attributes
    ----------
    omega : complex
        Circular frequency, can be complex to allow for finite bandwidth effects.
    Nx : int
        Number of pixels along the x direction.
    Ny : int
        Number of pixels along the y direction.
    Npmlx : int
        Size of the x direction PML in pixels.
    Npmly : TYPE
        Size of the y direction PML in pixels.
    dx : float
        Finite difference grid pixel size in x direction, in units of 1.
    dy : float
        Finite difference grid pixel size in y direction, in units of 1.
    chi : complex
        Bulk susceptibility of material used. 
    des_mask : boolean ndarray
        Boolean mask over computation domain that is TRUE for pixels in design region.
    ji : complex ndarray
        Incident current source that produces an incident field.
    ei : complex ndarray
        Incident field. 
    chi_background : complex ndarray
        The background structure. 
        The default is None, in which case it is set to vacuum.
    bloch_x : float
        x-direction phase shift associated with the periodic boundary condtions. 
    bloch_y : float
        y-direction phase shift associated with the periodic boundary condtions. 
    sparseQCQP : boolean
        Boolean flag indicating whether the sparse QCQP convention is used.
    A0 : complex np.ndarray or scipy.sparse.csc_array
        A0 array in the QCQP field design objective. 
    s0 : complex np.ndarray
        The vector s0 in the QCQP field design objective. 
    c0 : float
        The constant c0 in the QCQP field design objective. 
    """
    def __init__(self, omega, Nx, Ny, Npmlx, Npmly, dx, dy, # FDFD solver attr
                 chi, des_mask, ji=None, ei=None, chi_background=None,# design problem attr
                 bloch_x=0.0, bloch_y=0.0, # FDFD solver attr
                 sparseQCQP=True, A0=None, s0=None, c0=0.0): # design problem attr
        """
        A0, s0, c0 for design objective QCQP can be specified by user later
        """
        self.omega = omega
        self.Nx = Nx
        self.Ny = Ny
        self.Npmlx = Npmlx
        self.Npmly = Npmly
        self.dx = dx
        self.dy = dy
        self.bloch_x = bloch_x
        self.bloch_y = bloch_y
        
        self.chi = chi
        self.des_mask = des_mask
        self.ji = ji
        self.ei = ei
        self.chi_background = chi_background
        
        self.sparseQCQP = self.sparseQCQP
        self.A0 = A0
        self.s0 = s0
        self.c0 = c0
        self.Pdiags = None # constraints of QCQP; to be set up after init
        
        

class Photonics_TM_FDFD(Photonics_FDFD):
    def __init__(self, omega, Nx, Ny, Npmlx, Npmly, dl, # FDFD solver attr
                 chi, des_mask, ji=None, ei=None, chi_background=None, # design problem attr
                 bloch_x=0.0, bloch_y=0.0, # FDFD solver attr
                 sparseQCQP=True, A0=None, s0=None, c0=0.0): # design problem attr
        
        super().__init(omega, Nx, Ny, Npmlx, Npmly, dl, dl,
                     chi, des_mask, ji, ei,
                     bloch_x, bloch_y,
                     sparseQCQP, A0, s0, c0)
        
        self.FDFD = TM_FDFD(omega, Nx, Ny, Npmlx, Npmly, dl, bloch_x, bloch_y)
        self.QCQP = None # constructed later after the user specifies objective
    
    
    def setup_QCQP(self, Pdiags="global", verbose: float = 0):
        
        # check to make sure the objective function has been defined
        if (self.A0 is None) or (self.s0 is None):
            print("Please specify objective function A0 and s0 before QCQP setup.")
            return None
        
        # generate initial field
        if (self.ji is None) and (self.ei is None):
            raise ValueError("an initial current ji or field ei must be specified.")
        if not (self.ji is None) and not (self.ei is None):
            warnings.warn("If both ji and ei are specified then ji is ignored.")
        if self.ei is None:
            self.ei = self.FDFD.get_TM_field(self.ji, self.chi_background)
        
        Ndes = self.Ginv.shape[0]
        
        if Pdiags=="global":
            self.Pdiags = np.ones((Ndes,2), dtype=complex)
            self.Pdiags[:,1] = -1j
        else:
            raise ValueError("Not a valid Pdiags specification / needs implementation")
        
        if self.sparseQCQP: # rewrite later when sparse and dense QCQP classes are unified
            self.Ginv, self.M = self.FDFD.get_GaaInv(self.des_mask, self.chi_background)
            A1 = np.conj(1.0/self.chi) * self.Ginv.conj().T - sp.eye(self.Ginv.shape[0], dtype=sp.csc_array)
            A2 = self.Ginv
            self.QCQP = SparseSharedProjQCQP(self.A0, self.s0, self.c0,
                                            A1, A2, self.ei/2, 0.0, #c1 = 0.0, remove if c1 is patched out
                                            self.Pdiags, verbose
                                            )
        else:
            raise ValueError("dense QCQP formulation not fully implemented yet")



class Photonics_TE_Yee_FDFD(Photonics_FDFD):
    def __init__(self):
        pass