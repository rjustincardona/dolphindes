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



def check_attributes(self, *attrs):
    ### maybe this helper function for lazy initialization can also be useful elsewhere?
    missing = [attr for attr in attrs if getattr(self, attr) is None]
    if missing:
        raise AttributeError(f"{', '.join(missing)} undefined.")


class Photonics_FDFD():
    """
    Mother class for frequency domain photonics problems numerically described using FDFD.
    To allow for lazy initialization, only demands omega upon init
    
    Attributes
    ----------
    omega : complex
        Circular frequency, can be complex to allow for finite bandwidth effects.
    chi : complex
        Bulk susceptibility of material used. 
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
        Default: 0.0
    bloch_y : float
        y-direction phase shift associated with the periodic boundary condtions. 
        Default: 0.0
    sparseQCQP : boolean
        Boolean flag indicating whether the sparse QCQP convention is used.
    A0 : complex np.ndarray or scipy.sparse.csc_array
        A0 array in the QCQP field design objective. 
    s0 : complex np.ndarray
        The vector s0 in the QCQP field design objective. 
    c0 : float
        The constant c0 in the QCQP field design objective. 
    """
    def __init__(self, omega, chi=None, Nx=None, Ny=None, Npmlx=None, Npmly=None, dx=None, dy=None, # FDFD solver attr
                 des_mask=None, ji=None, ei=None, chi_background=None,# design problem attr
                 bloch_x=0.0, bloch_y=0.0, # FDFD solver attr
                 sparseQCQP=None, A0=None, s0=None, c0=0.0, Pdiags=None): # design problem attr
        """
        only omega is absoluted needed for initialization, other attributes can be added later
        """
        self.omega = omega
        self.chi = chi
        
        self.Nx = Nx
        self.Ny = Ny
        self.Npmlx = Npmlx
        self.Npmly = Npmly
        self.dx = dx
        self.dy = dy
        self.bloch_x = bloch_x
        self.bloch_y = bloch_y
        
        self.des_mask = des_mask
        self.ji = ji
        self.ei = ei
        self.chi_background = chi_background
        
        self.sparseQCQP = sparseQCQP
        self.A0 = A0
        self.s0 = s0
        self.c0 = c0
        self.Pdiags = Pdiags
        
        

class Photonics_TM_FDFD(Photonics_FDFD):
    def __init__(self, omega, chi=None, Nx=None, Ny=None, Npmlx=None, Npmly=None, dl=None, # FDFD solver attr
                 des_mask=None, ji=None, ei=None, chi_background=None,# design problem attr
                 bloch_x=0.0, bloch_y=0.0, # FDFD solver attr
                 sparseQCQP=True, A0=None, s0=None, c0=0.0): # design problem attr
        
        super().__init__(omega, chi, Nx, Ny, Npmlx, Npmly, dl, dl,
                         des_mask, ji, ei,
                         bloch_x, bloch_y,
                         sparseQCQP, A0, s0, c0)
        
        self.dl = dl
        

    def setup_FDFD(self, omega=None, Nx=None, Ny=None, Npmlx=None, Npmly=None, dl=None, bloch_x=None, bloch_y=None, des_mask=None):
        """
        setup the FDFD solver. non-None arguments will define / modify corresponding attributes
        """
        params = locals()
        params.pop('self')
        for param_name, param_value in params.items():
            if param_value is not None:
                setattr(self, param_name, param_value)

        check_attributes(self, 'Nx', 'Ny', 'Npmlx', 'Npmly', 'dl', 'bloch_x', 'bloch_y', 'des_mask')
        self.FDFD = TM_FDFD(self.omega, self.Nx, self.Ny, self.Npmlx, self.Npmly, self.dl, self.bloch_x, self.bloch_y)
        
    
    def setup_QCQP(self, Pdiags="global", verbose: float = 0):
        """
        
        """
        check_attributes(self, 'des_mask', 'A0', 's0', 'c0', 'Pdiags')
        
        self.Ndes = int(np.sum(self.des_mask)) # number of field degrees of freedom / pixels in design region
        
        # generate initial field
        if (self.ji is None) and (self.ei is None):
            raise AttributeError("an initial current ji or field ei must be specified.")
        if not (self.ji is None) and not (self.ei is None):
            warnings.warn("If both ji and ei are specified then ji is ignored.")
        if self.ei is None:
            self.ei = self.FDFD.get_TM_field(self.ji, self.chi_background)

        if Pdiags=="global":
            self.Pdiags = np.ones((self.Ndes,2), dtype=complex)
            self.Pdiags[:,1] = -1j
        else:
            raise ValueError("Not a valid Pdiags specification / needs implementation")
        
        if self.sparseQCQP: # rewrite later when sparse and dense QCQP classes are unified
            self.Ginv, self.M = self.FDFD.get_GaaInv(self.des_mask, self.chi_background)
            A1 = np.conj(1.0/self.chi) * self.Ginv.conj().T - sp.eye(self.Ndes)
            A2 = self.Ginv
            self.QCQP = SparseSharedProjQCQP(self.A0, self.s0, self.c0, 
                                            A1, A2, self.ei.flatten()/2, 
                                            self.Pdiags, verbose
                                            )
        else:
            raise ValueError("dense QCQP formulation not fully implemented yet")



class Photonics_TE_Yee_FDFD(Photonics_FDFD):
    def __init__(self):
        pass