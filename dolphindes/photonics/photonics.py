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
from dolphindes.util import check_attributes
from typing import Tuple 
import warnings

import warnings

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
    Npmly : int
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
    def __init__(self, omega, chi=None, 
                 grid_size : Tuple[int, int] = (None, None), 
                 pml_size : Tuple[int, int] = (None, None), 
                 dl=None, # FDFD solver attr
                 des_mask : np.ndarray = None, ji : np.ndarray = None, 
                 ei : np.ndarray = None, chi_background : np.ndarray = None, # design problem attr
                 bloch_x=0.0, bloch_y=0.0, # FDFD solver attr
                 sparseQCQP=True, A0=None, s0=None, c0=0.0): # design problem attr
        
        Nx, Ny = grid_size
        Npmlx, Npmly = pml_size
        self.dl = dl
        self.des_mask = des_mask
        self.Ginv = None
        self.M = None

        super().__init__(omega, chi, Nx, Ny, Npmlx, Npmly, dl, dl,
                         des_mask, ji, ei, chi_background,
                         bloch_x, bloch_y,
                         sparseQCQP, A0, s0, c0)
        
        try:
            check_attributes(self, 'omega', 'chi', 'Nx', 'Ny', 'Npmlx', 'Npmly', 'des_mask', 'chi_background', 'bloch_x', 'bloch_y')
            self.setup_EM_solver()
            self.setup_EM_operators()
            
        except AttributeError as e:
            warnings.warn(f"Photonics_TM_FDFD initialized with missing attributes (lazy initialization). We strongly recommend passing all arguments for expected behavior.")

    def __repr__(self):
        return (f"Photonics_TM_FDFD(omega={self.omega}, chi={self.chi}, Nx={self.Nx}, "
                f"Ny={self.Ny}, Npmlx={self.Npmlx}, Npmly={self.Npmly}, dl={self.dl}, "
                f"des_mask={self.des_mask is not None}, ji={self.ji is not None}, "
                f"ei={self.ei is not None}, chi_background={self.chi_background is not None}, "
                f"bloch_x={self.bloch_x}, bloch_y={self.bloch_y})")

    def set_objective(self, A0=None, s0=None, c0=0.0):
        """
        set the QCQP objective function
        """
        if A0 is not None:
            self.A0 = A0
        if s0 is not None:
            self.s0 = s0
        if c0 is not None:
            self.c0 = c0

            
    def setup_EM_solver(self, omega=None, Nx=None, Ny=None, Npmlx=None, Npmly=None, dl=None, bloch_x=None, bloch_y=None):
        """
        setup the solver. non-None arguments will define / modify corresponding attributes
        """
        params = locals()
        params.pop('self')
        for param_name, param_value in params.items():
            if param_value is not None:
                setattr(self, param_name, param_value)

        check_attributes(self, 'Nx', 'Ny', 'Npmlx', 'Npmly', 'dl', 'bloch_x', 'bloch_y')
        self.EM_solver = TM_FDFD(self.omega, self.Nx, self.Ny, self.Npmlx, self.Npmly, self.dl, self.bloch_x, self.bloch_y)
    
    
    def setup_EM_operators(self):
        """
        setup EM operators associated with the given design region and background
        """
        check_attributes(self, 'des_mask')
        if self.sparseQCQP:
            self.Ginv, self.M = self.EM_solver.get_GaaInv(self.des_mask, self.chi_background)
        else:
            raise ValueError("dense QCQP not implemented yet")
    
    def get_ei(self, ji = None, update=False):
        """
        get the incident field
        """
        if self.ei is None:
            ei = self.EM_solver.get_TM_field(ji, self.chi_background) if self.ji is None else self.EM_solver.get_TM_field(self.ji, self.chi_background)
        else:
            ei = self.ei        
        if update: self.ei = ei
        return ei
    
    def set_ei(self, ei):
        self.ei = ei 
        
    def setup_QCQP(self, Pdiags="global", verbose: float = 0):
        """
        
        """
        check_attributes(self, 'des_mask', 'A0', 's0', 'c0')
        
        self.Ndes = int(np.sum(self.des_mask)) # number of field degrees of freedom / pixels in design region
        
        # generate initial field
        if (self.ji is None) and (self.ei is None):
            raise AttributeError("an initial current ji or field ei must be specified.")
        if not (self.ji is None) and not (self.ei is None):
            warnings.warn("If both ji and ei are specified then ji is ignored.")
        
        self.get_ei(self.ji, update=True)

        if Pdiags=="global":
            self.Pdiags = np.ones((self.Ndes,2), dtype=complex)
            self.Pdiags[:,1] = -1j
        else:
            raise ValueError("Not a valid Pdiags specification / needs implementation")
        
        if self.sparseQCQP: # rewrite later when sparse and dense QCQP classes are unified
            if (self.Ginv is None) or (self.M is None):
                self.setup_EM_operators()
            
            A0_sparse = self.Ginv.T.conj() @ sp.csc_array(self.A0) @ self.Ginv
            A1_sparse = sp.csc_array(np.conj(1.0/self.chi) * self.Ginv.conj().T - sp.eye(self.Ndes))
            A2_sparse = sp.csc_array(self.Ginv)
            s0_sparse = self.Ginv.T.conj() @ self.s0

            self.QCQP = SparseSharedProjQCQP(A0_sparse, s0_sparse, self.c0, 
                                            A1_sparse, A2_sparse, self.ei[self.des_mask]/2, 
                                            self.Pdiags, verbose
                                            )
        else:
            raise ValueError("dense QCQP formulation not fully implemented yet")

    def bound_QCQP(self, method : str = 'bfgs', init_lags : np.ndarray = None, opt_params : dict = None):
        """
        Calculate a limit on the QCQP
        """
        return self.QCQP.solve_current_dual_problem(method = method, init_lags = init_lags, opt_params = opt_params)
        


class Photonics_TE_Yee_FDFD(Photonics_FDFD):
    def __init__(self):
        pass