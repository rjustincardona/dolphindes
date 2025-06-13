"""
useful mathematical operations
"""

def Sym(A):
    """
    Compute the symmetric Hermitian part of a matrix A
    TODO: add type hinting indicating that input is a generic matrix
    """
    return (A + A.T.conj()) / 2
