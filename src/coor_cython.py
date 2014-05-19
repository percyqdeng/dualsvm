__author__ = 'qdengpercy'




def stoc_coor(a, kk, lmda):
    """
    min 0.5/lmda * a'*kk*a - a'*1
    sub to 0<= a <=1/n
    """
    n = kk.shape[0]

