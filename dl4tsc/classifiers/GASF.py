from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import GramianAngularField
import numpy as np


def GASF_enc(Dataset, P=128):
    
    N = Dataset.shape[0]
    T = Dataset.shape[1]
    M = Dataset.shape[2]
    P = P #Length of the approximated representation
    
    Dataset_PAA = np.zeros((N,P, M))
    # PAA transformation
    paa = PiecewiseAggregateApproximation(window_size=None, output_size=P)
    for i in range(M):
        X_paa = paa.transform(Dataset[:,:,i])
        Dataset_PAA[:,:,i] = X_paa
        
    transformer = GramianAngularField()
    Dataset_PAA_GAF = np.zeros((N,P,P,M))

    for i in range(M):
        transformer.fit(Dataset_PAA[:,:,i])
        X_paa_gaf = transformer.transform(Dataset_PAA[:,:,i])
        Dataset_PAA_GAF[:,:,:,i] = X_paa_gaf
    
    return Dataset_PAA_GAF