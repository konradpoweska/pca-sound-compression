import numpy as np
from sklearn.decomposition import PCA

def compress(soundVector, coeff, matrixSize):
    #Reshaping soundVector into a matrix
    vectorSize = len(soundVector)
    filling = matrixSize - np.mod(vectorSize, matrixSize) #zeros to add in the padded vector
    paddedVector = np.lib.pad(soundVector, (0, filling), 'constant', constant_values=0)
    initialMatrix = paddedVector.reshape((len(paddedVector) // matrixSize, matrixSize)) # Reshape soundVector to be a matrix of dimension matrixSize

    #PCA
    pca = PCA(n_components=coeff)
    pca.fit(initialMatrix)
    compressedMatrix = pca.transform(initialMatrix)
    return pca, compressedMatrix
