import numpy as np
from numpy import linalg as la
from sklearn.decomposition import PCA
import wav

def getKey(item):
    return item[0]

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
    decompressed = pca.inverse_transform(compressedMatrix).reshape((len(paddedVector)))
    decompressed = decompressed[0:-filling]
    return pca, compressedMatrix, decompressed

def manualCompress(soundVector, coeff, matrixSize):
    #Reshaping soundVector into a matrix
    vectorSize = len(soundVector)
    filling = matrixSize - np.mod(vectorSize, matrixSize) #zeros to add in the padded vector
    paddedVector = np.lib.pad(soundVector, (0, filling), 'constant', constant_values=0)
    initialMatrix = paddedVector.reshape((len(paddedVector) // matrixSize, matrixSize)) # Reshape soundVector to be a matrix of dimension matrixSize

    #Calculation of eigen values
    productInitialMatrix = np.transpose(initialMatrix).dot(initialMatrix)
    V=np.true_divide(productInitialMatrix,(vectorSize-1)) #standardize data
    eigVals,eigVects = la.eig(V) # compute eigen values and eigen vectors
    eigPairs = [(np.abs(eigVals[i]), eigVects[:,i]) for i in range(len(eigVals))] #form pairs of (eigen value, eigen vector)
    eigPairs = sorted(eigPairs, key=getKey, reverse=True) #sort pairs by value from highest to lowest

    #Calculation of eigen matrix
    eigMatrix = eigPairs[0][1]
    for i in range (1,coeff):
        eigMatrix = np.c_[eigMatrix,eigPairs[i][1].reshape(matrixSize,1)] #selection of the right number of columns for pca
    #Calculation of compressed matrix
    compressedMatrix = initialMatrix.dot(eigMatrix)

    #decompression of compressed matrix
    filledComp = compressedMatrix
    filledEig = eigMatrix
    for i in range (1,matrixSize-coeff): #fill deleted columns with zeros
        filledComp = np.c_[compressedMatrix,np.zeros(compressedMatrix.shape[0])]
        filledEig = np.c_[eigMatrix,np.zeros(eigMatrix.shape[0])]
    decompressed = filledComp.dot(np.transpose(filledEig)) #calculate decompressed matrix
    decompressed = decompressed.reshape(len(paddedVector)) #reshape into vector
    decompressed = decompressed[0:-filling]
    return compressedMatrix, decompressed

#MAIN
def main():
    h, sample = wav.open("speech_8kHz.wav")
    x=2
    if x == 0:
        pca, matrix, out = compress(sample,64,256)
        out = out.astype(int)
        wav.save("compressed.wav",h,out)
    else:
        _,out = manualCompress(sample,64,256)
        out = out.astype(int)
        wav.save("compressed.wav",h,out)

if __name__ == "__main__":
    main()
