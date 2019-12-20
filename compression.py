import numpy as np
from numpy import linalg as la
from sklearn.decomposition import PCA
import wav

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
    V=np.true_divide(np.dot(initialMatrix,np.transpose(initialMatrix)),(len(soundVector)-1)) #standardize data
    eigVals,eigVects = la.eig(V) # compute eigen values and eigen vectors


#MAIN
def main():
    h, sample = wav.open("speech_8kHz.wav")
    pca, matrix, out = compress(sample,2,4)
    out = out.astype(int)
    print(np.amax(out))
    wav.save("compressed.wav",h,out)
    #compression.manualCompress(sample,4,4)

if name == "main":
    main()
