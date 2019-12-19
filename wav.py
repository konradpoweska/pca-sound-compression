import numpy as np
import wave, struct


widthStruct = {
  1: 'b',
  2: 'h'
}


def open(filename):
  file = wave.open(filename, 'r')

  nchannels = file.getnchannels()
  sampwidth = file.getsampwidth()
  framerate = file.getframerate()
  nfarmes = file.getnframes()
  headers = nchannels, sampwidth, framerate, nfarmes

  structFormat = "<{0}{1}".format(nfarmes, widthStruct[sampwidth])
  frames = file.readframes(nfarmes)
  samples = np.array(struct.unpack(structFormat, frames))

  file.close()
  return headers, samples


def save(filename, headers, samples):
  file = wave.open(filename, 'w')

  nchannels, sampwidth, framerate, nfarmes = headers
  file.setnchannels(nchannels)
  file.setsampwidth(sampwidth)
  file.setframerate(framerate)
  file.setnframes(nfarmes)

  structFormat = "<{0}{1}".format(nfarmes, widthStruct[sampwidth])
  frames = struct.pack(structFormat, *samples)
  file.writeframes(frames)

  file.close()
