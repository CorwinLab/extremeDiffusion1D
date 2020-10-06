# @Author: Eric Corwin <ecorwin>
# @Date:   2020-10-06T12:40:32-07:00
# @Email:  eric.corwin@gmail.com
# @Filename: runOneBCSimulation.py
# @Last modified by:   ecorwin
# @Last modified time: 2020-10-06T13:21:42-07:00

import BarraquandCorwin as bc
import numpy as np
import sys
import os

# Call this program with arguments topDir, numWalkers, biasFunction, uniqueId

if __name__ == '__main__':
    # Extract the information from the command line
    topDir = sys.argv[1]
    numWalkersStr = sys.argv[2]
    numWalkers = float( numWalkersStr )
    biasName = sys.argv[3]
    if biasName == 'einstein':
        biasFunction = bc.einsteinBias
    else:
        biasFunction = bc.uniformBias
    uniqueId = sys.argv[4]

    maxTime = np.int( np.log( numWalkers )**2 )
    # Run the simulation
    edges = bc.floatRunFixedTime(maxTime, biasFunction, numWalkers=numWalkers)
    # Save the data
    dirName = f'{topDir}/{biasName}/{numWalkersStr}'
    os.makedirs(dirName, exist_ok=True)
    fileName = f'{dirName}/{numWalkersStr}-{uniqueId}'
    np.savetxt(fileName, edges)
