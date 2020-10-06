# @Author: Eric Corwin <ecorwin>
# @Date:   2020-10-06T12:40:32-07:00
# @Email:  eric.corwin@gmail.com
# @Filename: runOneBCSimulation.py
# @Last modified by:   ecorwin
# @Last modified time: 2020-10-06T13:01:21-07:00

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
    if sys.argv[3] == 'einstein':
        biasFunction = bc.einsteinBias
    else:
        biasFunction = bc.uniformBias
    uniqueId = sys.argv[4]

    maxTime = np.int( np.log( numWalkers )**2 )
    # Run the simulation
    edges = bc.floatRunFixedTime(maxTime, biasFunction, numWalkers=numWalkers)
    # Save the data
    if not os.path.isdir(f'{topDir}/{numWalkersStr}'):
        os.mkdir(f'{topDir}/{numWalkersStr}')
    fileName = f'{topDir}/{numWalkersStr}/{numWalkersStr}-{uniqueId}'
    np.savetxt(fileName, edges)
