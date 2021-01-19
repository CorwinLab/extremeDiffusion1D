# @Author: Eric Corwin <ecorwin>
# @Date:   2021-01-19T11:57:41-08:00
# @Email:  eric.corwin@gmail.com
# @Filename: generateTalkFigures.py
# @Last modified by:   ecorwin
# @Last modified time: 2021-01-19T14:28:16-08:00

import pickle
import numpy as np

def extractOne(allData, N, bias):
    for d in allData:
        if d[3] == N and d[4] == bias:
            return d


if __name__ == '__main__':
    with open('extremeDiff.pickle', 'rb') as f:
        allData = pickle.load(f)

    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({'font.size': 18})
    plotDir = '/home/ecorwin/Dropbox/Talks/Simons2021/'
    ylim = (1e-1, 267851.4425517303)
    xlim = (0.5525248156835681, 257402.31246125474)
    plt.ylim(ylim)
    plt.xlim(xlim)
    tMax = np.log(1e50)**(5/2)
    tMin = np.log2(1e50)
    plt.xlabel('Time')
    plt.ylabel('Variance of Outlier')

    # steps = False
    steps = True


    # Start with just the regular diffusion
    # Expectation
    plt.loglog([tMin, tMax], .75/np.log(1e50) * np.array([tMin, tMax]), color=[.75, 0, 0])
    if steps:
        plt.savefig(plotDir+'RegularLine.pdf', bbox_inches='tight')

    # Data
    d = extractOne(allData, 1e50, 100)
    plt.loglog(np.array(d[0]), 4*d[2],color=[1,0,0])
    plt.loglog([tMin, tMax], .75/np.log(1e50) * np.array([tMin, tMax]), color=[.75, 0, 0])
    if steps:
        plt.savefig(plotDir+'RegularData.pdf', bbox_inches='tight')


    # Totally sticky
    # Expectation
    plt.loglog([1, tMax], [1/3, tMax/3], color=[0,0,.75])
    if steps:
        plt.savefig(plotDir+'StickyLine.pdf', bbox_inches='tight')
    # Data
    stickyData =  np.random.choice([-1,1], size=[np.int(tMax), 100])
    plt.loglog(np.arange(np.int(tMax))+1, np.var(np.abs(np.cumsum(stickyData, axis=0)), axis=1), color=[0,0,1])
    plt.loglog([1, tMax], [1/3, tMax/3], color=[0,0,.75])
    if steps:
        plt.savefig(plotDir+'StickyData.pdf', bbox_inches='tight')


    # Uniform
    alpha = 1.11111111
    d = extractOne(allData, 1e50, alpha)
    plt.loglog(np.array(d[0]), 4*d[2],color=[.5,0,.5])
    if steps:
        plt.savefig(plotDir+'UniformData.pdf', bbox_inches='tight')

    # Infill

    alpha = 0.5
    d = extractOne(allData, 1e50, alpha)
    plt.loglog(np.array(d[0]), 4*d[2],color=[.4, 0, .6])

    alpha = 0.1
    d = extractOne(allData, 1e50, alpha)
    plt.loglog(np.array(d[0]), 4*d[2],color=[.3, 0, .7])

    alpha = 0.01
    d = extractOne(allData, 1e50, alpha)
    plt.loglog(np.array(d[0]), 4*d[2],color=[.2, 0, .8])
    if steps:
        plt.savefig(plotDir+'GettingStickyData.pdf', bbox_inches='tight')


    alpha = 2.5
    d = extractOne(allData, 1e50, alpha)
    plt.loglog(np.array(d[0]), 4*d[2],color=[.75, 0, .25])
    if steps:
        plt.savefig(plotDir+'GettingLessStickyData.pdf', bbox_inches='tight')

    #Data as a function of system size
    plt.clf()
    # plt.ylim(ylim)
    # plt.xlim(xlim)
    plt.xlabel('Time')
    plt.ylabel('Variance of Outlier')

    #Einstein
    d = extractOne(allData, 1e50, np.inf)
    plt.loglog(np.array(d[0]), 4*d[2],color=[0, 0, 0])
    d = extractOne(allData, 1e100, np.inf)
    plt.loglog(np.array(d[0]), 4*d[2],color=[.2, 0, 0])
    d = extractOne(allData, 1e150, np.inf)
    plt.loglog(np.array(d[0]), 4*d[2],color=[.4, 0, 0])
    d = extractOne(allData, 1e200, np.inf)
    plt.loglog(np.array(d[0]), 4*d[2],color=[.6, 0, 0])
    d = extractOne(allData, 1e250, np.inf)
    plt.loglog(np.array(d[0]), 4*d[2],color=[.8, 0, 0])
    d = extractOne(allData, 1e300, np.inf)
    plt.loglog(np.array(d[0]), 4*d[2],color=[1, 0, 0])

    if steps:
        plt.savefig(plotDir+'EinsteinFiniteSize.pdf', bbox_inches='tight')

    plt.clf()
    # plt.ylim(ylim)
    # plt.xlim(xlim)
    plt.xlabel('Time')
    plt.ylabel('Variance of Outlier')

    #Uniform
    d = extractOne(allData, 1e50, 1)
    plt.loglog(np.array(d[0]), 4*d[2],color=[0, 0, 0])
    d = extractOne(allData, 1e100, 1)
    plt.loglog(np.array(d[0]), 4*d[2],color=[0, 0, .2])
    d = extractOne(allData, 1e150, 1)
    plt.loglog(np.array(d[0]), 4*d[2],color=[0, 0, .4])
    d = extractOne(allData, 1e200, 1)
    plt.loglog(np.array(d[0]), 4*d[2],color=[0, 0, .6])
    d = extractOne(allData, 1e250, 1)
    plt.loglog(np.array(d[0]), 4*d[2],color=[0, 0, .8])
    d = extractOne(allData, 1e300, 1)
    plt.loglog(np.array(d[0]), 4*d[2],color=[0, 0, 1])

    if steps:
        plt.savefig(plotDir+'UniformFiniteSize.pdf', bbox_inches='tight')

    # Collapse
    plt.clf()
    # plt.ylim(ylim)
    # plt.xlim(xlim)
    plt.xlabel('Time/Log(N)')
    plt.ylabel('Variance of Outlier')
    plt.xlim([.5, 1e3])
    #Einstein
    d = extractOne(allData, 1e300, np.inf)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2],color=[1, 0, 0])
    d = extractOne(allData, 1e250, np.inf)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2],color=[.8, 0, 0])
    d = extractOne(allData, 1e200, np.inf)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2],color=[.6, 0, 0])
    d = extractOne(allData, 1e150, np.inf)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2],color=[.4, 0, 0])
    d = extractOne(allData, 1e100, np.inf)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2],color=[.2, 0, 0])
    d = extractOne(allData, 1e50, np.inf)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2],color=[0, 0, 0])
    if steps:
        plt.savefig(plotDir+'EinsteinMasterCurve.pdf', bbox_inches='tight')

    plt.clf()
    # plt.ylim(ylim)
    # plt.xlim(xlim)
    plt.xlabel('Time/Log(N)')
    plt.ylabel('Variance of Outlier/Log(N)^(2/3)')
    plt.xlim([.5, 1e3])

    #Uniform
    d = extractOne(allData, 1e300, 1)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2]/np.log(d[3])**(2/3),color=[1, 0, 0])
    d = extractOne(allData, 1e250, 1)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2]/np.log(d[3])**(2/3),color=[.8, 0, 0])
    d = extractOne(allData, 1e200, 1)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2]/np.log(d[3])**(2/3),color=[.6, 0, 0])
    d = extractOne(allData, 1e150, 1)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2]/np.log(d[3])**(2/3),color=[.4, 0, 0])
    d = extractOne(allData, 1e100, 1)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2]/np.log(d[3])**(2/3),color=[.2, 0, 0])
    d = extractOne(allData, 1e50, 1)
    plt.loglog(np.array(d[0])/np.log(d[3]), 4*d[2]/np.log(d[3])**(2/3),color=[0, 0, 0])
    if steps:
        plt.savefig(plotDir+'UniformMasterCurve.pdf', bbox_inches='tight')


#    d = extractOne(allData, 1e50, 0.01)
#    plt.loglog(np.array(d[0]), 8*d[2]+1,'k')

    plt.savefig(plotDir+'test.pdf', bbox_inches='tight')
