import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def cusPlot(fileName, Type, Data):
    xy = np.vstack([Data[:, 0], Data[:, 1]])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    ax.scatter(Data[:, 0], Data[:, 1], c=z, edgecolor='')
    ax.set_title(f'Samples from {Type} of w')
    ax.set_xlabel('$w_{0}$')
    ax.set_ylabel('$w_{1}$')
    ax.set(xlim=(-10, 10), ylim=(-10, 10))
    fig.savefig(f'img/{fileName}')


def cusPlot2(fileName, Type, SampleX, SampleY, X, T, mean, cov):

    fig, ax = plt.subplots()
    lgnd = np.array(["variance"])
    ax.fill_between(SampleX[0], (mean-np.diag(cov)), (mean+np.diag(cov)), facecolor="y", alpha=0.5)
    lgnd = np.append(lgnd, ["mean"])
    ax.plot(SampleX[0], mean, c="y", linestyle="--")
    lgnd = np.append(lgnd, ["input data"])
    ax.scatter(X, T, c="r")
    for i, line in enumerate(SampleY):
        lgnd = np.append(lgnd, [f"Sample{i+1}"])
        ax.plot(SampleX[0], line)
    print(lgnd)
    ax.legend(lgnd)
    ax.set_title(f'Samples from {Type}')
    ax.set_xlabel('x')
    ax.set_ylabel('f')
    ax.set(xlim=(-6, 6), ylim=(-40, 40))
    fig.savefig(f'img/{fileName}')


def pickRandom(X, T, nrPoints, sigm, tau):
    i = np.random.randint(low=0, high=T.shape[1]-1, size=nrPoints)
    Xi = X[:, i]
    Ti = T[:, i]
    wCovPost = np.linalg.inv(np.dot(Xi, Xi.T)/sigm + 1/tau)
    wMeanPost = np.dot(wCovPost, np.dot(Xi, Ti.T))/sigm
    wPostSample = np.random.multivariate_normal(wMeanPost[:, 0], wCovPost, size=500)

    return wPostSample


def createData(sigm, nrSamples):
    noise = np.random.normal(0, sigm, (1, nrSamples))
    wTrue = np.array([[0.5, - 1.5]])
    X = np.vstack([np.linspace(-1, 1, nrSamples, endpoint=True), np.ones(nrSamples)])
    T = np.dot(wTrue, X) + noise
    return X, T


def squaredExpKernel(X1, X2, sigmf, L):
    K = sigmf*np.exp(np.dot(X1.T, X2)/(L**2))
    return K


def main():
    np.random.seed(seed=134678)
    n = 201

    # Question 9
    # SIGM = np.array([0.1, 0.2, 0.4, 0.8])
    # for sigm in SIGM:
    #
    #     X, T = createData(sigm=sigm, nrSamples=n)
    #
    #     # Create prior
    #     tau = 1
    #     wMean = np.array([0, 0])
    #     wCov = np.array([[tau, 0], [0, tau]])
    #     # w[w0, w1]
    #     WSamplePrior = np.random.multivariate_normal(wMean, wCov, size=500)
    #     # Q 9.1
    #     # Visualize prior
    #     cusPlot(fileName=f"question_9_1_sigma_{sigm}.png",
    #             Type=f"prior (sigma = {sigm})", Data=WSamplePrior)
    #
    #     # Q 9.2
    #     # T dim = D*n
    #     # X dim = q*n
    #     RNG = np.array([1, 5, 6, 7])
    #     for i in RNG:
    #         wPostSample = pickRandom(X=X, T=T, nrPoints=i, sigm=sigm, tau=tau)
    #         cusPlot(fileName=f"question_9_2_{i}_pts_sigm_{sigm}.png",
    #                 Type=f"posterior ({i} points, sigma = {sigm})", Data=wPostSample)

    # Question 10
    sigmF = 1
    # 0.1 and 0.2 is not positive semi definite
    # LL = np.array([0.4, 0.8, 5])
    # X, T = createData(sigm=0.2, nrSamples=n)
    # X = np.array([X[0]])
    # for L in LL:
    #     priorCov = squaredExpKernel(X=X, sigmf=sigmF, L=L)
    #     priorMean = np.zeros(shape=priorCov.shape[0])
    #     priorSample = np.random.multivariate_normal(priorMean, priorCov, size=10)
    #     cusPlot2(Data=[X, priorSample], Type=f"prior with exp kernel",
    #              fileName=f"quest_10_2_L_{L}.png")
    # Question 11
    n = 11
    L = 2
    X = np.array([np.linspace(-5, 5, n, endpoint=True)])
    sigm = 3
    noise = np.random.normal(0, sigm, (1, n))
    T = np.multiply(2+np.square(0.5*X-1), np.sin(2*X)) + noise

    priorCov = squaredExpKernel(X1=X, X2=X, sigmf=sigmF, L=L)
    priorMean = np.zeros(shape=priorCov.shape[0])
    priorSample = np.random.multivariate_normal(priorMean, priorCov, size=5)
    cusPlot2(SampleY=priorSample, SampleX=X, X=X, T=T, mean=priorMean, cov=priorCov, Type=f"prior with exp kernel",
             fileName=f"quest_10_L_{L}.png")
    n = 21
    XN1 = np.array([np.linspace(-5.5, 5.5, n, endpoint=True)])

    postMean = np.dot(np.dot(squaredExpKernel(X1=XN1, X2=X, sigmf=sigmF, L=L),
                             np.linalg.inv(priorCov + sigm*np.eye(priorCov.shape[0]))), T.T)
    postCov = squaredExpKernel(X1=XN1, X2=XN1, sigmf=sigmF, L=L) - \
        np.dot(np.dot(squaredExpKernel(X1=XN1, X2=X, sigmf=sigmF, L=L), np.linalg.inv(
            priorCov + sigm*np.eye(priorCov.shape[0]))), squaredExpKernel(X1=X, X2=XN1, sigmf=sigmF, L=L))

    postSamplef = np.random.multivariate_normal(postMean[:, 0], postCov, size=5)
    yPred = postSamplef + np.random.normal(0, sigm, postSamplef.shape)
    cusPlot2(SampleY=postSamplef, SampleX=XN1, X=X, T=T, mean=postMean[:, 0], cov=postCov, Type=f"posterior with exp kernel",
             fileName=f"quest_11_L_{L}.png")


if __name__ == "__main__":
    main()
