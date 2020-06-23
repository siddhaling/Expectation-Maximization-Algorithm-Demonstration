from numpy import hstack
from numpy.random import normal
import numpy as np

# Generate two lists of values with different scale and loc using normal distribution
X1 = np.round(normal(loc=10, scale=2.2, size=6),2)
X2 = np.round(normal(loc=70, scale=2.5, size=6),2)
X = hstack((X1, X2))
X = X.reshape((len(X), 1))

#Compute the likelihood that each point belong to a group
def likelihoodMeasureByGaussian(sd,m,X):
    p_xbym=np.zeros(len(X))
    i=0
    for xi in X:
        p_xbym[i]=(1/np.sqrt(2*np.pi*sd*sd))*(np.exp(-((xi-m)**2/(2*sd*sd))))
        i=i+1
    return p_xbym

#computer Posterior Probability
def posteriorProbability(X,likelihoodMForA,likelihoodMForB,priorProbForA,priorProbForB):
    
    posteriorProbB=np.zeros(len(X))
    for i in range(len(X)):
        posteriorProbB[i]=(likelihoodMForB[i]*priorProbForB)/(likelihoodMForB[i]*priorProbForB+likelihoodMForA[i]*priorProbForA)
    
    posteriorProbA=np.zeros(len(X))
    for i in range(len(X)):
        posteriorProbA[i]=1-posteriorProbB[i]

    return(posteriorProbA,posteriorProbB)

#Computer standard deviation
def standardDev(X,meanA,posteriorProbA):
    stdevA=sum([((X[i]-meanA)**2)*posteriorProbA[i] for i in range(len(X))])
    stdevA=stdevA[0]/sum(posteriorProbA)
    stdevA=np.sqrt(stdevA)
    return(stdevA)
#compute mean
def mean(X,posteriorProbA):
    meanA=sum([X[i]*posteriorProbA[i] for i in range(len(X))])
    meanA=meanA[0]/sum(posteriorProbA)
    return(meanA)


#Assume there are two groups and their mean and standard deviation
meanA=2.0
stdevA=1.5
meanB=7.0
stdevB=2.0
#prior probability for Group A
priorProbForA=0.5
#prior probability for Group B
priorProbForB=0.5


# 1st iteration
#Compute the likelihood that each point belong to A i.e P(X/A)
likelihoodMForA=likelihoodMeasureByGaussian(stdevA,meanA,X)

#Compute the likelihood that each point belong to B i.e P(X/B)
likelihoodMForB=likelihoodMeasureByGaussian(stdevB,meanB,X)

#Compute the posterior prob. that each i.e P(A/X) and P(B/X)
(posteriorProbA,posteriorProbB)=posteriorProbability(X,likelihoodMForA,likelihoodMForB,priorProbForA,priorProbForB)

#Recompute the meanA for group A and meanB for group B
meanA=mean(X,posteriorProbA)
meanB=mean(X,posteriorProbB)

#Recompute the standardDev for group A and standardDev for group B
stdevA=standardDev(X,meanA,posteriorProbA)
stdevB=standardDev(X,meanB,posteriorProbB)

#update prior for A and B
priorProbForB=sum(posteriorProbB)/len(posteriorProbB)
priorProbForA=1-priorProbForB

##2nd iteration

#Compute the likelihood that each point belong to A i.e P(X/A)
likelihoodMForA=likelihoodMeasureByGaussian(stdevA,meanA,X)

#Compute the likelihood that each point belong to B i.e P(X/B)
likelihoodMForB=likelihoodMeasureByGaussian(stdevB,meanB,X)

#Compute the posterior prob. that each i.e P(A/X) and P(B/X)
(posteriorProbA,posteriorProbB)=posteriorProbability(X,likelihoodMForA,likelihoodMForB,priorProbForA,priorProbForB)

#Recompute the meanA for group A and meanB for group B
meanA=mean(X,posteriorProbA)
meanB=mean(X,posteriorProbB)

#Recompute the standardDev for group A and standardDev for group B
stdevA=standardDev(X,meanA,posteriorProbA)
stdevB=standardDev(X,meanB,posteriorProbB)

#update prior for A and B
priorProbForB=sum(posteriorProbB)/len(posteriorProbB)
priorProbForA=1-priorProbForB