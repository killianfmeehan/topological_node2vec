# This file is released under MIT license, see file LICENSE.
# Author(s):       Theo Lacombe
#
# Copyright (C) 2022 Université Gustave Eiffel

import numpy as np
import ot
# TODO change ot dep to scipy. Only used for matrix building now.


def aprox(x, mode_divergence, eps=None, cdiag=None):
    """
    Perform the anisotropic proximal operation (transformation of the LogSumExp output),
    used in update of Sinkhorn divergence.
    Only implemented for some specific modes (choice of marginal divergence).

    :param x: output of the LogSumExp in Sinkhorn update.
    :param mode_divergence: "balanced" --> identity, "KL" --> scaling, "TV" --> cut-off, "boundary" --> min with cost.
    :param eps: Regularization parameter
                (only needed with KL and boundary, perhaps could be turned into a "hyperparams" field.)
    :param cdiag: matrix encoding the list of costs of throwing the elements of x : 'c(x_i,thediag)',
                  where thediag is the boundary of our space.
    :return: output of LSE after performing aprox.
    """
    if mode_divergence == "balanced":
        return x
    elif mode_divergence == "KL":
        return x / (1 + eps)
    elif mode_divergence == "TV":
        return np.maximum(-1, np.minimum(x, 1))
    elif mode_divergence == "boundary":
        return np.maximum(-cdiag, x + eps * np.log(cdiag))
    else:
        raise ValueError("mode %s is not available for aprox" % mode_divergence)


def varphi_star(q, mode_divergence, cdiag):
    if mode_divergence == "balanced":
        return q
    elif mode_divergence == "KL":
        return np.exp(q) - 1
    elif mode_divergence == "TV":
        return np.maximum(-1, q)
    elif mode_divergence == "boundary":
        return np.maximum(-1, np.divide(q, cdiag))
    else:
        raise ValueError("mode %s is not available for varphi_star" % mode_divergence)


def sinkhorn_map(f, a,
                 mode_divergence, mode_homogeneity,
                 corrected_marginals,
                 eps, cost_matrix, cdiag,
                 stab, b=None):
    """
        :param f: current eval of dual potential, same shape as mu (say n)
        :param a: distribution of weights of the measure (histogram)
        :param mode_divergence: Mode divergence to the marginals ("balanced", "KL", or "TV").
        :param mode_homogeneity: Are we in std model (no homogeneity) or homogene (L or G) one?
        :param corrected_marginals: Should we use the corrected version of the marginals.
        :param eps: smoothing param
        :param cost_matrix: cost matrix, size (n x m)
        :param cdiag: distance to the boundary (throwing cost).
        :param stab: boolean, should we use stabilized log-sum-exp version of the implementation.
        :param b: the second measure, useful for some homogene model only.

        Note: warning, will need a C.T when applying on g
    """
    h = (f[:, None] - cost_matrix) / eps
    # We Run Sinkhorn algorithm for renormalized version of the measure.
    # Idea: Sinkhorn loop is processed with a * X/Y, where X is the renormalization applied on exp(f+g-c)
    #       and Y is the normalization applied to the marginals.
    # Therefore, if we apply (harmonic or geometric) homogeneity, X = sqrt(m(a)*m(b))
    #        and if we apply marginal correction, Y = sqrt(m(a) / m(b))
    # Hence,  X / Y = 1/np.sum(a)  (or np.sum(b)) for the matter.

    if mode_homogeneity == "std":
        a_norm = a
    else:
        if corrected_marginals:
            a_norm = a / np.sum(a)
        else:
            a_norm = a / np.sqrt(np.sum(a) * np.sum(b))

    # Computation of the LogSumExp
    if stab:
        h_star = np.max(h, axis=0)
        tmp = (np.exp(h - h_star).T).dot(a_norm)
        res = - eps * (h_star + np.log(tmp))
    else:
        tmp = (np.exp(h).T).dot(a_norm)
        res = - eps * np.log(tmp)
    # Apply the aprox operator and return.
    return -aprox(-res, mode_divergence=mode_divergence, eps=eps, cdiag=cdiag)


def update(first_potential, second_potential,
           first_weights, second_weights,
           mode_divergence, mode_homogeneity,
           corrected_marginals,
           eps, C, cdiag1, cdiag2,
           stab):
    """
    Update of the dual potential (iteration of the Sinkhorn algorithm) :
        f_{t+1} = sinkhorn_map(g_t, **hyperparams).
        g_{t+1} = sinkhorn_nap(f_{t+1}, **hyperparams)   # note that f_t is not used.

    :param first_potential: the first potential. It is not used (yet), but put there for symmetry (should be removed?).
    :param second_potential: the potential from which we perform the update.
    :param first_weights: the weight of the measure corresponding to first_potential (f <--> alpha).
    :param second_weights: the weights of the measure corresopnding to second_potential (g <--> beta).
    :param mode_divergence: divergence to marginals, "balanced", "KL", "TV" or "boundary".
    :param mode_homogeneity: "std" (non-homogene), "harmonic" or "geometric".
    :param corrected_marginals: Boolean, should we apply the marginal renormalization.
    :param eps: the entropic regularization parameter.
    :param C: Distance matrix between the points in the "first" measure and the "second" one. Beware of transpositions.
    :param cdiag1: distance to diagonal of elements in the "first" measure.
    :param cdiag2: distance to diagonal of elements in the "second" measure.
    :param stab: used stabilized version of LogSumExp (Sinkhorn) update (should be True).
    """
    new_f = sinkhorn_map(second_potential, second_weights,
                         mode_divergence=mode_divergence,
                         mode_homogeneity=mode_homogeneity,
                         corrected_marginals=corrected_marginals,
                         eps=eps,
                         cost_matrix=C.T, cdiag=cdiag1,
                         stab=stab, b=first_weights)
    new_g = sinkhorn_map(new_f, first_weights,
                         mode_divergence=mode_divergence,
                         mode_homogeneity=mode_homogeneity,
                         corrected_marginals=corrected_marginals,
                         eps=eps, cost_matrix=C, cdiag=cdiag2,
                         stab=stab, b=second_weights)
    return new_f, new_g


def estim_dual(first_potential, second_potential,
               first_weights, second_weights,
               mode_divergence, mode_homogeneity,
               corrected_marginals,
               eps, C, cdiag1, cdiag2,
               withentropy):
    """
    Dual estimation
    <-varphi_star(-f) , a> + <-varphi_star(-g), b> - eps < exp((f(+)g-C)/eps)-1, a (x) b>
    """
    # Masses of the measures
    ma, mb = np.sum(first_weights), np.sum(second_weights)
    # Geometric mean of the masses
    m_g = np.sqrt(ma * mb)
    # (Inverted) harmonic mean of the masses
    m_h_inv = 0.5 * (1 / ma + 1 / mb)
    # Sqrted ratio of the masses
    r_ab = np.sqrt(ma / mb)

    # Term corresponding to transport + marginal error
    if corrected_marginals:
        z = np.dot(-varphi_star(-first_potential, mode_divergence, cdiag=cdiag1), first_weights * r_ab) \
            + np.dot(-varphi_star(-second_potential, mode_divergence, cdiag=cdiag2), second_weights / r_ab)
    else:
        z = np.dot(-varphi_star(-first_potential, mode_divergence, cdiag=cdiag1), first_weights) \
            + np.dot(-varphi_star(-second_potential, mode_divergence, cdiag=cdiag2), second_weights)
    if not withentropy:
        return z
    # Term corresponding to the entropic regularization.
    else:
        tmp1 = (np.add(first_potential[:, None], second_potential[None, :]) - C) / eps

        if mode_homogeneity == 'std':
            tmp2 = np.exp(tmp1) - 1
            tmp2bis = np.multiply(first_weights[:, None], second_weights[None, :])
        elif mode_homogeneity == 'harmonic':
            tmp2 = np.exp(tmp1) / m_g - m_h_inv
            tmp2bis = np.multiply(first_weights[:, None], second_weights[None, :])
        elif mode_homogeneity == "geometric":
            m_g = np.sqrt(np.sum(first_weights) * np.sum(second_weights))
            tmp2 = np.exp(tmp1) - 1
            tmp2bis = np.multiply(first_weights[:, None], second_weights[None, :]) / m_g
        else:
            raise ValueError("mode_homogeneity %s unknown" % mode_homogeneity)

        # Dot product entropic term
        tmp3 = np.sum(np.multiply(tmp2, tmp2bis))

        # Summing everything to get the dual.
        return z - eps * tmp3


def hurot(X, Y, a, b,
       mode_divergence,
       mode_homogeneity,
       corrected_marginals,
       eps,
       nb_step=10000, crit=0.0001,
       stab=True, verbose=1, init="unif", withentropy=True):
    """
    Iterate Sinkhorn loop until convergence, between two measures
    $$alpha = sum_i a_i delta_{X_i}$$
    and
    $$beta = sum_j b_j delta_{Y_j}$$

    :param X: Input point cloud (np.array, size n x d)
    :param Y: Input point cloud (np.array, size n x d)
    :param a: Input weight distribution for the first measure (np.array, >= 0, size n)
    :param b: Input weight distribution for the second measure (np.array, >= 0, size n)
    :param mode_divergence: Choice Phi-divergence for marginal error
    :param mode_homogeneity: Shall we use homogeneous OT model?
    :param corrected_marginals: Should we use the corrected version of the marginals (should be False).
    :param eps: Regularization parameter for the entropic smoothing.
    :param nb_step: Maximal number of step in the Sinkhorn Loop.
    :param crit: Stopping criterion : relative error of change in the dual estimation.
                    WARNING : when the pbm is not homogeneous, yield different convergence rates!
    :param stab: Should we use log-sum-exp stabilisation trick.
    :param verbose: 0: silent, 1: ok, 2: verbose.
    :param init: Mode to initialize the dual potentials (default: unif).
    :param withentropy: Do we keep entropic term in eval of dual (should be True).

    :return: P,f,g,e : Final transport plan, dual potentials, and objective value.
    """
    C = ot.utils.dist(X, Y)
    if mode_divergence == "boundary":
        cdiag1 = squared_dist_to_diag(X)
        cdiag2 = squared_dist_to_diag(Y)
        a_weighted = np.multiply(cdiag1, a)
        b_weighted = np.multiply(cdiag2, b)
    else:
        cdiag1, cdiag2 = None, None
        a_weighted = a
        b_weighted = b

    if init == "unif":
        f, g = np.ones(len(X)), np.ones(len(Y))
    elif init == "rand":
        f, g = np.random.rand(len(X)), np.random.rand(len(Y))
    else:
        print("Unknown init. Pick rand.")
        f, g = np.random.rand(len(X)), np.random.rand(len(Y))

    e = -np.inf
    converged = False
    for t in range(nb_step):
        f, g = update(first_potential=f, second_potential=g,
                      first_weights=a_weighted, second_weights=b_weighted,
                      mode_divergence=mode_divergence,
                      mode_homogeneity=mode_homogeneity,
                      corrected_marginals=corrected_marginals,
                      eps=eps, C=C,
                      cdiag1=cdiag1, cdiag2=cdiag2,
                      stab=stab)

        new_e = estim_dual(first_potential=f, second_potential=g,
                           first_weights=a_weighted, second_weights=b_weighted,
                           mode_divergence=mode_divergence,
                           mode_homogeneity=mode_homogeneity,
                           corrected_marginals=corrected_marginals,
                           eps=eps, C=C,
                           cdiag1=cdiag1, cdiag2=cdiag2,
                           withentropy=withentropy)

        if abs((new_e - e) / new_e) < crit:
            e = new_e
            converged = True
            if verbose >= 1:
                print("converged at step t =", t)
            break
        else:
            e = new_e

    if not converged:
        if crit > 0:
            print("Warning, nb step = %s was not sufficient to have convergence with relative criterion %s"
                  % (nb_step, crit))
        else:
            if verbose >= 1:
                print("(note: convergence criterion is 0. Ran the algorithm for %s steps)" % nb_step)
    P = get_P(f, g, a_weighted, b_weighted, eps, C, mode_homogeneity=mode_homogeneity)

    return P, f, g, e


def sk_div(X, Y, a, b,
           mode_divergence,
           mode_homogeneity,
           corrected_marginals,
           eps,
           nb_step=1000, crit=0.0001,
           stab=True, verbose=1, init="unif"):
    """
    Compute the Sinkhorn divergence between (possibly) unbalanced measures.

    :param X: Point cloud (n x d).
    :param Y: Point cloud (n x d).
    :param a: First distrib of weights (n, >= 0)
    :param b: Second ditrib of weights (n, >= 0)
    :param mode_divergence: Divergence to the marginal.
    :param mode_homogeneity: which type of entropic reg do we use (std, harmonic, geometric...)
    :param corrected_marginals: Should we use the normal or corrected version for the marginal.
    :param eps: Regularization parameter for entropic smoothing
    :param nb_step: Maximal number of step to run sinkhorn.
    :param crit: Stopping criterion.
    :param stab: Should we use LSE stabilization.
    :param verbose: Verobisity level.
    :param init: Initialization of dual potentials.

    :return: value of sinkhorn divergence.
    """
    # Cost mu-->nu
    xy = hurot(X, Y, a, b,
               mode_divergence=mode_divergence, mode_homogeneity=mode_homogeneity,
               corrected_marginals=corrected_marginals,
               eps=eps, nb_step=nb_step, crit=crit,
               stab=stab, verbose=verbose, init=init, withentropy=True)[-1]
    # Cost mu-->mu (self entropy)
    xx = hurot(X, X, a, a,
               mode_divergence=mode_divergence, mode_homogeneity=mode_homogeneity,
               corrected_marginals=corrected_marginals,
               eps=eps, nb_step=nb_step, crit=crit,
               stab=stab, verbose=verbose, init=init, withentropy=True)[-1]
    # Cost nu-->nu (self entropy)
    yy = hurot(Y, Y, b, b,
               mode_divergence=mode_divergence, mode_homogeneity=mode_homogeneity,
               corrected_marginals=corrected_marginals,
               eps=eps, nb_step=nb_step, crit=crit,
               stab=stab, verbose=verbose, init=init, withentropy=True)[-1]

    cost_brut = xy - 0.5 * xx - 0.5 * yy

    if mode_divergence == "boundary":
        cdiag1 = squared_dist_to_diag(X)
        cdiag2 = squared_dist_to_diag(Y)
        a_weighted = np.multiply(cdiag1, a)
        b_weighted = np.multiply(cdiag2, b)
    else:
        a_weighted = a
        b_weighted = b

    if mode_homogeneity == "std":
        mass_biais = 0.5 * eps * (np.sum(a_weighted) - np.sum(b_weighted)) ** 2
    elif mode_homogeneity == "geometric":
        mass_biais = 0.5 * eps * (np.sqrt(np.sum(a_weighted)) - np.sqrt(np.sum(b_weighted))) ** 2
    elif mode_homogeneity == "harmonic":
        mass_biais = 0
    else:
        raise ValueError("mode_homogeneity (%s) unknown." % mode_homogeneity)

    return cost_brut + mass_biais


def hurot_tda(X, Y, eps, mode_homogeneity='harmonic', verbose=1):
    """
    Just a simple wrapper to use hurot for two persistence diagrams. 
    We made the function as simple to use as possible, so it may miss some parameters of interest
    (e.g. stopping criterion for sinkhorn loop). 
    
    :note: Points are represented with multiplicity 1. 
    
    :param X: the support of the first diagram (coordinates in R^2). Shape n x 2.
    :param Y: the support of the second diagram (coordinates in R^2). Shape m x 2. 
    :param eps: entropic regularization parameter. 
    :param verbose: verbosity level. 
    
    :returns: P, e, the entropic transport plan and the corresponding cost. 
    
    :note: P represent the *off diagonal* mass transported from x_i to y_j. What is not transported should be understood as 
            being sent onto the diagonal (for the matter of computing the barycentric map). 
    """
    n = X.shape[0]
    m = Y.shape[0]
    a = np.ones(n)
    b = np.ones(m)
    
    P, f, g, e = hurot(X = X, Y = Y, a = a, b = b, 
                       eps = eps, 
                       mode_divergence='boundary', mode_homogeneity=mode_homogeneity,
                       corrected_marginals=False, 
                       verbose=verbose)
    
    return P, e


def _correction_term(P, P_self, X, eps):
    """
    A correction term to account for the fact that with the HUROT model, 
    the entropic regularization term *does* depend on the points location. 
    Indeed, it considers "weighted" measures, where the weigth is the distance
    to the diagonal. This is the price for using a spatially varying entropy. 
    
    This is the gradient (wrt X) of 
    
    KL(P1 | a x b / m(a)) - 0.5 * KL(P2  | a x a / m(a))
    
    (with a, b = weighted measures wrt distance to the diagonal)
    """
    DX = X[:,1] - X[:,0]   # the vector of "distances to the diagonal"
    
    #print("Diff mass tot:", np.sum(P) - np.sum(P_self))
    #print("Diff mass at i:", np.sum(P, axis=1) - np.sum(P_self, axis=1))
    
    Z = (0.5 * (np.sum(P) - np.sum(P_self)) / _perstot(X)) * DX - 2 * np.divide((np.sum(P, axis=1) - np.sum(P_self, axis=1)), DX)
    
    return eps * np.array([-Z, Z]).T


def barycentric_map_tda(P, X, Y):
    """
    A quick implementation (can be improved for sure) to turn P into the "naive" entropic barycentric map. 
    
    :note: leverage the fact that the mass/multiplicity of the points is 1, so we do not need to normalize. 
    
    TODO: understand if there is some more adapted notion of barycentric map. It is likely that there is. 
    
    :param P: the entropic homogeneous transport plan P (n x m). 
    :param X: the support of the source measure (n x 2). 
    :param Y: the support of the target measure (m x 2).
    :param with_correction: do we include the gradient with respect to the entropic regularization term.
    
    :returns: an array of size (n x 2) which tells you T(x) for each x in X. 
    """
    # off-diagonal barycenter
    T_off = P.dot(Y)
    # remaining mass (matched to the diagonal)
    R = (1 - np.sum(P, axis=1))[:,None]
    # barycenter for projection on diagonals of X
    T_diag = np.multiply(R, proj_on_diag(X))
    
    return T_off + T_diag


def sinkhorn_gradient_tda(X, Y, eps, with_correction=False):
    """
    A quick implementation for the Sinkhorn barycentric map. Gradient wrt the locations X. 
    
    Possibility to include a correction term, that account for the gradient of the KL terms. 
    If set to False, only account for the gradient wrt the cost. 
    """
    # Plan from alpha to beta
    P1, dist1 = hurot_tda(X, Y, eps=eps, verbose=0)
    # self alpha-alpha 
    P2, dist2 = hurot_tda(X, X, eps=eps, verbose=0)
    # self beta-beta (not used in the gradient but useful for the loss. Should we remove it?
    P3, dist3 = hurot_tda(Y, Y, eps=eps, verbose=0)
    
    # Compute the gradients
    grad1 = X - barycentric_map_tda(P1, X, Y)
    grad2 = 2*X - 2*barycentric_map_tda(P2, X, X)
    ### grad3 = Y - barycentric_map_tda(P3, Y, Y)  # Should not be needed
    # The Sinkhorn cost
    S = dist1-1/2*dist2-1/2*dist3
    # The Sinkhorn gradient (without the KL contribution)
    gradS = grad1-1/2*grad2
    
    # Correction term <=> compute the gradient of the KL term. 
    if with_correction:
        corr = _correction_term(P1, P2, X, eps)
        gradS = gradS + corr
    
    return gradS, S

    
###########################
### Complementary utils ###
###########################
def proj_on_diag(X):
    Z = (X[:,1] + X[:,0])/2
    return np.array([Z,Z]).T


def grad_to_diag(X):
    Z = (X[:,1] - X[:,0])
    return -np.array([Z,Z]).T


def squared_dist_to_diag(X):
    """
    :param X: (n x 2) array encoding the points of a persistent diagram.
    :returns: (n) array encoding the (respective orthogonal) distances of the points to the diagonal
    """
    return (X[:, 1] - X[:, 0])**2 / 2


def _perstot(X):
    return np.sum(squared_dist_to_diag(X))


def get_P(f, g, a, b, eps, C, mode_homogeneity):
    """
    Given two dual potentials and mass distributions (+ cost matrix and regularization param), returns the
    transport plan according to the primal-dual correspondence.
    """
    tmp1 = (np.add(f[:, None], g[None, :]) - C) / eps
    tmp2 = np.multiply(a[:, None], b[None, :])

    if mode_homogeneity == "std":
        return np.multiply(np.exp(tmp1), tmp2)
    elif mode_homogeneity == "geometric":
        return np.multiply(np.exp(tmp1), tmp2) / np.sqrt(np.sum(a) * np.sum(b))
    elif mode_homogeneity == "harmonic":
        return np.multiply(np.exp(tmp1), tmp2) / np.sqrt(np.sum(a) * np.sum(b))
    else:
        raise ValueError("mode_homogeneity (%s) unknown.")


def MMD(X, Y, a, b):
    """
    Quick implementation of MMD between two measures.

    :param X: Point cloud (locations).
    :param Y: Point cloud (locations).
    :param a: Weights (>= 0).
    :param b: Weights (>= 0).

    :return: Value of MMD for the Euclidean cost.
    TODO: implement with general cost.
    """
    Cxy = ot.utils.dist(X, Y)
    Cxx = ot.utils.dist(X, X)
    Cyy = ot.utils.dist(Y, Y)
    r1 = np.dot(a, np.dot(Cxy, b))
    r2 = np.dot(a, np.dot(Cxx, a))
    r3 = np.dot(b, np.dot(Cyy, b))
    return r1 - 0.5 * r2 - 0.5 * r3


#####################
### Drawing utils ###
#####################

def plot_transport_plan(ax, xs, xt, P, thr=1e-8):
    """
    An utils function to plot a transport plan between two measures.

    :param ax: the matplotlib ax on a fig.
    :param xs: points in the first measure (in R^2)
    :param xt: points in the second measure (in R^2)
    :param P: the transport plan we want to plot between xt and xs.
    :param thr: only draw edges in P above this threshold.
    """
    mx = np.max(P)
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if P[i, j] / mx > thr:
                ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], alpha=P[i, j] / mx, color='k')
