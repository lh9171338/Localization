import numpy as np
import sympy
from scipy import stats


class KalmanFilter:
    """
    Motion equation: x(n+1) = Fx(n) + Gu(n) + w[n]
    Observation equation: z(n) = Hx(n) + v[n]
    """
    def __init__(self, sys, Q, R):
        """
        Create Kalman Filter
        :param sys: system matrix, [F, G, H]
        :param Q: covariance matrix of w
        :param R: covariance matrix of v
        """
        self.F = sys[0]
        self.G = sys[1]
        self.H = sys[2]
        self.Q = Q
        self.R = R
        self.x = None
        self.P = None

    def init(self, x0, P0):
        """
        Initialize Kalman Filter
        :param x0: initial mean of x
        :param P0: initial covariance matrix of x
        """
        self.x = x0
        self.P = P0

    def update(self, z=None, u=None):
        """
        Update Kalman Filter
        :param z:
        :param u:
        :return:
        """
        F = self.F
        G = self.G
        H = self.H
        Q = self.Q
        R = self.R
        x = self.x
        P = self.P

        # predict
        if u is None:
            x_e = np.matmul(F, x)
        else:
            x_e = np.matmul(F, x) + np.matmul(G, u)
        P_e = np.matmul(np.matmul(F, P), F.T) + Q

        if z is None:
            x = x_e
            P = P_e
        else:
            # calculate Kalman gain
            Denom = np.matmul(np.matmul(H, P_e), H.T) + R
            G = np.matmul(np.matmul(P_e, H.T), np.linalg.inv(Denom))

            # update
            x = x_e + np.matmul(G, z - np.matmul(H, x_e))
            P = np.matmul(np.eye(P.shape[0]) - np.matmul(G, H), P_e)
        self.x = x
        self.P = P
        return x, P


class ExtendedKalmanFilter:
    """
    Motion equation: x(n+1) = f(x(n)) + g(u[n]) + w[n]
    Observation equation: z(n) = h(x(n)) + v[n]
    """
    def __init__(self, sys, Q, R):
        """
        Create Extended Kalman Filter
        :param sys: system matrix, [x, u, f, g, h]
        :param Q: covariance matrix of w
        :param R: covariance matrix of v
        """
        self.symx = sys[0]
        self.symu = sys[1]
        self.symf = sys[2]
        self.symg = sys[3]
        self.symh = sys[4]
        self.Q = Q
        self.R = R
        self.x = None
        self.P = None
        dimx = self.symf.shape[0]
        dimz = self.symh.shape[0]
        self.symF = np.zeros((dimx, dimx), dtype=self.symf.dtype)
        self.symH = np.zeros((dimz, dimx), dtype=self.symf.dtype)
        for i in range(dimx):
            for j in range(dimx):
                df = sympy.diff(self.symf[i], self.symx[j])
                self.symF[i, j] = df
        for i in range(dimz):
            for j in range(dimx):
                dh = sympy.diff(self.symh[i], self.symx[j])
                self.symH[i, j] = dh

    def init(self, x0, P0):
        """
        Initialize Extended Kalman Filter
        :param x0: initial mean of x
        :param P0: initial covariance matrix of x
        """
        self.x = x0
        self.P = P0

    def eval(self, symf, symx, xs):
        symf_ = symf.reshape(-1)
        fs = np.zeros(len(symf_), dtype=xs.dtype)
        subs = {}
        for i in range(symx.shape[0]):
            subs[symx[i]] = xs[i]
        for i in range(len(symf_)):
            fs[i] = symf_[i].evalf(subs=subs)
        fs = fs.reshape(symf.shape)
        return fs

    def update(self, z=None, u=None):
        """
        Update Extended Kalman Filter
        :param z:
        :param u:
        :return:
        """
        symx = self.symx
        symu = self.symu
        symf = self.symf
        symg = self.symg
        symh = self.symh
        symF = self.symF
        symH = self.symH
        Q = self.Q
        R = self.R
        x = self.x
        P = self.P
        dimx = symx.shape[0]
        dimu = symu.shape[0]

        if u is None:
            u = np.zeros(dimu, dtype=x.dtype)

        # predict
        F = self.eval(symF, symx, x)
        H = self.eval(symH, symx, x)
        x_e = self.eval(symf, symx, x) + self.eval(symg, symu, u)
        P_e = np.matmul(np.matmul(F, P), F.T) + Q

        if z is None:
            x = x_e
            P = P_e
        else:
            # calculate Kalman gain
            Denom = np.matmul(np.matmul(H, P_e), H.T) + R
            G = np.matmul(np.matmul(P_e, H.T), np.linalg.inv(Denom))

            # update
            x = x_e + np.matmul(G, z - self.eval(symh, symx, x_e))
            P = np.matmul(np.eye(dimx) - np.matmul(G, H), P_e)
        self.x = x
        self.P = P
        return x, P


class ParticleFilter:
    """
    Motion equation: x(n+1) = f(x(n)) + g(u[n]) + w[n]
    Observation equation: z(n) = h(x(n)) + v[n]
    """
    def __init__(self, sys, Q, R):
        """
        Create Particle Filter
        :param sys: system matrix, [f, g, h]
        :param Q: covariance matrix of w
        :param R: covariance matrix of v
        """
        self.f = sys[0]
        self.g = sys[1]
        self.h = sys[2]
        self.sqrtQ = np.sqrt(Q)
        self.R = R
        self.numParticles = None
        self.x = None
        self.particles = None

    def init(self, x0, P0, numParticles):
        """
        Initialize Particle Filter
        :param x0: initial mean of x
        :param P0: initial covariance matrix of x
        :param numParticles: number of particles
        """
        self.x = x0
        self.numParticles = numParticles
        particles = x0[None] + np.matmul(np.random.randn(numParticles, x0.shape[0]), np.sqrt(P0))
        self.particles = particles
        return particles

    def update(self, z, u=None):
        """
        Update Particle Filter
        :param z:
        :param u:
        :return:
        """
        f = self.f
        g = self.g
        h = self.h
        sqrtQ = self.sqrtQ
        R = self.R
        x = self.x
        numParticles = self.numParticles
        particles = self.particles

        # predict
        if u is None:
            newParticles = f(particles.T).T + np.matmul(np.random.randn(particles.shape[0], particles.shape[1]), sqrtQ)
        else:
            newParticles = f(particles.T).T + g(u).T + np.matmul(np.random.randn(particles.shape[0], particles.shape[1]), sqrtQ)

        # update
        zs = h(newParticles.T).T

        # weight
        weights = stats.multivariate_normal.logpdf(zs, z, R)
        weights -= weights.max()
        weights = np.exp(weights)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights[:] = 1.0 / numParticles

        # resample
        indices = np.random.choice(numParticles, size=numParticles, p=weights)
        particles = newParticles[indices]
        x = particles.mean(axis=0)
        self.x = x
        self.particles = particles
        return x, particles


