import numpy as np
import matplotlib.pyplot as plt
import sympy
from filter import ExtendedKalmanFilter, ParticleFilter


if __name__ == '__main__':
    # 生成模型参数
    s1 = np.array([0, 0], dtype=np.float32)
    s2 = np.array([200, 0], dtype=np.float32)
    s3 = np.array([0, 200], dtype=np.float32)
    dimx = 4
    dimz = 3
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    G = np.zeros((dimx, 1), dtype=np.float32)
    Q = 1 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    R = 100 * np.eye(dimz, dtype=np.float32)

    # 生成真实数据
    np.random.seed(0)
    N = 100
    v = np.array([15, 5], dtype=np.float32)
    x0 = np.array([100, 100], dtype=np.float32)
    ts = np.arange(N)[..., None]
    xs = np.matmul(ts, v[None]) + x0[None]
    xs = np.vstack((xs, xs))
    N *= 2
    zs1 = np.sqrt(((xs - s1[None]) ** 2).sum(axis=1))
    zs2 = np.sqrt(((xs - s2[None]) ** 2).sum(axis=1))
    zs3 = np.sqrt(((xs - s3[None]) ** 2).sum(axis=1))
    zs = np.hstack((zs1[..., None], zs2[..., None], zs3[..., None]))
    noise = np.matmul(np.random.randn(zs.shape[0], zs.shape[1]), np.sqrt(R))
    zs += noise

    # 扩展卡尔曼滤波
    x0 = np.zeros(dimx, dtype=np.float32)
    P0 = np.eye(dimx, dtype=np.float32)
    symx = np.asarray(list(sympy.symbols('x1 x2 x3 x4')))
    symu = np.array([sympy.symbols('u')])
    symf = np.matmul(F, symx)
    symg = np.matmul(G, symu)
    symh1 = sympy.sqrt((symx[0] - s1[0]) ** 2 + (symx[1] - s1[1]) ** 2)
    symh2 = sympy.sqrt((symx[0] - s2[0]) ** 2 + (symx[1] - s2[1]) ** 2)
    symh3 = sympy.sqrt((symx[0] - s3[0]) ** 2 + (symx[1] - s3[1]) ** 2)
    symh = np.array([symh1, symh2, symh3])
    extendedKalmanFilter = ExtendedKalmanFilter(sys=[symx, symu, symf, symg, symh], Q=Q, R=R)
    extendedKalmanFilter.init(x0=x0, P0=P0)
    xs1 = np.zeros_like(xs)
    xs1[0] = x0[:2]
    for i in range(1, N):
        x, _ = extendedKalmanFilter.update(z=zs[i])
        xs1[i] = x[:2]

    # 粒子滤波
    numParticles = 1000
    P0 = 100 * np.eye(dimx, dtype=np.float32)
    f = lambda x: np.matmul(F, x)
    g = lambda x: np.matmul(G, x)
    h = lambda x: np.vstack((np.sqrt(((x[:2] - s1[..., None]) ** 2).sum(axis=0))[None],
                             np.sqrt(((x[:2] - s2[..., None]) ** 2).sum(axis=0))[None],
                             np.sqrt(((x[:2] - s3[..., None]) ** 2).sum(axis=0))[None]))
    particleFilter = ParticleFilter(sys=[f, g, h], Q=Q, R=R)
    particles = particleFilter.init(x0=x0, P0=P0, numParticles=numParticles)
    xs2 = np.zeros_like(xs)
    particlesList = np.zeros((N, numParticles, 2), dtype=np.float32)
    xs2[0] = x0[:2]
    particlesList[0] = particles[:, :2]
    for i in range(1, N):
        x, particles = particleFilter.update(z=zs[i])
        xs2[i] = x[:2]
        particlesList[i] = particles[:, :2]

    # 绘制结果
    pts = np.vstack((xs, xs1, xs2, particlesList.reshape(-1, 2)))
    xlim = [pts[:, 0].min(), pts[:, 0].max()]
    ylim = [pts[:, 1].min(), pts[:, 1].max()]
    width = int(xlim[1] - xlim[0])
    height = int(ylim[1] - ylim[0])
    plt.figure()
    for i in range(N):
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.scatter(particlesList[i, :100, 0], particlesList[i, :100, 1], color='m', s=5)
        plt.plot(xs[:i + 1, 0], xs[:i + 1, 1], '-r', linewidth=1)
        plt.plot(xs1[:i + 1, 0], xs1[:i + 1, 1], '-b', linewidth=1)
        plt.plot(xs2[:i + 1, 0], xs2[:i + 1, 1], '-m', linewidth=1)
        plt.legend(['Ground truth', 'Kalman filter', 'Particle filter'], loc='upper left', fontsize=10)
        plt.ion()
        plt.pause(0.1)
        plt.clf()
