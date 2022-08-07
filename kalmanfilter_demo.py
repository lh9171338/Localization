import numpy as np
import matplotlib.pyplot as plt
from filter import KalmanFilter, ParticleFilter


if __name__ == '__main__':
    # 生成模型参数
    dimx = 4
    dimz = 2
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    G = np.zeros((4, 1), dtype=np.float32)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    Q = 10 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    R = 20 * np.eye(dimz, dtype=np.float32)

    # 生成真实数据
    np.random.seed(0)
    N = 100
    v = np.array([15, 5], dtype=np.float32)
    x0 = np.array([2, 2], dtype=np.float32)
    ts = np.arange(N)[..., None]
    xs = np.matmul(ts, v[None]) + x0[None]
    xs = np.vstack((xs, xs))
    N *= 2
    noise = np.matmul(np.random.randn(xs.shape[0], xs.shape[1]), np.sqrt(R))
    zs = xs + noise

    # 卡尔曼滤波
    x0 = np.array([zs[0, 0], zs[0, 1], 0, 0], dtype=np.float32)
    P0 = np.eye(dimx, dtype=np.float32)
    kalmanFilter = KalmanFilter(sys=[F, G, H], Q=Q, R=R)
    kalmanFilter.init(x0=x0, P0=P0)
    xs1 = np.zeros_like(xs)
    xs1[0] = x0[:2]
    for i in range(1, N):
        x, _ = kalmanFilter.update(z=zs[i])
        xs1[i] = x[:2]

    # 粒子滤波
    numParticles = 1000
    P0 = 100 * np.eye(dimx, dtype=np.float32)
    f = lambda x: np.matmul(F, x)
    g = lambda x: np.matmul(G, x)
    h = lambda x: np.matmul(H, x)
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
    pts = np.vstack((xs, zs, xs1, xs2, particlesList.reshape(-1, 2)))
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
        plt.plot(zs[:i + 1, 0], zs[:i + 1, 1], '-g', linewidth=1)
        plt.plot(xs1[:i + 1, 0], xs1[:i + 1, 1], '-b', linewidth=1)
        plt.plot(xs2[:i + 1, 0], xs2[:i + 1, 1], '-m', linewidth=1)
        plt.legend(['Ground truth', 'Measured value', 'Kalman filter', 'Particle filter'], loc='upper left', fontsize=10)
        plt.ion()
        plt.pause(0.1)
        plt.clf()
