"""Deep deterministic policy gradient (DDPG) in plain numpy (actor-critic, replay, target nets)."""
import numpy as np


class MLP:
    """Fully connected network with ReLU hidden layers and linear or tanh output."""

    def __init__(self, sizes, out_tanh, rng, final_scale=3e-3):
        self.W, self.b = [], []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            lim = final_scale if i == len(sizes) - 2 else 1.0 / np.sqrt(n_in)
            self.W.append(rng.uniform(-lim, lim, (n_in, n_out)))
            self.b.append(rng.uniform(-lim, lim, n_out))
        self.out_tanh = out_tanh

    def params(self):
        return self.W + self.b

    def forward(self, x):
        self._h = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else (np.tanh(z) if self.out_tanh else z)
            self._h.append(h)
        return h

    def backward(self, g_out):
        """Gradients of sum(g_out * output) w.r.t. weights, biases and input."""
        g = g_out * (1.0 - self._h[-1] ** 2) if self.out_tanh else g_out
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = self._h[i].T @ g
            gb[i] = g.sum(axis=0)
            g = g @ self.W[i].T
            if i > 0:
                g = g * (self._h[i] > 0)
        return gW + gb, g

    def copy_from(self, other, tau=1.0):
        for a, b in zip(self.params(), other.params()):
            a *= (1.0 - tau)
            a += tau * b


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for x, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            x -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


class Replay:
    def __init__(self, size, obs_dim, rng):
        self.s = np.zeros((size, obs_dim))
        self.a = np.zeros((size, 1))
        self.r = np.zeros((size, 1))
        self.s2 = np.zeros((size, obs_dim))
        self.d = np.zeros((size, 1))
        self.size, self.n, self.i, self.rng = size, 0, 0, rng

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch):
        k = self.rng.integers(0, self.n, batch)
        return self.s[k], self.a[k], self.r[k], self.s2[k], self.d[k]


class DDPG:
    """Actor mu(x | theta_mu) in [-1, 1] and critic Q(x, a | theta_Q)."""

    def __init__(self, obs_dim, hidden=(64, 64), gamma=0.99, tau=0.005,
                 lr_actor=1e-4, lr_critic=1e-3, buffer=200_000, batch=128, seed=0):
        self.rng = np.random.default_rng(seed)
        self.actor = MLP([obs_dim, *hidden, 1], True, self.rng)
        self.critic = MLP([obs_dim + 1, *hidden, 1], False, self.rng)
        self.actor_t = MLP([obs_dim, *hidden, 1], True, self.rng)
        self.critic_t = MLP([obs_dim + 1, *hidden, 1], False, self.rng)
        self.actor_t.copy_from(self.actor)
        self.critic_t.copy_from(self.critic)
        self.opt_a = Adam(self.actor.params(), lr_actor)
        self.opt_c = Adam(self.critic.params(), lr_critic)
        self.buf = Replay(buffer, obs_dim, self.rng)
        self.gamma, self.tau, self.batch = gamma, tau, batch

    def act(self, s, noise=0.0):
        a = self.actor.forward(s[None, :])[0, 0]
        if noise > 0:
            a += noise * self.rng.standard_normal()
        return float(np.clip(a, -1.0, 1.0))

    def update(self, train_actor=True):
        s, a, r, s2, d = self.buf.sample(self.batch)
        a2 = self.actor_t.forward(s2)
        y = r + self.gamma * (1 - d) * self.critic_t.forward(np.hstack([s2, a2]))
        # critic: minimise mean (Q - y)^2
        q = self.critic.forward(np.hstack([s, a]))
        grads, _ = self.critic.backward(2.0 * (q - y) / len(q))
        self.opt_c.step(grads)
        self.critic_t.copy_from(self.critic, self.tau)
        if not train_actor:
            return float(np.mean((q - y) ** 2))
        # actor: maximise mean Q(s, mu(s)) via dQ/da
        mu = self.actor.forward(s)
        self.critic.forward(np.hstack([s, mu]))
        _, g_in = self.critic.backward(-np.ones((len(s), 1)) / len(s))
        grads, _ = self.actor.backward(g_in[:, -1:])
        self.opt_a.step(grads)
        self.actor_t.copy_from(self.actor, self.tau)
        return float(np.mean((q - y) ** 2))

    def state_dict(self):
        return {f"actor_{i}": x.copy() for i, x in enumerate(self.actor.params())}

    def load_actor(self, state):
        for i, x in enumerate(self.actor.params()):
            x[...] = state[f"actor_{i}"]
