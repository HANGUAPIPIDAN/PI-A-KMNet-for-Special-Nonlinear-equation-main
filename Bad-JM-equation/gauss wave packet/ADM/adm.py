
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------- 参数 ----------------
A      = 0.3       # r0 幅值
W      = 0.25      # r0 宽度: r0 = A*exp(-W x^2)
L_BOX  = 30.0      # 计算域 [-15, 15]
N_X    = 1600      # 网格点数 (dx = 0.01875, Nyquist ~ 84 >> K_CUT)
K_CUT  = 3.5       # 带限截断波数
T_END  = 0.3       # 终止时间
H      = 2.5e-4    # ADM 步长 (1200 段)
M_TERM = 5         # 每段级数项数
N_FRAMES = 121     # 记录帧数 (含 t=0, 每帧间隔 = 0.0025)
# --------------------------------------


def make_grid(n=N_X, L=L_BOX):
    xs = np.linspace(-L / 2, L / 2, n, endpoint=False)
    kf = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    return xs, kf


def ddx(f, kf):
    return np.fft.ifft(1j * kf * np.fft.fft(f))


def d3x(f, kf):
    return np.fft.ifft((1j * kf) ** 3 * np.fft.fft(f))


def bandlimit(f, kf):
    """带限滤波: 抑制 |k| > K_CUT 的模式 (超高斯型, 截断边缘平滑)"""
    return np.fft.ifft(np.fft.fft(f) * np.exp(-(np.abs(kf) / K_CUT) ** 14))


def init_ics(n=N_X, L=L_BOX):
    xs, kf = make_grid(n, L)
    r = bandlimit(A * np.exp(-W * xs ** 2), kf).astype(complex)
    q = np.zeros(n, dtype=complex)
    return xs, kf, r, q


def adm_step(r, q, kf, h, m=M_TERM):
    """一段步长 h 的分段 ADM 递推 (等价于该段的 Adomian/Taylor 级数)"""
    rl = [r.copy()]; ql = [q.copy()]
    for n in range(1, m):
        a_rrx = np.zeros_like(r); a_rqx = np.zeros_like(r); a_qrx = np.zeros_like(r)
        for i in range(n):                       # Adomian 多项式: 卷积求和
            j = n - 1 - i
            dr_j = ddx(rl[j], kf); dq_j = ddx(ql[j], kf)
            a_rrx += rl[i] * dr_j
            a_rqx += rl[i] * dq_j
            a_qrx += ql[i] * dr_j
        r_n = (1.0 / n) * (4.0 * ddx(ql[n - 1], kf) + 6.0 * a_rrx)
        q_n = (1.0 / n) * (d3x(rl[n - 1], kf) + 2.0 * a_rqx + 4.0 * a_qrx)
        rl.append(r_n); ql.append(q_n)
    out_r = rl[0].copy(); out_q = ql[0].copy()
    hp = h
    for n in range(1, m):                        # r(t+h) = sum_n h^n * r_n
        out_r += hp * rl[n]; out_q += hp * ql[n]
        hp *= h
    return bandlimit(out_r, kf), bandlimit(out_q, kf)


def march():
    """分段 ADM 推进到 T_END, 返回均匀时间采样的快照"""
    xs, kf, r, q = init_ics()
    nst = int(round(T_END / H))
    rec = max(1, nst // (N_FRAMES - 1))
    snaps = [(0.0, r.real.copy(), q.real.copy())]
    for s in range(1, nst + 1):
        r, q = adm_step(r, q, kf, H, M_TERM)
        if s % rec == 0 and len(snaps) < N_FRAMES:
            snaps.append((s * H, r.real.copy(), q.real.copy()))
    return xs, kf, r, q, snaps


if __name__ == '__main__':
    import time
    t0 = time.time()
    print(f"q0=0, A={A}, W={W}, 域 [-{L_BOX/2:g},{L_BOX/2:g}], N={N_X}, "
          f"K_CUT={K_CUT}, T={T_END}, h={H}, M={M_TERM}")

    xs, kf, r_f, q_f, snaps = march()
    print(f"分段 ADM {int(round(T_END/H))} 段, {len(snaps)} 帧, 用时 {time.time()-t0:.1f}s")

    # ---- 诊断: 守恒量 / 边界干净度 / 高频泄漏 ----
    mass = np.trapz(snaps[0][1], xs)
    mass_f = np.trapz(snaps[-1][1], xs)
    edge = np.abs(xs) > 14.0
    lk = (np.abs(np.fft.fft(r_f)) ** 2 + np.abs(np.fft.fft(q_f)) ** 2)
    lk = float(lk[np.abs(kf) > 0.88 * K_CUT].sum() / lk.sum())
    print(f"质量: t=0 {mass:.6f} -> t={T_END} {mass_f:.6f} (漂移 {abs(mass_f-mass):.1e})")
    print(f"t={T_END}: r_max={r_f.real.max():+.5f}  "
          f"q: {q_f.real.max():+.5f} / {q_f.real.min():+.5f}")
    print(f"边界区 |x|>14: max|r| = {np.abs(r_f.real[edge]).max():.2e}  高频泄漏 = {lk:.2e}")

    # ---- 三维曲面 r(x,t), q(x,t) ----
    ts = np.array([s[0] for s in snaps])
    R = np.array([s[1] for s in snaps])
    Q = np.array([s[2] for s in snaps])
    stride = max(1, len(xs) // 300)
    X, T = np.meshgrid(xs[::stride], ts)

    fig = plt.figure(figsize=(14, 6))
    for k, (Z, cm_, nm) in enumerate([(R, cm.viridis, 'r'), (Q, cm.plasma, 'q')]):
        axk = fig.add_subplot(1, 2, k + 1, projection='3d')
        surfk = axk.plot_surface(X, T, Z[:, ::stride], cmap=cm_,
                                 linewidth=0, antialiased=True, rstride=1, cstride=1)
        axk.set_title(rf'${nm}(x,t)$ piecewise ADM, $q_0\equiv 0$, $A={A}$, '
                      rf'$t \in [0,{T_END:g}]$', fontsize=10)
        axk.set_xlabel('x'); axk.set_ylabel('t'); axk.set_zlabel(nm)
        fig.colorbar(surfk, ax=axk, shrink=0.5, aspect=10)
    fig.suptitle(rf'piecewise ADM, $h={H:g}$, $M={M_TERM}$, '
                 rf'$K_\mathrm{{cut}}={K_CUT}$,  domain $[-{L_BOX/2:g},{L_BOX/2:g}]$',
                 y=0.99, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig('adm_standalone_3d.png', dpi=115)
    print("saved adm_standalone_3d.png")
    plt.show()
