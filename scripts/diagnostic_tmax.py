import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scripts.run_all_to_all_parallel import Config, _one_traj
from src.simulators.all_to_all_stim import make_sample_times

def main():
    cfg = Config()
    cfg.N_values = [512]
    cfg.r_values = [0.40]
    cfg.n_traj = 150
    cfg.t_max_factor = 25
    cfg.n_log_times = 30
    cfg.n_lin_times = 150
    N, r = 512, 0.40
    t_max = cfg.t_max_factor * N
    sample_times = make_sample_times(t_max, cfg.n_log_times, cfg.n_lin_times)
    print("Diagnostic N=512 r=0.40 t_max=%d, %d samples, %d traj, %d workers" % (
        t_max, len(sample_times), cfg.n_traj, cfg.n_workers), flush=True)
    args = [(N, r, sample_times, t_max, cfg.seed + j) for j in range(cfg.n_traj)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
        traj = np.array(list(pool.map(_one_traj, args, chunksize=4)))
    print("  done in %ds" % (time.time()-t0), flush=True)
    mean = traj.mean(0)
    s = mean / N
    x = np.array(sample_times, float) / N
    grad = np.gradient(s, x)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(x, s, "-o", ms=3)
    ax1.set_ylabel("S_R / N")
    ax1.set_title("N=512 r=0.40 relaxation")
    ax1.grid(alpha=0.3)
    ax2.plot(x, np.abs(grad), "-o", ms=3, color="C3")
    ax2.set_ylabel("|grad|")
    ax2.set_xlabel("t/N (sweeps)")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")
    for thr in [1e-2, 1e-3]:
        below = np.where(np.abs(grad) < thr)[0]
        if below.size:
            xf = x[below[0]]
            ax1.axvline(xf, ls="--", lw=1)
            ax2.axvline(xf, ls="--", lw=1)
            print("  |grad| < %g first at t/N = %.1f" % (thr, xf), flush=True)
    fig.tight_layout()
    fig.savefig("diagnostic_tmax.png", dpi=150)
    print("  saved diagnostic_tmax.png", flush=True)
    print("  final S_R/N: %.4f" % s[-5:].mean(), flush=True)

if __name__ == "__main__":
    main()
