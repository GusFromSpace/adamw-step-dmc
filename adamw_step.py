"""adamw_step.py — compact AdamW optimizer step.

Faithful to the scalar core of torch.optim.AdamW: decoupled weight decay,
first/second moment updates, bias correction, and epsilon-stabilized update.

Fixed to four parameters and three gradient batches so the demoniC port can
stay close without needing classes or dynamic tensor write-back.
"""

import math


def adamw_step(w, m, v, g, step, lr=0.05, beta1=0.9, beta2=0.999,
               eps=1e-8, weight_decay=0.01):
    b1_corr = 1.0 - beta1 ** step
    b2_corr = 1.0 - beta2 ** step
    for i in range(len(w)):
        # Decoupled decay: AdamW applies it directly to the weight.
        w[i] -= lr * weight_decay * w[i]
        m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
        v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]
        m_hat = m[i] / b1_corr
        v_hat = v[i] / b2_corr
        w[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)


def main():
    w = [0.50, -1.00, 2.00, -0.25]
    m = [0.0, 0.0, 0.0, 0.0]
    v = [0.0, 0.0, 0.0, 0.0]
    grads = [
        [0.10, -0.20, 0.05, 0.30],
        [0.07, -0.15, 0.02, 0.25],
        [0.04, -0.10, 0.01, 0.20],
    ]

    for step, g in enumerate(grads, 1):
        adamw_step(w, m, v, g, step)
        print("step %d" % step)
        print("  w=" + " ".join("%.6f" % x for x in w))
        print("  m=" + " ".join("%.6f" % x for x in m))
        print("  v=" + " ".join("%.6f" % x for x in v))


if __name__ == "__main__":
    main()
