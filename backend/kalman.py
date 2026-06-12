"""Pure-Python 2-state Kalman filter for time-varying linear regression.

Tracks the dynamic relationship ``y_t = alpha_t + beta_t * x_t + noise`` where
alpha_t and beta_t drift as a slow random walk. Used by the live engine to
maintain a continually-updated WTI-vs-DXY hedge ratio, so the pair-trading
signal compares today's residual against today's relationship — not against
the static 5-year average.

No numpy dependency so the live runtime stays lightweight; the 2x2 matrix
algebra is small enough to write directly."""
from __future__ import annotations

from typing import Tuple


class KalmanPair:
    """Online estimator of (alpha_t, beta_t) for y = alpha + beta * x."""

    def __init__(self,
                 q_alpha: float = 1e-4,
                 q_beta: float = 1e-5,
                 r_obs: float = 4.0,
                 init_var: float = 1.0) -> None:
        self.alpha = 0.0
        self.beta = 0.0
        # full 2x2 posterior covariance (kept fully explicit to avoid
        # symmetry-drift bugs)
        self.p00 = init_var
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = init_var
        self.q_alpha = q_alpha
        self.q_beta = q_beta
        self.r_obs = r_obs

    def update(self, y: float, x: float) -> float:
        """Absorb one observation (y, x) and return the innovation
        (residual: y - predicted_y)."""
        # ---- predict step (random-walk transition) -------------------- #
        p00 = self.p00 + self.q_alpha
        p01 = self.p01
        p10 = self.p10
        p11 = self.p11 + self.q_beta

        # observation row H = [1, x]
        # P @ H^T  (2-vector)
        ph0 = p00 + p01 * x
        ph1 = p10 + p11 * x

        # innovation variance  S = H @ P @ H^T + R   (scalar)
        s = p00 + (p01 + p10) * x + p11 * x * x + self.r_obs

        # Kalman gain  K = P @ H^T / S   (2-vector)
        k0 = ph0 / s
        k1 = ph1 / s

        # innovation
        innovation = y - (self.alpha + self.beta * x)

        # ---- update step --------------------------------------------- #
        self.alpha += k0 * innovation
        self.beta += k1 * innovation

        # P_new = (I - K @ H) @ P_pred,  where  I - K @ H is
        #   [[1 - k0,  -k0*x],
        #    [  -k1,  1 - k1*x]]
        m00 = 1.0 - k0
        m01 = -k0 * x
        m10 = -k1
        m11 = 1.0 - k1 * x

        self.p00 = m00 * p00 + m01 * p10
        self.p01 = m00 * p01 + m01 * p11
        self.p10 = m10 * p00 + m11 * p10
        self.p11 = m10 * p01 + m11 * p11
        return innovation

    def residual(self, y: float, x: float) -> float:
        """Compute the residual for the current state without updating it.

        Used to read the live (intraday) residual from the latest alpha/beta
        without disturbing the Kalman state until a new daily observation
        is committed."""
        return y - (self.alpha + self.beta * x)

    def snapshot(self) -> Tuple[float, float]:
        """Current (alpha, beta) estimates."""
        return self.alpha, self.beta
