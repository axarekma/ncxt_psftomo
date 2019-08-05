import unittest
import ncxt_psftomo as module
import numpy as np


class MainTest(unittest.TestCase):
    def test_adjointness_psf(self):

        L = 5
        Lp = 7
        height = 5
        na = 3
        res, dof = 2, L // 2
        psf = module.psf_utils.psf_volume(res, dof, 1, Lp)

        x = np.zeros((L, L, height))
        y = np.zeros((na, Lp, height))
        py = x * 0
        angles = np.linspace(0, 2 * np.pi, na)

        errors = 0
        # for pixels in volume
        for vol_i in range(L):
            for vol_j in range(L):
                for vol_k in range(height):
                    for proj_i in range(na):
                        for proj_j in range(Lp):
                            for proj_k in range(height):
                                y *= 0
                                x *= 0
                                x[vol_i, vol_j, vol_k] = 1
                                module.forward(x, y, psf, angles)
                                val_a = y[proj_i, proj_j, proj_k]

                                y *= 0
                                x *= 0
                                y[proj_i, proj_j, proj_k] = 1
                                module.backward(x, y, psf, angles)
                                val_at = x[vol_i, vol_j, vol_k]

                                if not np.isclose(val_a, val_at):
                                    errors += 1
                                    print(f"pair ({i},{j}) ({p},{a}) differs")
                                    print(f"   A:{val_a:.3}  AT:{val_at:.3}")
                # print(f'errors {errors} of {L*L*Lp*na}')
        self.assertEqual(errors, 0)

    def test_adjointness_bl(self):

        L = 5
        Lp = 7
        height = 5
        na = 3

        x = np.zeros((L, L, height))
        y = np.zeros((na, Lp, height))
        py = x * 0
        angles = np.linspace(0, 2 * np.pi, na)

        errors = 0
        # for pixels in volume
        for vol_i in range(L):
            for vol_j in range(L):
                for vol_k in range(height):
                    for proj_i in range(na):
                        for proj_j in range(Lp):
                            for proj_k in range(height):
                                y *= 0
                                x *= 0
                                x[vol_i, vol_j, vol_k] = 1
                                module.forward_bl(x, y, angles)
                                val_a = y[proj_i, proj_j, proj_k]

                                y *= 0
                                x *= 0
                                y[proj_i, proj_j, proj_k] = 1
                                module.backward_bl(x, y, angles)
                                val_at = x[vol_i, vol_j, vol_k]

                                if not np.isclose(val_a, val_at):
                                    errors += 1
                                    print(f"pair ({i},{j}) ({p},{a}) differs")
                                    print(f"   A:{val_a:.3}  AT:{val_at:.3}")
                # print(f'errors {errors} of {L*L*Lp*na}')
        self.assertEqual(errors, 0)


if __name__ == "__main__":
    unittest.main()
