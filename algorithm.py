"""
    A Bayesian Approach to Digital Matting
    参考https://github.com/MarcoForte/bayesian-matting
    func  : 完整算法
    Author: Chen Yu
    Date  : 2020.10.21
"""
import cv2
import numpy as np
from numba import jit

from utils import gauss2d_distribution
from orchard_bouman_clust import clustFunc


sigma = 8
N = 25
minN = 10


class BayesianMatting(object):

    @staticmethod
    @jit(nopython=True, cache=True)
    def _solve(mu_F, cov_F, mu_B, cov_B, C, sigma_C, alpha_init, min_like, max_iter=50):
        I = np.eye(3)
        FMax = np.zeros(3)
        BMax = np.zeros(3)
        alphaMax = 0
        maxlike = - np.inf
        invsgma2 = 1 / sigma_C ** 2

        for i in range(mu_F.shape[0]):
            for j in range(mu_B.shape[0]):
                mu_Fi = mu_F[i]
                cov_Fi_inv = np.linalg.inv(cov_F[i])
                mu_Bj = mu_B[j]
                cov_Bi_inv = np.linalg.inv(cov_B[j])

                alpha = alpha_init
                cur_iter = 1
                last_like = -1.7977e+308
                while True:
                    # 根据(9)，求解F、B
                    A11 = cov_Fi_inv + (I * alpha ** 2) * (invsgma2)
                    A12 = I * alpha * (1-alpha) * invsgma2
                    A21 = A12
                    A22 = cov_Bi_inv + (I * (1 - alpha) ** 2) * invsgma2
                    A = np.vstack((np.hstack((A11, A12)), np.hstack((A21, A22))))
                    b1 = cov_Fi_inv @ mu_Fi + C * alpha * invsgma2
                    b2 = cov_Bi_inv @ mu_Bj + C * (1 - alpha) * invsgma2
                    b = np.atleast_2d(np.concatenate((b1, b2))).T

                    X = np.linalg.solve(A, b)
                    F = np.maximum(0, np.minimum(1, X[0:3]))
                    B = np.maximum(0, np.minimum(1, X[3:6]))

                    # 根据(10)，求解alpha
                    alpha = np.maximum(0, np.minimum(1,
                        ((np.atleast_2d(C).T - B).T @ (F - B)) / np.sum((F - B) ** 2)))[0, 0]

                    # 计算likelihood
                    L_C = - np.sum((np.atleast_2d(C).T - alpha * F - (1 - alpha) * B) ** 2) * invsgma2
                    L_F = (- ((F - np.atleast_2d(mu_Fi).T).T @ cov_Fi_inv @ (F - np.atleast_2d(mu_Fi).T)) / 2)[0, 0]
                    L_B = (- ((B - np.atleast_2d(mu_Bj).T).T @ cov_Bi_inv @ (B - np.atleast_2d(mu_Bj).T)) / 2)[0, 0]
                    like = (L_C + L_F + L_B)

                    if like > maxlike:
                        alphaMax = alpha
                        maxLike = like
                        FMax = F.ravel()
                        BMax = B.ravel()

                    if cur_iter >= max_iter or abs(like - last_like) <= min_like:
                        break

                    last_like = like
                    cur_iter += 1
        return FMax, BMax, alphaMax

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_window(m, x, y, N):
        h, w, c = m.shape
        halfN = N // 2
        r = np.zeros((N, N, c))
        xmin = max(0, x - halfN)
        xmax = min(w, x + (halfN + 1))
        ymin = max(0, y - halfN)
        ymax = min(h, y + (halfN + 1))
        pxmin = halfN - (x - xmin)
        pxmax = halfN + (xmax - x)
        pymin = halfN - (y - ymin)
        pymax = halfN + (ymax - y)
        r[pymin:pymax, pxmin:pxmax] = m[ymin:ymax, xmin:xmax]
        return r

    @staticmethod
    def _get_pixels_and_weights(pixels, gaussian_weights, a, N):
        weights = (a ** 2 * gaussian_weights).ravel()
        pixels = np.reshape(pixels, (N * N, 3))
        posInds = np.nan_to_num(weights) > 0
        pixels = pixels[posInds, :]
        weights = weights[posInds]
        return pixels, weights

    @staticmethod
    def bayesian_matte(img, trimap):
        img = img / 255
        h, w, c = img.shape
        alpha = np.zeros((h, w))

        fg_mask = trimap == 255
        bg_mask = trimap == 0
        uk_mask = True ^ np.logical_or(fg_mask, bg_mask)
        foreground = img * np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2)
        background = img * np.repeat(bg_mask[:, :, np.newaxis], 3, axis=2)

        alpha[fg_mask] = 1
        alpha[uk_mask] = np.nan

        gaussian_weights = gauss2d_distribution((N, N), sigma)
        gaussian_weights = gaussian_weights / np.max(gaussian_weights)

        num_uk = np.sum(uk_mask)
        uk_tmp_mask = uk_mask
        kernel = np.ones((3, 3))
        n = 1

        # 每次求解unknown区域最外一层像素
        while n < num_uk:
            uk_tmp_mask = cv2.erode(uk_tmp_mask.astype(np.uint8), kernel, iterations=1)
            uk_pixels = np.logical_and(np.logical_not(uk_tmp_mask), uk_mask)

            Y, X = np.nonzero(uk_pixels)
            for i in range(Y.shape[0]):
                if n % 100 == 0:
                    print(n, num_uk)
                y, x = Y[i], X[i]
                p = img[y, x]
                a = BayesianMatting.get_window(alpha[:, :, np.newaxis], x, y, N)[:, :, 0]

                f_pixels = BayesianMatting.get_window(foreground, x, y, N)
                f_pixels, f_weights = BayesianMatting._get_pixels_and_weights(f_pixels, gaussian_weights, a, N)

                b_pixels = BayesianMatting.get_window(background, x, y, N)
                b_pixels, b_weights = BayesianMatting._get_pixels_and_weights(b_pixels, gaussian_weights, (1-a), N)

                if len(f_weights) < minN or len(b_weights) < minN:
                    continue

                mu_f, sigma_f = clustFunc(f_pixels, f_weights)
                mu_b, sigma_b = clustFunc(b_pixels, b_weights)

                alpha_init = np.nanmean(a.ravel())
                # Solve for F,B for all cluster pairs
                f, b, alphaT = BayesianMatting._solve(mu_f, sigma_f, mu_b, sigma_b, p, 0.01, alpha_init, 1e-6, 50)
                foreground[y, x] = f.ravel()
                background[y, x] = b.ravel()
                alpha[y, x] = alphaT
                uk_mask[y, x] = 0
                n += 1
        return alpha



if __name__ == '__main__':
    from PIL import Image
    import imageio

    img = np.asarray(Image.open("sample/img/woman.png").convert('RGB'))
    trimap = np.asarray(Image.open("sample/trimap/woman.png").convert('L'))

    bm = BayesianMatting()
    alpha = bm.bayesian_matte(img, trimap)

    imageio.imwrite('res.png', alpha)


