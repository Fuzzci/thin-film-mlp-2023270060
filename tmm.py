import numpy as np
from config import WAVELENGTHS, N_H, N_L, N_S


def layer_matrix(n, d, lam):
    """单层薄膜的特征矩阵，正入射，无吸收"""
    delta = 2 * np.pi * n * d / lam
    return np.array([
        [np.cos(delta), (1j / n) * np.sin(delta)],
        [(1j * n) * np.sin(delta), np.cos(delta)]
    ], dtype=complex)


def reflectance_spectrum(thicknesses, wavelengths=WAVELENGTHS):
    """
    计算 Air / H / L / H / L / Glass 的反射光谱
    thicknesses = [d1, d2, d3, d4]，单位 nm
    d1, d3 是 H 层；d2, d4 是 L 层
    """
    d1, d2, d3, d4 = thicknesses
    R = []

    for lam in wavelengths:
        M = np.eye(2, dtype=complex)

        # 光从空气依次经过 H -> L -> H -> L -> 玻璃
        M = M @ layer_matrix(N_H, d1, lam)
        M = M @ layer_matrix(N_L, d2, lam)
        M = M @ layer_matrix(N_H, d3, lam)
        M = M @ layer_matrix(N_L, d4, lam)

        # 基底玻璃
        B = M[0, 0] * 1.0 + M[0, 1] * N_S
        C = M[1, 0] * 1.0 + M[1, 1] * N_S

        # 空气折射率 n0 = 1
        r = (1.0 * B - C) / (1.0 * B + C)
        R.append(abs(r) ** 2)

    return np.array(R)


if __name__ == "__main__":
    # 先随便给一个膜厚，测试能不能跑
    d_test = [100, 100, 100, 100]
    R = reflectance_spectrum(d_test)

    print("膜厚：", d_test)
    print("反射率数组：")
    print(R)
    print("数组长度：", len(R))