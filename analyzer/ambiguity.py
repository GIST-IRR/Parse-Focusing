import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"


# Optimization landscape 함수 정의
def landscape(x, y):
    return np.sin(x) + np.cos(y)


# x, y 범위 설정
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Optimization landscape 계산
a = -(
    np.log(X)
    + np.log(Y)
    + 5 * np.log(1 - X - Y)
    - 2
    * np.log(
        ((-Y - X + np.sqrt(4 * Y - 3 * Y**2 - 2 * Y * X + X**2)) / (2 * Y))
    )
)
b = -(np.log(X) + np.log(Y) + 5 * np.log(1 - X - Y))
Z = np.where(2 * X + 3 * Y > 1, a, b)
# for safe
Z[np.isnan(Z)] = np.inf
inf_idx = np.isinf(Z)
Z[inf_idx] = Z[~inf_idx].max()

# Optimum of a
a[np.isnan(a)] = np.inf
a[np.isinf(a)] = a[~np.isinf(a)].max()
a_optimum = np.unravel_index(
    np.argsort(a, axis=None)[0],
    a.shape,
)

# Optimum of b
b[np.isnan(b)] = np.inf
b[np.isinf(b)] = b[~np.isinf(b)].max()
b_optimum = np.unravel_index(
    np.argsort(b, axis=None)[0],
    b.shape,
)

# optimum = np.unravel_index(
#     np.argsort(Z, axis=None)[:2],
#     Z.shape,
# )
optimum = [a_optimum, b_optimum]

Z[inf_idx] = np.inf
# Z = -a
# Z = -b
# Z = -a - b

# 그래프 그리기
fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, Z, cmap="viridis")
ax = fig.add_subplot(111)
# ax.contour(X, Y, Z, 30, colors="k", linewidths=1, linestyles="--")
ax.contourf(X, Y, Z, 100, cmap="viridis")

# # optimum 위치 설정
# optimum1 = 0
# optimum2 = np.array([np.pi, np.pi])

# optimum 위치 표시
ax.scatter(
    X[optimum[0][0]][optimum[0][1]],
    Y[optimum[0][0]][optimum[0][1]],
    color="red",
    label="Optimum 1",
)
ax.scatter(
    X[optimum[1][0]][optimum[1][1]],
    Y[optimum[1][0]][optimum[1][1]],
    color="blue",
    label="Optimum 2",
)

# 그래프 설정
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
ax.set_aspect("equal")
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.set_zlabel("Loss")
# ax.set_zlim(10, 20)
# ax.legend()

# 그래프 출력
# plt.savefig("landscape.png", format="png", bbox_inches="tight")
plt.savefig("landscape.svg", format="svg", bbox_inches="tight")
plt.close()
