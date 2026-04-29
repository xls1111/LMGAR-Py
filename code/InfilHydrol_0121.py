import math
import numpy as np


def G(psi_upper_lim, psi_lower_lim, alpha, n, m, K_s, num_trapezoids=120):
    epsilon = 1e-10
    psi_upper_lim = max(psi_upper_lim, epsilon)
    psi_lower_lim = max(psi_lower_lim, epsilon)

    if abs(psi_upper_lim - psi_lower_lim) < epsilon:
        return 0
    else:
        psis_to_integrate = np.linspace(psi_lower_lim, psi_upper_lim, num_trapezoids + 1)
        G_result = 0

        for i in range(0, num_trapezoids):
            try:
                k1 = K(capital_theta_of_psi(psis_to_integrate[i + 1], alpha, n, m), K_s, m)
                k2 = K(capital_theta_of_psi(psis_to_integrate[i], alpha, n, m), K_s, m)
                G_result = G_result + (k1 + k2) / 2 * abs((psis_to_integrate[i + 1] - psis_to_integrate[i]))
            except:
                continue

        if abs(K_s) > epsilon:
            G_result = abs(G_result / K_s)
        else:
            G_result = 0.0

        if psi_upper_lim > psi_lower_lim:
            G_result = abs(G_result)

    return G_result


def psi_of_theta(theta, theta_s, theta_r, n, m, alpha):
    """计算基质势，添加数值保护"""
    epsilon = 1e-10

    # 确保 theta 在有效范围内 [theta_r, theta_s]
    theta = np.clip(theta, theta_r + epsilon, theta_s - epsilon)

    numerator = theta_s - theta_r
    denominator = theta - theta_r

    # 避免除零
    if denominator < epsilon:
        denominator = epsilon

    ratio = numerator / denominator

    # 确保 ratio >= 1
    if ratio < 1.0:
        ratio = 1.0

    try:
        result = ((ratio ** (1 / m) - 1) ** (1 / n)) / alpha
        return max(result, epsilon)
    except:
        return epsilon


def theta_of_psi(psi, theta_s, theta_r, n, m, alpha):
    """计算含水量，添加数值保护"""
    epsilon = 1e-10
    psi = max(psi, epsilon)

    try:
        theta = theta_r + (theta_s - theta_r) / (1 + (alpha * psi) ** n) ** m
        # 确保结果在有效范围内
        return np.clip(theta, theta_r, theta_s)
    except:
        return theta_r


def capital_theta_of_psi(psi, alpha, n, m):
    """计算有效饱和度，添加数值保护"""
    epsilon = 1e-10
    psi = max(psi, epsilon)

    try:
        result = 1 / ((1 + (alpha * psi) ** n) ** m)
        # 确保结果在 [0, 1] 范围内
        return np.clip(result, 0.0, 1.0)
    except:
        return 0.0


def K(capital_theta_temp, K_s, m):
    """计算导水率，添加数值保护"""
    epsilon = 1e-10

    # 确保 capital_theta_temp 在 [0, 1] 范围内
    capital_theta_temp = np.clip(capital_theta_temp, epsilon, 1.0 - epsilon)

    try:
        # 安全计算
        temp = capital_theta_temp ** (1 / m)
        temp = min(temp, 1.0 - epsilon)  # 确保 1-temp >= 0

        inner = max(1 - temp, 0.0)  # 确保非负

        result = K_s * capital_theta_temp ** 0.5 * (1 - inner ** m) ** 2

        return np.clip(result, 0.0, K_s)
    except:
        return epsilon * K_s


def capital_theta(theta, theta_r, theta_s):
    """计算有效饱和度，添加数值保护"""
    epsilon = 1e-10

    theta = np.clip(theta, theta_r, theta_s)
    denominator = theta_s - theta_r

    if abs(denominator) < epsilon:
        return 0.0

    result = (theta - theta_r) / denominator
    return np.clip(result, 0.0, 1.0)


def derivs(QP, rh, F, WC, y, WFZ, a):
    """计算导数，添加数值保护"""
    epsilon = 1e-10

    # 避免除零
    denominator = y - WC
    if abs(denominator) < epsilon:
        return 0.0

    z = F / denominator
    if abs(z) < epsilon:
        z = epsilon

    dydx = 0

    if a == 0:
        y = np.clip(y, QP.WCR[0] + epsilon, QP.WCS[0] - epsilon)
        WC = np.clip(WC, QP.WCR[0], QP.WCS[0])

        QP.k_composite = K(capital_theta(y, QP.WCR[0], QP.WCS[0]), QP.ks[0], QP.m[0])
        G_temp = G(psi_of_theta(y, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]),
                   psi_of_theta(WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]),
                   QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
        dydx = (rh - QP.k_composite - (QP.ks[0] * G_temp / max(WFZ, epsilon))) / z

    elif a == 1:
        y = np.clip(y, QP.WCR[1] + epsilon, QP.WCS[1] - epsilon)
        WC = np.clip(WC, QP.WCR[1], QP.WCS[1])

        psi_y = psi_of_theta(y, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
        theta_y_layer0 = theta_of_psi(psi_y, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])

        k0 = K(capital_theta(theta_y_layer0, QP.WCR[0], QP.WCS[0]), QP.ks[0], QP.m[0])
        k1 = K(capital_theta(y, QP.WCR[1], QP.WCS[1]), QP.ks[1], QP.m[1])

        QP.k_composite = WFZ / (QP.max_depth[0] / max(k0, epsilon) +
                                (WFZ - QP.max_depth[0]) / max(k1, epsilon))

        G_temp = G(psi_of_theta(y, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]),
                   psi_of_theta(WC, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]),
                   QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
        dydx = (rh - QP.k_composite - (QP.ks[1] * G_temp / max(WFZ, epsilon))) / z

    elif a == 2:
        y = np.clip(y, QP.WCR[2] + epsilon, QP.WCS[2] - epsilon)
        WC = np.clip(WC, QP.WCR[2], QP.WCS[2])

        psi_y = psi_of_theta(y, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
        theta_y_layer0 = theta_of_psi(psi_y, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
        theta_y_layer1 = theta_of_psi(psi_y, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])

        k0 = K(capital_theta(theta_y_layer0, QP.WCR[0], QP.WCS[0]), QP.ks[0], QP.m[0])
        k1 = K(capital_theta(theta_y_layer1, QP.WCR[1], QP.WCS[1]), QP.ks[1], QP.m[1])
        k2 = K(capital_theta(y, QP.WCR[2], QP.WCS[2]), QP.ks[2], QP.m[2])

        QP.k_composite = WFZ / (QP.max_depth[0] / max(k0, epsilon) +
                                QP.max_depth[1] / max(k1, epsilon) +
                                (WFZ - QP.max_depth[0] - QP.max_depth[1]) / max(k2, epsilon))

        G_temp = G(psi_of_theta(y, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]),
                   psi_of_theta(WC, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]),
                   QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])
        dydx = (rh - QP.k_composite - (QP.ks[2] * G_temp / max(WFZ, epsilon))) / z

    return dydx


#   x                   起始时间
#   y                   x 处的 y
#   dydx                x 处的 dydx
#   h                   时间步长
def rk4(QP, Rh, F, WC, y, dydx, h, WFZ, a):
    # 初始化局部变量
    hh = h / 2.0

    # 计算中间值 yt 和 dydx
    yt = y + hh * dydx
    dyt = derivs(QP, Rh, F, WC, yt, WFZ, a)

    yt = y + hh * dyt
    dym = derivs(QP, Rh, F, WC, yt, WFZ, a)

    yt = y + h * dym
    dym = dym + dyt

    dyt = derivs(QP, Rh, F, WC, yt, WFZ, a)

    # 计算 yout 并返回
    yout = y + h * (dydx + dyt + 2.0 * dym) / 6.0
    return yout


#   Q                   当前处理的方程组的索引
#   Rh, F, WC           可能是常微分方程中的参数
#   y                   当前解
#   dydx                当前导数
#   x                   当前自变量
#   htry                初始步长
#   eps                 允许的误差
#   yscal               用于误差计算的尺度因子
#   hdid                使用步长
#   hnext               下一个步长
def rkqc(QP, Rh, F, WC, y, dydx, x, htry, eps, yscal, WFZ, a):
    pgrow = -0.2                    # 步长增长因子
    pshrnk = -0.25                  # 步长缩减因子
    fcor = 1.0 / 15.0               # 误差控制因子
    safety = 0.9                    # 安全因子，用于调整步长
    errcon = 6.0e-4                 # 误差控制常数

    # 保存当前 x, y, 和 dydx 的值
    xsav = x
    ysav = y
    dysav = dydx
    h = htry

    while True:
        hh = h / 2.0
        # 第一步：使用半步长 hh 进行一次 rk4 计算
        ytemp = rk4(QP, Rh, F, WC, ysav, dysav, hh, WFZ, a)

        # 计算 dydx 的中间导数
        dydx = derivs(QP, Rh, F, WC, ytemp, WFZ, a)
        # 第二步：再次使用半步长 hh 进行 rk4 计算
        y = rk4(QP, Rh, F, WC, ytemp, dydx, hh, WFZ, a)

        # 第三步：用全步长 h 进行一次 rk4 计算
        ytemp_full = rk4(QP, Rh, F, WC, ysav, dysav, h, WFZ, a)

        # 计算最大误差 errmax
        ytemp = y - ytemp_full
        errmax = abs(ytemp / yscal)
        errmax = errmax/eps

        # 如果误差小于允许范围
        if errmax <= 1.0:
            hdid = h  # 记录实际使用的步长
            if errmax > errcon:
                hnext = safety * h * math.exp(pgrow * math.log(errmax))
            else:
                hnext = 4.0 * h  # 增大步长
            break
        else:
            # 误差过大，缩减步长
            h = safety * h * math.exp(pshrnk * math.log(errmax))

    # 返回 y 的新值和步长调整
    y += ytemp * fcor
    x = xsav + h
    return x, y, hdid, hnext


def odeint(QP, Rh, F, WC, ystart, x1, x2, eps, h1, WFZ, a):
    maxstp = 100  # 最大允许的步数
    tiny = 1.0e-3  # 非常小的数值，避免除以零或非常小的值
    x = x1
    h = abs(h1)/10
    y = ystart

    for nstp in range(maxstp):
        # 计算当前导数 dydx
        dydx = derivs(QP, Rh, F, WC, y, WFZ, a)

        # 误差尺度因子，用于误差估计
        yscal = abs(y) + abs(dydx * h) + tiny

        # 如果步长 h 导致 x 超过 x2，则调整 h 以确保积分不超出区间
        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        # 使用 rkqc 进行一次步长控制的 RK 计算
        x, y, hdid, hnext = rkqc(QP, Rh, F, WC, y, dydx, x, h, eps, yscal, WFZ, a)

        # 检查是否完成积分
        if (x - x2) * (x2 - x1) >= 0.0:
            ystart = y
            break

        # 更新步长 h 为下一个步长 hnext
        h = hnext

    return ystart


#  Q                    方程组的索引
#  num                  数值参数，用于避免除以零
#  redistT              重新分配时间
#  QP                   参数数据结构，包含 `ks` 属性
def adj_factor_calc(QP, num, redistT, z):
    # 常量系数
    a = 4.2951615111
    b = 154.6101111175
    c = 0.0020393887
    d = -0.0010402988
    e = -14.0032425382
    f = -61.5428564782

    # 当 num 为 0 时，返回 0
    if num == 0:
        return 0.0
    else:
        # 计算 alpha、beta 和 delta
        if z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
        elif QP.boundary_depths[0] < z <= QP.boundary_depths[1]:
            QP.ks_composite = z / (QP.max_depth[0] / QP.ks[0] + (z - QP.max_depth[0]) / QP.ks[1])
            # QP.ks_composite = QP.ks[0]
        elif QP.boundary_depths[1] < z <= QP.boundary_depths[2]:
            QP.ks_composite = z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
            # QP.ks_composite = QP.ks[0]

        ks = QP.ks_composite  # 假设 QP 是包含各参数的字典或类似结构
        alpha = ks / (a * ks + b)
        beta = c + d * math.sqrt(ks)
        delta = ks / (e * ks + f)

        # 计算结果
        result = (alpha + beta * math.log(redistT) + delta / num)/6

        # 微调因子以优化下渗量到120mm
        # result = result * 1.002  # 增加0.2%的调整因子

        # 如果结果为负值，调整为 0
        if result < 0.0:
            result = 0.0

    return result


# Redist11、Redist20、Redist21、Redist30、Redist31 和 Redist40 过程根据系统所处的再分配状态执行再分配过程
#  Q                    方程组的索引
#  rh                   降雨参数
#  ts, te               积分时间范围的起点和终点
#  dt                   时间步长
#  QP                   包含各方程组参数的字典
def redist11(QP, rh, ts, te, dt):
    """重新分配水分 - 简化版"""

    # ===== 辅助函数 =====
    def get_layer_index(z, boundary_depths):
        """根据深度获取土层索引"""
        for i, boundary in enumerate(boundary_depths):
            if z <= boundary:
                return i
        return len(boundary_depths)  # 超出最底层

    def convert_theta_between_layers(theta, psi, from_layer, to_layer, QP):
        """在不同土层间转换含水量（通过psi保持连续性）"""
        if from_layer == to_layer:
            return theta, psi

        # 如果psi未提供，先计算
        if psi is None:
            psi = psi_of_theta(theta, QP.WCS[from_layer], QP.WCR[from_layer],
                               QP.n[from_layer], QP.m[from_layer], QP.alpha[from_layer])

        # 转换到新层
        new_theta = theta_of_psi(psi, QP.WCS[to_layer], QP.WCR[to_layer],
                                 QP.n[to_layer], QP.m[to_layer], QP.alpha[to_layer])
        return new_theta, psi

    def calc_water_in_layer(psi, layer_idx, QP):
        """计算某层中的theta值"""
        return theta_of_psi(psi, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def calc_total_water_above_layer(psi, target_layer, QP):
        """计算目标层以上所有层的总水量"""
        total_water = 0.0
        for layer_idx in range(target_layer):
            theta_in_layer = calc_water_in_layer(psi, layer_idx, QP)
            water_in_layer = (theta_in_layer - QP.WCI[layer_idx]) * QP.max_depth[layer_idx]
            total_water += water_in_layer
        return total_water

    def calc_z_in_multilayer(FAmt, psi, current_layer, QP):
        """计算跨多层的wetting front深度"""
        # 计算当前层以上的水量
        water_above = calc_total_water_above_layer(psi, current_layer, QP)

        # 计算当前层的theta
        theta_current = calc_water_in_layer(psi, current_layer, QP)

        # 计算当前层的深度
        depth_above = sum(QP.max_depth[i] for i in range(current_layer))
        remaining_water = FAmt - water_above
        z_in_current = remaining_water / (theta_current - QP.WCI[current_layer])

        return depth_above + z_in_current

    def update_wf_properties(WF, layer_idx, QP):
        """更新wetting front的所有属性"""
        WF.psiHold = psi_of_theta(WF.WCHold, QP.WCS[layer_idx], QP.WCR[layer_idx],
                                  QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])
        WF.psi = psi_of_theta(WF.WC, QP.WCS[layer_idx], QP.WCR[layer_idx],
                              QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def handle_overflow(WF, psi, QP):
        """处理水量溢出"""
        # 计算最大容量
        max_capacity = 0.0
        for layer_idx in range(len(QP.max_depth)):
            theta_in_layer = calc_water_in_layer(psi, layer_idx, QP)
            max_capacity += (theta_in_layer - QP.WCI[layer_idx]) * QP.max_depth[layer_idx]

        # 计算溢出量
        overflow = WF.FAmt - max_capacity
        if overflow > 0:
            QP.runoff += overflow * 0
            WF.FAmt = max_capacity
            WF.z = QP.boundary_depths[-1]  # 设置到最底层

            # 更新到最底层的含水量
            bottom_layer = len(QP.max_depth) - 1
            WF.WC = calc_water_in_layer(psi, bottom_layer, QP)
            WF.WCHold = WF.WC + WF.adjFactor
            update_wf_properties(WF, bottom_layer, QP)
            return True
        return False

    # ===== 主逻辑 =====
    WF = QP.WF[0]

    # 1. 确定当前所在土层
    current_layer = get_layer_index(WF.z, QP.boundary_depths)

    # 防止越界
    if current_layer >= len(QP.max_depth):
        current_layer = len(QP.max_depth) - 1

    # 2. 使用ODE积分更新WCHold
    WF.WCHold = odeint(QP, rh, WF.FAmt, QP.WCI[current_layer],
                       WF.WCHold, ts, te, 1.0e-4, dt, WF.z, current_layer)

    # 3. 更新WC
    WF.WC = WF.WCHold - WF.adjFactor

    # 4. 更新psi值
    update_wf_properties(WF, current_layer, QP)

    # 5. 计算新的z位置
    if current_layer == 0:
        # 第一层简单计算
        WF.z = calc_redist_zf(QP, WF.FAmt, WF.WC, QP.WCI[0])
    else:
        # 多层计算
        WF.z = calc_z_in_multilayer(WF.FAmt, WF.psi, current_layer, QP)

    # 6. 检查是否跨层
    new_layer = get_layer_index(WF.z, QP.boundary_depths)

    # 防止越界
    if new_layer >= len(QP.max_depth):
        new_layer = len(QP.max_depth) - 1

    if new_layer != current_layer:
        # 跨层了，需要转换theta
        WF.WC, WF.psi = convert_theta_between_layers(
            WF.WC, WF.psi, current_layer, new_layer, QP)
        WF.WCHold, WF.psiHold = convert_theta_between_layers(
            WF.WCHold, WF.psiHold, current_layer, new_layer, QP)

        # 重新计算z（因为theta变了）
        WF.z = calc_z_in_multilayer(WF.FAmt, WF.psi, new_layer, QP)

        # 更新psi属性
        update_wf_properties(WF, new_layer, QP)

    # 7. 检查是否溢出
    if WF.z > QP.boundary_depths[-1]:
        handle_overflow(WF, WF.psi, QP)

    # 8. 更新潜在入渗速率
    QP.fp = fp_calc(QP)


def redist20(QP, rh, ts, te, dt):
    """两个wetting front的再分配 - 3层专用简化版"""

    # ===== 辅助函数 =====
    def get_layer(z):
        """获取深度z所在的土层索引 (0, 1, 2)"""
        if z <= QP.boundary_depths[0]:
            return 0
        elif z <= QP.boundary_depths[1]:
            return 1
        else:
            return 2

    def psi_to_theta(psi, layer_idx):
        """psi转theta"""
        return theta_of_psi(psi, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def theta_to_psi(theta, layer_idx):
        """theta转psi"""
        return psi_of_theta(theta, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def calc_z_from_water(FAmt, psi):
        """从总水量和psi计算z位置"""
        remaining = FAmt

        # Layer 0
        theta0 = psi_to_theta(psi, 0)
        water0 = (theta0 - QP.WCI[0]) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta0 - QP.WCI[0])
        remaining -= water0

        # Layer 1
        theta1 = psi_to_theta(psi, 1)
        water1 = (theta1 - QP.WCI[1]) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta1 - QP.WCI[1])
        remaining -= water1

        # Layer 2
        theta2 = psi_to_theta(psi, 2)
        water2 = (theta2 - QP.WCI[2]) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta2 - QP.WCI[2])

        # 溢出
        return QP.boundary_depths[2]

    def calc_z_relative(FAmt, psi_front, psi_base):
        """计算相对于另一个front的z位置 (用于WF[1])"""
        remaining = FAmt

        # Layer 0
        theta_front0 = psi_to_theta(psi_front, 0)
        theta_base0 = psi_to_theta(psi_base, 0)
        water0 = (theta_front0 - theta_base0) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta_front0 - theta_base0)
        remaining -= water0

        # Layer 1
        theta_front1 = psi_to_theta(psi_front, 1)
        theta_base1 = psi_to_theta(psi_base, 1)
        water1 = (theta_front1 - theta_base1) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta_front1 - theta_base1)
        remaining -= water1

        # Layer 2
        theta_front2 = psi_to_theta(psi_front, 2)
        theta_base2 = psi_to_theta(psi_base, 2)
        water2 = (theta_front2 - theta_base2) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta_front2 - theta_base2)

        # 溢出
        return QP.boundary_depths[2]

    def handle_overflow(WF, psi):
        """处理溢出"""
        # 计算最大容量
        max_capacity = 0
        for i in range(3):
            theta = psi_to_theta(psi, i)
            max_capacity += (theta - QP.WCI[i]) * QP.max_depth[i]

        if WF.FAmt > max_capacity:
            overflow = WF.FAmt - max_capacity
            QP.runoff += overflow * 0
            WF.FAmt = max_capacity
            WF.z = QP.boundary_depths[2]

            # 更新到第3层
            WF.WC = psi_to_theta(psi, 2)
            WF.WCHold = WF.WC + WF.adjFactor
            WF.psiHold = theta_to_psi(WF.WCHold, 2)
            WF.psi = theta_to_psi(WF.WC, 2)
            return True
        return False

    def update_to_new_layer(WF, old_layer, new_layer):
        """更新到新土层"""
        if old_layer == new_layer:
            return

        # 通过psi转换到新层
        WF.WC = psi_to_theta(WF.psi, new_layer)
        WF.WCHold = WF.WC + WF.adjFactor
        WF.psiHold = theta_to_psi(WF.WCHold, new_layer)
        WF.psi = theta_to_psi(WF.WC, new_layer)

    # ===== 处理 WF[0] (正在再分配) =====
    WF0 = QP.WF[0]
    layer0 = get_layer(WF0.z)

    # 1. ODE积分
    WF0.WCHold = odeint(QP, 0, WF0.FAmt, QP.WCI[layer0],
                        WF0.WCHold, ts, te, 1.0e-4, dt, WF0.z, layer0)

    # 2. 更新WC和psi
    WF0.WC = WF0.WCHold - WF0.adjFactor
    WF0.psiHold = theta_to_psi(WF0.WCHold, layer0)
    WF0.psi = theta_to_psi(WF0.WC, layer0)

    # 3. 计算新的z
    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)

    # 4. 检查跨层
    new_layer0 = get_layer(WF0.z)
    update_to_new_layer(WF0, layer0, new_layer0)

    # 5. 检查溢出
    if WF0.z > QP.boundary_depths[2]:
        handle_overflow(WF0, WF0.psi)

    # ===== 处理 WF[1] (正在入渗) =====
    WF1 = QP.WF[1]
    layer1 = get_layer(WF1.z)

    # 1. 更新psi (保持原值)
    WF1.psi = theta_to_psi(WF1.WC, layer1)
    WF1.psiHold = theta_to_psi(WF1.WCHold, layer1)

    # 2. 计算相对于WF[0]的z
    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

    # 3. 检查跨层
    new_layer1 = get_layer(WF1.z)
    update_to_new_layer(WF1, layer1, new_layer1)

    # 4. 检查溢出
    if WF1.z > QP.boundary_depths[2]:
        # 计算相对最大容量
        max_capacity = 0
        for i in range(3):
            theta_wf1 = psi_to_theta(WF1.psi, i)
            theta_wf0 = psi_to_theta(WF0.psi, i)
            max_capacity += (theta_wf1 - theta_wf0) * QP.max_depth[i]

        if WF1.FAmt > max_capacity:
            overflow = WF1.FAmt - max_capacity
            QP.runoff += overflow * 0
            WF1.FAmt = max_capacity
            WF1.z = QP.boundary_depths[2]
            update_to_new_layer(WF1, new_layer1, 2)

    # ===== 检查合并 =====
    if WF1.z >= WF0.z:
        # 合并到WF[0]
        WF0.FAmt += WF1.FAmt
        WF0.WC = WF1.WC
        WF0.WCHold = WF1.WCHold
        WF0.psi = WF1.psi
        WF0.psiHold = WF1.psiHold
        WF0.redistTime = WF1.redistTime
        WF0.adjFactor = WF1.adjFactor

        # 清空WF[1]
        WF1.FAmt = 0.0
        WF1.WC = 0.0
        WF1.WCHold = 0.0
        WF1.numRedist = 0.0
        WF1.redistTime = 0.0
        WF1.adjFactor = 0.0
        WF1.z = 0.0
        QP.redistStatus = 10

        # 重新计算WF[0]的位置
        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
        final_layer = get_layer(WF0.z)
        WF0.WC = psi_to_theta(WF0.psi, final_layer)
        WF0.WCHold = psi_to_theta(WF0.psiHold, final_layer)

        # 检查溢出
        handle_overflow(WF0, WF0.psi)

    # 更新潜在入渗速率
    QP.fp = fp_calc(QP)


def redist21(QP, rh, ts, te, dt):
    """两个wetting front都在再分配 - 3层专用简化版"""

    # ===== 辅助函数(与redist20相同) =====
    def get_layer(z):
        """获取深度z所在的土层索引 (0, 1, 2)"""
        if z <= QP.boundary_depths[0]:
            return 0
        elif z <= QP.boundary_depths[1]:
            return 1
        else:
            return 2

    def psi_to_theta(psi, layer_idx):
        """psi转theta"""
        return theta_of_psi(psi, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def theta_to_psi(theta, layer_idx):
        """theta转psi"""
        return psi_of_theta(theta, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def calc_z_from_water(FAmt, psi):
        """从总水量和psi计算z位置"""
        remaining = FAmt

        # Layer 0
        theta0 = psi_to_theta(psi, 0)
        water0 = (theta0 - QP.WCI[0]) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta0 - QP.WCI[0])
        remaining -= water0

        # Layer 1
        theta1 = psi_to_theta(psi, 1)
        water1 = (theta1 - QP.WCI[1]) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta1 - QP.WCI[1])
        remaining -= water1

        # Layer 2
        theta2 = psi_to_theta(psi, 2)
        water2 = (theta2 - QP.WCI[2]) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta2 - QP.WCI[2])

        # 溢出
        return QP.boundary_depths[2]

    def calc_z_relative(FAmt, psi_front, psi_base):
        """计算相对于另一个front的z位置 (用于WF[1])"""
        remaining = FAmt

        # Layer 0
        theta_front0 = psi_to_theta(psi_front, 0)
        theta_base0 = psi_to_theta(psi_base, 0)
        water0 = (theta_front0 - theta_base0) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta_front0 - theta_base0)
        remaining -= water0

        # Layer 1
        theta_front1 = psi_to_theta(psi_front, 1)
        theta_base1 = psi_to_theta(psi_base, 1)
        water1 = (theta_front1 - theta_base1) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta_front1 - theta_base1)
        remaining -= water1

        # Layer 2
        theta_front2 = psi_to_theta(psi_front, 2)
        theta_base2 = psi_to_theta(psi_base, 2)
        water2 = (theta_front2 - theta_base2) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta_front2 - theta_base2)

        # 溢出
        return QP.boundary_depths[2]

    def handle_overflow(WF, psi):
        """处理溢出"""
        # 计算最大容量
        max_capacity = 0
        for i in range(3):
            theta = psi_to_theta(psi, i)
            max_capacity += (theta - QP.WCI[i]) * QP.max_depth[i]

        if WF.FAmt > max_capacity:
            overflow = WF.FAmt - max_capacity
            QP.runoff += overflow * 0
            WF.FAmt = max_capacity
            WF.z = QP.boundary_depths[2]

            # 更新到第3层
            WF.WC = psi_to_theta(psi, 2)
            WF.WCHold = WF.WC + WF.adjFactor
            WF.psiHold = theta_to_psi(WF.WCHold, 2)
            WF.psi = theta_to_psi(WF.WC, 2)
            return True
        return False

    def handle_overflow_relative(WF, psi_front, psi_base):
        """处理相对溢出(用于WF[1])"""
        # 计算相对最大容量
        max_capacity = 0
        for i in range(3):
            theta_front = psi_to_theta(psi_front, i)
            theta_base = psi_to_theta(psi_base, i)
            max_capacity += (theta_front - theta_base) * QP.max_depth[i]

        if WF.FAmt > max_capacity:
            overflow = WF.FAmt - max_capacity
            QP.runoff += overflow * 0
            WF.FAmt = max_capacity
            WF.z = QP.boundary_depths[2]

            # 更新到第3层
            WF.WC = psi_to_theta(psi_front, 2)
            WF.WCHold = WF.WC + WF.adjFactor
            WF.psiHold = theta_to_psi(WF.WCHold, 2)
            WF.psi = theta_to_psi(WF.WC, 2)
            return True
        return False

    def update_to_new_layer(WF, old_layer, new_layer):
        """更新到新土层"""
        if old_layer == new_layer:
            return

        # 通过psi转换到新层
        WF.WC = psi_to_theta(WF.psi, new_layer)
        WF.WCHold = WF.WC + WF.adjFactor
        WF.psiHold = theta_to_psi(WF.WCHold, new_layer)
        WF.psi = theta_to_psi(WF.WC, new_layer)

    # ===== 处理 WF[0] (正在再分配) =====
    WF0 = QP.WF[0]
    layer0 = get_layer(WF0.z)

    # 1. ODE积分
    WF0.WCHold = odeint(QP, 0, WF0.FAmt, QP.WCI[layer0],
                        WF0.WCHold, ts, te, 1.0e-4, dt, WF0.z, layer0)

    # 2. 更新WC和psi
    WF0.WC = WF0.WCHold - WF0.adjFactor
    WF0.psiHold = theta_to_psi(WF0.WCHold, layer0)
    WF0.psi = theta_to_psi(WF0.WC, layer0)

    # 3. 计算新的z
    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)

    # 4. 检查跨层
    new_layer0 = get_layer(WF0.z)
    update_to_new_layer(WF0, layer0, new_layer0)

    # 5. 检查溢出
    if WF0.z > QP.boundary_depths[2]:
        handle_overflow(WF0, WF0.psi)

    # ===== 处理 WF[1] (也在再分配,但相对于WF[0]) =====
    WF1 = QP.WF[1]
    layer1 = get_layer(WF1.z)

    # 1. ODE积分 (注意:WCI参数使用WF[0]的psiHold转换的theta)
    wci_relative = psi_to_theta(WF0.psiHold, layer1)
    WF1.WCHold = odeint(QP, rh, WF1.FAmt, wci_relative,
                        WF1.WCHold, ts, te, 1.0e-4, dt, WF1.z, layer1)

    # 2. 更新WC和psi
    WF1.WC = WF1.WCHold - WF1.adjFactor
    WF1.psiHold = theta_to_psi(WF1.WCHold, layer1)
    WF1.psi = theta_to_psi(WF1.WC, layer1)

    # 3. 计算相对于WF[0]的z
    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

    # 4. 检查跨层
    new_layer1 = get_layer(WF1.z)
    update_to_new_layer(WF1, layer1, new_layer1)

    # 5. 检查溢出
    if WF1.z > QP.boundary_depths[2]:
        handle_overflow_relative(WF1, WF1.psi, WF0.psi)

    # ===== 检查合并 =====
    if WF1.z >= WF0.z:
        # 合并到WF[0]
        WF0.FAmt += WF1.FAmt
        WF0.WC = WF1.WC
        WF0.WCHold = WF1.WCHold
        WF0.psi = WF1.psi
        WF0.psiHold = WF1.psiHold
        WF0.redistTime = WF1.redistTime
        WF0.adjFactor = WF1.adjFactor

        # 清空WF[1]
        WF1.FAmt = 0.0
        WF1.WC = 0.0
        WF1.WCHold = 0.0
        WF1.numRedist = 0.0
        WF1.redistTime = 0.0
        WF1.adjFactor = 0.0
        WF1.z = 0.0
        QP.redistStatus = 10

        # 重新计算WF[0]的位置
        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
        final_layer = get_layer(WF0.z)
        WF0.WC = psi_to_theta(WF0.psi, final_layer)
        WF0.WCHold = psi_to_theta(WF0.psiHold, final_layer)

        # 检查溢出
        handle_overflow(WF0, WF0.psi)

    # 更新潜在入渗速率
    QP.fp = fp_calc(QP)


def redist30(QP, rh, ts, te, dt):
    """三个wetting front的再分配 - 3层专用简化版（完整修正版）"""

    # ===== 辅助函数 =====
    def get_layer(z):
        """获取深度z所在的土层索引 (0, 1, 2)"""
        if z <= QP.boundary_depths[0]:
            return 0
        elif z <= QP.boundary_depths[1]:
            return 1
        else:
            return 2

    def psi_to_theta(psi, layer_idx):
        """psi转theta"""
        return theta_of_psi(psi, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def theta_to_psi(theta, layer_idx):
        """theta转psi"""
        return psi_of_theta(theta, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def calc_z_from_water(FAmt, psi):
        """从总水量和psi计算z位置"""
        remaining = FAmt

        # Layer 0
        theta0 = psi_to_theta(psi, 0)
        water0 = (theta0 - QP.WCI[0]) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta0 - QP.WCI[0])
        remaining -= water0

        # Layer 1
        theta1 = psi_to_theta(psi, 1)
        water1 = (theta1 - QP.WCI[1]) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta1 - QP.WCI[1])
        remaining -= water1

        # Layer 2
        theta2 = psi_to_theta(psi, 2)
        water2 = (theta2 - QP.WCI[2]) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta2 - QP.WCI[2])

        return QP.boundary_depths[2]

    def calc_z_relative(FAmt, psi_front, psi_base):
        """计算相对于另一个front的z位置"""
        remaining = FAmt

        # Layer 0
        theta_front0 = psi_to_theta(psi_front, 0)
        theta_base0 = psi_to_theta(psi_base, 0)
        water0 = (theta_front0 - theta_base0) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta_front0 - theta_base0) if (theta_front0 - theta_base0) > 0 else 0
        remaining -= water0

        # Layer 1
        theta_front1 = psi_to_theta(psi_front, 1)
        theta_base1 = psi_to_theta(psi_base, 1)
        water1 = (theta_front1 - theta_base1) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta_front1 - theta_base1) if (theta_front1 - theta_base1) > 0 else QP.max_depth[0]
        remaining -= water1

        # Layer 2
        theta_front2 = psi_to_theta(psi_front, 2)
        theta_base2 = psi_to_theta(psi_base, 2)
        water2 = (theta_front2 - theta_base2) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta_front2 - theta_base2) if (theta_front2 - theta_base2) > 0 else QP.max_depth[0] + QP.max_depth[1]

        return QP.boundary_depths[2]

    def handle_overflow(FAmt, psi):
        """处理溢出,返回修正后的FAmt和overflow"""
        max_capacity = 0
        for i in range(3):
            theta = psi_to_theta(psi, i)
            max_capacity += (theta - QP.WCI[i]) * QP.max_depth[i]

        if FAmt > max_capacity:
            overflow = FAmt - max_capacity
            return max_capacity, overflow
        return FAmt, 0.0

    def handle_overflow_relative(FAmt, psi_front, psi_base):
        """处理相对溢出"""
        max_capacity = 0
        for i in range(3):
            theta_front = psi_to_theta(psi_front, i)
            theta_base = psi_to_theta(psi_base, i)
            max_capacity += (theta_front - theta_base) * QP.max_depth[i]

        if FAmt > max_capacity:
            overflow = FAmt - max_capacity
            return max_capacity, overflow
        return FAmt, 0.0

    def update_to_layer(WF, layer_idx):
        """更新WF到指定土层"""
        WF.WC = psi_to_theta(WF.psi, layer_idx)
        WF.WCHold = psi_to_theta(WF.psiHold, layer_idx)

    def clear_wf(WF):
        """清空wetting front"""
        WF.FAmt = 0.0
        WF.WC = 0.0
        WF.WCHold = 0.0
        WF.numRedist = 0.0
        WF.redistTime = 0.0
        WF.adjFactor = 0.0
        WF.z = 0.0

    # ===== 处理 WF[0] (正在再分配) =====
    WF0 = QP.WF[0]
    layer0 = get_layer(WF0.z)

    # 1. ODE积分
    WF0.WCHold = odeint(QP, 0, WF0.FAmt, QP.WCI[layer0],
                        WF0.WCHold, ts, te, 1.0e-4, dt, WF0.z, layer0)
    WF0.WC = WF0.WCHold - WF0.adjFactor

    # 2. 更新psi
    WF0.psiHold = theta_to_psi(WF0.WCHold, layer0)
    WF0.psi = theta_to_psi(WF0.WC, layer0)

    # 3. 计算新的z
    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)

    # 4. 检查跨层
    new_layer0 = get_layer(WF0.z)
    if new_layer0 != layer0:
        update_to_layer(WF0, new_layer0)
        WF0.psiHold = theta_to_psi(WF0.WCHold, new_layer0)
        WF0.psi = theta_to_psi(WF0.WC, new_layer0)

    # 5. 检查溢出
    if WF0.z > QP.boundary_depths[2]:
        WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
        QP.runoff += overflow * 0
        WF0.z = QP.boundary_depths[2]
        update_to_layer(WF0, 2)
        WF0.psiHold = theta_to_psi(WF0.WCHold, 2)
        WF0.psi = theta_to_psi(WF0.WC, 2)

    # ===== 处理 WF[1] (再分配,相对于WF[0]) =====
    WF1 = QP.WF[1]
    layer1 = get_layer(WF1.z)

    # 1. ODE积分
    wci_base = psi_to_theta(WF0.psiHold, layer1)
    WF1.WCHold = odeint(QP, 0, WF1.FAmt, wci_base,
                        WF1.WCHold, ts, te, 1.0e-4, dt, WF1.z, layer1)
    WF1.WC = WF1.WCHold - WF1.adjFactor

    # 2. 更新psi
    WF1.psiHold = theta_to_psi(WF1.WCHold, layer1)
    WF1.psi = theta_to_psi(WF1.WC, layer1)

    # 3. 计算相对于WF[0]的z
    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

    # 4. 检查跨层
    new_layer1 = get_layer(WF1.z)
    if new_layer1 != layer1:
        update_to_layer(WF1, new_layer1)
        WF1.psiHold = theta_to_psi(WF1.WCHold, new_layer1)
        WF1.psi = theta_to_psi(WF1.WC, new_layer1)

    # 5. 检查溢出
    if WF1.z > QP.boundary_depths[2]:
        WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
        QP.runoff += overflow * 0
        WF1.z = QP.boundary_depths[2]
        update_to_layer(WF1, 2)
        WF1.psiHold = theta_to_psi(WF1.WCHold, 2)
        WF1.psi = theta_to_psi(WF1.WC, 2)

    # ===== 检查WF[1]和WF[0]是否需要合并 =====
    if WF1.z >= WF0.z:
        # *** 合并WF[1]到WF[0] ***
        WF0.FAmt += WF1.FAmt

        # 先临时确定一个层
        temp_layer = get_layer(WF0.z)

        # 设置为该层的最大含水量
        WF0.WC = WF1.WC
        WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

        # 计算对应的psi
        WF0.psi = theta_to_psi(WF0.WC, temp_layer)
        WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

        WF0.numRedist = WF1.numRedist
        WF0.redistTime = WF1.redistTime
        WF0.adjFactor = WF1.adjFactor

        # WF[2]移到WF[1]
        WF1.FAmt = QP.WF[2].FAmt
        WF1.WC = QP.WF[2].WC
        WF1.WCHold = QP.WF[2].WCHold
        WF1.psi = QP.WF[2].psi
        WF1.psiHold = QP.WF[2].psiHold
        WF1.numRedist = QP.WF[2].numRedist
        WF1.redistTime = QP.WF[2].redistTime
        WF1.adjFactor = QP.WF[2].adjFactor
        WF1.z = QP.WF[2].z

        # 清空WF[2]
        clear_wf(QP.WF[2])

        QP.redistStatus = 20

        # **关键修正**: 重新计算WF[0]的位置，此时WF[0]右侧有WF[1]（原WF[2]）
        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)

        final_layer0 = get_layer(WF0.z)

        if final_layer0 != temp_layer:
            WF0.WC = QP.WCS[final_layer0]
            WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
            WF0.psi = theta_to_psi(WF0.WC, final_layer0)
            WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
            final_layer0 = get_layer(WF0.z)

        update_to_layer(WF0, final_layer0)

        # 检查WF[0]溢出
        if WF0.z > QP.boundary_depths[2]:
            WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
            QP.runoff += overflow * 0
            WF0.z = QP.boundary_depths[2]
            update_to_layer(WF0, 2)

        # 重新计算WF[1]的位置(现在是原WF[2],正在入渗)
        layer1_new = get_layer(WF1.z)

        # 更新psi (保持不变,因为是入渗front)
        WF1.psi = theta_to_psi(WF1.WC, layer1_new)
        WF1.psiHold = theta_to_psi(WF1.WCHold, layer1_new)

        # 计算相对于WF[0]的z
        WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

        # 检查跨层
        final_layer1 = get_layer(WF1.z)
        if final_layer1 != layer1_new:
            update_to_layer(WF1, final_layer1)

        # 检查溢出
        if WF1.z > QP.boundary_depths[2]:
            WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
            QP.runoff += overflow * 0
            WF1.z = QP.boundary_depths[2]
            update_to_layer(WF1, 2)

        # 再次检查WF[1]和WF[0]是否需要合并
        if WF1.z >= WF0.z:
            # *** 再次合并WF[1]到WF[0] ***
            WF0.FAmt += WF1.FAmt

            # 先临时确定一个层
            temp_layer = get_layer(WF0.z)

            # 设置为该层的最大含水量
            WF0.WC = QP.WCS[temp_layer]
            WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

            # 计算对应的psi
            WF0.psi = theta_to_psi(WF0.WC, temp_layer)
            WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

            WF0.numRedist = WF1.numRedist
            WF0.redistTime = WF1.redistTime
            WF0.adjFactor = WF1.adjFactor

            # 清空WF[1]
            clear_wf(WF1)

            QP.redistStatus = 10

            # 重新计算WF[0]
            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
            final_layer = get_layer(WF0.z)

            # 如果层发生变化，更新到新层的WCS
            if final_layer != temp_layer:
                WF0.WC = QP.WCS[final_layer]
                WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer])
                WF0.psi = theta_to_psi(WF0.WC, final_layer)
                WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer)
                WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                final_layer = get_layer(WF0.z)

            update_to_layer(WF0, final_layer)

            # 检查溢出
            if WF0.z > QP.boundary_depths[2]:
                WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                QP.runoff += overflow * 0
                WF0.z = QP.boundary_depths[2]
                update_to_layer(WF0, 2)

    else:
        # 仍然是3个front,更新WF[2] (正在入渗)
        WF2 = QP.WF[2]
        layer2 = get_layer(WF2.z)

        # 1. 更新psi (保持不变,因为是入渗front)
        WF2.psi = theta_to_psi(WF2.WC, layer2)
        WF2.psiHold = theta_to_psi(WF2.WCHold, layer2)

        # 2. 计算相对于WF[1]的z
        WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)

        # 3. 检查跨层
        new_layer2 = get_layer(WF2.z)
        if new_layer2 != layer2:
            update_to_layer(WF2, new_layer2)

        # 4. 检查溢出
        if WF2.z > QP.boundary_depths[2]:
            WF2.FAmt, overflow = handle_overflow_relative(WF2.FAmt, WF2.psi, WF1.psi)
            QP.runoff += overflow * 0
            WF2.z = QP.boundary_depths[2]
            update_to_layer(WF2, 2)

        # 5. 检查WF[2]和WF[1]是否需要合并
        if WF2.z >= WF1.z:
            # *** 合并WF[2]到WF[1] ***
            WF1.FAmt += WF2.FAmt

            # 先临时确定一个层
            temp_layer = get_layer(WF1.z)

            # 设置为该层的最大含水量
            WF1.WC = QP.WCS[temp_layer]
            WF1.WCHold = min(WF1.WC + WF2.adjFactor, QP.WCS[temp_layer])

            # 计算对应的psi
            WF1.psi = theta_to_psi(WF1.WC, temp_layer)
            WF1.psiHold = theta_to_psi(WF1.WCHold, temp_layer)

            WF1.numRedist = WF2.numRedist
            WF1.redistTime = WF2.redistTime
            WF1.adjFactor = WF2.adjFactor

            # 清空WF[2]
            clear_wf(WF2)

            QP.redistStatus = 20

            # 重新计算WF[1]的位置
            WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
            final_layer1 = get_layer(WF1.z)

            # 如果层发生变化，更新到新层的WCS
            if final_layer1 != temp_layer:
                WF1.WC = QP.WCS[final_layer1]
                WF1.WCHold = min(WF1.WC + WF1.adjFactor, QP.WCS[final_layer1])
                WF1.psi = theta_to_psi(WF1.WC, final_layer1)
                WF1.psiHold = theta_to_psi(WF1.WCHold, final_layer1)
                WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                final_layer1 = get_layer(WF1.z)

            update_to_layer(WF1, final_layer1)

            # 检查溢出
            if WF1.z > QP.boundary_depths[2]:
                WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                QP.runoff += overflow * 0
                WF1.z = QP.boundary_depths[2]
                update_to_layer(WF1, 2)

            # 再次检查WF[1]和WF[0]是否需要合并
            if WF1.z >= WF0.z:
                # *** 最后合并WF[1]到WF[0] ***
                WF0.FAmt += WF1.FAmt

                # 先临时确定一个层
                temp_layer = get_layer(WF0.z)

                # 设置为该层的最大含水量
                WF0.WC = QP.WCS[temp_layer]
                WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

                # 计算对应的psi
                WF0.psi = theta_to_psi(WF0.WC, temp_layer)
                WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

                WF0.numRedist = WF1.numRedist
                WF0.redistTime = WF1.redistTime
                WF0.adjFactor = WF1.adjFactor

                # 清空WF[1]
                clear_wf(WF1)

                QP.redistStatus = 10

                # 重新计算WF[0]
                WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                final_layer0 = get_layer(WF0.z)

                # 如果层发生变化，更新到新层的WCS
                if final_layer0 != temp_layer:
                    WF0.WC = QP.WCS[final_layer0]
                    WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
                    WF0.psi = theta_to_psi(WF0.WC, final_layer0)
                    WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
                    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                    final_layer0 = get_layer(WF0.z)

                update_to_layer(WF0, final_layer0)

                # 检查溢出
                if WF0.z > QP.boundary_depths[2]:
                    WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                    QP.runoff += overflow * 0
                    WF0.z = QP.boundary_depths[2]
                    update_to_layer(WF0, 2)

    # 更新潜在入渗速率
    QP.fp = fp_calc(QP)


def redist31(QP, rh, ts, te, dt):
    """三个wetting front都在再分配 - 3层专用简化版"""

    # ===== 辅助函数 =====
    def get_layer(z):
        """获取深度z所在的土层索引 (0, 1, 2)"""
        if z <= QP.boundary_depths[0]:
            return 0
        elif z <= QP.boundary_depths[1]:
            return 1
        else:
            return 2

    def psi_to_theta(psi, layer_idx):
        """psi转theta"""
        return theta_of_psi(psi, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def theta_to_psi(theta, layer_idx):
        """theta转psi"""
        return psi_of_theta(theta, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def calc_z_from_water(FAmt, psi):
        """从总水量和psi计算z位置"""
        remaining = FAmt

        # Layer 0
        theta0 = psi_to_theta(psi, 0)
        water0 = (theta0 - QP.WCI[0]) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta0 - QP.WCI[0])
        remaining -= water0

        # Layer 1
        theta1 = psi_to_theta(psi, 1)
        water1 = (theta1 - QP.WCI[1]) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta1 - QP.WCI[1])
        remaining -= water1

        # Layer 2
        theta2 = psi_to_theta(psi, 2)
        water2 = (theta2 - QP.WCI[2]) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta2 - QP.WCI[2])

        return QP.boundary_depths[2]

    def calc_z_relative(FAmt, psi_front, psi_base):
        """计算相对于另一个front的z位置"""
        remaining = FAmt

        # Layer 0
        theta_front0 = psi_to_theta(psi_front, 0)
        theta_base0 = psi_to_theta(psi_base, 0)
        water0 = (theta_front0 - theta_base0) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta_front0 - theta_base0) if (theta_front0 - theta_base0) > 0 else 0
        remaining -= water0

        # Layer 1
        theta_front1 = psi_to_theta(psi_front, 1)
        theta_base1 = psi_to_theta(psi_base, 1)
        water1 = (theta_front1 - theta_base1) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta_front1 - theta_base1) if (theta_front1 - theta_base1) > 0 else QP.max_depth[0]
        remaining -= water1

        # Layer 2
        theta_front2 = psi_to_theta(psi_front, 2)
        theta_base2 = psi_to_theta(psi_base, 2)
        water2 = (theta_front2 - theta_base2) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta_front2 - theta_base2) if (theta_front2 - theta_base2) > 0 else QP.max_depth[0] + QP.max_depth[1]

        return QP.boundary_depths[2]

    def handle_overflow(FAmt, psi):
        """处理溢出,返回修正后的FAmt和overflow"""
        max_capacity = 0
        for i in range(3):
            theta = psi_to_theta(psi, i)
            max_capacity += (theta - QP.WCI[i]) * QP.max_depth[i]

        if FAmt > max_capacity:
            overflow = FAmt - max_capacity
            return max_capacity, overflow
        return FAmt, 0.0

    def handle_overflow_relative(FAmt, psi_front, psi_base):
        """处理相对溢出"""
        max_capacity = 0
        for i in range(3):
            theta_front = psi_to_theta(psi_front, i)
            theta_base = psi_to_theta(psi_base, i)
            max_capacity += (theta_front - theta_base) * QP.max_depth[i]

        if FAmt > max_capacity:
            overflow = FAmt - max_capacity
            return max_capacity, overflow
        return FAmt, 0.0

    def update_to_layer(WF, layer_idx):
        """更新WF到指定土层"""
        WF.WC = psi_to_theta(WF.psi, layer_idx)
        WF.WCHold = psi_to_theta(WF.psiHold, layer_idx)

    def clear_wf(WF):
        """清空wetting front"""
        WF.FAmt = 0.0
        WF.WC = 0.0
        WF.WCHold = 0.0
        WF.numRedist = 0.0
        WF.redistTime = 0.0
        WF.adjFactor = 0.0
        WF.z = 0.0

    # ===== 处理 WF[0] (正在再分配) =====
    WF0 = QP.WF[0]
    layer0 = get_layer(WF0.z)

    # 1. ODE积分
    WF0.WCHold = odeint(QP, 0, WF0.FAmt, QP.WCI[layer0],
                        WF0.WCHold, ts, te, 1.0e-4, dt, WF0.z, layer0)
    WF0.WC = WF0.WCHold - WF0.adjFactor

    # 2. 更新psi
    WF0.psiHold = theta_to_psi(WF0.WCHold, layer0)
    WF0.psi = theta_to_psi(WF0.WC, layer0)

    # 3. 计算新的z
    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)

    # 4. 检查跨层
    new_layer0 = get_layer(WF0.z)
    if new_layer0 != layer0:
        update_to_layer(WF0, new_layer0)
        WF0.psiHold = theta_to_psi(WF0.WCHold, new_layer0)
        WF0.psi = theta_to_psi(WF0.WC, new_layer0)

    # 5. 检查溢出
    if WF0.z > QP.boundary_depths[2]:
        WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
        QP.runoff += overflow * 0
        WF0.z = QP.boundary_depths[2]
        update_to_layer(WF0, 2)
        WF0.psiHold = theta_to_psi(WF0.WCHold, 2)
        WF0.psi = theta_to_psi(WF0.WC, 2)

    # ===== 处理 WF[1] (再分配,相对于WF[0]) =====
    WF1 = QP.WF[1]
    layer1 = get_layer(WF1.z)

    # 1. ODE积分
    wci_base = psi_to_theta(WF0.psiHold, layer1)
    WF1.WCHold = odeint(QP, 0, WF1.FAmt, wci_base,
                        WF1.WCHold, ts, te, 1.0e-4, dt, WF1.z, layer1)
    WF1.WC = WF1.WCHold - WF1.adjFactor

    # 2. 更新psi
    WF1.psiHold = theta_to_psi(WF1.WCHold, layer1)
    WF1.psi = theta_to_psi(WF1.WC, layer1)

    # 3. 计算相对于WF[0]的z
    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

    # 4. 检查跨层
    new_layer1 = get_layer(WF1.z)
    if new_layer1 != layer1:
        update_to_layer(WF1, new_layer1)
        WF1.psiHold = theta_to_psi(WF1.WCHold, new_layer1)
        WF1.psi = theta_to_psi(WF1.WC, new_layer1)

    # 5. 检查溢出
    if WF1.z > QP.boundary_depths[2]:
        WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
        QP.runoff += overflow * 0
        WF1.z = QP.boundary_depths[2]
        update_to_layer(WF1, 2)
        WF1.psiHold = theta_to_psi(WF1.WCHold, 2)
        WF1.psi = theta_to_psi(WF1.WC, 2)

    # ===== 检查WF[1]和WF[0]是否需要合并 =====
    if WF1.z >= WF0.z:
        # *** 合并WF[1]到WF[0] ***
        WF0.FAmt += WF1.FAmt

        # 先临时确定一个层
        temp_layer = get_layer(WF0.z)

        # 设置为该层的最大含水量
        WF0.WC = WF1.WC
        WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

        # 计算对应的psi
        WF0.psi = theta_to_psi(WF0.WC, temp_layer)
        WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

        WF0.numRedist = WF1.numRedist
        WF0.redistTime = WF1.redistTime
        WF0.adjFactor = WF1.adjFactor

        # WF[2]移到WF[1]
        WF1.FAmt = QP.WF[2].FAmt
        WF1.WC = QP.WF[2].WC
        WF1.WCHold = QP.WF[2].WCHold
        WF1.psi = QP.WF[2].psi
        WF1.psiHold = QP.WF[2].psiHold
        WF1.numRedist = QP.WF[2].numRedist
        WF1.redistTime = QP.WF[2].redistTime
        WF1.adjFactor = QP.WF[2].adjFactor
        WF1.z = QP.WF[2].z

        # 清空WF[2]
        clear_wf(QP.WF[2])

        QP.redistStatus = 20

        # **关键修正**: 重新计算WF[0]的位置，此时WF[0]右侧有WF[1]（原WF[2]）
        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)

        final_layer0 = get_layer(WF0.z)

        # 如果层发生变化，更新到新层的WCS
        if final_layer0 != temp_layer:
            WF0.WC = QP.WCS[final_layer0]
            WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
            WF0.psi = theta_to_psi(WF0.WC, final_layer0)
            WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
            final_layer0 = get_layer(WF0.z)

        update_to_layer(WF0, final_layer0)

        # 检查WF[0]溢出
        if WF0.z > QP.boundary_depths[2]:
            WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
            QP.runoff += overflow * 0
            WF0.z = QP.boundary_depths[2]
            update_to_layer(WF0, 2)

        # 重新处理WF[1] (原WF[2],也在再分配)
        layer1_new = get_layer(WF1.z)

        # ODE积分
        wci_base_new = psi_to_theta(WF0.psiHold, layer1_new)
        WF1.WCHold = odeint(QP, rh, WF1.FAmt, wci_base_new,
                            WF1.WCHold, ts, te, 1.0e-4, dt, WF1.z, layer1_new)
        WF1.WC = WF1.WCHold - WF1.adjFactor

        # 更新psi
        WF1.psiHold = theta_to_psi(WF1.WCHold, layer1_new)
        WF1.psi = theta_to_psi(WF1.WC, layer1_new)

        # 计算相对于WF[0]的z
        WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

        # 检查跨层
        final_layer1 = get_layer(WF1.z)
        if final_layer1 != layer1_new:
            update_to_layer(WF1, final_layer1)
            WF1.psiHold = theta_to_psi(WF1.WCHold, final_layer1)
            WF1.psi = theta_to_psi(WF1.WC, final_layer1)

        # 检查溢出
        if WF1.z > QP.boundary_depths[2]:
            WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
            QP.runoff += overflow * 0
            WF1.z = QP.boundary_depths[2]
            update_to_layer(WF1, 2)
            WF1.psiHold = theta_to_psi(WF1.WCHold, 2)
            WF1.psi = theta_to_psi(WF1.WC, 2)

        # 再次检查WF[1]和WF[0]是否需要合并
        if WF1.z >= WF0.z:
            # *** 再次合并WF[1]到WF[0] ***
            WF0.FAmt += WF1.FAmt

            # 先临时确定一个层
            temp_layer = get_layer(WF0.z)

            # 设置为该层的最大含水量
            WF0.WC = QP.WCS[temp_layer]
            WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

            # 计算对应的psi
            WF0.psi = theta_to_psi(WF0.WC, temp_layer)
            WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

            WF0.numRedist = WF1.numRedist
            WF0.redistTime = WF1.redistTime
            WF0.adjFactor = WF1.adjFactor

            # 清空WF[1]
            clear_wf(WF1)

            QP.redistStatus = 10

            # 重新计算WF[0]
            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
            final_layer = get_layer(WF0.z)

            # 如果层发生变化，更新到新层的WCS
            if final_layer != temp_layer:
                WF0.WC = QP.WCS[final_layer]
                WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer])
                WF0.psi = theta_to_psi(WF0.WC, final_layer)
                WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer)
                WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                final_layer = get_layer(WF0.z)

            update_to_layer(WF0, final_layer)

            # 检查溢出
            if WF0.z > QP.boundary_depths[2]:
                WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                QP.runoff += overflow * 0
                WF0.z = QP.boundary_depths[2]
                update_to_layer(WF0, 2)

    else:
        # 仍然是3个front,更新WF[2] (也在再分配)
        WF2 = QP.WF[2]
        layer2 = get_layer(WF2.z)

        # 1. ODE积分
        wci_base2 = psi_to_theta(WF1.psiHold, layer2)
        WF2.WCHold = odeint(QP, rh, WF2.FAmt, wci_base2, WF2.WCHold, ts, te, 1.0e-4, dt, WF2.z, layer2)
        WF2.WC = WF2.WCHold - WF2.adjFactor

        # 2. 更新psi
        WF2.psiHold = theta_to_psi(WF2.WCHold, layer2)
        WF2.psi = theta_to_psi(WF2.WC, layer2)

        # 3. 计算相对于WF[1]的z
        WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)

        # 4. 检查跨层
        new_layer2 = get_layer(WF2.z)
        if new_layer2 != layer2:
            update_to_layer(WF2, new_layer2)
            WF2.psiHold = theta_to_psi(WF2.WCHold, new_layer2)
            WF2.psi = theta_to_psi(WF2.WC, new_layer2)

        # 5. 检查溢出
        if WF2.z > QP.boundary_depths[2]:
            WF2.FAmt, overflow = handle_overflow_relative(WF2.FAmt, WF2.psi, WF1.psi)
            QP.runoff += overflow * 0
            WF2.z = QP.boundary_depths[2]
            update_to_layer(WF2, 2)
            WF2.psiHold = theta_to_psi(WF2.WCHold, 2)
            WF2.psi = theta_to_psi(WF2.WC, 2)

        # 6. 检查WF[2]和WF[1]是否需要合并
        if WF2.z >= WF1.z:
            # *** 合并WF[2]到WF[1] ***
            WF1.FAmt += WF2.FAmt

            # 先临时确定一个层
            temp_layer = get_layer(WF1.z)

            # 设置为该层的最大含水量
            WF1.WC = QP.WCS[temp_layer]
            WF1.WCHold = min(WF1.WC + WF2.adjFactor, QP.WCS[temp_layer])

            # 计算对应的psi
            WF1.psi = theta_to_psi(WF1.WC, temp_layer)
            WF1.psiHold = theta_to_psi(WF1.WCHold, temp_layer)

            WF1.numRedist = WF2.numRedist
            WF1.redistTime = WF2.redistTime
            WF1.adjFactor = WF2.adjFactor

            # 清空WF[2]
            clear_wf(WF2)

            QP.redistStatus = 20

            # 重新计算WF[1]的位置
            WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
            final_layer1 = get_layer(WF1.z)

            # 如果层发生变化，更新到新层的WCS
            if final_layer1 != temp_layer:
                WF1.WC = QP.WCS[final_layer1]
                WF1.WCHold = min(WF1.WC + WF1.adjFactor, QP.WCS[final_layer1])
                WF1.psi = theta_to_psi(WF1.WC, final_layer1)
                WF1.psiHold = theta_to_psi(WF1.WCHold, final_layer1)
                WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                final_layer1 = get_layer(WF1.z)

            update_to_layer(WF1, final_layer1)

            # 检查溢出
            if WF1.z > QP.boundary_depths[2]:
                WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                QP.runoff += overflow * 0
                WF1.z = QP.boundary_depths[2]
                update_to_layer(WF1, 2)

            # 再次检查WF[1]和WF[0]是否需要合并
            if WF1.z >= WF0.z:
                # *** 最后合并WF[1]到WF[0] ***
                WF0.FAmt += WF1.FAmt

                # 先临时确定一个层
                temp_layer = get_layer(WF0.z)

                # 设置为该层的最大含水量
                WF0.WC = QP.WCS[temp_layer]
                WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

                # 计算对应的psi
                WF0.psi = theta_to_psi(WF0.WC, temp_layer)
                WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

                WF0.numRedist = WF1.numRedist
                WF0.redistTime = WF1.redistTime
                WF0.adjFactor = WF1.adjFactor

                # 清空WF[1]
                clear_wf(WF1)

                QP.redistStatus = 10

                # 重新计算WF[0]
                WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                final_layer0 = get_layer(WF0.z)

                # 如果层发生变化，更新到新层的WCS
                if final_layer0 != temp_layer:
                    WF0.WC = QP.WCS[final_layer0]
                    WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
                    WF0.psi = theta_to_psi(WF0.WC, final_layer0)
                    WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
                    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                    final_layer0 = get_layer(WF0.z)

                update_to_layer(WF0, final_layer0)

                # 检查溢出
                if WF0.z > QP.boundary_depths[2]:
                    WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                    QP.runoff += overflow * 0
                    WF0.z = QP.boundary_depths[2]
                    update_to_layer(WF0, 2)

    # 更新潜在入渗速率
    QP.fp = fp_calc(QP)


def redist40(QP, rh, ts, te, dt):
    """四个wetting front的再分配 - 3层专用简化版（完整修正版）"""

    # ===== 辅助函数 =====
    def get_layer(z):
        """获取深度z所在的土层索引 (0, 1, 2)"""
        if z <= QP.boundary_depths[0]:
            return 0
        elif z <= QP.boundary_depths[1]:
            return 1
        else:
            return 2

    def psi_to_theta(psi, layer_idx):
        """psi转theta"""
        return theta_of_psi(psi, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def theta_to_psi(theta, layer_idx):
        """theta转psi"""
        return psi_of_theta(theta, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def calc_z_from_water(FAmt, psi):
        """从总水量和psi计算z位置"""
        remaining = FAmt

        # Layer 0
        theta0 = psi_to_theta(psi, 0)
        water0 = (theta0 - QP.WCI[0]) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta0 - QP.WCI[0])
        remaining -= water0

        # Layer 1
        theta1 = psi_to_theta(psi, 1)
        water1 = (theta1 - QP.WCI[1]) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta1 - QP.WCI[1])
        remaining -= water1

        # Layer 2
        theta2 = psi_to_theta(psi, 2)
        water2 = (theta2 - QP.WCI[2]) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta2 - QP.WCI[2])

        return QP.boundary_depths[2]

    def calc_z_relative(FAmt, psi_front, psi_base):
        """计算相对于另一个front的z位置"""
        remaining = FAmt

        # Layer 0
        theta_front0 = psi_to_theta(psi_front, 0)
        theta_base0 = psi_to_theta(psi_base, 0)
        water0 = (theta_front0 - theta_base0) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta_front0 - theta_base0) if (theta_front0 - theta_base0) > 0 else 0
        remaining -= water0

        # Layer 1
        theta_front1 = psi_to_theta(psi_front, 1)
        theta_base1 = psi_to_theta(psi_base, 1)
        water1 = (theta_front1 - theta_base1) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta_front1 - theta_base1) if (theta_front1 - theta_base1) > 0 else QP.max_depth[0]
        remaining -= water1

        # Layer 2
        theta_front2 = psi_to_theta(psi_front, 2)
        theta_base2 = psi_to_theta(psi_base, 2)
        water2 = (theta_front2 - theta_base2) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta_front2 - theta_base2) if (theta_front2 - theta_base2) > 0 else QP.max_depth[0] + QP.max_depth[1]

        return QP.boundary_depths[2]

    def handle_overflow(FAmt, psi):
        """处理溢出,返回修正后的FAmt和overflow"""
        max_capacity = 0
        for i in range(3):
            theta = psi_to_theta(psi, i)
            max_capacity += (theta - QP.WCI[i]) * QP.max_depth[i]

        if FAmt > max_capacity:
            overflow = FAmt - max_capacity
            return max_capacity, overflow
        return FAmt, 0.0

    def handle_overflow_relative(FAmt, psi_front, psi_base):
        """处理相对溢出"""
        max_capacity = 0
        for i in range(3):
            theta_front = psi_to_theta(psi_front, i)
            theta_base = psi_to_theta(psi_base, i)
            max_capacity += (theta_front - theta_base) * QP.max_depth[i]

        if FAmt > max_capacity:
            overflow = FAmt - max_capacity
            return max_capacity, overflow
        return FAmt, 0.0

    def update_to_layer(WF, layer_idx):
        """更新WF到指定土层"""
        WF.WC = psi_to_theta(WF.psi, layer_idx)
        WF.WCHold = psi_to_theta(WF.psiHold, layer_idx)

    def clear_wf(WF):
        """清空wetting front"""
        WF.FAmt = 0.0
        WF.WC = 0.0
        WF.WCHold = 0.0
        WF.numRedist = 0.0
        WF.redistTime = 0.0
        WF.adjFactor = 0.0
        WF.z = 0.0

    # ===== 处理 WF[0] (正在再分配) =====
    WF0 = QP.WF[0]
    layer0 = get_layer(WF0.z)

    # 1. ODE积分
    WF0.WCHold = odeint(QP, 0, WF0.FAmt, QP.WCI[layer0],
                        WF0.WCHold, ts, te, 1.0e-4, dt, WF0.z, layer0)
    WF0.WC = WF0.WCHold - WF0.adjFactor

    # 2. 更新psi
    WF0.psiHold = theta_to_psi(WF0.WCHold, layer0)
    WF0.psi = theta_to_psi(WF0.WC, layer0)

    # 3. 计算新的z
    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)

    # 4. 检查跨层
    new_layer0 = get_layer(WF0.z)
    if new_layer0 != layer0:
        update_to_layer(WF0, new_layer0)
        WF0.psiHold = theta_to_psi(WF0.WCHold, new_layer0)
        WF0.psi = theta_to_psi(WF0.WC, new_layer0)

    # 5. 检查溢出
    if WF0.z > QP.boundary_depths[2]:
        WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
        QP.runoff += overflow * 0
        WF0.z = QP.boundary_depths[2]
        update_to_layer(WF0, 2)
        WF0.psiHold = theta_to_psi(WF0.WCHold, 2)
        WF0.psi = theta_to_psi(WF0.WC, 2)

    # ===== 处理 WF[1] (再分配,相对于WF[0]) =====
    WF1 = QP.WF[1]
    layer1 = get_layer(WF1.z)

    # 1. ODE积分
    wci_base = psi_to_theta(WF0.psiHold, layer1)
    WF1.WCHold = odeint(QP, 0, WF1.FAmt, wci_base,
                        WF1.WCHold, ts, te, 1.0e-4, dt, WF1.z, layer1)
    WF1.WC = WF1.WCHold - WF1.adjFactor

    # 2. 更新psi
    WF1.psiHold = theta_to_psi(WF1.WCHold, layer1)
    WF1.psi = theta_to_psi(WF1.WC, layer1)

    # 3. 计算相对于WF[0]的z
    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

    # 4. 检查跨层
    new_layer1 = get_layer(WF1.z)
    if new_layer1 != layer1:
        update_to_layer(WF1, new_layer1)
        WF1.psiHold = theta_to_psi(WF1.WCHold, new_layer1)
        WF1.psi = theta_to_psi(WF1.WC, new_layer1)

    # 5. 检查溢出
    if WF1.z > QP.boundary_depths[2]:
        WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
        QP.runoff += overflow * 0
        WF1.z = QP.boundary_depths[2]
        update_to_layer(WF1, 2)
        WF1.psiHold = theta_to_psi(WF1.WCHold, 2)
        WF1.psi = theta_to_psi(WF1.WC, 2)

    # ===== 检查WF[1]和WF[0]是否需要合并 =====
    if WF1.z >= WF0.z:
        # *** 合并WF[1]到WF[0] ***
        WF0.FAmt += WF1.FAmt

        # 先临时确定一个层
        temp_layer = get_layer(WF0.z)

        # 设置为该层的最大含水量
        WF0.WC = WF1.WC
        WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

        # 计算对应的psi
        WF0.psi = theta_to_psi(WF0.WC, temp_layer)
        WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

        WF0.numRedist = WF1.numRedist
        WF0.redistTime = WF1.redistTime
        WF0.adjFactor = WF1.adjFactor

        # WF[2]移到WF[1], WF[3]移到WF[2]
        WF1.FAmt = QP.WF[2].FAmt
        WF1.WC = QP.WF[2].WC
        WF1.WCHold = QP.WF[2].WCHold
        WF1.psi = QP.WF[2].psi
        WF1.psiHold = QP.WF[2].psiHold
        WF1.numRedist = QP.WF[2].numRedist
        WF1.redistTime = QP.WF[2].redistTime
        WF1.adjFactor = QP.WF[2].adjFactor
        WF1.z = QP.WF[2].z

        WF2 = QP.WF[2]
        WF2.FAmt = QP.WF[3].FAmt
        WF2.WC = QP.WF[3].WC
        WF2.WCHold = QP.WF[3].WCHold
        WF2.psi = QP.WF[3].psi
        WF2.psiHold = QP.WF[3].psiHold
        WF2.numRedist = QP.WF[3].numRedist
        WF2.redistTime = QP.WF[3].redistTime
        WF2.adjFactor = QP.WF[3].adjFactor
        WF2.z = QP.WF[3].z

        # 清空WF[3]
        clear_wf(QP.WF[3])

        QP.redistStatus = 30

        # **关键修正**: 重新计算WF[0]的位置，此时WF[0]右侧有WF[1]（原WF[2]）
        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
        final_layer0 = get_layer(WF0.z)

        # 如果层发生变化，更新到新层的WCS
        if final_layer0 != temp_layer:
            WF0.WC = QP.WCS[final_layer0]
            WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
            WF0.psi = theta_to_psi(WF0.WC, final_layer0)
            WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
            final_layer0 = get_layer(WF0.z)

        update_to_layer(WF0, final_layer0)

        # 检查WF[0]溢出
        if WF0.z > QP.boundary_depths[2]:
            WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
            QP.runoff += overflow * 0
            WF0.z = QP.boundary_depths[2]
            update_to_layer(WF0, 2)

        # 重新计算WF[1]的位置（现在是原WF[2]，再分配）
        layer1_new = get_layer(WF1.z)

        # ODE积分
        wci_base1 = psi_to_theta(WF0.psiHold, layer1_new)
        WF1.WCHold = odeint(QP, 0, WF1.FAmt, wci_base1,
                            WF1.WCHold, ts, te, 1.0e-4, dt, WF1.z, layer1_new)
        WF1.WC = WF1.WCHold - WF1.adjFactor

        # 更新psi
        WF1.psiHold = theta_to_psi(WF1.WCHold, layer1_new)
        WF1.psi = theta_to_psi(WF1.WC, layer1_new)

        # 计算相对于WF[0]的z
        WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

        # 检查跨层
        final_layer1 = get_layer(WF1.z)
        if final_layer1 != layer1_new:
            update_to_layer(WF1, final_layer1)

        # 检查溢出
        if WF1.z > QP.boundary_depths[2]:
            WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
            QP.runoff += overflow * 0
            WF1.z = QP.boundary_depths[2]
            update_to_layer(WF1, 2)

        # 再次检查WF[1]和WF[0]是否需要合并
        if WF1.z >= WF0.z:
            # *** 再次合并WF[1]到WF[0] ***
            WF0.FAmt += WF1.FAmt

            # 先临时确定一个层
            temp_layer = get_layer(WF0.z)

            # 设置为该层的最大含水量
            WF0.WC = WF1.WC
            WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

            # 计算对应的psi
            WF0.psi = theta_to_psi(WF0.WC, temp_layer)
            WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

            WF0.numRedist = WF1.numRedist
            WF0.redistTime = WF1.redistTime
            WF0.adjFactor = WF1.adjFactor

            # WF[2]移到WF[1]
            WF1.FAmt = WF2.FAmt
            WF1.WC = WF2.WC
            WF1.WCHold = WF2.WCHold
            WF1.psi = WF2.psi
            WF1.psiHold = WF2.psiHold
            WF1.numRedist = WF2.numRedist
            WF1.redistTime = WF2.redistTime
            WF1.adjFactor = WF2.adjFactor
            WF1.z = WF2.z

            # 清空WF[2]
            clear_wf(WF2)

            QP.redistStatus = 20

            # **关键修正**: 重新计算WF[0]的位置，此时WF[0]右侧有WF[1]（原WF[2]）
            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
            final_layer = get_layer(WF0.z)

            # 如果层发生变化，更新到新层的WCS
            if final_layer != temp_layer:
                WF0.WC = QP.WCS[final_layer]
                WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer])
                WF0.psi = theta_to_psi(WF0.WC, final_layer)
                WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer)
                WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                final_layer = get_layer(WF0.z)

            update_to_layer(WF0, final_layer)

            # 检查溢出
            if WF0.z > QP.boundary_depths[2]:
                WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                QP.runoff += overflow * 0
                WF0.z = QP.boundary_depths[2]
                update_to_layer(WF0, 2)

            # 重新计算WF[1]的位置（现在是原WF[3]，入渗）
            layer1_final = get_layer(WF1.z)

            # 更新psi（保持不变，因为是入渗front）
            WF1.psi = theta_to_psi(WF1.WC, layer1_final)
            WF1.psiHold = theta_to_psi(WF1.WCHold, layer1_final)

            # 计算相对于WF[0]的z
            WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

            # 检查跨层
            final_layer1 = get_layer(WF1.z)
            if final_layer1 != layer1_final:
                update_to_layer(WF1, final_layer1)

            # 检查溢出
            if WF1.z > QP.boundary_depths[2]:
                WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                QP.runoff += overflow * 0
                WF1.z = QP.boundary_depths[2]
                update_to_layer(WF1, 2)

            # 第三次检查WF[1]和WF[0]是否需要合并
            if WF1.z >= WF0.z:
                # *** 最后合并WF[1]到WF[0] ***
                WF0.FAmt += WF1.FAmt

                # 先临时确定一个层
                temp_layer = get_layer(WF0.z)

                # 设置为该层的最大含水量
                WF0.WC = QP.WCS[temp_layer]
                WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

                # 计算对应的psi
                WF0.psi = theta_to_psi(WF0.WC, temp_layer)
                WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

                WF0.numRedist = WF1.numRedist
                WF0.redistTime = WF1.redistTime
                WF0.adjFactor = WF1.adjFactor

                # 清空WF[1]
                clear_wf(WF1)

                QP.redistStatus = 10

                # 重新计算WF[0]
                WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                final_layer0 = get_layer(WF0.z)

                # 如果层发生变化，更新到新层的WCS
                if final_layer0 != temp_layer:
                    WF0.WC = QP.WCS[final_layer0]
                    WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
                    WF0.psi = theta_to_psi(WF0.WC, final_layer0)
                    WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
                    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                    final_layer0 = get_layer(WF0.z)

                update_to_layer(WF0, final_layer0)

                # 检查溢出
                if WF0.z > QP.boundary_depths[2]:
                    WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                    QP.runoff += overflow * 0
                    WF0.z = QP.boundary_depths[2]
                    update_to_layer(WF0, 2)
        else:
            # 仍然是3个front,更新WF[2] (正在入渗)
            WF2 = QP.WF[2]
            layer2 = get_layer(WF2.z)

            # 1. 更新psi (保持不变,因为是入渗front)
            WF2.psi = theta_to_psi(WF2.WC, layer2)
            WF2.psiHold = theta_to_psi(WF2.WCHold, layer2)

            # 2. 计算相对于WF[1]的z
            WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)

            # 3. 检查跨层
            new_layer2 = get_layer(WF2.z)
            if new_layer2 != layer2:
                update_to_layer(WF2, new_layer2)

            # 4. 检查溢出
            if WF2.z > QP.boundary_depths[2]:
                WF2.FAmt, overflow = handle_overflow_relative(WF2.FAmt, WF2.psi, WF1.psi)
                QP.runoff += overflow * 0
                WF2.z = QP.boundary_depths[2]
                update_to_layer(WF2, 2)

            # 5. 检查WF[2]和WF[1]是否需要合并
            if WF2.z >= WF1.z:
                # *** 合并WF[2]到WF[1] ***
                WF1.FAmt += WF2.FAmt

                # 先临时确定一个层
                temp_layer = get_layer(WF1.z)

                # 设置为该层的最大含水量
                WF1.WC = QP.WCS[temp_layer]
                WF1.WCHold = min(WF1.WC + WF2.adjFactor, QP.WCS[temp_layer])

                # 计算对应的psi
                WF1.psi = theta_to_psi(WF1.WC, temp_layer)
                WF1.psiHold = theta_to_psi(WF1.WCHold, temp_layer)

                WF1.numRedist = WF2.numRedist
                WF1.redistTime = WF2.redistTime
                WF1.adjFactor = WF2.adjFactor

                # 清空WF[2]
                clear_wf(WF2)

                QP.redistStatus = 20

                # 重新计算WF[1]的位置
                WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                final_layer1 = get_layer(WF1.z)

                # 如果层发生变化，更新到新层的WCS
                if final_layer1 != temp_layer:
                    WF1.WC = QP.WCS[final_layer1]
                    WF1.WCHold = min(WF1.WC + WF1.adjFactor, QP.WCS[final_layer1])
                    WF1.psi = theta_to_psi(WF1.WC, final_layer1)
                    WF1.psiHold = theta_to_psi(WF1.WCHold, final_layer1)
                    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                    final_layer1 = get_layer(WF1.z)

                update_to_layer(WF1, final_layer1)

                # 检查溢出
                if WF1.z > QP.boundary_depths[2]:
                    WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                    QP.runoff += overflow * 0
                    WF1.z = QP.boundary_depths[2]
                    update_to_layer(WF1, 2)

                # 再次检查WF[1]和WF[0]是否需要合并
                if WF1.z >= WF0.z:
                    # *** 最后合并WF[1]到WF[0] ***
                    WF0.FAmt += WF1.FAmt

                    # 先临时确定一个层
                    temp_layer = get_layer(WF0.z)

                    # 设置为该层的最大含水量
                    WF0.WC = QP.WCS[temp_layer]
                    WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

                    # 计算对应的psi
                    WF0.psi = theta_to_psi(WF0.WC, temp_layer)
                    WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

                    WF0.numRedist = WF1.numRedist
                    WF0.redistTime = WF1.redistTime
                    WF0.adjFactor = WF1.adjFactor

                    # 清空WF[1]
                    clear_wf(WF1)

                    QP.redistStatus = 10

                    # 重新计算WF[0]
                    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                    final_layer0 = get_layer(WF0.z)

                    # 如果层发生变化，更新到新层的WCS
                    if final_layer0 != temp_layer:
                        WF0.WC = QP.WCS[final_layer0]
                        WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
                        WF0.psi = theta_to_psi(WF0.WC, final_layer0)
                        WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
                        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                        final_layer0 = get_layer(WF0.z)

                    update_to_layer(WF0, final_layer0)

                    # 检查溢出
                    if WF0.z > QP.boundary_depths[2]:
                        WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                        QP.runoff += overflow * 0
                        WF0.z = QP.boundary_depths[2]
                        update_to_layer(WF0, 2)

    else:
        # 仍然是4个front，处理WF[2]（再分配，相对于WF[1]）
        WF2 = QP.WF[2]
        layer2 = get_layer(WF2.z)

        # 1. ODE积分
        wci_base2 = psi_to_theta(WF1.psiHold, layer2)
        WF2.WCHold = odeint(QP, 0, WF2.FAmt, wci_base2,
                            WF2.WCHold, ts, te, 1.0e-4, dt, WF2.z, layer2)
        WF2.WC = WF2.WCHold - WF2.adjFactor

        # 2. 更新psi
        WF2.psiHold = theta_to_psi(WF2.WCHold, layer2)
        WF2.psi = theta_to_psi(WF2.WC, layer2)

        # 3. 计算相对于WF[1]的z
        WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)

        # 4. 检查跨层
        new_layer2 = get_layer(WF2.z)
        if new_layer2 != layer2:
            update_to_layer(WF2, new_layer2)
            WF2.psiHold = theta_to_psi(WF2.WCHold, new_layer2)
            WF2.psi = theta_to_psi(WF2.WC, new_layer2)

        # 5. 检查溢出
        if WF2.z > QP.boundary_depths[2]:
            WF2.FAmt, overflow = handle_overflow_relative(WF2.FAmt, WF2.psi, WF1.psi)
            QP.runoff += overflow * 0
            WF2.z = QP.boundary_depths[2]
            update_to_layer(WF2, 2)
            WF2.psiHold = theta_to_psi(WF2.WCHold, 2)
            WF2.psi = theta_to_psi(WF2.WC, 2)

        # ===== 检查WF[2]和WF[1]是否需要合并 =====
        if WF2.z >= WF1.z:
            # *** 合并WF[2]到WF[1] ***
            WF1.FAmt += WF2.FAmt

            # 先临时确定一个层
            temp_layer = get_layer(WF1.z)

            # 设置为该层的最大含水量
            WF1.WC = WF2.WC
            WF1.WCHold = min(WF1.WC + WF2.adjFactor, QP.WCS[temp_layer])

            # 计算对应的psi
            WF1.psi = theta_to_psi(WF1.WC, temp_layer)
            WF1.psiHold = theta_to_psi(WF1.WCHold, temp_layer)

            WF1.numRedist = WF2.numRedist
            WF1.redistTime = WF2.redistTime
            WF1.adjFactor = WF2.adjFactor

            # WF[3]移到WF[2]
            WF2.FAmt = QP.WF[3].FAmt
            WF2.WC = QP.WF[3].WC
            WF2.WCHold = QP.WF[3].WCHold
            WF2.psi = QP.WF[3].psi
            WF2.psiHold = QP.WF[3].psiHold
            WF2.numRedist = QP.WF[3].numRedist
            WF2.redistTime = QP.WF[3].redistTime
            WF2.adjFactor = QP.WF[3].adjFactor
            WF2.z = QP.WF[3].z

            # 清空WF[3]
            clear_wf(QP.WF[3])

            QP.redistStatus = 30

            # **关键修正**: 重新计算WF[0]的位置，此时WF[0]右侧有WF[1]（原WF[2]）
            WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
            final_layer1 = get_layer(WF0.z)

            # 如果层发生变化，更新到新层的WCS
            if final_layer1 != temp_layer:
                WF1.WC = QP.WCS[final_layer1]
                WF1.WCHold = min(WF1.WC + WF1.adjFactor, QP.WCS[final_layer1])
                WF1.psi = theta_to_psi(WF1.WC, final_layer1)
                WF1.psiHold = theta_to_psi(WF1.WCHold, final_layer1)
                WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                final_layer1 = get_layer(WF1.z)

            update_to_layer(WF1, final_layer1)

            # 检查溢出
            if WF1.z > QP.boundary_depths[2]:
                WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                QP.runoff += overflow * 0
                WF1.z = QP.boundary_depths[2]
                update_to_layer(WF1, 2)

            # 再次检查WF[1]和WF[0]是否需要合并
            if WF1.z >= WF0.z:
                # *** 再次合并WF[1]到WF[0] ***
                WF0.FAmt += WF1.FAmt

                # 先临时确定一个层
                temp_layer = get_layer(WF0.z)

                # 设置为该层的最大含水量
                WF0.WC = WF1.WC
                WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

                # 计算对应的psi
                WF0.psi = theta_to_psi(WF0.WC, temp_layer)
                WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

                WF0.numRedist = WF1.numRedist
                WF0.redistTime = WF1.redistTime
                WF0.adjFactor = WF1.adjFactor

                # WF[2]移到WF[1]
                WF1.FAmt = WF2.FAmt
                WF1.WC = WF2.WC
                WF1.WCHold = WF2.WCHold
                WF1.psi = WF2.psi
                WF1.psiHold = WF2.psiHold
                WF1.numRedist = WF2.numRedist
                WF1.redistTime = WF2.redistTime
                WF1.adjFactor = WF2.adjFactor
                WF1.z = WF2.z

                # 清空WF[2]
                clear_wf(WF2)

                QP.redistStatus = 20

                # 重新计算WF[0]
                WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                final_layer0 = get_layer(WF0.z)

                # 如果层发生变化，更新到新层的WCS
                if final_layer0 != temp_layer:
                    WF0.WC = QP.WCS[final_layer0]
                    WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
                    WF0.psi = theta_to_psi(WF0.WC, final_layer0)
                    WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
                    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                    final_layer0 = get_layer(WF0.z)

                update_to_layer(WF0, final_layer0)

                # 检查溢出
                if WF0.z > QP.boundary_depths[2]:
                    WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                    QP.runoff += overflow * 0
                    WF0.z = QP.boundary_depths[2]
                    update_to_layer(WF0, 2)

                # 重新计算WF[1]的位置（现在是原WF[3]，入渗）
                layer1_final = get_layer(WF1.z)

                # 更新psi（保持不变，因为是入渗front）
                WF1.psi = theta_to_psi(WF1.WC, layer1_final)
                WF1.psiHold = theta_to_psi(WF1.WCHold, layer1_final)

                # 计算相对于WF[0]的z
                WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

                # 检查跨层
                final_layer1 = get_layer(WF1.z)
                if final_layer1 != layer1_final:
                    update_to_layer(WF1, final_layer1)

                # 检查溢出
                if WF1.z > QP.boundary_depths[2]:
                    WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                    QP.runoff += overflow * 0
                    WF1.z = QP.boundary_depths[2]
                    update_to_layer(WF1, 2)

                # 第三次检查WF[1]和WF[0]是否需要合并
                if WF1.z >= WF0.z:
                    # *** 最后合并WF[1]到WF[0] ***
                    WF0.FAmt += WF1.FAmt

                    # 先临时确定一个层
                    temp_layer = get_layer(WF0.z)

                    # 设置为该层的最大含水量
                    WF0.WC = QP.WCS[temp_layer]
                    WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

                    # 计算对应的psi
                    WF0.psi = theta_to_psi(WF0.WC, temp_layer)
                    WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

                    WF0.numRedist = WF1.numRedist
                    WF0.redistTime = WF1.redistTime
                    WF0.adjFactor = WF1.adjFactor

                    # 清空WF[1]
                    clear_wf(WF1)

                    QP.redistStatus = 10

                    # 重新计算WF[0]
                    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                    final_layer0 = get_layer(WF0.z)

                    # 如果层发生变化，更新到新层的WCS
                    if final_layer0 != temp_layer:
                        WF0.WC = QP.WCS[final_layer0]
                        WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
                        WF0.psi = theta_to_psi(WF0.WC, final_layer0)
                        WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
                        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                        final_layer0 = get_layer(WF0.z)

                    update_to_layer(WF0, final_layer0)

                    # 检查溢出
                    if WF0.z > QP.boundary_depths[2]:
                        WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                        QP.runoff += overflow * 0
                        WF0.z = QP.boundary_depths[2]
                        update_to_layer(WF0, 2)
            else:
                # 仍然是3个front,更新WF[2] (正在入渗)
                WF2 = QP.WF[2]
                layer2 = get_layer(WF2.z)

                # 1. 更新psi (保持不变,因为是入渗front)
                WF2.psi = theta_to_psi(WF2.WC, layer2)
                WF2.psiHold = theta_to_psi(WF2.WCHold, layer2)

                # 2. 计算相对于WF[1]的z
                WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)

                # 3. 检查跨层
                new_layer2 = get_layer(WF2.z)
                if new_layer2 != layer2:
                    update_to_layer(WF2, new_layer2)

                # 4. 检查溢出
                if WF2.z > QP.boundary_depths[2]:
                    WF2.FAmt, overflow = handle_overflow_relative(WF2.FAmt, WF2.psi, WF1.psi)
                    QP.runoff += overflow * 0
                    WF2.z = QP.boundary_depths[2]
                    update_to_layer(WF2, 2)

                # 5. 检查WF[2]和WF[1]是否需要合并
                if WF2.z >= WF1.z:
                    # *** 合并WF[2]到WF[1] ***
                    WF1.FAmt += WF2.FAmt

                    # 先临时确定一个层
                    temp_layer = get_layer(WF1.z)

                    # 设置为该层的最大含水量
                    WF1.WC = QP.WCS[temp_layer]
                    WF1.WCHold = min(WF1.WC + WF2.adjFactor, QP.WCS[temp_layer])

                    # 计算对应的psi
                    WF1.psi = theta_to_psi(WF1.WC, temp_layer)
                    WF1.psiHold = theta_to_psi(WF1.WCHold, temp_layer)

                    WF1.numRedist = WF2.numRedist
                    WF1.redistTime = WF2.redistTime
                    WF1.adjFactor = WF2.adjFactor

                    # 清空WF[2]
                    clear_wf(WF2)

                    QP.redistStatus = 20

                    # 重新计算WF[1]的位置
                    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                    final_layer1 = get_layer(WF1.z)

                    # 如果层发生变化，更新到新层的WCS
                    if final_layer1 != temp_layer:
                        WF1.WC = QP.WCS[final_layer1]
                        WF1.WCHold = min(WF1.WC + WF1.adjFactor, QP.WCS[final_layer1])
                        WF1.psi = theta_to_psi(WF1.WC, final_layer1)
                        WF1.psiHold = theta_to_psi(WF1.WCHold, final_layer1)
                        WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                        final_layer1 = get_layer(WF1.z)

                    update_to_layer(WF1, final_layer1)

                    # 检查溢出
                    if WF1.z > QP.boundary_depths[2]:
                        WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                        QP.runoff += overflow * 0
                        WF1.z = QP.boundary_depths[2]
                        update_to_layer(WF1, 2)

                    # 再次检查WF[1]和WF[0]是否需要合并
                    if WF1.z >= WF0.z:
                        # *** 最后合并WF[1]到WF[0] ***
                        WF0.FAmt += WF1.FAmt

                        # 先临时确定一个层
                        temp_layer = get_layer(WF0.z)

                        # 设置为该层的最大含水量
                        WF0.WC = QP.WCS[temp_layer]
                        WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

                        # 计算对应的psi
                        WF0.psi = theta_to_psi(WF0.WC, temp_layer)
                        WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

                        WF0.numRedist = WF1.numRedist
                        WF0.redistTime = WF1.redistTime
                        WF0.adjFactor = WF1.adjFactor

                        # 清空WF[1]
                        clear_wf(WF1)

                        QP.redistStatus = 10

                        # 重新计算WF[0]
                        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                        final_layer0 = get_layer(WF0.z)

                        # 如果层发生变化，更新到新层的WCS
                        if final_layer0 != temp_layer:
                            WF0.WC = QP.WCS[final_layer0]
                            WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
                            WF0.psi = theta_to_psi(WF0.WC, final_layer0)
                            WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
                            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                            final_layer0 = get_layer(WF0.z)

                        update_to_layer(WF0, final_layer0)

                        # 检查溢出
                        if WF0.z > QP.boundary_depths[2]:
                            WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                            QP.runoff += overflow * 0
                            WF0.z = QP.boundary_depths[2]
                            update_to_layer(WF0, 2)
        else:
            # 仍然是4个front，更新WF[3]（正在入渗）
            WF3 = QP.WF[3]
            layer3 = get_layer(WF3.z)

            # 1. 更新psi（保持不变，因为是入渗front）
            WF3.psi = theta_to_psi(WF3.WC, layer3)
            WF3.psiHold = theta_to_psi(WF3.WCHold, layer3)

            # 2. 计算相对于WF[2]的z
            WF3.z = calc_z_relative(WF3.FAmt, WF3.psi, WF2.psi)

            # 3. 检查跨层
            new_layer3 = get_layer(WF3.z)
            if new_layer3 != layer3:
                update_to_layer(WF3, new_layer3)

            # 4. 检查溢出
            if WF3.z > QP.boundary_depths[2]:
                WF3.FAmt, overflow = handle_overflow_relative(WF3.FAmt, WF3.psi, WF2.psi)
                QP.runoff += overflow * 0
                WF3.z = QP.boundary_depths[2]
                update_to_layer(WF3, 2)

            # 5. 检查WF[3]和WF[2]是否需要合并
            if WF3.z >= WF2.z:
                # *** 合并WF[3]到WF[2] ***
                WF2.FAmt += WF3.FAmt

                # 先临时确定一个层
                temp_layer = get_layer(WF2.z)

                # 设置为该层的最大含水量
                WF2.WC = QP.WCS[temp_layer]
                WF2.WCHold = min(WF2.WC + WF3.adjFactor, QP.WCS[temp_layer])

                # 计算对应的psi
                WF2.psi = theta_to_psi(WF2.WC, temp_layer)
                WF2.psiHold = theta_to_psi(WF2.WCHold, temp_layer)

                WF2.numRedist = WF3.numRedist
                WF2.redistTime = WF3.redistTime
                WF2.adjFactor = WF3.adjFactor

                # 清空WF[3]
                clear_wf(WF3)

                QP.redistStatus = 30

                # 重新计算WF[2]的位置
                WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)
                final_layer2 = get_layer(WF2.z)

                # 如果层发生变化，更新到新层的WCS
                if final_layer2 != temp_layer:
                    WF2.WC = QP.WCS[final_layer2]
                    WF2.WCHold = min(WF2.WC + WF2.adjFactor, QP.WCS[final_layer2])
                    WF2.psi = theta_to_psi(WF2.WC, final_layer2)
                    WF2.psiHold = theta_to_psi(WF2.WCHold, final_layer2)
                    WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)
                    final_layer2 = get_layer(WF2.z)

                update_to_layer(WF2, final_layer2)

                # 检查溢出
                if WF2.z > QP.boundary_depths[2]:
                    WF2.FAmt, overflow = handle_overflow_relative(WF2.FAmt, WF2.psi, WF1.psi)
                    QP.runoff += overflow * 0
                    WF2.z = QP.boundary_depths[2]
                    update_to_layer(WF2, 2)

                # 再次检查WF[2]和WF[1]是否需要合并
                if WF2.z >= WF1.z:
                    # *** 再次合并WF[2]到WF[1] ***
                    WF1.FAmt += WF2.FAmt

                    # 先临时确定一个层
                    temp_layer = get_layer(WF1.z)

                    # 设置为该层的最大含水量
                    WF1.WC = QP.WCS[temp_layer]
                    WF1.WCHold = min(WF1.WC + WF2.adjFactor, QP.WCS[temp_layer])

                    # 计算对应的psi
                    WF1.psi = theta_to_psi(WF1.WC, temp_layer)
                    WF1.psiHold = theta_to_psi(WF1.WCHold, temp_layer)

                    WF1.numRedist = WF2.numRedist
                    WF1.redistTime = WF2.redistTime
                    WF1.adjFactor = WF2.adjFactor

                    # 清空WF[2]
                    clear_wf(WF2)

                    QP.redistStatus = 20

                    # 重新计算WF[1]的位置
                    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                    final_layer1 = get_layer(WF1.z)

                    # 如果层发生变化，更新到新层的WCS
                    if final_layer1 != temp_layer:
                        WF1.WC = QP.WCS[final_layer1]
                        WF1.WCHold = min(WF1.WC + WF1.adjFactor, QP.WCS[final_layer1])
                        WF1.psi = theta_to_psi(WF1.WC, final_layer1)
                        WF1.psiHold = theta_to_psi(WF1.WCHold, final_layer1)
                        WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                        final_layer1 = get_layer(WF1.z)

                    update_to_layer(WF1, final_layer1)

                    # 检查溢出
                    if WF1.z > QP.boundary_depths[2]:
                        WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                        QP.runoff += overflow * 0
                        WF1.z = QP.boundary_depths[2]
                        update_to_layer(WF1, 2)

                    # 第三次检查WF[1]和WF[0]是否需要合并
                    if WF1.z >= WF0.z:
                        # *** 最后合并WF[1]到WF[0] ***
                        WF0.FAmt += WF1.FAmt

                        # 先临时确定一个层
                        temp_layer = get_layer(WF0.z)

                        # 设置为该层的最大含水量
                        WF0.WC = QP.WCS[temp_layer]
                        WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])

                        # 计算对应的psi
                        WF0.psi = theta_to_psi(WF0.WC, temp_layer)
                        WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

                        WF0.numRedist = WF1.numRedist
                        WF0.redistTime = WF1.redistTime
                        WF0.adjFactor = WF1.adjFactor

                        # 清空WF[1]
                        clear_wf(WF1)

                        QP.redistStatus = 10

                        # 重新计算WF[0]
                        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                        final_layer0 = get_layer(WF0.z)

                        # 如果层发生变化，更新到新层的WCS
                        if final_layer0 != temp_layer:
                            WF0.WC = QP.WCS[final_layer0]
                            WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer0])
                            WF0.psi = theta_to_psi(WF0.WC, final_layer0)
                            WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer0)
                            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
                            final_layer0 = get_layer(WF0.z)

                        update_to_layer(WF0, final_layer0)

                        # 检查溢出
                        if WF0.z > QP.boundary_depths[2]:
                            WF0.FAmt, overflow = handle_overflow(WF0.FAmt, WF0.psi)
                            QP.runoff += overflow * 0
                            WF0.z = QP.boundary_depths[2]
                            update_to_layer(WF0, 2)

    # 更新潜在入渗速率
    QP.fp = fp_calc(QP)


def fp_calc(QP):
    if QP.redistStatus == 40:
        QP.WF[3].FAmt = QP.cummInfil - QP.WF[2].FAmt - QP.WF[1].FAmt - QP.WF[0].FAmt
        if QP.WF[3].z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
            QP.Sav = G(psi_of_theta(QP.WCS[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]),
                       psi_of_theta(QP.WF[2].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0],
                       QP.n[0], QP.m[0], QP.ks[0])
            QP.fp = QP.ks_composite + (QP.ks[0] * QP.Sav / QP.WF[3].z)
        elif QP.boundary_depths[0] < QP.WF[3].z <= QP.boundary_depths[1]:
            QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[3].z - QP.max_depth[0]) / QP.ks[1])
            QP.Sav = G(psi_of_theta(QP.WCS[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]),
                       psi_of_theta(QP.WF[2].WC, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1],
                       QP.n[1], QP.m[1], QP.ks[1])
            QP.fp = QP.ks_composite + (QP.ks[1] * QP.Sav / QP.WF[3].z)
        elif QP.boundary_depths[1] < QP.WF[3].z <= QP.boundary_depths[2]:
            QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (
                    QP.WF[3].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
            QP.Sav = G(psi_of_theta(QP.WCS[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]),
                       psi_of_theta(QP.WF[2].WC, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2],
                       QP.n[2], QP.m[2], QP.ks[2])
            QP.fp = QP.ks_composite + (QP.ks[2] * QP.Sav / QP.WF[3].z)
    elif QP.redistStatus in [30, 31]:
        QP.WF[2].FAmt = QP.cummInfil - QP.WF[1].FAmt - QP.WF[0].FAmt
        if QP.WF[2].z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
            QP.Sav = G(psi_of_theta(QP.WCS[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]),
                       psi_of_theta(QP.WF[1].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0],
                       QP.n[0], QP.m[0], QP.ks[0])
            QP.fp = QP.ks_composite + (QP.ks[0] * QP.Sav / QP.WF[2].z)
        elif QP.boundary_depths[0] < QP.WF[2].z <= QP.boundary_depths[1]:
            QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[2].z - QP.max_depth[0]) / QP.ks[1])
            QP.Sav = G(psi_of_theta(QP.WCS[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]),
                       psi_of_theta(QP.WF[1].WC, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1],
                       QP.n[1], QP.m[1], QP.ks[1])
            QP.fp = QP.ks_composite + (QP.ks[1] * QP.Sav / QP.WF[2].z)
        elif QP.boundary_depths[1] < QP.WF[2].z <= QP.boundary_depths[2]:
            QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (
                    QP.WF[2].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
            QP.Sav = G(psi_of_theta(QP.WCS[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]),
                       psi_of_theta(QP.WF[1].WC, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2],
                       QP.n[2], QP.m[2], QP.ks[2])
            QP.fp = QP.ks_composite + (QP.ks[2] * QP.Sav / QP.WF[2].z)
    elif QP.redistStatus in [20, 21]:
        QP.WF[1].FAmt = QP.cummInfil - QP.WF[0].FAmt
        if QP.WF[1].z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
            QP.Sav = G(psi_of_theta(QP.WCS[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]),
                       psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0],
                       QP.n[0], QP.m[0], QP.ks[0])
            QP.fp = QP.ks_composite + (QP.ks[0] * QP.Sav / QP.WF[1].z)
        elif QP.boundary_depths[0] < QP.WF[1].z <= QP.boundary_depths[1]:
            QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[1].z - QP.max_depth[0]) / QP.ks[1])
            QP.Sav = G(psi_of_theta(QP.WCS[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]),
                       psi_of_theta(QP.WF[0].WC, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1],
                       QP.n[1], QP.m[1], QP.ks[1])
            QP.fp = QP.ks_composite + (QP.ks[1] * QP.Sav / QP.WF[1].z)
        elif QP.boundary_depths[1] < QP.WF[1].z <= QP.boundary_depths[2]:
            QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (
                    QP.WF[1].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
            QP.Sav = G(psi_of_theta(QP.WCS[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]),
                       psi_of_theta(QP.WF[0].WC, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2],
                       QP.n[2], QP.m[2], QP.ks[2])
            QP.fp = QP.ks_composite + (QP.ks[2] * QP.Sav / QP.WF[1].z)
    else:
        QP.WF[0].FAmt = QP.cummInfil
        if QP.WF[0].z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
            QP.Sav = G(psi_of_theta(QP.WCS[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]),
                       psi_of_theta(QP.WCI[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0],
                       QP.n[0], QP.m[0], QP.ks[0])
            QP.fp = QP.ks_composite + (QP.ks[0] * QP.Sav / QP.WF[0].z)
        elif QP.boundary_depths[0] < QP.WF[0].z <= QP.boundary_depths[1]:
            QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[0].z - QP.max_depth[0]) / QP.ks[1])
            QP.Sav = G(psi_of_theta(QP.WCS[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]),
                       psi_of_theta(QP.WCI[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1],
                       QP.n[1], QP.m[1], QP.ks[1])
            QP.fp = QP.ks_composite + (QP.ks[1] * QP.Sav / QP.WF[0].z)
        elif QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
            QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (
                    QP.WF[0].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
            QP.Sav = G(psi_of_theta(QP.WCS[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]),
                       psi_of_theta(QP.WCI[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2],
                       QP.n[2], QP.m[2], QP.ks[2])
            QP.fp = QP.ks_composite + (QP.ks[2] * QP.Sav / QP.WF[0].z)
    return QP.fp


# 根据土壤的再分配状态更新地表含水量 (WCO)
def calc_wco(QP):
    if QP.redistStatus == 10 or QP.redistStatus == 11:
        QP.WCO = QP.WF[0].WC
    elif QP.redistStatus == 20 or QP.redistStatus == 21:
        QP.WCO = QP.WF[1].WC
    elif QP.redistStatus == 30 or QP.redistStatus == 31:
        QP.WCO = QP.WF[2].WC
    else:
        QP.WCO = QP.WF[3].WC

    return


#  Q                    方程组的索引
#  F                    累积入渗量
#  WCA                  当前含水量
#  WCB                  之前的含水量
#  QP                   包含各方程组参数的字典
def calc_redist_zf(QP, F, WCA, WCB):
    result = F / (WCA - WCB)
    return result


# 计算相对含水量
def calc_rwc(QP):
    if QP.redistStatus in (10, 11):
        QP.RWC = (QP.WF[0].WC - QP.WCR) / (QP.WCS - QP.WCR)
    elif QP.redistStatus in (20, 21):
        QP.RWC = (QP.WF[1].WC - QP.WCR) / (QP.WCS - QP.WCR)
    elif QP.redistStatus in (30, 31):
        QP.RWC = (QP.WF[2].WC - QP.WCR) / (QP.WCS - QP.WCR)
    else:
        QP.RWC = (QP.WF[3].WC - QP.WCR) / (QP.WCS - QP.WCR)


# 当降水量小于渗透系数且无积水时，执行水分再分配过程
def redistribution(QP, tend, dT):
    ts = tend - dT
    te = tend

    # 计算每个湿润锋的gamma系数
    for j in range(0, 3):
        QP.WF[j].adjFactor = adj_factor_calc(QP, QP.WF[j].numRedist, QP.WF[j].redistTime, QP.WF[j].z)

    # 根据 redistStatus 进行不同的重分配处理
    if QP.redistStatus == 11:
        redist11(QP, 0.0, ts, te, dT)
    elif QP.redistStatus == 21:
        redist21(QP, 0.0, ts, te, dT)
    elif QP.redistStatus == 31:
        redist31(QP, 0.0, ts, te, dT)


# 计算潜在的累积入渗量
def cumm_infil_at_pond(QP, WC, S):
    rk = QP.precipRate / QP.ks
    result = S * (QP.WCS - WC) / (rk - 1)
    return result


# 计算积水时间
def calc_time_to_pond(QP):
    result = QP.BFp / QP.PrecipRate
    return result


# 计算时间偏移
def calc_tpp(QP, W, S):
    QP.tpp = (QP.Bfp - S * (QP.WCS - W) * math.log(1 + (QP.Bfp / (S * (QP.WCS - W))))) / QP.ks
    return QP.tpp


# 使用牛顿-拉夫森法计算累积入渗量
def calc_f_newton(QP, Fnew, t, tp, tpp, wcs, wci, Sav, Ks, ks_composite):
    tolerance = 1.0E-6
    error = 1.0
    Fold = Fnew + 1.0E-6
    WCS_WC = wcs - wci
    g = Fold - Ks * Sav * WCS_WC * math.log(1.0 + (ks_composite * Fold / (Ks * Sav * WCS_WC)))/ks_composite - ks_composite * (t - tp + tpp)

    while error > tolerance:
        dgdf = 1.0 - (Ks * Sav * WCS_WC) / (Ks * Sav * WCS_WC + ks_composite * Fold)
        Fnew = Fold - (g / dgdf)
        Fold = Fnew
        g = Fold - Ks * Sav * WCS_WC * math.log(1.0 + (ks_composite * Fold / (Ks * Sav * WCS_WC)))/ks_composite - ks_composite * (t - tp + tpp)
        error = abs(g)

    return Fnew


# 计算 Srivastava, Costello, Edwards 提出的近似累积入渗量解
def calc_f_approx(QP, Q, Fnew, t, tp, tpp, WC, S):
    alpha = 1.851
    beta = 0.565
    delta = 0.004
    phi = QP.ks * (t - tp + tpp) / (S * (QP.WCS - WC))

    if phi > 0.0001:
        if phi <= 0.095:
            alpha = 1.851
            beta = 0.565
            delta = 0.004
        elif phi <= 0.911:
            alpha = 2.137
            beta = 0.667
            delta = 0.021
        elif phi <= 17.00:
            alpha = 2.141
            beta = 0.689
            delta = 0.035
        else:
            return 0.0  # 如果 phi > 17，退出并返回 0.0

    lambda_val = math.pow(phi, (beta + delta * math.log(phi)))
    result = alpha * (S * (QP.WCS - WC)) * lambda_val

    return result


# 计算剩余积水入渗所需的时间
def calc_t_newton(QP, tstart, tend, tp, tpp, precip, total_infil, WC, S):
    tolerance = 1.0E-6
    error = 1.0
    t1new = tend
    t1old = t1new

    # 初始F计算
    F = (t1old - tstart) * precip + total_infil

    # 计算lnarg和g
    lnarg = 1.0 + (F / (S * (QP.WCS - WC)))
    g = F - (S * (QP.WCS - WC)) * math.log(lnarg) - QP.ks * (t1old - tp + tpp)

    while error > tolerance:
        # 计算 dg/d(deltaT1)
        dgdeltaT1 = precip - precip * (1.0 / lnarg) - QP.ks

        # 迭代t1new
        t1new = t1old - (g / dgdeltaT1)
        t1old = t1new

        # 更新F和lnarg
        F = (t1old - tstart) * precip + total_infil
        lnarg = 1.0 + (F / (S * (QP.WCS - WC)))

        # 重新计算g
        g = F - (S * (QP.WCS - WC)) * math.log(lnarg) - QP.ks * (t1old - tp + tpp)

        # 更新误差
        error = abs(g)

    return t1new


# ===================== 新增:蒸发相关函数 =====================

def calculate_et_demand(QP, PET_rate, time_step):
    """
    计算实际蒸散发需求(AET demand)
    使用van Genuchten方法,参考LGAR实现

    注意：此函数只读取QP的值进行计算，不修改QP的任何属性

    参数:
        QP: 参数对象（只读）
        PET_rate: 潜在蒸散发速率 (mm/h)
        time_step: 时间步长 (h)

    返回:
        actual_ET_demand: 实际蒸散发需求 (mm)
    """
    epsilon = 1e-10

    if PET_rate <= epsilon:
        return 0.0

    # 直接使用第一个湿润前缘的含水量（与LGAR一致）
    surface_theta = QP.WF[0].WC
    surface_theta = np.clip(surface_theta, QP.WCR[0], QP.WCS[0])

    # 计算field capacity对应的含水量
    relative_moisture_fc = getattr(QP, 'relative_moisture_at_which_PET_equals_AET', 0.5)
    theta_fc = (QP.WCS[0] - QP.WCR[0]) * relative_moisture_fc + QP.WCR[0]

    # 获取凋萎点(如果没有定义,计算它)
    wilting_point_theta = getattr(QP, 'wilting_point_theta', None)
    if wilting_point_theta is None:
        wilting_point_head = 154950  # cm（标准值 -1500 kPa）
        # 只计算第一层的凋萎点
        wilting_point_theta_0 = (QP.WCR[0] +
                                 (QP.WCS[0] - QP.WCR[0]) /
                                 (1 + (QP.alpha[0] * wilting_point_head) ** QP.n[0]) ** QP.m[0])
    else:
        # 如果QP有定义，提取第一层的值
        if isinstance(wilting_point_theta, (list, np.ndarray)):
            wilting_point_theta_0 = wilting_point_theta[0]
        else:
            wilting_point_theta_0 = wilting_point_theta

    # 计算50%有效含水量对应的含水量
    theta_50 = (theta_fc - wilting_point_theta_0) * 0.5 + wilting_point_theta_0

    # 转换为压力水头
    h_50 = psi_of_theta(theta_50, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])

    # 计算当前含水量对应的psi值
    h = psi_of_theta(surface_theta, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])

    # 使用van Genuchten型ET函数: AET = PET * 1/(1+(h/h_50)^3)
    try:
        et_factor = 1.0 / (1.0 + (h / h_50) ** 3)
        et_factor = np.clip(et_factor, 0.0, 1.0)
    except:
        et_factor = 0.0

    actual_ET_demand = PET_rate * et_factor * time_step

    # 确保ET不超过PET且非负
    actual_ET_demand = np.clip(actual_ET_demand, 0.0, PET_rate * time_step)

    return actual_ET_demand


# ========== 以下函数可以删除或注释掉 ==========
# 如果您不需要应用ET到土壤，可以删除这两个函数

# def apply_et_to_soil(QP, actual_ET_demand, wf_idx=0):
#     """
#     此函数已废弃 - 因为不需要修改QP状态
#     """
#     pass

# def _get_layer_index(QP, z):
#     """
#     此函数已废弃 - 因为不需要修改QP状态
#     """
#     pass

# ===================== 蒸发函数结束 =====================


# 在积水条件下计算入渗率、累积入渗量、径流和地表存储量；并调用下层的再分配过程
def calc_pond(QP, tstart, tend, tp, tpp, deltaT, R, upstream_water_depth):
    interTime = tstart
    deltaTerr = tend - interTime
    if deltaTerr > 1E-10:
        interTime = tend
        QP.time = interTime
        QP.pondingAmt = QP.pondingAmt
        QP.cummPrecip += deltaTerr * R
        amtWater = R * deltaTerr
        Fhold = QP.cummInfil
        t = interTime
        Fnew = QP.cummInfil + amtWater

        if QP.redistStatus in [10, 11]:
            if QP.WF[0].z <= QP.boundary_depths[0]:
                QP.wcs = QP.WCS[0]
                QP.wci = QP.WCI[0]
                QP.Ks = QP.ks[0]
                QP.ks_composite = QP.ks[0]
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
            elif QP.boundary_depths[0] < QP.WF[0].z <= QP.boundary_depths[1]:
                QP.wcs = QP.WCS[1]
                QP.wci = QP.WCI[1]
                QP.Ks = QP.ks[1]
                QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[0].z - QP.max_depth[0]) / QP.ks[1])
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
            elif QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
                QP.wcs = QP.WCS[2]
                QP.wci = QP.WCI[2]
                QP.Ks = QP.ks[2]
                QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[0].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

            QP.cummInfil = calc_f_newton(QP, Fnew, t, tp, tpp, QP.wcs, QP.wci, QP.Sav, QP.Ks, QP.ks_composite)
            QP.WF[0].FAmt = QP.WF[0].FAmt + QP.cummInfil - Fhold
            QP.runoff2 = Fnew - QP.cummInfil
            if (QP.runoff2 + QP.pondingAmt) >= upstream_water_depth:
                QP.runoff += QP.runoff2 - upstream_water_depth + QP.pondingAmt
                QP.runoff2 = 0
                QP.pondingAmt = upstream_water_depth
            else:
                QP.pondingAmt += QP.runoff2
                QP.runoff2 = 0
                QP.runoff = QP.runoff

        elif QP.redistStatus in [20, 21]:
            if QP.WF[1].z <= QP.boundary_depths[0]:
                QP.wcs = QP.WCS[0]
                QP.wci = theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                QP.Ks = QP.ks[0]
                QP.ks_composite = QP.ks[0]
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
            elif QP.boundary_depths[0] < QP.WF[1].z <= QP.boundary_depths[1]:
                QP.wcs = QP.WCS[1]
                QP.wci = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.Ks = QP.ks[1]
                QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[1].z - QP.max_depth[0]) / QP.ks[1])
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
            elif QP.boundary_depths[1] < QP.WF[1].z <= QP.boundary_depths[2]:
                QP.wcs = QP.WCS[2]
                QP.wci = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.Ks = QP.ks[2]
                QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[1].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

            QP.cummInfil = calc_f_newton(QP, Fnew, t, tp, tpp, QP.wcs, QP.wci, QP.Sav, QP.Ks, QP.ks_composite)
            QP.WF[1].FAmt = QP.WF[1].FAmt + QP.cummInfil - Fhold
            QP.runoff2 = Fnew - QP.cummInfil
            if (QP.runoff2 + QP.pondingAmt) >= upstream_water_depth:
                QP.runoff += QP.runoff2 - upstream_water_depth + QP.pondingAmt
                QP.runoff2 = 0
                QP.pondingAmt = upstream_water_depth
            else:
                QP.pondingAmt += QP.runoff2
                QP.runoff2 = 0
                QP.runoff = QP.runoff

        elif QP.redistStatus in [30, 31]:
            if QP.WF[2].z <= QP.boundary_depths[0]:
                QP.wcs = QP.WCS[0]
                QP.wci = theta_of_psi(QP.WF[1].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                QP.Ks = QP.ks[0]
                QP.ks_composite = QP.ks[0]
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
            elif QP.boundary_depths[0] < QP.WF[2].z <= QP.boundary_depths[1]:
                QP.wcs = QP.WCS[1]
                QP.wci = theta_of_psi(QP.WF[1].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.Ks = QP.ks[1]
                QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[2].z - QP.max_depth[0]) / QP.ks[1])
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
            elif QP.boundary_depths[1] < QP.WF[2].z <= QP.boundary_depths[2]:
                QP.wcs = QP.WCS[2]
                QP.wci = theta_of_psi(QP.WF[1].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.Ks = QP.ks[2]
                QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[2].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])
            QP.cummInfil = calc_f_newton(QP, Fnew, t, tp, tpp, QP.wcs, QP.wci, QP.Sav, QP.Ks, QP.ks_composite)
            QP.WF[2].FAmt = QP.WF[2].FAmt + QP.cummInfil - Fhold
            QP.runoff2 = Fnew - QP.cummInfil
            if (QP.runoff2 + QP.pondingAmt) >= upstream_water_depth:
                QP.runoff += QP.runoff2 - upstream_water_depth + QP.pondingAmt
                QP.runoff2 = 0
                QP.pondingAmt = upstream_water_depth
            else:
                QP.pondingAmt += QP.runoff2
                QP.runoff2 = 0
                QP.runoff = QP.runoff

        else:
            if QP.WF[3].z <= QP.boundary_depths[0]:
                QP.wcs = QP.WCS[0]
                QP.wci = theta_of_psi(QP.WF[2].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                QP.Ks = QP.ks[0]
                QP.ks_composite = QP.ks[0]
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
            elif QP.boundary_depths[0] < QP.WF[3].z <= QP.boundary_depths[1]:
                QP.wcs = QP.WCS[1]
                QP.wci = theta_of_psi(QP.WF[2].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.Ks = QP.ks[1]
                QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[3].z - QP.max_depth[0]) / QP.ks[1])
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
            elif QP.boundary_depths[1] < QP.WF[3].z <= QP.boundary_depths[2]:
                QP.wcs = QP.WCS[2]
                QP.wci = theta_of_psi(QP.WF[2].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.Ks = QP.ks[2]
                QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[3].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])
            QP.cummInfil = calc_f_newton(QP, Fnew, t, tp, tpp, QP.wcs, QP.wci, QP.Sav, QP.Ks, QP.ks_composite)
            QP.WF[3].FAmt = QP.WF[3].FAmt + QP.cummInfil - Fhold
            QP.runoff2 = Fnew - QP.cummInfil
            if (QP.runoff2 + QP.pondingAmt) >= upstream_water_depth:
                QP.runoff += QP.runoff2 - upstream_water_depth + QP.pondingAmt
                QP.runoff2 = 0
                QP.pondingAmt = upstream_water_depth
            else:
                QP.pondingAmt += QP.runoff2
                QP.runoff2 = 0
                QP.runoff = QP.runoff

        # ===== 新增: 计算并应用ET =====
        if hasattr(QP, 'PET_rate') and QP.PET_rate > 0:
            actual_ET_demand = calculate_et_demand(QP, QP.PET_rate, deltaTerr)
            if hasattr(QP, 'actual_ET_vec'):
                QP.actual_ET_vec.append(actual_ET_demand)
        # ===== ET处理结束 =====

        # Reallocate water
        for j in range(0, 3):
            QP.WF[j].adjFactor = adj_factor_calc(QP, QP.WF[j].numRedist, (QP.WF[j].redistTime - deltaT + deltaTerr), QP.WF[j].z)

        if QP.redistStatus == 11:
            redist11(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 20:
            redist20(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 21:
            redist21(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 30:
            redist30(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 31:
            redist31(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 40:
            redist40(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        else:  # redistStatus = 10 时，不进行再分配，只需将 wf1 向下移动
            if QP.WF[0].z <= QP.boundary_depths[0]:
                QP.WF[0].psi = psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                QP.WF[0].psiHold = psi_of_theta(QP.WF[0].WCHold, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                QP.WF[0].z = calc_redist_zf(QP, QP.WF[0].FAmt, theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCI[0])
                QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                if QP.boundary_depths[0] < QP.WF[0].z <= QP.boundary_depths[1]:
                    QP.WF[0].z = QP.max_depth[0] + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0]) / (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                elif QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
                    QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[1].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]) / (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                elif QP.WF[0].z > QP.boundary_depths[2]:
                    QP.WF[0].z = QP.boundary_depths[2]
                    # QP.runoff = QP.runoff + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] - (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].FAmt = (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
            elif QP.boundary_depths[0] < QP.WF[0].z <= QP.boundary_depths[1]:
                QP.WF[0].psi = psi_of_theta(QP.WF[0].WC, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.WF[0].psiHold = psi_of_theta(QP.WF[0].WCHold, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.WF[0].z = QP.max_depth[0] + calc_redist_zf(QP, QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0], theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.WCI[1])
                QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                if QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
                    QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]) / (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                elif QP.WF[0].z > QP.boundary_depths[2]:
                    QP.WF[0].z = QP.boundary_depths[2]
                    # QP.runoff = QP.runoff + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] - (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].FAmt = (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
            elif QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
                QP.WF[0].psi = psi_of_theta(QP.WF[0].WC, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].psiHold = psi_of_theta(QP.WF[0].WCHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + calc_redist_zf(QP, QP.WF[0].FAmt - ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]), theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.WCI[2])
                QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                if QP.WF[0].z > QP.boundary_depths[2]:
                    QP.WF[0].z = QP.boundary_depths[2]
                    # QP.runoff = QP.runoff + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] - (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].FAmt = (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])

    calc_wco(QP)
    calc_rwc(QP)

    return


# 在无积水条件下计算入渗率、累积入渗量、径流和地表存储量，并调用下层的再分配过程
def calc_no_pond(QP, tstart, tend, deltaT, R, upstream_water_depth):
    nok = 0
    nbad = 0
    nsteps = int((tend - tstart) / deltaT)
    interTime = tstart

    # 检查是否确实在周期结束时
    deltaTerr = tend - interTime
    if deltaTerr > 1E-10:
        interTime = tend
        QP.time = interTime
        QP.cummPrecip += deltaTerr * R
        QP.cummInfil += R * deltaTerr
        QP.pondingAmt = QP.pondingAmt
        QP.runoff = QP.runoff

        # ===== 新增: 计算并应用ET =====
        if hasattr(QP, 'PET_rate') and QP.PET_rate > 0:
            actual_ET_demand = calculate_et_demand(QP, QP.PET_rate, deltaTerr)
            if hasattr(QP, 'actual_ET_vec'):
                QP.actual_ET_vec.append(actual_ET_demand)
        # ===== ET处理结束 =====

        # 重新分配水分
        for j in range(0, 3):
            QP.WF[j].adjFactor = adj_factor_calc(QP, QP.WF[j].numRedist, (QP.WF[j].redistTime - deltaT + deltaTerr), QP.WF[j].z)

        # 估算积水前的水分含量上升
        if QP.pondFlag == 0 and R > 0.0 and QP.WF[0].WC == 0.0:
            # expon = 1 / (3 + (2 / QP.lamda))
            # QP.WF[0].WC = QP.WCI + (QP.WCS - QP.WCI) * ((R / QP.fp) ** expon)
            QP.WF[0].WC = QP.WCS[0]
            QP.WF[0].WCHold = QP.WF[0].WC
            QP.WF[0].FAmt = QP.WF[0].FAmt + deltaTerr * R
            if QP.WF[0].FAmt <= (QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0]:
                QP.WF[0].z = calc_redist_zf(QP, QP.WF[0].FAmt, QP.WF[0].WC, QP.WCI[0])
            elif (QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0] < QP.WF[0].FAmt <= ((QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]):
                QP.WF[0].z = QP.max_depth[0] + calc_redist_zf(QP, QP.WF[0].FAmt - (QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0], theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.WCI[1])
                QP.WF[0].WC = theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.WF[0].WCHold = theta_of_psi(psi_of_theta(QP.WF[0].WCHold, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
            elif ((QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]) < QP.WF[0].FAmt <= ((QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]):
                QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + calc_redist_zf(QP, QP.WF[0].FAmt - ((QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]), theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.WCI[2])
                QP.WF[0].WC = theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].WCHold = theta_of_psi(psi_of_theta(QP.WF[0].WCHold, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
            elif QP.WF[0].FAmt > ((QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]):
                QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + QP.max_depth[2]
                # QP.runoff = QP.runoff + QP.WF[0].FAmt - ((QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                QP.WF[0].FAmt = ((QP.WF[0].WC - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                QP.WF[0].WC = theta_of_psi(psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].WCHold = theta_of_psi(psi_of_theta(QP.WF[0].WCHold, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
        elif QP.redistStatus == 11:
            QP.WF[0].FAmt = QP.WF[0].FAmt + deltaTerr * R
            redist11(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 20:
            QP.WF[1].FAmt = QP.WF[1].FAmt + deltaTerr * R
            redist20(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 21:
            QP.WF[1].FAmt = QP.WF[1].FAmt + deltaTerr * R
            redist21(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 30:
            QP.WF[2].FAmt = QP.WF[2].FAmt + deltaTerr * R
            redist30(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 31:
            QP.WF[2].FAmt = QP.WF[2].FAmt + deltaTerr * R
            redist31(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        elif QP.redistStatus == 40:
            QP.WF[3].FAmt = QP.WF[3].FAmt + deltaTerr * R
            redist40(QP, R, interTime - deltaTerr, interTime, deltaTerr)
        else:  # redistStatus = 10, wf1 不需再分配，因为它正在入渗
            if QP.WF[0].z <= QP.boundary_depths[0]:
                QP.WF[0].psi = psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                QP.WF[0].psiHold = psi_of_theta(QP.WF[0].WCHold, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                QP.WF[0].z = calc_redist_zf(QP, QP.WF[0].FAmt, theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCI[0])
                QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                if QP.boundary_depths[0] < QP.WF[0].z <= QP.boundary_depths[1]:
                    QP.WF[0].z = QP.max_depth[0] + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0]) / (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                elif QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
                    QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[1].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]) / (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                elif QP.WF[0].z > QP.boundary_depths[2]:
                    QP.WF[0].z = QP.boundary_depths[2]
                    # QP.runoff = QP.runoff + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] - (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].FAmt = (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
            elif QP.boundary_depths[0] < QP.WF[0].z <= QP.boundary_depths[1]:
                QP.WF[0].psi = psi_of_theta(QP.WF[0].WC, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.WF[0].psiHold = psi_of_theta(QP.WF[0].WCHold, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.WF[0].z = QP.max_depth[0] + calc_redist_zf(QP, QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0], theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.WCI[1])
                QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                if QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
                    QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]) / (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                elif QP.WF[0].z > QP.boundary_depths[2]:
                    QP.WF[0].z = QP.boundary_depths[2]
                    # QP.runoff = QP.runoff + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] - (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].FAmt = (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
            elif QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
                QP.WF[0].psi = psi_of_theta(QP.WF[0].WC, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].psiHold = psi_of_theta(QP.WF[0].WCHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + calc_redist_zf(QP, QP.WF[0].FAmt - ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]), theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.WCI[2])
                QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                if QP.WF[0].z > QP.boundary_depths[2]:
                    QP.WF[0].z = QP.boundary_depths[2]
                    # QP.runoff = QP.runoff + (QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] - (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] - (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].FAmt = (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])

    return


# 计算潜在入渗量、积水时间，并调用相应的入渗子程序：积水或非积水
# 更新湿润锋的再分配状态
# 如果降水量小于渗透系数，则调用再分配过程
def calc_infil(QP, tstart, tend, deltaT, upstream_water_depth):
    # 更新再分配次数和时间
    Z_list = [QP.WF[0].z, QP.WF[1].z, QP.WF[2].z, QP.WF[3].z]
    index = np.count_nonzero(Z_list) - 1
    if QP.WF[index].z <= QP.boundary_depths[0]:
        QP.ks_composite = QP.ks[0]
    elif QP.boundary_depths[0] < QP.WF[index].z <= QP.boundary_depths[1]:
        QP.ks_composite = QP.WF[index].z / (
                QP.max_depth[0] / QP.ks[0] + (QP.WF[index].z - QP.max_depth[0]) / QP.ks[1])
    elif QP.boundary_depths[1] < QP.WF[index].z <= QP.boundary_depths[2]:
        QP.ks_composite = QP.WF[index].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (
                QP.WF[index].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])

    QP.precipRate += QP.pondingAmt/deltaT
    QP.pondingAmt = 0

    if (QP.redistStatus in [10, 20, 30, 40]) and (QP.precipRate < QP.ks_composite) and (QP.pondingAmt <= 1E-10) and QP.WF[0].z != 0:
        QP.numRedist += 1
        if QP.redistStatus == 10:
            QP.redistTime = 0.0
            QP.WF[0].redistTime = 0.0
            QP.WF[0].numRedist = QP.numRedist
        elif QP.redistStatus == 20:
            QP.redistTime = 0.0
            QP.WF[1].redistTime = 0.0
            QP.WF[1].numRedist = QP.numRedist
        elif QP.redistStatus == 30:
            QP.redistTime = 0.0
            QP.WF[2].redistTime = 0.0
            QP.WF[2].numRedist = QP.numRedist
        elif QP.redistStatus == 40:
            # wf1和wf2合并; 前wf1和2前wf1变成新wf1; 前wf3变成新wf2; 前wf4变成新wf3
            QP.WF[0].redistTime = QP.WF[1].redistTime
            QP.WF[0].numRedist = QP.WF[1].numRedist
            QP.WF[1].redistTime = QP.WF[2].redistTime
            QP.WF[1].numRedist = QP.WF[2].numRedist
            QP.WF[2].redistTime = 0.0
            QP.WF[2].numRedist = QP.numRedist

    QP.redistTime = QP.redistTime + deltaT

    for T in range(0, 3):
        QP.WF[T].redistTime += deltaT

    if QP.WF[0].z == 0 and QP.precipRate == 0:
        return
    elif (QP.precipRate > 0.0) or (QP.pondingAmt > 1E-10):
        # 开始土壤入渗阶段
        if QP.pondingAmt <= 1E-10:
            # 更新再分配状态
            if (QP.redistStatus == 10) and (QP.precipRate < QP.ks_composite) and index != 0:
                # 湿润锋可以进行再分配
                QP.redistStatus = 11
            elif (QP.redistStatus == 11) and (QP.precipRate > QP.ks_composite):
                # 两个湿润锋，其中一个正在进行再分配
                QP.redistStatus = 20
                QP.WF[1].FAmt = QP.cummInfil - QP.WF[0].FAmt
                QP.WF[1].WC = QP.WCS[0]
                QP.WF[1].WCHold = QP.WCS[0]
                QP.fp = 0.0
                QP.ks_composite = QP.ks[0]
            elif (QP.redistStatus == 20) and (QP.precipRate < QP.ks_composite):
                # 两个湿润锋均在进行再分配
                QP.redistStatus = 21
            elif (QP.redistStatus == 21) and (QP.precipRate > QP.ks_composite):
                # 三个湿润锋，其中两个正在进行再分配
                QP.redistStatus = 30
                QP.WF[2].FAmt = QP.cummInfil - QP.WF[0].FAmt - QP.WF[1].FAmt
                QP.WF[2].WC = QP.WCS[0]
                QP.WF[2].WCHold = QP.WCS[0]
                QP.fp = 0.0
            elif (QP.redistStatus == 30) and (QP.precipRate < QP.ks_composite):
                # 三个湿润锋均在进行再分配
                QP.redistStatus = 31
            elif (QP.redistStatus == 31) and (QP.precipRate > QP.ks_composite):
                # 四个湿润锋，其中三个正在进行再分配
                QP.redistStatus = 40
                QP.WF[3].FAmt = QP.cummInfil - QP.WF[0].FAmt - QP.WF[1].FAmt - QP.WF[2].FAmt
                QP.WF[3].WC = QP.WCS[0]
                QP.WF[3].WCHold = QP.WCS[0]
                QP.fp = 0.0
            elif (QP.redistStatus == 40) and (QP.precipRate < QP.ks_composite):
                # 合并湿润锋 1 和 2，当前有三个湿润锋在进行再分配
                QP.redistStatus = 31
                # 更新 wf1 的信息
                QP.WF[0].FAmt += QP.WF[1].FAmt
                QP.WF[0].WC = QP.WF[1].WC
                QP.WF[0].WCHold = QP.WF[1].WCHold
                QP.WF[0].psi = QP.WF[1].psi
                QP.WF[0].psiHold = QP.WF[1].psiHold
                QP.WF[0].numRedist = QP.WF[1].numRedist
                QP.WF[0].redistTime = QP.WF[1].redistTime
                QP.WF[0].adjFactor = QP.WF[1].adjFactor
                QP.WF[1].FAmt = QP.WF[2].FAmt
                QP.WF[1].WC = QP.WF[2].WC
                QP.WF[1].WCHold = QP.WF[2].WCHold
                QP.WF[1].numRedist = QP.WF[2].numRedist
                QP.WF[1].redistTime = QP.WF[2].redistTime
                QP.WF[1].adjFactor = QP.WF[2].adjFactor
                QP.WF[1].z = QP.WF[2].z
                QP.WF[2].FAmt = QP.WF[3].FAmt
                QP.WF[2].WC = QP.WF[3].WC
                QP.WF[2].WCHold = QP.WF[3].WCHold
                QP.WF[2].numRedist = QP.WF[3].numRedist
                QP.WF[2].redistTime = QP.WF[3].redistTime
                QP.WF[2].adjFactor = QP.WF[3].adjFactor
                QP.WF[2].z = QP.WF[3].z
                QP.WF[3].FAmt = 0.0
                QP.WF[3].WC = 0.0
                QP.WF[3].WCHold = 0.0
                QP.WF[3].numRedist = 0.0
                QP.WF[3].redistTime = 0.0
                QP.WF[3].adjFactor = 0.0
                QP.WF[3].z = 0.0
                QP.fp = 0.0

                if QP.WF[0].FAmt <= (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0]:
                    QP.WF[0].z = calc_redist_zf(QP, QP.WF[0].FAmt, theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCI[0])
                    QP.WF[0].WC = QP.WF[1].WC
                    QP.WF[0].WCHold = QP.WF[1].WCHold
                elif (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] < QP.WF[0].FAmt <= ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]):
                    QP.WF[0].z = QP.max_depth[0] + calc_redist_zf(QP, QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0], theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.WCI[1])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                elif ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]) < QP.WF[0].FAmt <= ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]):
                    QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + calc_redist_zf(QP, QP.WF[0].FAmt - ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]), theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.WCI[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                elif QP.WF[0].FAmt > ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]):
                    QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + QP.max_depth[2]
                    # QP.runoff = QP.runoff + QP.WF[0].FAmt - ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].FAmt = ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])

            if (QP.fp > 1E-10) and (QP.precipRate >= QP.fp):
                # 积水立即开始
                QP.timeToPond = tstart
                if QP.redistStatus in [10, 11]:
                    if QP.WF[0].z <= QP.boundary_depths[0]:
                        QP.wcs = QP.WCS[0]
                        QP.wci = QP.WCI[0]
                        QP.Ks = QP.ks[0]
                        QP.ks_composite = QP.ks[0]
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
                    elif QP.boundary_depths[0] < QP.WF[0].z <= QP.boundary_depths[1]:
                        QP.wcs = QP.WCS[1]
                        QP.wci = QP.WCI[1]
                        QP.Ks = QP.ks[1]
                        QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[0].z - QP.max_depth[0]) / QP.ks[1])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
                    elif QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
                        QP.wcs = QP.WCS[2]
                        QP.wci = QP.WCI[2]
                        QP.Ks = QP.ks[2]
                        QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[0].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

                    QP.Bfp = QP.cummInfil
                    # QP.Bfp = QP.Ks * QP.Sav * (QP.wcs - QP.wci) / (QP.fp - QP.ks_composite)
                    QP.tpp = (QP.Bfp - (QP.Ks * QP.Sav * (QP.wcs - QP.wci) * math.log(1 + (QP.ks_composite * QP.Bfp / (QP.Ks * QP.Sav * (QP.wcs - QP.wci)))) / QP.ks_composite)) / QP.ks_composite
                elif QP.redistStatus in [20, 21]:
                    if QP.WF[1].z <= QP.boundary_depths[0]:
                        QP.wcs = QP.WCS[0]
                        QP.wci = theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                        QP.Ks = QP.ks[0]
                        QP.ks_composite = QP.ks[0]
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
                    elif QP.boundary_depths[0] < QP.WF[1].z <= QP.boundary_depths[1]:
                        QP.wcs = QP.WCS[1]
                        QP.wci = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                        QP.Ks = QP.ks[1]
                        QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[1].z - QP.max_depth[0]) / QP.ks[1])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
                    elif QP.boundary_depths[1] < QP.WF[1].z <= QP.boundary_depths[2]:
                        QP.wcs = QP.WCS[2]
                        QP.wci = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                        QP.Ks = QP.ks[2]
                        QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[1].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

                    # QP.Bfp = QP.cummInfil + QP.Ks * QP.Sav * (QP.wcs - QP.wci) / (QP.fp - QP.ks_composite)
                    QP.Bfp = QP.cummInfil
                    QP.tpp = (QP.Bfp - (QP.Ks * QP.Sav * (QP.wcs - QP.wci) * math.log(1 + (QP.ks_composite * QP.Bfp / (QP.Ks * QP.Sav * (QP.wcs - QP.wci)))) / QP.ks_composite)) / QP.ks_composite
                elif QP.redistStatus in [30, 31]:
                    if QP.WF[2].z <= QP.boundary_depths[0]:
                        QP.wcs = QP.WCS[0]
                        QP.wci = theta_of_psi(QP.WF[1].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                        QP.Ks = QP.ks[0]
                        QP.ks_composite = QP.ks[0]
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
                    elif QP.boundary_depths[0] < QP.WF[2].z <= QP.boundary_depths[1]:
                        QP.wcs = QP.WCS[1]
                        QP.wci = theta_of_psi(QP.WF[1].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                        QP.Ks = QP.ks[1]
                        QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[2].z - QP.max_depth[0]) / QP.ks[1])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
                    elif QP.boundary_depths[1] < QP.WF[2].z <= QP.boundary_depths[2]:
                        QP.wcs = QP.WCS[2]
                        QP.wci = theta_of_psi(QP.WF[1].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                        QP.Ks = QP.ks[2]
                        QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[2].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

                    # QP.Bfp = QP.Ks * QP.Sav * (QP.wcs - QP.wci) / (QP.fp - QP.ks_composite)
                    QP.Bfp = QP.cummInfil
                    QP.tpp = (QP.Bfp - (QP.Ks * QP.Sav * (QP.wcs - QP.wci) * math.log(1 + (QP.ks_composite * QP.Bfp / (QP.Ks * QP.Sav * (QP.wcs - QP.wci)))) / QP.ks_composite)) / QP.ks_composite
                else:
                    if QP.WF[3].z <= QP.boundary_depths[0]:
                        QP.wcs = QP.WCS[0]
                        QP.wci = theta_of_psi(QP.WF[2].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                        QP.Ks = QP.ks[0]
                        QP.ks_composite = QP.ks[0]
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
                    elif QP.boundary_depths[0] < QP.WF[3].z <= QP.boundary_depths[1]:
                        QP.wcs = QP.WCS[1]
                        QP.wci = theta_of_psi(QP.WF[2].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                        QP.Ks = QP.ks[1]
                        QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[3].z - QP.max_depth[0]) / QP.ks[1])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
                    elif QP.boundary_depths[1] < QP.WF[3].z <= QP.boundary_depths[2]:
                        QP.wcs = QP.WCS[2]
                        QP.wci = theta_of_psi(QP.WF[2].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                        QP.Ks = QP.ks[2]
                        QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[3].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

                    # QP.Bfp = QP.Ks * QP.Sav * (QP.wcs - QP.wci) / (QP.fp - QP.ks_composite)
                    QP.Bfp = QP.cummInfil
                    QP.tpp = (QP.Bfp - (QP.Ks * QP.Sav * (QP.wcs - QP.wci) * math.log(1 + (QP.ks_composite * QP.Bfp / (QP.Ks * QP.Sav * (QP.wcs - QP.wci)))) / QP.ks_composite)) / QP.ks_composite
            elif QP.precipRate > QP.ks_composite:
                # 积水不一定会立即开始
                if QP.redistStatus in [10, 11]:
                    if QP.WF[0].z <= QP.boundary_depths[0]:
                        QP.wcs = QP.WCS[0]
                        QP.wci = QP.WCI[0]
                        QP.Ks = QP.ks[0]
                        QP.ks_composite = QP.ks[0]
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
                    elif QP.boundary_depths[0] < QP.WF[0].z <= QP.boundary_depths[1]:
                        QP.wcs = QP.WCS[1]
                        QP.wci = QP.WCI[1]
                        QP.Ks = QP.ks[1]
                        QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[0].z - QP.max_depth[0]) / QP.ks[1])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
                    elif QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
                        QP.wcs = QP.WCS[2]
                        QP.wci = QP.WCI[2]
                        QP.Ks = QP.ks[2]
                        QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[0].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

                    if QP.precipRate > QP.ks_composite:
                        QP.Bfp = QP.Ks * QP.Sav * (QP.wcs - QP.wci) / (QP.precipRate - QP.ks_composite)
                        QP.timeToPond = QP.Bfp / QP.precipRate + tstart
                        QP.tpp = (QP.Bfp - (QP.Ks * QP.Sav * (QP.wcs - QP.wci) * math.log(1 + (QP.ks_composite * QP.Bfp / (QP.Ks * QP.Sav * (QP.wcs - QP.wci)))) / QP.ks_composite)) / QP.ks_composite
                    else:  # 积水不会发生
                        QP.timeToPond = 99999999
                elif QP.redistStatus in [20, 21]:
                    if QP.WF[1].z <= QP.boundary_depths[0]:
                        QP.wcs = QP.WCS[0]
                        QP.wci = theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                        QP.Ks = QP.ks[0]
                        QP.ks_composite = QP.ks[0]
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
                    elif QP.boundary_depths[0] < QP.WF[1].z <= QP.boundary_depths[1]:
                        QP.wcs = QP.WCS[1]
                        QP.wci = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                        QP.Ks = QP.ks[1]
                        QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[1].z - QP.max_depth[0]) / QP.ks[1])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
                    elif QP.boundary_depths[1] < QP.WF[1].z <= QP.boundary_depths[2]:
                        QP.wcs = QP.WCS[2]
                        QP.wci = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                        QP.Ks = QP.ks[2]
                        QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[1].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

                    if QP.precipRate > QP.ks_composite:
                        QP.Bfp = QP.Ks * QP.Sav * (QP.wcs - QP.wci) / (QP.precipRate - QP.ks_composite)
                        QP.timeToPond = QP.Bfp / QP.precipRate + tstart
                        QP.tpp = (QP.Bfp - (QP.Ks * QP.Sav * (QP.wcs - QP.wci) * math.log(1 + (QP.ks_composite * QP.Bfp / (QP.Ks * QP.Sav * (QP.wcs - QP.wci)))) / QP.ks_composite)) / QP.ks_composite
                    else:  # 积水不会发生
                        QP.timeToPond = 99999999
                elif QP.redistStatus in [30, 31]:
                    if QP.WF[2].z <= QP.boundary_depths[0]:
                        QP.wcs = QP.WCS[0]
                        QP.wci = theta_of_psi(QP.WF[1].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                        QP.Ks = QP.ks[0]
                        QP.ks_composite = QP.ks[0]
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
                    elif QP.boundary_depths[0] < QP.WF[2].z <= QP.boundary_depths[1]:
                        QP.wcs = QP.WCS[1]
                        QP.wci = theta_of_psi(QP.WF[1].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                        QP.Ks = QP.ks[1]
                        QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[2].z - QP.max_depth[0]) / QP.ks[1])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
                    elif QP.boundary_depths[1] < QP.WF[2].z <= QP.boundary_depths[2]:
                        QP.wcs = QP.WCS[2]
                        QP.wci = theta_of_psi(QP.WF[1].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                        QP.Ks = QP.ks[2]
                        QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[2].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

                    if QP.precipRate > QP.ks_composite:
                        QP.Bfp = QP.Ks * QP.Sav * (QP.wcs - QP.wci) / (QP.precipRate - QP.ks_composite)
                        QP.timeToPond = QP.Bfp / QP.precipRate + tstart
                        QP.tpp = (QP.Bfp - (QP.Ks * QP.Sav * (QP.wcs - QP.wci) * math.log(1 + (QP.ks_composite * QP.Bfp / (QP.Ks * QP.Sav * (QP.wcs - QP.wci)))) / QP.ks_composite)) / QP.ks_composite
                    else:  # 积水不会发生
                        QP.timeToPond = 99999999
                else:
                    if QP.WF[3].z <= QP.boundary_depths[0]:
                        QP.wcs = QP.WCS[0]
                        QP.wci = theta_of_psi(QP.WF[2].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
                        QP.Ks = QP.ks[0]
                        QP.ks_composite = QP.ks[0]
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.wci, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
                    elif QP.boundary_depths[0] < QP.WF[3].z <= QP.boundary_depths[1]:
                        QP.wcs = QP.WCS[1]
                        QP.wci = theta_of_psi(QP.WF[2].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                        QP.Ks = QP.ks[1]
                        QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[3].z - QP.max_depth[0]) / QP.ks[1])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.wci, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
                    elif QP.boundary_depths[1] < QP.WF[3].z <= QP.boundary_depths[2]:
                        QP.wcs = QP.WCS[2]
                        QP.wci = theta_of_psi(QP.WF[2].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                        QP.Ks = QP.ks[2]
                        QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[3].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
                        QP.Sav = G(psi_of_theta(QP.wcs, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.wci, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])

                    if QP.precipRate > QP.ks_composite:
                        QP.Bfp = QP.Ks * QP.Sav * (QP.wcs - QP.wci) / (QP.precipRate - QP.ks_composite)
                        QP.timeToPond = QP.Bfp / QP.precipRate + tstart
                        QP.tpp = (QP.Bfp - (QP.Ks * QP.Sav * (QP.wcs - QP.wci) * math.log(1 + (QP.ks_composite * QP.Bfp / (QP.Ks * QP.Sav * (QP.wcs - QP.wci)))) / QP.ks_composite)) / QP.ks_composite
                    else:  # 积水不会发生
                        QP.timeToPond = 99999999
            else:  # 积水不会发生
                QP.timeToPond = 99999999

            if QP.timeToPond > tend:
                # 在时间步内没有积水发生；仅使用 calcNoPond 计算入渗量
                QP.timeToPond = 99999999
                QP.tpp = 99999999
                calc_no_pond(QP, tstart, tend, deltaT, QP.precipRate, upstream_water_depth)
            elif QP.timeToPond < tend:
                # 在时间步内将发生积水；同时使用 noPond 和 Pond 方法计算入渗量
                # 从开始到积水发生（在时间点 tp）期间的入渗率
                calc_no_pond(QP, tstart, QP.timeToPond, deltaT, QP.precipRate, upstream_water_depth)
                # 从时间点 tp 到周期结束期间的入渗率
                calc_pond(QP, QP.timeToPond, tend, QP.timeToPond, QP.tpp, deltaT, QP.precipRate, upstream_water_depth)

        # 从开始时就发生积水
        # 需要进一步修改，目前还未使用
        elif QP.pondingAmt >= 1E-10:
            QP.timeToPond = tstart

            # 如果在模拟开始前表面已经发生积水
            if QP.Bfp == 0.0:
                QP.tpp = tstart

            # 合并湿润锋 1 和 2，现在有三个湿润锋
            if QP.redistStatus == 40 and QP.precipRate < QP.ks_composite:
                QP.redistStatus = 30
                QP.WF[0].FAmt += QP.WF[1].FAmt
                QP.WF[0].WC = QP.WF[1].WC
                QP.WF[0].WCHold = QP.WF[1].WCHold
                QP.WF[0].psi = QP.WF[1].psi
                QP.WF[0].psiHold = QP.WF[1].psiHold
                QP.WF[0].numRedist = QP.WF[1].numRedist
                QP.WF[0].redistTime = QP.WF[1].redistTime
                QP.WF[0].adjFactor = QP.WF[1].adjFactor
                QP.WF[1].FAmt = QP.WF[2].FAmt
                QP.WF[1].WC = QP.WF[2].WC
                QP.WF[1].WCHold = QP.WF[2].WCHold
                QP.WF[1].numRedist = QP.WF[2].numRedist
                QP.WF[1].redistTime = QP.WF[2].redistTime
                QP.WF[1].adjFactor = QP.WF[2].adjFactor
                QP.WF[1].z = QP.WF[2].z
                QP.WF[2].FAmt = QP.WF[3].FAmt
                QP.WF[2].WC = QP.WF[3].WC
                QP.WF[2].WCHold = QP.WF[3].WCHold
                QP.WF[2].numRedist = QP.WF[3].numRedist
                QP.WF[2].redistTime = QP.WF[3].redistTime
                QP.WF[2].adjFactor = QP.WF[3].adjFactor
                QP.WF[2].z = QP.WF[3].z
                QP.WF[3].FAmt = 0.0
                QP.WF[3].WC = 0.0
                QP.WF[3].WCHold = 0.0
                QP.WF[3].numRedist = 0.0
                QP.WF[3].redistTime = 0.0
                QP.WF[3].adjFactor = 0.0
                QP.WF[3].z = 0.0
                QP.fp = 0.0

                if QP.WF[0].FAmt <= (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0]:
                    QP.WF[0].z = calc_redist_zf(QP, QP.WF[0].FAmt, theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCI[0])
                    QP.WF[0].WC = QP.WF[1].WC
                    QP.WF[0].WCHold = QP.WF[1].WCHold
                elif (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] < QP.WF[0].FAmt <= ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]):
                    QP.WF[0].z = QP.max_depth[0] + calc_redist_zf(QP, QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0], theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.WCI[1])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                elif ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]) < QP.WF[0].FAmt <= ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]):
                    QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + calc_redist_zf(QP, QP.WF[0].FAmt - ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]), theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.WCI[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                elif QP.WF[0].FAmt > ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]):
                    QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + QP.max_depth[2]
                    # QP.runoff = QP.runoff + QP.WF[0].FAmt - ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].FAmt = ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                    QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                    QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])

            # 有可能积水不会在整个时间段内持续,需要找到所有水分入渗完成的时间，即不再发生积水的时间
            if QP.precipRate < QP.fp:
                if QP.redistStatus == 10:
                    QP.Bfp = QP.WF[0].FAmt
                    QP.tpp = (QP.Bfp - QP.Sav * (QP.WCS - QP.WCI) * math.log(1 + (QP.Bfp / (QP.Sav * (QP.WCS - QP.WCI))))) / QP.ks
                    # QP.tpp = calc_tpp(QP, QP.WCI, QP.Sav)
                    total_infil = QP.WF[0].FAmt + QP.pondingAmt
                    t1 = calc_t_newton(QP, tstart, tend, QP.timeToPond, QP.tpp, QP.precipRate, total_infil, QP.WCI, QP.Sav)
                elif QP.redistStatus == 20:
                    QP.Bfp = QP.WF[1].FAmt
                    QP.tpp = (QP.Bfp - QP.Sav * (QP.WCS - QP.WF[0].WC) * math.log(1 + (QP.Bfp / (QP.Sav * (QP.WCS - QP.WF[0].WC))))) / QP.ks
                    # QP.tpp = calc_tpp(QP, QP.WF[0].WC, QP.Sav)
                    total_infil = QP.WF[1].FAmt + QP.pondingAmt
                    t1 = calc_t_newton(QP, tstart, tend, QP.timeToPond, QP.tpp, QP.precipRate, total_infil, QP.WF[1].WC, QP.Sav)
                elif QP.redistStatus == 30:
                    QP.Bfp = QP.WF[2].FAmt
                    QP.tpp = (QP.Bfp - QP.Sav * (QP.WCS - QP.WF[1].WC) * math.log(1 + (QP.Bfp / (QP.Sav * (QP.WCS - QP.WF[1].WC))))) / QP.ks
                    # QP.tpp = calc_tpp(QP, QP.WF[1].WC, QP.Sav)
                    total_infil = QP.WF[2].FAmt + QP.pondingAmt
                    t1 = calc_t_newton(QP, tstart, tend, QP.timeToPond, QP.tpp, QP.precipRate, total_infil, QP.WF[2].WC, QP.Sav)
                else:  # redistStatus = 40
                    QP.Bfp = QP.WF[3].FAmt
                    QP.tpp = (QP.Bfp - QP.Sav * (QP.WCS - QP.WF[2].WC) * math.log(1 + (QP.Bfp / (QP.Sav * (QP.WCS - QP.WF[2].WC))))) / QP.ks
                    # QP.tpp = calc_tpp(QP, QP.WF[2].WC, QP.Sav)
                    total_infil = QP.WF[3].FAmt + QP.pondingAmt
                    t1 = calc_t_newton(QP, tstart, tend, QP.timeToPond, QP.tpp, QP.precipRate, total_infil, QP.WF[3].WC, QP.Sav)
            # 没有可能完全失去所有积水
            else:
                t1 = tend + 99999999.0  # 设置时间，以确保积水状态持续

            if t1 > tend:
                # 将保持积水状态整个周期；仅在积水条件下计算入渗量
                calc_pond(QP, tstart, tend, QP.timeToPond, QP.tpp, deltaT, QP.precipRate, upstream_water_depth)
            elif t1 < tend:  # 积水将在 t1 结束；使用积水和非积水条件计算入渗量
                # 从开始时间（tstart）到时间点 t1 的积水部分
                calc_pond(QP, tstart, t1, QP.timeToPond, QP.tpp, deltaT, QP.precipRate, upstream_water_depth)
                if QP.redistStatus == 10 and QP.precipRate < QP.ks:
                    QP.redistStatus = 11
                elif QP.redistStatus == 20 and QP.precipRate < QP.ks:
                    QP.redistStatus = 21
                elif QP.redistStatus == 30 and QP.precipRate < QP.ks:
                    QP.redistStatus = 31
                # 从时间点 t1 到周期结束时间（tEND）的非积水部分
                calc_no_pond(QP, t1, tend, deltaT, QP.precipRate, upstream_water_depth)
    # 仅进行再分配，不进行入渗
    else:
        # 仍需重新分配土壤水分
        if QP.redistStatus == 40:  # 合并湿润锋 1 和 2，现在有三个湿润锋
            QP.redistStatus = 31
            QP.WF[0].FAmt += QP.WF[1].FAmt
            QP.WF[0].WC = QP.WF[1].WC
            QP.WF[0].WCHold = QP.WF[1].WCHold
            QP.WF[0].psi = QP.WF[1].psi
            QP.WF[0].psiHold = QP.WF[1].psiHold
            QP.WF[0].numRedist = QP.WF[1].numRedist
            QP.WF[0].redistTime = QP.WF[1].redistTime
            QP.WF[0].adjFactor = QP.WF[1].adjFactor
            QP.WF[1].FAmt = QP.WF[2].FAmt
            QP.WF[1].WC = QP.WF[2].WC
            QP.WF[1].WCHold = QP.WF[2].WCHold
            QP.WF[1].numRedist = QP.WF[2].numRedist
            QP.WF[1].redistTime = QP.WF[2].redistTime
            QP.WF[1].adjFactor = QP.WF[2].adjFactor
            QP.WF[1].z = QP.WF[2].z
            QP.WF[2].FAmt = QP.WF[3].FAmt
            QP.WF[2].WC = QP.WF[3].WC
            QP.WF[2].WCHold = QP.WF[3].WCHold
            QP.WF[2].numRedist = QP.WF[3].numRedist
            QP.WF[2].redistTime = QP.WF[3].redistTime
            QP.WF[2].adjFactor = QP.WF[3].adjFactor
            QP.WF[2].z = QP.WF[3].z
            QP.WF[3].FAmt = 0.0
            QP.WF[3].WC = 0.0
            QP.WF[3].WCHold = 0.0
            QP.WF[3].numRedist = 0.0
            QP.WF[3].redistTime = 0.0
            QP.WF[3].adjFactor = 0.0
            QP.WF[3].z = 0.0
            QP.fp = 0.0

            if QP.WF[0].FAmt <= (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0]:
                QP.WF[0].z = calc_redist_zf(QP, QP.WF[0].FAmt, theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.WCI[0])
                QP.WF[0].WC = QP.WF[1].WC
                QP.WF[0].WCHold = QP.WF[1].WCHold
            elif (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] < QP.WF[0].FAmt <= ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]):
                QP.WF[0].z = QP.max_depth[0] + calc_redist_zf(QP, QP.WF[0].FAmt - (theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0], theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.WCI[1])
                QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
                QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])
            elif ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]) < QP.WF[0].FAmt <= ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]):
                QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + calc_redist_zf(QP, QP.WF[0].FAmt - ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1]), theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.WCI[2])
                QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
            elif QP.WF[0].FAmt > ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2]):
                QP.WF[0].z = QP.max_depth[0] + QP.max_depth[1] + QP.max_depth[2]
                # QP.runoff = QP.runoff + QP.WF[0].FAmt - ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                QP.WF[0].FAmt = ((theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]) - QP.WCI[0]) * QP.max_depth[0] + (theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]) - QP.WCI[1]) * QP.max_depth[1] + (theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]) - QP.WCI[2]) * QP.max_depth[2])
                QP.WF[0].WC = theta_of_psi(QP.WF[0].psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
                QP.WF[0].WCHold = theta_of_psi(QP.WF[0].psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])

        elif QP.redistStatus == 30:
            QP.redistStatus = 31
        elif QP.redistStatus == 20:
            QP.redistStatus = 21
        elif QP.redistStatus == 10:
            QP.redistStatus = 11

        nsteps = int((tend - tstart) / deltaT)
        inter_time = tstart

        for T in range(nsteps):
            inter_time += deltaT
            QP.time = inter_time
            QP.cummPrecip = QP.cummPrecip
            QP.cummInfil = QP.cummInfil
            QP.pondingAmt = 0.0
            QP.runoff = QP.runoff

            # ===== 新增: 在再分配过程中也处理ET =====
            if hasattr(QP, 'PET_rate') and QP.PET_rate > 0:
                actual_ET_demand = calculate_et_demand(QP, QP.PET_rate, deltaT)
                if hasattr(QP, 'actual_ET_vec'):
                    QP.actual_ET_vec.append(actual_ET_demand)
            # ===== ET处理结束 =====

            redistribution(QP, inter_time, deltaT)
            calc_wco(QP)
            calc_rwc(QP)

        delta_terr = tend - inter_time
        if delta_terr > 1E-10:
            inter_time = tend
            QP.time = inter_time
            QP.cummPrecip = QP.cummPrecip
            QP.cummInfil = QP.cummInfil
            QP.pondingAmt = 0.0
            QP.runoff = QP.runoff

            # ===== 新增: 处理剩余时间的ET =====
            if hasattr(QP, 'PET_rate') and QP.PET_rate > 0:
                actual_ET_demand = calculate_et_demand(QP, QP.PET_rate, delta_terr)
                if hasattr(QP, 'actual_ET_vec'):
                    QP.actual_ET_vec.append(actual_ET_demand)
            # ===== ET处理结束 =====

            redistribution(QP, inter_time, delta_terr)
            calc_wco(QP)
            calc_rwc(QP)

    # 计算下一阶段入渗速率
    if QP.redistStatus == 40:
        QP.WF[3].FAmt = QP.cummInfil - QP.WF[2].FAmt - QP.WF[1].FAmt - QP.WF[0].FAmt
        if QP.WF[3].z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
            QP.Sav = G(psi_of_theta(QP.WCS[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.WF[2].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
            QP.fp = QP.ks_composite + (QP.ks[0] * QP.Sav / QP.WF[3].z)
        elif QP.boundary_depths[0] < QP.WF[3].z <= QP.boundary_depths[1]:
            QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[3].z - QP.max_depth[0]) / QP.ks[1])
            QP.Sav = G(psi_of_theta(QP.WCS[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.WF[2].WC, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
            QP.fp = QP.ks_composite + (QP.ks[1] * QP.Sav / QP.WF[3].z)
        elif QP.boundary_depths[1] < QP.WF[3].z <= QP.boundary_depths[2]:
            QP.ks_composite = QP.WF[3].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[3].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
            QP.Sav = G(psi_of_theta(QP.WCS[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.WF[2].WC, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])
            QP.fp = QP.ks_composite + (QP.ks[2] * QP.Sav / QP.WF[3].z)
    elif QP.redistStatus in [30, 31]:
        QP.WF[2].FAmt = QP.cummInfil - QP.WF[1].FAmt - QP.WF[0].FAmt
        if QP.WF[2].z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
            QP.Sav = G(psi_of_theta(QP.WCS[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.WF[1].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
            QP.fp = QP.ks_composite + (QP.ks[0] * QP.Sav / QP.WF[2].z)
        elif QP.boundary_depths[0] < QP.WF[2].z <= QP.boundary_depths[1]:
            QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[2].z - QP.max_depth[0]) / QP.ks[1])
            QP.Sav = G(psi_of_theta(QP.WCS[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.WF[1].WC, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
            QP.fp = QP.ks_composite + (QP.ks[1] * QP.Sav / QP.WF[2].z)
        elif QP.boundary_depths[1] < QP.WF[2].z <= QP.boundary_depths[2]:
            QP.ks_composite = QP.WF[2].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[2].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
            QP.Sav = G(psi_of_theta(QP.WCS[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.WF[1].WC, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])
            QP.fp = QP.ks_composite + (QP.ks[2] * QP.Sav / QP.WF[2].z)
    elif QP.redistStatus in [20, 21]:
        QP.WF[1].FAmt = QP.cummInfil - QP.WF[0].FAmt
        if QP.WF[1].z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
            QP.Sav = G(psi_of_theta(QP.WCS[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.WF[0].WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
            QP.fp = QP.ks_composite + (QP.ks[0] * QP.Sav / QP.WF[1].z)
        elif QP.boundary_depths[0] < QP.WF[1].z <= QP.boundary_depths[1]:
            QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[1].z - QP.max_depth[0]) / QP.ks[1])
            QP.Sav = G(psi_of_theta(QP.WCS[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.WF[0].WC, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
            # QP.fp = QP.ks_composite + (QP.ks[1] * QP.Sav / QP.WF[1].z)
            QP.fp = QP.ks_composite + (QP.ks[1] * QP.Sav * (QP.WCS[1] - theta_of_psi(QP.WF[0].psi, QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1])) / QP.WF[1].FAmt)
        elif QP.boundary_depths[1] < QP.WF[1].z <= QP.boundary_depths[2]:
            QP.ks_composite = QP.WF[1].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[1].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
            QP.Sav = G(psi_of_theta(QP.WCS[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.WF[0].WC, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])
            # QP.fp = QP.ks_composite + (QP.ks[2] * QP.Sav / QP.WF[1].z)
            QP.fp = QP.ks_composite + (QP.ks[2] * QP.Sav * (
                        QP.WCS[2] - theta_of_psi(QP.WF[0].psi, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])) /
                                       QP.WF[1].FAmt)
    else:
        QP.WF[0].FAmt = QP.cummInfil
        if QP.WF[0].z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
            QP.Sav = G(psi_of_theta(QP.WCS[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), psi_of_theta(QP.WCI[0], QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0]), QP.alpha[0], QP.n[0], QP.m[0], QP.ks[0])
            QP.fp = QP.ks_composite + (QP.ks[0] * QP.Sav / QP.WF[0].z)
        elif QP.boundary_depths[0] < QP.WF[0].z <= QP.boundary_depths[1]:
            QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + (QP.WF[0].z - QP.max_depth[0]) / QP.ks[1])
            QP.Sav = G(psi_of_theta(QP.WCS[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), psi_of_theta(QP.WCI[1], QP.WCS[1], QP.WCR[1], QP.n[1], QP.m[1], QP.alpha[1]), QP.alpha[1], QP.n[1], QP.m[1], QP.ks[1])
            QP.fp = QP.ks_composite + (QP.ks[1] * QP.Sav * (QP.WCS[1] - QP.WCI[1]) / QP.WF[0].FAmt)
        elif QP.boundary_depths[1] < QP.WF[0].z <= QP.boundary_depths[2]:
            QP.ks_composite = QP.WF[0].z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] + (QP.WF[0].z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])
            QP.Sav = G(psi_of_theta(QP.WCS[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), psi_of_theta(QP.WCI[2], QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]), QP.alpha[2], QP.n[2], QP.m[2], QP.ks[2])
            QP.fp = QP.ks_composite + (QP.ks[2] * QP.Sav * (QP.WCS[2] - QP.WCI[2]) / QP.WF[0].FAmt)
    return
