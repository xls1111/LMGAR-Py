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
    theta = np.clip(theta, theta_r + epsilon, theta_s - epsilon)
    numerator = theta_s - theta_r
    denominator = theta - theta_r

    if denominator < epsilon:
        denominator = epsilon

    ratio = numerator / denominator
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
        return np.clip(theta, theta_r, theta_s)
    except:
        return theta_r


def capital_theta_of_psi(psi, alpha, n, m):
    """计算有效饱和度，添加数值保护"""
    epsilon = 1e-10
    psi = max(psi, epsilon)

    try:
        result = 1 / ((1 + (alpha * psi) ** n) ** m)
        return np.clip(result, 0.0, 1.0)
    except:
        return 0.0


def K(capital_theta_temp, K_s, m):
    """计算导水率，添加数值保护"""
    epsilon = 1e-10
    capital_theta_temp = np.clip(capital_theta_temp, epsilon, 1.0 - epsilon)

    try:
        temp = capital_theta_temp ** (1 / m)
        temp = min(temp, 1.0 - epsilon)
        inner = max(1 - temp, 0.0)
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


def rk4(QP, Rh, F, WC, y, dydx, h, WFZ, a):
    hh = h / 2.0
    yt = y + hh * dydx
    dyt = derivs(QP, Rh, F, WC, yt, WFZ, a)
    yt = y + hh * dyt
    dym = derivs(QP, Rh, F, WC, yt, WFZ, a)
    yt = y + h * dym
    dym = dym + dyt
    dyt = derivs(QP, Rh, F, WC, yt, WFZ, a)
    yout = y + h * (dydx + dyt + 2.0 * dym) / 6.0
    return yout


def rkqc(QP, Rh, F, WC, y, dydx, x, htry, eps, yscal, WFZ, a):
    pgrow = -0.2
    pshrnk = -0.25
    fcor = 1.0 / 15.0
    safety = 0.9
    errcon = 6.0e-4

    xsav = x
    ysav = y
    dysav = dydx
    h = htry

    while True:
        hh = h / 2.0
        ytemp = rk4(QP, Rh, F, WC, ysav, dysav, hh, WFZ, a)
        dydx = derivs(QP, Rh, F, WC, ytemp, WFZ, a)
        y = rk4(QP, Rh, F, WC, ytemp, dydx, hh, WFZ, a)
        ytemp_full = rk4(QP, Rh, F, WC, ysav, dysav, h, WFZ, a)
        ytemp = y - ytemp_full
        errmax = abs(ytemp / yscal)
        errmax = errmax / eps

        if errmax <= 1.0:
            hdid = h
            if errmax > errcon:
                hnext = safety * h * math.exp(pgrow * math.log(errmax))
            else:
                hnext = 4.0 * h
            break
        else:
            h = safety * h * math.exp(pshrnk * math.log(errmax))

    y += ytemp * fcor
    x = xsav + h
    return x, y, hdid, hnext


def odeint(QP, Rh, F, WC, ystart, x1, x2, eps, h1, WFZ, a):
    maxstp = 100
    tiny = 1.0e-3
    x = x1
    h = abs(h1) / 10
    y = ystart

    for nstp in range(maxstp):
        dydx = derivs(QP, Rh, F, WC, y, WFZ, a)
        yscal = abs(y) + abs(dydx * h) + tiny

        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        x, y, hdid, hnext = rkqc(QP, Rh, F, WC, y, dydx, x, h, eps, yscal, WFZ, a)

        if (x - x2) * (x2 - x1) >= 0.0:
            ystart = y
            break

        h = hnext

    return ystart


def adj_factor_calc(QP, num, redistT, z):
    a = 4.2951615111
    b = 154.6101111175
    c = 0.0020393887
    d = -0.0010402988
    e = -14.0032425382
    f = -61.5428564782

    if num == 0:
        return 0.0
    else:
        if z <= QP.boundary_depths[0]:
            QP.ks_composite = QP.ks[0]
        elif QP.boundary_depths[0] < z <= QP.boundary_depths[1]:
            QP.ks_composite = z / (QP.max_depth[0] / QP.ks[0] + (z - QP.max_depth[0]) / QP.ks[1])
        elif QP.boundary_depths[1] < z <= QP.boundary_depths[2]:
            QP.ks_composite = z / (QP.max_depth[0] / QP.ks[0] + QP.max_depth[1] / QP.ks[1] +
                                   (z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])

        ks = QP.ks_composite
        alpha = ks / (a * ks + b)
        beta = c + d * math.sqrt(ks)
        delta = ks / (e * ks + f)
        result = (alpha + beta * math.log(redistT) + delta / num) / 10
        result = result * 1.002

        if result < 0.0:
            result = 0.0

    return result


# ===================== 新增:蒸发相关函数 =====================

def calculate_et_demand(QP, PET_rate, time_step):
    """
    计算实际蒸散发需求(AET demand)
    使用van Genuchten方法,参考LGAR实现

    参数:
        QP: 参数对象
        PET_rate: 潜在蒸散发速率 (mm/h)
        time_step: 时间步长 (h)

    返回:
        actual_ET_demand: 实际蒸散发需求 (mm)
    """
    epsilon = 1e-10

    if PET_rate <= epsilon:
        return 0.0

    # 获取表层土壤含水量
    surface_theta = QP.WF[0].WC if QP.WF[0].WC > 0 else QP.WCI[0]
    surface_theta = np.clip(surface_theta, QP.WCR[0], QP.WCS[0])

    # 计算field capacity对应的含水量
    relative_moisture_fc = getattr(QP, 'relative_moisture_at_which_PET_equals_AET', 0.5)
    theta_fc = (QP.WCS[0] - QP.WCR[0]) * relative_moisture_fc + QP.WCR[0]

    # 获取凋萎点(如果没有定义,使用theta_r)
    wilting_point = getattr(QP, 'wilting_point', [QP.WCR[0], QP.WCR[1], QP.WCR[2]])

    # 计算50%有效含水量对应的psi值
    theta_50 = (theta_fc - wilting_point[0]) * 0.5 + wilting_point[0]
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


def apply_et_to_soil(QP, actual_ET_demand):
    """
    将蒸散发从土壤中扣除
    优先从最表层湿润锋扣除,类似LGAR的实现

    参数:
        QP: 参数对象
        actual_ET_demand: 实际蒸散发需求量 (mm)

    返回:
        actual_ET: 实际从土壤中提取的蒸散发量 (mm)
    """
    epsilon = 1e-10

    if actual_ET_demand <= epsilon:
        return 0.0

    # 从WF[0]扣除ET
    wf_idx = 0

    # 确定当前土层
    current_layer = _get_layer_index(QP, QP.WF[wf_idx].z)

    # 获取凋萎点
    wilting_point = getattr(QP, 'wilting_point', [QP.WCR[0], QP.WCR[1], QP.WCR[2]])

    # 计算可用于ET的水量(高于凋萎点的水量)
    if QP.WF[wf_idx].z > epsilon:
        available_theta = QP.WF[wf_idx].WC - wilting_point[current_layer]
        available_theta = max(available_theta, 0.0)
        available_water = available_theta * QP.WF[wf_idx].z
    else:
        available_water = 0.0

    # 实际ET不能超过可用水量
    actual_ET = min(actual_ET_demand, available_water)

    if actual_ET > epsilon:
        # 从FAmt中扣除ET
        QP.WF[wf_idx].FAmt -= actual_ET

        # 确保FAmt非负
        if QP.WF[wf_idx].FAmt < 0:
            actual_ET += QP.WF[wf_idx].FAmt
            QP.WF[wf_idx].FAmt = 0.0

        # 更新含水量
        if QP.WF[wf_idx].z > epsilon:
            # 通过质量平衡更新theta
            new_theta = wilting_point[current_layer] + (QP.WF[wf_idx].FAmt / QP.WF[wf_idx].z)
            new_theta = np.clip(new_theta, QP.WCR[current_layer], QP.WCS[current_layer])

            QP.WF[wf_idx].WC = new_theta
            QP.WF[wf_idx].WCHold = new_theta + QP.WF[wf_idx].adjFactor

            # 更新psi
            QP.WF[wf_idx].psi = psi_of_theta(QP.WF[wf_idx].WC, QP.WCS[current_layer],
                                             QP.WCR[current_layer], QP.n[current_layer],
                                             QP.m[current_layer], QP.alpha[current_layer])
            QP.WF[wf_idx].psiHold = psi_of_theta(QP.WF[wf_idx].WCHold, QP.WCS[current_layer],
                                                 QP.WCR[current_layer], QP.n[current_layer],
                                                 QP.m[current_layer], QP.alpha[current_layer])

    return actual_ET


def _get_layer_index(QP, z):
    """根据深度确定土层索引"""
    if z <= QP.boundary_depths[0]:
        return 0
    elif z <= QP.boundary_depths[1]:
        return 1
    elif z <= QP.boundary_depths[2]:
        return 2
    return 2

# ===================== 蒸发函数结束 =====================


# ===================== Redist系列函数 =====================
def redist11(QP, rh, ts, te, dt):
    """重新分配水分 - 简化版"""

    def get_layer_index(z, boundary_depths):
        """根据深度获取土层索引"""
        for i, boundary in enumerate(boundary_depths):
            if z <= boundary:
                return i
        return len(boundary_depths)

    def convert_theta_between_layers(theta, psi, from_layer, to_layer, QP):
        """在不同土层间转换含水量"""
        if from_layer == to_layer:
            return theta, psi

        if psi is None:
            psi = psi_of_theta(theta, QP.WCS[from_layer], QP.WCR[from_layer],
                               QP.n[from_layer], QP.m[from_layer], QP.alpha[from_layer])

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
        water_above = calc_total_water_above_layer(psi, current_layer, QP)
        theta_current = calc_water_in_layer(psi, current_layer, QP)
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
        max_capacity = 0.0
        for layer_idx in range(len(QP.max_depth)):
            theta_in_layer = calc_water_in_layer(psi, layer_idx, QP)
            max_capacity += (theta_in_layer - QP.WCI[layer_idx]) * QP.max_depth[layer_idx]

        overflow = WF.FAmt - max_capacity
        if overflow > 0:
            QP.runoff += overflow * 0
            WF.FAmt = max_capacity
            WF.z = QP.boundary_depths[-1]
            bottom_layer = len(QP.max_depth) - 1
            WF.WC = calc_water_in_layer(psi, bottom_layer, QP)
            WF.WCHold = WF.WC + WF.adjFactor
            update_wf_properties(WF, bottom_layer, QP)
            return True
        return False

    # 主逻辑
    WF = QP.WF[0]
    current_layer = get_layer_index(WF.z, QP.boundary_depths)

    if current_layer >= len(QP.max_depth):
        current_layer = len(QP.max_depth) - 1

    WF.WCHold = odeint(QP, rh, WF.FAmt, QP.WCI[current_layer],
                       WF.WCHold, ts, te, 1.0e-4, dt, WF.z, current_layer)

    WF.WC = WF.WCHold - WF.adjFactor
    update_wf_properties(WF, current_layer, QP)

    if current_layer == 0:
        WF.z = calc_redist_zf(QP, WF.FAmt, WF.WC, QP.WCI[0])
    else:
        WF.z = calc_z_in_multilayer(WF.FAmt, WF.psi, current_layer, QP)

    new_layer = get_layer_index(WF.z, QP.boundary_depths)

    if new_layer >= len(QP.max_depth):
        new_layer = len(QP.max_depth) - 1

    if new_layer != current_layer:
        WF.WC, WF.psi = convert_theta_between_layers(
            WF.WC, WF.psi, current_layer, new_layer, QP)
        WF.WCHold, WF.psiHold = convert_theta_between_layers(
            WF.WCHold, WF.psiHold, current_layer, new_layer, QP)
        WF.z = calc_z_in_multilayer(WF.FAmt, WF.psi, new_layer, QP)
        update_wf_properties(WF, new_layer, QP)

    if WF.z > QP.boundary_depths[-1]:
        handle_overflow(WF, WF.psi, QP)

    QP.fp = fp_calc(QP)


def redist20(QP, rh, ts, te, dt):
    """两个wetting front的再分配 - 3层专用简化版"""

    def get_layer(z):
        if z <= QP.boundary_depths[0]:
            return 0
        elif z <= QP.boundary_depths[1]:
            return 1
        else:
            return 2

    def psi_to_theta(psi, layer_idx):
        return theta_of_psi(psi, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def theta_to_psi(theta, layer_idx):
        return psi_of_theta(theta, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def calc_z_from_water(FAmt, psi):
        remaining = FAmt
        theta0 = psi_to_theta(psi, 0)
        water0 = (theta0 - QP.WCI[0]) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta0 - QP.WCI[0])
        remaining -= water0

        theta1 = psi_to_theta(psi, 1)
        water1 = (theta1 - QP.WCI[1]) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta1 - QP.WCI[1])
        remaining -= water1

        theta2 = psi_to_theta(psi, 2)
        water2 = (theta2 - QP.WCI[2]) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta2 - QP.WCI[2])

        return QP.boundary_depths[2]

    def calc_z_relative(FAmt, psi_front, psi_base):
        remaining = FAmt
        theta_front0 = psi_to_theta(psi_front, 0)
        theta_base0 = psi_to_theta(psi_base, 0)
        water0 = (theta_front0 - theta_base0) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta_front0 - theta_base0)
        remaining -= water0

        theta_front1 = psi_to_theta(psi_front, 1)
        theta_base1 = psi_to_theta(psi_base, 1)
        water1 = (theta_front1 - theta_base1) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta_front1 - theta_base1)
        remaining -= water1

        theta_front2 = psi_to_theta(psi_front, 2)
        theta_base2 = psi_to_theta(psi_base, 2)
        water2 = (theta_front2 - theta_base2) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta_front2 - theta_base2)

        return QP.boundary_depths[2]

    def handle_overflow(WF, psi):
        max_capacity = 0
        for i in range(3):
            theta = psi_to_theta(psi, i)
            max_capacity += (theta - QP.WCI[i]) * QP.max_depth[i]

        if WF.FAmt > max_capacity:
            overflow = WF.FAmt - max_capacity
            QP.runoff += overflow * 0
            WF.FAmt = max_capacity
            WF.z = QP.boundary_depths[2]
            WF.WC = psi_to_theta(psi, 2)
            WF.WCHold = WF.WC + WF.adjFactor
            WF.psiHold = theta_to_psi(WF.WCHold, 2)
            WF.psi = theta_to_psi(WF.WC, 2)
            return True
        return False

    def update_to_new_layer(WF, old_layer, new_layer):
        if old_layer == new_layer:
            return
        WF.WC = psi_to_theta(WF.psi, new_layer)
        WF.WCHold = WF.WC + WF.adjFactor
        WF.psiHold = theta_to_psi(WF.WCHold, new_layer)
        WF.psi = theta_to_psi(WF.WC, new_layer)

    # 处理 WF[0]
    WF0 = QP.WF[0]
    layer0 = get_layer(WF0.z)

    WF0.WCHold = odeint(QP, 0, WF0.FAmt, QP.WCI[layer0],
                        WF0.WCHold, ts, te, 1.0e-4, dt, WF0.z, layer0)

    WF0.WC = WF0.WCHold - WF0.adjFactor
    WF0.psiHold = theta_to_psi(WF0.WCHold, layer0)
    WF0.psi = theta_to_psi(WF0.WC, layer0)

    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)

    new_layer0 = get_layer(WF0.z)
    update_to_new_layer(WF0, layer0, new_layer0)

    if WF0.z > QP.boundary_depths[2]:
        handle_overflow(WF0, WF0.psi)

    # 处理 WF[1]
    WF1 = QP.WF[1]
    layer1 = get_layer(WF1.z)

    WF1.psi = theta_to_psi(WF1.WC, layer1)
    WF1.psiHold = theta_to_psi(WF1.WCHold, layer1)

    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

    new_layer1 = get_layer(WF1.z)
    update_to_new_layer(WF1, layer1, new_layer1)

    if WF1.z > QP.boundary_depths[2]:
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

    # 检查合并
    if WF1.z >= WF0.z:
        WF0.FAmt += WF1.FAmt
        temp_layer = get_layer(WF0.z)
        WF0.WC = QP.WCS[temp_layer]
        WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])
        WF0.psi = theta_to_psi(WF0.WC, temp_layer)
        WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
        final_layer = get_layer(WF0.z)

        if final_layer != temp_layer:
            WF0.WC = QP.WCS[final_layer]
            WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[final_layer])
            WF0.psi = theta_to_psi(WF0.WC, final_layer)
            WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer)
            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
            final_layer = get_layer(WF0.z)

        WF0.redistTime = WF1.redistTime
        WF0.adjFactor = WF1.adjFactor

        WF1.FAmt = 0.0
        WF1.WC = 0.0
        WF1.WCHold = 0.0
        WF1.numRedist = 0.0
        WF1.redistTime = 0.0
        WF1.adjFactor = 0.0
        WF1.z = 0.0
        QP.redistStatus = 10

        handle_overflow(WF0, WF0.psi)

    QP.fp = fp_calc(QP)


def redist21(QP, rh, ts, te, dt):
    """两个wetting front都在再分配 - 3层专用简化版"""

    def get_layer(z):
        if z <= QP.boundary_depths[0]:
            return 0
        elif z <= QP.boundary_depths[1]:
            return 1
        else:
            return 2

    def psi_to_theta(psi, layer_idx):
        return theta_of_psi(psi, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def theta_to_psi(theta, layer_idx):
        return psi_of_theta(theta, QP.WCS[layer_idx], QP.WCR[layer_idx],
                            QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])

    def calc_z_from_water(FAmt, psi):
        remaining = FAmt
        theta0 = psi_to_theta(psi, 0)
        water0 = (theta0 - QP.WCI[0]) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta0 - QP.WCI[0])
        remaining -= water0

        theta1 = psi_to_theta(psi, 1)
        water1 = (theta1 - QP.WCI[1]) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta1 - QP.WCI[1])
        remaining -= water1

        theta2 = psi_to_theta(psi, 2)
        water2 = (theta2 - QP.WCI[2]) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta2 - QP.WCI[2])
        return QP.boundary_depths[2]

    def calc_z_relative(FAmt, psi_front, psi_base):
        remaining = FAmt
        theta_front0 = psi_to_theta(psi_front, 0)
        theta_base0 = psi_to_theta(psi_base, 0)
        water0 = (theta_front0 - theta_base0) * QP.max_depth[0]
        if remaining <= water0:
            return remaining / (theta_front0 - theta_base0)
        remaining -= water0

        theta_front1 = psi_to_theta(psi_front, 1)
        theta_base1 = psi_to_theta(psi_base, 1)
        water1 = (theta_front1 - theta_base1) * QP.max_depth[1]
        if remaining <= water1:
            return QP.max_depth[0] + remaining / (theta_front1 - theta_base1)
        remaining -= water1

        theta_front2 = psi_to_theta(psi_front, 2)
        theta_base2 = psi_to_theta(psi_base, 2)
        water2 = (theta_front2 - theta_base2) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta_front2 - theta_base2)
        return QP.boundary_depths[2]

    def handle_overflow(WF, psi):
        max_capacity = 0
        for i in range(3):
            theta = psi_to_theta(psi, i)
            max_capacity += (theta - QP.WCI[i]) * QP.max_depth[i]

        if WF.FAmt > max_capacity:
            overflow = WF.FAmt - max_capacity
            QP.runoff += overflow * 0
            WF.FAmt = max_capacity
            WF.z = QP.boundary_depths[2]
            WF.WC = psi_to_theta(psi, 2)
            WF.WCHold = WF.WC + WF.adjFactor
            WF.psiHold = theta_to_psi(WF.WCHold, 2)
            WF.psi = theta_to_psi(WF.WC, 2)
            return True
        return False

    def handle_overflow_relative(WF, psi_front, psi_base):
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
            WF.WC = psi_to_theta(psi_front, 2)
            WF.WCHold = WF.WC + WF.adjFactor
            WF.psiHold = theta_to_psi(WF.WCHold, 2)
            WF.psi = theta_to_psi(WF.WC, 2)
            return True
        return False

    def update_to_new_layer(WF, old_layer, new_layer):
        if old_layer == new_layer:
            return
        WF.WC = psi_to_theta(WF.psi, new_layer)
        WF.WCHold = WF.WC + WF.adjFactor
        WF.psiHold = theta_to_psi(WF.WCHold, new_layer)
        WF.psi = theta_to_psi(WF.WC, new_layer)

    # 处理 WF[0]
    WF0 = QP.WF[0]
    layer0 = get_layer(WF0.z)

    WF0.WCHold = odeint(QP, 0, WF0.FAmt, QP.WCI[layer0],
                        WF0.WCHold, ts, te, 1.0e-4, dt, WF0.z, layer0)
    WF0.WC = WF0.WCHold - WF0.adjFactor
    WF0.psiHold = theta_to_psi(WF0.WCHold, layer0)
    WF0.psi = theta_to_psi(WF0.WC, layer0)

    WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)

    new_layer0 = get_layer(WF0.z)
    update_to_new_layer(WF0, layer0, new_layer0)

    if WF0.z > QP.boundary_depths[2]:
        handle_overflow(WF0, WF0.psi)

    # 处理 WF[1]
    WF1 = QP.WF[1]
    layer1 = get_layer(WF1.z)

    wci_relative = psi_to_theta(WF0.psiHold, layer1)
    WF1.WCHold = odeint(QP, rh, WF1.FAmt, wci_relative,
                        WF1.WCHold, ts, te, 1.0e-4, dt, WF1.z, layer1)

    WF1.WC = WF1.WCHold - WF1.adjFactor
    WF1.psiHold = theta_to_psi(WF1.WCHold, layer1)
    WF1.psi = theta_to_psi(WF1.WC, layer1)

    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

    new_layer1 = get_layer(WF1.z)
    update_to_new_layer(WF1, layer1, new_layer1)

    if WF1.z > QP.boundary_depths[2]:
        handle_overflow_relative(WF1, WF1.psi, WF0.psi)

    # 检查合并
    if WF1.z >= WF0.z:
        WF0.FAmt += WF1.FAmt
        temp_layer = get_layer(WF0.z)
        WF0.WC = QP.WCS[temp_layer]
        WF0.WCHold = min(WF0.WC + WF1.adjFactor, QP.WCS[temp_layer])
        WF0.psi = theta_to_psi(WF0.WC, temp_layer)
        WF0.psiHold = theta_to_psi(WF0.WCHold, temp_layer)

        WF0.numRedist = WF1.numRedist
        WF0.redistTime = WF1.redistTime
        WF0.adjFactor = WF1.adjFactor

        WF1.FAmt = 0.0
        WF1.WC = 0.0
        WF1.WCHold = 0.0
        WF1.numRedist = 0.0
        WF1.redistTime = 0.0
        WF1.adjFactor = 0.0
        WF1.z = 0.0
        QP.redistStatus = 11

        WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
        final_layer = get_layer(WF0.z)

        if final_layer != temp_layer:
            WF0.WC = QP.WCS[final_layer]
            WF0.WCHold = min(WF0.WC + WF0.adjFactor, QP.WCS[final_layer])
            WF0.psi = theta_to_psi(WF0.WC, final_layer)
            WF0.psiHold = theta_to_psi(WF0.WCHold, final_layer)
            WF0.z = calc_z_from_water(WF0.FAmt, WF0.psi)
            final_layer = get_layer(WF0.z)

        handle_overflow(WF0, WF0.psi)

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
        WF0.WC = QP.WCS[temp_layer]
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

        # 重新计算WF[0]的位置
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
        WF0.WC = QP.WCS[temp_layer]
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

        # 重新计算WF[0]的位置
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
    """四个wetting front的再分配 - 3层专用简化版"""

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
            return QP.max_depth[0] + remaining / (theta_front1 - theta_base1) if (theta_front1 - theta_base1) > 0 else \
            QP.max_depth[0]
        remaining -= water1

        # Layer 2
        theta_front2 = psi_to_theta(psi_front, 2)
        theta_base2 = psi_to_theta(psi_base, 2)
        water2 = (theta_front2 - theta_base2) * QP.max_depth[2]
        if remaining <= water2:
            return QP.max_depth[0] + QP.max_depth[1] + remaining / (theta_front2 - theta_base2) if (theta_front2 - theta_base2) > 0 else \
            QP.max_depth[0] + QP.max_depth[1]

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

    def merge_wf_to_front(front_WF, merge_WF):
        """合并wetting front到前方的front"""
        front_WF.FAmt += merge_WF.FAmt

        # 临时确定层
        temp_layer = get_layer(front_WF.z)

        # 设置为该层的最大含水量
        front_WF.WC = QP.WCS[temp_layer]
        front_WF.WCHold = min(front_WF.WC + merge_WF.adjFactor, QP.WCS[temp_layer])

        # 计算对应的psi
        front_WF.psi = theta_to_psi(front_WF.WC, temp_layer)
        front_WF.psiHold = theta_to_psi(front_WF.WCHold, temp_layer)

        front_WF.numRedist = merge_WF.numRedist
        front_WF.redistTime = merge_WF.redistTime
        front_WF.adjFactor = merge_WF.adjFactor

    def recalc_wf_position_absolute(WF):
        """重新计算WF的绝对位置并更新到正确的层"""
        temp_layer = get_layer(WF.z)

        WF.z = calc_z_from_water(WF.FAmt, WF.psi)
        final_layer = get_layer(WF.z)

        # 如果层发生变化,更新到新层的WCS
        if final_layer != temp_layer:
            WF.WC = QP.WCS[final_layer]
            WF.WCHold = min(WF.WC + WF.adjFactor, QP.WCS[final_layer])
            WF.psi = theta_to_psi(WF.WC, final_layer)
            WF.psiHold = theta_to_psi(WF.WCHold, final_layer)
            WF.z = calc_z_from_water(WF.FAmt, WF.psi)
            final_layer = get_layer(WF.z)

        update_to_layer(WF, final_layer)

        # 检查溢出
        if WF.z > QP.boundary_depths[2]:
            WF.FAmt, overflow = handle_overflow(WF.FAmt, WF.psi)
            QP.runoff += overflow * 0
            WF.z = QP.boundary_depths[2]
            update_to_layer(WF, 2)

    def recalc_wf_position_relative(WF, base_WF):
        """重新计算WF相对于base_WF的位置"""
        layer = get_layer(WF.z)

        # 更新psi
        WF.psi = theta_to_psi(WF.WC, layer)
        WF.psiHold = theta_to_psi(WF.WCHold, layer)

        # 计算相对位置
        WF.z = calc_z_relative(WF.FAmt, WF.psi, base_WF.psi)

        # 检查跨层
        new_layer = get_layer(WF.z)
        if new_layer != layer:
            update_to_layer(WF, new_layer)

        # 检查溢出
        if WF.z > QP.boundary_depths[2]:
            WF.FAmt, overflow = handle_overflow_relative(WF.FAmt, WF.psi, base_WF.psi)
            QP.runoff += overflow * 0
            WF.z = QP.boundary_depths[2]
            update_to_layer(WF, 2)

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

    # ===== 检查WF[1]和WF[0]是否需要合并 =====
    if WF1.z >= WF0.z:
        # 合并WF[1]到WF[0]
        merge_wf_to_front(WF0, WF1)

        # WF[2]移到WF[1], WF[3]移到WF[2]
        for i in range(1, 3):
            QP.WF[i].FAmt = QP.WF[i + 1].FAmt
            QP.WF[i].WC = QP.WF[i + 1].WC
            QP.WF[i].WCHold = QP.WF[i + 1].WCHold
            QP.WF[i].psi = QP.WF[i + 1].psi
            QP.WF[i].psiHold = QP.WF[i + 1].psiHold
            QP.WF[i].numRedist = QP.WF[i + 1].numRedist
            QP.WF[i].redistTime = QP.WF[i + 1].redistTime
            QP.WF[i].adjFactor = QP.WF[i + 1].adjFactor
            QP.WF[i].z = QP.WF[i + 1].z

        # 清空WF[3]
        clear_wf(QP.WF[3])
        QP.redistStatus = 30

        # 重新计算WF[0]
        recalc_wf_position_absolute(WF0)

        # 重新计算WF[1] (原WF[2], 再分配)
        WF1 = QP.WF[1]
        layer1_new = get_layer(WF1.z)

        wci_base1 = psi_to_theta(WF0.psiHold, layer1_new)
        WF1.WCHold = odeint(QP, 0, WF1.FAmt, wci_base1,
                            WF1.WCHold, ts, te, 1.0e-4, dt, WF1.z, layer1_new)
        WF1.WC = WF1.WCHold - WF1.adjFactor
        WF1.psiHold = theta_to_psi(WF1.WCHold, layer1_new)
        WF1.psi = theta_to_psi(WF1.WC, layer1_new)
        WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)

        new_layer1_new = get_layer(WF1.z)
        if new_layer1_new != layer1_new:
            update_to_layer(WF1, new_layer1_new)

        if WF1.z > QP.boundary_depths[2]:
            WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
            QP.runoff += overflow * 0
            WF1.z = QP.boundary_depths[2]
            update_to_layer(WF1, 2)

        # 重新计算WF[2] (原WF[3], 入渗)
        recalc_wf_position_relative(QP.WF[2], WF1)

        # 再次检查WF[1]和WF[0]是否需要合并
        if WF1.z >= WF0.z:
            merge_wf_to_front(WF0, WF1)

            # WF[2]移到WF[1]
            QP.WF[1].FAmt = QP.WF[2].FAmt
            QP.WF[1].WC = QP.WF[2].WC
            QP.WF[1].WCHold = QP.WF[2].WCHold
            QP.WF[1].psi = QP.WF[2].psi
            QP.WF[1].psiHold = QP.WF[2].psiHold
            QP.WF[1].numRedist = QP.WF[2].numRedist
            QP.WF[1].redistTime = QP.WF[2].redistTime
            QP.WF[1].adjFactor = QP.WF[2].adjFactor
            QP.WF[1].z = QP.WF[2].z

            clear_wf(QP.WF[2])
            QP.redistStatus = 20

            recalc_wf_position_absolute(WF0)
            recalc_wf_position_relative(QP.WF[1], WF0)

            # 第三次检查
            if QP.WF[1].z >= WF0.z:
                merge_wf_to_front(WF0, QP.WF[1])
                clear_wf(QP.WF[1])
                QP.redistStatus = 10
                recalc_wf_position_absolute(WF0)

    else:
        # 仍然是4个front, 处理WF[2] (再分配,相对于WF[1])
        WF2 = QP.WF[2]
        layer2 = get_layer(WF2.z)

        wci_base2 = psi_to_theta(WF1.psiHold, layer2)
        WF2.WCHold = odeint(QP, 0, WF2.FAmt, wci_base2,
                            WF2.WCHold, ts, te, 1.0e-4, dt, WF2.z, layer2)
        WF2.WC = WF2.WCHold - WF2.adjFactor
        WF2.psiHold = theta_to_psi(WF2.WCHold, layer2)
        WF2.psi = theta_to_psi(WF2.WC, layer2)
        WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)

        new_layer2 = get_layer(WF2.z)
        if new_layer2 != layer2:
            update_to_layer(WF2, new_layer2)

        if WF2.z > QP.boundary_depths[2]:
            WF2.FAmt, overflow = handle_overflow_relative(WF2.FAmt, WF2.psi, WF1.psi)
            QP.runoff += overflow * 0
            WF2.z = QP.boundary_depths[2]
            update_to_layer(WF2, 2)

        # 检查WF[2]和WF[1]是否需要合并
        if WF2.z >= WF1.z:
            merge_wf_to_front(WF1, WF2)

            # WF[3]移到WF[2]
            QP.WF[2].FAmt = QP.WF[3].FAmt
            QP.WF[2].WC = QP.WF[3].WC
            QP.WF[2].WCHold = QP.WF[3].WCHold
            QP.WF[2].psi = QP.WF[3].psi
            QP.WF[2].psiHold = QP.WF[3].psiHold
            QP.WF[2].numRedist = QP.WF[3].numRedist
            QP.WF[2].redistTime = QP.WF[3].redistTime
            QP.WF[2].adjFactor = QP.WF[3].adjFactor
            QP.WF[2].z = QP.WF[3].z

            clear_wf(QP.WF[3])
            QP.redistStatus = 30

            # 重新计算WF[1]
            temp_layer = get_layer(WF1.z)
            WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
            final_layer1 = get_layer(WF1.z)

            if final_layer1 != temp_layer:
                WF1.WC = QP.WCS[final_layer1]
                WF1.WCHold = min(WF1.WC + WF1.adjFactor, QP.WCS[final_layer1])
                WF1.psi = theta_to_psi(WF1.WC, final_layer1)
                WF1.psiHold = theta_to_psi(WF1.WCHold, final_layer1)
                WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                final_layer1 = get_layer(WF1.z)

            update_to_layer(WF1, final_layer1)

            if WF1.z > QP.boundary_depths[2]:
                WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                QP.runoff += overflow * 0
                WF1.z = QP.boundary_depths[2]
                update_to_layer(WF1, 2)

            # 重新计算WF[2] (原WF[3], 入渗)
            recalc_wf_position_relative(QP.WF[2], WF1)

            # 再次检查WF[1]和WF[0]
            if WF1.z >= WF0.z:
                merge_wf_to_front(WF0, WF1)

                QP.WF[1].FAmt = QP.WF[2].FAmt
                QP.WF[1].WC = QP.WF[2].WC
                QP.WF[1].WCHold = QP.WF[2].WCHold
                QP.WF[1].psi = QP.WF[2].psi
                QP.WF[1].psiHold = QP.WF[2].psiHold
                QP.WF[1].numRedist = QP.WF[2].numRedist
                QP.WF[1].redistTime = QP.WF[2].redistTime
                QP.WF[1].adjFactor = QP.WF[2].adjFactor
                QP.WF[1].z = QP.WF[2].z

                clear_wf(QP.WF[2])
                QP.redistStatus = 20

                recalc_wf_position_absolute(WF0)
                recalc_wf_position_relative(QP.WF[1], WF0)

                if QP.WF[1].z >= WF0.z:
                    merge_wf_to_front(WF0, QP.WF[1])
                    clear_wf(QP.WF[1])
                    QP.redistStatus = 10
                    recalc_wf_position_absolute(WF0)

        else:
            # 仍然是4个front, 更新WF[3] (入渗)
            recalc_wf_position_relative(QP.WF[3], WF2)

            # 检查WF[3]和WF[2]是否需要合并
            if QP.WF[3].z >= WF2.z:
                merge_wf_to_front(WF2, QP.WF[3])
                clear_wf(QP.WF[3])
                QP.redistStatus = 30

                # 重新计算WF[2]
                temp_layer2 = get_layer(WF2.z)
                WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)
                final_layer2 = get_layer(WF2.z)

                if final_layer2 != temp_layer2:
                    WF2.WC = QP.WCS[final_layer2]
                    WF2.WCHold = min(WF2.WC + WF2.adjFactor, QP.WCS[final_layer2])
                    WF2.psi = theta_to_psi(WF2.WC, final_layer2)
                    WF2.psiHold = theta_to_psi(WF2.WCHold, final_layer2)
                    WF2.z = calc_z_relative(WF2.FAmt, WF2.psi, WF1.psi)
                    final_layer2 = get_layer(WF2.z)

                update_to_layer(WF2, final_layer2)

                if WF2.z > QP.boundary_depths[2]:
                    WF2.FAmt, overflow = handle_overflow_relative(WF2.FAmt, WF2.psi, WF1.psi)
                    QP.runoff += overflow * 0
                    WF2.z = QP.boundary_depths[2]
                    update_to_layer(WF2, 2)

                # 检查WF[2]和WF[1]
                if WF2.z >= WF1.z:
                    merge_wf_to_front(WF1, WF2)
                    clear_wf(WF2)
                    QP.redistStatus = 20

                    # 重新计算WF[1]
                    temp_layer1 = get_layer(WF1.z)
                    WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                    final_layer1 = get_layer(WF1.z)

                    if final_layer1 != temp_layer1:
                        WF1.WC = QP.WCS[final_layer1]
                        WF1.WCHold = min(WF1.WC + WF1.adjFactor, QP.WCS[final_layer1])
                        WF1.psi = theta_to_psi(WF1.WC, final_layer1)
                        WF1.psiHold = theta_to_psi(WF1.WCHold, final_layer1)
                        WF1.z = calc_z_relative(WF1.FAmt, WF1.psi, WF0.psi)
                        final_layer1 = get_layer(WF1.z)

                    update_to_layer(WF1, final_layer1)

                    if WF1.z > QP.boundary_depths[2]:
                        WF1.FAmt, overflow = handle_overflow_relative(WF1.FAmt, WF1.psi, WF0.psi)
                        QP.runoff += overflow * 0
                        WF1.z = QP.boundary_depths[2]
                        update_to_layer(WF1, 2)

                    # 检查WF[1]和WF[0]
                    if WF1.z >= WF0.z:
                        merge_wf_to_front(WF0, WF1)
                        clear_wf(WF1)
                        QP.redistStatus = 10
                        recalc_wf_position_absolute(WF0)

    # 更新潜在入渗速率
    QP.fp = fp_calc(QP)


# ===================== 修改后的主要计算函数(包含ET) =====================

def redistribution(QP, tend, dT):
    """
    再分配过程 - 修改为包含蒸发处理
    """
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


def cumm_infil_at_pond(QP, WC, S):
    rk = QP.precipRate / QP.ks
    result = S * (QP.WCS - WC) / (rk - 1)
    return result


def calc_time_to_pond(QP):
    result = QP.BFp / QP.PrecipRate
    return result


def calc_tpp(QP, W, S):
    QP.tpp = (QP.Bfp - S * (QP.WCS - W) * math.log(1 + (QP.Bfp / (S * (QP.WCS - W))))) / QP.ks
    return QP.tpp


def calc_f_newton(QP, Fnew, t, tp, tpp, wcs, wci, Sav, Ks, ks_composite):
    tolerance = 1.0E-6
    error = 1.0
    Fold = Fnew + 1.0E-6
    WCS_WC = wcs - wci
    g = Fold - Ks * Sav * WCS_WC * math.log(
        1.0 + (ks_composite * Fold / (Ks * Sav * WCS_WC))) / ks_composite - ks_composite * (t - tp + tpp)

    while error > tolerance:
        dgdf = 1.0 - (Ks * Sav * WCS_WC) / (Ks * Sav * WCS_WC + ks_composite * Fold)
        Fnew = Fold - (g / dgdf)
        Fold = Fnew
        g = Fold - Ks * Sav * WCS_WC * math.log(
            1.0 + (ks_composite * Fold / (Ks * Sav * WCS_WC))) / ks_composite - ks_composite * (t - tp + tpp)
        error = abs(g)

    return Fnew


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
            return 0.0

    lambda_val = math.pow(phi, (beta + delta * math.log(phi)))
    result = alpha * (S * (QP.WCS - WC)) * lambda_val

    return result


def calc_t_newton(QP, tstart, tend, tp, tpp, precip, total_infil, WC, S):
    tolerance = 1.0E-6
    error = 1.0
    t1new = tend
    t1old = t1new

    F = (t1old - tstart) * precip + total_infil
    lnarg = 1.0 + (F / (S * (QP.WCS - WC)))
    g = F - (S * (QP.WCS - WC)) * math.log(lnarg) - QP.ks * (t1old - tp + tpp)

    while error > tolerance:
        dgdeltaT1 = precip - precip * (1.0 / lnarg) - QP.ks
        t1new = t1old - (g / dgdeltaT1)
        t1old = t1new
        F = (t1old - tstart) * precip + total_infil
        lnarg = 1.0 + (F / (S * (QP.WCS - WC)))
        g = F - (S * (QP.WCS - WC)) * math.log(lnarg) - QP.ks * (t1old - tp + tpp)
        error = abs(g)

    return t1new


def calc_no_pond(QP, tstart, tend, deltaT, R, path):
    """
    无积水条件下的水分计算 - 修改为包含ET
    """
    interTime = tstart
    deltaTerr = tend - interTime

    if deltaTerr <= 1E-10:
        return

    # 更新基本参数
    interTime = tend
    QP.time = interTime
    QP.cummPrecip += deltaTerr * R
    QP.cummInfil += R * deltaTerr
    QP.pondingAmt = 0.0

    # ===== 新增: 计算并应用ET =====
    if hasattr(QP, 'PET_rate') and QP.PET_rate > 0:
        # 计算ET需求
        actual_ET_demand = calculate_et_demand(QP, QP.PET_rate, deltaTerr)

        # 应用ET到土壤
        actual_ET = apply_et_to_soil(QP, actual_ET_demand)

        # 记录实际ET(如果QP有相应属性)
        if hasattr(QP, 'actual_ET_vec'):
            QP.actual_ET_vec.append(actual_ET)
    # ===== ET处理结束 =====

    # 更新调整因子
    for j in range(3):
        QP.WF[j].adjFactor = adj_factor_calc(
            QP, QP.WF[j].numRedist,
            QP.WF[j].redistTime - deltaT + deltaTerr,
            QP.WF[j].z
        )

    # 处理不同的redistStatus
    _process_no_pond_redistribution(QP, R, interTime, deltaTerr, deltaT)


def _process_no_pond_redistribution(QP, R, interTime, deltaTerr, deltaT):
    """无积水条件下根据redistStatus处理水分再分配"""

    # 初始化阶段
    if QP.pondFlag == 0 and R > 0.0 and QP.WF[0].WC == 0.0:
        QP.WF[0].WC = QP.WCS[0]
        QP.WF[0].WCHold = QP.WF[0].WC
        QP.WF[0].FAmt += deltaTerr * R
        _update_no_pond_wf_position(QP, 0)
        return

    # 根据redistStatus分发处理
    handlers = {
        11: lambda: _handle_no_pond_status_11(QP, R, interTime, deltaTerr),
        20: lambda: _handle_no_pond_status_20_30(QP, R, interTime, deltaTerr, 1),
        21: lambda: _handle_no_pond_status_21_31(QP, R, interTime, deltaTerr, 1),
        30: lambda: _handle_no_pond_status_20_30(QP, R, interTime, deltaTerr, 2),
        31: lambda: _handle_no_pond_status_21_31(QP, R, interTime, deltaTerr, 2),
        40: lambda: redist40(QP, R, interTime - deltaTerr, interTime, deltaTerr),
        10: lambda: _handle_no_pond_status_10(QP),
    }

    handler = handlers.get(QP.redistStatus, lambda: _handle_no_pond_status_10(QP))
    handler()


def _handle_no_pond_status_11(QP, R, interTime, deltaTerr):
    """无积水条件下处理redistStatus=11"""
    QP.WF[0].FAmt += deltaTerr * R

    if QP.pondingAmt > 0.0:
        QP.redistStatus = 10
        _update_no_pond_wf_position(QP, 0)
    else:
        redist11(QP, R, interTime - deltaTerr, interTime, deltaTerr)


def _handle_no_pond_status_20_30(QP, R, interTime, deltaTerr, wf_idx):
    """无积水条件下处理redistStatus=20或30"""
    QP.WF[wf_idx].FAmt += deltaTerr * R
    redist_func = redist20 if wf_idx == 1 else redist30
    redist_func(QP, R, interTime - deltaTerr, interTime, deltaTerr)


def _handle_no_pond_status_21_31(QP, R, interTime, deltaTerr, wf_idx):
    """无积water条件下处理redistStatus=21或31"""
    QP.WF[wf_idx].FAmt += deltaTerr * R

    if QP.pondingAmt > 0.0:
        QP.redistStatus = 20 if wf_idx == 1 else 30
        redist_func = redist20 if wf_idx == 1 else redist30
        redist_func(QP, R, interTime - deltaTerr, interTime, deltaTerr)
    else:
        redist_func = redist21 if wf_idx == 1 else redist31
        redist_func(QP, R, interTime - deltaTerr, interTime, deltaTerr)


def _handle_no_pond_status_10(QP):
    """无积水条件下处理redistStatus=10"""
    current_layer = _get_layer_index(QP, QP.WF[0].z)

    QP.WF[0].psi = psi_of_theta(
        QP.WF[0].WC, QP.WCS[current_layer], QP.WCR[current_layer],
        QP.n[current_layer], QP.m[current_layer], QP.alpha[current_layer]
    )
    QP.WF[0].psiHold = psi_of_theta(
        QP.WF[0].WCHold, QP.WCS[current_layer], QP.WCR[current_layer],
        QP.n[current_layer], QP.m[current_layer], QP.alpha[current_layer]
    )

    _update_no_pond_wf_position(QP, 0)


def _update_no_pond_wf_position(QP, wf_idx):
    """无积水条件下更新湿润锋位置"""
    wf = QP.WF[wf_idx]

    psi = psi_of_theta(wf.WC, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])
    psi_hold = psi_of_theta(wf.WCHold, QP.WCS[0], QP.WCR[0], QP.n[0], QP.m[0], QP.alpha[0])

    theta_layers = [
        theta_of_psi(psi, QP.WCS[i], QP.WCR[i], QP.n[i], QP.m[i], QP.alpha[i])
        for i in range(3)
    ]

    layer_thresholds = _calculate_no_pond_layer_thresholds(QP, theta_layers, wf_idx)
    new_layer, new_z = _find_no_pond_target_layer(QP, wf.FAmt, layer_thresholds, theta_layers, wf_idx)

    if new_z > QP.boundary_depths[2]:
        _handle_no_pond_overflow(QP, wf, layer_thresholds[-1], theta_layers, psi, psi_hold)
    else:
        wf.z = new_z
        wf.WC = theta_of_psi(psi, QP.WCS[new_layer], QP.WCR[new_layer],
                             QP.n[new_layer], QP.m[new_layer], QP.alpha[new_layer])
        wf.WCHold = theta_of_psi(psi_hold, QP.WCS[new_layer], QP.WCR[new_layer],
                                 QP.n[new_layer], QP.m[new_layer], QP.alpha[new_layer])


def _calculate_no_pond_layer_thresholds(QP, theta_layers, wf_idx):
    """计算各土层的累积水量阈值"""
    if wf_idx == 0:
        water_deficit_0 = (theta_layers[0] - QP.WCI[0]) * QP.max_depth[0]
    else:
        water_deficit_0 = (QP.WF[wf_idx].WC - QP.WCI[0]) * QP.max_depth[0]

    water_deficit_1 = (theta_layers[1] - QP.WCI[1]) * QP.max_depth[1]
    water_deficit_2 = (theta_layers[2] - QP.WCI[2]) * QP.max_depth[2]

    return [
        water_deficit_0,
        water_deficit_0 + water_deficit_1,
        water_deficit_0 + water_deficit_1 + water_deficit_2
    ]


def _find_no_pond_target_layer(QP, FAmt, thresholds, theta_layers, wf_idx):
    """确定目标土层和深度"""
    if FAmt <= thresholds[0]:
        z = calc_redist_zf(QP, FAmt, theta_layers[0], QP.WCI[0])
        return 0, z

    elif FAmt <= thresholds[1]:
        z = QP.max_depth[0] + (FAmt - thresholds[0]) / (theta_layers[1] - QP.WCI[1])
        return 1, z

    elif FAmt <= thresholds[2]:
        z = (QP.max_depth[0] + QP.max_depth[1] +
             (FAmt - thresholds[1]) / (theta_layers[2] - QP.WCI[2]))
        return 2, z

    else:
        return 2, QP.max_depth[0] + QP.max_depth[1] + QP.max_depth[2]


def _handle_no_pond_overflow(QP, wf, max_capacity, theta_layers, psi, psi_hold):
    """处理超出土壤边界的情况"""
    wf.z = QP.boundary_depths[2]

    overflow = wf.FAmt - max_capacity
    QP.runoff += overflow * 0
    wf.FAmt = max_capacity

    wf.WC = theta_of_psi(psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])
    wf.WCHold = theta_of_psi(psi_hold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2])


def calc_pond(QP, tstart, tend, tp, tpp, deltaT, R, path):
    """
    积水条件下的水分计算 - 修改为包含ET
    """
    interTime = tstart
    deltaTerr = tend - interTime

    if deltaTerr <= 1E-10:
        calc_wco(QP)
        calc_rwc(QP)
        return

    # 更新时间和累积降水
    interTime = tend
    QP.time = interTime
    QP.cummPrecip += deltaTerr * R
    amtWater = R * deltaTerr
    Fhold = QP.cummInfil
    Fnew = QP.cummInfil + amtWater

    # 计算渗透和径流
    _calculate_pond_infiltration(QP, Fnew, interTime, tp, tpp, Fhold)

    # ===== 新增: 计算并应用ET =====
    if hasattr(QP, 'PET_rate') and QP.PET_rate > 0:
        # 计算ET需求
        actual_ET_demand = calculate_et_demand(QP, QP.PET_rate, deltaTerr)

        # 应用ET到土壤
        actual_ET = apply_et_to_soil(QP, actual_ET_demand)

        # 记录实际ET
        if hasattr(QP, 'actual_ET_vec'):
            QP.actual_ET_vec.append(actual_ET)
    # ===== ET处理结束 =====

    # 重新分配水分
    _reallocate_pond_water(QP, R, interTime, deltaTerr, deltaT)

    calc_wco(QP)
    calc_rwc(QP)


def _calculate_pond_infiltration(QP, Fnew, t, tp, tpp, Fhold):
    """积水条件下计算渗透量和径流"""
    status_to_wf_idx = {
        10: (0, lambda: QP.WCI),
        11: (0, lambda: QP.WCI),
        20: (1, lambda idx: theta_of_psi(QP.WF[0].psi, QP.WCS[idx], QP.WCR[idx],
                                         QP.n[idx], QP.m[idx], QP.alpha[idx])),
        21: (1, lambda idx: theta_of_psi(QP.WF[0].psi, QP.WCS[idx], QP.WCR[idx],
                                         QP.n[idx], QP.m[idx], QP.alpha[idx])),
        30: (2, lambda idx: theta_of_psi(QP.WF[1].psi, QP.WCS[idx], QP.WCR[idx],
                                         QP.n[idx], QP.m[idx], QP.alpha[idx])),
        31: (2, lambda idx: theta_of_psi(QP.WF[1].psi, QP.WCS[idx], QP.WCR[idx],
                                         QP.n[idx], QP.m[idx], QP.alpha[idx])),
        40: (3, lambda idx: theta_of_psi(QP.WF[2].psi, QP.WCS[idx], QP.WCR[idx],
                                         QP.n[idx], QP.m[idx], QP.alpha[idx])),
    }

    wf_idx, get_wci = status_to_wf_idx.get(QP.redistStatus, (0, lambda: QP.WCI))

    z = QP.WF[wf_idx].z
    layer_idx = _get_layer_index(QP, z)

    QP.wcs = QP.WCS[layer_idx]
    QP.Ks = QP.ks[layer_idx]

    if callable(get_wci) and get_wci.__code__.co_argcount > 0:
        QP.wci = get_wci(layer_idx)
    else:
        wci_array = get_wci()
        QP.wci = wci_array[layer_idx]

    QP.ks_composite = _calc_ks_composite(QP, z, layer_idx)

    QP.Sav = G(
        psi_of_theta(QP.wcs, QP.WCS[layer_idx], QP.WCR[layer_idx],
                     QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx]),
        psi_of_theta(QP.wci, QP.WCS[layer_idx], QP.WCR[layer_idx],
                     QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx]),
        QP.alpha[layer_idx], QP.n[layer_idx], QP.m[layer_idx], QP.ks[layer_idx]
    )

    QP.cummInfil = calc_f_newton(QP, Fnew, t, tp, tpp, QP.wcs, QP.wci,
                                 QP.Sav, QP.Ks, QP.ks_composite)

    QP.WF[wf_idx].FAmt += QP.cummInfil - Fhold
    QP.runoff += Fnew - QP.cummInfil


def _reallocate_pond_water(QP, R, interTime, deltaTerr, deltaT):
    """积水条件下重新分配水分"""
    for j in range(3):
        QP.WF[j].adjFactor = adj_factor_calc(
            QP, QP.WF[j].numRedist,
            QP.WF[j].redistTime - deltaT + deltaTerr,
            QP.WF[j].z
        )

    redist_handlers = {
        11: lambda: _handle_pond_redist_11(QP, R, interTime, deltaTerr),
        20: lambda: redist20(QP, R, interTime - deltaTerr, interTime, deltaTerr),
        21: lambda: _handle_pond_redist_21(QP, R, interTime, deltaTerr),
        30: lambda: redist30(QP, R, interTime - deltaTerr, interTime, deltaTerr),
        31: lambda: _handle_pond_redist_31(QP, R, interTime, deltaTerr),
        40: lambda: redist40(QP, R, interTime - deltaTerr, interTime, deltaTerr),
        10: lambda: _handle_pond_redist_10(QP),
    }

    handler = redist_handlers.get(QP.redistStatus, lambda: _handle_pond_redist_10(QP))
    handler()


def _handle_pond_redist_11(QP, R, interTime, deltaTerr):
    """积水条件下处理redistStatus=11"""
    if QP.pondingAmt > 0.0:
        QP.redistStatus = 10
        QP.WF[0].z = calc_redist_zf(QP, QP.WF[0].FAmt, QP.WF[0].WC, QP.WCI)
    else:
        redist11(QP, R, interTime - deltaTerr, interTime, deltaTerr)


def _handle_pond_redist_21(QP, R, interTime, deltaTerr):
    """积水条件下处理redistStatus=21"""
    if QP.pondingAmt > 0.0:
        QP.redistStatus = 20
        redist20(QP, R, interTime - deltaTerr, interTime, deltaTerr)
    else:
        redist21(QP, R, interTime - deltaTerr, interTime, deltaTerr)


def _handle_pond_redist_31(QP, R, interTime, deltaTerr):
    """积水条件下处理redistStatus=31"""
    if QP.pondingAmt > 0.0:
        QP.redistStatus = 30
        redist30(QP, R, interTime - deltaTerr, interTime, deltaTerr)
    else:
        redist31(QP, R, interTime - deltaTerr, interTime, deltaTerr)


def _handle_pond_redist_10(QP):
    """积水条件下处理redistStatus=10"""
    layer_idx = _get_layer_index(QP, QP.WF[0].z)

    QP.WF[0].psi = psi_of_theta(
        QP.WF[0].WC, QP.WCS[layer_idx], QP.WCR[layer_idx],
        QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx]
    )
    QP.WF[0].psiHold = psi_of_theta(
        QP.WF[0].WCHold, QP.WCS[layer_idx], QP.WCR[layer_idx],
        QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx]
    )

    new_z, new_layer = _calculate_pond_new_z(QP, QP.WF[0])

    if new_z > QP.boundary_depths[2]:
        _handle_pond_overflow(QP, QP.WF[0])
    else:
        QP.WF[0].z = new_z
        QP.WF[0].WCHold = theta_of_psi(
            QP.WF[0].psiHold, QP.WCS[new_layer], QP.WCR[new_layer],
            QP.n[new_layer], QP.m[new_layer], QP.alpha[new_layer]
        )
        QP.WF[0].WC = theta_of_psi(
            QP.WF[0].psi, QP.WCS[new_layer], QP.WCR[new_layer],
            QP.n[new_layer], QP.m[new_layer], QP.alpha[new_layer]
        )


def _calculate_pond_new_z(QP, wf):
    """积水条件下计算湿润锋的新深度位置"""
    theta_layers = [
        theta_of_psi(wf.psi, QP.WCS[i], QP.WCR[i], QP.n[i], QP.m[i], QP.alpha[i])
        for i in range(3)
    ]

    layer0_water = (theta_layers[0] - QP.WCI[0]) * QP.max_depth[0]
    layer01_water = layer0_water + (theta_layers[1] - QP.WCI[1]) * QP.max_depth[1]

    if wf.FAmt <= layer0_water:
        new_z = calc_redist_zf(QP, wf.FAmt, theta_layers[0], QP.WCI[0])
        return new_z, 0
    elif wf.FAmt <= layer01_water:
        new_z = QP.max_depth[0] + (wf.FAmt - layer0_water) / (theta_layers[1] - QP.WCI[1])
        return new_z, 1
    else:
        new_z = (QP.max_depth[0] + QP.max_depth[1] +
                 (wf.FAmt - layer01_water) / (theta_layers[2] - QP.WCI[2]))
        return new_z, 2


def _handle_pond_overflow(QP, wf):
    """积水条件下处理水分超出土壤深度边界"""
    theta_layers = [
        theta_of_psi(wf.psi, QP.WCS[i], QP.WCR[i], QP.n[i], QP.m[i], QP.alpha[i])
        for i in range(3)
    ]

    max_water = sum(
        (theta_layers[i] - QP.WCI[i]) * QP.max_depth[i]
        for i in range(3)
    )

    QP.runoff += wf.FAmt - max_water
    wf.FAmt = max_water
    wf.z = QP.boundary_depths[2]

    wf.WCHold = theta_of_psi(
        wf.psiHold, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]
    )
    wf.WC = theta_of_psi(
        wf.psi, QP.WCS[2], QP.WCR[2], QP.n[2], QP.m[2], QP.alpha[2]
    )


def fp_calc(QP):
    """计算潜在入渗速率"""
    status_config = {
        40: (3, lambda i: QP.WF[2].WC),
        30: (2, lambda i: QP.WF[1].WC),
        31: (2, lambda i: QP.WF[1].WC),
        20: (1, lambda i: QP.WF[0].WC),
        21: (1, lambda i: QP.WF[0].WC),
    }

    if QP.redistStatus in status_config:
        wf_idx, get_ref_wc = status_config[QP.redistStatus]
    else:
        wf_idx = 0
        get_ref_wc = lambda i: QP.WCI[i]

    QP.WF[wf_idx].FAmt = QP.cummInfil - sum(QP.WF[i].FAmt for i in range(wf_idx))

    z = QP.WF[wf_idx].z
    if z <= QP.boundary_depths[0]:
        layer_idx = 0
    elif z <= QP.boundary_depths[1]:
        layer_idx = 1
    elif z <= QP.boundary_depths[2]:
        layer_idx = 2
    else:
        raise ValueError(f"Depth {z} exceeds boundary_depths")

    QP.ks_composite = _calc_ks_composite(QP, z, layer_idx)

    ref_wc = get_ref_wc(layer_idx)
    QP.Sav = G(
        psi_of_theta(QP.WCS[layer_idx], QP.WCS[layer_idx], QP.WCR[layer_idx],
                     QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx]),
        psi_of_theta(ref_wc, QP.WCS[layer_idx], QP.WCR[layer_idx],
                     QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx]),
        QP.alpha[layer_idx], QP.n[layer_idx], QP.m[layer_idx], QP.ks[layer_idx]
    )

    QP.fp = QP.ks_composite + (QP.ks[layer_idx] * QP.Sav / z)

    return QP.fp


def _calc_ks_composite(QP, z, layer_idx):
    """计算复合导水率"""
    if layer_idx == 0:
        return QP.ks[0]
    elif layer_idx == 1:
        return z / (QP.max_depth[0] / QP.ks[0] +
                    (z - QP.max_depth[0]) / QP.ks[1])
    elif layer_idx == 2:
        return z / (QP.max_depth[0] / QP.ks[0] +
                    QP.max_depth[1] / QP.ks[1] +
                    (z - QP.max_depth[0] - QP.max_depth[1]) / QP.ks[2])


def calc_wco(QP):
    """更新地表含水量"""
    if QP.redistStatus == 10 or QP.redistStatus == 11:
        QP.WCO = QP.WF[0].WC
    elif QP.redistStatus == 20 or QP.redistStatus == 21:
        QP.WCO = QP.WF[1].WC
    elif QP.redistStatus == 30 or QP.redistStatus == 31:
        QP.WCO = QP.WF[2].WC
    else:
        QP.WCO = QP.WF[3].WC

    return


def calc_redist_zf(QP, F, WCA, WCB):
    """计算再分配深度"""
    result = F / (WCA - WCB)
    return result


def calc_rwc(QP):
    """计算相对含水量"""
    if QP.redistStatus in (10, 11):
        QP.RWC = (QP.WF[0].WC - QP.WCR) / (QP.WCS - QP.WCR)
    elif QP.redistStatus in (20, 21):
        QP.RWC = (QP.WF[1].WC - QP.WCR) / (QP.WCS - QP.WCR)
    elif QP.redistStatus in (30, 31):
        QP.RWC = (QP.WF[2].WC - QP.WCR) / (QP.WCS - QP.WCR)
    else:
        QP.RWC = (QP.WF[3].WC - QP.WCR) / (QP.WCS - QP.WCR)


# ===================== 主入口函数 calc_infil =====================

def calc_infil(QP, tstart, tend, deltaT, path):
    """
    计算入渗过程的主函数 - 修改为包含ET支持
    """
    # 初始化和更新
    _update_redistribution_params(QP, tstart, tend, deltaT)

    # 早期退出条件
    if QP.WF[0].z == 0 and QP.precipRate == 0:
        return

    # 主要处理逻辑
    if (QP.precipRate > 0.0) or (QP.pondingAmt > 1E-10):
        if QP.pondingAmt <= 1E-10:
            _process_no_ponding_phase(QP, tstart, tend, deltaT, path)
        else:
            _process_ponding_phase(QP, tstart, tend, deltaT, path)
    else:
        _process_redistribution_only(QP, tstart, tend, deltaT)

    # 计算下一阶段入渗速率
    _calculate_next_fp(QP)


def _update_redistribution_params(QP, tstart, tend, deltaT):
    """更新再分配次数和时间"""
    index = _get_active_wf_index(QP)
    QP.ks_composite = _calc_ks_composite(QP, QP.WF[index].z, _get_layer_index(QP, QP.WF[index].z))

    if ((QP.redistStatus in [10, 20, 30, 40]) and
            (QP.precipRate < QP.ks_composite) and
            (QP.pondingAmt <= 1E-10) and
            QP.WF[0].z != 0):
        QP.numRedist += 1
        _update_wf_redist_params(QP)

    QP.redistTime += deltaT
    for T in range(3):
        QP.WF[T].redistTime += deltaT


def _update_wf_redist_params(QP):
    """根据redistStatus更新湿润锋再分配参数"""
    status_wf_map = {10: 0, 20: 1, 30: 2}

    if QP.redistStatus in status_wf_map:
        wf_idx = status_wf_map[QP.redistStatus]
        QP.redistTime = 0.0
        QP.WF[wf_idx].redistTime = 0.0
        QP.WF[wf_idx].numRedist = QP.numRedist
    elif QP.redistStatus == 40:
        _merge_wetting_fronts(QP)


def _process_no_ponding_phase(QP, tstart, tend, deltaT, path):
    """处理无积水阶段"""
    _update_redist_status(QP)
    _calculate_ponding_time(QP, tstart)

    if QP.timeToPond > tend:
        QP.timeToPond = 99999999
        QP.tpp = 99999999
        calc_no_pond(QP, tstart, tend, deltaT, QP.precipRate, path)
    elif QP.timeToPond < tend:
        calc_no_pond(QP, tstart, QP.timeToPond, deltaT, QP.precipRate, path)
        calc_pond(QP, QP.timeToPond, tend, QP.timeToPond, QP.tpp, deltaT, QP.precipRate, path)


def _update_redist_status(QP):
    """更新再分配状态"""
    status_transitions = {
        (10, 'low'): 11,
        (11, 'high'): 20,
        (20, 'low'): 21,
        (21, 'high'): 30,
        (30, 'low'): 31,
        (31, 'high'): 40,
        (40, 'low'): 31,
    }

    rate_type = 'low' if QP.precipRate < QP.ks_composite else 'high'
    key = (QP.redistStatus, rate_type)

    if key in status_transitions:
        new_status = status_transitions[key]

        if new_status in [20, 30, 40] and new_status > QP.redistStatus:
            _create_new_wetting_front(QP, new_status)
        elif QP.redistStatus == 40 and new_status == 31:
            _merge_and_update_wf(QP)

        QP.redistStatus = new_status


def _create_new_wetting_front(QP, new_status):
    """创建新的湿润锋"""
    wf_index_map = {20: 1, 30: 2, 40: 3}
    wf_idx = wf_index_map[new_status]

    QP.WF[wf_idx].FAmt = QP.cummInfil - sum(QP.WF[i].FAmt for i in range(wf_idx))
    QP.WF[wf_idx].WC = QP.WCS[0]
    QP.WF[wf_idx].WCHold = QP.WCS[0]
    QP.fp = 0.0


def _calculate_ponding_time(QP, tstart):
    """计算积水开始时间"""
    wf_idx = _get_redist_wf_index(QP.redistStatus)
    _set_soil_parameters(QP, wf_idx)

    if (QP.fp > 1E-10) and (QP.precipRate >= QP.fp):
        QP.timeToPond = tstart
        QP.Bfp = QP.cummInfil
    elif QP.precipRate > QP.ks_composite:
        if QP.precipRate > QP.ks_composite:
            QP.Bfp = QP.Ks * QP.Sav * (QP.wcs - QP.wci) / (QP.precipRate - QP.ks_composite)
            QP.timeToPond = QP.Bfp / QP.precipRate + tstart
        else:
            QP.timeToPond = 99999999
    else:
        QP.timeToPond = 99999999

    if QP.timeToPond < 99999999:
        _calculate_tpp(QP)


def _set_soil_parameters(QP, wf_idx):
    """设置土壤参数"""
    z = QP.WF[wf_idx].z
    layer_idx = _get_layer_index(QP, z)

    QP.wcs = QP.WCS[layer_idx]
    QP.Ks = QP.ks[layer_idx]
    QP.ks_composite = _calc_ks_composite(QP, z, layer_idx)

    wci_source_map = {
        (10, 11): lambda i: QP.WCI[i],
        (20, 21): lambda i: theta_of_psi(QP.WF[0].psi, QP.WCS[i], QP.WCR[i], QP.n[i], QP.m[i], QP.alpha[i]),
        (30, 31): lambda i: theta_of_psi(QP.WF[1].psi, QP.WCS[i], QP.WCR[i], QP.n[i], QP.m[i], QP.alpha[i]),
        (40,): lambda i: theta_of_psi(QP.WF[2].psi, QP.WCS[i], QP.WCR[i], QP.n[i], QP.m[i], QP.alpha[i]),
    }

    for statuses, func in wci_source_map.items():
        if QP.redistStatus in statuses:
            QP.wci = func(layer_idx)
            break

    QP.Sav = G(
        psi_of_theta(QP.wcs, QP.WCS[layer_idx], QP.WCR[layer_idx], QP.n[layer_idx], QP.m[layer_idx],
                     QP.alpha[layer_idx]),
        psi_of_theta(QP.wci, QP.WCS[layer_idx], QP.WCR[layer_idx], QP.n[layer_idx], QP.m[layer_idx],
                     QP.alpha[layer_idx]),
        QP.alpha[layer_idx], QP.n[layer_idx], QP.m[layer_idx], QP.ks[layer_idx]
    )


def _calculate_tpp(QP):
    """计算tpp参数"""
    denominator = QP.Ks * QP.Sav * (QP.wcs - QP.wci)
    if denominator > 0:
        log_term = math.log(1 + (QP.ks_composite * QP.Bfp / denominator))
        QP.tpp = (QP.Bfp - denominator * log_term / QP.ks_composite) / QP.ks_composite
    else:
        QP.tpp = 0


def _process_ponding_phase(QP, tstart, tend, deltaT, path):
    """处理积水阶段"""
    QP.timeToPond = tstart

    if QP.Bfp == 0.0:
        QP.tpp = tstart

    if QP.redistStatus == 40 and QP.precipRate < QP.ks_composite:
        _merge_and_update_wf(QP)

    t1 = _calculate_ponding_end_time(QP, tstart, tend)

    if t1 > tend:
        calc_pond(QP, tstart, tend, QP.timeToPond, QP.tpp, deltaT, QP.precipRate, path)
    elif t1 < tend:
        calc_pond(QP, tstart, t1, QP.timeToPond, QP.tpp, deltaT, QP.precipRate, path)
        _transition_to_redistribution(QP)
        calc_no_pond(QP, t1, tend, deltaT, QP.precipRate, path)


def _calculate_ponding_end_time(QP, tstart, tend):
    """计算积water结束时间"""
    if QP.precipRate >= QP.fp:
        return tend + 99999999.0

    wf_idx = _get_redist_wf_index(QP.redistStatus)
    ref_wf_idx = max(0, wf_idx - 1) if wf_idx > 0 else 0

    wci = QP.WCI if wf_idx == 0 else QP.WF[ref_wf_idx].WC

    QP.Bfp = QP.WF[wf_idx].FAmt
    denominator = QP.Sav * (QP.WCS - wci)
    if denominator > 0:
        QP.tpp = (QP.Bfp - denominator * math.log(1 + (QP.Bfp / denominator))) / QP.ks

    total_infil = QP.WF[wf_idx].FAmt + QP.pondingAmt
    return calc_t_newton(QP, tstart, tend, QP.timeToPond, QP.tpp, QP.precipRate, total_infil, wci, QP.Sav)


def _transition_to_redistribution(QP):
    """从积水状态转换到再分配状态"""
    transitions = {10: 11, 20: 21, 30: 31}
    if QP.redistStatus in transitions and QP.precipRate < QP.ks:
        QP.redistStatus = transitions[QP.redistStatus]


def _process_redistribution_only(QP, tstart, tend, deltaT):
    """
    仅进行再分配,不进行入渗 - 修改为包含ET
    """
    if QP.redistStatus == 40:
        _merge_and_update_wf(QP)
    elif QP.redistStatus in [10, 20, 30]:
        QP.redistStatus += 1

    nsteps = int((tend - tstart) / deltaT)
    inter_time = tstart

    for _ in range(nsteps):
        inter_time += deltaT
        _update_time_step(QP, inter_time)

        # ===== 新增: 在再分配过程中也处理ET =====
        if hasattr(QP, 'PET_rate') and QP.PET_rate > 0:
            actual_ET_demand = calculate_et_demand(QP, QP.PET_rate, deltaT)
            actual_ET = apply_et_to_soil(QP, actual_ET_demand)
            if hasattr(QP, 'actual_ET_vec'):
                QP.actual_ET_vec.append(actual_ET)
        # ===== ET处理结束 =====

        redistribution(QP, inter_time, deltaT)
        calc_wco(QP)
        calc_rwc(QP)

    delta_terr = tend - inter_time
    if delta_terr > 1E-10:
        inter_time = tend
        _update_time_step(QP, inter_time)

        # ===== 新增: 处理剩余时间的ET =====
        if hasattr(QP, 'PET_rate') and QP.PET_rate > 0:
            actual_ET_demand = calculate_et_demand(QP, QP.PET_rate, delta_terr)
            actual_ET = apply_et_to_soil(QP, actual_ET_demand)
            if hasattr(QP, 'actual_ET_vec'):
                QP.actual_ET_vec.append(actual_ET)
        # ===== ET处理结束 =====

        redistribution(QP, inter_time, delta_terr)
        calc_wco(QP)
        calc_rwc(QP)


def _update_time_step(QP, inter_time):
    """更新时间步参数"""
    QP.time = inter_time
    QP.pondingAmt = 0.0


def _merge_wetting_fronts(QP):
    """合并湿润锋(从40到31状态)"""
    QP.WF[0].redistTime = QP.WF[1].redistTime
    QP.WF[0].numRedist = QP.WF[1].numRedist

    QP.WF[1].redistTime = QP.WF[2].redistTime
    QP.WF[1].numRedist = QP.WF[2].numRedist

    QP.WF[2].redistTime = 0.0
    QP.WF[2].numRedist = QP.numRedist


def _merge_and_update_wf(QP):
    """合并并更新湿润锋"""
    QP.redistStatus = 31 if QP.redistStatus == 40 else 30

    QP.WF[0].FAmt += QP.WF[1].FAmt
    _copy_wf_properties(QP.WF[0], QP.WF[1])

    _copy_wf_all_properties(QP.WF[1], QP.WF[2])
    _copy_wf_all_properties(QP.WF[2], QP.WF[3])

    _clear_wf(QP.WF[3])
    QP.fp = 0.0

    _update_merged_wf_position(QP)


def _copy_wf_properties(target, source):
    """复制湿润锋属性(不包括z和FAmt)"""
    target.WC = source.WC
    target.WCHold = source.WCHold
    target.psi = source.psi
    target.psiHold = source.psiHold
    target.numRedist = source.numRedist
    target.redistTime = source.redistTime
    target.adjFactor = source.adjFactor


def _copy_wf_all_properties(target, source):
    """复制湿润锋全部属性"""
    target.FAmt = source.FAmt
    target.z = source.z
    _copy_wf_properties(target, source)


def _clear_wf(wf):
    """清空湿润锋"""
    wf.FAmt = 0.0
    wf.WC = 0.0
    wf.WCHold = 0.0
    wf.numRedist = 0.0
    wf.redistTime = 0.0
    wf.adjFactor = 0.0
    wf.z = 0.0


def _update_merged_wf_position(QP):
    """更新合并后的湿润锋位置"""
    _update_no_pond_wf_position(QP, 0)


def _calculate_next_fp(QP):
    """计算下一阶段的入渗速率fp"""
    status_wf_map = {
        40: (3, lambda: QP.WF[2].WC),
        30: (2, lambda: QP.WF[1].WC),
        31: (2, lambda: QP.WF[1].WC),
        20: (1, lambda: QP.WF[0].WC),
        21: (1, lambda: QP.WF[0].WC),
    }

    if QP.redistStatus in status_wf_map:
        wf_idx, get_ref_wc = status_wf_map[QP.redistStatus]
        ref_wc = get_ref_wc()
    else:
        wf_idx = 0
        ref_wc = QP.WCI

    QP.WF[wf_idx].FAmt = QP.cummInfil - sum(QP.WF[i].FAmt for i in range(wf_idx))

    z = QP.WF[wf_idx].z
    layer_idx = _get_layer_index(QP, z)

    QP.ks_composite = _calc_ks_composite(QP, z, layer_idx)

    if isinstance(ref_wc, (list, tuple, np.ndarray)):
        ref_wc_value = ref_wc[layer_idx]
    else:
        ref_wc_value = ref_wc

    QP.Sav = G(
        psi_of_theta(QP.WCS[layer_idx], QP.WCS[layer_idx], QP.WCR[layer_idx],
                     QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx]),
        psi_of_theta(ref_wc_value, QP.WCS[layer_idx], QP.WCR[layer_idx],
                     QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx]),
        QP.alpha[layer_idx], QP.n[layer_idx], QP.m[layer_idx], QP.ks[layer_idx]
    )

    _calculate_fp_value(QP, wf_idx, layer_idx, ref_wc_value)


def _calculate_fp_value(QP, wf_idx, layer_idx, ref_wc):
    """计算fp值"""
    if QP.redistStatus in [20, 21] and layer_idx > 0:
        theta_ref = theta_of_psi(QP.WF[0].psi, QP.WCS[layer_idx], QP.WCR[layer_idx],
                                 QP.n[layer_idx], QP.m[layer_idx], QP.alpha[layer_idx])
        QP.fp = QP.ks_composite + (QP.ks[layer_idx] * QP.Sav * (QP.WCS[layer_idx] - theta_ref) / QP.WF[wf_idx].FAmt)
    elif QP.redistStatus in [10, 11] and layer_idx > 0:
        QP.fp = QP.ks_composite + (
                QP.ks[layer_idx] * QP.Sav * (QP.WCS[layer_idx] - QP.WCI[layer_idx]) / QP.WF[wf_idx].FAmt)
    else:
        QP.fp = QP.ks_composite + (QP.ks[layer_idx] * QP.Sav / QP.WF[wf_idx].z)


def _get_active_wf_index(QP):
    """获取活跃的湿润锋索引"""
    Z_list = [QP.WF[i].z for i in range(4)]
    return np.count_nonzero(Z_list) - 1


def _get_redist_wf_index(redist_status):
    """根据redistStatus获取对应的湿润锋索引"""
    status_map = {10: 0, 11: 0, 20: 1, 21: 1, 30: 2, 31: 2, 40: 3}
    return status_map.get(redist_status, 0)

