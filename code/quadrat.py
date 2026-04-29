class WettingFrontData:
    def __init__(self):
        self.z = 0.0  # 湿润前沿深度
        self.FAmt = 0.0  # 湿润前沿的水分量
        self.WC = 0.0  # 湿润前沿的水分含量
        self.WCHold = 0.0  # 湿润前沿保留的水分含量
        self.psi = 0.0
        self.psiHold = 0.0
        self.numRedist = 0  # 重新分布的阶段数量
        self.redistTime = 0.0  # 重新分布的时间
        self.adjFactor = 0.0  # 调整因子

    def __repr__(self):
        return (f"WettingFrontData:\n"
                f"  z: {self.z}\n"
                f"  FAmt: {self.FAmt}\n"
                f"  WC: {self.WC}\n"
                f"  WCHold: {self.WCHold}\n"
                f"  psi: {self.psi}\n"
                f"  psiHold: {self.psiHold}\n"
                f"  numRedist: {self.numRedist}\n"
                f"  redistTime: {self.redistTime}\n"
                f"  adjFactor: {self.adjFactor}\n")


class QuadratData:
    def __init__(self):
        self.Sav = 0.0  # 平均湿润前沿吸力或毛细压力
        self.PsiB = 0.0  # 气泡压力
        self.lamda = 0.0  # 孔隙大小分布指数
        self.WCI = 0.0  # 初始水分含量（萎蔫点）
        self.WCS = 0.0  # 饱和水分含量
        self.WCR = 0.0  # 残余水分含量
        self.ks = 0.0  # 饱和导水率
        self.Smax = 0.0  # 最大洼地蓄水量
        self.WCMin = 0.0  # 最小水分含量
        self.NumObsLayers = 0  # 观测层的数量
        self.precips = []  # 降水强度数组

        # 以下是计算出的数据
        self.pondFlag = 0  # 标志是否发生积水
        self.redistStatus = 10  # 标志重新分布的状态
        self.WCO = 0.0  # 表面水分含量
        self.RWC = 0.0  # 相对水分含量
        self.fp = 0.0  # 实际渗透率
        self.Bfp = 0.0  # 潜在渗透率
        self.timeToPond = 0.0  # 积水所需的时间
        self.tpp = 0.0  # 时间偏移
        self.precipRate = 0.0  # 降水率
        self.cummPrecip = 0.0  # 累积降水量
        self.pondingAmt = 0.0  # 表面积水量
        self.infilRate = 0.0  # 渗透率
        self.cummInfil = 0.0  # 累积渗透量
        self.runoff = 0.0  # 可用于径流的水量

        # 重新分布阶段计数
        self.numRedist = 0
        self.time = 0.0
        self.redistTime = 0.0
        self.tpAdj = 0

        # 初始化湿润前沿
        self.WF = [WettingFrontData() for _ in range(4)]  # 最多4个湿润前沿

    def __repr__(self):
        return (f"QuadratData:\n"
                f"  Sav: {self.Sav}\n"
                f"  PsiB: {self.PsiB}\n"
                f"  lamda: {self.lamda}\n"
                f"  WCI: {self.WCI}\n"
                f"  WCS: {self.WCS}\n"
                f"  WCR: {self.WCR}\n"
                f"  ks: {self.ks}\n"
                f"  Smax: {self.Smax}\n"
                f"  WCMin: {self.WCMin}\n"
                f"  NumObsLayers: {self.NumObsLayers}\n"
                f"  pondFlag: {self.pondFlag}"
                f"  redistStatus: {self.redistStatus}"
                f"  WCO: {self.WCO}"
                f"  RWC: {self.RWC}"
                f"  fp: {self.fp}"
                f"  Bfp: {self.Bfp}"
                f"  timeToPond: {self.timeToPond}"
                f"  tpp: {self.tpp}"
                f"  precipRate: {self.precipRate}"
                f"  cummPrecip: {self.cummPrecip}"
                f"  pondingAmt: {self.pondingAmt}"
                f"  infilRate: {self.infilRate}"
                f"  cummInfil: {self.cummInfil}"
                f"  runoff: {self.runoff}"
                f"  numRedist: {self.numRedist}"
                f"  time: {self.time}"
                f"  redistTime: {self.redistTime}"
                f"  tpAdj: {self.tpAdj}"
                f"  WF: {self.WF}")


def build_initial_quadrat_records(precip_data: object, parameters: object, initial_psi: object, num_layers: object, boundary_depths: object) -> object:
    QP = QuadratData()

    # 遍历每个网格进行数据初始化
    QP.precips = precip_data
    QP.parameters = parameters

    # 读取土壤属性数据
    QP.WCR = parameters[:, 0]
    QP.WCS = parameters[:, 1]
    QP.ks = parameters[:, 2]
    QP.alpha = parameters[:, 3]
    QP.n = parameters[:, 4]
    QP.m = parameters[:, 5]
    QP.max_depth = parameters[:, 6]

    QP.WCI = []
    for k in range(0, num_layers):
        # 通过循环遍历参数列表中的每个土壤层
        temp_theta_for_init = theta_of_psi(initial_psi, QP.WCR[k], QP.WCS[k], QP.alpha[k], QP.n[k], QP.m[k])
        # 根据当前土壤层的参数计算临时的初始θ值
        QP.WCI.append(temp_theta_for_init)
    QP.boundary_depths = boundary_depths

    QP.lambda_vec = []
    QP.psi_b_vec = []
    for u in range(0, num_layers):
        p = 1 + 2 / QP.m[u]
        QP.lambda_vec.append(2 / (p - 3))
        QP.psi_b_vec.append((p + 3) * (147.8 + 8.1 * p + 0.092 * p ** 2) / (2 * QP.alpha[u] * p * (p - 1) * (55.6 + 7.4 * p + p ** 2)))

    return QP


def theta_of_psi(psi, theta_r, theta_s, alpha, n, m):
    return theta_r + (theta_s - theta_r) / (1 + (alpha * psi) ** n) ** m


def initialize_computation_variables(QP):
    QP.pondFlag = 0
    QP.redistStatus = 10
    QP.WCO = 0.0
    QP.RWC = 0.0
    QP.fp = 0.0
    QP.Bfp = 0.0
    QP.timeToPond = 0.0
    QP.tpp = 0.0
    QP.precipRate = 0.0
    QP.cummPrecip = 0.0
    QP.pondingAmt = 0.0
    QP.infilRate = 0.0
    QP.cummInfil = 0.0
    QP.runoffAmt = 0.0
    QP.numRedist = 0
    QP.redistTime = 0.0
    QP.time = 0.0
    QP.tpAdj = 0

    # 初始化湿润前沿
    for T in range(4):
        QP.WF[T].z = 0.0
        QP.WF[T].FAmt = 0.0
        QP.WF[T].WC = QP.WCI
        QP.WF[T].WCHold = 0.0
        QP.WF[T].psi = 0.0
        QP.WF[T].psiHold = 0.0
        QP.WF[T].numRedist = 0
        QP.WF[T].redistTime = 0.0
        QP.WF[T].adjFactor = 0.0
