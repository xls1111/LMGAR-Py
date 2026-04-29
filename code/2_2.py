import datetime
from quadrat import build_initial_quadrat_records
# from InfilHydrol_simplified import calc_infil
from InfilHydrol_0115 import calc_infil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 创建 Result 文件夹 ==========
result_folder = r'D:\文件\paper2\DATA\4-1\2'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    print(f'已创建文件夹: {result_folder}')


# ========== 定义误差统计函数 ==========
def calculate_kge(obs, sim):
    """计算 Kling-Gupta Efficiency (KGE)"""
    obs = np.array(obs)
    sim = np.array(sim)

    # 移除NaN值
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    # 计算相关系数
    r = np.corrcoef(obs, sim)[0, 1]

    # 计算偏差比
    alpha = np.std(sim) / np.std(obs)

    # 计算均值比
    beta = np.mean(sim) / np.mean(obs)

    # 计算KGE
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge


def calculate_nse(obs, sim):
    """计算 Nash-Sutcliffe Efficiency (NSE)"""
    obs = np.array(obs)
    sim = np.array(sim)

    # 移除NaN值
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    # 计算NSE
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    if denominator == 0:
        return np.nan

    nse = 1 - (numerator / denominator)

    return nse


def calculate_pbias(obs, sim):
    """计算 Percent Bias (PBIAS) in %"""
    obs = np.array(obs)
    sim = np.array(sim)

    # 移除NaN值
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0 or np.sum(obs) == 0:
        return np.nan

    # 计算PBIAS
    pbias = 100 * np.sum(obs - sim) / np.sum(obs)

    return pbias


def calculate_rmse(obs, sim):
    """计算 Root Mean Square Error (RMSE) in mm/h"""
    obs = np.array(obs)
    sim = np.array(sim)

    # 移除NaN值
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    # 计算RMSE (mm/h)
    rmse = np.sqrt(np.mean((obs - sim) ** 2))

    return rmse


# ---------------------------------------------------------------------------------------------------------------------#
precipFilename = r"D:\program\LGAR-Py\Figure\2\MLGAR-2\DATA\forcing_data_resampled_synth_2.csv"
SomeTxtFile = os.path.join(result_folder, 'sample_MLGAR_output.txt')
# ---------------------------------------------------------------------------------------------------------------------#
MaxNumQuadrats = 1
MaxNumLayers = 3
NumQuadrats = 1
deltaT = 1.0
length_of_simulation = 144
forcing_data = pd.read_csv(precipFilename, index_col=0, parse_dates=True)
precip_data = np.array(forcing_data['P(mm/h)'])[0:length_of_simulation]
PET_data = np.array(forcing_data['PET(mm/h)'])[0:length_of_simulation]
# ---------------------------------------------------------------------------------------------------------------------#
h_p_max = 0
initial_psi = 1000
num_layers = 3
# 第一层: loam
theta_r_layer_0 = 0.034
theta_s_layer_0 = 0.46
K_s_layer_0 = 2.5
alpha_layer_0 = 0.0016
n_layer_0 = 1.37
m_layer_0 = 1-1/n_layer_0
max_depth_layer_0 = 100
param_set_0 = [theta_r_layer_0, theta_s_layer_0, K_s_layer_0, alpha_layer_0, n_layer_0, m_layer_0, max_depth_layer_0]
# 第二层: clay loam
theta_r_layer_1 = 0.089
theta_s_layer_1 = 0.43
K_s_layer_1 = 0.7
alpha_layer_1 = 0.0010
n_layer_1 = 1.23
m_layer_1 = 1-1/n_layer_1
max_depth_layer_1 = 300
param_set_1 = [theta_r_layer_1, theta_s_layer_1, K_s_layer_1, alpha_layer_1, n_layer_1, m_layer_1, max_depth_layer_1]
# 第三层: silty clay loam
theta_r_layer_2 = 0.068
theta_s_layer_2 = 0.38
K_s_layer_2 = 2
alpha_layer_2 = 0.0008
n_layer_2 = 1.09
m_layer_2 = 1-1/n_layer_2
max_depth_layer_2 = 300
param_set_2 = [theta_r_layer_2, theta_s_layer_2, K_s_layer_2, alpha_layer_2, n_layer_2, m_layer_2, max_depth_layer_2]

parameters = np.vstack([param_set_0, param_set_1, param_set_2])

if len(parameters) > num_layers:
    while len(parameters) > num_layers:
        parameters = np.delete(parameters, len(parameters) - 1, 0)

theta_r_vec = []
theta_s_vec = []
K_s_vec = []
alpha_vec = []
n_vec = []
m_vec = []
max_depth_vec = []

for param_set_num in range(0, num_layers):
    theta_r_vec.append(parameters[param_set_num][0])
    theta_s_vec.append(parameters[param_set_num][1])
    K_s_vec.append(parameters[param_set_num][2])
    alpha_vec.append(parameters[param_set_num][3])
    n_vec.append(parameters[param_set_num][4])
    m_vec.append(parameters[param_set_num][5])
    max_depth_vec.append(parameters[param_set_num][6])

max_layer = len(parameters) - 1
boundary_depths = []
temp_bdy_depth = 0
for s in range(0, len(parameters)):
    temp_bdy_depth = temp_bdy_depth + parameters[s][-1]
    boundary_depths.append(temp_bdy_depth)
# ---------------------------------------------------------------------------------------------------------------------#
init_time = datetime.datetime.now()

# 创建样方并初始化变量
QP = build_initial_quadrat_records(precip_data, parameters, initial_psi, num_layers, boundary_depths)

tstart = 0
tend = 1.0 / 12
deltaT = 1.0 / 12

# 打开文件以写入数据
with open(SomeTxtFile, 'w') as file:
    file.write('Time F1 Z1 WC1 F2 Z2 WC2 F3 Z3 WC3 F4 Z4 WC4 runoff CummInfil\n')

# ← 修正：runoff 和 cummInfil 都是累积量
runoff = [0] * length_of_simulation  # 累积径流量 (mm)
cummInfil = [0] * length_of_simulation  # 累积下渗量 (mm)

# 主计算循环
print('开始计算...')
for x in range(length_of_simulation):
    # if x > 1705:
    # print(x)
    QP.precipRate = QP.precips[x]
    calc_infil(QP, tstart, tend, deltaT, SomeTxtFile)
    runoff[x] = QP.runoff  # ← 累积径流量 (mm)
    cummInfil[x] = QP.cummInfil  # ← 累积下渗量 (mm)
    tstart += 1.0 / 12
    tend += 1.0 / 12

    with open(SomeTxtFile, 'a') as f:
        f.write(f"{int(x)} ")
        for i in range(0, 4):
            f.write(f"{QP.WF[i].FAmt:.3f} ")
            f.write(f"{QP.WF[i].z:.3f} ")
            f.write(f"{QP.WF[i].WC:.3f} ")
        f.write(f"{QP.runoff:.3f} ")
        f.write(f"{QP.cummInfil:.3f}\n")

end_time = datetime.datetime.now()
print(f'计算完成！用时: {end_time - init_time}')

# ========== 构建 output DataFrame ==========
time_step = 300 / 3600  # 5分钟 = 1/12小时

# 创建索引
row_names = forcing_data.index[:length_of_simulation].tolist()

# ← 修正：先存储累积量
output = pd.DataFrame({
    'P(mm/h)': precip_data,
    'PET(mm/h)': PET_data,
    'cumulative_runoff': runoff,  # ← 累积径流量 (mm)
    'cummInfil': cummInfil,  # ← 累积下渗量 (mm)
}, index=row_names)

# ← 修正：从累积量计算速率
# 1. 计算每个时间步的径流量（mm）
runoff_per_step = output['cumulative_runoff'].diff().fillna(output['cumulative_runoff'].iloc[0])
# 2. 转换为径流速率（mm/h）
output['runoff[mm/h]'] = runoff_per_step / time_step

# 3. 计算每个时间步的下渗量（mm）
infil_per_step = output['cummInfil'].diff().fillna(output['cummInfil'].iloc[0])
# 4. 转换为下渗速率（mm/h）
output['actual_infil[mm/h]'] = infil_per_step / time_step

# 添加其他字段
output['actual_ET_per_step(mm)'] = 0
output['actual_ET[mm/h]'] = output['actual_ET_per_step(mm)'] / time_step
output['bottom_flux[mm/h]'] = 0
output['water_in_soil[mm]'] = 0
output['mass_bal_error(mm)'] = 0

# ========== 打印统计信息（与原始代码逻辑一致）==========
print('\n' + '=' * 60)
print('当前计算结果统计：')
print('=' * 60)

# 累积降水
print('cumulative precip')
cumulative_precip_mm_1 = sum(output['P(mm/h)']) * 5 / 60
print(cumulative_precip_mm_1)
print(' ')

# ← 修正：累积径流的正确计算
print('cumulative runoff')
cumulative_runoff_mm_1 = sum(output['runoff[mm/h]']) * 5 / 60  # 速率求和×时间步
print(cumulative_runoff_mm_1)
print(' ')

# 径流系数
print('runoff efficiency')
runoff_efficiency = sum(output['runoff[mm/h]']) / sum(output['P(mm/h)'])
print(runoff_efficiency)
print(' ')

# 使用 time_step 的计算
print("cumulative precip (mm)")
cumulative_precip_mm = sum(output['P(mm/h)']) * time_step
print(cumulative_precip_mm)

print("runoff (mm)")
cumulative_runoff_mm = sum(output['runoff[mm/h]']) * time_step  # ← 修正：速率求和×时间步
print(cumulative_runoff_mm)

print("actual_infil (mm)")
actual_infil_sum = sum(output['actual_infil[mm/h]']) * time_step
print(actual_infil_sum)

print("cumulative infil + runoff (mm)")
check_sum = actual_infil_sum + cumulative_runoff_mm
print(check_sum)

# ← 修正：验证累积量的一致性
print("\n" + "-" * 60)
print("数据一致性验证：")
print("-" * 60)

print("累积径流量：")
print(f"  从 runoff[mm/h] 求和: {cumulative_runoff_mm:.6f} mm")
print(f"  从 cumulative_runoff 最后值: {output['cumulative_runoff'].iloc[-1]:.6f} mm")
print(f"  差异: {abs(cumulative_runoff_mm - output['cumulative_runoff'].iloc[-1]):.9f} mm")

print("\n累积下渗量：")
print(f"  从 actual_infil[mm/h] 求和: {actual_infil_sum:.6f} mm")
print(f"  从 cummInfil 最后值: {output['cummInfil'].iloc[-1]:.6f} mm")
print(f"  差异: {abs(actual_infil_sum - output['cummInfil'].iloc[-1]):.9f} mm")

print('=' * 60)

# ========== 保存统计信息 ==========
stats_file = os.path.join(result_folder, 'statistics_summary.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write('=' * 60 + '\n')
    f.write('MLGAR 计算结果统计摘要\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'运行时间: {end_time - init_time}\n')
    f.write(f'模拟时长: {length_of_simulation} 个时间步\n')
    f.write(f'时间步长: {time_step:.6f} 小时 ({time_step * 60:.1f} 分钟)\n\n')
    f.write('-' * 60 + '\n')
    f.write('水量平衡:\n')
    f.write('-' * 60 + '\n')
    f.write(f'累积降水量 (mm):           {cumulative_precip_mm:.6f}\n')
    f.write(f'累积径流量 (mm):           {cumulative_runoff_mm:.6f}\n')
    f.write(f'累积下渗量 (mm):           {actual_infil_sum:.6f}\n')
    f.write(f'下渗+径流 (mm):            {check_sum:.6f}\n')
    f.write(f'水量平衡误差 (mm):         {abs(cumulative_precip_mm - check_sum):.9f}\n\n')
    f.write('-' * 60 + '\n')
    f.write('数据一致性验证:\n')
    f.write('-' * 60 + '\n')
    f.write('累积径流量：\n')
    f.write(f'  从 runoff[mm/h] 求和:      {cumulative_runoff_mm:.6f} mm\n')
    f.write(f'  从 cumulative_runoff 最后值: {output["cumulative_runoff"].iloc[-1]:.6f} mm\n')
    f.write(
        f'  差异:                      {abs(cumulative_runoff_mm - output["cumulative_runoff"].iloc[-1]):.9f} mm\n\n')
    f.write('累积下渗量：\n')
    f.write(f'  从 actual_infil[mm/h] 求和: {actual_infil_sum:.6f} mm\n')
    f.write(f'  从 cummInfil 最后值:       {output["cummInfil"].iloc[-1]:.6f} mm\n')
    f.write(f'  差异:                      {abs(actual_infil_sum - output["cummInfil"].iloc[-1]):.9f} mm\n\n')
    f.write('-' * 60 + '\n')
    f.write('效率指标:\n')
    f.write('-' * 60 + '\n')
    f.write(f'径流系数:                  {runoff_efficiency:.6f}\n')
    f.write(f'下渗系数:                  {actual_infil_sum / cumulative_precip_mm:.6f}\n')
    f.write('=' * 60 + '\n')

print(f'\n统计摘要已保存到: {stats_file}')

# ========== 绘图 ==========
print('\n开始绘图...')

plt.figure(figsize=(10, 6))
plt.plot(output['P(mm/h)'], label='Precip (mm/h)', linewidth=2)
plt.plot(output['runoff[mm/h]'], label='runoff (mm/h)', linewidth=2)
plt.plot(output['actual_infil[mm/h]'], label='actual infiltration (mm/h)', linewidth=2)
plt.plot(output['actual_ET[mm/h]'], label='actual ET (mm/h)', linewidth=2)
plt.legend()
plt.ylabel('通量 (mm/h)', fontsize=12)
plt.xlabel('时间', fontsize=12)
plt.title('降水-径流-下渗-蒸发过程')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_folder, 'Precip_runoff_infiltration_ET.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(output['bottom_flux[mm/h]'])
plt.legend(labels=['bottom_flux'])
plt.ylabel('底部通量 (mm/h)', fontsize=12)
plt.xlabel('时间', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_folder, 'bottom_flux.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(output['water_in_soil[mm]'])
plt.xlabel('时间', fontsize=12)
plt.ylabel('土壤含水量 (mm)', fontsize=12)
plt.legend(labels=['water_in_soil'])
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_folder, 'water_in_soil.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(output['mass_bal_error(mm)'])
plt.xlabel('时间', fontsize=12)
plt.ylabel('质量平衡误差 (mm)', fontsize=12)
plt.legend(labels=['mass_bal_error'])
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_folder, 'mass_bal_error.png'), dpi=300, bbox_inches='tight')
plt.close()

# ========== 计算累积量 ==========
cumulative_precip = []
cumulative_runoff = []
cumulative_infilt = []
cumulative_botflx = []
cumulative_evptrs = []
cumulative_evptrs_pot = []

accumulated_precip = 0
accumulated_runoff = 0
accumulated_infilt = 0
accumulated_botflx = 0
accumulated_evptrs = 0
accumulated_evptrs_pot = 0

for i in range(len(output)):
    accumulated_precip = accumulated_precip + output['P(mm/h)'].iloc[i] * time_step
    cumulative_precip.append(accumulated_precip)

    accumulated_runoff = accumulated_runoff + output['runoff[mm/h]'].iloc[i] * time_step
    cumulative_runoff.append(accumulated_runoff)

    accumulated_infilt = accumulated_infilt + output['actual_infil[mm/h]'].iloc[i] * time_step
    cumulative_infilt.append(accumulated_infilt)

    accumulated_botflx = accumulated_botflx + output['bottom_flux[mm/h]'].iloc[i] * time_step
    cumulative_botflx.append(accumulated_botflx)

    accumulated_evptrs = accumulated_evptrs + output['actual_ET_per_step(mm)'].iloc[i]
    cumulative_evptrs.append(accumulated_evptrs)

    accumulated_evptrs_pot = accumulated_evptrs_pot + output['PET(mm/h)'].iloc[i] * time_step
    cumulative_evptrs_pot.append(accumulated_evptrs_pot)

output['cumulative_precip'] = cumulative_precip
output['cumulative_runoff_calc'] = cumulative_runoff  # ← 从速率累加计算的
output['cumulative_infilt'] = cumulative_infilt
output['cumulative_botflx'] = cumulative_botflx
output['cumulative_evptrs'] = cumulative_evptrs
output['cumulative_evptrs_pot'] = cumulative_evptrs_pot

# ========== 读取 LGAR 模型数据 ==========
print('\n' + '=' * 60)
print('读取 LGAR 模型对照数据...')
print('=' * 60)

try:
    LGAR_output = pd.read_pickle(r"D:\program\LGAR-Py-LGAR-Py_public\outputs\output_synth_2.pkl")
    print(f'成功读取 LGAR 数据，共 {len(LGAR_output)} 行')
    print(f'LGAR 数据列名: {list(LGAR_output.columns)}')

    # 确保 LGAR 数据长度与当前数据一致
    lgar_length = min(len(LGAR_output), length_of_simulation)
    LGAR_output = LGAR_output.iloc[:lgar_length]

    # 检查 LGAR 数据中是否有需要的列
    lgar_has_data = True
    required_cols = ['cumulative_runoff', 'cumulative_infilt']
    missing_cols = [col for col in required_cols if col not in LGAR_output.columns]

    if missing_cols:
        print(f'警告: LGAR 数据缺少以下列: {missing_cols}')
        print('尝试从速率数据计算累积量...')

        # 如果有速率数据，尝试计算累积量
        if 'runoff[mm/h]' in LGAR_output.columns:
            LGAR_output['cumulative_runoff'] = (LGAR_output['runoff[mm/h]'] * time_step).cumsum()
            print('  已从 runoff[mm/h] 计算累积径流')

        if 'actual_infil[mm/h]' in LGAR_output.columns:
            LGAR_output['cumulative_infilt'] = (LGAR_output['actual_infil[mm/h]'] * time_step).cumsum()
            print('  已从 actual_infil[mm/h] 计算累积下渗')

    # 打印 LGAR 统计信息
    if 'cumulative_runoff' in LGAR_output.columns and 'cumulative_infilt' in LGAR_output.columns:
        print(f'\nLGAR 模型统计:')
        print(f'  累积径流量: {LGAR_output["cumulative_runoff"].iloc[-1]:.6f} mm')
        print(f'  累积下渗量: {LGAR_output["cumulative_infilt"].iloc[-1]:.6f} mm')

except FileNotFoundError:
    print('警告: 未找到 LGAR 模型数据文件')
    print('路径: D:\\program\\LGAR-Py\\outputs\\output_synth_2.pkl')
    lgar_has_data = False
    LGAR_output = None
except Exception as e:
    print(f'读取 LGAR 数据时出错: {e}')
    lgar_has_data = False
    LGAR_output = None

# ========== 读取 HYDRUS 数据 ==========
print('\n' + '=' * 60)
print('读取 HYDRUS 数据...')
print('=' * 60)

HYDRUS_output = pd.read_fwf(r'D:\Program Files\Hydrus1D_4.17.0140\3\2-降雨\T_Level.txt',
                            widths=[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                                    13])

# ========== 修正：确保 Time 列为数值类型 ==========
HYDRUS_output['Time'] = pd.to_numeric(HYDRUS_output['Time'], errors='coerce')

HYDRUS_datetime_vec = []
for i in range(len(HYDRUS_output['Time'])):
    current_dt = output.index[0] + dt.timedelta(minutes=float(HYDRUS_output['Time'].iloc[i]))
    HYDRUS_datetime_vec.append(current_dt)
HYDRUS_output['HYDRUS_datetime_vec'] = HYDRUS_datetime_vec
HYDRUS_output = HYDRUS_output.set_index('HYDRUS_datetime_vec')
HYDRUS_output = HYDRUS_output.resample('300S').asfreq().interpolate()

print(f'HYDRUS 数据读取完成，共 {len(HYDRUS_output)} 行')

# ========== 新增：计算 HYDRUS 的速率数据 ==========
# 从累积量计算每个时间步的通量速率
HYDRUS_output['infil_rate[mm/h]'] = HYDRUS_output['sum(Infil'].diff().fillna(0) / time_step
HYDRUS_output['runoff_rate[mm/h]'] = HYDRUS_output['sum(RunOff'].diff().fillna(0) / time_step

# LGAR 从时间 0 开始，而 HYDRUS 不从时间 0 开始
output_backup = output.copy()  # 备份完整数据
output.drop(index=output.index[0], axis=0, inplace=True)
HYDRUS_output.drop(index=HYDRUS_output.index[len(HYDRUS_output) - 1], axis=0, inplace=True)

# 对 LGAR 数据做相同处理
if lgar_has_data and LGAR_output is not None:
    LGAR_output = LGAR_output.iloc[1:]  # 删除第一行以对齐

# ========== 绘制累积强迫数据曲线 ==========
plt.figure(figsize=(8, 6))
plt.plot(output['cumulative_precip'], color='black', linewidth=2, label='precip')
plt.plot(output['cumulative_evptrs_pot'], color='black', linestyle='dotted', linewidth=2, label='PET')
plt.ylabel('cumulative precipitation \n or PET (mm)', fontsize=12)
plt.xlabel('date', fontsize=12)
plt.legend()
plt.title('cumulative forcing data mass curves')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_folder, 'cumulative_forcing_data_mass_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# ========== 准备 MLGAR 输出数据 ==========
# ← 修正：使用累积量
MLGAR_output = pd.DataFrame({
    'runoff': runoff,  # 累积径流量 (mm)
    'cummInfil': cummInfil  # 累积下渗量 (mm)
}, index=row_names)
MLGAR_output = MLGAR_output[1:]
LMGAR_output = MLGAR_output

# ========== 对比 MLGAR、LGAR 和 HYDRUS 结果 ==========
times_to_plot = np.arange(time_step, 12, time_step)
min_plt_len = min(len(output), len(HYDRUS_output))

# 如果有 LGAR 数据，调整最小长度
if lgar_has_data and LGAR_output is not None:
    min_plt_len = min(min_plt_len, len(LGAR_output))

# ========== 图1: 累积通量对比（分栏显示，避免重叠）==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 子图1: 累积下渗
line1 = ax1.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(Infil'].iloc[:min_plt_len],
                 linewidth=3, color='#1f77b4', label='HYDRUS', linestyle='-', alpha=0.9)
line2 = ax1.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['cummInfil'],
                 linewidth=2.5, color='#ff7f0e', label='MLGAR', linestyle='--',
                 marker='o', markersize=5, markevery=12, alpha=0.85)
if lgar_has_data and LGAR_output is not None:
    line3 = ax1.plot(times_to_plot[:min_plt_len], LGAR_output['cumulative_infilt'].iloc[:min_plt_len],
                     linewidth=2.5, color='#2ca02c', label='LGAR', linestyle=':',
                     marker='s', markersize=5, markevery=8, alpha=0.85)

ax1.set_ylabel('累积下渗 (mm)', fontsize=14, fontweight='bold')
ax1.set_xlabel('时间 (h)', fontsize=13)
ax1.legend(fontsize=12, loc='upper left', framealpha=0.95, edgecolor='black')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_title('(a) 累积下渗对比', fontsize=15, fontweight='bold', pad=10)
ax1.tick_params(labelsize=11)

# 子图2: 累积径流
line4 = ax2.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(RunOff'].iloc[:min_plt_len],
                 linewidth=3, color='#1f77b4', label='HYDRUS', linestyle='-', alpha=0.9)
line5 = ax2.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['runoff'],
                 linewidth=2.5, color='#ff7f0e', label='MLGAR', linestyle='--',
                 marker='o', markersize=5, markevery=12, alpha=0.85)
if lgar_has_data and LGAR_output is not None:
    line6 = ax2.plot(times_to_plot[:min_plt_len], LGAR_output['cumulative_runoff'].iloc[:min_plt_len],
                     linewidth=2.5, color='#2ca02c', label='LGAR', linestyle=':',
                     marker='s', markersize=5, markevery=8, alpha=0.85)

ax2.set_ylabel('累积径流 (mm)', fontsize=14, fontweight='bold')
ax2.set_xlabel('时间 (h)', fontsize=13)
ax2.legend(fontsize=12, loc='upper left', framealpha=0.95, edgecolor='black')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_title('(b) 累积径流对比', fontsize=15, fontweight='bold', pad=10)
ax2.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(result_folder, 'cumulative_fluxes_3models.png'), dpi=300, bbox_inches='tight')
plt.close()

# ========== 图2: 组合对比图（优化版，减少重叠）==========
fig = plt.figure(figsize=(14, 8))
ax = plt.gca()

# 使用更明显区分的颜色和线型
# HYDRUS - 实线，较粗
ax.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(Infil'].iloc[:min_plt_len],
        linewidth=3.5, color='#0066CC', label='HYDRUS - 累积下渗',
        linestyle='-', alpha=0.9, zorder=3)
ax.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(RunOff'].iloc[:min_plt_len],
        linewidth=3.5, color='#CC0000', label='HYDRUS - 累积径流',
        linestyle='-', alpha=0.9, zorder=3)

# MLGAR - 虚线 + 圆形标记
ax.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['cummInfil'],
        linewidth=2.8, color='#FF8800', label='MLGAR - 累积下渗',
        linestyle='--', marker='o', markersize=6, markevery=15,
        markerfacecolor='white', markeredgewidth=2, alpha=0.95, zorder=4)
ax.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['runoff'],
        linewidth=2.8, color='#FF0088', label='MLGAR - 累积径流',
        linestyle='--', marker='o', markersize=6, markevery=15,
        markerfacecolor='white', markeredgewidth=2, alpha=0.95, zorder=4)

# LGAR - 点线 + 方形标记
if lgar_has_data and LGAR_output is not None:
    ax.plot(times_to_plot[:min_plt_len], LGAR_output['cumulative_infilt'].iloc[:min_plt_len],
            linewidth=2.8, color='#00AA44', label='LGAR - 累积下渗',
            linestyle=':', marker='s', markersize=6, markevery=10,
            markerfacecolor='white', markeredgewidth=2, alpha=0.95, zorder=5)
    ax.plot(times_to_plot[:min_plt_len], LGAR_output['cumulative_runoff'].iloc[:min_plt_len],
            linewidth=2.8, color='#AA00AA', label='LGAR - 累积径流',
            linestyle=':', marker='s', markersize=6, markevery=10,
            markerfacecolor='white', markeredgewidth=2, alpha=0.95, zorder=5)

ax.legend(fontsize=11, loc='upper left', framealpha=0.95,
          edgecolor='black', fancybox=True, shadow=True, ncol=2)
ax.set_ylabel('累计通量 (mm)', fontsize=14, fontweight='bold', labelpad=8)
ax.set_xlabel('时间 (h)', fontsize=14, fontweight='bold', labelpad=8)
ax.set_title('MLGAR vs LGAR vs HYDRUS 累积通量对比', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig(os.path.join(result_folder, 'cumulative_fluxes_combined.png'), dpi=300, bbox_inches='tight')
plt.close()

# ========== Solution 4 Fixed Version A: Combined Plot with Local Zoom ==========
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots(figsize=(16, 10))

# Main plot
ax.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(Infil'].iloc[:min_plt_len],
        linewidth=3.5, color='#0066CC', label='HYDRUS - Cumulative Infiltration', linestyle='-', alpha=0.8)
ax.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['cummInfil'],
        linewidth=3, color='#FF8800', label='MLGAR - Cumulative Infiltration', linestyle='--',
        marker='o', markersize=6, markevery=15, alpha=0.9)
if lgar_has_data and LGAR_output is not None:
    ax.plot(times_to_plot[:min_plt_len], LGAR_output['cumulative_infilt'].iloc[:min_plt_len],
            linewidth=3, color='#00CC44', label='LGAR - Cumulative Infiltration', linestyle='-.',
            marker='^', markersize=6, markevery=10, alpha=0.9)

# Create zoomed inset axes
axins = inset_axes(ax, width="40%", height="35%", loc='lower right',
                   bbox_to_anchor=(0, 0.05, 0.95, 0.95), bbox_transform=ax.transAxes)

# Zoom region (select time period with notable differences, e.g., 5-7 hours)
zoom_start = np.where(times_to_plot >= 5)[0][0] if len(np.where(times_to_plot >= 5)[0]) > 0 else 0
zoom_end = np.where(times_to_plot <= 7)[0][-1] if len(np.where(times_to_plot <= 7)[0]) > 0 else min_plt_len

axins.plot(times_to_plot[zoom_start:zoom_end],
           HYDRUS_output['sum(Infil'].iloc[zoom_start:zoom_end],
           linewidth=3, color='#0066CC', linestyle='-', alpha=0.8)
axins.plot(times_to_plot[zoom_start:zoom_end],
           MLGAR_output['cummInfil'].iloc[zoom_start:zoom_end],
           linewidth=2.5, color='#FF8800', linestyle='--',
           marker='o', markersize=5, markevery=3, alpha=0.9)
if lgar_has_data and LGAR_output is not None:
    axins.plot(times_to_plot[zoom_start:zoom_end],
               LGAR_output['cumulative_infilt'].iloc[zoom_start:zoom_end],
               linewidth=2.5, color='#00CC44', linestyle='-.',
               marker='^', markersize=5, markevery=2, alpha=0.9)

# Mark the zoomed region
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
axins.grid(True, alpha=0.3)
axins.set_title('Zoomed View', fontsize=10)

ax.set_ylabel('Cumulative Infiltration (mm)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time (h)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_title('Cumulative Infiltration Comparison (with Zoom)', fontsize=16, fontweight='bold')

# Fix: Remove tight_layout and bbox_inches='tight'
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
plt.savefig(os.path.join(result_folder, 'infiltration_with_zoom.png'), dpi=300)
plt.close()
print('Saved figure with zoom: infiltration_with_zoom.png')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots(figsize=(16, 10))

# Main plot - Cumulative runoff comparison
# Note: Please adjust column names based on actual data
ax.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(RunOff'].iloc[:min_plt_len],
        linewidth=3.5, color='#CC0066', label='HYDRUS - Cumulative Runoff', linestyle='-', alpha=0.8)
ax.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['runoff'],
        linewidth=3, color='#9900FF', label='MLGAR - Cumulative Runoff', linestyle='--',
        marker='s', markersize=6, markevery=15, alpha=0.9)
if lgar_has_data and LGAR_output is not None:
    ax.plot(times_to_plot[:min_plt_len], LGAR_output['cumulative_runoff'].iloc[:min_plt_len],
            linewidth=3, color='#FF6600', label='LGAR - Cumulative Runoff', linestyle='-.',
            marker='D', markersize=6, markevery=10, alpha=0.9)

# Create zoomed inset axes
axins = inset_axes(ax, width="40%", height="35%", loc='center left',
                   bbox_to_anchor=(0.05, 0.1, 0.95, 0.95), bbox_transform=ax.transAxes)

# Zoom region (select time period with notable differences based on runoff characteristics, here 5-7.5 hours)
# Tip: Adjust time range based on actual data
zoom_start = np.where(times_to_plot >= 5)[0][0] if len(np.where(times_to_plot >= 3)[0]) > 0 else 0
zoom_end = np.where(times_to_plot <= 7.5)[0][-1] if len(np.where(times_to_plot <= 5)[0]) > 0 else min_plt_len

axins.plot(times_to_plot[zoom_start:zoom_end],
           HYDRUS_output['sum(RunOff'].iloc[zoom_start:zoom_end],
           linewidth=3, color='#CC0066', linestyle='-', alpha=0.8)
axins.plot(times_to_plot[zoom_start:zoom_end],
           MLGAR_output['runoff'].iloc[zoom_start:zoom_end],
           linewidth=2.5, color='#9900FF', linestyle='--',
           marker='s', markersize=5, markevery=3, alpha=0.9)
if lgar_has_data and LGAR_output is not None:
    axins.plot(times_to_plot[zoom_start:zoom_end],
               LGAR_output['cumulative_runoff'].iloc[zoom_start:zoom_end],
               linewidth=2.5, color='#FF6600', linestyle='-.',
               marker='D', markersize=5, markevery=2, alpha=0.9)

# Mark the zoomed region
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
axins.grid(True, alpha=0.3)
axins.set_title('Zoomed View', fontsize=10)

ax.set_ylabel('Cumulative Runoff (mm)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time (h)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_title('Cumulative Runoff Comparison (with Zoom)', fontsize=16, fontweight='bold')

# Manually adjust margins
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
plt.savefig(os.path.join(result_folder, 'runoff_with_zoom.png'), dpi=300)
plt.close()
print('Saved figure with zoom: runoff_with_zoom.png')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 创建包含两个子图的图像（2行1列）
fig, axes = plt.subplots(2, 1, figsize=(16, 18))

# ==================== 子图 (a): 累积渗流 ====================
ax1 = axes[0]

# 主图 - 累积渗流
ax1.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(Infil'].iloc[:min_plt_len],
        linewidth=3.5, color='#0066CC', label='HYDRUS - Cumulative Infiltration',
        linestyle='-', alpha=0.8)
ax1.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['cummInfil'],
        linewidth=3, color='#FF8800', label='LMGAR - Cumulative Infiltration',
        linestyle='--', marker='o', markersize=6, markevery=15, alpha=0.9)
if lgar_has_data and LGAR_output is not None:
    ax1.plot(times_to_plot[:min_plt_len], LGAR_output['cumulative_infilt'].iloc[:min_plt_len],
            linewidth=3, color='#00CC44', label='LGAR - Cumulative Infiltration',
            linestyle='-.', marker='^', markersize=6, markevery=10, alpha=0.9)

# 创建缩放插图
axins1 = inset_axes(ax1, width="40%", height="35%", loc='lower right',
                   bbox_to_anchor=(0, 0.05, 0.95, 0.95), bbox_transform=ax1.transAxes)

# 缩放区域 (5-7小时)
zoom_start1 = np.where(times_to_plot >= 5)[0][0] if len(np.where(times_to_plot >= 5)[0]) > 0 else 0
zoom_end1 = np.where(times_to_plot <= 7)[0][-1] if len(np.where(times_to_plot <= 7)[0]) > 0 else min_plt_len

axins1.plot(times_to_plot[zoom_start1:zoom_end1],
           HYDRUS_output['sum(Infil'].iloc[zoom_start1:zoom_end1],
           linewidth=3, color='#0066CC', linestyle='-', alpha=0.8)
axins1.plot(times_to_plot[zoom_start1:zoom_end1],
           MLGAR_output['cummInfil'].iloc[zoom_start1:zoom_end1],
           linewidth=2.5, color='#FF8800', linestyle='--',
           marker='o', markersize=5, markevery=3, alpha=0.9)
if lgar_has_data and LGAR_output is not None:
    axins1.plot(times_to_plot[zoom_start1:zoom_end1],
               LGAR_output['cumulative_infilt'].iloc[zoom_start1:zoom_end1],
               linewidth=2.5, color='#00CC44', linestyle='-.',
               marker='^', markersize=5, markevery=2, alpha=0.9)

# 标记缩放区域
mark_inset(ax1, axins1, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
axins1.grid(True, alpha=0.3)
axins1.set_title('Zoomed View', fontsize=10)

# 设置子图(a)属性
ax1.set_ylabel('Cumulative Infiltration (mm)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (h)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax1.grid(True, alpha=0.3)
ax1.set_title('(a) Cumulative Infiltration Comparison',
              fontsize=16, fontweight='bold', loc='left', pad=12)

# ==================== 子图 (b): 累积径流 ====================
ax2 = axes[1]

# 主图 - 累积径流
ax2.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(RunOff'].iloc[:min_plt_len],
        linewidth=3.5, color='#CC0066', label='HYDRUS - Cumulative Runoff',
        linestyle='-', alpha=0.8)
ax2.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['runoff'],
        linewidth=3, color='#9900FF', label='LMGAR - Cumulative Runoff',
        linestyle='--', marker='s', markersize=6, markevery=15, alpha=0.9)
if lgar_has_data and LGAR_output is not None:
    ax2.plot(times_to_plot[:min_plt_len], LGAR_output['cumulative_runoff'].iloc[:min_plt_len],
            linewidth=3, color='#FF6600', label='LGAR - Cumulative Runoff',
            linestyle='-.', marker='D', markersize=6, markevery=10, alpha=0.9)

# 创建缩放插图
axins2 = inset_axes(ax2, width="40%", height="35%", loc='center left',
                   bbox_to_anchor=(0.05, 0.1, 0.95, 0.95), bbox_transform=ax2.transAxes)

# 缩放区域 (5-7.5小时)
zoom_start2 = np.where(times_to_plot >= 5)[0][0] if len(np.where(times_to_plot >= 3)[0]) > 0 else 0
zoom_end2 = np.where(times_to_plot <= 7.5)[0][-1] if len(np.where(times_to_plot <= 5)[0]) > 0 else min_plt_len

axins2.plot(times_to_plot[zoom_start2:zoom_end2],
           HYDRUS_output['sum(RunOff'].iloc[zoom_start2:zoom_end2],
           linewidth=3, color='#CC0066', linestyle='-', alpha=0.8)
axins2.plot(times_to_plot[zoom_start2:zoom_end2],
           MLGAR_output['runoff'].iloc[zoom_start2:zoom_end2],
           linewidth=2.5, color='#9900FF', linestyle='--',
           marker='s', markersize=5, markevery=3, alpha=0.9)
if lgar_has_data and LGAR_output is not None:
    axins2.plot(times_to_plot[zoom_start2:zoom_end2],
               LGAR_output['cumulative_runoff'].iloc[zoom_start2:zoom_end2],
               linewidth=2.5, color='#FF6600', linestyle='-.',
               marker='D', markersize=5, markevery=2, alpha=0.9)

# 标记缩放区域
mark_inset(ax2, axins2, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
axins2.grid(True, alpha=0.3)
axins2.set_title('Zoomed View', fontsize=10)

# 设置子图(b)属性
ax2.set_ylabel('Cumulative Runoff (mm)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (h)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax2.grid(True, alpha=0.3)
ax2.set_title('(b) Cumulative Runoff Comparison',
              fontsize=16, fontweight='bold', loc='left', pad=12)

# 调整子图间距 - 防止标题被截断
plt.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.05, hspace=0.28)

# 保存图片
plt.savefig(os.path.join(result_folder, 'infiltration_runoff_combined_with_zoom.png'), dpi=300)
plt.close()
print('Saved combined figure with zoom: infiltration_runoff_combined_with_zoom.png')


# ========== 计算误差统计指标 ==========
print('\n' + '=' * 60)
print('计算误差统计指标...')
print('=' * 60)

# ------------------- 方法1: 基于累积量（仅供参考，会虚高）-------------------
hydrus_infil_cumul = HYDRUS_output['sum(Infil'].iloc[:min_plt_len].values
hydrus_runoff_cumul = HYDRUS_output['sum(RunOff'].iloc[:min_plt_len].values
mlgar_infil_cumul = MLGAR_output['cummInfil'].iloc[:min_plt_len].values
mlgar_runoff_cumul = MLGAR_output['runoff'].iloc[:min_plt_len].values

mlgar_vs_hydrus_cumulative = {
    'Infiltration': {
        'KGE': calculate_kge(hydrus_infil_cumul, mlgar_infil_cumul),
        'NSE': calculate_nse(hydrus_infil_cumul, mlgar_infil_cumul),
        'PBIAS': calculate_pbias(hydrus_infil_cumul, mlgar_infil_cumul),
        'RMSE': calculate_rmse(hydrus_infil_cumul, mlgar_infil_cumul),
        'MLGAR_cumulation': mlgar_infil_cumul[-1],
        'HYDRUS_cumulation': hydrus_infil_cumul[-1],
        'Difference': mlgar_infil_cumul[-1] - hydrus_infil_cumul[-1]
    },
    'Runoff': {
        'KGE': calculate_kge(hydrus_runoff_cumul, mlgar_runoff_cumul),
        'NSE': calculate_nse(hydrus_runoff_cumul, mlgar_runoff_cumul),
        'PBIAS': calculate_pbias(hydrus_runoff_cumul, mlgar_runoff_cumul),
        'RMSE': calculate_rmse(hydrus_runoff_cumul, mlgar_runoff_cumul),
        'MLGAR_cumulation': mlgar_runoff_cumul[-1],
        'HYDRUS_cumulation': hydrus_runoff_cumul[-1],
        'Difference': mlgar_runoff_cumul[-1] - hydrus_runoff_cumul[-1]
    }
}

# ------------------- 方法2: 基于瞬时速率（推荐，真实反映模型性能）-------------------
# HYDRUS 速率数据（从累积量差分计算）
hydrus_infil_rate = HYDRUS_output['infil_rate[mm/h]'].iloc[:min_plt_len].values
hydrus_runoff_rate = HYDRUS_output['runoff_rate[mm/h]'].iloc[:min_plt_len].values

# MLGAR 速率数据（已经计算好的）
mlgar_infil_rate = output['actual_infil[mm/h]'].iloc[:min_plt_len].values
mlgar_runoff_rate = output['runoff[mm/h]'].iloc[:min_plt_len].values

mlgar_vs_hydrus_rate = {
    'Infiltration': {
        'KGE': calculate_kge(hydrus_infil_rate, mlgar_infil_rate),
        'NSE': calculate_nse(hydrus_infil_rate, mlgar_infil_rate),
        'PBIAS': calculate_pbias(hydrus_infil_rate, mlgar_infil_rate),
        'RMSE': calculate_rmse(hydrus_infil_rate, mlgar_infil_rate),
    },
    'Runoff': {
        'KGE': calculate_kge(hydrus_runoff_rate, mlgar_runoff_rate),
        'NSE': calculate_nse(hydrus_runoff_rate, mlgar_runoff_rate),
        'PBIAS': calculate_pbias(hydrus_runoff_rate, mlgar_runoff_rate),
        'RMSE': calculate_rmse(hydrus_runoff_rate, mlgar_runoff_rate),
    }
}

# 如果有LGAR数据，计算 LGAR vs HYDRUS 的误差指标
if lgar_has_data and LGAR_output is not None:
    # 累积量
    lgar_infil_cumul = LGAR_output['cumulative_infilt'].iloc[:min_plt_len].values
    lgar_runoff_cumul = LGAR_output['cumulative_runoff'].iloc[:min_plt_len].values

    lgar_vs_hydrus_cumulative = {
        'Infiltration': {
            'KGE': calculate_kge(hydrus_infil_cumul, lgar_infil_cumul),
            'NSE': calculate_nse(hydrus_infil_cumul, lgar_infil_cumul),
            'PBIAS': calculate_pbias(hydrus_infil_cumul, lgar_infil_cumul),
            'RMSE': calculate_rmse(hydrus_infil_cumul, lgar_infil_cumul),
            'LGAR_cumulation': lgar_infil_cumul[-1],
            'HYDRUS_cumulation': hydrus_infil_cumul[-1],
            'Difference': lgar_infil_cumul[-1] - hydrus_infil_cumul[-1]
        },
        'Runoff': {
            'KGE': calculate_kge(hydrus_runoff_cumul, lgar_runoff_cumul),
            'NSE': calculate_nse(hydrus_runoff_cumul, lgar_runoff_cumul),
            'PBIAS': calculate_pbias(hydrus_runoff_cumul, lgar_runoff_cumul),
            'RMSE': calculate_rmse(hydrus_runoff_cumul, lgar_runoff_cumul),
            'LGAR_cumulation': lgar_runoff_cumul[-1],
            'HYDRUS_cumulation': hydrus_runoff_cumul[-1],
            'Difference': lgar_runoff_cumul[-1] - hydrus_runoff_cumul[-1]
        }
    }

    # 速率（如果LGAR有速率数据）
    if 'actual_infil[mm/h]' in LGAR_output.columns and 'runoff[mm/h]' in LGAR_output.columns:
        lgar_infil_rate = LGAR_output['actual_infil[mm/h]'].iloc[:min_plt_len].values
        lgar_runoff_rate = LGAR_output['runoff[mm/h]'].iloc[:min_plt_len].values

        lgar_vs_hydrus_rate = {
            'Infiltration': {
                'KGE': calculate_kge(hydrus_infil_rate, lgar_infil_rate),
                'NSE': calculate_nse(hydrus_infil_rate, lgar_infil_rate),
                'PBIAS': calculate_pbias(hydrus_infil_rate, lgar_infil_rate),
                'RMSE': calculate_rmse(hydrus_infil_rate, lgar_infil_rate),
            },
            'Runoff': {
                'KGE': calculate_kge(hydrus_runoff_rate, lgar_runoff_rate),
                'NSE': calculate_nse(hydrus_runoff_rate, lgar_runoff_rate),
                'PBIAS': calculate_pbias(hydrus_runoff_rate, lgar_runoff_rate),
                'RMSE': calculate_rmse(hydrus_runoff_rate, lgar_runoff_rate),
            }
        }
    else:
        lgar_vs_hydrus_rate = None
else:
    lgar_vs_hydrus_cumulative = None
    lgar_vs_hydrus_rate = None

# ========== 打印误差统计结果 ==========
print('\n' + '⚠️ ' * 40)
print('【仅供参考】基于累积量的指标（会因单调性虚高，不推荐用于评估模型性能）')
print('=' * 80)
print('误差统计指标 - 累积量 (MLGAR vs HYDRUS):')
print('=' * 80)
print(f"{'指标':<20} {'下渗 (Infiltration)':<25} {'径流 (Runoff)':<25}")
print('-' * 80)
print(
    f"{'KGE(-)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['KGE']:>24.3f} {mlgar_vs_hydrus_cumulative['Runoff']['KGE']:>24.3f}")
print(
    f"{'NSE(-)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['NSE']:>24.3f} {mlgar_vs_hydrus_cumulative['Runoff']['NSE']:>24.3f}")
print(
    f"{'PBIAS(%)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['PBIAS']:>24.3f} {mlgar_vs_hydrus_cumulative['Runoff']['PBIAS']:>24.3f}")
print(
    f"{'RMSE(mm)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['RMSE']:>24.3f} {mlgar_vs_hydrus_cumulative['Runoff']['RMSE']:>24.3f}")
print('=' * 80)

print('\n' + '✅ ' * 40)
print('【推荐使用】基于瞬时速率的指标（真实反映模型性能）')
print('=' * 80)
print('误差统计指标 - 瞬时速率 (MLGAR vs HYDRUS):')
print('=' * 80)
print(f"{'指标':<20} {'下渗速率':<25} {'径流速率':<25}")
print('-' * 80)
print(
    f"{'KGE(-)':<20} {mlgar_vs_hydrus_rate['Infiltration']['KGE']:>24.3f} {mlgar_vs_hydrus_rate['Runoff']['KGE']:>24.3f}")
print(
    f"{'NSE(-)':<20} {mlgar_vs_hydrus_rate['Infiltration']['NSE']:>24.3f} {mlgar_vs_hydrus_rate['Runoff']['NSE']:>24.3f}")
print(
    f"{'PBIAS(%)':<20} {mlgar_vs_hydrus_rate['Infiltration']['PBIAS']:>24.3f} {mlgar_vs_hydrus_rate['Runoff']['PBIAS']:>24.3f}")
print(
    f"{'RMSE(mm/h)':<20} {mlgar_vs_hydrus_rate['Infiltration']['RMSE']:>24.3f} {mlgar_vs_hydrus_rate['Runoff']['RMSE']:>24.3f}")
print('=' * 80)

# 如果有 LGAR 数据，打印 LGAR 的指标
if lgar_vs_hydrus_cumulative is not None:
    print('\n' + '⚠️ ' * 40)
    print('【仅供参考】基于累积量的指标 (LGAR vs HYDRUS):')
    print('=' * 80)
    print(f"{'指标':<20} {'下渗 (Infiltration)':<25} {'径流 (Runoff)':<25}")
    print('-' * 80)
    print(
        f"{'KGE(-)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['KGE']:>24.3f} {lgar_vs_hydrus_cumulative['Runoff']['KGE']:>24.3f}")
    print(
        f"{'NSE(-)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['NSE']:>24.3f} {lgar_vs_hydrus_cumulative['Runoff']['NSE']:>24.3f}")
    print(
        f"{'PBIAS(%)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['PBIAS']:>24.3f} {lgar_vs_hydrus_cumulative['Runoff']['PBIAS']:>24.3f}")
    print(
        f"{'RMSE(mm)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['RMSE']:>24.3f} {lgar_vs_hydrus_cumulative['Runoff']['RMSE']:>24.3f}")
    print('=' * 80)

    if lgar_vs_hydrus_rate is not None:
        print('\n' + '✅ ' * 40)
        print('【推荐使用】基于瞬时速率的指标 (LGAR vs HYDRUS):')
        print('=' * 80)
        print(f"{'指标':<20} {'下渗速率':<25} {'径流速率':<25}")
        print('-' * 80)
        print(
            f"{'KGE(-)':<20} {lgar_vs_hydrus_rate['Infiltration']['KGE']:>24.3f} {lgar_vs_hydrus_rate['Runoff']['KGE']:>24.3f}")
        print(
            f"{'NSE(-)':<20} {lgar_vs_hydrus_rate['Infiltration']['NSE']:>24.3f} {lgar_vs_hydrus_rate['Runoff']['NSE']:>24.3f}")
        print(
            f"{'PBIAS(%)':<20} {lgar_vs_hydrus_rate['Infiltration']['PBIAS']:>24.3f} {lgar_vs_hydrus_rate['Runoff']['PBIAS']:>24.3f}")
        print(
            f"{'RMSE(mm/h)':<20} {lgar_vs_hydrus_rate['Infiltration']['RMSE']:>24.3f} {lgar_vs_hydrus_rate['Runoff']['RMSE']:>24.3f}")
        print('=' * 80)

print('\n' + '📊 ' * 40)
print('说明：')
print('  1. 累积量指标因单调递增特性会虚高（KGE、NSE接近1），不能真实反映模型性能')
print('  2. 瞬时速率指标能真实反映模型对过程的模拟能力，建议论文中使用该指标')
print('  3. 两种指标均已保存到文件中供对比参考')
print('=' * 80)

# ========== 创建误差统计表格并保存 ==========
# 创建 DataFrame 格式的对比表（包含两种方法）
error_metrics_table_cumul = pd.DataFrame({
    'Calculation_Method': ['Cumulative'] * 2 + (['Cumulative'] * 2 if lgar_vs_hydrus_cumulative else []),
    'Test_Number': ['MLGAR vs HYDRUS'] * 2 + (['LGAR vs HYDRUS'] * 2 if lgar_vs_hydrus_cumulative else []),
    'Metric': ['Infiltration', 'Runoff'] * (2 if lgar_vs_hydrus_cumulative else 1),
    'KGE(-)': [
                  mlgar_vs_hydrus_cumulative['Infiltration']['KGE'],
                  mlgar_vs_hydrus_cumulative['Runoff']['KGE']
              ] + ([
                       lgar_vs_hydrus_cumulative['Infiltration']['KGE'],
                       lgar_vs_hydrus_cumulative['Runoff']['KGE']
                   ] if lgar_vs_hydrus_cumulative else []),
    'NSE(-)': [
                  mlgar_vs_hydrus_cumulative['Infiltration']['NSE'],
                  mlgar_vs_hydrus_cumulative['Runoff']['NSE']
              ] + ([
                       lgar_vs_hydrus_cumulative['Infiltration']['NSE'],
                       lgar_vs_hydrus_cumulative['Runoff']['NSE']
                   ] if lgar_vs_hydrus_cumulative else []),
    'PBIAS(%)': [
                    mlgar_vs_hydrus_cumulative['Infiltration']['PBIAS'],
                    mlgar_vs_hydrus_cumulative['Runoff']['PBIAS']
                ] + ([
                         lgar_vs_hydrus_cumulative['Infiltration']['PBIAS'],
                         lgar_vs_hydrus_cumulative['Runoff']['PBIAS']
                     ] if lgar_vs_hydrus_cumulative else []),
    'RMSE(mm)': [
                    mlgar_vs_hydrus_cumulative['Infiltration']['RMSE'],
                    mlgar_vs_hydrus_cumulative['Runoff']['RMSE']
                ] + ([
                         lgar_vs_hydrus_cumulative['Infiltration']['RMSE'],
                         lgar_vs_hydrus_cumulative['Runoff']['RMSE']
                     ] if lgar_vs_hydrus_cumulative else []),
    'Model_cumulation(mm)': [
                                mlgar_vs_hydrus_cumulative['Infiltration']['MLGAR_cumulation'],
                                mlgar_vs_hydrus_cumulative['Runoff']['MLGAR_cumulation']
                            ] + ([
                                     lgar_vs_hydrus_cumulative['Infiltration']['LGAR_cumulation'],
                                     lgar_vs_hydrus_cumulative['Runoff']['LGAR_cumulation']
                                 ] if lgar_vs_hydrus_cumulative else []),
    'HYDRUS_cumulation(mm)': [
                                 mlgar_vs_hydrus_cumulative['Infiltration']['HYDRUS_cumulation'],
                                 mlgar_vs_hydrus_cumulative['Runoff']['HYDRUS_cumulation']
                             ] + ([
                                      lgar_vs_hydrus_cumulative['Infiltration']['HYDRUS_cumulation'],
                                      lgar_vs_hydrus_cumulative['Runoff']['HYDRUS_cumulation']
                                  ] if lgar_vs_hydrus_cumulative else []),
    'Difference(mm)': [
                          mlgar_vs_hydrus_cumulative['Infiltration']['Difference'],
                          mlgar_vs_hydrus_cumulative['Runoff']['Difference']
                      ] + ([
                               lgar_vs_hydrus_cumulative['Infiltration']['Difference'],
                               lgar_vs_hydrus_cumulative['Runoff']['Difference']
                           ] if lgar_vs_hydrus_cumulative else [])
})

error_metrics_table_rate = pd.DataFrame({
    'Calculation_Method': ['Instantaneous_Rate'] * 2 + (
        ['Instantaneous_Rate'] * 2 if lgar_vs_hydrus_rate else []),
    'Test_Number': ['MLGAR vs HYDRUS'] * 2 + (['LGAR vs HYDRUS'] * 2 if lgar_vs_hydrus_rate else []),
    'Metric': ['Infiltration', 'Runoff'] * (2 if lgar_vs_hydrus_rate else 1),
    'KGE(-)': [
                  mlgar_vs_hydrus_rate['Infiltration']['KGE'],
                  mlgar_vs_hydrus_rate['Runoff']['KGE']
              ] + ([
                       lgar_vs_hydrus_rate['Infiltration']['KGE'],
                       lgar_vs_hydrus_rate['Runoff']['KGE']
                   ] if lgar_vs_hydrus_rate else []),
    'NSE(-)': [
                  mlgar_vs_hydrus_rate['Infiltration']['NSE'],
                  mlgar_vs_hydrus_rate['Runoff']['NSE']
              ] + ([
                       lgar_vs_hydrus_rate['Infiltration']['NSE'],
                       lgar_vs_hydrus_rate['Runoff']['NSE']
                   ] if lgar_vs_hydrus_rate else []),
    'PBIAS(%)': [
                    mlgar_vs_hydrus_rate['Infiltration']['PBIAS'],
                    mlgar_vs_hydrus_rate['Runoff']['PBIAS']
                ] + ([
                         lgar_vs_hydrus_rate['Infiltration']['PBIAS'],
                         lgar_vs_hydrus_rate['Runoff']['PBIAS']
                     ] if lgar_vs_hydrus_rate else []),
    'RMSE(mm/h)': [
                      mlgar_vs_hydrus_rate['Infiltration']['RMSE'],
                      mlgar_vs_hydrus_rate['Runoff']['RMSE']
                  ] + ([
                           lgar_vs_hydrus_rate['Infiltration']['RMSE'],
                           lgar_vs_hydrus_rate['Runoff']['RMSE']
                       ] if lgar_vs_hydrus_rate else [])
})

# ============ 对数值列进行四舍五入（在创建表格之后）============
# 对累积量表格的数值列四舍五入到4位小数
numeric_cols_cumul = ['KGE(-)', 'NSE(-)', 'PBIAS(%)', 'RMSE(mm)',
                      'Model_cumulation(mm)', 'HYDRUS_cumulation(mm)', 'Difference(mm)']
for col in numeric_cols_cumul:
    if col in error_metrics_table_cumul.columns:
        error_metrics_table_cumul[col] = error_metrics_table_cumul[col].round(4)

# 对速率表格的数值列四舍五入到4位小数
numeric_cols_rate = ['KGE(-)', 'NSE(-)', 'PBIAS(%)', 'RMSE(mm/h)']
for col in numeric_cols_rate:
    if col in error_metrics_table_rate.columns:
        error_metrics_table_rate[col] = error_metrics_table_rate[col].round(4)

# 合并两个表格
error_metrics_table = pd.concat([error_metrics_table_rate, error_metrics_table_cumul], ignore_index=True)

# ============ 保存CSV（使用float_format控制精度）============
error_metrics_file = os.path.join(result_folder, 'error_metrics_table.csv')
error_metrics_table.to_csv(error_metrics_file, index=False, encoding='utf-8', float_format='%.4f')
print(f'\n误差统计表已保存到: {error_metrics_file}')

# ============ 保存Excel（使用xlsxwriter控制格式）============
error_metrics_excel = os.path.join(result_folder, 'error_metrics_table.xlsx')

with pd.ExcelWriter(error_metrics_excel, engine='xlsxwriter') as writer:
    # 写入三个sheet
    error_metrics_table_rate.to_excel(writer, sheet_name='Rate_Based_RECOMMENDED', index=False)
    error_metrics_table_cumul.to_excel(writer, sheet_name='Cumulative_Based_Reference', index=False)
    error_metrics_table.to_excel(writer, sheet_name='All_Metrics', index=False)

    # 获取workbook对象
    workbook = writer.book

    # 定义数字格式（保留4位小数）
    number_format = workbook.add_format({'num_format': '0.0000', 'align': 'right'})

    # 定义表头格式
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'vcenter',
        'align': 'center',
        'bg_color': '#D7E4BD',
        'border': 1
    })

    # 对每个sheet应用格式
    sheet_configs = [
        ('Rate_Based_RECOMMENDED', error_metrics_table_rate),
        ('Cumulative_Based_Reference', error_metrics_table_cumul),
        ('All_Metrics', error_metrics_table)
    ]

    for sheet_name, df in sheet_configs:
        worksheet = writer.sheets[sheet_name]

        # 设置列宽和格式
        worksheet.set_column('A:A', 22)  # Calculation_Method
        worksheet.set_column('B:B', 20)  # Test_Number
        worksheet.set_column('C:C', 15)  # Metric

        # 为数值列设置格式和宽度
        # 获取列名
        for col_num, col_name in enumerate(df.columns):
            if col_num >= 3:  # 从第4列开始是数值列
                worksheet.set_column(col_num, col_num, 20, number_format)

        # 设置表头格式（重写第一行）
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_format)

        # 冻结首行
        worksheet.freeze_panes(1, 0)

print(f'误差统计表已保存到: {error_metrics_excel}')
print('  - Sheet1: Rate_Based_RECOMMENDED (推荐使用的速率指标)')
print('  - Sheet2: Cumulative_Based_Reference (仅供参考的累积量指标)')
print('  - Sheet3: All_Metrics (所有指标汇总)')
print('  - 所有数值已格式化为4位小数，并添加了表头样式')

# ========== 保存对比数据 ==========
# 径流对比（包含累积量和速率）
runoff_comparison = pd.DataFrame({
    'Time(h)': times_to_plot[:min_plt_len],
    'HYDRUS_cumul': HYDRUS_output['sum(RunOff'].iloc[:min_plt_len].values,
    'MLGAR_cumul': MLGAR_output['runoff'].iloc[:min_plt_len].values,
    'HYDRUS_rate': HYDRUS_output['runoff_rate[mm/h]'].iloc[:min_plt_len].values,
    'MLGAR_rate': mlgar_runoff_rate
})

# 下渗对比（包含累积量和速率）
infil_comparison = pd.DataFrame({
    'Time(h)': times_to_plot[:min_plt_len],
    'HYDRUS_cumul': HYDRUS_output['sum(Infil'].iloc[:min_plt_len].values,
    'MLGAR_cumul': MLGAR_output['cummInfil'].iloc[:min_plt_len].values,
    'HYDRUS_rate': HYDRUS_output['infil_rate[mm/h]'].iloc[:min_plt_len].values,
    'MLGAR_rate': mlgar_infil_rate
})

# 如果有 LGAR 数据，添加到对比数据框
if lgar_has_data and LGAR_output is not None:
    runoff_comparison['LGAR_cumul'] = LGAR_output['cumulative_runoff'].iloc[:min_plt_len].values
    infil_comparison['LGAR_cumul'] = LGAR_output['cumulative_infilt'].iloc[:min_plt_len].values

    if 'runoff[mm/h]' in LGAR_output.columns:
        runoff_comparison['LGAR_rate'] = LGAR_output['runoff[mm/h]'].iloc[:min_plt_len].values
    if 'actual_infil[mm/h]' in LGAR_output.columns:
        infil_comparison['LGAR_rate'] = LGAR_output['actual_infil[mm/h]'].iloc[:min_plt_len].values

runoff_comparison.to_csv(os.path.join(result_folder, 'runoff_comparison.csv'), index=False, encoding='utf-8')
infil_comparison.to_csv(os.path.join(result_folder, 'infil_comparison.csv'), index=False, encoding='utf-8')

print(f'\n对比数据已保存到:')
print(f'  {result_folder}/runoff_comparison.csv')
print(f'  {result_folder}/infil_comparison.csv')

# ========== 保存详细的误差统计报告 ==========
error_report_file = os.path.join(result_folder, 'error_analysis_report.txt')
with open(error_report_file, 'w', encoding='utf-8') as f:
    f.write('=' * 100 + '\n')
    f.write('误差分析报告 - Error Analysis Report\n')
    f.write('=' * 100 + '\n\n')

    f.write('【重要说明 Important Notes】\n')
    f.write('-' * 100 + '\n')
    f.write('本报告包含两种计算方法的误差指标：\n')
    f.write('1. 基于瞬时速率 (Instantaneous Rate) - 推荐用于论文发表\n')
    f.write('   - 真实反映模型对水文过程的模拟能力\n')
    f.write('   - 能捕捉峰值和时序变化特征\n\n')
    f.write('2. 基于累积量 (Cumulative) - 仅供参考\n')
    f.write('   - 因单调递增特性，指标会虚高（KGE、NSE接近1）\n')
    f.write('   - 不能真实反映模型性能，不建议用于模型评估\n')
    f.write('=' * 100 + '\n\n')

    f.write('Table 1: Error Metrics Based on Instantaneous Rates (RECOMMENDED)\n')
    f.write('=' * 100 + '\n')
    f.write('MLGAR vs HYDRUS-1D:\n')
    f.write('-' * 100 + '\n')
    f.write(f"{'指标 Metric':<20} {'下渗速率 Infiltration Rate':<35} {'径流速率 Runoff Rate':<35}\n")
    f.write('-' * 100 + '\n')
    f.write(
        f"{'KGE(-)':<20} {mlgar_vs_hydrus_rate['Infiltration']['KGE']:>34.4f} {mlgar_vs_hydrus_rate['Runoff']['KGE']:>34.4f}\n")
    f.write(
        f"{'NSE(-)':<20} {mlgar_vs_hydrus_rate['Infiltration']['NSE']:>34.4f} {mlgar_vs_hydrus_rate['Runoff']['NSE']:>34.4f}\n")
    f.write(
        f"{'PBIAS(%)':<20} {mlgar_vs_hydrus_rate['Infiltration']['PBIAS']:>34.4f} {mlgar_vs_hydrus_rate['Runoff']['PBIAS']:>34.4f}\n")
    f.write(
        f"{'RMSE(mm/h)':<20} {mlgar_vs_hydrus_rate['Infiltration']['RMSE']:>34.4f} {mlgar_vs_hydrus_rate['Runoff']['RMSE']:>34.4f}\n")
    f.write('-' * 100 + '\n\n')

    if lgar_vs_hydrus_rate is not None:
        f.write('LGAR vs HYDRUS-1D:\n')
        f.write('-' * 100 + '\n')
        f.write(f"{'指标 Metric':<20} {'下渗速率 Infiltration Rate':<35} {'径流速率 Runoff Rate':<35}\n")
        f.write('-' * 100 + '\n')
        f.write(
            f"{'KGE(-)':<20} {lgar_vs_hydrus_rate['Infiltration']['KGE']:>34.4f} {lgar_vs_hydrus_rate['Runoff']['KGE']:>34.4f}\n")
        f.write(
            f"{'NSE(-)':<20} {lgar_vs_hydrus_rate['Infiltration']['NSE']:>34.4f} {lgar_vs_hydrus_rate['Runoff']['NSE']:>34.4f}\n")
        f.write(
            f"{'PBIAS(%)':<20} {lgar_vs_hydrus_rate['Infiltration']['PBIAS']:>34.4f} {lgar_vs_hydrus_rate['Runoff']['PBIAS']:>34.4f}\n")
        f.write(
            f"{'RMSE(mm/h)':<20} {lgar_vs_hydrus_rate['Infiltration']['RMSE']:>34.4f} {lgar_vs_hydrus_rate['Runoff']['RMSE']:>34.4f}\n")
        f.write('-' * 100 + '\n\n')

    f.write('\n' + '=' * 100 + '\n')
    f.write('Table 2: Error Metrics Based on Cumulative Values (FOR REFERENCE ONLY)\n')
    f.write('=' * 100 + '\n')
    f.write('MLGAR vs HYDRUS-1D:\n')
    f.write('-' * 100 + '\n')
    f.write(f"{'指标 Metric':<20} {'累积下渗 Cumul. Infil.':<35} {'累积径流 Cumul. Runoff':<35}\n")
    f.write('-' * 100 + '\n')
    f.write(
        f"{'KGE(-)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['KGE']:>34.4f} {mlgar_vs_hydrus_cumulative['Runoff']['KGE']:>34.4f}\n")
    f.write(
        f"{'NSE(-)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['NSE']:>34.4f} {mlgar_vs_hydrus_cumulative['Runoff']['NSE']:>34.4f}\n")
    f.write(
        f"{'PBIAS(%)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['PBIAS']:>34.4f} {mlgar_vs_hydrus_cumulative['Runoff']['PBIAS']:>34.4f}\n")
    f.write(
        f"{'RMSE(mm)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['RMSE']:>34.4f} {mlgar_vs_hydrus_cumulative['Runoff']['RMSE']:>34.4f}\n")
    f.write(
        f"{'MLGAR累积量(mm)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['MLGAR_cumulation']:>34.4f} {mlgar_vs_hydrus_cumulative['Runoff']['MLGAR_cumulation']:>34.4f}\n")
    f.write(
        f"{'HYDRUS累积量(mm)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['HYDRUS_cumulation']:>34.4f} {mlgar_vs_hydrus_cumulative['Runoff']['HYDRUS_cumulation']:>34.4f}\n")
    f.write(
        f"{'差异Difference(mm)':<20} {mlgar_vs_hydrus_cumulative['Infiltration']['Difference']:>34.4f} {mlgar_vs_hydrus_cumulative['Runoff']['Difference']:>34.4f}\n")
    f.write('-' * 100 + '\n\n')

    if lgar_vs_hydrus_cumulative is not None:
        f.write('LGAR vs HYDRUS-1D:\n')
        f.write('-' * 100 + '\n')
        f.write(f"{'指标 Metric':<20} {'累积下渗 Cumul. Infil.':<35} {'累积径流 Cumul. Runoff':<35}\n")
        f.write('-' * 100 + '\n')
        f.write(
            f"{'KGE(-)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['KGE']:>34.4f} {lgar_vs_hydrus_cumulative['Runoff']['KGE']:>34.4f}\n")
        f.write(
            f"{'NSE(-)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['NSE']:>34.4f} {lgar_vs_hydrus_cumulative['Runoff']['NSE']:>34.4f}\n")
        f.write(
            f"{'PBIAS(%)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['PBIAS']:>34.4f} {lgar_vs_hydrus_cumulative['Runoff']['PBIAS']:>34.4f}\n")
        f.write(
            f"{'RMSE(mm)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['RMSE']:>34.4f} {lgar_vs_hydrus_cumulative['Runoff']['RMSE']:>34.4f}\n")
        f.write(
            f"{'LGAR累积量(mm)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['LGAR_cumulation']:>34.4f} {lgar_vs_hydrus_cumulative['Runoff']['LGAR_cumulation']:>34.4f}\n")
        f.write(
            f"{'HYDRUS累积量(mm)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['HYDRUS_cumulation']:>34.4f} {lgar_vs_hydrus_cumulative['Runoff']['HYDRUS_cumulation']:>34.4f}\n")
        f.write(
            f"{'差异Difference(mm)':<20} {lgar_vs_hydrus_cumulative['Infiltration']['Difference']:>34.4f} {lgar_vs_hydrus_cumulative['Runoff']['Difference']:>34.4f}\n")
        f.write('-' * 100 + '\n\n')

    f.write('\n指标说明 (Metric Descriptions):\n')
    f.write('-' * 100 + '\n')
    f.write('KGE  : Kling-Gupta Efficiency, 范围[-∞, 1], 1为完美匹配, >0.5为良好\n')
    f.write('NSE  : Nash-Sutcliffe Efficiency, 范围[-∞, 1], 1为完美匹配, >0.5为满意\n')
    f.write('PBIAS: Percent Bias, 范围(-∞, +∞), 0为无偏差\n')
    f.write('       负值表示模型高估observed, 正值表示模型低估observed\n')
    f.write('       ±10%以内为very good, ±15%以内为good, ±25%以内为satisfactory\n')
    f.write('RMSE : Root Mean Square Error, 值越小越好\n')
    f.write('       速率指标单位: mm/h, 累积量指标单位: mm\n')
    f.write('=' * 100 + '\n\n')

    f.write('【建议 Recommendations】\n')
    f.write('-' * 100 + '\n')
    f.write('1. 论文中应使用 Table 1 (基于瞬时速率) 的指标\n')
    f.write('2. Table 2 (基于累积量) 仅作为水量平衡检验的参考\n')
    f.write('3. 如需展示模型的时序拟合能力，建议绘制速率过程对比图\n')
    f.write('4. 累积量图可用于展示长期水量平衡，但不应作为主要评估依据\n')
    f.write('=' * 100 + '\n')

print(f'误差分析报告已保存到: {error_report_file}')

# ========== 计算累积下渗的百分比差异 ==========
percent_diff_cumulative_infiltration = []
for i in range(min(len(output) - 2, len(HYDRUS_output))):
    hydrus_val = HYDRUS_output['sum(Infil'].iloc[i]
    lgar_val = output['cumulative_infilt'].iloc[i]
    if hydrus_val != 0:
        percent_diff = abs((hydrus_val - lgar_val) / hydrus_val * 100)
        percent_diff_cumulative_infiltration.append(percent_diff)
    else:
        percent_diff_cumulative_infiltration.append(np.nan)

panel_letter = 'A'
plt.figure(figsize=(6, 4))
plt.plot(times_to_plot[58:129], percent_diff_cumulative_infiltration[58:129], linewidth=2)
mean_diff = np.nanmean(percent_diff_cumulative_infiltration[:])
plt.hlines(mean_diff, xmin=0, xmax=12, color='red', linestyle='--', linewidth=2)
print(f'\nMLGAR vs HYDRUS 平均累积下渗百分比差异: {mean_diff:.6f}%')
plt.ylabel('cumulative infiltration \n % difference', fontsize=18)
plt.xlabel('time (h)', fontsize=18)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.text(s=panel_letter, x=9.75, y=mean_diff / 15.8, fontweight='bold', fontsize=18)
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
plt.savefig(os.path.join(result_folder, 'cumulative_infiltration_%_difference.png'), dpi=300, bbox_inches='tight')
plt.close()

# ========== 保存完整的 output DataFrame ==========
output_backup.to_pickle(os.path.join(result_folder, 'output_complete.pkl'))
output_backup.to_csv(os.path.join(result_folder, 'output_complete.csv'), encoding='utf-8')
print(f'完整输出数据已保存到: {result_folder}/output_complete.pkl 和 output_complete.csv')

print('\n' + '=' * 80)
print('所有结果已保存完毕！')
print(f'结果文件夹: {os.path.abspath(result_folder)}')
print('=' * 80)
print('\n生成的文件列表:')
print('  图表文件:')
print('    1. cumulative_fluxes_3models.png      - 三模型累积通量对比（分栏，推荐）')
print('    2. cumulative_fluxes_combined.png     - 三模型累积通量对比（组合，优化版）')
print('\n  数据文件:')
print('    3. runoff_comparison.csv              - 径流对比数据（含累积量和速率）')
print('    4. infil_comparison.csv               - 下渗对比数据（含累积量和速率）')
print('    5. error_metrics_table.csv            - 误差统计表（CSV格式，含两种方法）')
print('    6. error_metrics_table.xlsx           - 误差统计表（Excel格式，分sheet）')
print('       - Sheet1: 推荐使用的速率指标')
print('       - Sheet2: 仅供参考的累积量指标')
print('       - Sheet3: 所有指标汇总')
print('    7. error_analysis_report.txt          - 详细误差分析报告（含使用建议）')
print('=' * 80)
print('\n' + '⚠️ ' * 40)
print('重要提示：')
print('  ✅ 论文中请使用基于"瞬时速率"的误差指标（KGE、NSE等）')
print('  ❌ 基于"累积量"的指标会虚高，仅供水量平衡检验参考')
print('  📊 详细说明请查看 error_analysis_report.txt')
print('=' * 80)
