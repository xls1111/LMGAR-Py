import datetime
from quadrat import build_initial_quadrat_records
from InfilHydrol import calc_infil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 创建 Result 文件夹 ==========
result_folder = r'Result\3'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    print(f'已创建文件夹: {result_folder}')

# ---------------------------------------------------------------------------------------------------------------------#
precipFilename = r"D:\program\LGAR-Py\Figure\3\MLGAR-3\DATA\forcing_data_resampled_synth_3.csv"
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
theta_r_layer_0 = 0.1
theta_s_layer_0 = 0.39
K_s_layer_0 = 13.1
alpha_layer_0 = 0.0059
n_layer_0 = 1.48
m_layer_0 = 1-1/n_layer_0
max_depth_layer_0 = 100
param_set_0 = [theta_r_layer_0, theta_s_layer_0, K_s_layer_0, alpha_layer_0, n_layer_0, m_layer_0, max_depth_layer_0]
# 第二层: clay loam
theta_r_layer_1 = 0.1
theta_s_layer_1 = 0.38
K_s_layer_1 = 1.2
alpha_layer_1 = 0.0027
n_layer_1 = 1.23
m_layer_1 = 1-1/n_layer_1
max_depth_layer_1 = 300
param_set_1 = [theta_r_layer_1, theta_s_layer_1, K_s_layer_1, alpha_layer_1, n_layer_1, m_layer_1, max_depth_layer_1]
# 第三层: silty clay loam
theta_r_layer_2 = 0.067
theta_s_layer_2 = 0.45
K_s_layer_2 = 4.5
alpha_layer_2 = 0.002
n_layer_2 = 1.41
m_layer_2 = 1-1/n_layer_2
max_depth_layer_2 = 300
param_set_2 = [theta_r_layer_2, theta_s_layer_2, K_s_layer_2, alpha_layer_2, n_layer_2, m_layer_2, max_depth_layer_2]

parameters = np.vstack([param_set_0, param_set_1, param_set_2])

if len(parameters) > num_layers:
    while len(parameters) > num_layers:
        parameters = np.delete(parameters, len(parameters)-1, 0)

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

max_layer = len(parameters)-1
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
tend = 1.0/12
deltaT = 1.0/12

# 打开文件以写入数据
with open(SomeTxtFile, 'w') as file:
    file.write('Time F1 Z1 WC1 F2 Z2 WC2 F3 Z3 WC3 F4 Z4 WC4 runoff CummInfil\n')

# ← 修正：runoff 和 cummInfil 都是累积量
runoff = [0] * length_of_simulation        # 累积径流量 (mm)
cummInfil = [0] * length_of_simulation     # 累积下渗量 (mm)

# 主计算循环
print('开始计算...')
for x in range(length_of_simulation):
    QP.precipRate = QP.precips[x]
    calc_infil(QP, tstart, tend, deltaT, SomeTxtFile)
    runoff[x] = QP.runoff          # ← 累积径流量 (mm)
    cummInfil[x] = QP.cummInfil    # ← 累积下渗量 (mm)
    tstart += 1.0/12
    tend += 1.0/12

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
time_step = 300/3600  # 5分钟 = 1/12小时

# 创建索引
row_names = forcing_data.index[:length_of_simulation].tolist()

# ← 修正：先存储累积量
output = pd.DataFrame({
    'P(mm/h)': precip_data,
    'PET(mm/h)': PET_data,
    'cumulative_runoff': runoff,      # ← 累积径流量 (mm)
    'cummInfil': cummInfil,           # ← 累积下渗量 (mm)
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
cumulative_precip_mm_1 = sum(output['P(mm/h)']) * 5/60
print(cumulative_precip_mm_1)
print(' ')

# ← 修正：累积径流的正确计算
print('cumulative runoff')
cumulative_runoff_mm_1 = sum(output['runoff[mm/h]']) * 5/60  # 速率求和×时间步
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
    f.write(f'时间步长: {time_step:.6f} 小时 ({time_step*60:.1f} 分钟)\n\n')
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
    f.write(f'  差异:                      {abs(cumulative_runoff_mm - output["cumulative_runoff"].iloc[-1]):.9f} mm\n\n')
    f.write('累积下渗量：\n')
    f.write(f'  从 actual_infil[mm/h] 求和: {actual_infil_sum:.6f} mm\n')
    f.write(f'  从 cummInfil 最后值:       {output["cummInfil"].iloc[-1]:.6f} mm\n')
    f.write(f'  差异:                      {abs(actual_infil_sum - output["cummInfil"].iloc[-1]):.9f} mm\n\n')
    f.write('-' * 60 + '\n')
    f.write('效率指标:\n')
    f.write('-' * 60 + '\n')
    f.write(f'径流系数:                  {runoff_efficiency:.6f}\n')
    f.write(f'下渗系数:                  {actual_infil_sum/cumulative_precip_mm:.6f}\n')
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

# ========== 读取 HYDRUS 数据 ==========
HYDRUS_output = pd.read_fwf(r'D:\Program Files\Hydrus1D_4.17.0140\3\3-降雨\T_Level.txt',
                            widths=[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])

HYDRUS_datetime_vec = []
for i in range(len(HYDRUS_output['Time'])):
    current_dt = output.index[0] + dt.timedelta(minutes=HYDRUS_output['Time'].iloc[i])
    HYDRUS_datetime_vec.append(current_dt)
HYDRUS_output['HYDRUS_datetime_vec'] = HYDRUS_datetime_vec
HYDRUS_output = HYDRUS_output.set_index('HYDRUS_datetime_vec')
HYDRUS_output = HYDRUS_output.resample('300S').asfreq().interpolate()

# LGAR 从时间 0 开始，而 HYDRUS 不从时间 0 开始
output.drop(index=output.index[0], axis=0, inplace=True)
HYDRUS_output.drop(index=HYDRUS_output.index[len(HYDRUS_output)-1], axis=0, inplace=True)

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
    'runoff': runoff,         # 累积径流量 (mm)
    'cummInfil': cummInfil    # 累积下渗量 (mm)
}, index=row_names)
MLGAR_output = MLGAR_output[1:]

# ========== 对比 LGAR 和 HYDRUS 结果 ==========
times_to_plot = np.arange(time_step, 12, time_step)
min_plt_len = min(len(output), len(HYDRUS_output))

plt.figure(figsize=(10, 6))
plt.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['cummInfil'],
         linewidth=2, color='blue', label='MLGAR cumulative infiltration')
plt.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(Infil'].iloc[:min_plt_len],
         ls='dashed', linewidth=2, color='blue', alpha=0.7, label='HYDRUS cumulative infiltration')
plt.plot(times_to_plot[:len(MLGAR_output)], MLGAR_output['runoff'],
         linewidth=2, color='red', label='MLGAR cumulative runoff')
plt.plot(times_to_plot[:min_plt_len], HYDRUS_output['sum(RunOff'].iloc[:min_plt_len],
         ls='dashed', linewidth=2, color='red', alpha=0.7, label='HYDRUS cumulative runoff')

plt.legend()
plt.ylabel('累计通量 (mm)', fontsize=12, labelpad=5)
plt.xlabel('时间 (h)', fontsize=12, labelpad=5)
plt.title('MLGAR 与 HYDRUS 对比')
plt.grid(True, alpha=0.3)
plt.text(0.98, 0.02, '(c)', transform=plt.gca().transAxes, fontsize=12,
         horizontalalignment='right', verticalalignment='bottom')
plt.savefig(os.path.join(result_folder, 'cumulative_fluxes.png'), dpi=300, bbox_inches='tight')
plt.close()

# ========== 保存对比数据 ==========
runoff_output = pd.DataFrame({
    'obs': HYDRUS_output['sum(RunOff'].iloc[:min_plt_len],
    'sim': MLGAR_output['runoff'].iloc[:min_plt_len]
})
cummInfil_output = pd.DataFrame({
    'obs': HYDRUS_output['sum(Infil'].iloc[:min_plt_len],
    'sim': MLGAR_output['cummInfil'].iloc[:min_plt_len]
})

runoff_output.to_csv(os.path.join(result_folder, 'runoff.csv'), index=False, encoding='utf-8')
cummInfil_output.to_csv(os.path.join(result_folder, 'cummInfil.csv'), index=False, encoding='utf-8')

print(f'对比数据已保存到: {result_folder}/runoff.csv 和 cummInfil.csv')

# ========== 计算累积下渗的百分比差异 ==========
percent_diff_cumulative_infiltration = []
for i in range(min(len(output)-2, len(HYDRUS_output))):
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
mean_diff = np.nanmean(percent_diff_cumulative_infiltration[58:129])
plt.hlines(mean_diff, xmin=0, xmax=12, color='red', linestyle='--', linewidth=2)
print(f'\n平均累积下渗百分比差异: {mean_diff:.6f}%')
plt.ylabel('cumulative infiltration \n % difference', fontsize=18)
plt.xlabel('time (h)', fontsize=18)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.text(s=panel_letter, x=9.75, y=mean_diff/15.8, fontweight='bold', fontsize=18)
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
plt.savefig(os.path.join(result_folder, 'cumulative_infiltration_%_difference.png'), dpi=300, bbox_inches='tight')
plt.close()

# ========== 保存完整的 output DataFrame ==========
output.to_pickle(os.path.join(result_folder, 'output_complete.pkl'))
output.to_csv(os.path.join(result_folder, 'output_complete.csv'), encoding='utf-8')
print(f'完整输出数据已保存到: {result_folder}/output_complete.pkl 和 output_complete.csv')

print('\n' + '=' * 60)
print('所有结果已保存完毕！')
print(f'结果文件夹: {os.path.abspath(result_folder)}')
print('=' * 60)
