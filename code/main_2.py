import datetime
from quadrat import build_initial_quadrat_records
from InfilHydrol import calc_infil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------------------------------------------------#
precipFilename = r"D:\program\LGAR-Py\Figure\2\MLGAR-2\DATA\forcing_data_resampled_synth_2.csv"
SomeTxtFile = r"D:\program\LGAR-Py\Figure\2\MLGAR-2\sample_MLGAR_output.txt"
# ---------------------------------------------------------------------------------------------------------------------#
MaxNumQuadrats = 1  # 表示最大允许的象限数量，当前设置为 1。象限可以代表不同的模拟区域
MaxNumLayers = 3   # 表示观测层的最大数量，当前设置为 10。每个象限可能包含多个观测层，用于表示不同的土壤层或其他参数
NumQuadrats = 1      # 表示实际使用的象限数量，当前设置为 1。这个值可以根据需要调整
deltaT = 1.0         # 时间步长，初始设置为 1.0，表示 1 小时的时间步长
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
init_time = datetime.datetime.now()  # 获取当前时间

# 创建样方并初始化变量
QP = build_initial_quadrat_records(precip_data, parameters, initial_psi, num_layers, boundary_depths)

tstart = 0
tend = 1.0/12
deltaT = 1.0/12

# 打开文件以写入数据
with open(SomeTxtFile, 'w') as file:
    # file.write('Time F1 Z1 WC1 F2 Z2 WC2 F3 Z3 WC3 F4 Z4 WC4 WCO RWC CummInfil\n')
    file.write('Time F1 Z1 WC1 F2 Z2 WC2 F3 Z3 WC3 F4 Z4 WC4 runoff CummInfil\n')

runoff = [0] * length_of_simulation
cummInfil = [0] * length_of_simulation

for x in range(length_of_simulation):
    if x > 72:
        print("---")
    QP.precipRate = QP.precips[x]  # 假设有降水速率数据
    calc_infil(QP, tstart, tend, deltaT, SomeTxtFile)
    runoff[x] = QP.runoff
    cummInfil[x] = QP.cummInfil
    tstart += 1.0/12
    tend += 1.0/12

    with open(SomeTxtFile, 'a') as f:
        # 写入 inter_time，格式化为整数
        f.write(f"{int(x)} ")

        # 遍历 WF 数组并写入 FAmt, Z, WC
        for i in range(0, 4):
            f.write(f"{QP.WF[i].FAmt:.3f} ")
            f.write(f"{QP.WF[i].z:.3f} ")
            f.write(f"{QP.WF[i].WC:.3f} ")

        # 写入 WCO 和 RWC
        # f.write(f"{QP.WCO:.6f} ")
        # f.write(f"{QP.RWC:.6f} ")
        f.write(f"{QP.runoff:.3f} ")
        f.write(f"{QP.cummInfil:.3f}\n")

end_time = datetime.datetime.now()  # 获取结束时间


# 验证
HYDRUS_output = pd.read_fwf(r'D:\Program Files\Hydrus1D_4.17.0140\3\2-降雨\T_Level.txt', widths=[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])
output = pd.read_pickle(r'D:\program\LGAR-Py\outputs\output_synth_2.pkl')

row_names = output.index.tolist()
MLGAR_output = pd.DataFrame({'runoff': runoff, 'cummInfil': cummInfil}, index=row_names)
MLGAR_output = MLGAR_output[1:]

time_step = 300/3600
actual_ET = []
for i in range(0, len(output)):
    actual_ET.append((1/time_step)*output['actual_ET_per_step(mm)'][i])
output['actual_ET[mm/h]'] = actual_ET

# 创建一个空白图像
# plt.figure()
plt.plot(output['P(mm/h)'])
plt.plot(output['runoff[mm/h]'])
plt.plot(output['actual_infil[mm/h]'])
plt.plot(output['actual_ET[mm/h]'])
plt.legend(labels=['Precip (mm/h)', 'runoff (mm/h)', 'actual infiltration (mm/h)', 'actual ET (mm/h)'])
plt.savefig(r"D:\program\LGAR-Py\Figure\Precip_runoff_infiltration_ET.png")
plt.close()

print('cumulative precip')
print(sum(output['P(mm/h)'])*5/60)
print(' ')
print('cumulative runoff')
print(sum(output['runoff[mm/h]'])*5/60)
print(' ')
#runoff_efficiency
print('runoff efficiency')
print(sum(output['runoff[mm/h]'])/sum(output['P(mm/h)']))

plt.plot(output['bottom_flux[mm/h]'])
np.mean(output['bottom_flux[mm/h]'])
np.min(output['bottom_flux[mm/h]'])
plt.legend(labels=['bottom_flux'])
plt.savefig(r"D:\program\LGAR-Py\Figure\bottom_flux.png")
plt.close()

plt.plot(output['water_in_soil[mm]'])
plt.xlabel('date')
plt.ylabel('storage (mm)')
plt.legend(labels=['water_in_soil'])
plt.savefig(r"D:\program\LGAR-Py\Figure\water_in_soil.png")
plt.close()

plt.plot(output['mass_bal_error(mm)'])
plt.xlabel('date')
plt.ylabel('storage (mm)')
plt.legend(labels=['mass_bal_error'])
plt.savefig(r"D:\program\LGAR-Py\Figure\mass_bal_error.png")
plt.close()

# cumulative precip (mm)
print("cumulative precip (mm)")
print(sum(output['P(mm/h)'])*time_step)

# runoff (mm)
print("runoff (mm)")
print(sum(output['runoff[mm/h]'])*time_step)

# actual_infil (mm)
print("actual_infil (mm)")
print(sum(output['actual_infil[mm/h]'])*time_step)

# checking that cumulative infiltration + cumulative runoff is the same as cumulative precip
print("cumulative infil + runoff (mm)")
print(sum(output['actual_infil[mm/h]'])*time_step + sum(output['runoff[mm/h]'])*time_step)

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

for i in range(0, len(output)):
    accumulated_precip = accumulated_precip + output['P(mm/h)'][i] * time_step
    cumulative_precip = np.append(cumulative_precip, accumulated_precip)

    accumulated_runoff = accumulated_runoff + output['runoff[mm/h]'][i] * time_step
    cumulative_runoff = np.append(cumulative_runoff, accumulated_runoff)

    accumulated_infilt = accumulated_infilt + output['actual_infil[mm/h]'][i] * time_step
    cumulative_infilt = np.append(cumulative_infilt, accumulated_infilt)

    accumulated_botflx = accumulated_botflx + output['bottom_flux[mm/h]'][i] * time_step
    cumulative_botflx = np.append(cumulative_botflx, accumulated_botflx)

    accumulated_evptrs = accumulated_evptrs + output['actual_ET_per_step(mm)'][i]
    cumulative_evptrs = np.append(cumulative_evptrs, accumulated_evptrs)

    accumulated_evptrs_pot = accumulated_evptrs_pot + output['PET(mm/h)'][i] * time_step
    cumulative_evptrs_pot = np.append(cumulative_evptrs_pot, accumulated_evptrs_pot)

output['cumulative_precip'] = cumulative_precip
output['cumulative_runoff'] = cumulative_runoff
output['cumulative_infilt'] = cumulative_infilt
output['cumulative_botflx'] = cumulative_botflx
output['cumulative_evptrs'] = cumulative_evptrs
output['cumulative_evptrs_pot'] = cumulative_evptrs_pot

HYDRUS_datetime_vec = []
for i in range(0, len(HYDRUS_output['Time'])):
    current_dt = output.index[0] + dt.timedelta(minutes=HYDRUS_output['Time'][i])
    HYDRUS_datetime_vec.append(current_dt)
HYDRUS_output['HYDRUS_datetime_vec'] = HYDRUS_datetime_vec
HYDRUS_output = HYDRUS_output.set_index('HYDRUS_datetime_vec')
HYDRUS_output = HYDRUS_output.resample('300S').interpolate()

# LGAR 从时间 0 开始，而 HYDRUS 不从时间 0 开始。此单元格及下一个单元格将它们格式化，以便可以一起绘制
output.drop(index=output.index[0], axis=0, inplace=True)
HYDRUS_output.drop(index=HYDRUS_output.index[len(HYDRUS_output)-1], axis=0, inplace=True)

plt.plot(output['cumulative_precip'], color='black')
plt.plot(output['cumulative_evptrs_pot'], color='black', linestyle='dotted')
leg = plt.legend(['precip', 'PET'])
plt.ylabel('cumulative precipitation \n or PET (mm)')
plt.xlabel('date')
plt.legend(labels=['cumulative forcing data mass curves'])
plt.savefig(r"D:\program\LGAR-Py\Figure\cumulative_forcing_data_mass_curves.png")
plt.close()

# 更多的格式化用于绘制 LGAR 和 HYDRUS 结果
times_to_plot = np.arange(time_step, 12, time_step)

min_plt_len = min(len(output), len(HYDRUS_output))
# plt.plot(output['cumulative_infilt'][0:min_plt_len])
plt.plot(times_to_plot, MLGAR_output['cummInfil'])
plt.plot(times_to_plot, HYDRUS_output['sum(Infil'], ls='dashed'[0:min_plt_len])
# plt.plot(output['cumulative_runoff'], ls='dotted'[0:min_plt_len])
plt.plot(times_to_plot, MLGAR_output['runoff'])
plt.plot(times_to_plot, HYDRUS_output['sum(RunOff'], ls='dashed'[0:min_plt_len])
# HYDRUS_AET = HYDRUS_output[')    sum(Evap'] + HYDRUS_output[')   sum(vRoot']
# plt.plot(output['cumulative_evptrs'], ls='dashed'[0:min_plt_len])
# plt.plot(HYDRUS_AET, ls='dashed'[0:min_plt_len])

plt.legend(labels=['LMGAR cumulative infiltration', 'HYDRUS cumulative infiltration', 'LMGAR cumulative runoff', 'HYDRUS cumulative runoff'])
# plt.legend(labels=['发明方法累计渗流', 'HYDRUS累计渗流', '发明方法累计径流', 'HYDRUS累计径流'])
plt.ylabel('累计通量(mm)', fontsize=12, labelpad=5)
plt.xlabel('时间(h)', fontsize=12, labelpad=5)
# 在右下方添加文本 "(a)"
plt.text(0.98, 0.02, '(b)', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', verticalalignment='bottom')
plt.savefig(r"D:\program\LGAR-Py\Figure\cumulative_fluxes.png")
runoff_output = pd.DataFrame({'obs': HYDRUS_output['sum(RunOff'][0:min_plt_len], 'sim': MLGAR_output['runoff']}, index=row_names)
cummInfil_output = pd.DataFrame({'obs': HYDRUS_output['sum(Infil'][0:min_plt_len], 'sim': MLGAR_output['cummInfil']}, index=row_names)
runoff_output = runoff_output[1:]
cummInfil_output = cummInfil_output[1:]
runoff_output.to_csv(r'D:\program\LGAR-Py\Figure\2\runoff.csv', index=False, encoding='utf-8')
cummInfil_output.to_csv(r'D:\program\LGAR-Py\Figure\2\cummInfil.csv', index=False, encoding='utf-8')
plt.show()
plt.close()

# 创建一个向量，用于显示 LGAR 的累积入渗与 HYDRUS 的累积入渗的接近程度
percent_diff_cumulative_infiltration = []
for i in range(0, min(len(output)-2, len(HYDRUS_output))):
    percent_diff_cumulative_infiltration.append(abs((HYDRUS_output['sum(Infil'][i] - output['cumulative_infilt'][i])/HYDRUS_output['sum(Infil'][i]*100))

panel_letter = 'A'
plt.figure(figsize=(6, 4))
plt.plot(times_to_plot[58:129], percent_diff_cumulative_infiltration[58:129])
plt.hlines(np.mean(percent_diff_cumulative_infiltration[0:]), xmin=0, xmax=144, color='red')
print(np.mean(percent_diff_cumulative_infiltration[58:129]))
plt.ylabel('cumulative infiltration \n % difference', fontsize=18)
plt.xlabel('time (h)', fontsize=18)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.text(s=panel_letter, x=9.75, y=np.mean(percent_diff_cumulative_infiltration[58:129])/15.8, fontweight='bold', fontsize=18)
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
plt.savefig(r"D:\program\LGAR-Py\Figure\cumulative_infiltration_%_difference.png")
plt.close()



