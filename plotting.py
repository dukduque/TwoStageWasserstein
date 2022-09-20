import os
import pickle
from OutputAnalysis.SimulationAnalysis import plot_sim_results

m = 10
n = 30
N = 10
dro_type = 'transport_uni'
ptf = './output/'
experiment_files = os.listdir(ptf)
print(experiment_files)
sim_results = []
dro_radii = [a * (10**b) for b in [-2, -1, 0] for a in range(1, 10)]
dro_radii.sort()
for dro_r in dro_radii:
    try:
        instance_name = f"dro_cw_{dro_type}_{m}_{n}_{N}_{dro_r}"
        new_sim = pickle.load(open('%s%s_OOS.pickle' % (ptf, instance_name), 'rb'))
        sim_results.append(new_sim)
    except:
        print(dro_r, ' not fount')
instance_name = f"dro_cw_{dro_type}_{m}_{n}_{N}_{0}"
sp_sim = pickle.load(open('%s%s_OOS.pickle' % (ptf, instance_name), 'rb'))

for plot_type in ['means']:
    instance_name = f"dro_cw_{dro_type}_{m}_{n}_{N}_{plot_type}"
    plot_path = '%s%s.pdf' % (ptf, instance_name)
    plot_sim_results(sp_sim,
                     sim_results,
                     plot_path,
                     N,
                     y_lims=(90, 160),
                     plot_type=plot_type,
                     excel_file=False,
                     show=True)

# Lognormal y_lims=(20, 140)