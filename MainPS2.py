'''
Created on Apr 11, 2017

@author: dduque
'''

from IO_handler import Datahandler
from L_ShapeMethod import L_Shape
import sys
print(sys.path)
'''
Parameters
'''
#n = 10  # number of customers
#scen = 10  # number of scenarios
#B = 7.5 * n  # budget
rho = 10  # 1.42  # shortfall penalty

dro_radii = [2]  #[a * (10**b) for b in [-2, -1, 0, 1, 2] for a in [1]]
Problems = ['dro_cw']  # ['a','b','c','d','dd']
NumFacilities = [5, 10, 20]
worst_case_dist_r = {}
for n in [20]:
    B = 1E5 * n  # 7.5 * n  # budget
    for scen in [20, 30, 50]:
        for facilities in NumFacilities:
            for r in dro_radii:
                for p in Problems:
                    '''
                    Build a data set
                    Instantiate L_Shape
                    Solve problem a
                    '''
                    # Finite case lognormal
                    # data = Datahandler(n, facilities, scen, B, rho)
                    # solver = L_Shape(data)
                    # probName = f"{p}_finite_{facilities}_{n}_{scen}_{r}"
                    # solver.run(type='dro_cw',
                    #            probName=probName,
                    #            dro_r=r,
                    #            discrete_wasserstein=True,
                    #            add_scenario=False,
                    #            simulate=True,
                    #            output={},
                    #            distance_type='wasserstein')
                    
                    # Box BLP
                    # data = Datahandler(n, facilities, scen, B, rho)
                    # solver = L_Shape(data)
                    # probName = f"{p}_box_blp_{facilities}_{n}_{scen}_{r}"
                    # solver.run(type='dro_cw',
                    #            probName=probName,
                    #            dro_r=r,
                    #            discrete_wasserstein=False,
                    #            add_scenario=False,
                    #            simulate=False,
                    #            output={},
                    #            distance_type='wasserstein')
                    
                    # Box mip lognormal
                    # alg_output = {}
                    # data = Datahandler(n, facilities, scen, B, rho)
                    # solver = L_Shape(data)
                    # probName = f"{p}_box_mip_{facilities}_{n}_{scen}_{r}"
                    # solver.run(type='dro_cw',
                    #            probName=probName,
                    #            dro_r=r,
                    #            discrete_wasserstein=False,
                    #            add_scenario=False,
                    #            simulate=False,
                    #            output=alg_output,
                    #            distance_type='wasserstein')
                    # worst_case_dist_r[r] = alg_output['worst_case_dist']
                    
                    # Optimal transport example lognormal
                    alg_output = {}
                    data = Datahandler(n, facilities, scen, B, rho)
                    solver = L_Shape(data)
                    probName = f"{p}_transport_{facilities}_{n}_{scen}_{r}"
                    solver.run(type='dro_cw',
                               probName=probName,
                               dro_r=r,
                               discrete_wasserstein=False,
                               add_scenario=False,
                               simulate=False,
                               output=alg_output,
                               distance_type='optimal_transport')
                    worst_case_dist_r[r] = alg_output['worst_case_dist']
                    
                    # Finite case lognormal
                    # data = Datahandler(n, facilities, scen, B, rho, 'uniform')
                    # solver = L_Shape(data)
                    # probName = f"{p}_finite_uni_{facilities}_{n}_{scen}_{r}"
                    # solver.run(type='dro_cw',
                    #            probName=probName,
                    #            dro_r=r,
                    #            discrete_wasserstein=True,
                    #            add_scenario=False,
                    #            simulate=True,
                    #            output={},
                    #            distance_type='wasserstein')
                    
                    # Box mip uniform
                    # alg_output = {}
                    # data = Datahandler(n, facilities, scen, B, rho, 'uniform')
                    # solver = L_Shape(data)
                    # probName = f"{p}_box_mip_uni_{facilities}_{n}_{scen}_{r}"
                    # solver.run(type='dro_cw',
                    #            probName=probName,
                    #            dro_r=r,
                    #            discrete_wasserstein=False,
                    #            add_scenario=False,
                    #            simulate=True,
                    #            output=alg_output,
                    #            distance_type='wasserstein')
                    # worst_case_dist_r[r] = alg_output['worst_case_dist']
                    
                    # Optimal transport example uniform
                    # alg_output = {}
                    # data = Datahandler(n, facilities, scen, B, rho, 'uniform')
                    # solver = L_Shape(data)
                    # probName = f"{p}_transport_uni_{facilities}_{n}_{scen}_{r}"
                    # solver.run(type='dro_cw',
                    #            probName=probName,
                    #            dro_r=r,
                    #            discrete_wasserstein=False,
                    #            add_scenario=False,
                    #            simulate=True,
                    #            output=alg_output,
                    #            distance_type='optimal_transport')
                    # worst_case_dist_r[r] = alg_output['worst_case_dist']

#plot changes
import matplotlib.pyplot as plt
import colorsys
N = 100
HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
RGB_tuples = ['red', 'lightblue', 'black', 'orange', 'blue', 'teal', 'grey']
plot_markers = ['o', 'v', '*', '+', 'x', '_']
# list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
# plt.scatter(x=data.d_out_of_sample[0, :1000],
#             y=data.d_out_of_sample[1, :1000],
#             color='red',
#             alpha=0.2,
#             label=f'OutOfSample')

RGB_tuples.reverse()
# dro_radii.reverse()
f, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
#fig = plt.figure(figsize=(6, 6), dpi=200)
#ax2 = fig.add_subplot(111, projection='3d')
for ix_r, r in enumerate(dro_radii):
    print(f'Radius: {r} in color {RGB_tuples[ix_r]}')
    wcd = worst_case_dist_r[r]
    xi_1 = [wcd[i, j]['new_sup'][0] for (i, j) in wcd]
    xi_2 = [wcd[i, j]['new_sup'][1] for (i, j) in wcd]
    label_str = r"$\epsilon=$" + f'{r}' if r > 0.001 else r"$\epsilon=$" + f'{0}'
    ax.scatter(x=xi_1, y=xi_2, marker=plot_markers[ix_r], s=90, color=RGB_tuples[ix_r], label=label_str)
    
    probs = [wcd[i, j]['prob'] for (i, j) in wcd]
    #ax2.bar3d(xi_1, xi_2, [0] * len(xi_1), 0.2, 0.2, probs, shade=True)

#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlabel('Demand at site 1')
ax.set_ylabel('Demand at site 2')
plt.legend()
plt.show()
