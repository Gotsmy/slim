#!/home/users/mgotsmy/.conda/envs/2206test3.10/bin/python

import numpy as np
import pandas as pd
import cobra
import snek
import tqdm
from scipy.integrate import solve_ivp
from datetime import datetime
import pickle
from multiprocessing import Pool


def run_pFBA(_in):
    def infeasible_ATPM(t, y, model, tol):
        '''
        Infeasibility event. Usually occures when the glucose uptake flux is smaller than the ATPM requirement.
        '''

        V,X,S,P,G = y  # expand the boundary species
        # feed rate
        r_F_growth = 1.52*11.27/1.126*.1*np.exp(.1*t)/1000 # L/h
        # glucose conc in feed
        G_F = 329.88 # g/L glucose
        # glucse feed rate
        r_F_G = r_F_growth*G_F # gG/h
        r_F_G = r_F_G / snek.elements.molecular_weight('C6H12O6') * 1000 # mmol G/h
        
        q_G = -r_F_G/X
        with model as tmpmodel:
            snek.set_bounds(tmpmodel,'EX_glc__D_e',lb=q_G,ub=q_G)
            if S <= tol:
                snek.set_bounds(tmpmodel,'EX_so4_e',lb=0)
            fluxes = np.array([])
            for reaction, direction in zip(['BIOMASS_Ec_iML1515_core_75p37M'],['max']):
                tmpmodel.objective = reaction
                tmpmodel.objective_direction = direction
                tmp = tmpmodel.slim_optimize()
                if np.isnan(tmp):
                    return 0
                else:
                    return 1
    infeasible_ATPM.terminal = False
    infeasible_ATPM.direction = 0

    def infeasible_V(t, y, model, tol):
        '''
        Infeasibility event. Usually occures when the volume of the reactor is above 1.
        '''

        V,X,S,P,G = y  # expand the boundary species
        return 1-V
    infeasible_V.terminal = True
    infeasible_V.direction = 0

    def dFBA(t, y, model, tol):
        """
        Calculate the time derivative of external species.
        """

        V,X,S,P,G = y  # expand the boundary species
        # feed rate
        r_F_growth = 1.52*11.27/1.126*.1*np.exp(.1*t)/1000 # L/h
        # glucose conc in feed
        G_F = 329.88 # g/L glucose
        # glucse feed rate
        r_F_G = r_F_growth*G_F # gG/h
        r_F_G = r_F_G / snek.elements.molecular_weight('C6H12O6') * 1000 # mmol G/h
        
        dV_dt = r_F_growth
        q_G = -r_F_G/X
        # Calculate the specific exchanges fluxes at the given external concentrations.
        with model as tmpmodel:
            snek.set_bounds(tmpmodel,'EX_glc__D_e',lb=q_G,ub=q_G)
            if S <= tol:
                snek.set_bounds(tmpmodel,'EX_so4_e',lb=0)
            fluxes = np.array([])
            for reaction, direction in zip(['BIOMASS_Ec_iML1515_core_75p37M','EX_so4_e','pDNA_synthesis','EX_glc__D_e'],['max','max','max','min']):
                tmpmodel.objective = reaction
                tmpmodel.objective_direction = direction
                tmp = tmpmodel.slim_optimize()
                # the nan is excepted here but will trigger the terminal event.
                tmp = np.nan_to_num(tmp)
                fluxes = np.append(fluxes,tmp)
                snek.set_bounds(tmpmodel,reaction,tmp,tmp)
            del tmpmodel

        fluxes *= X
        return np.append(dV_dt,fluxes)
    
    S_0, q_pdna_max = _in 
    ecoli = cobra.io.read_sbml_model('../models/iML1515_pDNA.xml')
    ecoli.solver = 'cplex'
    ecoli.tolerance = 1e-8
    snek.set_bounds(ecoli,'pDNA_synthesis',lb=.1/100,ub=q_pdna_max)
    
    ts = np.linspace(0, 60, 10000)
    V_0 = .5    # L batch volume
    X_0 = 1.52  # g
    P_0 = 0     # mmol
    y0 = [V_0,X_0,S_0,P_0,0]
    tol = 1e-6
    sol = solve_ivp(
        fun=dFBA,
        t_span=(ts.min(), ts.max()),
        y0=y0,
        t_eval=ts,
        args = (ecoli,tol),
        events= [infeasible_ATPM,infeasible_V],
        rtol=tol,
        atol=tol,
        method='RK45'
    )
    
    return [S_0, sol]

def do_grid_simulations(q_pdna_max):
    print(f'Grid Simulations for q_pdna_max = {q_pdna_max}')
    t1 = datetime.now()

    # multiprocessing 
    nr_samples = 301
    S_0    = np.linspace(0,20,nr_samples)
    _input = np.vstack([S_0,np.ones(nr_samples)*q_pdna_max]).T
    output = []
    n_cpu = np.min([nr_samples,150])
    with Pool(processes = n_cpu) as p:
        for _ in tqdm.tqdm(p.imap_unordered(run_pFBA,_input),total=len(_input)):
            output.append(_)

    results = {}
    for out in output:
        S_0, sol = out
        results[S_0] = sol
    results = dict(sorted(results.items()))


    t2 = datetime.now()
    print('done')
    print('process time',t2-t1)
    return results

combined_results = {}
q_pdna_max_levels = np.linspace(.1,.5,41)/100
for q_pdna_max in q_pdna_max_levels:
    combined_results[q_pdna_max] = do_grid_simulations(q_pdna_max)
    # break

version = '220829_dFBA_exp_feed_v04'
loc     = f'/mnt/itching_scratch/mgotsmy/220829_slim_paper_v2/results/{version}.pkl'
with open(loc,'wb') as file:
    pickle.dump(combined_results,file)        

print('Scriped finished sucessfully.')
