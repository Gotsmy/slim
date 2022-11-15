#!/home/users/mgotsmy/.conda/envs/2206test3.10/bin/python

import os
os.environ["OMP_NUM_THREADS"] = str(1)
import numpy as np
import pandas as pd
import cobra
import snek
import tqdm
from scipy.integrate import solve_ivp
from datetime import datetime
from pathlib import Path
import pickle
from multiprocessing import Pool


def run_pFBA(_in):
    def michaelis_menten(G):
        """
        Returns the uptake flux of glucose depending on the current glucose concentration.

        Parameters
        ----------
            G : Float
                Glucose concentration in mmol/L.

        Returns
        -------
            v : Float
                Uptake flux in mmol/(gX L) with negative sign.
                Clipped at (-inf, 0).
        """

        # values used for dFBA by     [1]
        vmax = 10.5     # mmol/(gX h) [2]
        kM   = 0.0027   # g/L         [3]
        kM_mmol = kM / snek.elements.molecular_weight('C6H12O6') * 1000
        v = -vmax * G / (kM_mmol + G)

        # [1] Höffner, K., Harwood, S. M., & Barton, P. I. (2013). A reliable 
        # simulator for dynamic flux balance analysis. Biotechnology and 
        # bioengineering, 110(3), 792-802.

        # [2] Varma, A., & Palsson, B. O. (1994). Stoichiometric flux balance 
        # models quantitatively predict growth and metabolic by-product secretion 
        # in wild-type Escherichia coli W3110. Applied and environmental microbiology, 
        # 60(10), 3724-3731.

        # [3] Wong, P., Gladney, S., & Keasling, J. D. (1997). Mathematical model 
        # of the lac operon: inducer exclusion, catabolite repression, and diauxic 
        # growth on glucose and lactose. Biotechnology progress, 13(2), 132-143.
        return np.clip(v,a_min=-np.inf,a_max=0)

    def infeasible_ATPMv2(t, y, model, tol):
        '''
        Infeasibility event. Usually occures when the glucose uptake flux is smaller than the ATPM requirement.
        '''

        X,S,P,G = y  # expand the boundary species

        with model as tmpmodel:
            q_G_max = michaelis_menten(G)
            snek.set_bounds(tmpmodel,'EX_glc__D_e',lb=q_G_max,ub=q_G_max)
            if S <= tol:
                snek.set_bounds(tmpmodel,'EX_so4_e',lb=0,ub=1000)
            fluxes = np.array([])
            for reaction, direction in zip(['BIOMASS_Ec_iML1515_core_75p37M','EX_so4_e','pDNA_synthesis','EX_glc__D_e'],['max','max','max','min']):
                tmpmodel.objective = reaction
                tmpmodel.objective_direction = direction
                tmp = tmpmodel.slim_optimize()
                if np.isnan(tmp):
                    return 0
                else:
                    return 1
    infeasible_ATPMv2.terminal = False
    infeasible_ATPMv2.direction = 0
    
    def infeasible_G(t, y, model, tol):
        ''' Infeasibility if G <= 0.'''
        
        X,S,P,G = y  # expand the boundary species
        
        if G <= 0:
            return 0
        else:
            return 1
    infeasible_G.terminal = False
    infeasible_G.direction = 0
    
    def dFBAv2(t, y, model, tol):
        """
        Calculate the time derivative of external species.
        """

        X,S,P,G = y  # expand the boundary species
        # Calculate the specific exchanges fluxes at the given external concentrations.
        with model as tmpmodel:
            q_G_max = np.min([michaelis_menten(G),0])
            snek.set_bounds(tmpmodel,'EX_glc__D_e',lb=q_G_max,ub=q_G_max)
            if S <= tol:
                snek.set_bounds(tmpmodel,'EX_so4_e',lb=0,ub=1000)
            fluxes = np.array([])
            for reaction, direction in zip(['BIOMASS_Ec_iML1515_core_75p37M','EX_so4_e','pDNA_synthesis','EX_glc__D_e'],['max','max','max','min']):
                tmpmodel.objective = reaction
                tmpmodel.objective_direction = direction
                tmp = tmpmodel.slim_optimize()
                # the nan is excepted here but will trigger the termination event.
                tmp = np.nan_to_num(tmp)
                fluxes = np.append(fluxes,tmp)
                snek.set_bounds(tmpmodel,reaction,tmp,tmp)
            del tmpmodel
        fluxes *= X
        return fluxes
    
    S_0, q_pDNA_max = _in 
    ecoli = cobra.io.read_sbml_model('../models/iML1515_pDNA.xml')
    ecoli.solver = 'cplex'
    ecoli.tolerance = 1e-8
    snek.set_bounds(ecoli,'pDNA_synthesis',lb=.1/100,ub=q_pDNA_max)
    
    ts = np.linspace(0, 100, 10000)
    G_0_gL = 20 # g/L
    G_0 = G_0_gL / snek.elements.molecular_weight('C6H12O6') * 1000 # ~ 222 mmol/L
    X_0 = .02  # g/L
    P_0 = 0  # mmol/L
    y0 = [X_0,S_0,P_0,G_0]
    tol = 1e-6
    sol = solve_ivp(
        fun=dFBAv2,
        t_span=(ts.min(), ts.max()),
        y0=y0,
        t_eval=ts,
        args = (ecoli,tol),
        events= [infeasible_ATPMv2,infeasible_G],
        rtol=1e-8,
        atol=1e-8,
        method='DOP853',
        # max_step=1/60 # i.e. 1 min
    )
    return [S_0, sol, q_pDNA_max]


def do_grid_simulations(q_pDNA_max):
    print(f'Grid Simulations for q_pDNA_max = {q_pDNA_max}')
    t1 = datetime.now()

    # multiprocessing 
    nr_samples = 301
    S_0 = np.linspace(0,3,nr_samples)
    _input = np.vstack([S_0,np.ones(nr_samples)*q_pDNA_max]).T
    output = []
    n_cpu = np.min([nr_samples,120])
    with Pool(processes = n_cpu) as p:
        for _ in tqdm.tqdm(p.imap_unordered(run_pFBA,_input),total=len(_input)):
            output.append(_)

    results = {}
    for out in output:
        S_0, sol, q_pDNA_max = out
        results[S_0] = sol
    results = dict(sorted(results.items()))
    
    with open('{}/q_pDNA_max_{:.5f}.pkl'.format(loc,q_pDNA_max),'wb') as file:
        pickle.dump(results,file)        

    
    t2 = datetime.now()
    print('done')
    print('process time',t2-t1)
    return results

loc = '/mnt/itching_scratch/mgotsmy/220829_slim_paper_v2/results/220829_dFBA_batch_v06'
Path(loc).mkdir(exist_ok=True)
q_pDNA_max_levels = np.linspace(.1,.5,41)/100
for q_pDNA_max in q_pDNA_max_levels:
    do_grid_simulations(q_pDNA_max)


print('Scriped finished sucessfully.')
