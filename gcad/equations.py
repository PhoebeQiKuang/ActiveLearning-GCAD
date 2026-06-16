# Adopted from GCAD Repo
# Cleaned for OED & Active Learning by Phoebe Kuang

import numpy as np

# ==========================================
# 1. STANDARD PHYSICS (For "Weak" Circuits)
# ==========================================
def system_equations_pop(
        x,
        t,
        state: str,
        Z_list: list,
        topology: object,
        promo_params: dict,  # REQUIRED: Must pass parameters explicitly
        parts_params: dict   # REQUIRED: Must pass parameters explicitly
):
    """
    Builds the system of ODEs using Standard Hill Kinetics.
    Used when 'DsRed_inhibitor' is False.
    """
    system = []
    index = 0
    
    # Iterate through every gene node in the topology
    for n in topology.in_dict.keys():
        # mRNA degradation (fixed rate)
        eq = -2.7 * x[2 * topology.var_dict[n]]
        
        b = []      # Basal rates accumulator
        num = 0     # Numerator accumulator
        denom = 1   # Denominator accumulator
        Z_ZF = 0    # Zinc Finger concentration
        
        # A. PROMOTER REGULATION (Input P1/P2)
        for k in topology.in_dict[n]['P']:
            # Physics: (Dose/Pool)^0.5 scaling implicit in original code
            conc_factor = ((float(topology.dose[n]) / topology.pool[n])/200)**0.5
            eq += conc_factor * promo_params[k][state] * promo_params['k_txn'] * Z_list[index]
            index += 1
            
        # B. ACTIVATOR (Z) REGULATION
        for k in topology.in_dict[n]['Z']:
            b.append(parts_params[k][0]) # Add basal rate
            num += parts_params[k][1] * parts_params[k][2] * x[2 * topology.var_dict[k] + 1]
            denom += parts_params[k][2] * x[2 * topology.var_dict[k] + 1]
            
        # C. INHIBITOR (I) REGULATION
        for k in topology.in_dict[n]['I']:
            # Check if this inhibitor corresponds to an activator in the same node
            # (Logic from original paper regarding competing parts)
            if ('Z' + k[1:]) not in topology.in_dict[n]['Z']:
                b.append(parts_params['Z' + k[1:]][0])
            denom += parts_params[k][0] * x[2 * topology.var_dict[k] + 1]
            
        # D. COMBINE TERMS
        if len(b) == 0:
            b = 0
        else:
            b = np.mean(b)
            Z_ZF = Z_list[index]
            index += 1
            
        # Final Transcription Term
        conc_factor = ((float(topology.dose[n]) / topology.pool[n])/200)**0.5
        eq += conc_factor * (b + num) / denom * 9. * Z_ZF
        
        # E. PROTEIN DEGRADATION & TRANSLATION
        # dx_prot/dt = mRNA - deg * protein
        system.extend([eq, -topology.protein_deg[n[0]] * x[2 * topology.var_dict[n] + 1] + x[2 * topology.var_dict[n]]])
        
    return system


# ==========================================
# 2. DsRed PHYSICS (For "Strong" Circuits)
# ==========================================
def system_equations_DsRed_pop(
        x,
        t,
        state: str,
        Z_list: list,
        topology: object,
        promo_params: dict,
        parts_params: dict
):
    """
    Builds the system of ODEs using Piecewise Sequestration Kinetics.
    Used when 'DsRed_inhibitor' is True.
    """
    system = []
    index = 0
    for n in topology.in_dict.keys():
        eq = -2.7 * x[2 * topology.var_dict[n]]
        b = []
        num = 0
        denom = 1
        Z_ZF = 0
        
        # A. PROMOTER REGULATION
        for k in topology.in_dict[n]['P']:
            conc_factor = ((float(topology.dose[n]) / topology.pool[n])/200)**0.5
            eq += conc_factor * promo_params[k][state] * promo_params['k_txn'] * Z_list[index]
            index += 1
            
        # B. LOGIC BRANCH: Standard vs DsRed
        # If NO inhibitors, use standard math
        if len(topology.in_dict[n]['I']) == 0:
            for k in topology.in_dict[n]['Z']:
                b.append(parts_params[k][0])
                num += parts_params[k][1] * parts_params[k][2] * x[2 * topology.var_dict[k] + 1]
                denom += parts_params[k][2] * x[2 * topology.var_dict[k] + 1]
        
        # If INHIBITORS exist, use DsRed Complex Logic
        else:
            zfa = topology.in_dict[n]['Z'][0] # Activator
            zfi = topology.in_dict[n]['I'][0] # Inhibitor
            b.extend([parts_params[zfa][0], parts_params['Z' + zfi[1:]][0]])
            
            # --- THE COMPLEX PIECEWISE MATH ---
            # This models the strong binding/sequestration of DsRed
            term1 = 4 * parts_params[zfi][0] * x[2 * topology.var_dict[zfi] + 1]
            term2 = (1e-10 + parts_params[zfa][2] * x[2 * topology.var_dict[zfa] + 1])
            A = parts_params[zfa][1] + (1 - parts_params[zfa][1])/6 * (term1/term2 - 2)
            
            m = np.piecewise(A, [A < 1, A >= parts_params[zfa][1]], [1., parts_params[zfa][1], A])
            
            num += m * parts_params[zfa][2] * x[2 * topology.var_dict[zfa] + 1]
            denom += parts_params[zfa][2] * x[2 * topology.var_dict[zfa] + 1] + 4 * parts_params[zfi][0] * x[2 * topology.var_dict[zfi] + 1]
            # ----------------------------------
            
        if len(b) == 0:
            b = 0
        else:
            b = np.mean(b)
            Z_ZF = Z_list[index]
            index += 1
            
        conc_factor = ((float(topology.dose[n]) / topology.pool[n])/200)**0.5
        eq += conc_factor * (b + num) / denom * 9. * Z_ZF
        system.extend([eq, -topology.protein_deg[n[0]] * x[2 * topology.var_dict[n] + 1] + x[2 * topology.var_dict[n]]])
        
    return system