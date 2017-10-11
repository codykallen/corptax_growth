import pandas as pd
import numpy as np

params_econ = {
    'real_int_rate': 0.075,
    'inflation_rate': 0.024,
    'real_financial_return': 0.2,
    'debt_financing': 0.32,
    'new_equity_financing': 0,
    'corp_investment_share': 0.545,
    'incshare_capital': 0.406,
    'capincshare_equip': 0.297,
    'capincshare_struc': 0.275,
    'capincshare_ip': 0.202,
    'capincshare_rr': 0.061
}

params_corptax_base = {
    'ccr_method': 'DDB',
    'corptax_rate': 0.35,
    'int_deductibility': 1,
    'dividend_credit': 0
}

params_iitax_base = {
    'mtr_interest': 0.392,
    'mtr_dividends': 0.2,
    'mtr_capitalgains': 0.2,
    'cg_holding_period': 5
}

def calcA(life, method, i):
    if method == 'DDB':
        if life == 5:
            deductions = [0.2, 0.32, 0.192, 0.1152, 0.1152, 0.0567]
        else:
            deductions = [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446]
    elif method == 'SL':
        if life == 5:
            deductions = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
        else:
            deductions = [0.07145, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0715]
    else:
        deductions = [1]
    pv_deductions = []
    for j in range(len(deductions)):
        pv_deductions.append(deductions[j] / (1 + i)**j)
    return sum(pv_deductions)

def calcEATR(econ_params, corptax_params, iitax_params, L):
    r = econ_params['real_int_rate']
    pi = econ_params['inflation_rate']
    p = econ_params['real_financial_return']
    dB = econ_params['debt_financing']
    dN = econ_params['new_equity_financing']
    i = (1 + r) * (1 + pi) - 1
    assert dB >= 0 or db <= 1
    assert dN >= 0 or db <= 1
    assert i > 0

    mi = iitax_params['mtr_interest']
    md = iitax_params['mtr_dividends']
    mcg = iitax_params['mtr_capitalgains']
    n_cg = iitax_params['cg_holding_period']
    c = corptax_params['dividend_credit']
    assert n_cg > 0
    z = 1 - (((1 + i) ** n_cg * (1 - mcg) + mcg) ** (1.0 / n_cg) - 1) / i
    rho = (1 - mi) * i / (1 - z)
    gamma = (1 - md) / (1 - c) / (1 - z)

    assert L in [5, 7]
    delta = 1.0 / L

    tau = corptax_params['corptax_rate']
    phi = corptax_params['int_deductibility']
    ccr_method = corptax_params['ccr_method']
    assert ccr_method in ['DDB', 'SL', 'EXP']
    A = calcA(L, ccr_method, i)

    F = gamma * dB* (1 - (1 + i * (1 - phi * tau)) / (1 + rho)) - (1 - gamma) * dN * (1 - 1 / (1 + rho))
    p_coc = (1 - tau * A) / (1 - tau) * (r + delta) - F * (1 + r) / gamma / (1 - tau) - delta
    p = max(p, p_coc)
    R = (p - p_coc) * gamma * (1 - tau) * (1 + pi) / (1 + rho)
    Rstar = (p - r) / (1 + r)
    EATR = (Rstar - R) / (p / (1 + r))
    user_cost = p_coc + delta
    return (EATR, user_cost)

def investmentResponse(corptax_params_ref, iitax_params_ref, elast_type, elast_value):
    (eatr5_base, coc5_base) = calcEATR(params_econ, params_corptax_base, params_iitax_base, 5)
    (eatr7_base, coc7_base) = calcEATR(params_econ, params_corptax_base, params_iitax_base, 7)
    (eatr5_ref, coc5_ref) = calcEATR(params_econ, corptax_params_ref, iitax_params_ref, 5)
    (eatr7_ref, coc7_ref) = calcEATR(params_econ, corptax_params_ref, iitax_params_ref, 7)

    assert elast_type in ['EATR', 'net of EATR', 'semi', 'user cost']
    if elast_type == 'EATR':
        assert elast_value <= 0
        pctchg5 = eatr5_ref / eatr5_base - 1
        pctchg7 = eatr7_ref / eatr7_base - 1
        deltaI = elast_value * (pctchg5 + pctchg7) / 2
    elif elast_type == 'net of EATR':
        assert elast_value >= 0
        pctchg5 = (1 - eatr5_ref) / (1 - eatr5_base) - 1
        pctchg7 = (1 - eatr7_ref) / (1 - eatr7_base) - 1
        deltaI = elast_value * (pctchg5 + pctchg7) / 2
    elif elast_type == 'semi':
        assert elast_value <= 0
        deltaI = elast_value * (eatr5_ref + eatr7_ref - eatr5_base - eatr7_base) / 2
    else:
        assert elast_value <= 0
        deltaI = elast_value * ((coc5_ref + coc7_ref) / (coc5_base + coc7_base) - 1)
    return deltaI


def growth_accounting(econ_params, deltaI, cap_data_path, growth_data_path, response_start_year):
    assert response_start_year > 2016
    alpha_equip = econ_params['incshare_capital'] * econ_params['capincshare_equip']
    alpha_struc = econ_params['incshare_capital'] * econ_params['capincshare_struc']
    alpha_ip = econ_params['incshare_capital'] * econ_params['capincshare_ip']
    alpha_rr = econ_params['incshare_capital'] * econ_params['capincshare_rr']
    base_data = pd.read_csv(cap_data_path)
    K_equip_old = np.asarray(base_data['K_equip'])
    K_struc_old = np.asarray(base_data['K_struc'])
    K_ip_old = np.asarray(base_data['K_ip'])
    K_rr_old = np.asarray(base_data['K_rr'])
    I_equip_old = np.asarray(base_data['I_equip'])
    I_struc_old = np.asarray(base_data['I_struc'])
    I_ip_old = np.asarray(base_data['I_ip'])
    I_rr_old = np.asarray(base_data['I_rr'])
    D_equip_old = [K_equip_old[i+1] - K_equip_old[i] - I_equip_old[i] for i in range(len(K_equip_old) - 1)]
    D_struc_old = [K_struc_old[i+1] - K_struc_old[i] - I_struc_old[i] for i in range(len(K_struc_old) - 1)]
    D_ip_old = [K_ip_old[i+1] - K_ip_old[i] - I_ip_old[i] for i in range(len(K_ip_old) - 1)]
    D_rr_old = [K_rr_old[i+1] - K_rr_old[i] - I_rr_old[i] for i in range(len(K_rr_old) - 1)]
    delta_equip = -sum([D_equip_old[i] * K_equip_old[i] for i in range(len(D_equip_old))]) / sum([K_equip_old[i] ** 2 for i in range(len(D_equip_old))])
    delta_struc = -sum([D_struc_old[i] * K_struc_old[i] for i in range(len(D_struc_old))]) / sum([K_struc_old[i] ** 2 for i in range(len(D_struc_old))])
    delta_ip = -sum([D_ip_old[i] * K_ip_old[i] for i in range(len(D_ip_old))]) / sum([K_ip_old[i] ** 2 for i in range(len(D_ip_old))])
    delta_rr = -sum([D_rr_old[i] * K_rr_old[i] for i in range(len(D_rr_old))]) / sum([K_rr_old[i] ** 2 for i in range(len(D_rr_old))])
    corpshare = econ_params['corp_investment_share']

    growth_data = pd.read_csv(growth_data_path)
    maxyear = 2250

    ## set up first year (2015) for all variables
    K_equip_base = [K_equip_old[-1]]
    K_struc_base = [K_struc_old[-1]]
    K_ip_base = [K_ip_old[-1]]
    K_rr_base = [K_rr_old[-1]]
    K_equip_ref = [K_equip_old[-1]]
    K_struc_ref = [K_struc_old[-1]]
    K_ip_ref = [K_ip_old[-1]]
    K_rr_ref = [K_rr_old[-1]]
    I_equip_base = [I_equip_old[-1]]
    I_struc_base = [I_struc_old[-1]]
    I_ip_base = [I_ip_old[-1]]
    I_rr_base = [I_rr_old[-1]]
    I_equip_ref = [I_equip_old[-1]]
    I_struc_ref = [I_struc_old[-1]]
    I_ip_ref = [I_ip_old[-1]]
    I_rr_ref = [I_rr_old[-1]]
    GDP_base = [growth_data['gdp'][0]]
    govshare = [growth_data['govshare'][0]]
    dy_equip = [0]
    dy_struc = [0]
    dy_ip = [0]
    dy_rr = [0]
    dy_tot = [0]
    gov_inc = [govshare[0] * GDP_base[0]]
    priv_inc_base = [(1 - govshare[0]) * GDP_base[0]]
    priv_inc_ref = [(1 - govshare[0]) * GDP_base[0]]
    GDP_ref = [gov_inc[0] + priv_inc_ref[0]]
    ## extrapolate forward
    for j in range(1, maxyear - 2015):
        if j + 2015 < 2028:
            GDP_base.append(growth_data['gdp'][j])
            govshare.append(growth_data['govshare'][j])
        elif j + 2015 < 2048:
            GDP_base.append(GDP_base[j-1] * (1 + growth_data['gdp_growth'][j]))
            govshare.append(govshare[j-1])
        else:
            GDP_base.append(GDP_base[j-1] * (1 + 0.02))
            govshare.append(govshare[j-1])
        K_equip_base.append(K_equip_base[j-1] * (1 - delta_equip) + I_equip_base[j-1])
        K_struc_base.append(K_struc_base[j-1] * (1 - delta_struc) + I_struc_base[j-1])
        K_ip_base.append(K_ip_base[j-1] * (1 - delta_ip) + I_ip_base[j-1])
        K_rr_base.append(K_rr_base[j-1] * (1 - delta_rr) + I_rr_base[j-1])
        K_equip_ref.append(K_equip_ref[j-1] * (1 - delta_equip) + I_equip_ref[j-1])
        K_struc_ref.append(K_struc_ref[j-1] * (1 - delta_struc) + I_struc_ref[j-1])
        K_ip_ref.append(K_ip_ref[j-1] * (1 - delta_ip) + I_ip_ref[j-1])
        K_rr_ref.append(K_rr_ref[j-1] * (1 - delta_rr) + I_rr_ref[j-1])
        I_equip_base.append(I_equip_base[j-1] * GDP_base[j] / GDP_base[j-1])
        I_struc_base.append(I_struc_base[j-1] * GDP_base[j] / GDP_base[j-1])
        I_ip_base.append(I_ip_base[j-1] * GDP_base[j] / GDP_base[j-1])
        I_rr_base.append(I_rr_base[j-1] * GDP_base[j] / GDP_base[j-1])
        if j + 2015 < response_start_year:
            I_equip_ref.append(I_equip_base[j])
            I_struc_ref.append(I_struc_base[j])
            I_ip_ref.append(I_ip_base[j])
            I_rr_ref.append(I_rr_base[j])
        else:
            I_equip_ref.append(I_equip_base[j] * (1 + deltaI * corpshare))
            I_struc_ref.append(I_struc_base[j] * (1 + deltaI * corpshare))
            I_ip_ref.append(I_ip_base[j] * (1 + deltaI * corpshare))
            I_rr_ref.append(I_rr_base[j] * (1 + deltaI * corpshare))
        dy_equip.append((K_equip_ref[j] / K_equip_ref[j-1] - K_equip_base[j] / K_equip_base[j-1]) * alpha_equip)
        dy_struc.append((K_struc_ref[j] / K_struc_ref[j-1] - K_struc_base[j] / K_struc_base[j-1]) * alpha_struc)
        dy_ip.append((K_ip_ref[j] / K_ip_ref[j-1] - K_ip_base[j] / K_ip_base[j-1]) * alpha_ip)
        dy_rr.append((K_rr_ref[j] / K_rr_ref[j-1] - K_rr_base[j] / K_rr_base[j-1]) * alpha_rr)
        dy_tot.append(dy_equip[j] + dy_struc[j] + dy_ip[j] + dy_rr[j])
        gov_inc.append(govshare[j] * GDP_base[j])
        priv_inc_base.append((1 - govshare[j]) * GDP_base[j])
        priv_inc_ref.append(priv_inc_ref[j-1] * (priv_inc_base[j] / priv_inc_base[j-1] + dy_tot[j]))
        GDP_ref.append(gov_inc[j] + priv_inc_ref[j])
    change_steadystate = GDP_ref[-1] / GDP_base[-1] - 1
    change_2028 = GDP_ref[13] / GDP_base[13] - 1
    change_growth_2018_2028 = (GDP_ref[13] / GDP_ref[3])**0.1 - (GDP_base[13] / GDP_base[3])**0.1
    return (change_steadystate, change_2028, change_growth_2018_2028)
