import json

DEFAULT_PARAM_PATH = "default_params.json"
PARAM_TYPES = ["economic assumption",
               "corporate tax baseline",
               "individual income tax baseline"]

def get_default_params():
    """"
    Read parameters from default_params.json

    returns: dictionary with keys "economic assumption", "corporate tax baseline"
             and "individual income tax baseline" with all params
             sorted into the dictionary corresponding to param_type
    """
    param_dict = {"economic assumption": {},
                  "corporate tax baseline": {},
                  "individual income tax baseline": {}}

    with open(DEFAULT_PARAM_PATH) as js:
        default_params = json.loads(js.read())

    for param in default_params:
        param_type = default_params[param]["param_type"]
        v = default_params[param]["value"]
        assert isinstance(v, list)
        if len(v) > 0:
            param_dict[param_type][param] = v[0]
        else:
            param_dict[param_type][param] = None
    return param_dict


def read_reform(reform):
    """
    Read reform and fill in default parameters where paramters are not specified

    returns: dictionary with keys "economic assumption", "corporate tax baseline"
             and "individual income tax baseline" with all params
             sorted into the dictionary corresponding to param_type
    """
    default_params = get_default_params()
    for param_type in default_params:
        for param in default_params[param_type]:
            if param in reform[param_type]:
                reform[param_type][param] = reform[param_type][param]
            else:
                reform[param_type][param] = default_params[param_type][param]

    return reform
