
import numpy as np

def test_get_default():
    import parameters
    parameters.DEFAULT_PARAM_PATH = "test_params.json"

    act = parameters.get_default_params()
    exp = {"economic assumption": {
               "real_int_rate": 0.075
           },
           "corporate tax baseline": {
               "ccr_method": "DDB"
           },
           "individual income tax baseline": {
               "mtr_interest": 0.392
           }
    }

    assert act == exp


def test_load_reform():
    import parameters
    parameters.DEFAULT_PARAM_PATH = "default_params.json"

    ref = {"economic assumption": {
               "real_int_rate": 0.05
           },
           "corporate tax baseline": {
               "ccr_method": "SL"
           },
           "individual income tax baseline": {
               "mtr_interest": 0.4
           }
    }

    processed_reform = parameters.read_reform(ref)
    assert processed_reform["economic assumption"]["real_int_rate"] == 0.05
    assert processed_reform["corporate tax baseline"]["ccr_method"] == "SL"
    assert processed_reform["individual income tax baseline"]["mtr_interest"] == 0.4
    assert processed_reform["economic assumption"]["inflation_rate"] == 0.024


if __name__ == "__main__":
    test_get_default()
    test_load_reform()
