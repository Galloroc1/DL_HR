def check_params_type_int(*args):
    for i in args:
        assert isinstance(i, int), f'params {i} should be {int} , not {type(i)}'