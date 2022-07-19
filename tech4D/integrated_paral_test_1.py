#########################################
# Optimization of post jump HJB
#########################################


#########################################
# Library Loading
#########################################


import numpy as np
import joblib
from joblib import Parallel, delayed
import pickle



path_name = "./tech4D/data/PostJump/"

epsilon_list = np.array([0.1])

OutsideVar = 1


def model(epsilon ):
    InsideVar = 2
    # filename = filename
    my_shelf = {}
    print("print dir():")
    print(dir())
    print("print locals():")    
    print(locals())
    print("print globals():")
    print(globals())
    for key in dir():
        if isinstance(locals()[key], (int, float, str, bool, np.ndarray,list)):
            try:
                my_shelf[key] = locals()[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
        else:
            pass
    for key in globals():
        if isinstance(globals()[key], (int, float, bool, np.ndarray)):
            try:
                my_shelf[key] = globals()[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
        else:
            pass

    file = open(path_name+"test", 'wb')
    pickle.dump(my_shelf, file)
    file.close()




number_of_cpu = joblib.cpu_count()
delayed_funcs = [delayed(model)(epsilon) for epsilon in epsilon_list]
parallel_pool = Parallel(n_jobs=number_of_cpu,require = 'sharedmem')
res = parallel_pool(delayed_funcs)


with open(path_name+"test","rb") as f:
    res = pickle.load(f)
    print("print pickle.load()")
    print(res)