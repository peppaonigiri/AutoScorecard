
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials


# define an objective function
def objective(args):
    return -(args['x'] + args['y'])**2


space = {"x": hp.uniform('x', -10, 10),
         "y": hp.uniform('y', -10, 10)
         }

trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

print(best)
print(space_eval(space, best))