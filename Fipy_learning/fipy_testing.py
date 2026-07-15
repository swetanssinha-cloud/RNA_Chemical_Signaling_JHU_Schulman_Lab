import inspect
from fipy import CellVariable, DiffusionTerm, Grid2D, Viewer, TransientTerm

# Print the code to your console
print(inspect.getsource(CellVariable))
print(inspect.getsource(DiffusionTerm))
print(inspect.getsource(Grid2D))
print(inspect.getsource(Viewer))
print(inspect.getsource(TransientTerm))



