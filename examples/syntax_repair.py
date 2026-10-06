"""Review a repair without executing it."""

import pybms as pb

source = "import pybms as pb\nvalues = pb.array([1, 2]"
result = pb.fix_syntax(source)
print(result.source)
print(result.changes)
print("Valid:", result.valid)
