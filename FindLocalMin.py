#find local minimu of x^4-3x^3+2 get derivative and use gradient descent

xOld = 0
xNew = 6
eps = 0.01
precision = 0.0001

def fDerivative(x):
	return 4 * x**3 - 9 * x**2

while abs(xNew - xOld) > precision:
	xOld = xNew
	xNew = xOld - eps * fDerivative(xOld)

print("Local minimum ocurrs at:", xNew)