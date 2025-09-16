from verry import FloatInterval as FI
from verry.affineform import AffineForm, summarize

x = [AffineForm(FI()), AffineForm(FI())]


def henon(x, y):
    a = 1.400000009849371
    b = 0.300000019143266
    return (1 + y - a * x * x, b * x)


for i in range(5):
    y0, y1 = henon(x[0], x[1])
    x[0] = y0
    x[1] = y1
    summarize(x, 100, 300)

    ran = x[0].range()
    print(i)
    print(f"  diam: {ran.diam()}")
    print(f"  appx: {ran.mid()}")

print(x[0].range())
print(x[1].range())
