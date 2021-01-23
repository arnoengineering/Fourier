
file = 'poly_file.csv'
n_file = 'polynomial_form.txt'

with open(file) as f:
    coefs = f.readlines()

poly_ord = len(coefs)

newlines = []
for n in range(poly_ord):
    li = coefs[n]
    x_val = poly_ord - (n + 1)
    if x_val > 0:
        st = f'({li})x^({x_val}) +'
    else:
        st = f'({li})'
    newlines.append(st)

with open(n_file) as f:
    f.writelines(newlines)