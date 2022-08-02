import numpy as np

x = 3
biases = np.ones(shape=2+x) * 1/2

# Initialize the pdf
pdf = np.zeros(shape=2+x)
pdf[0] = 1
t = 0

''' This does in fact work, but it's super ugly :('''
def iteratePDF(biases, pdf, t, x):
    pdf_new = np.zeros(pdf.shape)

    if t <= x: 
        for i in range(len(pdf)-1):
            pdf_new[i] += pdf[i] * 1/2
            pdf_new[i+1] += pdf[i] * 1/2
    else:
        for i in range(len(pdf)):
            if i == 0:
                pdf_new[i] += pdf[i]
            
            elif (pdf[-1] != 0):
                if i == (len(pdf) - 1):
                    pdf_new[i-1] += pdf[i]
                else:
                    pdf_new[i] += pdf[i] * 1/2
                    pdf_new[i-1] += pdf[i] * 1/2
            else:
                if i == (len(pdf) - 1):
                    continue 
                elif i == (len(pdf) - 2):
                    pdf_new[i+1] += pdf[i]
                else:
                    pdf_new[i] += pdf[i] * 1/2
                    pdf_new[i+1] += pdf[i] * 1/2
    return pdf_new

for _ in range(10):
    pdf = iteratePDF(biases, pdf, t, x)
    t += 1
    print(t, pdf, sum(pdf))