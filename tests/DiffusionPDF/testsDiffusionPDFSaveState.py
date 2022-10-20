from pyDiffusion import DiffusionPDF

pdf = DiffusionPDF(1e24, 'beta', [1, 1], 1000)
for _ in range(100):
    pdf.iterateTimestep()
pdf.saveState()

pdf1 = DiffusionPDF.fromFiles('ScalarsNone.json', 'OccupancyNone.txt')
print(pdf == pdf1)