import highspy

h = highspy.Highs()
scenname = 'Flight_intention_30_1_20240605132230'
h.readModel(f'data/output/{scenname}/{scenname}.mps')
h.run()
print('Model ', scenname, ' has status ', h.getModelStatus())