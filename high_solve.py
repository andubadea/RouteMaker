import highspy

h = highspy.Highs()
scenname = 'Flight_intention_120_1_3_B_C1_T20_20240606172059'
h.readModel(f'data/output/{scenname}/{scenname}.mps')
h.run()
print('Model ', scenname, ' has status ', h.getModelStatus())