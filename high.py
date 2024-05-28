import highspy

h = highspy.Highs()
filename = 'Flight_intention_120_4.mps'
h.readModel(filename)
h.run()
print('Model ', filename, ' has status ', h.getModelStatus())