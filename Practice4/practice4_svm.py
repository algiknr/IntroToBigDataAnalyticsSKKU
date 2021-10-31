from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LabeledPoint, SVMWithSGD

conf = SparkConf()
conf.set("spark.master", "local")
sc = SparkContext(conf=conf)

def parsePoint(line):
	try:
		values = [float(x) for x in line.replace(',',' ').split(' ')]
		return LabeledPoint(values[0], values[1:])
	except:
		return None

data = sc.textFile("train_practice4.csv", 3)
trainData = data.map(parsePoint)

data = sc.textFile("test_practice4.csv")
testData = data.map(parsePoint)

# Support Vector Machine
model_SVM = SVMWithSGD.train(trainData, iterations=100,
                             step=1.0, regParam=0.01, regType="l2")


prediction = testData.map(lambda p: (p.label, model_SVM.predict(p.features)))

f = open('result.txt','w')

# Label 0
tp = float(prediction.filter(lambda p: (p[0]==p[1]) & (p[0]==0)).count())
fn = float(prediction.filter(lambda p: (p[0]!=p[1]) & (p[0]==0)).count())
fp = float(prediction.filter(lambda p: (p[0]!=p[1]) & (p[0]==1)).count())
tn = float(prediction.filter(lambda p: (p[0]==p[1]) & (p[0]==1)).count())

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

f.write('Label 0\n')
f.write('F1 Score : {:.4f}\n'.format(f1_score))
f.write('Precision : {:.4f}\n'.format(precision))
f.write('Recall : {:.4f}\n\n'.format(recall))

# Label 1
tp = float(prediction.filter(lambda p: (p[0]==p[1]) & (p[0]==1)).count())
fn = float(prediction.filter(lambda p: (p[0]!=p[1]) & (p[0]==1)).count())
fp = float(prediction.filter(lambda p: (p[0]!=p[1]) & (p[0]==0)).count())
tn = float(prediction.filter(lambda p: (p[0]==p[1]) & (p[0]==0)).count())

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

f.write('Label 1\n')
f.write('F1 Score : {:.4f}\n'.format(f1_score))
f.write('Precision : {:.4f}\n'.format(precision))
f.write('Recall : {:.4f}\n\n'.format(recall))

accuracy = (tp + tn) / (tp + fn + fp + tn)
f.write('Accuracy : {:.4f}'.format(accuracy))

sc.stop()
