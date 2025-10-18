import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas
from matplotlib import pyplot as plt

model = joblib.load("./artifacts/model.joblib")
data = pandas.read_csv("./data.csv")

X = data
y = X.species
X.drop("species", axis=1, inplace=True)

y_pred = model.predict(X)

report = classification_report(y,y_pred,output_dict=True)

df = pandas.DataFrame(report).transpose()
df.to_csv("./metrics/report.csv", index_label='Metric')

cm = confusion_matrix(y, y_pred)

cmd = ConfusionMatrixDisplay(cm)

cmd.plot()
plt.savefig("./metrics/confusion_matrix")

print(df)