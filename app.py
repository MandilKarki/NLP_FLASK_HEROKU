from flask import Flask,render_template,url_for,request
import pickle

app = Flask(__name__,static_folder='app/static/')
pickle_in = open("model_pickle","rb")
svm_model = pickle.load(pickle_in)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		comment = request.form['text']
		data = [comment]
		my_prediction = svm_model.predict(data)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug = True)
