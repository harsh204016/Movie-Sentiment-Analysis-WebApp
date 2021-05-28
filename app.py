from flask import Flask, request, render_template
import feature
from tensorflow.keras.models import load_model

#model = joblib.load('./pipeline.sav')
app = Flask(__name__)
model = load_model("model.h5")
model.compile()




@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def get_delay():

    result=request.form
    query_text = result['review']
    query_text = feature.text_cleaning(query_text)
    query_text = feature.text_encoding(query_text)
    print(query_text)
    
    pred = model.predict_classes(query_text)
    print(pred)
    print("predicted value",pred[0])
    label = {0:'negative',1:'somewhat negative',2:'neutral',3:'somewhat positive',4:'positive'}
    
    return render_template('index.html', prediction_text='Predicted Movie Review is {}'.format(label[pred[0]]))
    
if __name__ == '__main__':
    app.run(port=8080, debug=True)
