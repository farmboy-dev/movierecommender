#import
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from flask import Flask , render_template
from flask import request

app =Flask(__name__)
model = None

#model class
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(1703, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 800)
        self.fc4 = nn.Linear(800, 1703)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

#load model
def load_model():
    global model
    model = SAE()
    model.load_state_dict(torch.load('./model/movie_recommendation.pth'))
    print("success")
    # model.eval()

#movies_list
movies_list = pd.read_csv('movies_list_final.csv')
genre_list = pd.read_csv('genre_90%.csv')
train = pd.read_csv('train.csv', index_col=None)
def pre_data():
    # data_dict = request.args.to_dict()
    data_dict = request.form
    movie = data_dict['Movie']
    genre = data_dict['Genre']
    age = torch.tensor([int(data_dict['Age'])])
    gender = (torch.tensor([1,0]) if data_dict['Gender']=='Female' else torch.tensor([0,1]))
    print(len(gender))
    print(age)
    genre_point = genre_list[genre_list['genre'] == genre]['top_90%'].values
    genre_index = genre_list[genre_list['genre'] == genre]['index'].values - 1
    movie_index = movies_list[movies_list['name'] == movie]['index'].values - 1

    movie_tensor = torch.zeros(1700)
    user_data = torch.cat((movie_tensor,gender,age), dim=0)
    user_data[movie_index] = 5
    user_data[1682+genre_index] = genre_point[0]
    return user_data

@app.route("/")
def info():
    return render_template('info.html')

@app.route("/predict", methods=['POST'])
def predict():
    data = pre_data()
    final_data = {"success": False}
    result = model(data.float()).detach().numpy()
    #top 20 movies
    result_top = sorted(range(len(result)), key=lambda i: result[i])[-20:]

    final_data['poster']= []
    for index in result_top:
        final_data['poster'].append(movies_list[movies_list['index']==index]['poster'].values[0])
        print(final_data)
    final_data['success']= True
    # flask.jsonify(final_data)

    return render_template("index.html", links=final_data['poster'])
if __name__ == '__main__':
    load_model()
    # app.run(host='0.0.0.0', port=8000)
    app.run()