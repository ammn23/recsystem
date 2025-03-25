from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from recipe_recsystem import get_recipe_vector, recipes, recommend_knn

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query = data.get("query", "")
    recommendations = recommend_knn(query)
    return jsonify(recommendations)


@app.route('/recipe/<recipe_name>')
def recipe_detail(recipe_name):
    recipe = recipes[recipes["name"] == recipe_name].iloc[0]
    return render_template(
        'recipe.html',
        name=recipe["name"],
        description=recipe["description"],
        ingredients=recipe["ingredients_raw_str"],
        steps=recipe["steps"]
    )

if __name__ == '__main__':
    app.run(debug=True)
