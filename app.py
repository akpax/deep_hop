from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
from model_manager import GPTJ
import transformers
import torch
from sqlalchemy.orm import relationship
from flask_sqlalchemy import SQLAlchemy
from forms import VerseInputForm


app = Flask(__name__)
app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'

# CONNECT TO DB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///model_logging.db'
db = SQLAlchemy()
db.init_app(app)

#Configure Tables
class UserInput(db.Model):
    __tablename__="user_input"
    id = db.Column(db.Integer, primary_key=True)
    input = db.Column(db.String, nullable=False)
    generated_verses = relationship("GeneratedVerse", back_populates="input")

class GeneratedVerse(db.Model):
    __tablename__="generated_verses"
    id=db.Column(db.Integer, primary_key=True)
    input_id = db.Column(db.Integer, db.ForeignKey("user_input.id")) 
    input = relationship("UserInput", back_populates="generated_verses")
    output = db.Column(db.String, nullable=False) #Field to store generated verse

with app.app_context():
    db.create_all()


# Global variable to keep track of the model's state
loaded_model = None

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/load_model", methods=["POST"])
def load_model():
    global loaded_model 
    # Check if the model is already loaded
    print("Before loading, model is:", "Loaded" if loaded_model else "None")
    if loaded_model is not None:
        return jsonify({"message": "Model is already loaded"})  
    try:
        loaded_model = GPTJ("model/gpt-j-6b.bin")
        print("After loading, model is:", "Loaded" if loaded_model else "None")
        return jsonify({"message": "Model loaded successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/generate_lyrics", methods=["GET", "POST"])
def generate_lyrics():
    global loaded_model
    form = VerseInputForm()
    print("On entering generate_lyrics, model is:", "Loaded" if loaded_model else "None")
    # handle
    if loaded_model is None:
        return redirect(url_for("home"))
    
    # Check if the user clicked the "Generate Again" button
    if request.method == "POST" and "generate_again" in request.form:
        return render_template("generate.html", form=form)  # Set the flag to indicate the model is loaded

    if form.validate_on_submit():
        user_input = form.user_input.data

        generated_verses=loaded_model.generate_verse(user_input)
        
        # log input and output to SQL Database
        new_input = UserInput(input=user_input)
        db.session.add(new_input)
        db.session.commit() #necessary to get input id
        for verse in generated_verses:
            new_verse = GeneratedVerse(output=verse, input_id=new_input.id)
            db.session.add(new_verse)
        db.session.commit()
        return redirect(url_for("display", user_input=user_input, generated_verses=generated_verses))
    return render_template("generate.html", form=form)

@app.route("/display", methods=["GET"])
def display():
    if request.method == "POST":
        return redirect(url_for("generate_lyrics"))
    # Access URL parameters using request.args
    user_input = request.args.get("user_input")
    generated_verses = request.args.getlist("generated_verses")
    return render_template("display.html", user_input=user_input, generated_verses=generated_verses)


if __name__ == "__main__":
    app.run(debug=True, port=8000)