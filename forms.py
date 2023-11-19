from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class VerseInputForm(FlaskForm):
    user_input = StringField('First Verse', validators=[DataRequired(), Length(max=200)])
    submit = SubmitField('Generate!')