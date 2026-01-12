from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    code = StringField('Code', validators=[DataRequired()])
    submit = SubmitField('Submit')

