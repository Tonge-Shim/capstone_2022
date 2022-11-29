from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, FloatField
from wtforms.validators import DataRequired, Length
from flask_pagedown.fields import PageDownField


class NameForm(FlaskForm):
    name = StringField('What is your name?', validators = [DataRequired()])
    submit = SubmitField('Submit')


class EditProfileForm(FlaskForm):
    name = StringField('Real name', validators=[Length(0, 64)])
    location = StringField('Location', validators=[Length(0, 64)])
    about_me = TextAreaField('About me')
    height = FloatField('Height')
    weight = FloatField('Weight')
    submit = SubmitField('Submit')


class PostForm(FlaskForm):
    body = TextAreaField("What's in your mind?", validators = [DataRequired()])
    submit = SubmitField('Submit')
    # TO-DO: upload video file button -> route to upload page -> go to profile page?


# class UploadForm(FlaskForm):
    # TO-DO: file uploading flask form!! search for it please