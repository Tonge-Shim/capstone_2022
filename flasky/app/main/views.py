from datetime import datetime
from flask import render_template, \
    session, redirect, url_for, \
    request, flash, current_app
from flask_login import login_required, current_user
from . import main
from .forms import NameForm, EditProfileForm, PostForm
from .. import db
from ..models import User, Post
import os


@main.route('/', methods=['GET', 'POST'])
def index():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(body=form.body.data,author=current_user._get_current_object())
        db.session.add(post)
        db.session.commit()
        return redirect(url_for('.index'))
    page = request.args.get('page', 1, type=int)
    query = Post.query
    pagination = query.order_by(Post.timestamp.desc()).paginate(
        page=page, per_page=current_app.config['FLASKY_POSTS_PER_PAGE'],
        error_out=False)
    posts = pagination.items
    return render_template('index.html', form=form, posts=posts, pagination=pagination)




@main.route('/video_upload/test')
def upload_file():
    return render_template('upload.html')

@main.route('/success', methods = ['POST'])
def success():
    filepath = '/Users/simchaeeun/Desktop/2022-2/flask/flasky/videos'
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(filepath, file.filename))
        return render_template('acknowledgement.html', name = file.filename)

@main.route('/user/<username>')
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    posts = user.posts.order_by(Post.timestamp.desc()).all()
    return render_template('user.html', user = user, posts = posts)

@main.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm()
    if form.validate_on_submit():
        current_user.name = form.name.data
        current_user.location = form.location.data
        current_user.about_me = form.about_me.data
        current_user.height = form.height.data
        db.session.add(current_user._get_current_object())
        db.session.commit()
        flash('Your profile has been updated.')
        return redirect(url_for('.user', username = current_user.username))
    form.name.data = current_user.name
    form.location.data = current_user.location
    form.height.data = current_user.height
    form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', form = form)

@main.route('/post/<int:id>')
def post(id):
    post = Post.query.get_or_404(id)
    return render_template('post.html', posts=[post])