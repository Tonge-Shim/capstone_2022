from datetime import datetime
from flask import render_template, \
    session, redirect, url_for, \
    request, flash, current_app
from flask_login import login_required, current_user
from . import main
from .forms import NameForm, EditProfileForm, PostForm
from .. import db
from ..models import User, Post, JIresult
from datetime import datetime
import os
import numpy as np
import torch

import sys
sys.path.append("/Users/simchaeeun/Desktop/2022-2/flask/flasky/app/main")

from models.with_mobilenet import PoseEstimationWithMobileNet
from utils import *
from modules.load_state import load_state




@main.route('/', methods=['GET', 'POST'])
def index():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(body=form.body.data, author=current_user._get_current_object())
        db.session.add(post)
        db.session.commit()
        return redirect(url_for('.index'))
    page = request.args.get('page', 1, type=int)
    query = Post.query
    pagination = query.order_by(Post.timestamp.desc()).paginate(
        page=page, per_page=current_app.config['FLASKY_POSTS_PER_PAGE'],
        error_out=False)
    posts = Post.query.order_by(Post.timestamp.desc()).all()
    return render_template('index.html', form=form, posts=posts, pagination=pagination)

@main.route('/video_upload/test', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        height = request.form['height']
        weight = request.form['weight']
        file = request.files['video']
        path = '/Users/simchaeeun/Desktop/2022-2/flask/flasky/videos/'
        file.save(path + file.filename)
        video = path + file.filename
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load("/Users/simchaeeun/Desktop/2022-2/flask/flasky/app/main/weight.pth", map_location=torch.device('cpu'))
        load_state(net, checkpoint)
        results = evaluate(video, "testresult", net, True, False)
        vectors = []
        for i in range(len(results)):
            if 0 in results:
                results = [0, 0, 0, 0]
            x = -(results[i][2] - results[i][0])
            y = results[i][3] - results[i][1]
            if x == 0 and y == 0:
                x = 0.00001
                y = 0.00001
            vectors.append(unit_vector((y, x)))
        arctanres = []
        for vector in vectors:
            rad = np.arctan2(vector[1], vector[0])
            if rad < 0:
                if -1.57 < rad <= 0: rad = 0
                else: rad += 2 * 3.14
            arctan_angle = rad / np.pi * 180
            arctanres.append(arctan_angle)
        max_angle = np.max(arctanres)
        max_angle = round(max_angle, 2)
        frame_count = np.argmax(arctanres)
        speed = max_angle * (1 / 30 * (frame_count) + 1)
        speed = round(speed, 2)
        jp = max_angle * ((float(height) / 2) / 100) * speed
        jp = round(jp, 2)
        now = datetime.now()
        data = JIresult(str(now.date()), str(now.time()), max_angle, jp, float(height), float(weight))
        post = Post(rom = max_angle,
                    jp = jp,
                    height=float(height),
                    weight = float(weight),
                    author=current_user._get_current_object())
        db.session.add(post)
        db.session.commit()
        return redirect(url_for('.user', username = current_user.username))
        # return redirect(url_for('.show_res')) # url_for('.user')
    return render_template("upload.html")

@main.route('/result',methods = ['GET','POST'])
def show_res():
    return render_template("result.html", results=JIresult.query.all())


@main.route('/success', methods = ['POST'])
def success():
    filepath = '/Users/simchaeeun/Desktop/2022-2/flask/flasky/videos'
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(filepath, file.filename))
        return render_template('acknowledgement.html', name = file.filename)

@main.route('/user/<username>')
def user(username):
    usery = User.query.filter_by(username=username).first_or_404()
    posts = usery.posts.order_by(Post.timestamp.desc()).all()
    return render_template('user.html', user = usery, posts = posts)


@main.route('/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit(id):
    post = Post.query.get_or_404(id)
    form = PostForm()
    if form.validate_on_submit():
        post.body = form.body.data
        db.session.add(post)
        db.session.commit()
        flash('The post has been updated.')
        return redirect(url_for('.post', id=post.id))
    form.body.data = post.body
    return render_template('edit_post.html', form=form)


@main.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm()
    if form.validate_on_submit():
        current_user.name = form.name.data
        current_user.location = form.location.data
        current_user.about_me = form.about_me.data
        current_user.height = form.height.data
        current_user.weight = form.weight.data
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