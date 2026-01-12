from flask import Flask, render_template, request, make_response
import pandas as pd
import sqlite3
import csv
from app.models import User, UserPII, Log, Question, Answer
from app import create_app
from app import db


app = create_app()


def display_users():
    with app.app_context():
        users = User.query.all()
        for user in users:
            print(user)

def display_userpiis():
    with app.app_context():
        userpiis = UserPII.query.all()
        for userpii in userpiis:
            print(userpii)


def display_logs():
    with app.app_context():
        logs = Log.query.all()
        for log in logs:
            print(log)


def display_questions():
    with app.app_context():
        questions = Question.query.all()
        for question in questions:
            print(question)

def display_answers():
    with app.app_context():
        answers = Answer.query.all()
        for answer in answers:
            print(answer)

def display_activity():
    with app.app_context():
        for user in User.query.all():
            print(user)
            print("User logs:")
            query = user.logs.select()
            logs = db.session.scalars(query).all()
            for log in logs:
                print(log)
