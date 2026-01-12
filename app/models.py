from datetime import datetime, timezone
from typing import Optional
import sqlalchemy as sa
import sqlalchemy.orm as so
from app import db
from app import login
import random

from flask_login import UserMixin

@login.user_loader
def load_user(id):
    return User.query.get(id)


class User(UserMixin, db.Model):
    user_id: so.Mapped[str] = so.mapped_column(sa.String(64), primary_key=True)
    condition_id: so.Mapped[Optional[str]] = so.mapped_column(sa.String(64), nullable=True)                                   

    logs: so.WriteOnlyMapped['Log'] = so.relationship(
        back_populates='participant')
    
    # uselist=False to specify it is a one-to-one relationship
    # cascade="all, delete-orphan" removes userpii related to a user if this user is deleted in User
    pii: so.Mapped['UserPII'] = so.relationship(
        "UserPII", uselist=False, back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f'{self.user_id} - condition : {self.condition_id}'
    
    def get_id(self):
        return self.user_id

    def assign_condition(self, condition_id):
        """Assigns a condition ID to the user."""
        self.condition_id = condition_id
        db.session.commit()

class UserPII(db.Model):
    user_id: so.Mapped[str] = so.mapped_column(sa.String(64), sa.ForeignKey('user.user_id'), primary_key=True)
    email: so.Mapped[Optional[str]] = so.mapped_column(sa.String(120), index=True, unique=True, nullable=True)
    interac_email: so.Mapped[Optional[str]] = so.mapped_column(sa.String(120), index=True, unique=True, nullable=True)

    user: so.Mapped['User'] = so.relationship("User", back_populates="pii")
    
    def __repr__(self):
        return f'{self.user_id} - email : {self.email} - interac email: {self.interac_email}'

class Question(db.Model):
    question_id: so.Mapped[str] = so.mapped_column(sa.String(64), primary_key=True)
    question_content: so.Mapped[str] = so.mapped_column(sa.String(1000)) 
    group_id: so.Mapped[str] = so.mapped_column(sa.String(64))
    form_id: so.Mapped[str] = so.mapped_column(sa.String(64))  
    pilote_id: so.Mapped[str] = so.mapped_column(sa.String(64), nullable=True) 
    question_info: so.Mapped[str] = so.mapped_column(sa.String(1000), nullable=True)                            

    logs: so.WriteOnlyMapped['Log'] = so.relationship(
        back_populates='question')
    
    answers: so.WriteOnlyMapped['Answer'] = so.relationship(
        back_populates='question')

    def __repr__(self):
        return f'{self.question_id} ({self.group_id}) - {self.form_id}'


    def get_answer(self):
        query = self.answers.select()
        answers = db.session.scalars(query).all()
        return answers
    
    def get_form(self):
        form = []
        query = self.answers.select()
        answers = db.session.scalars(query).all()
        for answer in answers:
            form.append((answer.answer_id,answer.answer_content))
        return form

    def get_random_answer(self, seed=None):
        """
        Get a random answer for the question with the given question_id.
        
        :param seed: Seed value for reproducibility (optional).
        :return: Randomly selected Answer object or None if no answers exist.
        """
        # Get all answers related to the question
        query = self.answers.select()
        answers = db.session.scalars(query).all()

        if not answers:
            return None  # No answers available for this question

        # Set the seed if provided
        if seed is not None:
            random.seed(seed)

        # Return a random answer
        return random.choice(answers)

class Answer(db.Model):
    answer_id: so.Mapped[str] = so.mapped_column(sa.String(64), primary_key=True)
    answer_content: so.Mapped[str] = so.mapped_column(sa.String(1000)) 
    answer_weight: so.Mapped[float] = so.mapped_column(sa.Float)

    question_id: so.Mapped[str] = so.mapped_column(sa.ForeignKey(Question.question_id),index=True)                           
    question: so.Mapped['Question'] = so.relationship(back_populates='answers')

    logs: so.WriteOnlyMapped['Log'] = so.relationship(
        back_populates='answer')

    def __repr__(self):
        return f'{self.answer_id} - weight: {self.answer_weight}'



class Log(db.Model):
    log_id: so.Mapped[int] = so.mapped_column(primary_key=True)
    timestamp: so.Mapped[datetime] = so.mapped_column(
        index=True, default=lambda: datetime.now(timezone.utc))
    
    # Type of log: answer, score_computation
    # By default, it is an answer to a question
    log_type: so.Mapped[str] = so.mapped_column(sa.String(64), default='answer')

    # Additionnal info about the log
    log_info: so.Mapped[Optional[str]] = so.mapped_column(sa.String(64))
    
    user_id: so.Mapped[str] = so.mapped_column(sa.ForeignKey(User.user_id),index=True)
    participant: so.Mapped['User'] = so.relationship(back_populates='logs')

    question_id: so.Mapped[Optional[str]] = so.mapped_column(sa.ForeignKey(Question.question_id),index=True)
    question: so.Mapped['Question'] = so.relationship(back_populates='logs')

    answer_id: so.Mapped[Optional[str]] = so.mapped_column(sa.ForeignKey(Answer.answer_id),index=True)
    answer: so.Mapped['Answer'] = so.relationship(back_populates='logs')

    phase_id: so.Mapped[str] = so.mapped_column(sa.String(64))

    def __repr__(self):
        if self.log_type=='answer':
            return f'{self.timestamp} - User:{self.user_id} - Q:{self.question_id} - A:{self.answer_id} - Ph:{self.phase_id}'
        elif self.log_type=='score_computation':
            return f'{self.timestamp} - User:{self.user_id} - Phase:{self.phase_id} - Score:{self.log_info} - Ph:{self.phase_id}'
        elif self.log_type=='finished':
            return f'{self.timestamp} - User:{self.user_id} - Type:{self.log_type} - Time:{self.timestamp} - Ph:{self.phase_id}'
        else:
            return f'{self.timestamp} - User:{self.user_id} - Type:{self.log_type} - Info:{self.log_info} - Ph:{self.phase_id}'

