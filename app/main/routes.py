from app.main import bp
from flask import render_template, flash, redirect, url_for, current_app, make_response, g, send_file, Response
from flask_login import current_user, login_required
from app.models import UserPII, Log, Question, Answer
from flask import request
from app import db
from app import create_app
import pickle
from datetime import datetime, timezone
import pandas as pd
from app.ml.deploy import predict_for_example
import numpy as np
import matplotlib.pyplot as plt
# Use the Agg backend for non-GUI rendering
import matplotlib
matplotlib.use('Agg')
import io
from sqlalchemy import desc
# Treat multiple answers for checkbox questions
from werkzeug.datastructures import ImmutableMultiDict


#####################################
############# Functions #############
#####################################

## Function to initialize the feature_content dictionnary
def get_feature_content_dic():
    feature_content_dic_init = {
    'sommeil_1': [
        'Sleep quality',
        'During the past seven days, how would you rate your sleep quality overall?',
        "Sleep quality is a critical factor influencing mental health. Poor sleep can contribute to increased stress, anxiety, and depression, while good sleep supports emotional regulation and cognitive function. Over a seven-day period, tracking sleep quality provides insights into an individual’s ability to recover and manage daily challenges. Since sleep disturbances often correlate with mental health challenges, incorporating this variable into a mental health prediction model improves the model's ability to identify patterns and make accurate predictions, ultimately supporting more personalized and actionable feedback.",
        ],
    'autogestion_9': [
        'Healthy diet',
        'Indicate how often you have used a healthy diet it in the past month.',
        "Diet plays a vital role in mental health, as a nutritious diet supports brain function, reduces inflammation, and stabilizes mood. Frequent consumption of healthy foods is linked to lower risks of depression and anxiety, while poor dietary habits can exacerbate mental health challenges. By asking about the frequency of healthy diet use over the past month, the model can identify patterns between diet consistency and mental health outcomes. This information enhances the ability to provide accurate predictions and tailored feedback to support individuals in improving their overall well-being.",
        ],
    'act_friends': [
        'Social activities',
        'How often do you do activities with one or more friend(s)?',
        "Social interactions are closely tied to mental health, as spending time with friends provides emotional support, reduces feelings of loneliness, and fosters a sense of belonging. Regular social activities help buffer stress, improve mood, and promote resilience against mental health challenges. By assessing the frequency of activities with friends, the model can capture the impact of social connections on mental health, leading to more accurate predictions and meaningful feedback to help individuals enhance their social well-being.",
        ],
    'quartier_domicile_3': [
        'Friendly neighborhood',
        'Do you perceive your neighborhood as friendly ?',
        "Perceptions of neighborhood friendliness significantly influence mental health. A friendly neighborhood fosters a sense of safety, social support, and community, which can reduce stress and feelings of isolation. Positive social environments promote well-being by encouraging interactions and creating a buffer against mental health challenges. Including this variable allows the model to capture the impact of local social dynamics on mental health, enabling more accurate predictions and actionable insights to enhance community-based interventions.",
        ],
    'act_volunteer': [
        'Volunteering',
        'How often do you volunteer or involve yourself in a cause?',
        "Volunteering and involvement in a cause are strongly associated with improved mental health. Engaging in altruistic activities provides a sense of purpose, boosts self-esteem, and fosters social connections, all of which contribute to emotional well-being. Such activities also encourage positive thinking and reduce stress by shifting focus away from personal challenges. By evaluating the frequency of volunteering, the model can better understand the relationship between community engagement and mental health, enhancing the accuracy of predictions and the relevance of feedback.",
        ],
    }
    return feature_content_dic_init

# # Function to get the most recent answers
def get_most_recent_answers(user_id, question_list):
    """
    Retrieves all answers linked to the most recent timestamp for each question in the question_list for the specified user_id.

    :param user_id: The ID of the user.
    :param question_list: A list of Question instances.
    :return: A dictionary mapping question_id to a list of tuples (answer_id, answer_content, answer_weight).
    """
    recent_answers = {}

    for question in question_list:
        # Find the most recent timestamp for the given user and question
        recent_timestamp = (
            db.session.query(Log.timestamp)
            .filter_by(user_id=user_id, question_id=question.question_id)
            .order_by(desc(Log.timestamp))
            .limit(1)
            .scalar()
        )

        if recent_timestamp:
            # Retrieve all answers linked to this timestamp
            recent_logs = (
                db.session.query(Log)
                .filter_by(user_id=user_id, question_id=question.question_id, timestamp=recent_timestamp)
                .all()
            )

            recent_answers[question.question_id] = [
                (log.answer.answer_id, log.answer.answer_content, log.answer.answer_weight)
                for log in recent_logs
            ]
        else:
            recent_answers[question.question_id] = []  # No answer recorded

    return recent_answers


# Function to transform MultiDict (due to checkbox questions) of the form to a simple dict
def form_todict(request_form):
    form_data = request_form.to_dict(flat=False)
    cleaned_form_data = {}
    for key, value in form_data.items():
        # If [] in the key, it is a checkbox question
        if "[]" in key:
            cleaned_form_data[key.rstrip("[]")] = value
        else:
            cleaned_form_data[key] = value[0]
    return cleaned_form_data

# Function to transform a list of questions into a dictionnary where a key is question_id and a value is a tuple of:
# question.question_content: content of the question
# question_info_list: list of additionnal info for this question, may be empty
# question.form_id: form_id of the question, for example scroll
# question.get_form(): list of answers of the question where each element of the list is a tuple (answer_id,answer_content)
def questionnaire(questions):
    questionnaire_dico = {}
    for question in questions:
        if question.question_info==None:
            question_info_list = []
        else:
            question_info_list = question.question_info.split(";")
        questionnaire_dico[question.question_id] = (question.question_content, question_info_list, question.form_id, question.get_form())
    return questionnaire_dico

# Function to get a list of ids of questions
def get_question_ids(questions):
    question_ids = [question.question_id for question in questions]
    return question_ids

#####################################
############## Routes ###############
#####################################

# @bp.before_app_request
# def before_request():
#     pass

@bp.route('/consent')
@login_required
def consent():
    return render_template('main/consent.html')

@bp.route('/sociodemo', methods=["POST"])
@login_required
def sociodemo():
     # step 1 : extract questions for sociodemo
    questions = Question.query.filter(Question.group_id == "sociodemo").all()
    questionnaire_dico = questionnaire(questions)
    return render_template(
        'main/sociodemo.html',
        questionnaire_dico = questionnaire_dico,
        skip_valid=current_app.config['SKIP_VALID'],
    )

@bp.route('/knowledge_before', methods=["POST"])
@login_required
def knowledge_before():

    # Convert form from sociodemo with list for questions with multiple answers
    form_data = form_todict(request.form)

    # step 1 : extract questions for sociodemo
    questions = Question.query.filter(Question.group_id == "sociodemo").all()
    questionnaire_dico_responses = questionnaire(questions)
    
    # step 2 : extract and load answer values for sociodemo
    timestamp = datetime.now(timezone.utc)
    seed = 0
    for question_id, (_,_, form_id, questionnaire_value) in questionnaire_dico_responses.items():

        # For cursor, answer_content is registered instead of answerd_id
        if form_id=="cursor":
            answer_content = form_data[question_id]
            # Find the answer_id associated to this answer_content
            for a_id, a_content in questionnaire_value:
                if a_content == answer_content:
                    answer_id = a_id
            
        else:
            answer_id = form_data[question_id]

        # Chose random value if not answered for debug
        if not answer_id and current_app.config['SKIP_VALID']:
            question = db.session.get(Question, question_id)
            answer_id = question.get_random_answer(seed=seed).answer_id
                
        new_log = Log(
            timestamp=timestamp,
            user_id=current_user.user_id,
            question_id=question_id,
            answer_id=answer_id,
            phase_id='sociodemo'
        )
        db.session.add(new_log)
    db.session.commit()

     # step 3 : extract questions for knowledge
    questions = Question.query.filter(Question.group_id == "knowledge").all()
    questionnaire_dico = questionnaire(questions)
    return render_template(
        'main/knowledge_before.html',
        questionnaire_dico = questionnaire_dico,
        skip_valid=current_app.config['SKIP_VALID'],
    )

@bp.route('/lifestyle', methods=['GET', 'POST'])
@login_required
def lifestyle():
    
    # step 1 : extract questions ids for knowledge before
    questions = Question.query.filter(Question.group_id == "knowledge").all()
    question_ids = get_question_ids(questions)
    
    # step 2 : extract and load answer values for knowledge before
    timestamp = datetime.now(timezone.utc)
    for seed, question_id in enumerate(question_ids):
        answer_id = request.form[question_id]

        # chose random value if not answered for debug
        if not answer_id and current_app.config['SKIP_VALID']:
            question = db.session.get(Question, question_id)
            answer_id = question.get_random_answer(seed=seed).answer_id

        new_log = Log(
            timestamp=timestamp,
            user_id=current_user.user_id,
            question_id=question_id,
            answer_id=answer_id,
            phase_id='knowledge_before'
        )
        db.session.add(new_log)
    db.session.commit()

    # step 3 : extract questions for lifestyle
    questions = Question.query.filter(Question.group_id == "lifestyle").all()
    questionnaire_dico = questionnaire(questions)

    return render_template(
        'main/lifestyle.html',
        questionnaire_dico = questionnaire_dico,
        skip_valid=current_app.config['SKIP_VALID'],
    )


@bp.route('/radar_chart', methods=["GET"])
def radar_chart():
    labels = request.args.get('labels').split(',')
    values = [float(value) for value in request.args.get('values').split(',')]

    # Number of variables
    num_vars = len(labels)

    # Compute angle for each category on the radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Make the plot circular by closing the loop
    values += values[:1]
    angles += angles[:1]

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))  # Increased size for better visibility

    # Plot the data
    ax.fill(angles, values, color='b', alpha=0.25)
    ax.plot(angles, values, color='b', linewidth=2)

    # Fix radial axis limits to [0, 1]
    ax.set_ylim(0, 1)

    # Adjust labels and styling
    ax.set_yticklabels([])  # Remove radial labels for a cleaner look
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, ha='center')  # Adjust font size and alignment

    # Improve layout to avoid cutting text
    fig.tight_layout(pad=2)

    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)  # Increased DPI for better quality
    buf.seek(0)

    # Return the image as a response
    return Response(buf.getvalue(), mimetype='image/png')

def get_random_answer_id(question_id, seed):
    question = db.session.get(Question, question_id)
    answer_id = question.get_random_answer(seed=seed).answer_id
    return answer_id

def get_answer_ids(form_data, form_id, question_id, questionnaire_value, seed):
    if form_id=="cursor":
        answer_content = form_data[question_id]
        # Find the answer_id associated to this answer_content
        for a_id, a_content in questionnaire_value:
            if a_content == answer_content:
                answer_id = a_id
        # in skip_mode, choose random cursor value if not answered
        if not answer_id and current_app.config['SKIP_VALID']:
            answer_id = get_random_answer_id(question_id, seed)
        answer_ids = [answer_id]
    
    elif form_id=="checkbox":
        if question_id not in form_data:
            #answer_ids = [get_random_answer_id(question_id, seed)]
            answer_ids = []
        else:
            answer_ids = form_data[question_id]
    
    else: # likert, scroll
        answer_id = form_data[question_id]
        if not answer_id and current_app.config['SKIP_VALID']:
            answer_id = get_random_answer_id(question_id, seed)
        answer_ids = [answer_id]
    return answer_ids

def log_answer_ids(answer_ids, timestamp, question_id, phase_id):
    for answer_id in answer_ids:
        new_log = Log(
            timestamp=timestamp,
            user_id=current_user.user_id,
            question_id=question_id,
            answer_id=answer_id,
            phase_id=phase_id,
        )
        db.session.add(new_log)

def update_features_dico(features_dico, answer_ids, question_id, form_id):
    for answer_id in answer_ids:
        # Find pilote_id for this question and answer_weight 
        pilote_id = Question.query.filter(Question.question_id == question_id).first().pilote_id
        answer_weight = Answer.query.filter(Answer.answer_id == answer_id).first().answer_weight

        if form_id=="scroll":
            pilote_id_nominal_single = pilote_id + "_" + str(float(answer_weight))
            if pilote_id_nominal_single in features_dico:
                features_dico[pilote_id_nominal_single] = 1.0

        elif form_id=="checkbox":
            pilote_id_nominal_multiple = pilote_id + "_" + str(int(answer_weight))
            if pilote_id_nominal_multiple in features_dico:
                features_dico[pilote_id_nominal_multiple] = 1.0
        
        else: # likert, cursor
            if pilote_id in features_dico:
                features_dico[pilote_id] = answer_weight

    return features_dico


@bp.route('/explain', methods=["POST"])
@login_required
def explain():

    # (for checkboxes) Parse form for lifestyle questions with multiple answers
    form_data = form_todict(request.form)

    # Extract list of lifestyle questions in questionnaire
    questions = Question.query.filter(Question.group_id == "lifestyle").all()
    questionnaire_dico = questionnaire(questions)

    # Extract list of lifestyle questions in questionnaire for explainability
    feature_content_dic = get_feature_content_dic()
    displayed_feature_list = list(feature_content_dic.keys())
    questions_explain = Question.query.filter(
        Question.group_id == "lifestyle",
        Question.pilote_id.in_(displayed_feature_list)
    ).all()
    questionnaire_explain_dico = questionnaire(questions_explain)

    # Preallocate features to be populated
    selected_features = current_app.selected_features
    features_dico = {feature: 0.0 for feature in selected_features}
    
    # Record timestamp and set seed
    timestamp = datetime.now(timezone.utc)
    seed = 0

    # If coming from lifestyle.html, then 
    # - extract and log lifestyle answers
    # - create create features based on lifestyle answers
    if form_data['source_page'] == 'lifestyle.html':
        for question_id, (_,_, form_id, questionnaire_value) in questionnaire_dico.items():
            answer_ids = get_answer_ids(form_data, form_id, question_id, questionnaire_value, seed)
            log_answer_ids(answer_ids, timestamp, question_id, 'lifestyle')
            features_dico = update_features_dico(features_dico, answer_ids, question_id, form_id)
            seed += 1
        db.session.commit()
    # If coming from explain_interactive.html, then 
    # - extract and log new answers from explain_interactive answers
    # - extract most recent  answers
    # - create create features based on most recent answers    
    elif form_data['source_page'] == 'explain_interactive.html':
        for question_id, (_,_, form_id, questionnaire_value) in questionnaire_explain_dico.items():
            answer_ids = get_answer_ids(form_data, form_id, question_id, questionnaire_value, seed)
            log_answer_ids(answer_ids, timestamp, question_id, 'explain_interactive')
        most_recent_answers = get_most_recent_answers(current_user.user_id, questions)
        for question_id, answers in most_recent_answers.items():
            answer_ids = [answer_id for answer_id, _, _ in answers]
            _, _, form_id, _ = questionnaire_dico[question_id]
            features_dico = update_features_dico(features_dico, answer_ids, question_id, form_id)
    else:
        raise


    # Predict score from features
    features_df = pd.DataFrame([features_dico])
    best_model = current_app.best_model
    df_y = predict_for_example(
        df_example = features_df,
        model_info = (best_model, selected_features),
        is_df_features = True
    )
    predicted_score = df_y['score_tot_prediction'].iloc[0]

    # Log predicted score
    new_log = Log(
        timestamp=datetime.now(timezone.utc),
        log_type='score_computation',
        log_info=predicted_score,
        user_id=current_user.user_id,
        phase_id='explain'
        )
    db.session.add(new_log)
    db.session.commit()

    # Extract model additional info
    ## 1 - regression coefficient
    coefficients = best_model.named_steps["regressor"].coef_
    feature_coeff_dict = dict(zip(selected_features, coefficients))
    sorted_feature_coeff_dict = dict(sorted(feature_coeff_dict.items(), key=lambda item: item[1], reverse=True))

    ## 2 - values after scaler steps
    X = features_df[selected_features].values
    scaler = best_model.named_steps['scaler']
    X_scaled = scaler.transform(best_model.named_steps['imputer'].transform(X))
    values = X_scaled[0]
    values_coeff_dict = dict(zip(selected_features, values))
    sorted_values_coeff_dict = dict(sorted(values_coeff_dict.items(), key=lambda item: item[1], reverse=True))

    ## 3 - double check model prediction
    predicted_score_bis = sum(coef * val for coef, val in zip(coefficients, values))
    predicted_score_bis += best_model.named_steps["regressor"].intercept_ 

    ## 4 - information about features
    intermediate_predicted_score = 0
    for displayed_feature in displayed_feature_list:
        feature_coeff = feature_coeff_dict[displayed_feature]
        value_coeff = values_coeff_dict[displayed_feature]
        feature_content_dic[displayed_feature].append(feature_coeff_dict[displayed_feature])
        feature_content_dic[displayed_feature].append(values_coeff_dict[displayed_feature])
        intermediate_predicted_score += feature_coeff * value_coeff

    # 5) extract most recent answers (logs) from the 5 lifestyle features
    most_recent_answers_explain = get_most_recent_answers(current_user.user_id, questions_explain)

    # Prepare dictionary with all explainable information
    explain_dic = {
        "predicted_score": round(predicted_score),
        "intermediate_predicted_score": round(intermediate_predicted_score),
        "n_informative": len(feature_content_dic),
        "feature_content_dic": feature_content_dic,
    }
    return render_template(
        f'main/{current_user.condition_id}.html', 
        explain_dic=explain_dic,
        questionnaire_dico=questionnaire_explain_dico,
        predefined_values=most_recent_answers_explain,
    )


@bp.route('/satisfaction', methods=['GET', 'POST'])
@login_required
def satisfaction():
    # step 1 : extract questions for satisfaction
    questions = Question.query.filter(Question.group_id == "satisfaction").all()
    questionnaire_dico = questionnaire(questions)

    return render_template(
        'main/satisfaction.html',
        questionnaire_dico = questionnaire_dico,
        skip_valid=current_app.config['SKIP_VALID'],
        )

@bp.route('/intent', methods=['GET', 'POST'])
@login_required
def intent():
    # step 1 : extract questions ids for satisfaction
    questions = Question.query.filter(Question.group_id == "satisfaction").all()
    question_ids = get_question_ids(questions)
    
    # step 2 : extract and load answer values for satisfaction
    timestamp = datetime.now(timezone.utc)
    for question_id in question_ids:
        answer_id = request.form[question_id]

        # chose random value if not answered for debug
        if not answer_id and current_app.config['SKIP_VALID']:
            question = db.session.get(Question, question_id)
            answer_id = question.get_random_answer(seed=42).answer_id

        new_log = Log(
            timestamp=timestamp,
            user_id=current_user.user_id,
            question_id=question_id,
            answer_id=answer_id,
            phase_id='satisfaction'
        )
        db.session.add(new_log)
    db.session.commit()

    # step 3 : extract questions for intent
    questions = Question.query.filter(Question.group_id == "intent").all()
    questionnaire_dico = questionnaire(questions)

    return render_template(
        'main/intent.html',
        questionnaire_dico = questionnaire_dico,
        skip_valid=current_app.config['SKIP_VALID'],
        )


@bp.route('/knowledge_after', methods=["POST"])
@login_required
def knowledge_after():
    # step 1 : extract questions ids for intent
    questions = Question.query.filter(Question.group_id == "intent").all()
    question_ids = get_question_ids(questions)
    
    # step 2 : extract and load answer values for intent
    timestamp = datetime.now(timezone.utc)
    for question_id in question_ids:
        answer_id = request.form[question_id]

        # chose random value if not answered for debug
        if not answer_id and current_app.config['SKIP_VALID']:
            question = db.session.get(Question, question_id)
            answer_id = question.get_random_answer(seed=42).answer_id
            
        new_log = Log(
            timestamp=timestamp,
            user_id=current_user.user_id,
            question_id=question_id,
            answer_id=answer_id,
            phase_id='intent'
        )
        db.session.add(new_log)
    db.session.commit()
    
    # step 3 : extract questions for knowledge
    questions = Question.query.filter(Question.group_id == "knowledge").all()
    questionnaire_dico = questionnaire(questions)

    return render_template(
        'main/knowledge_after.html',
        questionnaire_dico = questionnaire_dico,
        skip_valid=current_app.config['SKIP_VALID'],
    )


@bp.route('/essaim', methods=["POST"])
@login_required
def essaim():

    # step 1 : extract questions ids for knowledge after
    questions = Question.query.filter(Question.group_id == "knowledge").all()
    question_ids = get_question_ids(questions)
    
    # step 2 : extract and load answer values for knowledge after
    timestamp = datetime.now(timezone.utc)
    for question_id in question_ids:
        answer_id = request.form[question_id]

        # chose random value if not answered for debug
        if not answer_id and current_app.config['SKIP_VALID']:
            question = db.session.get(Question, question_id)
            answer_id = question.get_random_answer().answer_id

        new_log = Log(
            timestamp=timestamp,
            user_id=current_user.user_id,
            question_id=question_id,
            answer_id=answer_id,
            phase_id='knowledge_after'
        )
        db.session.add(new_log)
    db.session.commit()

    # step 3 : extract questions for essaim
    questions = Question.query.filter(Question.group_id == "essaim").all()
    questionnaire_dico = questionnaire(questions)

    return render_template(
        'main/essaim.html',
        questionnaire_dico = questionnaire_dico,
        skip_valid=current_app.config['SKIP_VALID'],
    )

@bp.route('/interac', methods=["POST"])
@login_required
def interac():

    # step 1 : extract questions ids for essaim
    questions = Question.query.filter(Question.group_id == "essaim").all()
    question_ids = get_question_ids(questions)
    
    # step 2 : extract and load answer values for essaim
    timestamp = datetime.now(timezone.utc)
    for question_id in question_ids:
        answer_id = request.form[question_id]

        # chose random value if not answered for debug
        if not answer_id and current_app.config['SKIP_VALID']:
            question = db.session.get(Question, question_id)
            answer_id = question.get_random_answer().answer_id
            
        new_log = Log(
            timestamp=timestamp,
            user_id=current_user.user_id,
            question_id=question_id,
            answer_id=answer_id,
            phase_id='essaim'
        )
        db.session.add(new_log)
    db.session.commit()

    # step 3 : extract questions for interac
    questions = Question.query.filter(Question.group_id == "interac").all()
    questionnaire_dico = questionnaire(questions)

    return render_template(
        'main/interac.html',
        questionnaire_dico = questionnaire_dico,
        skip_valid=current_app.config['SKIP_VALID'],
    )

@bp.route('/merci', methods=["POST"])
@login_required
def merci():
    
    # Extract question id for interac (only one question so we take the first)
    question_id = Question.query.filter(Question.group_id == "interac").first().question_id

    # New log for finished time
    timestamp = datetime.now(timezone.utc)
    new_log_finished = Log(
            timestamp=timestamp,
            log_type='finished',
            user_id=current_user.user_id,
            question_id=question_id,
            phase_id='interac'
        )
    db.session.add(new_log_finished)

    # Find userpii
    user_pii = UserPII.query.get(current_user.user_id)
    # Save interac email (not in log because sensitive data)
    user_pii.interac_email = request.form[question_id]

    db.session.commit()
    
    return render_template('main/merci.html')

