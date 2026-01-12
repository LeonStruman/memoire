### 1. explain_quantitative

#### 1.1) LLM - interaction 1

I need to provide an high-quality explanation for users using a tool that aims to predict their mental health based on their answer on lifestyle questions.

I would like to provide an affordable quantitative explanation of the predicted mental health score. 

Here are more details about the algorithm. 

The mental health score is a value varying from 0 to 100, with 100 representing excellent mental health and 0 representing poor mental health. 

This prediction is based on the individual's responses to a series of lifestyle and socio-demographic questions. Using pilot data from 2,000 participants, we experimented with various machine learning models and found that linear regression performed best, with three key questions selected through feature selection. The mental health score is predicted by the equation: y = a1 * x1 + a2 * x2 + a3 * x3 where x1, x2, ..., x20 represent the responses to 20 normalized questions, and y is the predicted mental health score. 


The top 3 most informative questions are :
x1: How many friends do you have?
x2: How many hours of sleep do you get?
x3: How often do you exercise? 

We found that a1 = 70, a2 = 10, and a3 = 10 gives the best fit.

For the current user, the predicted mental health score is 45.
Given that the user's responses are x1 = 0.5, x2 = 0.1, and x3 = 0.2, there are 38 points that are already explained by those three factors. 

In the context of explainable AI, please provide to the user (who is not an expert in AI) an explanation about why his score is 45.


#### 1.2) LLM - interaction 2

I now want the same text, to be pasted in a html file between paragraph quotes (<p>, </p>), where some of the values will be replaced by variables send by a "render_template" command in a routes.py file. A variable can be injected with the syntax {{ my_variable }}. 

The information to replace dynamically are :
- the predicted score (45)
- the number of most informative questions
- the content of the most informative questions
- the coefficient of the most informative questions
- the intermediate predicted score obtained by summing the most informative questions (38)

This is injected via a dictionary variable explain_dic with the following corresponding keys:
- "predicted_score" (float)
- "n_informative" (int)
- "informative_questions_content_dic" (dictionary mapping question ids to a tuple (question_content, question_details, question_coefficient)). The variable question_details should not be used here and is kept for a content-based information later on.
- "intermediate_predicted_score" (float)


### 2. explain_textual

#### 2.1) LLM - interaction 1

I need to provide an high-quality explanation for users using a tool that aims to predict their mental health based on their answer on lifestyle questions.

I would like to provide an affordable textual explanation of the predicted mental health score, without any mathematics or numbers.

Here are more details about the algorithm. 

The mental health score is a value varying from 0 to 100, with 100 representing excellent mental health and 0 representing poor mental health. 

This prediction is based on the individual's responses to a series of lifestyle and socio-demographic questions. Using pilot data from 2,000 participants, we experimented with various machine learning models and found that linear regression performed best, with three key questions selected through feature selection. The mental health score is predicted by the equation: y = a1 * x1 + a2 * x2 + a3 * x3 where x1, x2, ..., x20 represent the responses to 20 normalized questions, and y is the predicted mental health score. 


The top 3 most informative questions are :
x1: How many friends do you have?
x2: How many hours of sleep do you get?
x3: How often do you exercise? 

We found that a1 = 70, a2 = 10, and a3 = 10 gives the best fit.

For the current user, the predicted mental health score is 45.
Given that the user's responses are x1 = 0.5, x2 = 0.1, and x3 = 0.2, there are 38 points that are already explained by those three factors. 

In the context of explainable AI, please provide to the user (who is not an expert in AI) an explanation about why his score is 45.

#### 2.2) LLM - interaction 2

I now want the same text, to be pasted in a html file between paragraph quotes (<p>, </p>), where some of the values will be replaced by variables send by a "render_template" command in a routes.py file. A variable can be injected with the syntax {{ my_variable }}. 

The information to replace dynamically are :
- the predicted score (45)
- the number of most informative questions
- the content of the most informative questions
- a more detailed information on the content of the most informative questions


This is injected via a dictionary variable explain_dic with the following corresponding keys:
- "predicted_score" (float)
- "n_informative" (int)
- "informative_questions_content_dic" (dictionary mapping question ids to a tuple (question_content, question_details, question_coefficient)). The variable question_coefficient should not be used here and is kept for a quantitative-based information later on.

#### 2.3) LLM - interaction 3

In my application, I am asking the following question : 
            'During the past seven days, how would you rate your sleep quality overall?'

Please explain me in a short paragraph why this information is relevant to predict my mental health score, in .txt format

### 3. explain_visual

#### 3.1) LLM - interaction 1

I need to provide an high-quality explanation for users using a tool that aims to predict their mental health based on their answer on lifestyle questions.

I would like to provide an affordable visual explanation of the predicted mental health score, displaying the importance of the most important features on a radarchart. I do not want a mathematical explanation. I already produced the radarchart and I need you to provide me with the textual explanation that goes with the chart. 

Here are more details about the algorithm. 

The mental health score is a value varying from 0 to 100, with 100 representing excellent mental health and 0 representing poor mental health. 

This prediction is based on the individual's responses to a series of lifestyle and socio-demographic questions. Using pilot data from 2,000 participants, we experimented with various machine learning models and found that linear regression performed best, with three key questions selected through feature selection. The mental health score is predicted by the equation: y = a1 * x1 + a2 * x2 + a3 * x3 where x1, x2, ..., x20 represent the responses to 20 normalized questions, and y is the predicted mental health score. 

The top 3 most informative questions are :
x1: How many friends do you have?
x2: How many hours of sleep do you get?
x3: How often do you exercise? 

We found that a1 = 70, a2 = 10, and a3 = 10 gives the best fit.

For the current user, the predicted mental health score is 45.
Given that the user's responses are x1 = 0.5, x2 = 0.1, and x3 = 0.2, there are 38 points that are already explained by those three factors. 

The radarchart is displaying the value of x1, x2 and x3. 

In the context of explainable AI, please provide to the user (who is not an expert in AI) an explanation about why his score is 45.

#### 3.2) LLM - interaction 2


