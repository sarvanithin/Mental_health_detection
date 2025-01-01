import pickle
import logging
import cv2
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
import mysql.connector
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import joblib
import numpy as np
import openai
import pandas as pd
import sklearn
from flask import Flask, render_template, request, jsonify
import requests
from flask_cors import cross_origin
from sklearn.preprocessing import RobustScaler
from pymongo import MongoClient
import openai
from flask import Flask, render_template, request, jsonify
from urllib.parse import quote_plus
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
from tensorflow import keras
import mysql.connector
from datetime import datetime
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

app = Flask(__name__)
# @app.route('/')
# def index():
#     return render_template('index1.html')
# MySQL Configuration
app.config['MYSQL_HOST'] = 'your_mysql_host'
app.config['MYSQL_USER'] = 'your_mysql_user'
app.config['MYSQL_PASSWORD'] = 'your_mysql_password'
app.config['MYSQL_DB'] = 'your_mysql_database'
mysql = MySQL(app)

# Validate Email Format
def is_valid_email(email):
    email_regex = r'^\S+@\S+\.\S+$'
    return re.match(email_regex, email)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    if not is_valid_email(email):
        return render_template('login.html', error='Invalid email format')

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()

    if user and user['password'] == password:
        # Successful login - you may implement session handling here
        # Redirect to index1.html upon successful login
        return redirect(url_for('index1'))
    else:
        return render_template('login.html', error='Invalid email or password')

@app.route('/index1')
def index1():
    # Render the index1.html page after successful login
    return render_template('index1.html')
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if not is_valid_email(email):
            return render_template('signup.html', error='Invalid email format')

        cur = mysql.connection.cursor()
        # Check if the user already exists
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            cur.close()
            return render_template('signup.html', error='User with this email already exists. Please choose another email.')

        # Insert the new user into the database
        cur.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, password))
        mysql.connection.commit()
        cur.close()

        # Redirect to login page after successful signup
        return redirect(url_for('login'))

    return render_template('signup.html')

#
#                                   BLOG
#
@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

#about us
@app.route('/about_us')
def about_us():
    return render_template('aboutus.html')

# username creation
# MySQL database configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '040503',
    'database': 'mental_health',
}

# Function to create a table if it doesn't exist
def create_table(username):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    # Define column names
    columns = [
        'middle_age', 'voice_recog', 'face_recog', 'food_analysis',
        'detect_songs', 'ocd', 'social_anxiety', 'anxiety_and_stress',
        'depression_pred', 'bipolar_disorder','predicted_outcome'
    ]

    # Construct SQL query to create a table
    create_table_query = f"CREATE TABLE IF NOT EXISTS {username} (id INT AUTO_INCREMENT PRIMARY KEY, {', '.join([f'{col} INT' for col in columns])})"

    cursor.execute(create_table_query)
    # Construct SQL query to insert a default row
    # Check if the table is empty
    check_empty_query = f"SELECT COUNT(*) FROM {username}"
    cursor.execute(check_empty_query)
    result = cursor.fetchone()[0]

    if result == 0:
        # Insert default row if the table is empty
        insert_default_row_query = f"INSERT INTO {username} ({', '.join(columns)}) VALUES ({', '.join(['0' for _ in columns])})"
        cursor.execute(insert_default_row_query)

    connection.commit()

    cursor.close()
    connection.close()

@app.route('/create_table', methods=['POST'])
def create_table_route():
    username = request.form['username']
    
    # Create a table for the given username
    create_table(username)

    return jsonify({'success': True})
app.secret_key = 'akhil123teja456reddy789'
# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    try:
        username = request.form['username']
        session['username'] = username  # Store username in session
        create_table(username)  # Create a table for the given username
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error processing form data: {e}")
        return jsonify({'success': False, 'error': str(e)})


#######################################################################################################################################
#                                            AGE RELATED MENTAL HEALTH PREDICTION                                                     #
#######################################################################################################################################
#                                                      ADULT                                                                              #

#######################################################################################################################################
#                                                     Employeee                                                                       #
with open('E:/Mental Health Detection Project/mental health site/lgr_model_emp', 'rb') as model_file:
    model_emp = pickle.load(model_file)

@app.route('/emp_pred')
def emp_pred():
    username = session.get('username')
    return render_template('emp_pred.html',username=username)

@app.route('/predict_mental_health', methods=['POST'])
def predict_mental_health():
    username = session.get('username')
    # Collect data from HTML form
    self_employed = float(request.form['self_employed'])
    family_history = float(request.form['family_history'])
    work_interfere = float(request.form['work_interfere'])
    remote_work = float(request.form['remote_work'])
    care_options = float(request.form['care_options'])
    wellness_program = float(request.form['wellness_program'])
    anonymity = float(request.form['anonymity'])
    leave = float(request.form['leave'])
    mental_health_consequence = float(request.form['mental_health_consequence'])
    phys_health_consequence = float(request.form['phys_health_consequence'])
    coworkers = float(request.form['coworkers'])
    supervisor = float(request.form['supervisor'])
    mental_health_interview = float(request.form['mental_health_interview'])
    phys_health_interview = float(request.form['phys_health_interview'])
    mental_vs_physical = float(request.form['mental_vs_physical'])
    obs_consequence = float(request.form['obs_consequence'])

    # Create feature array
    features = np.array([self_employed, family_history, work_interfere, remote_work, care_options, wellness_program,
                         anonymity, leave, mental_health_consequence, phys_health_consequence, coworkers, supervisor,
                         mental_health_interview, phys_health_interview, mental_vs_physical, obs_consequence])

    # Make prediction
    prediction_emp = model_emp.predict([features])[0]

    # Convert prediction to 'No' or 'Yes'
    prediction_result_emp = 'No' if prediction_emp == 0 else 'Yes'

    update_emp_value(username,prediction_emp)

    return render_template('emp_pred.html', prediction=prediction_result_emp)

def update_emp_value(username, prediction_emp):
    logging.debug(f"Updating emp value for username: {username}")
    connection = mysql.connector.connect(**db_config, autocommit=True)
    cursor = connection.cursor()

    # Construct SQL query to update the "depression_pred" column for the given username
    update_query = f"UPDATE {username} SET middle_age = {(prediction_emp)}"
    
    logging.debug(f"SQL Query: {update_query} WHERE id =1 ")  # Assuming id = 1 for simplicity

    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()
    logging.debug("Update successful")


#
#                                   CHATBOT
#
# Set your OpenAI API key
# MongoDB configuration
# username = 'yvakhilteja2003'
# password = 'AKHILteja2003'
# cluster_url = 'cluster0.deyd5n5.mongodb.net'

# Encode username and password
# encoded_username = quote_plus(username)
# encoded_password = quote_plus(password)

mongo_client = MongoClient('mongodb+srv://yvakhilteja2003:AKHILteja2003@cluster0.ebsadfl.mongodb.net/?retryWrites=true&w=majority')
db = mongo_client['mental_health']
collection = db['user_data']

# Example prompts
prompt = "Provide information on mental health detection."
system_message = "You are an assistant specialized in providing information on mental health detection."

def save_user_question(question):
    # Save user question to MongoDB
    collection.insert_one({"question": question})

# Set predefined prompts
predefined_prompts = [
    "Digital detox", "Therapeutic communication", "Mind-body interventions", "Psychiatric rehabilitation",
    "Recovery-oriented care", "Dual diagnosis treatment", "Mental health assessment", "Peer counseling",
    "Emotional intelligence training", "Mental health apps", "Trauma-informed care", "Substance use prevention",
    "Dialectical behavior therapy (DBT)", "Mental health community", "Cognitive distortions", "Holistic therapies",
    "Emotional well-being at work", "Coping skills development", "Wellness retreats", "Online therapy platforms",
    "Mental health screenings", "Mental health workshops", "Self-help strategies", "Mental health and nutrition",
    "Emotional resilience", "Stigma reduction", "Mindfulness-based stress reduction (MBSR)",
    "Crisis intervention training", "Mental health forums", "Mental health and technology", "Teletherapy",
    "Mindfulness meditation", "Psychosocial factors", "Mental health advocacy", "Emotional regulation",
    "Mental health resources", "Dual diagnosis", "Emotional support animals", "Neurotransmitters",
    "Therapeutic techniques", "Holistic healing", "Mental health initiatives", "Expressive arts therapy",
    "Mental health parity", "Neuroplasticity", "Compassion fatigue", "Stress management", "Self-care practices",
    "Mental health policy", "Resilience training", "Lifestyle changes", "Mental health promotion", "Coping mechanisms",
    "Mental health first aid", "Positive affirmations", "Crisis helpline", "Sleep hygiene", "Emotional boundaries",
    "Workplace mental health", "Mental health research", "Social isolation", "Burnout", "Substance abuse",
    "Eating disorders", "Sleep disorders", "Psychiatric medication", "Mood swings", "Trauma", "Grief and loss",
    "Cognitive-behavioral therapy (CBT)", "Mind-body connection", "Emotional intelligence", "Positive psychology",
    "Neurodiversity", "Work-life balance", "Mental health education", "Wellness programs", "Crisis intervention",
    "Support groups", "Holistic health", "Depression", "Anxiety", "Stress", "Bipolar disorder", "Schizophrenia",
    "PTSD (Post-Traumatic Stress Disorder)", "OCD (Obsessive-Compulsive Disorder)", "ADHD (Attention-Deficit/Hyperactivity Disorder)",
    "Self-esteem", "Therapy", "Counseling", "Psychotherapy", "Mental wellness", "Mindfulness", "Coping strategies",
    "Emotional well-being", "Mental health stigma", "Peer support", "Resilience", "Mental health awareness","bye"
]

# Set your OpenAI API key
openai.api_key = "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"

def generate_response(messages):
    user_input = messages[-1]["content"]

    if any(prompt.lower() in user_input.lower() for prompt in predefined_prompts):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]

        temperature = 0.5
        top_p = 0.9

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=150,
            top_p=top_p
        )

        output = response['choices'][0]['message']['content'].strip()
        return output
    else:
        # Save user question to MongoDB if not in predefined prompts
        save_user_question(user_input)
        return f"Error: The entered context does not contain any of the required mental health information."


# Example conversation initialization
conversation = [
    {"role": "system", "content": system_message},
]

# @app.route('/')
# def index():
#     return render_template('index1.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.form['user_input']

    # Add user message to the conversation
    conversation.append({"role": "user", "content": user_input})

    # Generate response based on the entire conversation
    assistant_response = generate_response(conversation)

    # Add assistant message to the conversation
    conversation.append({"role": "assistant", "content": assistant_response})

    return jsonify({'assistant_response': assistant_response})

##################################################################################
#                                     FOOD Screening                             #
##################################################################################

#food screening
#--------------------------------------------------------------------------------
# Set up MySQL connection
db_connection = mysql.connector.connect(
    host='127.0.0.1',
    user='root',
    password='040503',
    database='mental_health',
    autocommit=True
)

# Create a cursor object to execute SQL queries
cursor = db_connection.cursor()

# SQL query to create the table
create_table_query = """
CREATE TABLE IF NOT EXISTS nutrition_value (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255),
    calories INT,
    protein FLOAT,
    fat FLOAT,
    carbohydrates FLOAT,
    fiber FLOAT,
    sugar FLOAT
)
"""
cursor.execute(create_table_query)


@app.route('/food_analysis')
def food_analysis1():
    return render_template('food_main.html')


@app.route('/calculate', methods=['POST'])
def calculate_nutrition():
    try:
        # Retrieve the form data from the request
        title = request.form.get('title')
        ingredients = request.form.get('ingredients')
        servings = request.form.get('servings')

        # Prepare the ingredient list with servings information
        ingr_list = ingredients.split('\n')
        ingr_with_servings = [f'{servings} {ingr}' for ingr in ingr_list]

        # Make a POST request to the Edamam API
        response = requests.post('https://api.edamam.com/api/nutrition-details', params={
            'app_id': '2affd363',
            'app_key': 'f24fccaef17eceb9d60b11d2efe9a1a4',
        }, json={
            'title': title,
            'ingr': ingr_with_servings,
        })

        # Print the API request URL and response content
        print(response.url)
        print(response.content)

        # Extract the nutritional information from the API response
        if response.status_code == 200:
            nutrition = response.json()

            # Insert data into the nutrition table
            insert_query = """
            INSERT INTO nutrition_value (title, calories, protein, fat, carbohydrates, fiber, sugar)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            data = (
                title,
                nutrition['calories'],
                nutrition['totalNutrients']['PROCNT']['quantity'],
                nutrition['totalNutrients']['FAT']['quantity'],
                nutrition['totalNutrients']['CHOCDF']['quantity'],
                nutrition['totalNutrients']['FIBTG']['quantity'],
                nutrition['totalNutrients']['SUGAR']['quantity']
            )
            cursor.execute(insert_query, data)
            print("data inserted",insert_query)

            # Commit the changes to the database
            db_connection.commit()
            print("changes committed in database",data)

            # # After db_connection.commit()
            # cursor.close()
            # db_connection.close()


            return render_template('food_result.html',title=title, nutrition=nutrition)
        else:
            return render_template('food_result.html', error='Failed to retrieve nutrition information')
    except Exception as e:
        print(e)
        return render_template('food_result.html', error='Failed to calculate nutrition')

#
#                                   SOCIAL ANEXITY PREDICTION
#
# Load the pickled model
# Database connection
# Function to establish a database connection
def get_db_connection():
    return mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='040503',
        database='mental_health'
    )
# Create a table to store the data
table_creation_query_social = """
CREATE TABLE IF NOT EXISTS social_anxiety_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age FLOAT,
    gender INT,
    family_history INT,
    occupation INT,
    atf FLOAT,
    tkf FLOAT,
    def FLOAT,
    smf FLOAT,
    daf FLOAT,
    hr INT,
    sw INT,
    tr INT,
    dr INT,
    br INT,
    ck INT,
    cp INT,
    ns INT,
    dz INT,
    ur INT,
    ub INT,
    md INT,
    tg INT,
    prediction INT
);
"""
# Load the pickled model
with open('E:/Mental Health Detection Project/mental health site/stc_model_social', 'rb') as model_file:
    model_social = pickle.load(model_file)

@app.route('/social', methods=['GET', 'POST'])
def social_anxiety():
    username = session.get('username')
    if request.method == 'POST':
        try:
            # Retrieve form data
            age = float(request.form.get('age'))
            gender = int(request.form.get('gender'))
            family_history = int(request.form.get('familyHistory'))
            occupation = int(request.form.get('occupation'))

            atf = float(request.form.get('atf'))
            tkf = float(request.form.get('tkf'))
            def_ = float(request.form.get('def'))
            smf = float(request.form.get('smf'))
            daf = float(request.form.get('daf'))
            hr = int(request.form.get('hr'))
            sw = int(request.form.get('sw'))
            tr = int(request.form.get('tr'))
            dr = int(request.form.get('dr'))
            br = int(request.form.get('br'))
            ck = int(request.form.get('ck'))
            cp = int(request.form.get('cp'))
            ns = int(request.form.get('ns'))
            dz = int(request.form.get('dz'))
            ur = int(request.form.get('ur'))
            ub = int(request.form.get('ub'))
            md = int(request.form.get('md'))
            tg = int(request.form.get('tg'))

            # Prepare the input data for prediction
            input_data = np.array([[atf, tkf, def_, smf, daf, hr, sw, tr, dr, br, ck, cp, ns, dz, ur, ub, md, tg, age, gender, family_history, occupation]])

            # Make a prediction
            prediction_social = model_social.predict(input_data)
            db_prediction_social = 1 if prediction_social[0] == 1 else 0

            # Insert the data into the database
            db_connection = get_db_connection()
            cursor = db_connection.cursor()

            insert_query_social = """
            INSERT INTO social_anxiety_predictions 
            (age, gender, family_history, occupation, atf, tkf, def, smf, daf, hr, sw, tr, dr, br, ck, cp, ns, dz, ur, ub, md, tg, prediction) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            data_values_social = (age, gender, family_history, occupation, atf, tkf, def_, smf, daf, hr, sw, tr, dr, br, ck, cp, ns, dz, ur, ub, md, tg, int(prediction_social[0]))

            cursor.execute(insert_query_social, data_values_social)
            db_connection.commit()

            # Close the cursor and database connection
            cursor.close()
            db_connection.close()

            update_social_value(username,db_prediction_social)

            # Return the predicted value
            return f"The predicted value is: {prediction_social[0]} and data inserted successfully!"

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template('social.html',username=username)

def update_social_value(username, db_prediction_social):
    logging.debug(f"Updating social_pred value for username: {username}")
    connection = mysql.connector.connect(**db_config, autocommit=True)
    cursor = connection.cursor()

    # Construct SQL query to update the "depression_pred" column for the given username
    update_query = f"UPDATE {username} SET social_anxiety = {(db_prediction_social)}"
    
    logging.debug(f"SQL Query: {update_query} WHERE id =1 ")  # Assuming id = 1 for simplicity

    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()
    logging.debug("Update successful")
#
#                                   SENTIMENT PREDICTION
#

# Load the VADER model
sid = joblib.load('E:/Mental Health Detection Project/mental health site/vader_model.joblib')

@app.route('/passage_pred')
def sentiment_analysis():
    username = session.get('username')
    return render_template('passage.html',username=username)

@app.route('/passage_result', methods=['POST'])
def sentiment_pred():
    username = session.get('username')
    if request.method == 'POST':
        text = request.form['text']
        #username = request.form['username']
        sentiment_score = sid.polarity_scores(text)['compound']
        sent_score_data = 0 if sentiment_score >= 0.05 else (2 if sentiment_score <= -0.05 else 1)
        sentiment = 'positive' if sentiment_score >= 0.05 else ('negative' if sentiment_score <= -0.05 else 'neutral')
        update_sentiment_value(username,sent_score_data)
        return render_template('passage.html', text=text, sentiment=sentiment)

logging.basicConfig(level=logging.DEBUG)

def update_sentiment_value(username, sent_score_data):
    logging.debug(f"Updating sentiment value for username: {username}")
    connection = mysql.connector.connect(**db_config, autocommit=True)
    cursor = connection.cursor()

    # Construct SQL query to update the "depression_pred" column for the given username
    update_query = f"UPDATE {username} SET depression_pred = {(sent_score_data)}"
    
    logging.debug(f"SQL Query: {update_query} WHERE id =1 ")  # Assuming id = 1 for simplicity

    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()
    logging.debug("Update successful")
#
#                           PREDICTION THROUGH SONGS
#
model = pickle.load(open("E:/Mental Health Detection Project/mental health site/stc_model", "rb"))


@app.route("/songs")
@cross_origin()
def songs():
    return render_template("songs1.html")
    

@app.route("/songs_predict", methods = ["GET", "POST"])
@cross_origin()
def songs_predict():
    username = session.get('username')
    if request.method == "POST":

        # Total Stops
        Hours_per_day = int(request.form["Hours_per_day"])
        
        # print(Total_stops)
        

        Frequency_Classical = int(request.form["Frequency_Classical"])

        Frequency_Country = int(request.form["Frequency_Country"])

        Frequency_EDM = int(request.form["Frequency_EDM"])

        Frequency_Folk = int(request.form["Frequency_Folk"])

        Frequency_Gospel = int(request.form["Frequency_Gospel"])

        Frequency_Hip_hop = int(request.form["Frequency_Hip_hop"])

        Frequency_Jazz = int(request.form["Frequency_Jazz"])

        Frequency_K_pop = int(request.form["Frequency_K_pop"])

        Frequency_Latin = float(request.form["Frequency_Latin"])

        Frequency_Lofi = float(request.form["Frequency_Lofi"])

        Frequency_Metal = int(request.form["Frequency_Metal"])

        Frequency_Pop = int(request.form["Frequency_Pop"])

        Frequency_RB= int(request.form["Frequency_RB"])

        Frequency_Rock = int(request.form["Frequency_Rock"])

        Frequency_Video_game_music = int(request.form["Frequency_Video_game_music"])

        Insomnia = int(request.form["Insomnia"])

        OCD = int(request.form["OCD"])

        Music_effects = int(request.form["Music_effects"])

        Fav_genre=request.form['Fav_genre']
        if(Fav_genre=='Fav_genre_Classical'):
            Fav_genre_Classical = 1
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0
            

        elif (Fav_genre=='Fav_genre_Country'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 1
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_EDM'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 1
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0
            
        elif (Fav_genre=='Fav_genre_Folk'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 1
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0
            
        elif (Fav_genre=='Fav_genre_Hip_hop'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 1
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Jazz'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=1
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_K_pop'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 1
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Latin'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 1
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Lofi'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 1
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Metal'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 1
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Pop'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 1
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_RB'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 1
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Rap'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 1
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Rock'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 1
            Fav_genre_Video_game_music= 0
            
        else:
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 1

        Foreign_languages = request.form["Foreign_languages"]
        if (Foreign_languages == 'Foreign_languages_Yes'):
            Foreign_languages_Yes = 1
            Foreign_languages_No = 0
            

        else:
            Foreign_languages_Yes = 0
            Foreign_languages_No = 1

        
            
        prediction=model.predict([[
            Hours_per_day,
            Fav_genre_Classical,
            Fav_genre_Country,
            Fav_genre_EDM,
            Fav_genre_Folk,
            Fav_genre_Hip_hop,
            Fav_genre_Jazz,
            Fav_genre_K_pop,
            Fav_genre_Latin,
            Fav_genre_Lofi,
            Fav_genre_Metal,
            Fav_genre_Pop,
            Fav_genre_RB,
            Fav_genre_Rap,
            Fav_genre_Rock,
            Fav_genre_Video_game_music,
            Foreign_languages_No,
            Foreign_languages_Yes,
            Frequency_Classical,
            Frequency_Country,
            Frequency_EDM,
            Frequency_Folk,
            Frequency_Gospel,
            Frequency_Hip_hop,
            Frequency_Jazz,
            Frequency_K_pop,
            Frequency_Latin,
            Frequency_Lofi,
            Frequency_Metal,
            Frequency_Pop,
            Frequency_RB,
            Frequency_Rock,
            Frequency_Video_game_music,
            Insomnia,
            OCD,
            Music_effects,

        ]])

        output_songs=round(prediction[0],2)
        update_songs_value(username,output_songs)

        return render_template('songs1.html',prediction_text="Your mental health is {}, Where 1 is mental health not detected, 0 is mental health detected".format(output_songs))


    return render_template("songs1.html",username=username)

def update_songs_value(username, output_songs):
    logging.debug(f"Updating songs value for username: {username}")
    connection = mysql.connector.connect(**db_config, autocommit=True)
    cursor = connection.cursor()

    # Construct SQL query to update the "depression_pred" column for the given username
    update_query = f"UPDATE {username} SET detect_songs = {(output_songs)}"
    
    logging.debug(f"SQL Query: {update_query} WHERE id =1 ")  # Assuming id = 1 for simplicity

    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()
    logging.debug("Update successful")

#
#                               MENTAL HEALTH USING FACIAL EXPRESSION RECOGNITION
#
model_face = tf.keras.models.load_model("E:/Mental Health Detection Project/face_reg/facial_expression_model1.h5")

# Function to detect facial expression and store in the database
def detect_expression_and_store(img_data):
    username = session.get('username')
    try:
        # Convert base64 image data to OpenCV format
        img_data = base64.b64decode(img_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise Exception("Failed to decode image data")

        # Print some information for debugging
        print(f"Image shape: {img.shape}")
        print(f"Image type: {img.dtype}")

        # Preprocess the image
        img_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_array = cv2.resize(img_array, (48, 48))
        img_array = img_array.reshape(1, 48, 48, 1) / 255.0

        # Make a prediction
        predictions = model_face.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Map predicted class to emotion
        #emotions = ['sadness','contempt','happiness','anger','disgust','neutrality','fear','suprise']
        sentiments_list = ['anger','contempt', 'disgust','fear','happiness','neutrality','sadness','suprise']
        #detected_emotion_face = emotions[predicted_class]
        detected_emotion_face = sentiments_list[predicted_class]

        
          # Replace 'happiness' with the actual sentiment you want to check

        #sentiment_index = sentiments_list.index(detected_emotion_face)

        sentiment_face = (
            0 if detected_emotion_face == 'sadness' else
            1 if detected_emotion_face == 'anger' else
            2 if detected_emotion_face == 'disgust' else
            3 if detected_emotion_face == 'fear' else
            4 if detected_emotion_face == 'contempt' else
            5 if detected_emotion_face == 'neutrality' else
            6 if detected_emotion_face == 'suprise' else
            7  
        )

        #print(sentiment)


        # Store in the database
        store_in_database(detected_emotion_face)
        update_face_value(username,sentiment_face)

        return detected_emotion_face
    except Exception as e:
        print(f"Error in detect_expression_and_store: {e}")
        return "Error"


# Function to store detected emotion in the database
def store_in_database(emotion):
    connection = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="040503",
        database="mental_health"
    )

    cursor = connection.cursor()
    query = "INSERT INTO emotions_table (emotion) VALUES (%s)"
    cursor.execute(query, (emotion,))
    connection.commit()

    cursor.close()
    connection.close()

# Web routes
@app.route('/face_recog')
def face_recog1():
    return render_template('face.html')

@app.route('/capture', methods=['POST'])
def capture():
    img_data = request.form['img_data']
    emotion = detect_expression_and_store(img_data)
    return jsonify({'emotion': emotion})

def update_face_value(username, sentiment_face):
    logging.debug(f"Updating face value for username: {username}")
    connection = mysql.connector.connect(**db_config, autocommit=True)
    cursor = connection.cursor()

    # Construct SQL query to update the "depression_pred" column for the given username
    update_query = f"UPDATE {username} SET face_recog = {(sentiment_face)}"
    
    logging.debug(f"SQL Query: {update_query} WHERE id =1 ")  # Assuming id = 1 for simplicity

    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()
    logging.debug("Update successful")

#
#                               MENTAL HEALTH USING VOICE RECOGNITION
#
# Load the trained model
model_voice = keras.models.load_model('E:/Mental Health Detection Project/voice_detec/voice_emotion_model1.h5')  # Replace 'your_model_path' with the actual path to your trained model

# MySQL database configuration
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="040503",
    database="mental_health"
)

cursor = db.cursor()

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Function to convert any audio or video file to WAV format
def convert_to_wav(file_path):
    try:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        print(f"Original file extension: {file_extension}")

        if file_extension in ['.wav', '.mp3', '.ogg', '.flac']:  # Audio file formats
            audio = AudioSegment.from_file(file_path)
            wav_filename = os.path.splitext(os.path.basename(file_path))[0] + '.wav'
            wav_path = os.path.join(app.root_path, 'voice_detec', 'recordings', wav_filename)
            audio.export(wav_path, format='wav')
            return wav_path

        elif file_extension in ['.mp4', '.mkv', '.avi']:  # Video file formats
            clip = VideoFileClip(file_path)
            wav_filename = os.path.splitext(os.path.basename(file_path))[0] + '.wav'
            wav_path = os.path.join(app.root_path, 'voice_detec', 'recordings', wav_filename)
            clip.audio.write_audiofile(wav_path)
            return wav_path

        elif file_extension == '.m4a':  # Additional handling for .m4a files
            audio = AudioSegment.from_file(file_path, format='m4a')
            wav_filename = os.path.splitext(os.path.basename(file_path))[0] + '.wav'
            wav_path = os.path.join(app.root_path, 'voice_detec', 'recordings', wav_filename)
            audio.export(wav_path, format='wav')
            return wav_path

        else:
            raise ValueError("Unsupported file format")

    except CouldntDecodeError as e:
        print(f"Error converting to WAV: {e}")
        return None

# Function to extract MFCC features from the audio file
# Function to extract MFCC features from the audio file
def extract_mfcc(file_path):
    try:
        print(f"Attempting to load file: {file_path}")
        y, sr = librosa.load(file_path, sr=16000)
        print(f"Loaded file successfully")
        if len(y) == 0:
            raise ValueError("Empty audio file")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
        mfccs = resize_array(mfccs)
        mfccs = (mfccs - 0.5) / 0.5  # Apply the same scaling as during training
        mfccs = mfccs[..., None]
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None



# Function to resize the 2D arrays
def resize_array(array):
    new_matrix = np.zeros((30, 150))
    for i in range(30):
        for j in range(150):
            try:
                new_matrix[i][j] = array[i][j]
            except IndexError:
                pass
    return new_matrix

# Function to predict emotion
def predict_emotion(file_path):
    mfccs = extract_mfcc(file_path)

    if mfccs is not None:
        prediction = model_voice.predict(np.array([mfccs]))
        emotion_id = np.argmax(prediction)
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        predicted_emotion = emotions[emotion_id]
        return predicted_emotion
    else:
        return "Error"

# Function to save the recording information in the database
def save_to_database(emotion, timestamp):
    query = "INSERT INTO voice_recog (emotion, timestamp) VALUES (%s, %s)"
    values = (emotion, timestamp)
    cursor.execute(query, values)
    db.commit()

# Route for the main page
@app.route('/voice')
def voice_index():
    return render_template('voice_index.html')

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload():
    username = session.get('username')
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Ensure 'recordings' directory exists
        recordings_dir = os.path.join(app.root_path, 'voice_detec', 'recordings')
        os.makedirs(recordings_dir, exist_ok=True)

        # Save the uploaded file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Use '_' instead of ':'
        filename = os.path.join(recordings_dir, f"{timestamp}_{secure_filename(file.filename)}")

        file.save(filename)

        # Predict emotion
        emotion_voice = predict_emotion(filename)

        sentiment_voice = (
            0 if emotion_voice == 'sad' else
            1 if emotion_voice == 'angry' else
            2 if emotion_voice == 'disgust' else
            3 if emotion_voice == 'fear' else
            4 if emotion_voice == 'neutral' else
            5 if emotion_voice == 'surprise' else
            6  
        )

        # Save to database
        save_to_database(emotion_voice, timestamp)
        update_voice_value(username,sentiment_voice)

        return render_template('voice_result.html', emotion=emotion_voice,username=username)

def update_voice_value(username, sentiment_voice):
    logging.debug(f"Updating voice value for username: {username}")
    connection = mysql.connector.connect(**db_config, autocommit=True)
    cursor = connection.cursor()

    # Construct SQL query to update the "depression_pred" column for the given username
    update_query = f"UPDATE {username} SET voice_recog = {(sentiment_voice)}"
    
    logging.debug(f"SQL Query: {update_query} WHERE id =1 ")  # Assuming id = 1 for simplicity

    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()
    logging.debug("Update successful")

#############################################################
    #OCD
########################################################
# Establish database connection
# Database configuration
# db_config = {
#     'host': '127.0.0.1',
#     'user': 'root',
#     'password': '040503',
#     'database': 'mental_health'
# }

# # Connect to MySQL database
# conn = mysql.connector.connect(**db_config)
# cursor = conn.cursor()

# Create a table to store the data
table_creation_query_ocd = """
CREATE TABLE IF NOT EXISTS ocd_prediction (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    q1 INT,
    q2 INT,
    q3 INT,
    q4 INT,
    q5 INT,
    q6 INT,
    q7 INT,
    q8 INT,
    q9 INT,
    q10 INT,
    q11 INT,
    q12 INT,
    q13 INT,
    q14 INT,
    q15 INT,
    q16 INT,
    Age FLOAT,
    Gender INT,
    Education INT,
    Country INT
);
"""
cursor.execute(table_creation_query_ocd)
db_connection.commit()

model_ocd = pickle.load(open("E:/Mental Health Detection Project/ocd_pred/models/cat_model_ocd", "rb"))


@app.route('/ocd')
def ocd():
    username = session.get('username')
    return render_template('index_ocd.html',username=username)

@app.route('/predict', methods=['POST'])
def predict():
    username = session.get('username')
    if request.method == 'POST':
        try:
            
            # Retrieve form data
            feature_values_ocd = {key: request.form[key] for key in request.form}
            print("Form Data:", feature_values_ocd)

            # Convert form data to feature vector
            feature_vector_ocd = [int(feature_values_ocd.get('q1')),
                                  int(feature_values_ocd.get('q2')),
                                  int(feature_values_ocd.get('q3')),
                                  int(feature_values_ocd.get('q4')),
                                  int(feature_values_ocd.get('q5')),
                                  int(feature_values_ocd.get('q6')),
                                  int(feature_values_ocd.get('q7')),
                                  int(feature_values_ocd.get('q8')),
                                  int(feature_values_ocd.get('q9')),
                                  int(feature_values_ocd.get('q10')),
                                  int(feature_values_ocd.get('q11')),
                                  int(feature_values_ocd.get('q12')),
                                  int(feature_values_ocd.get('q13')),
                                  int(feature_values_ocd.get('q14')),
                                  int(feature_values_ocd.get('q15')),
                                  int(feature_values_ocd.get('q16')),
                                  float(feature_values_ocd.get('Age')),
                                  int(feature_values_ocd.get('Gender')),
                                  int(feature_values_ocd.get('Education')),
                                  int(feature_values_ocd.get('Country'))]

            print("Feature Vector OCD:", feature_vector_ocd)

            # Make a prediction
            prediction = model_ocd.predict([feature_vector_ocd])

            # Return the predicted value
            prediction_ocd = "Positive, OCD Detected" if prediction[0] == 1 else "Negative, OCD Not Detected"
            db_prediction_ocd = 1 if prediction[0] == 1 else 0

            # Insert the data into the database
            db_connection = get_db_connection()
            cursor = db_connection.cursor()

            # Insert the data into the database
            insert_query_ocd = """
            INSERT INTO ocd_prediction 
            (timestamp,q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, Age,Gender,Education,Country) 
            VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query_ocd, feature_vector_ocd)
            db_connection.commit()

            print("Data Inserted Successfully")

            update_ocd_value(username,db_prediction_ocd)

            return render_template('index_ocd.html', prediction=prediction_ocd)

        except Exception as e:
            print("Error:", e)

        finally:
            # Close the cursor and the connection
            cursor.close()
            db_connection.close()

def update_ocd_value(username, db_prediction_ocd):
    logging.debug(f"Updating ocd value for username: {username}")
    connection = mysql.connector.connect(**db_config, autocommit=True)
    cursor = connection.cursor()

    # Construct SQL query to update the "depression_pred" column for the given username
    update_query = f"UPDATE {username} SET ocd = {(db_prediction_ocd)}"
    
    logging.debug(f"SQL Query: {update_query} WHERE id =1 ")  # Assuming id = 1 for simplicity

    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()
    logging.debug("Update successful")
        
#########################################################################################################################################
#                                                 Bipolar Prediction                                                                    #
#########################################################################################################################################

# Load the pickled model
with open('E:/Mental Health Detection Project/mental health site/lgr_model_bipolar', 'rb') as model_file:
    model_bipolar = pickle.load(model_file)

# Database configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '040503',
    'database': 'mental_health'
}

# Connect to MySQL database
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Create a table to store form data if not exists
create_table_query = """
CREATE TABLE IF NOT EXISTS bipolar_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    depressed_mood_q1 INT,
    sleep_disturbance_q2 INT,
    appetite_disturbance_q3 INT,
    social_engagement_energy_q4 INT,
    not_feeling_motivated_q5 INT,
    anxiety_q6 INT,
    feel_worthlessness_q7 INT,
    suicidal_thoughts_q8 INT,
    talkative_or_speak_q9 INT,
    felt_both_high_elated_q10 INT,
    more_interested_in_sex_q11 INT,
    interest_in_being_with_people_q12 INT,
    self_confidence_ranges_q13 INT,
    blood_relatives_q14 INT,
    period_of_time_when_q15 INT,
    over_involved_in_q16 INT,
    totally_confident_q17 INT,
    talk_over_people_q18 INT,
    sleep_less_and_q19 INT,
    see_things_in_a_new_q20 INT,
    age FLOAT,
    gender INT,
    education INT,
    country INT
)
"""
cursor.execute(create_table_query)
conn.commit()

@app.route('/bipolar_predict')
def bipolar_predict():
    username = session.get('username')
    return render_template('bipolar.html',username=username)

@app.route('/predict_bipolar', methods=['POST'])
def predict_bipolar():
    try:
        username = session.get('username')
        # Extract feature values from the form data
        feature_values = {key: request.form[key] for key in request.form}
        feature_vector = [int(feature_values.get('depressed_mood')),
                          int(feature_values.get('sleep_disturbance')),
                          int(feature_values.get('appetite_disturbance')),
                          int(feature_values.get('reduction_in_social_engagement_energy')),
                          int(feature_values.get('not_feeling_motivated_concentrated')),
                          int(feature_values.get('anxiety')),
                          int(feature_values.get('feel_worthlessness_hopeless_helpless')),
                          int(feature_values.get('suicidal_thoughts')),
                          int(feature_values.get('talkative_or_speak_faster_than_usual')),
                          int(feature_values.get('felt_both_high_elated_and_low_depressed_at_the_same_time')),
                          int(feature_values.get('more_interested_in_sex_than_usual')),
                          int(feature_values.get('interest_in_being_with_people_vs_wanting_to_be_left_alone')),
                          int(feature_values.get('self_confidence_ranges_from_self_doubt_to_overconfidence')),
                          int(feature_values.get('blood_relatives_with_manic_depressive_illness_or_bipolar_disorder')),
                          int(feature_values.get('period_of_time_when_thoughts_raced_through_your_head')),
                          int(feature_values.get('over_involved_in_new_plans_and_projects')),
                          int(feature_values.get('totally_confident_that_everything_you_do_will_succeed')),
                          int(feature_values.get('talk_over_people')),
                          int(feature_values.get('sleep_less_and_not_feel_tired')),
                          int(feature_values.get('see_things_in_a_new_and_exciting_light')),
                          float(feature_values.get('age')),
                          int(feature_values.get('gender')),
                          int(feature_values.get('education')),
                          int(feature_values.get('country'))]

        # Insert form data into the database
        insert_query_bipolar = """
        INSERT INTO bipolar_data (
        timestamp,depressed_mood_q1 ,sleep_disturbance_q2 ,appetite_disturbance_q3 ,social_engagement_energy_q4,not_feeling_motivated_q5 ,anxiety_q6,feel_worthlessness_q7,
        suicidal_thoughts_q8,talkative_or_speak_q9,felt_both_high_elated_q10,more_interested_in_sex_q11,interest_in_being_with_people_q12,self_confidence_ranges_q13,
        blood_relatives_q14,period_of_time_when_q15,
        over_involved_in_q16,totally_confident_q17,talk_over_people_q18,sleep_less_and_q19,see_things_in_a_new_q20, age ,gender ,education,country
        ) 
        VALUES (NOW(),%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query_bipolar, tuple(feature_vector))
        conn.commit()

        # Make the prediction using the loaded model
        prediction_bipolar = model_bipolar.predict([feature_vector])[0]

        # Map the prediction to the corresponding outcome
        outcomes = {0: 'Less Detected', 1: 'Mild Detected', 2: 'Highly Detected'}
        predicted_bipolar = outcomes.get(prediction_bipolar, 'Invalid Prediction')
    

        # Determine the current set based on the form data
        current_set = int(request.form.get('current_set', 1))
        update_bipolar_value(username,prediction_bipolar)
        return render_template('bipolar.html', prediction=predicted_bipolar, current_set=current_set)

    except Exception as e:
        # Handle exceptions, e.g., invalid input
        print("Exception:", e)
        return render_template('bipolar.html', prediction="Error: Invalid input", current_set=1)
    
def update_bipolar_value(username, prediction_bipolar):
    logging.debug(f"Updating bipolar value for username: {username}")
    connection = mysql.connector.connect(**db_config, autocommit=True)
    cursor = connection.cursor()

    # Construct SQL query to update the "depression_pred" column for the given username
    update_query = f"UPDATE {username} SET bipolar_disorder = {(prediction_bipolar)}"
    
    logging.debug(f"SQL Query: {update_query} WHERE id =1 ")  # Assuming id = 1 for simplicity

    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()
    logging.debug("Update successful")
    
#############################################################################################################
#                                         Anxiety and stress                                                #
#############################################################################################################
    
# connection = mysql.connector.connect(**db_config)
# cursor = connection.cursor()

def determine_mental_health_level(score):
    if score <= 20:
        return "Low"
    elif score >= 40:
        return "High"
    else:
        return "Mild"

@app.route('/anxiety_stress', methods=['GET', 'POST'])
def anxiety_stress():
    username = session.get('username')
    if request.method == 'POST':
        # Process the form data here if needed
        return render_template('anexity_stress.html', username=username)
    
    # If it's a GET request, render the form
    return render_template('anexity_stress.html', username=username)


@app.route('/predict_anxiety', methods=['GET', 'POST'])
def predict_anxiety():
    username = session.get('username')
    if request.method == 'POST':
        print(request.form)
        # Retrieve the form data
        q1 = int(request.form['q1'])
        q2 = int(request.form['q2'])
        q3 = int(request.form['q3'])
        q4 = int(request.form['q4'])
        q5 = int(request.form['q5'])
        q6 = int(request.form['q6'])
        q7 = int(request.form['q7'])
        q8 = int(request.form['q8'])
        q9 = int(request.form['q9'])
        q10 = int(request.form['q10'])
        q11 = int(request.form['q11'])
        q12 = int(request.form['q12'])

        # Calculate the total score
        total_score = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11 + q12

        # Determine the mental health level
        mental_health_level = determine_mental_health_level(total_score)
        mental_health_score= 0 if mental_health_level == 'Low' else (2 if mental_health_level == 'High' else 1)

        update_anxiety_value(username,mental_health_score)

        return render_template('anexity_stress.html', prediction=mental_health_level)

    

def update_anxiety_value(username, mental_health_score):
    logging.debug(f"Updating anxiety value for username: {username}")
    connection = mysql.connector.connect(**db_config, autocommit=True)
    cursor = connection.cursor()

    # Construct SQL query to update the "depression_pred" column for the given username
    update_query = f"UPDATE {username} SET anxiety_and_stress = {(mental_health_score)}"
    
    logging.debug(f"SQL Query: {update_query} WHERE id =1 ")  # Assuming id = 1 for simplicity

    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()
    logging.debug("Update successful")

if __name__ == '__main__':
    app.run(debug=True)

'''
import pickle

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, render_template, request
from flask_cors import cross_origin
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import load_model


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index1.html')
#
#                                   SOCIAL ANEXITY PREDICTION
#
# Load the pickled model

with open('E:/Mental Health Detection Project/mental health site/stc_model_social', 'rb') as model_file:
    model1 = pickle.load(model_file)

#model1 = load_model('E:/mental health site/deep_le_model_social.h5')

@app.route('/social', methods=['GET', 'POST'])
def social_anxiety():
    if request.method == 'POST':
        # Retrieve form data
        age = float(request.form.get('age'))
        gender = int(request.form.get('gender'))
        family_history = int(request.form.get('familyHistory'))
        occupation = int(request.form.get('occupation'))

        atf = float(request.form.get('atf'))
        tkf = float(request.form.get('tkf'))
        def_ = float(request.form.get('def'))
        smf = float(request.form.get('smf'))
        daf = float(request.form.get('daf'))
        hr = int(request.form.get('hr'))
        sw = int(request.form.get('sw'))
        tr = int(request.form.get('tr'))
        dr = int(request.form.get('dr'))
        br = int(request.form.get('br'))
        ck = int(request.form.get('ck'))
        cp = int(request.form.get('cp'))
        ns = int(request.form.get('ns'))
        dz = int(request.form.get('dz'))
        ur = int(request.form.get('ur'))
        ub = int(request.form.get('ub'))
        md = int(request.form.get('md'))
        tg = int(request.form.get('tg'))

        # Prepare the input data for prediction
        input_data = np.array([[age, gender, family_history, occupation, atf, tkf, def_, smf, daf, hr, sw, tr, dr, br, ck, cp, ns, dz, ur, ub, md, tg]])

        # Make a prediction
        prediction1 = model1.predict(input_data)

        # Return the predicted value
        return f"The predicted value is: {prediction1[0]}"

    return render_template('social.html')
#
#                                   SENTIMENT PREDICTION
#

# Load the VADER model
sid = joblib.load('E:/mental health site/vader_model.joblib')

@app.route('/passage_pred')
def sentiment_analysis():
    return render_template('passage.html')

@app.route('/passage_result', methods=['POST'])
def sentiment_pred():
    if request.method == 'POST':
        text = request.form['text']
        sentiment_score = sid.polarity_scores(text)['compound']
        sentiment = 'positive' if sentiment_score >= 0.05 else ('negative' if sentiment_score <= -0.05 else 'neutral')
        return render_template('passage.html', text=text, sentiment=sentiment)

#
#                           PREDICTION THROUGH SONGS
#
#model = pickle.load(open("E:/mental health site/stc_model", "rb"))
 
with open('E:/Mental Health Detection Project/detecting_mental_health_through_songs/pickle files/stc_model', 'rb') as model_file:
    model = pickle.load(model_file)

#model = load_model('E:/mental health site/deep_le_model_adult_songs.h5')

@app.route("/songs")
@cross_origin()
def songs():
    return render_template("songs1.html")
    

@app.route("/songs_predict", methods = ["GET", "POST"])
@cross_origin()
def songs_predict():
    if request.method == "POST":

        # Total Stops
        Hours_per_day = int(request.form["Hours_per_day"])
        
        # print(Total_stops)
        

        Frequency_Classical = int(request.form["Frequency_Classical"])

        Frequency_Country = int(request.form["Frequency_Country"])

        Frequency_EDM = int(request.form["Frequency_EDM"])

        Frequency_Folk = int(request.form["Frequency_Folk"])

        Frequency_Gospel = int(request.form["Frequency_Gospel"])

        Frequency_Hip_hop = int(request.form["Frequency_Hip_hop"])

        Frequency_Jazz = int(request.form["Frequency_Jazz"])

        Frequency_K_pop = int(request.form["Frequency_K_pop"])

        Frequency_Latin = float(request.form["Frequency_Latin"])

        Frequency_Lofi = float(request.form["Frequency_Lofi"])

        Frequency_Metal = int(request.form["Frequency_Metal"])

        Frequency_Pop = int(request.form["Frequency_Pop"])

        Frequency_RB= int(request.form["Frequency_RB"])

        Frequency_Rock = int(request.form["Frequency_Rock"])

        Frequency_Video_game_music = int(request.form["Frequency_Video_game_music"])

        Insomnia = int(request.form["Insomnia"])

        OCD = int(request.form["OCD"])

        Music_effects = int(request.form["Music_effects"])

        Fav_genre=request.form['Fav_genre']
        if(Fav_genre=='Fav_genre_Classical'):
            Fav_genre_Classical = 1
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0
            

        elif (Fav_genre=='Fav_genre_Country'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 1
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_EDM'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 1
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0
            
        elif (Fav_genre=='Fav_genre_Folk'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 1
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0
            
        elif (Fav_genre=='Fav_genre_Hip_hop'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 1
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Jazz'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=1
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_K_pop'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 1
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Latin'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 1
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Lofi'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 1
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Metal'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 1
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Pop'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 1
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_RB'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 1
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Rap'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 1
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 0

        elif (Fav_genre=='Fav_genre_Rock'):
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 1
            Fav_genre_Video_game_music= 0
            
        else:
            Fav_genre_Classical = 0
            Fav_genre_Country = 0
            Fav_genre_EDM = 0
            Fav_genre_Folk = 0
            Fav_genre_Hip_hop= 0
            Fav_genre_Jazz=0
            Fav_genre_K_pop = 0
            Fav_genre_Latin = 0
            Fav_genre_Lofi = 0
            Fav_genre_Metal = 0
            Fav_genre_Pop = 0
            Fav_genre_RB = 0
            Fav_genre_Rap= 0
            Fav_genre_Rock= 0
            Fav_genre_Video_game_music= 1

        Foreign_languages = request.form["Foreign_languages"]
        if (Foreign_languages == 'Foreign_languages_Yes'):
            Foreign_languages_Yes = 1
            Foreign_languages_No = 0
            

        else:
            Foreign_languages_Yes = 0
            Foreign_languages_No = 1

        
            
        prediction=model.predict([[
            Hours_per_day,
            Fav_genre_Classical,
            Fav_genre_Country,
            Fav_genre_EDM,
            Fav_genre_Folk,
            Fav_genre_Hip_hop,
            Fav_genre_Jazz,
            Fav_genre_K_pop,
            Fav_genre_Latin,
            Fav_genre_Lofi,
            Fav_genre_Metal,
            Fav_genre_Pop,
            Fav_genre_RB,
            Fav_genre_Rap,
            Fav_genre_Rock,
            Fav_genre_Video_game_music,
            Foreign_languages_No,
            Foreign_languages_Yes,
            Frequency_Classical,
            Frequency_Country,
            Frequency_EDM,
            Frequency_Folk,
            Frequency_Gospel,
            Frequency_Hip_hop,
            Frequency_Jazz,
            Frequency_K_pop,
            Frequency_Latin,
            Frequency_Lofi,
            Frequency_Metal,
            Frequency_Pop,
            Frequency_RB,
            Frequency_Rock,
            Frequency_Video_game_music,
            Insomnia,
            OCD,
            Music_effects,
                       
        ]])
          
        output=round(prediction[0],2)

        return render_template('songs1.html',prediction_text="Employee Attrition is {}, Where 1 is mental health not detected, 0 is mental health detected".format(output))


    return render_template("songs1.html")



if __name__ == '__main__':
    app.run(debug=True)
'''