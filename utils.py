# utils.py
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json
import logging
import redis
import pickle
import hashlib
import json
import time
import threading
import redis.exceptions
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables from .env file
load_dotenv()

# Cấu hình Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

# 1. Load mô hình và các đối tượng đã lưu
best_model = joblib.load('./model/best_model.pkl')
scaler = joblib.load('./model/scaler.pkl')
new_user_scaler = joblib.load('./model/new_user_scaler.pkl')
le_exercise = joblib.load('./model/le_exercise.pkl')
le_experience = joblib.load('./model/le_experience.pkl')
le_activity = joblib.load('./model/le_activity.pkl')
le_desired_exp = joblib.load('./model/le_desired_exp.pkl')

# 2. Đọc và xử lý dữ liệu bài tập
data = pd.read_csv("./data/exercises_dataset.csv")
data['Exercise Type Original'] = data['Exercise Type']
data['Experience Level Original'] = data['Experience Level']
data['Exercise Type'] = le_exercise.transform(data['Exercise Type'])
data['Experience Level'] = le_experience.transform(data['Experience Level'])
data = pd.get_dummies(data, columns=['Target Muscle Group', 'Equipment Required'])

# Gán giá trị MET
met_values = {
    'Strength': 5.0, 'Warmup': 2.5, 'Conditioning': 10.0, 'Olympic Weightlifting': 6.0,
    'SMR': 2.0, 'Plyometrics': 9.0, 'Activation': 2.5, 'Powerlifting': 6.0,
    'Strongman': 8.0, 'Stretching': 2.0
}
data['MET'] = data['Exercise Type'].map(lambda x: met_values[le_exercise.classes_[x]])

# 3. Đọc dữ liệu từ exercise_details.csv
exercise_details_df = pd.read_csv("./data/exercise_details.csv")

# 4. Hàm lọc khuyến nghị
def filter_recommendations(df, experience_level=None, top_n=10):
    sorted_df = df.sort_values('Predicted Suitability', ascending=False).copy()
    sorted_df['Exercise Type Original'] = le_exercise.inverse_transform(sorted_df['Exercise Type'])
    sorted_df['Experience Level Original'] = le_experience.inverse_transform(sorted_df['Experience Level'])

    if experience_level is not None:
        filtered_df = sorted_df[sorted_df['Experience Level Original'] == experience_level]
        if len(filtered_df) == 0:
            return pd.DataFrame()
        result = filtered_df.drop_duplicates('Exercise Type Original').head(top_n)
    else:
        result = pd.DataFrame()
        levels = ['Beginner', 'Intermediate', 'Advanced']
        max_per_level = min(2, top_n // len(levels) + 1)

        for level in levels:
            level_df = sorted_df[sorted_df['Experience Level Original'] == level]
            level_result = level_df.drop_duplicates('Exercise Type Original').head(max_per_level)
            result = pd.concat([result, level_result])

        if len(result) < top_n:
            remaining = sorted_df[~sorted_df.index.isin(result.index)]
            additional = remaining.drop_duplicates('Exercise Type Original').head(top_n - len(result))
            result = pd.concat([result, additional])

    return result[['Name', 'Exercise Type Original', 'Experience Level Original', 'Predicted Calories per Minute', 'Predicted Suitability']].head(top_n)

# 5. Hàm gán thông số chi tiết
def assign_exercise_details(exercise_type, experience_level):
    details = exercise_details_df[
        (exercise_details_df['Exercise Type'] == exercise_type) &
        (exercise_details_df['Experience Level'] == experience_level)
    ]

    if not details.empty:
        details = details.iloc[0]
        rest = min(details['Rest Between Sets (seconds)'], 60)
        if details['Time per Set (seconds)'] > 0:
            return {
                'sets': details['Sets'],
                'time_per_set': details['Time per Set (seconds)'],
                'rest': rest
            }
        else:
            return {
                'sets': details['Sets'],
                'reps': details['Reps'],
                'time_per_rep': details['Time per Rep (seconds)'],
                'rest': rest
            }
    else:
        return {'sets': 3, 'reps': 10, 'time_per_rep': 3, 'rest': 60}

# 6. Hàm tạo Workout Plan
def generate_workout_plans(recommended_exercises, target_calories_burned, num_combinations=3, max_time_per_exercise=20):
    workout_plans = []
    exercises_list = recommended_exercises.to_dict('records')
    
    for _ in range(num_combinations):
        plan = []
        remaining_calories = target_calories_burned
        shuffled_exercises = exercises_list.copy()
        random.shuffle(shuffled_exercises)
        exercise_index = 0
        used_exercises = set()

        while remaining_calories > 0:
            available_exercises = [e for e in shuffled_exercises if e['Name'] not in used_exercises]
            if not available_exercises:
                available_exercises = shuffled_exercises
            
            exercise = available_exercises[exercise_index % len(available_exercises)]
            exercise_index += 1
            used_exercises.add(exercise['Name'])
            
            details = assign_exercise_details(exercise['Exercise Type Original'], exercise['Experience Level Original'])
            sets = details['sets']
            
            if 'reps' in details:
                reps = details['reps']
                time_per_set = reps * details['time_per_rep']
                reps_or_time_per_set = f"{reps} reps"
            else:
                time_per_set = details['time_per_set']
                reps_or_time_per_set = f"{time_per_set} seconds"
            
            rest_between_sets = details['rest']
            
            total_active_time_seconds = sets * time_per_set
            total_time_with_rest_seconds = total_active_time_seconds + (sets - 1) * rest_between_sets
            calories_per_minute = exercise['Predicted Calories per Minute']
            total_active_time_minutes = total_active_time_seconds / 60
            calories_burned = total_active_time_minutes * calories_per_minute
            
            if calories_burned > remaining_calories:
                total_active_time_minutes = remaining_calories / calories_per_minute
                total_active_time_seconds = total_active_time_minutes * 60
                sets = int(total_active_time_seconds // time_per_set)
                if sets < 1:
                    sets = 1
                total_active_time_seconds = sets * time_per_set
                total_active_time_minutes = total_active_time_seconds / 60
                total_time_with_rest_seconds = total_active_time_seconds + (sets - 1) * rest_between_sets
                calories_burned = total_active_time_minutes * calories_per_minute
            
            if total_time_with_rest_seconds / 60 > max_time_per_exercise:
                excess_time = total_time_with_rest_seconds / 60 - max_time_per_exercise
                excess_sets = int(excess_time * 60 // (time_per_set + rest_between_sets)) + 1
                sets = max(1, sets - excess_sets)
                total_active_time_seconds = sets * time_per_set
                total_active_time_minutes = total_active_time_seconds / 60
                total_time_with_rest_seconds = total_active_time_seconds + (sets - 1) * rest_between_sets
                calories_burned = total_active_time_minutes * calories_per_minute
            
            remaining_calories -= calories_burned
            
            plan.append({
                'Name': exercise['Name'],
                'Exercise Type': exercise['Exercise Type Original'],
                'Experience Level': exercise['Experience Level Original'],
                'Sets': sets,
                'Reps/Time per Set': reps_or_time_per_set,
                'Time per Set (seconds)': time_per_set,
                'Rest Between Sets (seconds)': rest_between_sets,
                'Total Active Time (minutes)': round(total_active_time_minutes, 2),
                'Total Time with Rest (minutes)': round(total_time_with_rest_seconds / 60, 2),
                'Calories Burned': round(calories_burned, 2)
            })
        
        total_calories = sum(exercise['Calories Burned'] for exercise in plan)
        if total_calories > 0:
            workout_plans.append((pd.DataFrame(plan), total_calories))
    
    return workout_plans

# 7. Hàm dự đoán cho người dùng mới
def predict_for_new_user(new_user_data, experience_level, top_n, target_calories_burned, num_combinations):
    new_user = pd.DataFrame(new_user_data)
    new_user['Activity Level'] = le_activity.transform(new_user['Activity Level'])
    new_user['Desired Experience Level'] = le_desired_exp.transform(new_user['Desired Experience Level'])
    new_user['Exercise Type'] = le_exercise.transform(new_user['Exercise Type'])
    new_user['Experience Level'] = le_experience.transform(new_user['Experience Level'])

    # Tính BMI_Calorie_Interaction dựa trên MET
    new_user['BMI_Calorie_Interaction'] = new_user['BMI'] * (data['MET'] * new_user['BMI'][0] / 60)
    new_user['Age_Activity_Interaction'] = new_user['Age'] * new_user['Activity Level']
    new_user_numeric_columns = ['BMI', 'Age', 'BMI_Calorie_Interaction', 'Age_Activity_Interaction']
    new_user[new_user_numeric_columns] = new_user_scaler.transform(new_user[new_user_numeric_columns])

    predictions = best_model.predict(new_user)
    data['Predicted Calories per Minute'] = predictions[:, 0]
    data['Predicted Suitability'] = predictions[:, 1]

    # Hiệu chỉnh Suitability
    mask = data['Experience Level Original'] == experience_level
    data.loc[mask, 'Predicted Suitability'] = np.clip(data.loc[mask, 'Predicted Suitability'] + 0.25, 0, 1)

    # Lọc bài tập
    recommended_exercises = filter_recommendations(data, experience_level=experience_level, top_n=top_n)
    if recommended_exercises.empty:
        return None, None


   # Tạo Workout Plans bằng Gemini AI
    workout_plans = generate_workout_plans_with_gemini(new_user_data, recommended_exercises, target_calories_burned, num_combinations)
    
    return recommended_exercises, workout_plans


def generate_workout_plans_with_gemini(new_user_data, recommended_exercises, target_calories, num_combinations):
    # Chuẩn bị thông tin người dùng từ new_user_data
    gender = "Male" if new_user_data['Gender'][0] == 0 else "Female"
    experience_level = new_user_data['Experience Level'][0]  # Đã được mã hóa, cần giải mã
    # experience_level = le_experience.inverse_transform([experience_level])[0]  # Giải mã Experience Level
    age = new_user_data['Age'][0]
    bmi = new_user_data['BMI'][0]
    activity_level =  new_user_data['Activity Level'][0] # Giải mã Activity Level

    # In giá trị target_calories bằng logging
    logger.info(f'target_calories: {target_calories}')

    # Chuẩn bị danh sách bài tập từ recommended_exercises
    exercises_str = "\n".join([
        f"- {row['Name']} ({row['Exercise Type Original']}, {row['Experience Level Original']}, {row['Predicted Calories per Minute']} cal/min)."
        for _, row in recommended_exercises.iterrows()
    ])

    # Tạo prompt cho Gemini AI
    # prompt = f"""
    #             You are a professional personal trainer. I am a {age}-year-old {gender} with a BMI of {bmi}, a {activity_level} activity level, and a {experience_level} experience level. I want to burn {target_calories} calories per session. Below is a list of recommended exercises for me, along with their type, experience level, and calories burned per minute:

    #             {exercises_str}

    #             Please create {num_combinations} different workout plans, each burning approximately {target_calories} calories (within ±15 calories of the target). Each plan must include some exercises from the list above, with the following details for each exercise: Name, ExerciseType, ExperienceLevel, Sets, RepsOrTimePerSet, TimePerSetSeconds, RestBetweenSetsSeconds, TotalActiveTimeMinutes, TotalTimeWithRestMinutes, CaloriesBurned. Additionally, each plan should include the total calories burned (total_calories_burned) and total completion time (total_completion_time_minutes). The output format must be JSON as follows:

    #             ```json
    #             {{
    #             "exercises": [
    #                 {{
    #                 "Name": "",
    #                 "ExerciseType": "",
    #                 "ExperienceLevel": "",
    #                 "Sets": 0,
    #                 "RepsOrTimePerSet": "",
    #                 "TimePerSetSeconds": 0,
    #                 "RestBetweenSetsSeconds": 0,
    #                 "TotalActiveTimeMinutes": 0,
    #                 "TotalTimeWithRestMinutes": 0,
    #                 "CaloriesBurned": 0
    #                 }}
    #             ],
    #             "total_calories_burned": 0,
    #             "total_completion_time_minutes": 0
    #             }}
    #         ```
    # To ensure variety, aim to use as many different exercises as possible from the list across all workout plans. Ideally, each exercise should be used in at least one of the {num_combinations} workout plans, and each workout plan should include a diverse mix of exercise types (e.g., Warmup, Strength, Conditioning).Adjust the sets, reps, or time per set to ensure the total calories burned for each workout plan is within ±15 calories of {target_calories}.Ensure the exercises are suitable for a {experience_level}, and accurately calculate calories based on the exercise duration and calories burned per minute. Output the result as a JSON array containing {num_combinations} workout plans.Ensure the output is a valid JSON string without any additional text before or after the JSON.
    # """

# Đây là prompt dùng caloriesPerMinutes ước tính của Gemini chứ không phải của hệ thống khuyến nghị
#     prompt = f"""
#                 You are a professional personal trainer with extensive knowledge of real-world calorie burn rates for various exercises. I am a {age}-year-old {gender} with a BMI of {bmi}, a {activity_level} activity level, and a {experience_level} experience level. I want to burn {target_calories} calories per session. Below is a list of recommended exercises for me, along with their type, experience level, and predicted calories burned per minute (for reference only):

#                 {exercises_str}

#                 Please create {num_combinations} different workout plans, each burning approximately {target_calories} calories (within ±15 calories of the target). Each plan must include some exercises from the list above, with the following details for each exercise: Name, ExerciseType, ExperienceLevel, Sets, RepsOrTimePerSet, TimePerSetSeconds, RestBetweenSetsSeconds, TotalActiveTimeMinutes, TotalTimeWithRestMinutes, CaloriesBurned. Additionally, each plan should include the total calories burned (total_calories_burned) and total completion time (total_completion_time_minutes). The output format must be JSON as follows:

#                 ```json
#                 {{
#                 "exercises": [
#                     {{
#                     "Name": "",
#                     "ExerciseType": "",
#                     "ExperienceLevel": "",
#                     "Sets": 0,
#                     "RepsOrTimePerSet": "",
#                     "TimePerSetSeconds": 0,
#                     "RestBetweenSetsSeconds": 0,
#                     "TotalActiveTimeMinutes": 0,
#                     "TotalTimeWithRestMinutes": 0,
#                     "CaloriesBurned": 0
#                     }}
#                 ],
#                 "total_calories_burned": 0,
#                 "total_completion_time_minutes": 0
#                 }}```
#                 When calculating the calories burned for each exercise, do NOT strictly follow the predicted calories per minute provided by the system. Instead, use your knowledge as a professional trainer to estimate the calories burned per minute based on real-world values for the given exercise type and experience level. For example:

#                 Warmup exercises (e.g., stretching, light movements) typically burn 2–5 cal/min.
#                 Strength exercises (e.g., weightlifting) typically burn 5–8 cal/min.
#                 Conditioning or cardio exercises (e.g., battling ropes, running) typically burn 8–12 cal/min.
#                 Important: You MUST ONLY use exercises from the list provided above. Do NOT add any new exercises that are not in the list.To ensure variety, aim to use as many different exercises as possible from the list across all workout plans. Ideally, each exercise should be used in at least one of the {num_combinations} workout plans, and each workout plan should include a diverse mix of exercise types (e.g., Warmup, Strength, Conditioning).Adjust the sets, reps, or time per set to ensure the total calories burned for each workout plan is within ±15 calories of {target_calories}. Ensure the exercises are suitable for a {experience_level}, and output the result as a JSON array containing {num_combinations} workout plans.
#    """
    prompt = f"""
            You are a professional personal trainer. I am a {age}-year-old {gender} with a BMI of {bmi}, a {activity_level} activity level, and a {experience_level} experience level. I want to burn {target_calories} calories per session. Below is a list of recommended exercises for me, along with their type, experience level, and calories burned per minute:

            {exercises_str}

            Please create {num_combinations} different workout plans, each burning approximately {target_calories} calories (within ±15 calories of the target, i.e., between {target_calories - 15} and {target_calories + 15} calories). Each plan must include ALL exercises from the list above (exactly {len(recommended_exercises)} exercises), ensuring that every exercise in the list is used in each plan. Each plan should be independent, meaning I can choose any single plan to achieve my calorie-burning goal without combining plans.

            Each plan must include the following details for each exercise: Name, ExerciseType, ExperienceLevel, Sets, RepsOrTimePerSet, TimePerSetSeconds, RestBetweenSetsSeconds, TotalActiveTimeMinutes, TotalTimeWithRestMinutes, and CaloriesBurned. Additionally, each plan should include the total calories burned (total_calories_burned) and total completion time (total_completion_time_minutes). The output format must be JSON as follows:

            ```json
            {{
            "exercises": [
                {{
                "Name": "",
                "ExerciseType": "",
                "ExperienceLevel": "",
                "Sets": 0,
                "RepsOrTimePerSet": "",
                "TimePerSetSeconds": 0,
                "RestBetweenSetsSeconds": 0,
                "TotalActiveTimeMinutes": 0,
                "TotalTimeWithRestMinutes": 0,
                "CaloriesBurned": 0
                }}
            ],
            "total_calories_burned": 0,
            "total_completion_time_minutes": 0
            }}
            ```
            Follow these steps to create the workout plans:

            Include all exercises in each plan:
            Each workout plan must include ALL {len(recommended_exercises)} exercises listed above.
            Do not skip any exercise; every plan must use exactly these {len(recommended_exercises)} exercises in the order they are listed.
            Calculate the required active time:
            The average calories burned per minute across all exercises is approximately {sum(row['Predicted Calories per Minute'] for _, row in recommended_exercises.iterrows() if pd.notna(row['Predicted Calories per Minute'])) / len(recommended_exercises):.3f} cal/min.
            To burn {target_calories} calories, the total active time needed is approximately {target_calories / (sum(row['Predicted Calories per Minute'] for _, row in recommended_exercises.iterrows() if pd.notna(row['Predicted Calories per Minute'])) / len(recommended_exercises)):.1f} minutes.
            Distribute this time equally across the {len(recommended_exercises)} exercises, so each exercise should have a TotalActiveTimeMinutes of approximately {(target_calories / (sum(row['Predicted Calories per Minute'] for _, row in recommended_exercises.iterrows() if pd.notna(row['Predicted Calories per Minute'])) / len(recommended_exercises))) / len(recommended_exercises):.1f} minutes.
            Adjust Sets and TimePerSetSeconds:
            For each exercise, adjust the Sets and TimePerSetSeconds to achieve the required TotalActiveTimeMinutes. For example:
            If TimePerSetSeconds = 60 seconds, then Sets = (TotalActiveTimeMinutes × 60 seconds/minute) ÷ 60 seconds/set.
            If TimePerSetSeconds = 45 seconds, then Sets = (TotalActiveTimeMinutes × 60 seconds/minute) ÷ 45 seconds/set.
            Choose TimePerSetSeconds between 30 and 120 seconds, and calculate Sets accordingly to match the required TotalActiveTimeMinutes.
            Set rest time:
            Use a reasonable RestBetweenSetsSeconds for each exercise (e.g., 30–90 seconds, depending on the exercise type). For example:
            Warmup, Activation, Stretching, SMR: 30 seconds.
            Strength, Conditioning, Plyometrics, Olympic Weightlifting: 60–90 seconds.
            Calculate exercise details:
            TotalActiveTimeMinutes: Calculate as (Sets × TimePerSetSeconds) ÷ 60, and ensure it matches the required value from step 2.
            TotalTimeWithRestMinutes: Calculate as (Sets × TimePerSetSeconds + (Sets - 1) × RestBetweenSetsSeconds) ÷ 60.
            CaloriesBurned: Calculate as TotalActiveTimeMinutes × (calories burned per minute for that exercise). This is the total calories burned for the exercise.
            Calculate workout plan totals:
            total_calories_burned: Must be the sum of CaloriesBurned of all exercises in the plan.
            total_completion_time_minutes: Must be the sum of TotalTimeWithRestMinutes of all exercises in the plan.
            Ensure the exercises are suitable for a {experience_level}, and accurately calculate calories based on the exercise duration and calories burned per minute. If the total_calories_burned for any plan is not within {target_calories - 15} to {target_calories + 15} calories, adjust the Sets or TimePerSetSeconds to meet this requirement. Output the result as a JSON array containing {num_combinations} workout plans. Ensure the output is a valid JSON string without any additional text before or after the JSON.
            """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash', generation_config=GenerationConfig(temperature=0.7))
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        if "json" in response_text:  
            json_str = response_text.split("json")[1].split("```")[0].strip()
        else:
            json_str = response_text
        # Chuyển đổi JSON string thành Python object
        workout_plans = json.loads(json_str)
        # Đảm bảo workout_plans là một list
        if not isinstance(workout_plans, list):
            workout_plans = [workout_plans]
        # Kiểm tra số lượng workout plans
        if len(workout_plans) < num_combinations:
            raise ValueError(f"Gemini AI returned only {len(workout_plans)} workout plans, but {num_combinations} were requested")
        
        # Định nghĩa thứ tự hợp lý của các thể loại bài tập
        exercise_type_order = {
        "Warmup": 1,
        "SMR": 2,
        "Activation": 3,
        "Plyometrics": 4,
        "Olympic Weightlifting": 5,
        "Powerlifting": 6,
        "Strength": 7,
        "Strongman": 8,
        "Conditioning": 9,
        "Stretching": 10
        }
        # Sắp xếp lại các bài tập trong mỗi workout plan theo thứ tự hợp lý
        for plan in workout_plans:
            if "exercises" in plan:
                # Sắp xếp danh sách exercises dựa trên ExerciseType
                plan["exercises"] = sorted(
                    plan["exercises"],
                    key=lambda x: exercise_type_order.get(x["ExerciseType"], 999)  # 999 để các loại không xác định xếp cuối
                )

        #Validate định dạng của workout_plans
        for plan in workout_plans:
            if not isinstance(plan, dict):
                raise ValueError("Each workout plan must be a dictionary")
            if "exercises" not in plan or "total_calories_burned" not in plan or "total_completion_time_minutes" not in plan:
                raise ValueError("Workout plan missing required fields: exercises, total_calories_burned, total_completion_time_minutes")
            if not isinstance(plan["exercises"], list):
                raise ValueError("Exercises in workout plan must be a list")
            for exercise in plan["exercises"]:
                required_exercise_fields = [
                "Name", "ExerciseType", "ExperienceLevel", "Sets", "RepsOrTimePerSet",
                "TimePerSetSeconds", "RestBetweenSetsSeconds", "TotalActiveTimeMinutes",
                "TotalTimeWithRestMinutes", "CaloriesBurned"
                ]
                for field in required_exercise_fields:
                    if field not in exercise:
                        raise ValueError(f"Exercise in workout plan missing required field: {field}")
        return workout_plans[:num_combinations]
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing Gemini AI response as JSON: {str(e)}")
    except Exception as e:
        raise Exception(f"Error calling Gemini AI: {str(e)}")

def clean_for_json(obj):
    """
    Đệ quy biến obj thành kiểu JSON-serializable hoàn toàn.
    """
    if isinstance(obj, pd.Series):
        obj = obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        obj = obj.to_dict(orient='records')
    elif isinstance(obj, np.generic):  # numpy scalar (int, float)
        obj = obj.item()
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()

    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    else:
        return obj

def make_cache_key(prefix, new_user_data, experience_level, top_n, target_calories_burned, num_combinations):
    cleaned_user_data = clean_for_json(new_user_data)

    key_source = {
        "user_data": cleaned_user_data,
        "experience_level": experience_level,
        "top_n": top_n,
        "target_calories": target_calories_burned,
        "num_combinations": num_combinations
    }

    key_str = json.dumps(key_source, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    return f"{prefix}:{key_hash}"