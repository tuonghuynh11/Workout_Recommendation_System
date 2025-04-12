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
def filter_recommendations(df, experience_level=None, top_n=5):
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

    # Tạo Workout Plans
    workout_plans = generate_workout_plans(recommended_exercises, target_calories_burned, num_combinations=num_combinations)
    
    return recommended_exercises, workout_plans



    