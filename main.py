# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np  # Thêm import numpy để xử lý np.nan
from utils import predict_for_new_user, data

app = FastAPI(title="Workout Plan API", description="API for generating workout plans and exercise recommendations")

# 1. Pydantic models để định nghĩa request/response
class UserInput(BaseModel):
    BMI: float
    Age: int
    Gender: int  # 0: Male, 1: Female
    ActivityLevel: str  # ['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active']
    DesiredExperienceLevel: str  # ['Beginner', 'Intermediate', 'Advanced']
    ExperienceLevel: Optional[str] = None  # Nếu None, sẽ lấy từ DesiredExperienceLevel

class WorkoutPlanRequest(BaseModel):
    user_input: UserInput
    top_n: int = 10
    target_calories_burned: float = 400
    num_combinations: int = 3

class Exercise(BaseModel):
    Name: str
    Video: Optional[str] = None
    TargetMuscleGroup: Optional[str] = None
    ExerciseType: str
    EquipmentRequired: Optional[str] = None
    Mechanics: Optional[str] = None
    ForceType: Optional[str] = None
    ExperienceLevel: str
    SecondaryMuscles: Optional[str] = None
    MuscleGroupImageSrc: Optional[str] = None
    Overview: Optional[str] = None
    Instructions: Optional[str] = None
    Tips: Optional[str] = None
    PredictedCaloriesPerMinute: float
    PredictedSuitability: float
    AdditionalFields: Dict[str, Any]  # Để chứa các cột khác nếu có

class WorkoutPlan(BaseModel):
    exercises: List[dict]
    total_calories_burned: float
    total_completion_time_minutes: float

class WorkoutPlanResponse(BaseModel):
    recommended_exercises: List[Exercise]
    workout_plans: List[WorkoutPlan]

# 2. API endpoint
@app.post("/generate_workout_plan", response_model=WorkoutPlanResponse, summary="Generate workout plans for a user")
async def generate_workout_plan(request: WorkoutPlanRequest):
    user_input = request.user_input
    top_n = request.top_n
    target_calories_burned = request.target_calories_burned
    num_combinations = request.num_combinations

    # Lấy thông tin từ request
    bmi = user_input.BMI
    age = user_input.Age
    gender = user_input.Gender
    activity_level = user_input.ActivityLevel
    desired_experience_level = user_input.DesiredExperienceLevel
    experience_level = user_input.ExperienceLevel if user_input.ExperienceLevel else desired_experience_level

    # Chuẩn bị dữ liệu người dùng
    new_user_data = {
        'BMI': [bmi] * len(data),
        'Age': [age] * len(data),
        'Gender': [gender] * len(data),
        'Activity Level': [activity_level] * len(data),
        'Desired Experience Level': [desired_experience_level] * len(data),
        'Exercise Type': data['Exercise Type Original'],
        'Experience Level': data['Experience Level Original']
    }

    # Dự đoán và tạo Workout Plan
    recommended_exercises, workout_plans = predict_for_new_user(new_user_data, experience_level, top_n, target_calories_burned, num_combinations)

    if recommended_exercises is None:
        raise HTTPException(status_code=400, detail=f"No exercises found for {experience_level} level")

    # Đọc lại file exercises_dataset.csv để lấy thông tin đầy đủ
    exercises_dataset = pd.read_csv("./data/exercises_dataset.csv")

    # Đổi tên cột Muscle Group Image-src thành MuscleGroupImageSrc để khớp với response
    if 'Muscle Group Image-src' in exercises_dataset.columns:
        exercises_dataset = exercises_dataset.rename(columns={'Muscle Group Image-src': 'MuscleGroupImageSrc'})

    # Lấy thông tin đầy đủ của các bài tập được đề xuất
    recommended_exercises_full = recommended_exercises.merge(
        exercises_dataset,
        left_on=['Name', 'Exercise Type Original', 'Experience Level Original'],
        right_on=['Name', 'Exercise Type', 'Experience Level'],
        how='left'
    )

    # Thay thế tất cả giá trị np.nan bằng None để tránh lỗi validation
    recommended_exercises_full = recommended_exercises_full.replace({np.nan: None})

    # Chuẩn bị response cho recommended_exercises
    recommended_exercises_list = []
    for _, row in recommended_exercises_full.iterrows():
        # Lấy tất cả các cột không thuộc các cột chính
        additional_fields = row.drop([
            'Name', 'Video', 'Target Muscle Group', 'Exercise Type', 'Equipment Required',
            'Mechanics', 'Force Type', 'Experience Level', 'Secondary Muscles',
            'MuscleGroupImageSrc', 'Overview', 'Instructions', 'Tips',
            'Predicted Calories per Minute', 'Predicted Suitability',
            'Exercise Type Original', 'Experience Level Original'
        ], errors='ignore').to_dict()

        recommended_exercises_list.append({
            "Name": row['Name'],
            "Video": row.get('Video', None),
            "TargetMuscleGroup": row.get('Target Muscle Group', None),
            "ExerciseType": row['Exercise Type Original'],
            "EquipmentRequired": row.get('Equipment Required', None),
            "Mechanics": row.get('Mechanics', None),
            "ForceType": row.get('Force Type', None),
            "ExperienceLevel": row['Experience Level Original'],
            "SecondaryMuscles": row.get('Secondary Muscles', None),
            "MuscleGroupImageSrc": row.get('MuscleGroupImageSrc', None),
            "Overview": row.get('Overview', None),
            "Instructions": row.get('Instructions', None),
            "Tips": row.get('Tips', None),
            "PredictedCaloriesPerMinute": row['Predicted Calories per Minute'],
            "PredictedSuitability": row['Predicted Suitability'],
            "AdditionalFields": additional_fields
        })

    # Chuẩn bị response cho workout_plans
    workout_plans_response = []
    rest_between_exercises = 1  # 1 phút nghỉ giữa các bài tập
    for plan, total_calories in workout_plans:
        grouped_plan = plan.groupby(['Name', 'Exercise Type', 'Experience Level']).agg({
            'Sets': 'sum',
            'Reps/Time per Set': 'last',
            'Time per Set (seconds)': 'last',
            'Rest Between Sets (seconds)': 'last',
            'Total Active Time (minutes)': 'sum',
            'Total Time with Rest (minutes)': 'sum',
            'Calories Burned': 'sum'
        }).reset_index()

        total_completion_time = grouped_plan['Total Time with Rest (minutes)'].sum()
        total_completion_time += (len(grouped_plan) - 1) * rest_between_exercises

        workout_plans_response.append({
            "exercises": grouped_plan.rename(columns={
                'Exercise Type': 'ExerciseType',
                'Experience Level': 'ExperienceLevel',
                'Reps/Time per Set': 'RepsOrTimePerSet',
                'Time per Set (seconds)': 'TimePerSetSeconds',
                'Rest Between Sets (seconds)': 'RestBetweenSetsSeconds',
                'Total Active Time (minutes)': 'TotalActiveTimeMinutes',
                'Total Time with Rest (minutes)': 'TotalTimeWithRestMinutes',
                'Calories Burned': 'CaloriesBurned'
            }).to_dict(orient='records'),
            "total_calories_burned": total_calories,
            "total_completion_time_minutes": total_completion_time
        })

    return {
        "recommended_exercises": recommended_exercises_list,
        "workout_plans": workout_plans_response
    }