# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np  # Thêm import numpy để xử lý np.nan
from utils import predict_for_new_user, data, make_cache_key, get_list_exercises_recommend
from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import logging

import redis
import pickle
import hashlib
import json
import time
import threading
import redis.exceptions

CACHE_EXPIRE_SECONDS = 3600 # 1 hour
REFRESH_THRESHOLD_SECONDS = 300 

## cấu hình redis
pool = redis.ConnectionPool(
    host='redis-13116.c1.ap-southeast-1-1.ec2.redns.redis-cloud.com',
    port=13116,
    username='default',
    password='mx3EPshCQtweT1lvrlSO4NeesRtAyzso',
    decode_responses=False,  # vì dùng pickle
    max_connections=20       # giới hạn số connection mở ra cùng lúc
)

r = redis.Redis(connection_pool=pool)



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

class RecommendExerciseRequest(BaseModel):
    user_input: UserInput
    top_n: int = 10

class Exercise(BaseModel):
    id: str = Field(..., alias='_id')
    Name: str
    Thumbnail: Optional[str] = None
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
    name: Optional[str] = None
    description: Optional[str] = None

class WorkoutPlanResponse(BaseModel):
    recommended_exercises: List[Exercise]
    workout_plans: List[WorkoutPlan]

class ListRecommendExerciseResponse(BaseModel):
    recommended_exercises: List[Exercise]

@app.get("/")
def home():
    return {"health_check": "OK"}

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

    # Kiểm tra cache redis 
    cache_key = make_cache_key("generate_workout_plan", new_user_data, experience_level, top_n, target_calories_burned, num_combinations)

    # Try to get cached result
    try:
        cached_result = r.get(cache_key)
        if cached_result:
            print("🔁 Cache hit")
            return pickle.loads(cached_result)  # dùng pickle.loads thay vì json.loads
    except Exception as e:
        print(f"⚠️ Redis error: {e}")


    # Dự đoán và tạo Workout Plan
    recommended_exercises, workout_plans = predict_for_new_user(new_user_data, experience_level, top_n, target_calories_burned, num_combinations)

    if recommended_exercises is None:
        raise HTTPException(status_code=400, detail=f"No exercises found for {experience_level} level")

    # Đọc lại file exercises_dataset.csv để lấy thông tin đầy đủ
    exercises_dataset = pd.read_csv("./data/exercises_dataset_final.csv")

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
           "id", 'Name', 'Video', 'Target Muscle Group', 'Exercise Type', 'Equipment Required',
            'Mechanics', 'Force Type', 'Experience Level', 'Secondary Muscles',
            'MuscleGroupImageSrc', 'Overview', 'Instructions', 'Tips',
            'Predicted Calories per Minute', 'Predicted Suitability',
            'Thumbnail',
            'Exercise Type Original', 'Experience Level Original'
        ], errors='ignore').to_dict()

        recommended_exercises_list.append({
            "_id": row.get('id', None), 
            "Name": row['Name'],
            "Thumbnail":row.get('Thumbnail', None),
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
    for plan in workout_plans:
        # Chuyển dữ liệu từ Gemini AI thành DataFrame để xử lý
        plan_df = pd.DataFrame(plan['exercises'])

        # Thêm cột index để bảo toàn thứ tự ban đầu
        plan_df['original_index'] = plan_df.index

        # Nhóm các bài tập trùng lặp (nếu có)
        grouped_plan = plan_df.groupby(['Name', 'ExerciseType', 'ExperienceLevel']).agg({
            'Sets': 'sum',
            'Reps': 'last',
            'TimePerSetSeconds': 'last',
            'RestBetweenSetsSeconds': 'last',
            'TotalActiveTimeMinutes': 'sum',
            'TotalTimeWithRestMinutes': 'sum',
            'CaloriesBurned': 'sum',
            'original_index': 'min'
        }).reset_index()

        # Thêm trường 'id' cho mỗi bài tập trong workout_plan
        grouped_plan['_id'] = grouped_plan.apply(lambda row: recommended_exercises_full[recommended_exercises_full['Name'] == row['Name']].iloc[0]['id'], axis=1)

        # Sắp xếp lại theo thứ tự ban đầu (dựa trên original_index)
        grouped_plan = grouped_plan.sort_values('original_index').drop(columns=['original_index'])

        total_completion_time = grouped_plan['TotalTimeWithRestMinutes'].sum()
        total_completion_time += (len(grouped_plan) - 1) * rest_between_exercises

        workout_plans_response.append({
            "exercises": grouped_plan.rename(columns={
                'ExerciseType': 'ExerciseType',
                'ExperienceLevel': 'ExperienceLevel',
                'Reps': 'Reps',
                'TimePerSetSeconds': 'TimePerSetSeconds',
                'RestBetweenSetsSeconds': 'RestBetweenSetsSeconds',
                'TotalActiveTimeMinutes': 'TotalActiveTimeMinutes',
                'TotalTimeWithRestMinutes': 'TotalTimeWithRestMinutes',
                'CaloriesBurned': 'CaloriesBurned'
            }).to_dict(orient='records'),
            "total_calories_burned": plan['total_calories_burned'],
            "total_completion_time_minutes": total_completion_time,
            "name":plan["name"],
            "description":plan["description"]
        })

     # --- Try set cache ---
     # Set cache đúng
    try:
        r.setex(cache_key, CACHE_EXPIRE_SECONDS, pickle.dumps(
            {
                "recommended_exercises": recommended_exercises_list,
                "workout_plans": workout_plans_response
            }
        ))
        print(f"[CACHE] Set: {cache_key}")
    except Exception as e:
        print(f"[CACHE] Redis set failed: {str(e)}")

    return {
        "recommended_exercises": recommended_exercises_list,
        "workout_plans": workout_plans_response
    }

@app.post("/recommend-exercises", response_model=ListRecommendExerciseResponse, summary="Get list recommend exercises")
async def get_recommend_exercises(request: RecommendExerciseRequest):
    user_input = request.user_input
    top_n = request.top_n

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

    # Kiểm tra cache redis 
    cache_key = make_cache_key("recommend_exercises", new_user_data, experience_level, top_n, 0, 0)

    # Try to get cached result
    try:
        cached_result = r.get(cache_key)
        if cached_result:
            print("🔁 Cache hit")
            return pickle.loads(cached_result)  # dùng pickle.loads thay vì json.loads
    except Exception as e:
        print(f"⚠️ Redis error: {e}")


    # Dự đoán và tạo Workout Plan
    recommended_exercises = get_list_exercises_recommend(new_user_data, experience_level, top_n)

    if recommended_exercises is None:
        raise HTTPException(status_code=400, detail=f"No exercises found for {experience_level} level")

    # Đọc lại file exercises_dataset.csv để lấy thông tin đầy đủ
    exercises_dataset = pd.read_csv("./data/exercises_dataset_final.csv")

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
           "id", 'Name', 'Video', 'Target Muscle Group', 'Exercise Type', 'Equipment Required',
            'Mechanics', 'Force Type', 'Experience Level', 'Secondary Muscles',
            'MuscleGroupImageSrc', 'Overview', 'Instructions', 'Tips',
            'Predicted Calories per Minute', 'Predicted Suitability',
            'Thumbnail',
            'Exercise Type Original', 'Experience Level Original'
        ], errors='ignore').to_dict()

        recommended_exercises_list.append({
            "_id": row.get('id', None), 
            "Name": row['Name'],
            "Thumbnail":row.get('Thumbnail', None),
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

     # --- Try set cache ---
     # Set cache đúng
    try:
        r.setex(cache_key, CACHE_EXPIRE_SECONDS, pickle.dumps(
            {
                "recommended_exercises": recommended_exercises_list,
            }
        ))
        print(f"[CACHE] Set: {cache_key}")
    except Exception as e:
        print(f"[CACHE] Redis set failed: {str(e)}")

    return {
        "recommended_exercises": recommended_exercises_list
    }
