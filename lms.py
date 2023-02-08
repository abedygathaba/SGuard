from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Course(BaseModel):
    name: str
    description: str
    price: float

@app.post("/courses")
async def create_course(course: Course):
    return {"name": course.name, "description": course.description, "price": course.price}

@app.get("/courses/{course_id}")
async def read_course(course_id: int):
    return {"course_id": course_id}

@app.put("/courses/{course_id}")
async def update_course(course_id: int, course: Course):
    return {"course_id": course_id, "name": course.name, "description": course.description, "price": course.price}

@app.delete("/courses/{course_id}")
async def delete_course(course_id: int):
    return {"course_id": course_id}
