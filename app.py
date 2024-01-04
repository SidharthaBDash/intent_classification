from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import prediction

app = FastAPI()


# Define the input data model
class InputData(BaseModel):
    sentences: list


# Define the output data model
class OutputData(BaseModel):
    predictions: list


# Define the predict endpoint
@app.post("/predict", response_model=OutputData)
async def predict(data: InputData):
    try:
        # Call the predict method
        sentences = data.sentences
        predictions = prediction(sentences)

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
