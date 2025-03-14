import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sched, time
import pandas as pd
from openai import OpenAI
import json

# load environment variables
load_dotenv(find_dotenv())

# OpenAI Key
openAI = OpenAI(
    api_key = os.getenv("OPENAI_KEY"),
)

# define the training batch
batchFile = 'dataset-batch-1'

system_message = {"role": "system", "content": "This is a trained system message."}

# prepare dataset for model training
def prepare_data():
    raw = batchFile  + '.csv'
    normalised = 'models/' + batchFile  + '.jsonl'
    
    # Load the CSV dataset
    dataset = pd.read_csv(raw)

    # Prepare the JSONL format
    with open(normalised, 'w') as outfile:
        for _, row in dataset.iterrows():
            # Construct the JSON object
            json_obj = {
                "messages": [
                    system_message,
                    {"role": "user", "content": row['instruction']},
                    {"role": "assistant", "content": row['response']}
                ]
            }
            
            # Write the JSON object as a line in the JSONL file
            outfile.write(json.dumps(json_obj) + "\n")

    print(f"JSONL file created at: {normalised}")

# train and fine-tune the GPT model    
def fine_tune():
    # normalise dataset and convert to JSONL for model training
    prepare_data() 
    
    # Upload the training file
    training_file = openAI.files.create(
        file = Path('models/' + batchFile + '.jsonl'),
        purpose = 'fine-tune'
    )

    # Fine-tune the model
    fine_tune_response = openAI.fine_tuning.jobs.create(
        training_file = training_file.id,
        model = "gpt-3.5-turbo" 
    )
    
    fine_tune_id = fine_tune_response.id

    print("Fine-tuning initiated. Fine-tune ID:", fine_tune_id)
    
    # monitor training progress and status
    while True:
        status = training_status(fine_tune_id)
        
        if status == 'succeeded':
            print("Fine-tuning completed successfully!")
            break
        elif status == 'failed':
            print("Fine-tuning failed.")
            break
        elif status == 'cancelled':
            print("Fine-tuning was cancelled.")
            break
        else:
            print(f"Fine-tuning status: {status}. Checking...")
        
        # Wait for 5 seconds before checking again
        time.sleep(5)

# monitor fine-tuning status
def training_status(id):
    response = openAI.fine_tuning.jobs.retrieve(id)
    return response.status

# validate training data
def validate():
    jsonl_file = 'models/' + batchFile + '.jsonl'

    with open(jsonl_file, 'r') as file:
        for i, line in enumerate(file, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                print(f"Line {i} is not valid JSON: {line}")

# run model dataset validation
# validate()

# run model train and fine tuning
fine_tune()