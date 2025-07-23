from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import os
from datetime import datetime
import logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
from io import BytesIO
import time
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path


# Load environment variables
load_dotenv()

app = FastAPI(title="OpenAI Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

ASSISTANT_ID = "asst_DV1ZHZBemzwymJSjqV8XlMCy"  # track it globally

def create_or_get_assistant():
    global ASSISTANT_ID
    assistant = client.beta.assistants.create(
        name="AI Assistant of Institute of Health",
        instructions=(
            "You are a knowledgeable and polite assistant for the Institute of Health. "
            "You help students by answering questions based on transcribed lecture documents uploaded by the faculty. "
            "Use the content of these documents to give clear, helpful, and accurate answers to student queries. "
            "Do not refer to the documents, files, or document's names in your responses. "
            "Strictly do not add file names in the response. and also do not add any source in the response just answer to their question and that's it."
            "Do not mention uploaded files, lecture titles, just provide clean, natural answers. "
            "If the user greets you, respond politely and briefly (e.g., 'Hello! How can I assist you today?'). "
            "Avoid any statements about being an AI or making disclaimers. Keep the tone professional and student-friendly."
        ),
        model="gpt-4o",  # or gpt-4-turbo/gpt-4o-mini
        tools=[{"type": "file_search"}]
    )
    ASSISTANT_ID = assistant.id
    logger.info(f"Created assistant: {ASSISTANT_ID}")
    return assistant

def upload_and_vectorize(file_content: bytes, filename: str, store_name: str):

    # Step 1: Upload raw PDF to OpenAI (for Vision + text extraction)
    file_obj = BytesIO(file_content)
    file_obj.name = filename
    openai_file = client.files.create(file=file_obj, purpose="user_data")
    logger.info(f"Uploaded for OCR/text: {filename}, id={openai_file.id}")

    # Step 2: Use Vision-capable model (e.g., gpt-4o) to extract text
    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "user", "content": [
                {"type": "input_file", "file_id": openai_file.id},
                {"type": "input_text", "text": "Please extract the full text from this PDF."}
            ]}
        ]
    )
    extracted_text = resp.choices[0].message.content.strip()
    logger.info(f"Extracted text from {filename}: {len(extracted_text)} chars")

    # Step 3: Upload extracted text as a plain .txt file
    text_obj = BytesIO(extracted_text.encode("utf-8"))
    text_obj.name = filename.replace(".pdf", ".txt")
    txt_file = client.files.create(file=text_obj, purpose="assistants")
    logger.info(f"Uploaded extracted text file: {txt_file.id}")

    # Step 4: Create vector store and embed the text file
    vector = client.vector_stores.create(name=store_name)
    batch = client.vector_stores.file_batches.create(
        vector_store_id=vector.id, file_ids=[txt_file.id]
    )
    while True:
        batch = client.vector_stores.file_batches.poll(vector_store_id=vector.id, batch_id=batch.id)
        if batch.status in ("completed", "failed"):
            break
        time.sleep(1)

    return txt_file.id, vector.id


def update_assistant_store(assistant_id, vector_store_id):
    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store_id]
            }
        }
    )
    logger.info(f"Updated assistant with vector store {vector_store_id}")
    return assistant


def save_thread_id(thread_id: str, user_info: str = "unknown"):
    """Save thread ID to a tracking file"""
    threads_file = Path("thread_history.txt")
    
    thread_data = {
        "thread_id": thread_id,
        "created_at": datetime.now().isoformat(),
        "user_info": user_info,
        "status": "active"
    }
    
    # Create file if it doesn't exist
    if not threads_file.exists():
        threads_file.touch()
        logger.info("Created new thread history file")
    
    # Append new thread data
    with open(threads_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(thread_data) + "\n")
    
    logger.info(f"Saved thread ID: {thread_id}")

def load_thread_history():
    """Load all thread IDs from the tracking file"""
    threads_file = Path("thread_history.txt")
    threads = []
    
    if threads_file.exists():
        try:
            with open(threads_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        threads.append(json.loads(line.strip()))
        except Exception as e:
            logger.error(f"Error reading thread history: {e}")
    
    return threads

def get_thread_count():
    """Get total number of threads created"""
    return len(load_thread_history())



# @app.post("/upload")
# async def upload_pdfs(file: List[UploadFile] = File(...), title: str = Form(None)):
#     if not ASSISTANT_ID:
#         create_or_get_assistant()

#     uploaded_file_ids = []
#     file_objects = []
#     filenames = []

#     for f in file:
#         if not f.filename.lower().endswith(".pdf"):
#             raise HTTPException(400, f"Only PDF files allowed, but got: {f.filename}")
        
#         content = await f.read()
#         file_obj = BytesIO(content)
#         file_obj.name = f.filename

#         # Upload file to OpenAI for assistant reference
#         openai_file = client.files.create(file=file_obj, purpose="assistants")
#         logger.info(f"Uploaded file {f.filename}, id={openai_file.id}")

#         uploaded_file_ids.append(openai_file.id)
#         filenames.append(f.filename)

#         # Rewind for vector store
#         file_obj.seek(0)
#         file_objects.append(file_obj)

#     # Create shared vector store
#     vector_store_name = f"Business_{title or filenames[0]}"
#     vector = client.vector_stores.create(name=vector_store_name)
#     logger.info(f"Created vector store {vector.id}")

#     # Upload all file objects to vector store
#     file_batch = client.vector_stores.file_batches.upload_and_poll(
#         vector_store_id=vector.id,
#         files=file_objects
#     )
#     logger.info(f"File batch uploaded: {file_batch.file_counts}")

#     # Wait for vector store readiness
#     while True:
#         vector = client.vector_stores.retrieve(vector_store_id=vector.id)
#         if vector.status == "completed":
#             break
#         time.sleep(1)

#     # Attach vector store to assistant
#     update_assistant_store(ASSISTANT_ID, vector.id)

#     return {
#         "success": True,
#         "message": f"{len(filenames)} PDF(s) uploaded & processed",
#         "data": {
#             "filenames": filenames,
#             "file_ids": uploaded_file_ids,
#             "vector_store_id": vector.id,
#             "assistant_id": ASSISTANT_ID
#         }
#     }


@app.post("/upload")
async def upload_pdfs(file: List[UploadFile] = File(...), title: str = Form(None)):
    if not ASSISTANT_ID:
        create_or_get_assistant()

    uploaded_file_ids = []
    file_objects = []
    filenames = []

    for f in file:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files allowed, but got: {f.filename}")
        
        content = await f.read()
        file_obj = BytesIO(content)
        file_obj.name = f.filename

        # Upload the original PDF to OpenAI
        openai_file = client.files.create(file=file_obj, purpose="assistants")
        uploaded_file_ids.append(openai_file.id)
        filenames.append(f.filename)
        logger.info(f"Uploaded {f.filename}, id={openai_file.id}")

        # Extract text using GPT-4o with vision capabilities
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_id": openai_file.id}},
                    {"type": "text", "text": "You are a PDF summarization expert. Read the entire provided PDF and generate a complete and detailed summary that includes all relevant context, key details, and supporting information from every section of the document. Your summary must not skip, omit, or ignore any part, including author names, individual contributions, section titles, data, and even the smallest details. Ensure that every section, paragraph, figure, reference, link, and concept is accurately represented in the summary. If the document includes URLs, references, citations, or footnotes, include them exactly as they appear in the original text—do not reword or remove them. The goal is to produce a fully comprehensive, information-rich, and structured summary of the entire content, while still maintaining clarity and conciseness. Do not alter the meaning, generalize critical points, or leave out any important or minor content. Return only the complete summary as your response, with no additional commentary, disclaimers, or meta-statements."}
                ]
            }],
            temperature=0
        )

        extracted_text = response.choices[0].message.content
        print(f"Extracted from {f.filename}:\n{extracted_text}\n")

        # Convert extracted text to in-memory .txt file
        txt_file = BytesIO(extracted_text.encode("utf-8"))
        txt_file.name = f.filename.replace(".pdf", ".txt")
        file_objects.append(txt_file)

    # Create a vector store
    vector_store_name = f"Business_{title or filenames[0]}"
    vector = client.vector_stores.create(name=vector_store_name)
    logger.info(f"Created vector store {vector.id}")

    # Upload all .txt files to vector store
    file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector.id,
        files=file_objects
    )
    logger.info(f"File batch uploaded: {file_batch.file_counts}")

    # Wait for vector store to complete
    while True:
        vector = client.vector_stores.retrieve(vector_store_id=vector.id)
        if vector.status == "completed":
            break
        time.sleep(1)

    # Attach vector store to assistant
    update_assistant_store(ASSISTANT_ID, vector.id)

    return {
        "success": True,
        "message": f"{len(filenames)} PDF(s) uploaded & processed",
        "data": {
            "filenames": filenames,
            "file_ids": uploaded_file_ids,
            "vector_store_id": vector.id,
            "assistant_id": ASSISTANT_ID
        }
    }


@app.post("/chat")
async def chat_with_assistant(message: str = Form(...), thread_id: str = Form(None)):
    if not ASSISTANT_ID:
        raise HTTPException(400, "Upload a document first")

    # Validate thread_id or create one
    if not thread_id or not thread_id.strip().startswith("thread_"):
        thread = client.beta.threads.create()
        thread_id = thread.id
        
        # Save new thread ID to file
        save_thread_id(thread_id, f"User message: {message[:50]}...")
        logger.info(f"New thread created and saved: {thread_id}")
    else:
        try:
            thread = client.beta.threads.retrieve(thread_id)
            logger.info(f"Using existing thread: {thread_id}")
        except Exception:
            thread = client.beta.threads.create()
            thread_id = thread.id
            
            # Save fallback thread ID to file
            save_thread_id(thread_id, f"Fallback thread for: {message[:50]}...")
            logger.info(f"Fallback thread created and saved: {thread_id}")

    # Add user message
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message
    )

    # Run with instructions so assistant knows its role
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        instructions=(
            "Please answer the user's question clearly and accurately, using only the uploaded documents. "
            "Do not include any file names, lecture titles in your response. "
            "Strictly do not add file names in the response. and also do not add any source in the response just answer to their question and that's it."
            "Keep answers natural, informative, and easy to understand. "
            "Avoid filler, disclaimers, or references to the source documents. "
            "If the input is a greeting or casual message, reply politely and briefly without offering extra help unless asked."
        )

    )

    if run.status == "completed":
        msgs = client.beta.threads.messages.list(thread_id)
        
        # Get the most recent message (first in the list) which should be the assistant's response
        latest_message = msgs.data[0]
        
        # Verify it's from the assistant (not the user)
        if latest_message.role == "assistant":
            resp = latest_message.content[0].text.value
        else:
            # Fallback: look for the first assistant message
            assistant_message = next((msg for msg in msgs.data if msg.role == "assistant"), None)
            if assistant_message:
                resp = assistant_message.content[0].text.value
            else:
                raise HTTPException(500, "No assistant response found")
        
        return {
            "success": True,
            "response": resp,
            "thread_id": thread_id,
            "run_id": run.id
        }

    raise HTTPException(500, f"Run failed: {run.status}")

@app.get("/list-files")
async def list_files():
    files = client.files.list(purpose="assistants")
    return {"files": [{"id": f.id, "filename": f.filename, "created_at": f.created_at} for f in files.data]}

@app.delete("/delete-file/{file_id}")
async def delete_file(file_id: str):
    client.files.delete(file_id)
    return {"success": True, "message": f"Deleted file {file_id}"}

@app.delete("/cleanup")
async def cleanup_all():
    # Delete assistant files
    files = client.files.list(purpose="assistants").data
    for f in files:
        client.files.delete(f.id)
    
    # Delete vector stores
    vss = client.vector_stores.list().data
    for vs in vss:
        client.vector_stores.delete(vector_store_id=vs.id)

    return {"success": True, "deleted_files": len(files), "deleted_vector_stores": len(vss)}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "assistant_id": ASSISTANT_ID
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
