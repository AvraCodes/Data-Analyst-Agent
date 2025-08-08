from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import os
import json
import subprocess
import re
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

app = FastAPI(title="Data Analyst Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from LLM response"""
    # Try to find code between ```python and ``` or just between ```
    patterns = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
        r'```(.*?)```'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Basic validation
            if any(keyword in code for keyword in ['import', 'def', 'print', 'json']):
                return code
    
    return None

def get_gemini_response(prompt: str) -> str:
    """Get response from Gemini with proper error handling"""
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        
        response = model.generate_content(prompt)
        
        # Extract text properly
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                text = ''.join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                return text.strip()
        
        return ""
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return ""

def analyze_data(questions_content: str, additional_files: List[UploadFile] = None):
    """Main data analysis function"""
    try:
        # Read system prompt
        with open("prompt.txt", "r") as f:
            system_prompt = f.read()

        # Phase 1: Try direct answers
        phase1_prompt = f"""
{system_prompt}

Questions to answer:
{questions_content}

Instructions: Try to answer these questions directly using your knowledge. 
If you can provide ALL answers with confidence, respond with ONLY a clean JSON array/object (depending on the format requested in questions).
If you need to scrape data or cannot answer with certainty, respond with exactly: "NEED_SCRAPING"

Respond now:
"""

        direct_response = get_gemini_response(phase1_prompt)
        logging.info(f"Phase 1 response: {direct_response[:200]}...")

        # Check if we got direct answers
        if direct_response and direct_response != "NEED_SCRAPING":
            try:
                # Try to parse as JSON
                if (direct_response.startswith('[') or direct_response.startswith('{')):
                    answers = json.loads(direct_response)
                    logging.info("Phase 1 successful - returning direct answers")
                    return answers
            except json.JSONDecodeError:
                logging.info("Phase 1 response not valid JSON, proceeding to Phase 2")

        # Phase 2: Generate scraping code
        logging.info("Phase 2: Generating scraping code")
        
        # Detect output format from questions
        is_json_object = "respond with a JSON object" in questions_content.lower()
        output_format = "JSON object" if is_json_object else "JSON array"

        phase2_prompt = f"""
{system_prompt}

Questions to answer:
{questions_content}

Generate a complete Python script that:
1. Scrapes/analyzes the required data
2. Answers ALL questions precisely
3. Outputs results as a {output_format} using: print(json.dumps(answers))

Requirements:
- Use requests, pandas, matplotlib, scipy, duckdb as needed
- Handle all errors gracefully with try/except
- For plots: return base64 PNG data URI under 100KB
- Clean data properly (remove symbols, handle missing values)
- Make the script robust and self-contained

Generate ONLY the Python code:
"""

        code_response = get_gemini_response(phase2_prompt)
        
        if not code_response:
            return ["Error: Could not generate analysis code"]

        # Extract and save code
        code = extract_code_from_response(code_response)
        if not code:
            # If no code blocks found, assume entire response is code
            code = code_response

        with open("test_scraper.py", "w") as f:
            f.write(code)

        logging.info(f"Generated code saved ({len(code)} chars)")

        # Execute the generated code
        result = subprocess.run(
            ["python", "test_scraper.py"],
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes max
        )

        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()

        # Parse results
        try:
            answers = json.loads(output)
            logging.info("Code execution successful")
            return answers
        except json.JSONDecodeError:
            return [f"Execution output: {output[:500]}"]

    except subprocess.TimeoutExpired:
        return ["Error: Analysis timed out (3 minutes exceeded)"]
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return [f"Error: {str(e)}"]

@app.post("/api/")
async def data_analyst_endpoint(
    questions_txt: UploadFile = File(alias="questions.txt"),
    additional_files: List[UploadFile] = File(default=[])
):
    """
    Data Analyst Agent endpoint
    Accepts questions.txt (required) and optional additional files
    """
    try:
        # Read questions
        questions_content = (await questions_txt.read()).decode("utf-8")
        if not questions_content.strip():
            return JSONResponse(
                content={"error": "questions.txt is empty"}, 
                status_code=400
            )

        # Process additional files if any
        file_info = []
        for file in additional_files:
            if file.filename:
                content = await file.read()
                file_info.append({
                    "filename": file.filename,
                    "size": len(content),
                    "type": file.content_type
                })
                # Save file for potential use by generated code
                with open(file.filename, "wb") as f:
                    f.write(content)

        logging.info(f"Processing questions with {len(file_info)} additional files")

        # Analyze data
        result = analyze_data(questions_content, additional_files)
        
        return JSONResponse(content=result)

    except Exception as e:
        logging.error(f"Endpoint error: {str(e)}")
        return JSONResponse(
            content={"error": f"Internal error: {str(e)}"}, 
            status_code=500
        )

@app.get("/")
async def root():
    return {
        "message": "Data Analyst Agent API",
        "usage": "POST /api/ with questions.txt and optional files",
        "example": 'curl -F "questions.txt=@questions.txt" -F "data.csv=@data.csv" http://localhost:8000/api/'
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "data-analyst-agent"}
