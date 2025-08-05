from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from typing import Dict, Any
import re
import json
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
import subprocess
import openai
import logging

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging to a file
logging.basicConfig(
    filename="agent_logs.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def get_relevant_data(file_name: str, css_selector: str = None) -> Dict[str, Any]:
    """Extract data from HTML file using BeautifulSoup."""
    with open(file_name, encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    
    if css_selector:
        elements = soup.select(css_selector)
        return {"data": [el.get_text(strip=True) for el in elements]}
    return {"data": soup.get_text(strip=True)}

async def scrape_website(url: str) -> str:
    """Scrape website content using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            # Save content to file
            with open("scraped_content.html", "w", encoding="utf-8") as file:
                file.write(content)
            return content
        except Exception as e:
            return f"Error scraping website: {str(e)}"
        finally:
            await browser.close()

async def scrape_specific_data(url: str, selector: str = None) -> Dict[str, Any]:
    """
    Scrape specific data from a website using selectors.
    Args:
        url: Website URL to scrape
        selector: CSS selector to target specific elements (e.g., "div.article", "h1.title")
    Returns:
        Dictionary containing scraped data
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            # Navigate to the page and wait for content
            await page.goto(url, wait_until="networkidle")
            
            # If selector is provided, wait for it and get specific content
            if selector:
                await page.wait_for_selector(selector)
                elements = await page.query_selector_all(selector)
                data = []
                for element in elements:
                    text = await element.text_content()
                    data.append(text.strip())
                return {"data": data, "url": url}
            
            # If no selector, get full page content
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            return {"data": soup.get_text(strip=True), "url": url}
            
        except Exception as e:
            return {"error": f"Failed to scrape {url}: {str(e)}"}
        finally:
            await browser.close()

def task_breakdown(task: str):
    """Breaks down a task into smaller programmable steps using Google GenAI."""
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        
        # Read prompt template
        try:
            with open('prompt.txt', 'r') as f:
                task_breakdown_prompt = f.read()
        except FileNotFoundError:
            task_breakdown_prompt = """Break down this task into clear steps:

When parsing tables:
- Use BeautifulSoup + pandas.read_html with StringIO.
- If the DataFrame has a MultiIndex as columns, flatten it:
    df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
- Use partial/lowercase substring matching to locate relevant columns.
    Example:
    for col in df.columns:
        if "population" in col.lower() and "density" not in col.lower():
            population_col = col
            break
- If the column isn’t found, print: f"Available columns: {df.columns}" and raise a clear error.
"""
        
        prompt = f"{task_breakdown_prompt}\n{task}"
        response = model.generate_content(prompt)
        
        # Save for debugging
        with open('broken_task.txt', 'w') as f:
            f.write(response.text)
        
        return response.text
    except Exception as e:
        return f"Error in task breakdown: {str(e)}"

async def scrape_with_query(url: str, query: str) -> Dict[str, Any]:
    """
    Scrape website based on a natural language query.
    Args:
        url: Website URL to scrape
        query: Natural language query (e.g., "find all article titles", "get contact information")
    Returns:
        Dictionary containing scraped data
    """
    # Configure Gemini to help determine selectors
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    
    try:
        # Get CSS selector from query using Gemini
        prompt = f"""
        Convert this query into appropriate CSS selectors for web scraping:
        Query: {query}
        Website: {url}
        Return only the CSS selector without explanation.
        """
        response = model.generate_content(prompt)
        selector = response.text.strip()
        
        # Use the generated selector to scrape
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="networkidle")
                await page.wait_for_selector(selector, timeout=5000)
                elements = await page.query_selector_all(selector)
                
                data = []
                for element in elements:
                    text = await element.text_content()
                    data.append(text.strip())
                
                return {
                    "query": query,
                    "url": url,
                    "selector_used": selector,
                    "data": data
                }
            
            except Exception as e:
                return {"error": f"Failed to scrape: {str(e)}"}
            finally:
                await browser.close()
                
    except Exception as e:
        return {"error": f"Failed to process query: {str(e)}"}

async def process_query_file(file_content: str, url: str) -> Dict[str, Any]:
    """
    Process multiple queries from a text file for a given URL.
    Each line in the file is treated as a separate query.
    
    Args:
        file_content: Content of the text file with queries
        url: Website URL to scrape
    Returns:
        Dictionary containing results for each query
    """
    queries = [q.strip() for q in file_content.splitlines() if q.strip()]
    results = []
    
    for query in queries:
        result = await scrape_with_query(url, query)
        results.append({
            "query": query,
            "result": result
        })
    
    return {
        "url": url,
        "total_queries": len(queries),
        "results": results
    }

@app.get("/")
async def root():
    return {"message": "Hello!"}

@app.post("/api/scrape")
async def scrape_url(url: str):
    content = await scrape_website(url)
    return {"content": get_relevant_data("scraped_content.html")}

@app.post("/api/process")
async def process_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        return {"error": "Only .txt files are supported"}
    contents = await file.read()
    text_content = contents.decode('utf-8')
    result = task_breakdown(text_content)
    return {
        "file_name": file.filename,
        "result": result,
        "message": "File processed successfully!"
    }

@app.post("/api/scrape-data")
async def scrape_endpoint(url: str, selector: str = None):
    """API endpoint to scrape specific data from a website"""
    result = await scrape_specific_data(url, selector)
    return result

@app.post("/api/smart-scrape")
async def smart_scrape_endpoint(url: str, query: str):
    """API endpoint that accepts a URL and natural language query"""
    result = await scrape_with_query(url, query)
    return result

@app.post("/api/bulk-scrape")
async def bulk_scrape_endpoint(
    file: UploadFile = File(...),
    url: str = None
):
    """API endpoint that accepts a text file with multiple queries"""
    if not file.filename.endswith('.txt'):
        return {"error": "Only .txt files are supported"}
    if not url:
        return {"error": "URL parameter is required"}
    
    try:
        contents = await file.read()
        text_content = contents.decode('utf-8')
        result = await process_query_file(text_content, url)
        return {
            "file_name": file.filename,
            "url": url,
            "results": result,
            "message": "Queries processed successfully!"
        }
    except Exception as e:
        return {"error": f"Failed to process queries: {str(e)}"}

MAX_RETRIES = 2  # Prevent infinite loops

@app.post("/api/")
async def universal_question_endpoint(file: UploadFile = File(...)):
    question_content = (await file.read()).decode("utf-8")
    with open("prompt.txt", "r") as f:
        prompt_content = f.read()
    llm_prompt = prompt_content + "\n\n" + question_content

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")

    error_message = None
    for attempt in range(MAX_RETRIES):
        prompt_to_send = llm_prompt
        if error_message:
            prompt_to_send += (
                f"\n\nThe previous code failed with this error:\n{error_message}\n"
                "Please fix the code and try again. Only output a complete, corrected Python script."
            )

        response = model.generate_content(prompt_to_send)
        response_text = response.text
        code = extract_code_from_response(response_text)
        logging.info(f"Generated code (attempt {attempt+1}):\n{code}")
        with open("test_scraper.py", "w") as f:
            f.write(code)
        result = subprocess.run(
            ["python", "test_scraper.py"],
            capture_output=True, text=True, timeout=180
        )
        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()
        try:
            answers = json.loads(output)
            return answers  # Success! Return immediately without validation
        except Exception:
            error_message = output
            logging.error(f"Attempt {attempt+1} failed. Error:\n{error_message}\n")

    logging.error(f"Failed after {MAX_RETRIES} attempts. Last error: {error_message}")
    return {"error": f"Failed after {MAX_RETRIES} attempts. Last error: {error_message}"}

def extract_code_from_response(response_text):
    match = re.search(r"```(?:python)?(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

async def process_universal_question(url: str, questions: list):
    """
    Scrape the site and answer the questions.
    """
    # --- Scrape the site ---
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        html = await page.content()
        await browser.close()
    soup = BeautifulSoup(html, "html.parser")

def answer_questions_from_file():
    """
    Reads questions from questions.txt, tries to answer each using Gemini (web/LLM),
    and if needed, generates and executes code to answer, returning a JSON array of answers.
    """
    try:
        # 1. Read questions from file
        with open("questions.txt", "r") as f:
            questions = [q.strip() for q in f.readlines() if q.strip()]
        
        if not questions:
            return {"error": "No questions found in questions.txt"}
        
        answers = []
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        
        for i, question in enumerate(questions):
            logging.info(f"Processing question {i+1}: {question}")
            
            # 2a. Try answering directly via LLM with web search capability
            direct_prompt = f"""
Question: {question}

Can you answer this question directly using your knowledge or general web information? 
If you can provide a confident, accurate answer without needing to scrape specific websites, 
data analysis, or custom code execution, please provide the answer.

If this question requires:
- Scraping specific websites for current data
- Numerical analysis or calculations on scraped data  
- Creating visualizations/plots
- Complex data processing

Then respond with exactly: "CODE_REQUIRED"

Otherwise, provide your direct answer.
"""
            
            try:
                response = model.generate_content(direct_prompt)
                direct_answer = response.text.strip()
                
                # 2b. If direct answer is possible, use it
                if "CODE_REQUIRED" not in direct_answer:
                    answers.append(direct_answer)
                    logging.info(f"Question {i+1} answered directly")
                    continue
                
                # 2c. Generate and execute code for complex questions
                logging.info(f"Question {i+1} requires code generation")
                
                code_prompt = f"""
You need to write a complete Python script to answer this question:
{question}

Requirements:
- Write a complete, standalone Python script
- Include all necessary imports
- Handle errors gracefully
- At the end, print the answer as a JSON array of strings using:
  import json
  print(json.dumps([answer]))
- If creating plots, encode as base64 PNG data URI under 100,000 bytes
- Make the code robust and handle edge cases

Generate the complete Python script:
"""
                
                code_response = model.generate_content(code_prompt)
                code = extract_code_from_response(code_response.text)
                
                if not code:
                    answers.append(f"Error: Could not generate code for question: {question}")
                    continue
                
                logging.info(f"Generated code for question {i+1}:\n{code}")
                
                # Execute the generated code
                temp_file = f"temp_question_{i+1}.py"
                with open(temp_file, "w") as f:
                    f.write(code)
                
                result = subprocess.run(
                    ["python", temp_file],
                    capture_output=True, 
                    text=True, 
                    timeout=180
                )
                
                output = result.stdout.strip()
                if not output:
                    output = result.stderr.strip()
                
                try:
                    # Parse JSON output from code
                    code_answers = json.loads(output)
                    if isinstance(code_answers, list) and len(code_answers) > 0:
                        answers.append(str(code_answers[0]))
                    else:
                        answers.append(str(code_answers))
                    logging.info(f"Question {i+1} answered via code execution")
                    
                except json.JSONDecodeError:
                    # If not valid JSON, use raw output as answer
                    answers.append(f"Code output: {output}")
                    logging.warning(f"Question {i+1} code output was not valid JSON")
                
                # Clean up temp file
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
            except Exception as e:
                error_msg = f"Error processing question '{question}': {str(e)}"
                answers.append(error_msg)
                logging.error(error_msg)
        
        return answers
        
    except FileNotFoundError:
        return {"error": "questions.txt file not found"}
    except Exception as e:
        return {"error": f"Failed to process questions: {str(e)}"}

# Add this endpoint to use the function
@app.post("/api/auto-answer")
async def auto_answer_endpoint():
    """
    Automatically answers questions from questions.txt file
    """
    try:
        answers = answer_questions_from_file()
        return {
            "answers": answers,
            "message": "Questions processed successfully!"
        }
    except Exception as e:
        return {"error": f"Failed to process questions: {str(e)}"}