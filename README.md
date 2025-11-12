# LLM Conversation Grader

This tool automatically grades system design interview conversations using GPT-4o-mini according to predefined rubrics.

## Features

- Grades 50 conversations from `all_samples.json`
- Uses 10 evaluation criteria from `rubrics.txt`
- Scores each conversation on a 0-2 scale per criterion
- Outputs results in JSON format
- Includes rate limiting and error handling

## Quick Setup

### Windows (Recommended)

1. **Run the setup script:**
   ```cmd
   setup.bat
   ```

2. **Add your OpenAI API key:**
   - Edit the `.env` file that was created
   - Replace `your_openai_api_key_here` with your actual OpenAI API key

3. **Run the grader:**
   ```cmd
   venv\Scripts\activate
   python grade_conversations.py
   ```

### Manual Setup (All Platforms)

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux  
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Run the grader:**
   ```bash
   python grade_conversations.py
   ```

## Input Files Required

- `all_samples.json` - Conversation data (already exists)
- `rubrics.txt` - Grading criteria (already exists)
- `.env` - OpenAI API key configuration

## Output

The script creates `graded_conversations.json` with this format:

```json
{
  "1": {
    "Problem Understanding & Requirement Gathering": 2,
    "Structured Problem-Solving Approach": 1,
    "High-Level Architecture & Design Evolution": 2,
    "Technical Depth & Implementation Details": 1,
    "Scalability & Performance Reasoning": 2,
    "Trade-off Analysis & Decision Justification": 1,
    "Handling Follow-up Questions & Adaptability": 2,
    "Reliability & Fault Tolerance Considerations": 0,
    "Communication & Collaboration": 2,
    "Completeness & Time tracking": 1
  },
  "2": {
    // ... more conversations
  }
}
```

## Scoring Scale

- **0**: Not addressed or fundamentally incorrect responses
- **1**: Basic responses with some correct elements
- **2**: Excellent, detailed responses demonstrating expertise

## Cost Estimation

- Model: GPT-4o-mini ($0.150 per 1M input tokens, $0.600 per 1M output tokens)
- Estimated cost for 50 conversations: ~$0.50-$1.00
- Processing time: ~5-10 minutes (with rate limiting)

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Ensure `.env` file exists and contains your API key
- Activate the virtual environment before running

**"FileNotFoundError"**
- Ensure `all_samples.json` and `rubrics.txt` exist in the project directory

**Rate limit errors**
- The script includes automatic delays between requests
- If you hit limits, wait a few minutes and restart

## Files Created

- `venv/` - Virtual environment directory
- `.env` - Environment configuration (add to .gitignore)
- `graded_conversations.json` - Output results

## Dependencies

- `openai>=1.3.0` - OpenAI API client
- `python-dotenv>=1.0.0` - Environment variable loading
- `tqdm>=4.65.0` - Progress bar display