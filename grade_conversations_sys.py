#!/usr/bin/env python3
"""
LLM Conversation Grader

This script grades system design interview conversations using GPT-5.1
according to predefined rubrics.
"""

import json
import os
import time
from typing import Dict, List, Any
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConversationGrader:
    def __init__(self):
        """Initialize the grader with OpenAI client and load rubrics."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5.1"
        self.rubrics = self._load_rubrics()
        
    def _load_rubrics(self) -> List[str]:
        """Load rubric criteria from rubrics_sys.txt file."""
        try:
            with open('rubrics_sys.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract rubric names (lines that start with a number)
            rubrics = []
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit() and '.' in line:
                    # Extract rubric name (everything after the number and period)
                    rubric_name = line.split('.', 1)[1].strip()
                    rubrics.append(rubric_name)
            
            return rubrics
        except FileNotFoundError:
            raise FileNotFoundError("rubrics_sys.txt file not found. Please ensure it exists in the current directory.")
    
    def _format_conversation(self, conversation: Dict[str, Any]) -> str:
        """Format a conversation for LLM evaluation."""
        turns = conversation.get('turns', [])
        formatted_turns = []
        
        for turn in turns:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            formatted_turns.append(f"{speaker}: {text}")
        
        conversation_text = '\n'.join(formatted_turns)
        
        metadata = f"""
Conversation Metadata:
- Student Level: {conversation.get('student_level', 'N/A')}
- Difficulty: {conversation.get('difficulty', 'N/A')}
- Number of Exchanges: {conversation.get('exchanges', 'N/A')}
- Kickoff Question: {conversation.get('kickoff_question', 'N/A')}

Conversation:
{conversation_text}
        """
        
        return metadata.strip()
    
    def _create_grading_prompt(self, conversation_text: str) -> str:
        """Create the grading prompt for the LLM."""
        rubric_list = '\n'.join([f"{i+1}. {rubric}" for i, rubric in enumerate(self.rubrics)])
        
        prompt = f"""
You are an expert system design interview evaluator. Please grade the following conversation between an interviewer and candidate according to these 10 rubrics:

{rubric_list}

Scoring Scale:
- 0: Not addressed at all, or fundamentally incorrect/problematic responses
- 1: Partially addressed with basic understanding, some correct elements but lacks depth or completeness
- 2: Thoroughly addressed with excellent detail, demonstrating strong expertise and comprehensive understanding

IMPORTANT: Be fair and generous in your evaluation. If a candidate demonstrates solid understanding and provides reasonable technical solutions, award appropriate scores. Score based on what the candidate DOES demonstrate, not what they might have missed. Consider both unprompted insights and responses to interviewer questions.

{conversation_text}

Please provide your evaluation as a JSON object with this exact format:
{{
    "Problem Understanding & Requirement Gathering": <score 0-2>,
    "Structured Problem-Solving Approach": <score 0-2>,
    "High-Level Architecture & Design Evolution": <score 0-2>,
    "Technical Depth & Implementation Details": <score 0-2>,
    "Scalability & Performance Reasoning": <score 0-2>,
    "Trade-off Analysis & Decision Justification": <score 0-2>,
    "Handling Follow-up Questions & Adaptability": <score 0-2>,
    "Reliability & Fault Tolerance Considerations": <score 0-2>,
    "Communication & Collaboration": <score 0-2>,
    "Completeness & Time tracking": <score 0-2>
}}

Respond with ONLY the JSON object, no additional text.
        """
        
        return prompt.strip()
    
    def _parse_llm_response(self, response: str) -> Dict[str, int]:
        """Parse the LLM response to extract scores."""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            scores = json.loads(json_str)
            
            # Validate that all rubrics are present and scores are 0-2
            expected_rubrics = [
                "Problem Understanding & Requirement Gathering",
                "Structured Problem-Solving Approach", 
                "High-Level Architecture & Design Evolution",
                "Technical Depth & Implementation Details",
                "Scalability & Performance Reasoning",
                "Trade-off Analysis & Decision Justification",
                "Handling Follow-up Questions & Adaptability",
                "Reliability & Fault Tolerance Considerations",
                "Communication & Collaboration",
                "Completeness & Time tracking"
            ]
            
            validated_scores = {}
            for rubric in expected_rubrics:
                if rubric in scores:
                    score = int(scores[rubric])
                    if 0 <= score <= 2:
                        validated_scores[rubric] = score
                    else:
                        validated_scores[rubric] = 0  # Default to 0 if invalid score
                else:
                    validated_scores[rubric] = 0  # Default to 0 if missing
            
            return validated_scores
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            # Return default scores if parsing fails
            return {rubric: 0 for rubric in [
                "Problem Understanding & Requirement Gathering",
                "Structured Problem-Solving Approach",
                "High-Level Architecture & Design Evolution",
                "Technical Depth & Implementation Details",
                "Scalability & Performance Reasoning",
                "Trade-off Analysis & Decision Justification",
                "Handling Follow-up Questions & Adaptability",
                "Reliability & Fault Tolerance Considerations",
                "Communication & Collaboration",
                "Completeness & Time tracking"
            ]}
    
    def grade_conversation(self, conversation: Dict[str, Any]) -> Dict[str, int]:
        """Grade a single conversation using GPT-5.1."""
        conversation_text = self._format_conversation(conversation)
        prompt = self._create_grading_prompt(conversation_text)
        
        try:
            # GPT-5.1 uses the Responses API instead of Chat Completions
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                reasoning={"effort": "medium"},  # Use medium reasoning for balanced speed and quality
                text={"verbosity": "low"},  # Low verbosity for concise output
                max_output_tokens=1500
            )
            
            response_text = response.output_text
            scores = self._parse_llm_response(response_text)
            
            # Add a small delay to respect rate limits
            time.sleep(0.5)
            
            return scores
            
        except Exception as e:
            error_msg = str(e)
            print(f"\nError grading conversation {conversation.get('id', 'unknown')}: {error_msg}")
            
            # Handle rate limit errors specifically
            if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                print("⚠️ Rate limit hit. Waiting 60 seconds before retry...")
                time.sleep(60)
                # Retry once
                try:
                    response = self.client.responses.create(
                        model=self.model,
                        input=prompt,
                        reasoning={"effort": "medium"},
                        text={"verbosity": "low"},
                        max_output_tokens=1500
                    )
                    response_text = response.output_text
                    scores = self._parse_llm_response(response_text)
                    time.sleep(0.5)
                    return scores
                except Exception as retry_error:
                    print(f"Retry failed: {retry_error}")
            
            # Return default scores on error
            return {rubric: 0 for rubric in [
                "Problem Understanding & Requirement Gathering",
                "Structured Problem-Solving Approach",
                "High-Level Architecture & Design Evolution",
                "Technical Depth & Implementation Details", 
                "Scalability & Performance Reasoning",
                "Trade-off Analysis & Decision Justification",
                "Handling Follow-up Questions & Adaptability",
                "Reliability & Fault Tolerance Considerations",
                "Communication & Collaboration",
                "Completeness & Time tracking"
            ]}
    
    def grade_all_conversations(self, input_file: str = 'all_samples.json', 
                              output_file: str = 'graded_conversations.json',
                              test_mode: bool = False,
                              batch_size: int = None) -> None:
        """Grade all conversations and save results."""
        print("Loading conversations...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {input_file} not found.")
        
        # Load existing grades if they exist
        existing_grades = {}
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_grades = json.load(f)
            print(f"📁 Found {len(existing_grades)} existing grades")
        except FileNotFoundError:
            print("📁 No existing grades found, starting fresh")
        
        # Filter out already graded conversations
        ungraded_conversations = []
        for conv in conversations:
            conv_id = conv.get('id')
            if conv_id is None:
                print(f"Warning: Conversation missing ID, skipping...")
                continue
            if str(conv_id) not in existing_grades:
                ungraded_conversations.append(conv)
        
        if test_mode:
            ungraded_conversations = ungraded_conversations[:1]  # Only process first ungraded conversation
            print("🧪 TEST MODE: Processing only the first ungraded conversation")
        elif batch_size and batch_size > 0:
            ungraded_conversations = ungraded_conversations[:batch_size]
            print(f"📦 BATCH MODE: Processing {len(ungraded_conversations)} conversations")
        
        print(f"📊 Total conversations: {len(conversations)}")
        print(f"📊 Already graded: {len(existing_grades)}")
        print(f"📊 Need to grade: {len(ungraded_conversations)}")
        print(f"Using model: {self.model}")
        print("Starting grading process...\n")
        
        if len(ungraded_conversations) == 0:
            print("✅ All conversations are already graded!")
            return
        
        new_results = {}
        
        for conversation in tqdm(ungraded_conversations, desc="Grading conversations"):
            conv_id = conversation.get('id')
            scores = self.grade_conversation(conversation)
            new_results[str(conv_id)] = scores
            
            if test_mode:
                print(f"\n📊 Test Results for Conversation {conv_id}:")
                for rubric, score in scores.items():
                    print(f"  {rubric}: {score}")
                print(f"\nTotal Score: {sum(scores.values())}/20")
        
        # Merge with existing results
        all_results = {**existing_grades, **new_results}
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Grading complete! Results saved to {output_file}")
        print(f"📊 Newly processed: {len(new_results)} conversations")
        print(f"📊 Total graded conversations: {len(all_results)}")


def main():
    """Main function to run the grading process."""
    print("🤖 LLM Conversation Grader")
    print("=" * 50)
    
    # Check if user wants test mode or batch mode
    import sys
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    # Check for batch size argument
    batch_size = None
    for arg in sys.argv:
        if arg.startswith("--batch="):
            try:
                batch_size = int(arg.split("=")[1])
                print(f"📦 Batch mode: Processing {batch_size} conversations at a time")
            except ValueError:
                print("⚠️ Invalid batch size, ignoring...")
    
    if test_mode:
        print("🧪 Running in TEST MODE")
    
    try:
        grader = ConversationGrader()
        grader.grade_all_conversations(test_mode=test_mode, batch_size=batch_size)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. .env file exists with OPENAI_API_KEY set")
        print("2. all_samples.json file exists")  
        print("3. rubrics_sys.txt file exists")
        print("4. Virtual environment is activated and dependencies installed")
        print("\nUsage:")
        print("  python grade_conversations_sys.py              # Grade all conversations")
        print("  python grade_conversations_sys.py --test       # Test with 1 conversation")
        print("  python grade_conversations_sys.py --batch=50   # Grade 50 conversations at a time")


if __name__ == "__main__":
    main()