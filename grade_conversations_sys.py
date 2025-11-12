#!/usr/bin/env python3
"""
LLM Conversation Grader

This script grades system design interview conversations using GPT-4o-mini
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
        self.model = "gpt-4o-mini"
        self.rubrics = self._load_rubrics()
        
    def _load_rubrics(self) -> List[str]:
        """Load rubric criteria from rubrics.txt file."""
        try:
            with open('rubrics.txt', 'r', encoding='utf-8') as f:
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
            raise FileNotFoundError("rubrics.txt file not found. Please ensure it exists in the current directory.")
    
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
- 0: Not addressed or fundamentally incorrect responses
- 1: Basic responses with some correct elements  
- 2: Excellent, detailed responses demonstrating expertise

Grade based ONLY on what actually happened in the conversation turns. Consider both unprompted insights and responses to interviewer questions.

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
        """Grade a single conversation using GPT-4o-mini."""
        conversation_text = self._format_conversation(conversation)
        prompt = self._create_grading_prompt(conversation_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent grading
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            scores = self._parse_llm_response(response_text)
            
            # Add a small delay to respect rate limits
            time.sleep(0.5)
            
            return scores
            
        except Exception as e:
            print(f"Error grading conversation {conversation.get('id', 'unknown')}: {e}")
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
                              test_mode: bool = False) -> None:
        """Grade all conversations and save results."""
        print("Loading conversations...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {input_file} not found.")
        
        if test_mode:
            conversations = conversations[:1]  # Only process first conversation
            print("🧪 TEST MODE: Processing only the first conversation")
        
        print(f"Found {len(conversations)} conversations to grade.")
        print(f"Using model: {self.model}")
        print("Starting grading process...\n")
        
        results = {}
        
        for conversation in tqdm(conversations, desc="Grading conversations"):
            conv_id = conversation.get('id')
            if conv_id is None:
                print(f"Warning: Conversation missing ID, skipping...")
                continue
                
            scores = self.grade_conversation(conversation)
            results[conv_id] = scores
            
            if test_mode:
                print(f"\n📊 Test Results for Conversation {conv_id}:")
                for rubric, score in scores.items():
                    print(f"  {rubric}: {score}")
                print(f"\nTotal Score: {sum(scores.values())}/20")
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Grading complete! Results saved to {output_file}")
        print(f"Processed {len(results)} conversations.")


def main():
    """Main function to run the grading process."""
    print("🤖 LLM Conversation Grader")
    print("=" * 50)
    
    # Check if user wants test mode
    import sys
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    if test_mode:
        print("🧪 Running in TEST MODE")
    
    try:
        grader = ConversationGrader()
        grader.grade_all_conversations(test_mode=test_mode)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. .env file exists with OPENAI_API_KEY set")
        print("2. all_samples.json file exists")  
        print("3. rubrics.txt file exists")
        print("4. Virtual environment is activated and dependencies installed")
        print("\nUsage:")
        print("  python grade_conversations.py        # Grade all conversations")
        print("  python grade_conversations.py --test # Test with 1 conversation")


if __name__ == "__main__":
    main()