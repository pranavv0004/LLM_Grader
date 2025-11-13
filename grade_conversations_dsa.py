#!/usr/bin/env python3
"""
DSA Interview Conversation Grader

This script grades DSA interview conversations using GPT-4o-mini
according to predefined rubrics.
"""

import json
import os
import time
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DSAConversationGrader:
    def __init__(self):
        """Initialize the grader with OpenAI client and load rubrics."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.rubrics = self._load_rubrics()
        
    def _load_rubrics(self) -> List[str]:
        """Load rubric criteria from rubrics_dsa.txt file."""
        try:
            with open('rubrics_dsa.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract rubric names from the DSA rubrics file
            rubrics = [
                "Ask clarifying questions",
                "Propose brute force", 
                "Provide space+time complexity",
                "Reach optimal sol finally",
                "Handle edge cases",
                "Correct explanation of approach",
                "Polite + respectful tone",
                "Logical progression of conversation"
            ]
            
            return rubrics
        except FileNotFoundError:
            raise FileNotFoundError("rubrics_dsa.txt file not found. Please ensure it exists in the current directory.")
    
    def _format_conversation(self, conversation: Dict[str, Any]) -> str:
        """Format a DSA conversation for LLM evaluation."""
        conv_parts = conversation.get('conversation', [])
        formatted_turns = []
        
        for turn in conv_parts:
            role = turn.get('role', 'Unknown')
            parts = turn.get('parts', '')
            
            # Map roles to more readable names
            if role == 'user':
                speaker = 'Interviewer'
            elif role == 'model':
                # Check if this is interviewer or interviewee based on content
                if parts.strip().startswith('Interviewer:'):
                    speaker = 'Interviewer'
                    parts = parts.replace('Interviewer:', '').strip()
                elif parts.strip().startswith('Interviewee:'):
                    speaker = 'Interviewee'
                    parts = parts.replace('Interviewee:', '').strip()
                else:
                    # Determine based on context - if previous was user, this is likely interviewee
                    if formatted_turns and formatted_turns[-1].startswith('Interviewer:'):
                        speaker = 'Interviewee'
                    else:
                        speaker = 'Interviewer'
            else:
                speaker = role
                
            formatted_turns.append(f"{speaker}: {parts}")
        
        conversation_text = '\n'.join(formatted_turns)
        
        metadata = f"""
DSA Interview Metadata:
- Problem: {conversation.get('prompt', 'N/A')}
- Student Type: {conversation.get('student_type', 'N/A')}
- Interview ID: {conversation.get('id', 'N/A')}

Conversation:
{conversation_text}
        """
        
        return metadata.strip()
    
    def _create_grading_prompt(self, conversation_text: str) -> str:
        """Create the grading prompt for the LLM."""
        rubric_list = '\n'.join([f"{i+1}. {rubric}" for i, rubric in enumerate(self.rubrics)])
        
        prompt = f"""
You are an expert DSA (Data Structures and Algorithms) interview evaluator. Please grade the following conversation between an interviewer and candidate according to these 8 rubrics:

{rubric_list}

Scoring Scale:
- 0: Not addressed or fundamentally incorrect responses
- 1: Basic responses with some correct elements  
- 2: Excellent, detailed responses demonstrating expertise

Grade based ONLY on what actually happened in the conversation turns. Look for evidence of each rubric in the candidate's responses and behavior throughout the interview.

{conversation_text}

Please provide your evaluation as a JSON object with this exact format:
{{
    "Ask clarifying questions": <score 0-2>,
    "Propose brute force": <score 0-2>,
    "Provide space+time complexity": <score 0-2>,
    "Reach optimal sol finally": <score 0-2>,
    "Handle edge cases": <score 0-2>,
    "Correct explanation of approach": <score 0-2>,
    "Polite + respectful tone": <score 0-2>,
    "Logical progression of conversation": <score 0-2>
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
                "Ask clarifying questions",
                "Propose brute force", 
                "Provide space+time complexity",
                "Reach optimal sol finally",
                "Handle edge cases",
                "Correct explanation of approach",
                "Polite + respectful tone",
                "Logical progression of conversation"
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
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            # Return default scores of 0 for all rubrics
            return {rubric: 0 for rubric in self.rubrics}
    
    def grade_conversation(self, conversation: Dict[str, Any]) -> Dict[str, int]:
        """Grade a single conversation."""
        try:
            # Format the conversation for evaluation
            formatted_conversation = self._format_conversation(conversation)
            
            # Create the grading prompt
            prompt = self._create_grading_prompt(formatted_conversation)
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent grading
                max_tokens=500
            )
            
            # Parse the response
            llm_response = response.choices[0].message.content
            scores = self._parse_llm_response(llm_response)
            
            return scores
            
        except Exception as e:
            print(f"Error grading conversation: {e}")
            # Return default scores of 0 for all rubrics
            return {rubric: 0 for rubric in self.rubrics}
    
    def grade_all_conversations(self, input_file: str, output_file: str, test_mode: bool = False):
        """Grade all conversations in the input file and save results."""
        try:
            # Load conversations
            with open(input_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            print(f"Loaded {len(conversations)} DSA interviews from {input_file}")
            
            # In test mode, only process the first conversation
            if test_mode:
                conversations = conversations[:1]
                print("Running in TEST MODE - processing only 1 conversation")
            
            results = {}
            
            # Process conversations with progress bar
            for conversation in tqdm(conversations, desc="Grading DSA interviews"):
                conv_id = conversation.get('id', 0)
                
                if test_mode:
                    print(f"\nGrading conversation ID: {conv_id}")
                
                scores = self.grade_conversation(conversation)
                
                if test_mode:
                    print(f"Scores: {scores}")
                
                # Store results with ID as key
                results[conv_id] = scores
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.1)
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            print(f"\n✅ Grading complete! Results saved to {output_file}")
            
            if not test_mode:
                # Print summary statistics
                total_conversations = len(results)
                avg_scores = {}
                
                for rubric in self.rubrics:
                    total_score = sum(result.get(rubric, 0) for result in results.values())
                    avg_scores[rubric] = total_score / total_conversations if total_conversations > 0 else 0
                
                print(f"\n📊 Summary Statistics ({total_conversations} conversations):")
                for rubric, avg_score in avg_scores.items():
                    print(f"  {rubric}: {avg_score:.2f}/2.00")
                
        except FileNotFoundError:
            print(f"❌ Error: Input file '{input_file}' not found.")
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON in input file '{input_file}'.")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function to run the grader."""
    parser = argparse.ArgumentParser(description='Grade DSA interview conversations using GPT-4o-mini')
    parser.add_argument('--test', action='store_true', help='Run in test mode (grade only 1 conversation)')
    parser.add_argument('--input', default='all_interviews(1)_dsa.json', help='Input JSON file with conversations')
    parser.add_argument('--output', default='graded_dsa_conversations.json', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Initialize grader
    try:
        grader = DSAConversationGrader()
        print("✅ DSA Conversation Grader initialized successfully!")
        print(f"📋 Loaded {len(grader.rubrics)} rubric criteria")
        
        if args.test:
            print("🧪 Running in TEST mode...")
        
        # Grade conversations
        grader.grade_all_conversations(args.input, args.output, args.test)
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())