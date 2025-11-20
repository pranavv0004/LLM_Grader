
#!/usr/bin/env python3
"""
OpenAI System Design Interview Generator

This script generates system design interview conversations using OpenAI API
with the same format as all_samples.json, starting from ID 101.

Usage:
    python openai_sysdesign_generator.py --test           # Generate 1 conversation for testing
    python openai_sysdesign_generator.py --count 50       # Generate 50 conversations
    python openai_sysdesign_generator.py                  # Generate 50 conversations (default)
"""

import os
import time
import random
import re
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================
# Configuration
# =============================
def get_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    return OpenAI(api_key=api_key)

# Initialize OpenAI client
client = get_openai_client()
MODEL_NAME = "gpt-4o-mini"  # Cost-effective model for conversation generation

# Constants for random generation (configurable via command line)
DEFAULT_SAMPLE_COUNT = 50  # Default number of conversations per run
OUTPUT_JSON = "all_samples.json"

LEVEL_CHOICES = ["excellent", "average", "struggling"]
DIFFICULTY_CHOICES = ["easy", "medium", "hard"]
MIN_EXCHANGES, MAX_EXCHANGES = 6, 10

# =============================
# Difficulty-Based Interviewer Prompts
# =============================
INTERVIEWER_DIFFICULTY_PROMPTS = {
    "easy": """
You are a senior system design interviewer conducting a beginner-friendly interview.
Focus on fundamental concepts and basic architecture (e.g., client-server, simple databases, basic caching).
Ask ONE short focused follow-up (<=15 words) that progresses gradually through: basic components -> simple data flow -> basic scaling.
Use concrete but simple examples. Avoid complex distributed systems concepts.
No bullet lists. No multi-sentence output.
Format strictly: <question>
""".strip(),
    "medium": """
You are a senior system design interviewer conducting a progressive interview.
CRITICAL: Build ONLY on the candidate's previous answers; never restate earlier covered basics.
Ask ONE short focused follow-up (<=15 words) that advances: high-level -> components -> scaling -> trade-offs.
Reference prior statements briefly (e.g., "You mentioned cache; how handle eviction?").
Be concrete (QPS, latency, storage). No bullet lists. No multi-sentence output.
Format strictly: <question>
""".strip(),
    "hard": """
You are a senior system design interviewer conducting an advanced-level interview.
Focus on complex distributed systems challenges: consistency models, partitioning strategies, failure scenarios, performance optimization.
Ask ONE short focused follow-up (<=15 words) that dives deep into: CAP trade-offs -> distributed consensus -> failure handling -> performance bottlenecks -> advanced scaling.
Expect specific numbers (millions of QPS, sub-millisecond latency, petabyte scale) and deep technical reasoning.
Challenge assumptions. No bullet lists. No multi-sentence output.
Format strictly: <question>
""".strip(),
}

# =============================
# Kickoff Question Bank (Multiple Questions per Difficulty)
# =============================
KICKOFF_QUESTION_BANK = {
    "easy": [
        "Design a basic URL shortening service; what's your approach?",
        "Design a simple online bookstore; how would you structure it?",
        "Design a basic chat application; what components do you need?",
        "Design a simple to-do list app; what's your database design?",
        "Design a basic image upload service; how would you store images?",
        "Design a simple blog platform; what's your high-level architecture?",
        "Design a basic voting system; how would you count votes?",
        "Design a simple notification service; what's your approach?",
    ],
    "medium": [
        "Design a scalable URL shortening service; what's your high-level approach?",
        "Design a news feed system for 10M users; what's your architecture?",
        "Design a rate limiter handling 100K requests/sec; how would you build it?",
        "Design a video streaming platform; what are your key components?",
        "Design a ride-sharing service like Uber; what's your matching strategy?",
        "Design a distributed cache system; how do you handle consistency?",
        "Design a real-time leaderboard for 1M gamers; what's your approach?",
        "Design a scalable web crawler; how do you avoid duplicates?",
        "Design a food delivery system; how do you handle real-time tracking?",
        "Design a messaging system like WhatsApp; what's your delivery guarantee?",
    ],
    "hard": [
        "Design a globally distributed URL shortening service handling 100K writes/sec; what's your architecture?",
        "Design YouTube's video recommendation engine at petabyte scale; how do you optimize latency?",
        "Design a distributed transaction system with strict consistency; what's your consensus protocol?",
        "Design Google Search's indexing pipeline processing 100M pages/hour; what's your partitioning strategy?",
        "Design a global payment system handling 1M TPS with <100ms latency; how handle failures?",
        "Design Twitter's timeline at 500M users with <50ms p99 latency; what's your caching strategy?",
        "Design a distributed file system like GFS storing 100PB; how ensure consistency?",
        "Design a real-time bidding system processing 10M bids/sec; what's your optimization approach?",
        "Design Netflix's CDN serving 200M concurrent streams; how minimize buffering globally?",
        "Design a distributed database with multi-region writes; how resolve conflicts?",
    ],
}

# =============================
# System / Persona Prompts
# =============================
STUDENT_PROMPTS = {
    "excellent": """
You are an excellent system design candidate.
Answer in ONE precise sentence (<20 words) using concrete terms (throughput numbers, components, trade-offs) confidently.
No filler. Format: <answer>
""".strip(),
    "average": """
You are an average system design candidate.
Answer in ONE sentence (<20 words) mixing some specifics and mild uncertainty ("maybe", "probably").
Brief filler allowed. Format: <answer>
""".strip(),
    "struggling": """
You are a struggling system design candidate.
Answer in ONE short sentence (<15 words) with visible uncertainty or partial gaps ("Um", "I guess").
Often simplistic or incomplete. Format: <answer>
""".strip(),
}

# =============================
# Hesitation / Speech Patterns
# =============================
HESITATION_PATTERNS = {
    "excellent": {"rate": 0.05, "patterns": ["Right...", "Actually..."]},
    "average": {"rate": 0.35, "patterns": ["Hmm...", "I think...", "Maybe...", "Probably..."]},
    "struggling": {"rate": 0.65, "patterns": ["Um...", "Uh...", "I guess...", "Let me think..."]},
}

# =============================
# Helper Functions
# =============================
def call_openai_chat(messages: List[Dict[str, str]], max_tokens: int = 50) -> str:
    """Make a call to OpenAI Chat API with retry logic."""
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=30
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle rate limiting
            if "rate_limit" in error_msg or "429" in error_msg:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
                
            # Handle other API errors
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                raise
                
            time.sleep(base_delay * (2 ** attempt))
    
    return ""

def trim_response_length(text: str, max_sentences: int = 1, max_words: int = 18) -> str:
    """Trim response to specified length limits."""
    # Extract sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        sentences = sentences[:max_sentences]
        result = '. '.join(sentences)
    else:
        result = text.strip()
    
    # Ensure terminal punctuation
    if result and result[-1] not in ".!?":
        result += '.'
    
    # Word trim
    words = result.split()
    if len(words) > max_words:
        result = ' '.join(words[:max_words]) + '...'
    
    return result

def apply_hesitation(body: str, student_level: str) -> str:
    """Apply hesitation patterns based on student level."""
    cfg = HESITATION_PATTERNS[student_level]
    lower = body.lower()
    starts_hes = any(lower.startswith(p.split('...')[0].strip(' .').lower()) for p in cfg["patterns"])
    
    # Remove existing hesitation sometimes if too frequent
    if starts_hes and random.random() > cfg["rate"]:
        body = re.sub(r'^(right|actually|hmm+|um+|uh+|i think|maybe|probably|i guess|let me think)[, .\-–—]*', '', body, flags=re.I).lstrip()
    elif not starts_hes and random.random() < cfg["rate"]:
        body = f"{random.choice(cfg['patterns'])} {body}".strip()
    
    return body

def postprocess_candidate(raw: str, student_level: str) -> str:
    """Process candidate response."""
    raw = raw.strip().replace('\n', ' ')
    raw = trim_response_length(raw, max_sentences=1, max_words=18)
    raw = apply_hesitation(raw, student_level)
    return raw

def postprocess_interviewer(raw: str) -> str:
    """Process interviewer response."""
    raw = raw.strip().replace('\n', ' ')
    raw = trim_response_length(raw, max_sentences=1, max_words=15)
    return raw

def to_turn(speaker: str, text: str) -> Dict[str, str]:
    """Convert speaker and text into structured turn."""
    return {"speaker": speaker, "text": text}

# =============================
# Conversation Generation
# =============================
def generate_single_conversation(student_level: str, difficulty: str, num_exchanges: int, conv_id: int) -> Dict[str, Any]:
    """Generate a single interview conversation."""
    
    # Select random kickoff question
    kickoff_text = random.choice(KICKOFF_QUESTION_BANK[difficulty])
    
    # Initialize conversation history and turns
    turns = [to_turn("Interviewer", kickoff_text)]
    conversation_history = [f"Interviewer: {kickoff_text}"]
    
    # Generate conversation exchanges
    for exchange_num in range(num_exchanges - 1):
        # Candidate answers the last interviewer question
        candidate_messages = [
            {"role": "system", "content": STUDENT_PROMPTS[student_level]},
            {"role": "user", "content": conversation_history[-1]}
        ]
        
        candidate_raw = call_openai_chat(candidate_messages, max_tokens=60)
        candidate_response = postprocess_candidate(candidate_raw, student_level)
        
        turns.append(to_turn("Candidate", candidate_response))
        conversation_history.append(f"Candidate: {candidate_response}")
        
        # Interviewer asks follow-up (except for the last exchange)
        if exchange_num < num_exchanges - 2:
            interviewer_messages = [
                {"role": "system", "content": INTERVIEWER_DIFFICULTY_PROMPTS[difficulty]},
                {"role": "user", "content": "\n".join(conversation_history)}
            ]
            
            interviewer_raw = call_openai_chat(interviewer_messages, max_tokens=50)
            interviewer_response = postprocess_interviewer(interviewer_raw)
            
            turns.append(to_turn("Interviewer", interviewer_response))
            conversation_history.append(f"Interviewer: {interviewer_response}")
        
        # Small delay to avoid hitting rate limits
        time.sleep(0.1)
    
    # Build structured conversation
    return {
        "id": conv_id,
        "student_level": student_level,
        "difficulty": difficulty,
        "exchanges": num_exchanges,
        "kickoff_question": kickoff_text,
        "turns": turns,
        "meta": {
            "model": MODEL_NAME,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    }

def load_existing_conversations() -> List[Dict[str, Any]]:
    """Load existing conversations from the output file."""
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️  Could not read existing file {OUTPUT_JSON}. Starting fresh.")
    
    return []

def save_conversations(conversations: List[Dict[str, Any]]):
    """Save conversations to the output file."""
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

def main():
    """Main function to generate conversations with test mode support."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate system design conversations using OpenAI API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python openai_sysdesign_generator.py --test           # Generate 1 conversation for testing
  python openai_sysdesign_generator.py --count 50       # Generate 50 conversations
  python openai_sysdesign_generator.py                  # Generate 50 conversations (default)
        """
    )
    parser.add_argument('--test', action='store_true', help='Run in test mode (generate only 1 conversation)')
    parser.add_argument('--count', type=int, default=DEFAULT_SAMPLE_COUNT, help=f'Number of conversations to generate (default: {DEFAULT_SAMPLE_COUNT})')
    
    args = parser.parse_args()
    
    # Determine sample count and mode
    if args.test:
        sample_count = 1
        print(f"🧪 Running in TEST mode - generating 1 conversation")
        print("=" * 60)
    else:
        sample_count = args.count
        print(f"🚀 Running in FULL mode - generating {sample_count} conversations")
        print("=" * 60)
    
    print(f"Model: {MODEL_NAME}")
    print(f"Target: {sample_count} new conversations")
    
    # Load existing conversations  
    existing_conversations = load_existing_conversations()
    # Force start from ID 102 regardless of existing conversations
    start_id = 102
    
    if existing_conversations:
        print(f"📂 Found {len(existing_conversations)} existing conversations")
        max_existing_id = max(conv.get('id', 0) for conv in existing_conversations)
        if max_existing_id >= 102:
            start_id = max_existing_id + 1
        print(f"🆔 Starting new conversations from ID: {start_id}")
    else:
        print(f"📄 No existing conversations found, starting from ID: {start_id}")
    
    print()
    
    # Generate new conversations
    new_conversations = []
    failed_count = 0
    
    desc = "Testing conversation" if args.test else "Generating conversations"
    for i in tqdm(range(sample_count), desc=desc):
        conv_id = start_id + i
        
        # Random configuration
        student_level = random.choice(LEVEL_CHOICES)
        difficulty = random.choice(DIFFICULTY_CHOICES)
        num_exchanges = random.randint(MIN_EXCHANGES, MAX_EXCHANGES)
        
        # Show details in test mode
        if args.test:
            print(f"\\n📋 Generating conversation with:")
            print(f"   ID: {conv_id}")
            print(f"   Student Level: {student_level}")
            print(f"   Difficulty: {difficulty}")
            print(f"   Target Exchanges: {num_exchanges}")
            print()
        
        try:
            conversation = generate_single_conversation(
                student_level, difficulty, num_exchanges, conv_id
            )
            new_conversations.append(conversation)
            
            # Show success in test mode
            if args.test:
                print(f"✅ Successfully generated conversation ID {conv_id}")
                print(f"   Actual exchanges: {conversation['exchanges']}")
                print(f"   Generated at: {conversation['meta']['generated_at']}")
                print(f"   Kickoff question: {conversation['kickoff_question'][:100]}...")
            elif (i + 1) % 10 == 0:
                print(f"✓ Generated {i + 1}/{sample_count} conversations")
                
        except Exception as e:
            print(f"❌ Failed to generate conversation {conv_id}: {e}")
            failed_count += 1
            # Continue with next conversation
            continue
    
    # Combine and save
    all_conversations = existing_conversations + new_conversations
    save_conversations(all_conversations)
    
    # Summary
    print("\n" + "=" * 60)
    if args.test:
        print("🧪 TEST GENERATION COMPLETE")
    else:
        print("✅ GENERATION COMPLETE")
    print("=" * 60)
    
    print(f"📊 Generated: {len(new_conversations)} new conversations")
    if failed_count > 0:
        print(f"⚠️  Failed: {failed_count} conversations")
    print(f"📈 Total conversations: {len(all_conversations)}")
    if len(new_conversations) > 0:
        print(f"🆔 ID range: {start_id} to {start_id + len(new_conversations) - 1}")
    print(f"💾 Saved to: {OUTPUT_JSON}")
    
    # Test mode specific messages
    if args.test:
        if len(new_conversations) > 0:
            print("\n🎯 Test Results:")
            print(f"✅ Successfully generated test conversation")
            print(f"🚀 Ready to run full generation with: --count {DEFAULT_SAMPLE_COUNT}")
        else:
            print("\n❌ Test Failed:")
            print("🔧 Please check your OpenAI API key and configuration")
    
    # Configuration summary (only for multiple conversations)
    if len(new_conversations) > 1:
        print("\n📋 Generated Configurations:")
        config_counts = {}
        for conv in new_conversations:
            key = f"{conv['student_level']}-{conv['difficulty']}-{conv['exchanges']}ex"
            config_counts[key] = config_counts.get(key, 0) + 1
        
        for config, count in sorted(config_counts.items()):
            print(f"  {config}: {count}")
    
    if len(new_conversations) > 0:
        print(f"\n💰 Estimated API cost: ~${len(new_conversations) * 0.001:.3f}")

if __name__ == "__main__":
    main()