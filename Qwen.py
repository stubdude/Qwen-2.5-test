import time
import json
import pandas as pd
import re
from mlx_lm import load, generate

# ==========================================
# 1. CONFIGURATION: THE SIX OPTIONS
# ==========================================

# 1. Qwen2.5-7B: High reasoning, best for complex stress tests.
# 2. Llama-3.2-3B: The industry standard for balanced SLM performance.
# 3. Qwen2.5-1.5B: High-density intelligence for structured output.
# 4. Phi-3.5-mini: Excellent at logic and "reasoning" through constraints.
# 5. Gemma-2-2B: Superior linguistic flair for the 'vector_query' generation.
# 6. Qwen2.5-0.5B: Ultra-low latency for edge/mobile deployments.

MODELS_TO_TEST = [
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/gemma-2-2b-it-4bit",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
]

# --- SYSTEM PROMPT (PER USER REQUEST: UNCHANGED) ---
SYSTEM_PROMPT = """<|im_start|>system
You are a real estate Search Agent, responsible for extracting relevant information from user queries to facilitate property searches. Your primary objectives are to identify HARD FILTERS (explicit facts, no deviations) and SOFT SEARCH (vibes or implicit preferences) from user input.

### INSTRUCTIONS
1. **EXPLICIT FILTERS ONLY**: Extract filters (features, material, location_signals) if the user explicitly mentions them. Do not infer or assume information not provided. Only use the features, material, and location_signals that are explicitly mentioned.
2. **DEFAULT VALUES**: If a user does not specify a material or any features, return an empty list [] for the respective field.
3. **AVOID ASSUMPTIONS**: Refrain from making assumptions about the user's preferences unless explicitly stated.

### DEFINED FIELDS FOR FILTERS
- **features**: ["Kitchen", "Garage", "Backyard", "Basement", "Pool", "Porch", "Driveway", "Office", "Fireplace"]
- **material**: ["Stone", "Brick", "Wood", "Stucco"]
- **location_signals**: ["School", "Park", "Bars", "Water", "Highway", "Subway", "Quiet"]

### OUTPUT REQUIREMENTS
Provide your response in the following JSON format:
{
"filters": {
"features": [...],  // List of explicitly mentioned features
"material": [...],  // List of explicitly mentioned materials
"location_signals": [...]  // List of explicitly mentioned location signals
},
"vector_query": "...",  // A string representing the soft search query (vibes or implicit preferences)
}


### EXAMPLE RESPONSES
- For the query "I want a cozy cottage near the lake.", the output should be:
{
"filters": {
"features": [],
"material": [],
"location_signals": ["Water"]
},
"vector_query": "Cozy cottage, lakefront, water view, cabin aesthetic, rustic, warm lighting"
}

- For the query "Find me a modern brick house with a garage.", the output should be:
{
"filters": {
"features": ["Garage"],
"material": ["Brick"],
"location_signals": []
},
"vector_query": "Modern architecture, red brick, contemporary design, sleek"
}

- For the query "A place to walk to get coffee.", the output should be:
{
"filters": {
"features": [],
"material": [],
"location_signals": []
},
"vector_query": "Walkable neighborhood, coffee shops nearby, sunday morning vibe, urban village"
}

Given a user query, please extract the relevant filters and generate an appropriate vector query, following the specified rules and output format.
<|im_end|>"""

# ==========================================
# 2. TEST PROMPTS (55 Real + 5 Stress)
# ==========================================

TEST_PROMPTS = [
    "I need a place where I can walk to get coffee on Sunday mornings.",
    "Find me a house that looks like a cottage from a storybook.",
    "Something with a big ugly kitchen that I can rip out and redo myself.",
    "A backyard that feels totally private, where I won't see my neighbors.",
    "I have three dogs, so I need a massive fenced yard, not just a patio.",
    "A place with huge windows—I have a lot of houseplants that need light.",
    "Show me homes near the elementary school so my kids can walk there safely.",
    "I want a house that feels 'historic' but doesn't have drafty old windows.",
    "A condo where I won't hear the person upstairs walking around.",
    "Something close to the bars on Hertel but far enough away that it’s quiet at night.",
    "A starter home that isn't falling apart, maybe just needs some paint.",
    "I need a garage big enough for my truck and a workbench.",
    "Is there a place where I can legally build a rental unit in the back for extra income?",
    "A house with a porch where I can actually sit and watch the rain.",
    "Find me a modern-looking apartment, I hate carpet and popcorn ceilings.",
    "A place near the park with the good running trails.",
    "I want a kitchen with an island where my friends can hang out while I cook.",
    "Something single-story—my knees can't do stairs anymore.",
    "A house that smells like old wood and books, like a library.",
    "I need a spare room that would make a good quiet office for Zoom calls.",
    "Find me a cheap house in a neighborhood that’s getting better.",
    "A place with a driveway, I'm tired of fighting for street parking.",
    "I want a dining room big enough for my grandmother's 10-person table.",
    "Show me homes near the highway on-ramp, I have a long commute.",
    "A place with a 'granny flat' or separate entrance for my mom.",
    "I need a house with gas cooking, electric stoves are a dealbreaker.",
    "Something with a finished basement where the kids can play video games.",
    "A house with character—arches, crown molding, weird little nooks.",
    "I want to live near other young families, not a retirement community.",
    "Find me a place with a view of the water, even if it's just a sliver.",
    "A house with a big front tree that’s perfect for a tire swing.",
    "I need a laundry room on the main floor, not in the scary basement.",
    "Something with a low-maintenance yard, I travel too much to mow grass.",
    "A place that gets good afternoon sun in the living room.",
    "I want a bathroom that feels like a spa, with a deep tub.",
    "Find me a house near a grocery store so I don't have to drive for milk.",
    "A place with exposed brick, I love that industrial loft look.",
    "I need a house that’s not in a flood zone, I’m worried about the basement.",
    "Something with a fireplace for the winter, real wood if possible.",
    "A house where I can have chickens in the backyard without the city fining me.",
    "I want a place with a 'mudroom' area for all our winter boots and coats.",
    "Find me a house that feels bright and airy, not dark and cave-like.",
    "A place near the subway station so I can sell my second car.",
    "I need a backyard that’s flat enough for a skating rink in the winter.",
    "Something with two actual bathrooms, not one and a half.",
    "A house where I can build a massive deck for summer parties.",
    "I want a bedroom that’s dark and quiet, away from the street lights.",
    "Find me a place with 'good bones' that I can restore over time.",
    "A house near the community center where they have the seniors' pottery class.",
    "Something that feels safe for a single woman living alone.",
    "Find me a stucco house with a pool in a quiet neighborhood.", # Verification test
    "A stone house with a basement and a fireplace.", # Verification test
    "I want a wood cabin near a park with a garage.", # Verification test
    "Show me a brick home near the subway with a porch.", # Verification test
    "A modern place with a driveway near the school.", # Verification test
    # --- The 5 Stress Tests ---
    "I want a house that feels like it's in the woods, but is actually in the city.",
    "Show me the cheapest house where I won't get robbed.",
    "I need a place where I can play my drums at 11 PM without the cops showing up.",
    "Find me a house that looks ugly now but is in a neighborhood where everyone else is renovating.",
    "I want a kitchen exactly like the one in 'Something's Gotta Give'."
]

# ==========================================
# 3. UTILITIES: PARSER & SANITIZER
# ==========================================

def extract_and_parse_json(text):
    """
    Robustly extracts JSON from model output, handling markdown blocks and stray text.
    """
    try:
        clean_text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(clean_text)
    except:
        return None

def sanitize_filters(user_query, data):
    """
    Ensures that extracted filters actually exist in the prompt to prevent hallucinations.
    """
    if not data or "filters" not in data:
        return data
        
    user_text = user_query.lower()
    clean_filters = data.get("filters", {})
    
    # Material Verification
    if "material" in clean_filters:
        synonyms = {
            "Stone": ["stone", "rock", "granite"],
            "Brick": ["brick", "masonry"],
            "Wood": ["wood", "timber", "cedar", "log", "cabin"],
            "Stucco": ["stucco", "plaster"]
        }
        clean_filters["material"] = [
            m for m in clean_filters["material"] 
            if any(syn in user_text for syn in synonyms.get(m, [m.lower()]))
        ]

    # Features Verification
    if "features" in clean_filters:
        mapping = {
            "Garage": ["garage", "car", "truck", "parking"],
            "Backyard": ["yard", "backyard", "garden", "outdoor"],
            "Office": ["office", "work", "zoom", "desk", "den"],
            "Pool": ["pool", "swim"]
        }
        verified = []
        for feat in clean_filters["features"]:
            keywords = mapping.get(feat, [feat.lower()])
            if any(k in user_text for k in keywords):
                verified.append(feat)
        clean_filters["features"] = verified

    data["filters"] = clean_filters
    return data

# ==========================================
# 4. BENCHMARKING ENGINE
# ==========================================

def run_benchmark():
    results = []
    print(f"🚀 Initializing Benchmark: 6 Models | 60 Queries")

    for model_path in MODELS_TO_TEST:
        short_name = model_path.split("/")[-1]
        print(f"\n📥 Loading Model: {short_name}...")
        
        try:
            model, tokenizer = load(model_path)
        except Exception as e:
            print(f"❌ Failed to load {short_name}: {e}")
            continue
        
        for i, prompt_text in enumerate(TEST_PROMPTS):
            if i % 15 == 0:
                print(f"   📊 Progress: {i}/{len(TEST_PROMPTS)} queries completed.")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Map this: '{prompt_text}'"}
            ]

            try:
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                if "System role not supported" in str(e):
                    # Gemma and some models don't support system role - fold instruction into user message
                    instruction = SYSTEM_PROMPT.replace("<|im_start|>system\n", "").strip()
                    messages = [{"role": "user", "content": f"{instruction}\n\nUser query: Map this: '{prompt_text}'"}]
                    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    raise

            start_time = time.perf_counter()
            response = generate(model, tokenizer, prompt=input_text, max_tokens=400, verbose=False)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse and Sanitize
            raw_data = extract_and_parse_json(response)
            clean_data = sanitize_filters(prompt_text, raw_data)
            
            if clean_data:
                filters_str = json.dumps(clean_data.get("filters", {}))
                vector_str = clean_data.get("vector_query", "N/A")
                status = "SUCCESS"
            else:
                filters_str = "PARSE_ERROR"
                vector_str = "PARSE_ERROR"
                status = "FAILED"

            results.append({
                "Model": short_name,
                "Query": prompt_text,
                "Status": status,
                "SQL_Filters": filters_str,
                "Vector_Vibe": vector_str,
                "Latency_ms": round(latency_ms, 2),
                "Raw_Output": response[:500] # Truncated for Excel readability
            })

    # ==========================================
    # 5. REPORT GENERATION
    # ==========================================
    
    print("\n📊 Compiling Results into Excel...")
    df = pd.DataFrame(results)
    
    with pd.ExcelWriter("RealEstate_Agent_Benchmark_2026.xlsx", engine='openpyxl') as writer:
        # Full Raw Data
        df.to_excel(writer, sheet_name="Raw Data", index=False)
        
        # Cross-Model Logic Comparison
        comparison = df.pivot(index="Query", columns="Model", values="SQL_Filters")
        comparison.to_excel(writer, sheet_name="Logic Comparison")
        
        # Performance Leaderboard
        leaderboard = df.groupby("Model").agg({
            "Latency_ms": ["mean", "min", "max"],
            "Status": lambda x: (x == "SUCCESS").sum()
        }).reset_index()
        leaderboard.to_excel(writer, sheet_name="Leaderboard")

    print(f"🎉 Done! Report saved as 'RealEstate_Agent_Benchmark_2026.xlsx'")

if __name__ == "__main__":
    run_benchmark()