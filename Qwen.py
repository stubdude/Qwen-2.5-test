import time
import json
import pandas as pd
import re  # <--- NEW: Required for robust parsing
from mlx_lm import load, generate

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODELS_TO_TEST = [
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit", # The Smartest
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit", # The Fastest
]

# --- YOUR EXACT SYSTEM PROMPT ---
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

# The Gauntlet: 55 Real-World Queries
TEST_PROMPTS = [
    # --- The 50 Realistic Queries ---
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

    # --- The 5 Stress Tests (Boundary Pushing) ---
    "I want a house that feels like it's in the woods, but is actually in the city.",
    "Show me the cheapest house where I won't get robbed.",
    "I need a place where I can play my drums at 11 PM without the cops showing up.",
    "Find me a house that looks ugly now but is in a neighborhood where everyone else is renovating.",
    "I want a kitchen exactly like the one in 'Something's Gotta Give'."
]

# ==========================================
# 2. THE PARSER (The Fix for 1.5B)
# ==========================================

def extract_and_parse_json(text):
    """
    Robust JSON extractor.
    1. Removes Markdown Code Blocks (```json ... ```)
    2. Finds the outermost { ... } using regex
    3. Parses it.
    """
    try:
        # Strategy 1: Clean Markdown
        clean_text = text.replace("```json", "").replace("```", "").strip()
        
        # Strategy 2: Regex Hunt for { ... }
        # This looks for the first '{' and the last '}' and grabs everything in between
        # re.DOTALL allows the dot to match newlines
        match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: maybe we manually added '{' in the generation loop?
            json_str = clean_text

        return json.loads(json_str)
    except:
        return None # Failed to parse

def sanitize_filters(user_query, data):
    """
    Removes hallucinated filters that don't appear in the user's text.
    """
    try:
        clean_filters = data.get("filters", {})
        user_text = user_query.lower()
        
        # 1. Verify Materials
        if "material" in clean_filters:
            verified_materials = []
            synonyms = {
                "Stone": ["stone", "rock", "granite", "limestone", "masonry"],
                "Brick": ["brick", "masonry"],
                "Wood": ["wood", "timber", "cedar", "log", "cabin"],
                "Stucco": ["stucco", "plaster"]
            }
            for mat in clean_filters["material"]:
                keywords = synonyms.get(mat, [mat.lower()])
                if any(k in user_text for k in keywords):
                    verified_materials.append(mat)
            clean_filters["material"] = verified_materials

        # 2. Verify Features
        if "features" in clean_filters:
            verified_features = []
            for feat in clean_filters["features"]:
                if feat.lower() in user_text:
                    verified_features.append(feat)
                elif feat == "Garage" and any(x in user_text for x in ["car", "truck", "parking"]):
                     verified_features.append(feat)
                elif feat == "Pool" and "swim" in user_text:
                     verified_features.append(feat)
                elif feat == "Fireplace" and "fire" in user_text:
                     verified_features.append(feat)
                elif feat == "Office" and ("work" in user_text or "zoom" in user_text or "desk" in user_text):
                     verified_features.append(feat)
            clean_filters["features"] = verified_features

        data["filters"] = clean_filters
        return data # Return Object, not String
        
    except:
        return data

# ==========================================
# 3. THE ENGINE
# ==========================================

def run_benchmark():
    results = []
    print(f"🚀 Starting Benchmark on {len(MODELS_TO_TEST)} models...")

    for model_path in MODELS_TO_TEST:
        short_name = model_path.split("/")[-1]
        print(f"📥 Loading: {short_name}...")
        
        try:
            model, tokenizer = load(model_path)
        except Exception as e:
            print(f"❌ Error loading {short_name}: {e}")
            continue
        
        start_batch_time = time.perf_counter()
        
        for i, prompt_text in enumerate(TEST_PROMPTS):
            if i % 10 == 0:
                print(f"   ...processing query {i+1}/{len(TEST_PROMPTS)}")

            # Prepare Input
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Map this: '{prompt_text}'"}
            ]
            
            # Note: We do NOT force the '{' character here anymore.
            # We let 1.5B generate the markdown block, and we clean it up later.
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # --- TIMER START ---
            start_time = time.perf_counter()
            
            response = generate(
                model, 
                tokenizer, 
                prompt=input_text, 
                max_tokens=300, # Increased max tokens for safe JSON generation
                verbose=False
            )
            
            # --- TIMER END ---
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # 1. PARSE (Regex Rescue)
            data = extract_and_parse_json(response)
            
            if data:
                # 2. SANITIZE
                data = sanitize_filters(prompt_text, data)
                
                # Format for Excel
                filters_str = json.dumps(data.get("filters", {}))
                vector_str = data.get("vector_query", "ERROR")
                formatted_output = f"SQL: {filters_str}\nVECTOR: {vector_str}"
            else:
                formatted_output = "JSON PARSE ERROR: " + response
                filters_str = "Error"
                vector_str = "Error"

            results.append({
                "Model": short_name,
                "Prompt": prompt_text,
                "Full Output": formatted_output,
                "SQL (Cleaned)": filters_str,
                "Vector (Vibe)": vector_str,
                "Time (ms)": round(latency_ms, 2)
            })

        total_batch_time = time.perf_counter() - start_batch_time
        print(f"✅ Finished {short_name} in {round(total_batch_time, 2)}s\n")

    # ==========================================
    # 4. EXCEL EXPORT
    # ==========================================
    
    print("📊 Generating Excel Report...")
    df = pd.DataFrame(results)
    output_filename = "Offerwell_Benchmark_Fixed.xlsx"

    # Create Pivot Tables
    quality_pivot = df.pivot(index="Prompt", columns="Model", values="Full Output")
    speed_pivot = df.pivot(index="Prompt", columns="Model", values="Time (ms)")
    
    # Leaderboard
    leaderboard = df.groupby("Model")["Time (ms)"].agg(['mean', 'min', 'max']).reset_index()

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False)
        quality_pivot.to_excel(writer, sheet_name="Compare Outputs")
        speed_pivot.to_excel(writer, sheet_name="Compare Speed")
        leaderboard.to_excel(writer, sheet_name="Leaderboard", index=False)

    print(f"🎉 Benchmark Complete! Results saved to: {output_filename}")

if __name__ == "__main__":
    run_benchmark()