import time
import json
import pandas as pd
import re
from mlx_lm import load, generate

# ==========================================
# 1. CONFIGURATION: MODEL SELECTION
# ==========================================

MODELS_TO_TEST = [
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
]

# ==========================================
# 2. PRODUCTION SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """<|im_start|>system
You are the Offerwell Search Agent. Your goal is to split user intent into HARD FACTS (SQL) and VISUAL AESTHETICS (Vector).

### CORE OBJECTIVE
You must generate a `vector_query` that describes **ONLY what is visible in a property photograph**. 
The Vector Database uses CLIP (Computer Vision), which cannot "see" concepts like "safe", "quiet", "near schools", or "good investment".

### INSTRUCTIONS
1. **EXTRACT HARD FILTERS**: specific features, materials, or location categories mentioned explicitly.
2. **GENERATE VECTOR VISUALS**: Convert the user's "Vibe" into physical, photographic descriptors.
3. **APPLY THE CAMERA TEST**: 
   - Can a camera see "Near Hertel"? NO -> Do not put in vector_query.
   - Can a camera see "Exposed Brick"? YES -> Put in vector_query.
   - Can a camera see "Quiet"? NO -> Do not put in vector_query.

### DEFINED FIELDS (STRICT ENUMERATION)
- **features**: [
    "Kitchen", "Chef's Kitchen", "Island", "Pantry",
    "Garage", "Carport", "Driveway", "EV Charger",
    "Backyard", "Fenced Yard", "Pool", "Hot Tub", "Deck", "Patio", "Porch", "Balcony", "Garden",
    "Basement", "Finished Basement", "Attic",
    "Office", "Den", "Gym", "Home Theater", "Mudroom", "Laundry Room",
    "Fireplace", "Wood Stove",
    "Hardwood", "Carpet", "Exposed Brick",
    "Central Air", "AC", "Solar Panels",
    "Single Story", "Open Floor Plan", "ADU", "Guest House", "In-Law Suite", "Loft"
  ]
- **material**: ["Brick", "Stone", "Wood", "Vinyl", "Stucco", "Concrete", "Log", "Metal", "Glass"]
- **location_signals**: [
    "School", "University",
    "Park", "Trail", "Lake", "River", "Ocean", "Beach", "Mountain",
    "Subway", "Bus", "Train", "Highway", "Airport", "Walkable",
    "Coffee", "Bar", "Restaurant", "Grocery", "Shopping",
    "Quiet", "Private", "Gated", "Cul-de-sac", "City", "Country"
  ]

### OUTPUT FORMAT (JSON)
{
  "filters": {
    "features": [],
    "material": [],
    "location_signals": []
  },
  "vector_query": "string"
}

### EXAMPLES

User: "I want a quiet house near Hertel with a storybook vibe."
Output: {
  "filters": {"features": [], "material": [], "location_signals": ["Quiet", "Bar", "Restaurant"]}, 
  "vector_query": "Storybook cottage, tudor style, thatched roof, whimsical, stone exterior, arched doorway, ivy"
}
*Note: "Hertel" (Location) and "Quiet" (Sound) are removed from vector_query. "Storybook" is expanded to visual terms.*

User: "Find me a modern concrete home with solar panels near the subway."
Output: {
  "filters": {"features": ["Solar Panels"], "material": ["Concrete"], "location_signals": ["Subway"]},
  "vector_query": "Modern architecture, brutalist design, concrete facade, minimalist, floor to ceiling windows, flat roof"
}
*Note: "Subway" is removed from vector_query because you cannot see the subway in the house photo.*

User: "Something with a big ugly kitchen I can rip out."
Output: {
  "filters": {"features": ["Kitchen"], "material": [], "location_signals": []},
  "vector_query": "Dated kitchen, old cabinetry, linoleum floor, fluorescent lighting, wood paneling, 1970s style, fixer upper interior"
}
*Note: "Ugly" is translated into specific visual features of an ugly room.*
<|im_end|>"""

# ==========================================
# 3. EXPANDED TEST PROMPTS
# ==========================================

TEST_PROMPTS = [
    # --- Simple / Direct Home Searches ---
    "3 bedroom home with a big backyard in a safe neighborhood under $500,000",
    "Condo near coffee shops and restaurants in a walkable downtown area",
    "Family home near top-rated elementary schools in the suburbs",
    "Quiet neighborhood with low crime and good schools in the Phoenix area",
    "Move-in ready house with a garage in a well-kept neighborhood",
    "Home with a covered porch in a warm climate with mild winters",
    "Townhouse in a walkable area with cafes and parks within walking distance",
    "Single-family home in a neighborhood with young families and good schools",
    "Newer construction home in a growing suburb with strong community feel",
    "Apartment near trendy coffee shops and a farmers market",

    # --- Weather & Climate Preferences ---
    "A home in a warm, sunny climate where it rarely gets below 50 degrees in winter",
    "Something in the Pacific Northwest with a cozy feel — I don't mind the rain but want a dry, insulated home",
    "A property in a mild four-season climate, nothing too extreme in summer or winter",
    "A house in Florida but in an area that doesn't feel the worst of hurricane season",
    "A mountain town home that gets real snow in winter but isn't brutally cold all year",

    # --- Schools & Family-Oriented ---
    "A home zoned for one of the top-rated public high schools in the district",
    "Family neighborhood with highly rated schools, a park nearby, and low traffic streets",
    "Something in a suburb known for strong public schools and safe streets for kids to play outside",
    "A home where I can walk my kids to school and feel safe doing it",
    "Neighborhood with good middle schools, other families around, and enough space for the kids to have their own rooms",

    # --- Safety & Crime ---
    "A neighborhood where I'd feel comfortable going for a jog at night",
    "Low crime area with a strong sense of community — the kind where neighbors look out for each other",
    "Safe, quiet street in a well-established neighborhood, nothing transitional or up-and-coming",
    "A home in a neighborhood that feels genuinely safe, not just technically okay on paper",
    "Family-friendly area with low crime where I don't have to worry about leaving my car in the driveway",

    # --- Walkability & Lifestyle ---
    "Walking distance to a good coffee shop, ideally also near a gym and a grocery store",
    "A neighborhood where I can run errands on foot and grab brunch without getting in the car",
    "Somewhere with a real walkable main street — independent cafes, bookstores, maybe a farmers market on weekends",
    "A place where I can walk to dinner and feel like I'm actually living somewhere, not just sleeping there",
    "Urban or near-urban home where most of my daily needs are within a 10-minute walk",

    # --- Neighborhood Quality & Demographics ---
    "An established, well-maintained neighborhood where homes are clearly taken care of and pride of ownership shows",
    "A neighborhood that feels upscale without being pretentious — good restaurants, clean streets, well-landscaped homes",
    "A diverse, inclusive neighborhood with a strong local identity and community events",
    "A neighborhood on the rise — still affordable but clearly improving, with new businesses coming in",
    "A mature neighborhood with older trees, wide sidewalks, and homes that have character and history",

    # --- Permits & Home Features ---
    "A home with an existing permitted ADU I could rent out for extra income",
    "A property with a finished basement that was properly permitted and isn't a DIY situation",
    "A house where the addition or renovation was done with permits pulled — I don't want surprises",
    "A home with a permitted garage conversion that could work as a home office or guest suite",
    "Something with a pool that was built properly with permits and has been well maintained",

    # --- Complex & Multi-Variable ---
    "I work from home and want a dedicated office space, walkable neighborhood with cafes, and a low-crime area — budget around $550,000",
    "Relocating from NYC with two kids and a dog. Need top schools, a yard, safe streets, and a neighborhood with some personality. Budget is $700,000.",
    "Looking for a forever home in a warm climate, great schools, walkable to at least a few restaurants, and a neighborhood that's clearly well cared for",
    "We want something affordable but in a neighborhood that's heading in the right direction — improving schools, new cafes opening, safer than it was five years ago",
    "Retiring soon and want a low-maintenance home in a warm, walkable town where we can age in place comfortably and feel safe",

    # --- Narrative / Conversational ---
    "My wife and I both work remotely and just want a beautiful, calm neighborhood where we can take walks, grab coffee, and not worry about crime. Price isn't our only concern — neighborhood feel matters more.",
    "First-time buyer here. I want a starter home in a neighborhood I'll actually enjoy living in — safe, some walkability, decent schools even if we don't have kids yet",
    "I grew up in a small town and want that same feeling — neighbors who say hi, kids playing outside, local shops. What does that look like in a mid-size metro area?",
    "We've been burned before by a neighborhood that looked fine online but felt off in person. I want data and real insight on what the neighborhood is actually like day to day",
    "Looking for a home that checks all the boxes — safe, walkable, great schools, good weather — but also just feels like somewhere we'd be proud to live and happy to come home to every day",

    # --- General ---
    "3 bed 2 bath in a safe quiet neighborhood in Buffalo under $400k",
    "Walkable area, good coffee shops, 2 bedroom, don't care about yard",
    "Family home near good schools, big backyard, safe street, around $500k",
    "Something cozy with character, established neighborhood, not a cookie cutter development",
    "Need a home office room, quiet area, close to amenities, budget $450k",
    "Warm weather, low crime, walkable — open to any city, just show me options",
    "Starter home, safe area, decent schools, anything under $350k",
    "Downsizing, want something small and low maintenance in a nice walkable neighborhood",
    "2 bed condo downtown, want to walk to restaurants and coffee, modern building",
    "Moving with two kids, need top schools, yard, safe neighborhood, up to $600k",
]

# ==========================================
# 4. UTILITIES: PARSER & UPGRADED SANITIZER
# ==========================================

def extract_and_parse_json(text):
    """
    Robustly extracts JSON from model output.
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
    Ensures that extracted filters actually exist in the prompt.
    Expanded to handle the new production fields.
    """
    if not data or "filters" not in data:
        return data
        
    user_text = user_query.lower()
    clean_filters = data.get("filters", {})
    
    # 1. Define Synonyms for Common Mismatches
    # If the user says "rental unit", the model outputs "ADU". We need to allow that.
    synonym_map = {
        "ADU": ["rental unit", "granny flat", "guest house", "in-law", "income"],
        "In-Law Suite": ["mother", "granny", "guest", "separate entrance"],
        "EV Charger": ["electric car", "tesla", "plug", "charging"],
        "Solar Panels": ["solar", "energy", "green"],
        "Office": ["work", "zoom", "desk", "den", "study"],
        "Gym": ["workout", "fitness", "yoga", "weights"],
        "Cul-de-sac": ["dead end", "court", "circle"],
        "Hardwood": ["wood floor", "hard wood", "oak", "maple"],
        "Garage": ["car", "parking", "storage"],
        "Fenced Yard": ["fence", "dog", "secure"],
        "Subway": ["train", "metro", "station", "transit"],
        "Walkable": ["walk", "pedestrian"],
    }

    # 2. General Verification Loop
    for category in ["features", "material", "location_signals"]:
        if category in clean_filters:
            verified = []
            for item in clean_filters[category]:
                # Check A: Exact match in text (e.g., user said "Pool")
                if item.lower() in user_text:
                    verified.append(item)
                    continue
                
                # Check B: Synonym match (e.g., user said "Tesla" -> "EV Charger")
                # Look up the item in our map, get list of keywords, check if ANY exist in user text
                keywords = synonym_map.get(item, [])
                if any(k in user_text for k in keywords):
                    verified.append(item)
            
            clean_filters[category] = verified

    data["filters"] = clean_filters
    return data

# ==========================================
# 5. BENCHMARKING ENGINE
# ==========================================

def run_benchmark():
    results = []
    print(f"🚀 Initializing Production Benchmark: {len(MODELS_TO_TEST)} Models | {len(TEST_PROMPTS)} Queries")

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
                # Handle models that don't support system roles elegantly
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                # Fallback format
                instruction = SYSTEM_PROMPT.replace("<|im_start|>system\n", "").strip()
                prompt_content = f"{instruction}\n\nUser query: Map this: '{prompt_text}'"
                input_text = f"<|user|>\n{prompt_content}\n<|assistant|>\n"

            start_time = time.perf_counter()
            response = generate(model, tokenizer, prompt=input_text, max_tokens=500, verbose=False)
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
                "Raw_Output": response[:500] 
            })

    # ==========================================
    # 6. REPORT GENERATION
    # ==========================================
    
    print("\n📊 Compiling Results into Excel...")
    df = pd.DataFrame(results)
    
    with pd.ExcelWriter("RealEstate_Production_Benchmark.xlsx", engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False)
        
        comparison = df.pivot(index="Query", columns="Model", values="SQL_Filters")
        comparison.to_excel(writer, sheet_name="Logic Comparison")
        
        leaderboard = df.groupby("Model").agg({
            "Latency_ms": ["mean", "min", "max"],
            "Status": lambda x: (x == "SUCCESS").sum()
        }).reset_index()
        leaderboard.to_excel(writer, sheet_name="Leaderboard")

    print(f"🎉 Done! Report saved as 'RealEstate_Production_Benchmark.xlsx'")

if __name__ == "__main__":
    run_benchmark()