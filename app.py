import streamlit as st
import json
import requests
import base64
from io import BytesIO
from PIL import Image
import time
import pandas as pd
import re
import zipfile

# --- 1. CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio 12/9")

# Model IDs
MODEL_IMAGE_GEN = "models/gemini-3-pro-image-preview"
MODEL_TEXT_GEMINI = "models/gemini-3.1-pro-preview"
MODEL_TEXT_GEMINI_FALLBACK = "models/gemini-3-pro-preview"

# Claude Models
CLAUDE_MODELS = {
    "Claude Sonnet 4.5": "claude-sonnet-4-5-20250929",
    "Claude Opus 4.6": "claude-opus-4-6",
}

# OpenAI Models (Chat Completions API compatible)
OPENAI_MODELS = {
    "GPT-5.2": "gpt-5.2",
}

# --- HELPER: CLEANER ---
def clean_key(value):
    if value is None: return ""
    return str(value).strip().replace(" ", "").replace('"', "").replace("'", "").replace("\n", "")

# --- HELPER: SAFE IMAGE LOADER ---
def safe_st_image(url, width=None, caption=None):
    if not url: return
    try:
        clean_url = str(url).strip().replace(" ", "").replace("\n", "")
        if clean_url.startswith("http"):
            st.image(clean_url, width=width, caption=caption)
    except Exception:
        st.warning("⚠️ Image unavailable")

# --- PROMPTS ---
SEO_PROMPT_POST_GEN = """
You are an SEO specialist with 15-20 years of experience. 
Help write SEO-optimized image file name with image alt tags in English for the product image with a model created, having product details according to this url: {product_url}
To rank well on organic search engines by customer groups interested in this type of product.
IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PROMPT_SMART_GEN = """
You are an SEO & Visual Content Specialist for Jewelry e-commerce.
Your task is to generate an SEO-optimized **Image File Name** and **Alt Tag** based on the visual description and product context provided.

**Inputs:**
1. **Visual Instruction (The image is generated from this):** "{context}"
2. **Product Reference URL (Context):** "{product_url}"

**Instructions:**
- **File Name:** Create a lowercase, hyphenated file name ending in .jpg (e.g., `silver-ring-blue-gemstone-side-view.jpg`).
    - COMBINE keywords from the URL (if valid) with the VISUAL details from the instruction.
    - Do NOT simply copy the URL slug. The filename MUST describe the visual look of the image (e.g., pose, angle, lighting, material).
- **Alt Tag:** Write a natural English sentence describing the image for accessibility and SEO. Mention the material, stone, and style visible in the instruction.

IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PROMPT_IMAGE_ANALYSIS = """
You are an SEO & Visual Content Specialist for Jewelry & Leather Product e-commerce with 15-20 years of experience.
Your task is to analyze the GENERATED IMAGE provided and create SEO-optimized **Image File Name** and **Alt Tag**.

**Product Reference URL:** "{product_url}"

**Instructions:**
1. **ANALYZE THE IMAGE** - Look at the actual generated image and describe what you see:
   - Type of jewelry (ring, necklace, bracelet, earrings, etc.)
   - Materials visible (gold, silver, platinum, etc.)
   - Gemstones (diamond, sapphire, ruby, etc.)
   - Style (modern, vintage, minimalist, luxury, etc.)
   - Visual elements (model wearing it, product shot, lifestyle, etc.)
   - Background and lighting style

2. **File Name:** Create a lowercase, hyphenated file name ending in .jpg
   - COMBINE product keywords from URL with VISUAL details from the image
   - Include: material, product type, style, and visual context
   - Example: `gold-diamond-ring-elegant-hand-model-lifestyle.jpg`

3. **Alt Tag:** Write a natural English sentence describing exactly what is shown in the image
   - Be specific about what you SEE in the image
   - Mention materials, style, and context visible
   - Good for accessibility and SEO

IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PROMPT_BULK_EXISTING = """
คุณคือ SEO specialist ที่มีประสบการณ์ 15-20 ปี ช่วยเขียน SEO-optimized image file name with image alt tags เป็นภาษาอังกฤษ สำหรับสินค้าของฉันตามแต่ละรูปที่แนบมาให้ {product_url} เพื่อให้ได้ติดอันดับที่ดีบน organic search engine โดยกลุ่มลูกค้าเป็นผู้สนใจสินค้าชนิดนี้
IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PRODUCT_WRITER_PROMPT = """
# Product Description Prompt v2.1
## Optimized for Google Search Organic — Updated for 2026 Algorithm

---

## ROLE:

You are a Senior E-commerce Copywriter with 20 years of hands-on experience.
Your style is inspired by real product reviewers who have physically touched,
tested, and lived with the item. You hate generic marketing fluff.
You "show" instead of "tell." You write the way a knowledgeable friend
gives a recommendation — honest, specific, and a little opinionated.

---

## CRITICAL RULES — READ BEFORE WRITING ANYTHING:

### [RULE 1 — SENTENCE RHYTHM]

Vary length aggressively. Short punchy sentences. Then a longer one that
builds context and earns the reader's trust. Then maybe just four words.
Never write three sentences of the same approximate length in a row.
This is non-negotiable.

### [RULE 2 — BAN LIST]

NEVER use these words or phrases:

> Delve, Elevate, Comprehensive, Cutting-edge, Unleash, Ultimate, Testament,
> Precision-engineered, Game-changer, Furthermore, Moreover, In conclusion,
> Seamlessly, Robust, Leverage, In today's world, Look no further,
> It's worth noting, Revolutionize, State-of-the-art, Best-in-class,
> Unparalleled, Groundbreaking, Next-level, Innovative (unless quoting
> a specific patent or feature name).

Natural alternatives to use instead:

- "Delve" → Look into / dig into
- "Elevate" → Improve / step up
- "Comprehensive" → Complete / thorough
- "Game-changer" → Big shift / real difference
- "Furthermore" → And / On top of that / Also
- "Seamlessly" → Smoothly / without fuss
- "Robust" → Solid / sturdy / tough
- "Leverage" → Use / take advantage of

### [RULE 3 — SENSORY & FIRST-HAND EXPERIENCE (E-E-A-T)]

Write as if you have physically used this product. Include at least 2
specific sensory details. Think:

- **Texture:** "The matte finish feels slightly grippy — not that slippery
  plastic you usually get with cheaper options."
- **Sound:** "The box opened with that satisfying whoosh of air."
- **Weight:** "A bit heavier than I expected, but that actually makes
  it feel like it'll last."
- **Setup:** "First-time setup took me about 8 minutes — not zero effort,
  but nothing to stress about."
- **Smell:** "No chemical smell out of the box — a good sign for the materials."
- **Temperature:** "The metal body stays cool even after an hour of heavy use."

> **Why this matters (2026 context):** Google's ranking systems now
> heavily reward content that demonstrates first-hand experience.
> A product review from someone who has actually used the product
> carries significantly more weight than a summary of manufacturer specs.
> This is the core of Google's E-E-A-T framework — Experience, Expertise,
> Authoritativeness, Trustworthiness.

### [RULE 4 — HONEST OBSERVATION]

Include exactly ONE small, honest imperfection or caveat.
This builds credibility. Real reviewers don't only praise things.

Example: "The cord is a bit short — you'll want to position this near
an outlet. Minor complaint for what you get."

> **Why this matters:** Google's December 2025 Core Update continued
> rewarding high-quality, original content while reducing visibility
> for overly optimized pages. One-sided praise reads as marketing copy.
> A balanced view signals genuine experience.

### [RULE 5 — HUMAN NUANCES]

Use contractions naturally: don't, it's, you'll, there's, won't,
that's. Occasionally start a sentence with "And" or "But."
Use em dashes — like this — to break your own thought mid-sentence.

### [RULE 6 — NEVER OPEN WITH THE PRODUCT NAME]

Start with a problem, a scene, or a blunt observation that pulls
the reader in immediately.

### [RULE 7 — PRODUCT SCHEMA-FRIENDLY CONTENT]

Write so the content maps naturally to Google's Product structured data
(JSON-LD). This is critical for earning rich snippets (stars, price,
availability badges) in search results.

Ensure the description naturally includes:

- **Product name** — must appear clearly within the first
  2 paragraphs. Don't bury it in metaphor.
- **Category context** — mention what type of product this is in
  plain language (e.g., "wireless noise-cancelling headphones")
  so Google can classify it correctly.
- **Value positioning** — reference the price tier naturally
  (e.g., "for under ฿2,000" or "in the mid-range bracket")
  without stating the exact price (which changes and should
  come from your CMS/schema, not hardcoded in copy).
- **Availability/shipping context** — if relevant, mention it
  naturally (e.g., "ships in the original box" or "available
  in three colorways").
- **The Specs table** must contain real, measurable data that maps
  directly to Product schema properties: weight, dimensions,
  material, battery life, connectivity, etc.

> **Do NOT write FAQ-schema-style Q&As.** Your Q&A section lives on
> the product page itself as buyer-objection handling — it uses
> Product schema, not FAQPage schema.

### [RULE 8 — TRANSACTIONAL SEARCH POSITIONING]

The people landing on this product page are searching with buying intent.
They're typing things like "best [category] for [use case]" or
"[product] vs [competitor category]."

Include ONE natural sentence that positions this product against its
category — not a specific competitor brand name.

Examples:
- "Most budget wireless earbuds sacrifice bass. This one doesn't."
- "Where other compact blenders struggle with ice, this handles
  frozen fruit without stalling."

This helps Google match the page to comparative/transactional queries
without keyword stuffing.

### [RULE 9 — KEYWORD ANALYSIS & NATURAL INTEGRATION]

**Step 1 — Analyze before writing:**

Before writing any content, analyze the provided product description
and extract/determine the following keywords:

- **Main Keyword:**
  **First, check if the provided product description contains
  "main keyword :" (or "Main Keyword :").** If found, use that
  exact keyword as the Main Keyword — do not override it.
  If NOT found, identify the most searchable combination of
  [Product Name] + [Product Category] from the description.
  Example: "skull flame stainless steel ring"

- **Secondary Keywords (2-3):** Identify feature-based or
  attribute-based keyword variations from the product specs.
  Example: "biker skull ring", "stainless steel gothic ring"

- **Long-tail Keywords (2-3):** Infer what real buyers would
  search based on the product's use cases and target audience.
  Think in terms of [product type] + [for whom / for what purpose].
  Example: "men's skull ring for bikers",
  "heavy stainless steel ring that won't tarnish"

- **Comparison Phrase (1):** Create a natural category comparison
  that positions this product without naming competitors.
  Example: "unlike most mass-produced biker rings"

Keep this analysis in memory for Step 2. Then output it at the
very end of your content as an HTML comment block (see OUTPUT
STRUCTURE — Keyword Analysis Note at bottom).

**Step 2 — Integrate naturally:**

- **Main Keyword** must appear in: Hook section (first 2 paragraphs),
  at least 1 H2 heading, and the Meta Title.
  Total: 2-3 natural mentions across the body.
- **Secondary Keywords** should appear 1 time each, spread across
  different sections.
- **Long-tail Keywords** should appear in the "Who This Is Actually For"
  section or the Q&A section — embedded in natural sentences.
- **Comparison Phrase** should appear once in the "What It's Like to Use"
  section.

All keywords must feel invisible — embedded in genuine observations,
not bolted on. If a keyword feels forced when you read it aloud,
rewrite the sentence. Do NOT keyword-stuff.

### [RULE 10 — FRESHNESS WITHOUT EXPIRY]

Include ONE subtle time-anchor that signals this content is current
without dating it too fast:

- ✅ "The 2026 version finally adds..."
- ✅ "Since the latest firmware update..."
- ✅ "The newest colorway just dropped and..."
- ❌ "As of February 2026..." (expires too quickly)
- ❌ "This month's best pick..." (stale in weeks)
- ❌ No time reference at all (Google can't assess freshness)

---

## OUTPUT STRUCTURE:

### [Hook — no H2 tag]

2-3 sentences. Open with a pain point or a scene.
Make the reader feel recognized before you sell anything.

> The product name + category (Main Keyword) must appear naturally
> within this section (Rule 7 + Rule 9).

---

### ## Who This Is Actually For

Address 3 distinct user types using "If you..." framing.
Write each as a short paragraph (2-3 sentences). Don't list features —
describe how this fixes their specific Tuesday afternoon struggle.

> Each "If you..." paragraph should contain a natural long-tail keyword
> variation (Rule 9).

---

### ## What It's Like to Use (The Honest Take)

Your E-E-A-T section. Describe the physical experience of using
this product:

- Include your **2 sensory details** here (Rule 3).
- Include your **1 honest observation** here (Rule 4).
- Include your **1 comparison phrase** here (Rule 8 + Rule 9).

Write 3-5 short paragraphs. Vary rhythm aggressively (Rule 1).

---

### ## The Specs — And What They Actually Mean

List 4-6 key specs. Format each spec using this EXACT HTML pattern:

<p><span style="color:#1a3a6b; font-weight:700; font-size:1.05em;">[Spec Name]:</span> Plain-English benefit in one conversational sentence. Wrap the <strong style="color:#1a3a6b;">most important keyword or value</strong> in the benefit sentence with a bold dark-blue span to help readers scan key takeaways quickly.</p>

Example output:
<p><span style="color:#1a3a6b; font-weight:700; font-size:1.05em;">Material:</span> Solid <strong style="color:#1a3a6b;">316L stainless steel</strong> — it won't tarnish, scratch easily, or turn your finger green.</p>
<p><span style="color:#1a3a6b; font-weight:700; font-size:1.05em;">Weight:</span> Comes in at <strong style="color:#1a3a6b;">28 grams</strong> — heavy enough to feel premium, not heavy enough to bother you all day.</p>

IMPORTANT: Every spec line MUST use the dark-blue color (#1a3a6b) for the spec name AND bold-highlight at least one key value or keyword inside the benefit sentence with <strong style="color:#1a3a6b;">. This makes specs scannable at a glance.

> These specs should contain measurable data that maps to Product
> schema properties (Rule 7): weight, dimensions, material,
> battery capacity, connectivity standard, etc.
> Include secondary keywords naturally where relevant (Rule 9).

---

### ## Questions You're Probably Asking

Write 3-4 Q&As from a skeptical buyer's perspective.

**Question format:** Blunt, real-language questions a buyer would
actually think (not SEO-bait questions).

- "Will this actually fit in my bag?"
- "Is the noise cancelling good enough for an open office?"
- "Do I need to buy anything else to get started?"

**Answer format:** Open with a direct yes/no or factual statement,
THEN elaborate in 2-3 sentences max.

> These are buyer-objection handlers, not FAQ schema entries.
> They should reduce purchase hesitation on the page itself.
> Include long-tail keywords naturally in questions or answers (Rule 9).

---

### ## Quick Specs & Real-World Performance

Generate this section as a styled HTML table using this EXACT format:

<div style="overflow-x:auto; margin:1.5em 0;">
<table style="width:100%; border-collapse:separate; border-spacing:0; border-radius:10px; overflow:hidden; box-shadow:0 2px 12px rgba(26,58,107,0.10); font-size:0.98em;">
<thead>
<tr style="background:linear-gradient(135deg,#1a3a6b 0%,#2d5aa0 100%); color:#ffffff;">
<th style="padding:14px 18px; text-align:left; font-weight:700; letter-spacing:0.3px;">Technical Detail</th>
<th style="padding:14px 18px; text-align:left; font-weight:700; letter-spacing:0.3px;">What It Actually Does for You</th>
</tr>
</thead>
<tbody>
<tr style="background:#f8fafd;">
<td style="padding:12px 18px; border-bottom:1px solid #e4eaf2; font-weight:600; color:#1a3a6b;">[Spec]</td>
<td style="padding:12px 18px; border-bottom:1px solid #e4eaf2; color:#333;">[Benefit]</td>
</tr>
<tr style="background:#ffffff;">
<td style="padding:12px 18px; border-bottom:1px solid #e4eaf2; font-weight:600; color:#1a3a6b;">[Spec]</td>
<td style="padding:12px 18px; border-bottom:1px solid #e4eaf2; color:#333;">[Benefit]</td>
</tr>
<!-- alternate #f8fafd and #ffffff for each row -->
</tbody>
</table>
</div>

IMPORTANT:
- The table header MUST use the dark-blue gradient background (#1a3a6b to #2d5aa0) with white text.
- Alternate row backgrounds between #f8fafd (light blue-gray) and #ffffff (white) for readability.
- Column 1 (Technical Detail) must be bold dark-blue (#1a3a6b).
- Column 2 (Benefit) must be plain-English, no jargon.
- Include 5-8 rows of real, measurable specs.
- The table must have rounded corners and a subtle shadow as shown above.

> This table is your Product schema goldmine. Every row should contain
> a real, measurable spec in column 1 and a plain-English benefit
> in column 2.

---

### ## You Might Also Want

Write 2-3 natural "bridge" sentences pointing to complementary
or alternative products. Frame as genuine advice, not a sales push.

**CRITICAL — INTERNAL LINKING FROM REAL STORE DATA:**

You will receive REAL STORE CATALOG DATA at the end of this prompt
containing actual collections and products that exist on the store.
You MUST ONLY link to paths that appear in that catalog data.
NEVER invent or guess URLs — every href must come from the provided list.

**How to write this section:**
1. First, write the recommendation sentences naturally — as if
   giving honest advice to a friend. Do NOT write around keywords.
2. Then, identify which real collection or product from the catalog
   data best matches your recommendation.
3. Finally, wrap the most natural phrase in the sentence with a link
   to that real path. The linked phrase should flow seamlessly in the
   sentence — it should NOT look like a keyword was inserted.

**Link format:**
<a href="[exact path from catalog]" title="[Product or Collection Title from catalog]" style="color:#1a3a6b; font-weight:600; text-decoration:underline;">[natural phrase from your sentence]</a>

**GOOD — natural sentence first, link added to a phrase that fits:**
<p>If you like the tribal look but want something for your wrist too, we've got <a href="/collections/bracelets" title="Bracelets" style="color:#1a3a6b; font-weight:600; text-decoration:underline;">a whole section of bracelets</a> that pair well with heavy chains.</p>
<p>A lot of people who grab this end up coming back for <a href="/products/skull-cross-sterling-silver-wallet-chain" title="Skull Cross Sterling Silver Wallet Chain" style="color:#1a3a6b; font-weight:600; text-decoration:underline;">the skull cross version</a> — different vibe, same solid build.</p>

**BAD — keyword-stuffed, unnatural:**
<p>Check out our <a href="/collections/skull-rings">skull rings collection</a> for more skull rings.</p>

**RULES:**
1. ONLY use paths from the provided REAL STORE CATALOG DATA.
   If no catalog data is provided, skip the links and write plain text recommendations.
2. Use PATH URLs only — start with / (e.g., /collections/... or /products/...).
3. Link 2-3 items total — mix of collections and products when possible.
4. The linked text must be a natural part of the sentence, NOT a standalone keyword.
5. The title attribute should match the real product/collection title from the catalog.

> This section creates internal links to related product/collection pages,
> which strengthens your site's crawlability and topical authority.
> Every link MUST point to a real, existing page from the store catalog.

---

### ## META (for CMS use — do not publish on page)

**Meta Title:** `[Product Name] — [One Key Benefit] | Bikerringshop`
Keep under 60 characters. Lead with product name, not benefit.
Must include the Main Keyword (Rule 9).

**Meta Description:** Under 155 characters. Include the Main Keyword
+ one sensory detail or the honest caveat to stand out in SERPs.
No "shop now" or "buy today" CTAs — those waste characters and
don't improve CTR.

Example:
> Meta Title: Skull Flame Ring — Heavy Stainless Steel That Lasts | Bikerringshop
> Meta Description: Slightly heavier than the XM4, but the ANC blocks out open-office chatter completely. Setup takes 3 minutes flat.

---

### ## Keyword Analysis Note (HTML comment — hidden from visitors)

At the very end of the content, output the keyword analysis as an
HTML comment. This will be invisible to visitors but useful for
the SEO/content team to review.

Format:
```html
<!--
🔍 Keyword Analysis:
- Main: [keyword]
- Secondary: [keyword 1], [keyword 2]
- Long-tail: [keyword 1], [keyword 2]
- Comparison: [phrase]
- Placement check:
  - Main in Hook: ✅/❌
  - Main in H2: ✅/❌
  - Main in Meta Title: ✅/❌
  - Secondary spread across sections: ✅/❌
  - Long-tail in "Who This Is For" or Q&A: ✅/❌
  - Comparison in "Honest Take": ✅/❌
-->
```

> This HTML comment is invisible on the published page but remains
> in the source code for internal review. The placement checklist
> helps the team verify keyword integration without re-reading
> the entire description.

---

## TONE:

Helpful, slightly opinionated, expert but human.
No corporate speak. No hype. Sound like you'd actually buy this.

## TARGET LENGTH:

480-600 words (body content, excluding table and meta section).

## READING LEVEL:

Grade 8-10. Clear, not dumbed down.

---

## CONTEXT FOR THE AI (do not output this section):

**Google Algorithm Context (2025-2026):**

- Google's December 2025 Core Update (4th core update of 2025) heavily
  rewarded e-commerce and retail brands while penalizing thin content.
- E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)
  is the primary quality framework. "Experience" — first-hand product
  use — is the differentiator for product pages.
- Google penalizes low-value AI-generated content more aggressively.
  Content must show human insight, specific details, and balanced views.
- Product structured data (JSON-LD) remains critical for rich snippets.
  Core schema types — Product, Review, Breadcrumb, Organization —
  are confirmed as long-term priorities by Google.
- User engagement metrics (time on page, scroll depth, bounce rate)
  are stronger ranking signals than ever.
- Content freshness signals matter — but product pages should use
  subtle time-anchors, not hard dates that expire.
- Zero-click searches (~60% of Google queries) mean product pages
  must provide enough structured data (via schema) to appear in
  rich results even when users don't click through.
- The February 2026 Discover Core Update prioritizes original,
  in-depth content from sites with demonstrated expertise.
  Topic-by-topic expertise matters — a site doesn't need to be
  an authority on everything, just on what it covers.

**What this prompt is NOT for:**
- Blog posts or informational content (use Blog Post Prompt v2.1)
- Category/collection pages (different optimization approach)
- Landing pages for ads (different conversion strategy)
---

Input Data: {raw_input}

IMPORTANT OUTPUT FORMAT:
You MUST return the result in RAW JSON format ONLY. Do not include markdown backticks.
{
  "url_slug": "url-slug-example",
  "meta_title": "Meta Title Example (Max 60 chars)",
  "meta_description": "Meta Description Example (Max 160 chars)",
  "product_title_h1": "Product Title Example",
  "html_content": "<p>Your full HTML product description here...</p>",
  "image_seo": [
    { "file_name": "image-name.jpg", "alt_tag": "Image description" }
  ]
}
"""

SEO_PROMPT_NAME_SLUG = """
You are an SEO expert with 10-15 years of experience. 
Analyze the provided product images and description. Generate:
1. An attractive, SEO-optimized Product Name.
2. A suitable, clean URL Slug (using hyphens).

User Input Description: "{user_desc}"

IMPORTANT: Return RAW JSON format ONLY (no markdown backticks).
Structure: {"product_name": "...", "url_slug": "..."}
"""

# Default Data
DEFAULT_PROMPTS = [
    {"id": "p1", "name": "Luxury Hand (Ring)", "category": "Ring",
     "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
     "variables": "face_size", "sample_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg"},
    {"id": "rt1", "name": "Clean Studio Look", "category": "Retouch",
     "template": "Retouch this jewelry product to have a clean white studio background. Enhance the metal shine of {metal_type} and gemstone clarity. Professional product photography.",
     "variables": "metal_type", "sample_url": ""}
]

# --- 2. CLOUD DATABASE FUNCTIONS ---
def get_prompts():
    try:
        raw_key = st.secrets.get("JSONBIN_API_KEY", "")
        raw_bin = st.secrets.get("JSONBIN_BIN_ID", "")
        API_KEY = clean_key(raw_key)
        BIN_ID = clean_key(raw_bin)
        if not API_KEY or not BIN_ID: return DEFAULT_PROMPTS
        url = f"https://api.jsonbin.io/v3/b/{BIN_ID}/latest"
        headers = {"X-Master-Key": API_KEY}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json().get("record", DEFAULT_PROMPTS)
        return DEFAULT_PROMPTS
    except: return DEFAULT_PROMPTS

def save_prompts(data):
    try:
        raw_key = st.secrets.get("JSONBIN_API_KEY", "")
        raw_bin = st.secrets.get("JSONBIN_BIN_ID", "")
        API_KEY = clean_key(raw_key)
        BIN_ID = clean_key(raw_bin)
        url = f"https://api.jsonbin.io/v3/b/{BIN_ID}"
        headers = {"Content-Type": "application/json", "X-Master-Key": API_KEY}
        requests.put(url, json=data, headers=headers, timeout=10)
    except Exception as e: st.error(f"Save failed: {e}")

# --- 3. HELPER FUNCTIONS ---
def img_to_base64(img):
    buf = BytesIO()
    if img.mode == 'RGBA': img = img.convert('RGB')
    img.thumbnail((1024, 1024)) 
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def parse_json_response(text):
    if not text: return None
    # Step 1: Try direct parse (cleanest case)
    try:
        return json.loads(text.strip())
    except: pass
    
    # Step 2: Remove markdown code fences
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except: pass
    
    # Step 3: Extract JSON object/array from surrounding text
    # Find the first { or [ and match to its closing } or ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1: continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False; continue
            if c == '\\' and in_string:
                escape_next = True; continue
            if c == '"' and not escape_next:
                in_string = not in_string; continue
            if in_string: continue
            if c == start_char: depth += 1
            elif c == end_char: depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start_idx:i+1])
                except: break
    
    return None

def clean_filename(name):
    if not name: return "N/A"
    clean = re.sub(r'[^a-zA-Z0-9\-\_\.]', '', str(name))
    return clean.rsplit('.', 1)[0]

def remove_html_tags(text):
    if not text: return ""
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
    return "\n".join([line.strip() for line in text.split('\n') if line.strip()])

# --- SHOPIFY HELPER FUNCTIONS ---
def update_shopify_product_v2(shop_url, access_token, product_id, data, images_pil=None, upload_images=False):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    product_payload = {
        "id": product_id,
        "title": data.get('product_title_h1'),
        "body_html": data.get('html_content'),
        "metafields": [
            {"namespace": "global", "key": "title_tag", "value": data.get('meta_title', ''), "type": "single_line_text_field"},
            {"namespace": "global", "key": "description_tag", "value": data.get('meta_description', ''), "type": "multi_line_text_field"}
        ]
    }
    
    if upload_images and images_pil and "image_seo" in data:
        img_payloads = []
        image_seo_list = data.get("image_seo", [])
        for i, img in enumerate(images_pil):
            seo_info = image_seo_list[i] if i < len(image_seo_list) else {}
            img_payloads.append({
                "attachment": img_to_base64(img),
                "filename": seo_info.get("file_name", f"image_{i+1}.jpg"),
                "alt": seo_info.get("alt_tag", "")
            })
        if img_payloads: product_payload["images"] = img_payloads

    try:
        response = requests.put(url, json={"product": product_payload}, headers=headers)
        if response.status_code in [200, 201]: return True, "✅ Update Successful!"
        return False, f"Shopify API Error {response.status_code}: {response.text}"
    except Exception as e: return False, f"Connection Error: {str(e)}"

def add_single_image_to_shopify(shop_url, access_token, product_id, image_bytes, file_name=None, alt_tag=None):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}/images.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    if not image_bytes: return False, "No valid image data."
    b64_str = base64.b64encode(image_bytes).decode('utf-8')
    payload = {"image": {"attachment": b64_str, "filename": file_name or f"gen_ai_image_{int(time.time())}.jpg", "alt": alt_tag or "AI Generated Product Image"}}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code in [200, 201]: return True, "✅ Added Successful!"
        return False, f"Shopify Error {response.status_code}: {response.text}"
    except Exception as e: return False, f"Connection Error: {str(e)}"

def upload_only_images_to_shopify(shop_url, access_token, product_id, image_bytes_list):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    img_payloads = []
    for i, img_bytes in enumerate(image_bytes_list):
        if img_bytes:
            img_payloads.append({"attachment": base64.b64encode(img_bytes).decode('utf-8'), "filename": f"retouched_image_{i+1}.jpg", "alt": f"Retouched Product Image {i+1}"})
    if not img_payloads: return False, "No valid images to upload."
    
    try:
        response = requests.put(url, json={"product": {"id": product_id, "images": img_payloads}}, headers=headers)
        if response.status_code in [200, 201]: return True, "✅ Upload Successful!"
        return False, f"Shopify Error {response.status_code}: {response.text}"
    except Exception as e: return False, f"Connection Error: {str(e)}"

def get_shopify_product_images(shop_url, access_token, product_id):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}/images.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            pil_images = []
            for img_info in response.json().get("images", []):
                src = img_info.get("src")
                if src:
                    img_resp = requests.get(src, stream=True)
                    if img_resp.status_code == 200:
                        img_pil = Image.open(BytesIO(img_resp.content))
                        if img_pil.mode in ('RGBA', 'P'): img_pil = img_pil.convert('RGB')
                        pil_images.append(img_pil)
            return pil_images, None
        return None, f"Shopify API Error {response.status_code}: {response.text}"
    except Exception as e: return None, f"Connection Error: {str(e)}"

def get_shopify_product_details(shop_url, access_token, product_id):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            prod = response.json().get("product", {})
            return prod.get("body_html", ""), prod.get("title", ""), prod.get("handle", ""), None
        return None, None, None, f"Error {response.status_code}: {response.text}"
    except Exception as e: return None, None, None, str(e)

# --- SHOPIFY ADMIN: LIST PRODUCTS & COLLECTIONS ---
def _shopify_admin_get(shop_url, access_token, endpoint, timeout=30, retries=3):
    """Robust GET for Shopify Admin API with retry and timeout."""
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    url = f"https://{shop_url}/admin/api/2024-01/{endpoint}"
    
    for attempt in range(retries):
        try:
            res = requests.get(url, headers=headers, timeout=timeout)
            if res.status_code == 200:
                return res, None
            elif res.status_code == 429:  # Rate limited
                time.sleep(2 * (attempt + 1)); continue
            else:
                return None, f"Error {res.status_code}: {res.text[:200]}"
        except requests.exceptions.Timeout:
            if attempt < retries - 1: time.sleep(2); continue
            return None, f"Timeout after {timeout}s (attempt {attempt+1})"
        except Exception as e:
            if attempt < retries - 1: time.sleep(1); continue
            return None, str(e)
    return None, "Failed after retries"

def get_shopify_all_collections(shop_url, access_token):
    """Fetch all custom + smart collections from Shopify admin."""
    all_collections = []
    for ctype in ["custom_collections", "smart_collections"]:
        res, err = _shopify_admin_get(shop_url, access_token, f"{ctype}.json?limit=250")
        if res:
            for c in res.json().get(ctype, []):
                all_collections.append({"id": c["id"], "title": c.get("title", ""), "handle": c.get("handle", "")})
    return all_collections

def get_shopify_products_page(shop_url, access_token, limit=50, page_info=None, collection_id=None):
    """Fetch one page of products from Shopify admin (smaller batches for reliability)."""
    if collection_id:
        endpoint = f"collections/{collection_id}/products.json?limit={limit}"
    elif page_info:
        endpoint = f"products.json?limit={limit}&page_info={page_info}"
    else:
        endpoint = f"products.json?limit={limit}"
    
    res, err = _shopify_admin_get(shop_url, access_token, endpoint, timeout=45)
    if err:
        return [], None, err
    
    products_raw = res.json().get("products", [])
    products = []
    for p in products_raw:
        total_inv = sum(v.get("inventory_quantity", 0) for v in p.get("variants", []))
        sku_list = [v.get("sku", "") for v in p.get("variants", []) if v.get("sku")]
        products.append({
            "id": str(p["id"]),
            "title": p.get("title", ""),
            "handle": p.get("handle", ""),
            "product_type": p.get("product_type", ""),
            "status": p.get("status", ""),
            "total_inventory": total_inv,
            "sku": sku_list[0] if sku_list else "",
            "all_skus": ", ".join(sku_list[:3]),
            "variants_count": len(p.get("variants", [])),
            "image_url": p.get("image", {}).get("src", "") if p.get("image") else "",
            "body_html": p.get("body_html", ""),
        })
    
    # Parse next page cursor from Link header
    next_cursor = None
    link_header = res.headers.get("Link", "")
    if 'rel="next"' in link_header:
        import urllib.parse
        for part in link_header.split(","):
            if 'rel="next"' in part:
                url_part = part.split(";")[0].strip().strip("<>")
                parsed = urllib.parse.urlparse(url_part)
                params = urllib.parse.parse_qs(parsed.query)
                next_cursor = params.get("page_info", [None])[0]
    
    return products, next_cursor, None

def get_shopify_all_products(shop_url, access_token, collection_id=None, max_pages=25, progress_callback=None):
    """Fetch all products (paginated with smaller batches) from Shopify admin."""
    all_products = []
    cursor = None
    for page_num in range(max_pages):
        products, next_cursor, err = get_shopify_products_page(
            shop_url, access_token, limit=50, page_info=cursor, collection_id=collection_id
        )
        if err and not products:
            if all_products:  # Return what we have so far
                return all_products, f"Partial load ({len(all_products)} products). Stopped: {err}"
            return all_products, err
        all_products.extend(products)
        if progress_callback:
            progress_callback(len(all_products), page_num + 1)
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(0.3)  # Small delay to avoid rate limits
    return all_products, None

def update_shopify_description_only(shop_url, access_token, product_id, data):
    """Update only title, body_html, and meta fields — no images."""
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    product_payload = {
        "id": product_id,
        "body_html": data.get('html_content'),
        "metafields": [
            {"namespace": "global", "key": "title_tag", "value": data.get('meta_title', ''), "type": "single_line_text_field"},
            {"namespace": "global", "key": "description_tag", "value": data.get('meta_description', ''), "type": "multi_line_text_field"}
        ]
    }
    
    try:
        response = requests.put(url, json={"product": product_payload}, headers=headers)
        if response.status_code in [200, 201]: return True, "✅ Updated"
        return False, f"Error {response.status_code}: {response.text[:200]}"
    except Exception as e: return False, str(e)

# ============================================================
# --- STORE CATALOG FETCHER (for internal linking) ---
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_store_catalog(store_domain="www.bikerringshop.com"):
    """Fetch real collections and products from the public Shopify storefront for internal linking."""
    catalog = {"collections": [], "products": []}
    
    # Fetch collections
    try:
        res = requests.get(f"https://{store_domain}/collections.json?limit=250", timeout=15)
        if res.status_code == 200:
            for c in res.json().get("collections", []):
                catalog["collections"].append({
                    "title": c.get("title", ""),
                    "handle": c.get("handle", ""),
                    "path": f"/collections/{c.get('handle', '')}"
                })
    except: pass
    
    # Fetch products (multiple pages for larger stores)
    page = 1
    while page <= 5:  # Max 5 pages = 1250 products
        try:
            res = requests.get(f"https://{store_domain}/products.json?limit=250&page={page}", timeout=15)
            if res.status_code == 200:
                products = res.json().get("products", [])
                if not products: break
                for p in products:
                    catalog["products"].append({
                        "title": p.get("title", ""),
                        "handle": p.get("handle", ""),
                        "path": f"/products/{p.get('handle', '')}",
                        "type": p.get("product_type", ""),
                        "tags": ", ".join(p.get("tags", [])[:5]) if p.get("tags") else ""
                    })
                page += 1
            else: break
        except: break
    
    return catalog

def format_catalog_for_prompt(catalog, max_collections=30, max_products=80):
    """Format catalog data into a compact string for the AI prompt."""
    lines = []
    if catalog.get("collections"):
        lines.append("=== REAL COLLECTIONS (use these paths) ===")
        for c in catalog["collections"][:max_collections]:
            lines.append(f"- {c['path']}  →  \"{c['title']}\"")
    
    if catalog.get("products"):
        lines.append("\n=== REAL PRODUCTS (use these paths) ===")
        for p in catalog["products"][:max_products]:
            extra = f"  [{p['type']}]" if p.get('type') else ""
            lines.append(f"- {p['path']}  →  \"{p['title']}\"{extra}")
    
    return "\n".join(lines)

# ============================================================
# --- CLAUDE API FUNCTION ---
# ============================================================
def call_claude_api(claude_key, prompt, img_pil_list=None, model_id="claude-sonnet-4-20250514"):
    """Call Claude API for Text/SEO tasks with optional image support"""
    url = "https://api.anthropic.com/v1/messages"
    headers = {"Content-Type": "application/json", "x-api-key": claude_key, "anthropic-version": "2023-06-01"}
    
    content = []
    if img_pil_list:
        for img in img_pil_list:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_to_base64(img)}})
    content.append({"type": "text", "text": prompt})
    
    payload = {"model": model_id, "max_tokens": 4096, "messages": [{"role": "user", "content": content}]}
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=120)
            if res.status_code == 200:
                text_content = ""
                for block in res.json().get("content", []):
                    if block.get("type") == "text": text_content += block.get("text", "")
                return text_content, None
            elif res.status_code == 529: time.sleep(3); continue
            else: return None, f"Claude API Error {res.status_code}: {res.text}"
        except: time.sleep(2)
    return None, "Claude API failed after retries"

# ============================================================
# --- OPENAI API FUNCTION (NEW) ---
# ============================================================
def call_openai_api(openai_key, prompt, img_pil_list=None, model_id="gpt-5.2"):
    """Call OpenAI API for Text/SEO tasks with optional image support"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"}
    
    content = []
    if img_pil_list:
        for img in img_pil_list:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_to_base64(img)}"}})
    content.append({"type": "text", "text": prompt})
    
    payload = {
        "model": model_id, 
        "max_completion_tokens": 4096,  # GPT-5.2 uses max_completion_tokens instead of max_tokens
        "messages": [{"role": "user", "content": content}]
    }
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=120)
            if res.status_code == 200:
                return res.json().get("choices", [{}])[0].get("message", {}).get("content", ""), None
            elif res.status_code == 429: time.sleep(3); continue
            else: return None, f"OpenAI API Error {res.status_code}: {res.text}"
        except: time.sleep(2)
    return None, "OpenAI API failed after retries"

# ============================================================
# --- AI FUNCTIONS (GEMINI & CLAUDE) ---
# ============================================================
def generate_image(api_key, image_list, prompt):
    """Image Generation - Gemini Only"""
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_IMAGE_GEN}:generateContent?key={key}"
    full_prompt = f"Instruction: {prompt} \nImportant Constraint: Keep the main jewelry product in the input image EXACTLY as it looks. Only improve lighting, background, and photography quality."
    
    parts = [{"text": full_prompt}]
    for img in image_list: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    
    try:
        res = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.3}}, headers={"Content-Type": "application/json"})
        if res.status_code != 200: return None, f"API Error {res.status_code}: {res.text}"
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        if "inline_data" in content: return base64.b64decode(content["inline_data"]["data"]), None
        if "inlineData" in content: return base64.b64decode(content["inlineData"]["data"]), None
        if "text" in content: return None, f"Model returned text: {content['text']}"
        return None, "Unknown format"
    except Exception as e: return None, str(e)

def _call_gemini_text(gemini_key, payload, timeout=60):
    """Helper: Call Gemini text model with automatic fallback from 3.1 to 3.0"""
    key = clean_key(gemini_key)
    models_to_try = [MODEL_TEXT_GEMINI, MODEL_TEXT_GEMINI_FALLBACK]
    last_error = "Failed"
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={key}"
        for attempt in range(3):
            try:
                res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
                if res.status_code == 200:
                    return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
                elif res.status_code in (503, 429):
                    time.sleep(2); continue
                else:
                    last_error = f"Error {res.status_code} ({model}): {res.text}"
                    break  # Try next model
            except Exception as e:
                last_error = f"Exception ({model}): {str(e)}"
                time.sleep(1)
        else:
            continue  # All 3 attempts were 503/429 retries, try next model
        continue  # Broke out of retry loop due to non-retryable error, try next model
    return None, last_error

def generate_seo_tags_smart(gemini_key, claude_key, openai_key, selected_model, context, product_url=""):
    prompt = SEO_PROMPT_SMART_GEN.replace("{context}", context).replace("{product_url}", product_url)
    
    # Claude models
    if selected_model in CLAUDE_MODELS and claude_key:
        model_id = CLAUDE_MODELS[selected_model]
        return call_claude_api(claude_key, prompt, model_id=model_id)
    
    # OpenAI models
    if selected_model in OPENAI_MODELS and openai_key:
        model_id = OPENAI_MODELS[selected_model]
        return call_openai_api(openai_key, prompt, model_id=model_id)
    
    # Default: Gemini (with fallback)
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}}
    return _call_gemini_text(gemini_key, payload)

def generate_seo_from_generated_image(gemini_key, claude_key, openai_key, selected_model, generated_image_bytes, product_url=""):
    """Analyze the generated image and create SEO tags based on visual content + product URL"""
    prompt = SEO_PROMPT_IMAGE_ANALYSIS.replace("{product_url}", product_url if product_url else "No URL provided")
    
    # Convert bytes to PIL Image
    try:
        img_pil = Image.open(BytesIO(generated_image_bytes))
    except:
        return None, "Failed to process generated image"
    
    # Claude models
    if selected_model in CLAUDE_MODELS and claude_key:
        model_id = CLAUDE_MODELS[selected_model]
        return call_claude_api(claude_key, prompt, [img_pil], model_id=model_id)
    
    # OpenAI models
    if selected_model in OPENAI_MODELS and openai_key:
        model_id = OPENAI_MODELS[selected_model]
        return call_openai_api(openai_key, prompt, [img_pil], model_id=model_id)
    
    # Default: Gemini (with fallback)
    parts = [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}]
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}}
    return _call_gemini_text(gemini_key, payload)

def generate_seo_for_existing_image(gemini_key, claude_key, openai_key, selected_model, img_pil, product_url):
    prompt = SEO_PROMPT_BULK_EXISTING.replace("{product_url}", product_url)
    
    # Claude models
    if selected_model in CLAUDE_MODELS and claude_key:
        model_id = CLAUDE_MODELS[selected_model]
        return call_claude_api(claude_key, prompt, [img_pil], model_id=model_id)
    
    # OpenAI models
    if selected_model in OPENAI_MODELS and openai_key:
        model_id = OPENAI_MODELS[selected_model]
        return call_openai_api(openai_key, prompt, [img_pil], model_id=model_id)
    
    # Default: Gemini (with fallback)
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}]}], "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}}
    return _call_gemini_text(gemini_key, payload)

def generate_full_product_content(gemini_key, claude_key, openai_key, selected_model, img_pil_list, raw_input, catalog_text=""):
    prompt = SEO_PRODUCT_WRITER_PROMPT.replace("{raw_input}", raw_input)
    num_images = len(img_pil_list) if img_pil_list else 0
    if num_images > 0: prompt += f"\n\nCRITICAL: You received {num_images} images. Return exactly {num_images} objects in 'image_seo' array."
    if catalog_text:
        prompt += f"\n\n--- REAL STORE CATALOG DATA (for 'You Might Also Want' section) ---\n{catalog_text}\n--- END CATALOG DATA ---"
    
    # Claude models
    if selected_model in CLAUDE_MODELS and claude_key:
        model_id = CLAUDE_MODELS[selected_model]
        return call_claude_api(claude_key, prompt, img_pil_list, model_id=model_id)
    
    # OpenAI models
    if selected_model in OPENAI_MODELS and openai_key:
        model_id = OPENAI_MODELS[selected_model]
        return call_openai_api(openai_key, prompt, img_pil_list, model_id=model_id)
    
    # Default: Gemini (with fallback)
    parts = [{"text": prompt}]
    if img_pil_list:
        for img in img_pil_list: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}}
    return _call_gemini_text(gemini_key, payload)

def generate_seo_name_slug(gemini_key, claude_key, openai_key, selected_model, img_list, user_desc):
    prompt = SEO_PROMPT_NAME_SLUG.replace("{user_desc}", user_desc)
    pil_images = []
    if img_list:
        for item in img_list:
            if isinstance(item, bytes):
                try: pil_images.append(Image.open(BytesIO(item)))
                except: pass
            elif isinstance(item, Image.Image): pil_images.append(item)
    
    # Claude models
    if selected_model in CLAUDE_MODELS and claude_key:
        model_id = CLAUDE_MODELS[selected_model]
        return call_claude_api(claude_key, prompt, pil_images if pil_images else None, model_id=model_id)
    
    # OpenAI models
    if selected_model in OPENAI_MODELS and openai_key:
        model_id = OPENAI_MODELS[selected_model]
        return call_openai_api(openai_key, prompt, pil_images if pil_images else None, model_id=model_id)
    
    # Default: Gemini (with fallback)
    parts = [{"text": prompt}]
    for img in pil_images: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}}
    return _call_gemini_text(gemini_key, payload)

def list_available_models(api_key):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200: return response.json().get("models", [])
        return None
    except: return None

# ============================================================
# --- UI LOGIC ---
# ============================================================
if "library" not in st.session_state: st.session_state.library = get_prompts()
if "edit_target" not in st.session_state: st.session_state.edit_target = None
if "image_generated_success" not in st.session_state: st.session_state.image_generated_success = False
if "current_generated_image" not in st.session_state: st.session_state.current_generated_image = None
if "gen_tags_result" not in st.session_state: st.session_state.gen_tags_result = {}
if "bulk_results" not in st.session_state: st.session_state.bulk_results = None
if "writer_result" not in st.session_state: st.session_state.writer_result = None
if "retouch_results" not in st.session_state: st.session_state.retouch_results = None
if "seo_name_result" not in st.session_state: st.session_state.seo_name_result = None
if "bulk_key_counter" not in st.session_state: st.session_state.bulk_key_counter = 0
if "writer_key_counter" not in st.session_state: st.session_state.writer_key_counter = 0
if "retouch_key_counter" not in st.session_state: st.session_state.retouch_key_counter = 0
if "batch_products" not in st.session_state: st.session_state.batch_products = []
if "batch_collections" not in st.session_state: st.session_state.batch_collections = []
if "batch_results" not in st.session_state: st.session_state.batch_results = {}
if "batch_running" not in st.session_state: st.session_state.batch_running = False

# ============================================================
# --- SIDEBAR (WITH MODEL SELECTOR) ---
# ============================================================
with st.sidebar:
    st.title("⚙️ Config")
    
    # MODEL SELECTOR
    st.subheader("🤖 AI Model Selection")
    
    # Provider selection
    all_models = ["Gemini"] + list(CLAUDE_MODELS.keys()) + list(OPENAI_MODELS.keys())
    selected_text_model = st.selectbox(
        "Text/SEO Model:", 
        all_models, 
        index=0, 
        help="เลือก Model สำหรับงาน SEO Writing", 
        key="sidebar_model_select"
    )
    st.session_state['selected_text_model'] = selected_text_model
    
    # Show model info
    if selected_text_model == "Gemini":
        st.caption("🔹 Google Gemini - Fast & Free tier available")
    elif selected_text_model in CLAUDE_MODELS:
        model_id = CLAUDE_MODELS[selected_text_model]
        st.caption(f"🔹 Anthropic - `{model_id}`")
        if "Opus" in selected_text_model:
            st.caption("⚠️ Opus = Higher quality but more expensive")
    elif selected_text_model in OPENAI_MODELS:
        model_id = OPENAI_MODELS[selected_text_model]
        st.caption(f"🔹 OpenAI - `{model_id}`")
    
    st.caption("📸 Image Gen: Gemini (Fixed)")
    st.divider()
    
    # API KEYS
    st.subheader("🔑 API Keys")
    
    # Gemini Key
    if "GEMINI_API_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini Key Loaded")
    elif "GOOGLE_API_KEY" in st.secrets:
        gemini_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ Google Key Loaded")
    else:
        gemini_key = st.text_input("Gemini API Key", type="password", key="sidebar_gemini_key")
    gemini_key = clean_key(gemini_key)
    
    # Claude Key
    if "CLAUDE_API_KEY" in st.secrets:
        claude_key = clean_key(st.secrets["CLAUDE_API_KEY"])
        st.success("✅ Claude Key Loaded")
    else:
        claude_key = st.text_input("Claude API Key", type="password", key="sidebar_claude_key")
        claude_key = clean_key(claude_key)
    
    # OpenAI Key
    if "OPENAI_API_KEY" in st.secrets:
        openai_key = clean_key(st.secrets["OPENAI_API_KEY"])
        st.success("✅ OpenAI Key Loaded")
    else:
        openai_key = st.text_input("OpenAI API Key", type="password", key="sidebar_openai_key")
        openai_key = clean_key(openai_key)
    
    # Validation warnings
    if selected_text_model in CLAUDE_MODELS and not claude_key:
        st.warning("⚠️ Claude selected but no API Key!")
    if selected_text_model in OPENAI_MODELS and not openai_key:
        st.warning("⚠️ OpenAI selected but no API Key!")
    
    st.divider()
    if "JSONBIN_API_KEY" in st.secrets: st.caption("✅ Database Connected")
    else: st.warning("⚠️ Local Mode")
    st.divider()
    st.caption(f"**Active Text Model:** {selected_text_model}")
    st.caption(f"**Active Image Model:** Gemini")

st.title("💎 Jewelry AI Studio")
tab1, tab_retouch, tab2, tab3, tab_batch, tab4, tab5 = st.tabs(["✨ Gen Image", "🎨 Retouch", "🏷️ Bulk SEO", "📝 Writer", "📝 Batch Writer", "📚 Library", "ℹ️ Models"])

# === TAB 1: GEN IMAGE ===
with tab1:
    if "gen_shopify_imgs" not in st.session_state: st.session_state.gen_shopify_imgs = []
    if "gen_key_counter" not in st.session_state: st.session_state.gen_key_counter = 0
    
    gen_key_id = st.session_state.gen_key_counter
    
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.subheader("1. Source Images")
        with st.expander("🛍️ Import from Shopify", expanded=True):
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            if sh_secret_shop and sh_secret_token:
                sh_gen_id = st.text_input("Product ID", key=f"gen_shopify_id_{gen_key_id}")
                col_fetch, col_clear = st.columns([2, 1])
                if col_fetch.button("⬇️ Fetch Images", key=f"gen_fetch_btn_{gen_key_id}"):
                    if not sh_gen_id: st.warning("Enter ID")
                    else:
                        with st.spinner("Downloading..."):
                            imgs, err = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_gen_id)
                            if imgs:
                                _, _, handle, _ = get_shopify_product_details(sh_secret_shop, sh_secret_token, sh_gen_id)
                                if handle:
                                    clean_shop = sh_secret_shop.replace("https://", "").replace("http://", "").strip()
                                    if not clean_shop.endswith(".myshopify.com"): clean_shop += ".myshopify.com"
                                    # Save URL directly to the widget key
                                    product_url = f"https://{clean_shop.replace('.myshopify.com', '.com')}/products/{handle}"
                                    st.session_state[f"gen_post_url_{gen_key_id}"] = product_url
                                st.session_state.gen_shopify_imgs = imgs
                                # Save fetched Product ID and increment upload counter to refresh Upload section
                                st.session_state['gen_fetched_prod_id'] = sh_gen_id
                                if "gen_upload_counter" not in st.session_state: st.session_state.gen_upload_counter = 0
                                st.session_state.gen_upload_counter += 1
                                st.success(f"Loaded {len(imgs)} images"); st.rerun()
                            else: st.error(err)
                if col_clear.button("❌ Clear", key=f"gen_clear_btn_{gen_key_id}"):
                    st.session_state.gen_shopify_imgs = []
                    st.session_state.image_generated_success = False
                    st.session_state.current_generated_image = None
                    st.session_state.gen_tags_result = {}
                    st.session_state.gen_fetched_prod_id = ""
                    # Clear URL by incrementing key counter (creates new widget)
                    st.session_state.gen_key_counter += 1
                    st.rerun()
            else: st.info("Set Secrets to use Import")

        images_to_send = []
        if st.session_state.gen_shopify_imgs:
            images_to_send = st.session_state.gen_shopify_imgs
            st.info(f"Using {len(images_to_send)} images from Shopify")
            try:
                zip_gen = BytesIO()
                with zipfile.ZipFile(zip_gen, "w") as zf:
                    for i, img in enumerate(images_to_send):
                        buf = BytesIO(); img.save(buf, format="JPEG", quality=95)
                        zf.writestr(f"shopify_orig_{i+1}.jpg", buf.getvalue())
                st.download_button("💾 Download All Originals (.zip)", data=zip_gen.getvalue(), file_name="shopify_original_images.zip", mime="application/zip", key=f"gen_download_zip_{gen_key_id}")
            except: pass
        else:
            files = st.file_uploader("Upload Manual", accept_multiple_files=True, type=["jpg","png"], key=f"gen_up_{gen_key_id}")
            images_to_send = [Image.open(f) for f in files] if files else []
        if images_to_send:
            cols = st.columns(4)
            for i, img in enumerate(images_to_send): cols[i%4].image(img, use_column_width=True)

    with c2:
        st.subheader("2. Settings")
        current_text_model = st.session_state.get('selected_text_model', 'Gemini')
        st.caption(f"🤖 SEO Tags Model: **{current_text_model}**")
        lib = st.session_state.library
        cats = list(set(p.get('category','Other') for p in lib)) if lib else []
        sel_cat = st.selectbox("Category", cats, key=f"gen_cat_{gen_key_id}") if cats else None
        filtered = [p for p in lib if p.get('category') == sel_cat]
        if filtered:
            sel_style = st.selectbox("Style", filtered, format_func=lambda x: x.get('name','Unknown'), key=f"gen_style_{gen_key_id}")
            if sel_style.get("sample_url"): safe_st_image(sel_style["sample_url"], width=100)
            
            # Get current style id
            current_style_id = sel_style.get('id', 'default')
            
            # Widget key for the text_area
            prompt_widget_key = f"gen_prompt_display_{gen_key_id}_{current_style_id}"
            
            # Key to track last variable values
            vars_tracker_key = f"gen_vars_tracker_{gen_key_id}_{current_style_id}"
            
            # Initialize with template if widget key not exists
            if prompt_widget_key not in st.session_state:
                st.session_state[prompt_widget_key] = sel_style.get('template','')
            
            # Variables input
            vars_list = [v.strip() for v in sel_style.get('variables','').split(",") if v.strip()]
            
            if vars_list:
                st.write("**Variables:**")
                cols_vars = st.columns(len(vars_list)) if len(vars_list) <= 3 else st.columns(3)
                user_vals = {}
                for idx, v in enumerate(vars_list):
                    with cols_vars[idx % len(cols_vars)]:
                        user_vals[v] = st.text_input(v, key=f"gen_var_{v}_{gen_key_id}_{current_style_id}")
                
                # Check if variables changed from last time
                last_vars = st.session_state.get(vars_tracker_key, {})
                vars_changed = (user_vals != last_vars)
                
                # Only update prompt if variables changed
                if vars_changed:
                    current_prompt = sel_style.get('template','')
                    for k, val in user_vals.items(): 
                        current_prompt = current_prompt.replace(f"{{{k}}}", val)
                    st.session_state[prompt_widget_key] = current_prompt
                    st.session_state[vars_tracker_key] = user_vals.copy()
            
            # Editable text area - value comes from session state key automatically
            prompt_edit = st.text_area("✏️ Custom Instruction (edit if needed)", height=100, key=prompt_widget_key)
            
            # Product URL - auto-filled when Fetch from Shopify (value set directly to session state key)
            url_input = st.text_input("Product URL (Optional):", key=f"gen_post_url_{gen_key_id}", help="Auto-filled from Shopify. AI will use URL context for SEO tags")

            if st.button("🚀 GENERATE", type="primary", use_container_width=True, key=f"gen_run_btn_{gen_key_id}"):
                if not gemini_key or not images_to_send: st.error("Check Key & Images")
                else:
                    with st.spinner("Generating Image..."):
                        d, e = generate_image(gemini_key, images_to_send, prompt_edit)
                        if d:
                            st.session_state.current_generated_image = d
                            st.session_state.image_generated_success = True
                            # Increment tags counter to force refresh of file name/alt tag fields
                            if "gen_tags_counter" not in st.session_state: st.session_state.gen_tags_counter = 0
                            st.session_state.gen_tags_counter += 1
                            
                            # Generate SEO tags by analyzing the GENERATED IMAGE + Product URL
                            current_url = url_input
                            with st.spinner("Analyzing image for SEO tags..."):
                                tags_json, _ = generate_seo_from_generated_image(gemini_key, claude_key, openai_key, current_text_model, d, current_url)
                                if tags_json:
                                    parsed_tags = parse_json_response(tags_json)
                                    st.session_state.gen_tags_result = parsed_tags if parsed_tags else {}
                                else: st.session_state.gen_tags_result = {}
                            st.rerun()
                        else: st.error(e)

            if st.session_state.image_generated_success and st.session_state.current_generated_image:
                st.divider(); st.subheader("✨ Result")
                st.image(st.session_state.current_generated_image, use_column_width=True)
                
                # Edit/Refine section
                with st.container(border=True):
                    st.write("🔄 **Edit / Refine Image**")
                    edit_counter = st.session_state.get("gen_edit_counter", 0)
                    edit_prompt = st.text_input(
                        "Edit Instruction:", 
                        placeholder="เช่น: ให้ใส่แหวนในนิ้วนางแทน, ให้มือขาวกว่าในรูป, ให้แหวนวงเล็กกว่านี้ 30%",
                        key=f"gen_edit_prompt_{gen_key_id}_{edit_counter}"
                    )
                    if st.button("🔄 Regenerate", type="secondary", use_container_width=True, key=f"gen_edit_btn_{gen_key_id}_{edit_counter}"):
                        if not edit_prompt:
                            st.warning("กรุณาใส่คำสั่งแก้ไข")
                        elif not gemini_key:
                            st.error("Missing Gemini API Key")
                        else:
                            with st.spinner("Regenerating with edit..."):
                                # Convert current generated image bytes to PIL
                                current_img = Image.open(BytesIO(st.session_state.current_generated_image))
                                # Generate with edit prompt using the current generated image
                                d, e = generate_image(gemini_key, [current_img], edit_prompt)
                                if d:
                                    st.session_state.current_generated_image = d
                                    # Increment counters to refresh widgets
                                    if "gen_tags_counter" not in st.session_state: st.session_state.gen_tags_counter = 0
                                    st.session_state.gen_tags_counter += 1
                                    if "gen_edit_counter" not in st.session_state: st.session_state.gen_edit_counter = 0
                                    st.session_state.gen_edit_counter += 1
                                    
                                    # Regenerate SEO tags for edited image
                                    current_url = st.session_state.get(f"gen_post_url_{gen_key_id}", "")
                                    with st.spinner("Analyzing edited image for SEO tags..."):
                                        tags_json, _ = generate_seo_from_generated_image(gemini_key, claude_key, openai_key, current_text_model, d, current_url)
                                        if tags_json:
                                            parsed_tags = parse_json_response(tags_json)
                                            st.session_state.gen_tags_result = parsed_tags if parsed_tags else {}
                                        else: st.session_state.gen_tags_result = {}
                                    st.rerun()
                                else:
                                    st.error(e)
                
                st.download_button("💾 Download Image", st.session_state.current_generated_image, "gen.jpg", "image/jpeg", type="secondary", key=f"gen_dl_img_{gen_key_id}")
                st.divider(); st.subheader("☁️ Upload to Shopify")
                with st.container(border=True):
                    tags_data = st.session_state.get("gen_tags_result", {})
                    tags_counter = st.session_state.get("gen_tags_counter", 0)
                    col_tags1, col_tags2 = st.columns(2)
                    # Use tags_counter in key to force refresh when new image is generated
                    final_filename = col_tags1.text_input("File Name", value=tags_data.get("file_name", ""), key=f"gen_filename_{gen_key_id}_{tags_counter}")
                    final_alt = col_tags2.text_input("Alt Tag", value=tags_data.get("alt_tag", ""), key=f"gen_alt_{gen_key_id}_{tags_counter}")
                    s_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
                    s_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
                    # Get Product ID from fetched value (updates on each Fetch)
                    fetched_prod_id = st.session_state.get("gen_fetched_prod_id", "")
                    upload_counter = st.session_state.get("gen_upload_counter", 0)
                    col_u1, col_u2 = st.columns([3, 1])
                    # Use upload_counter in key so widget refreshes when Fetch is clicked
                    u_prod_id = col_u1.text_input("Product ID", value=fetched_prod_id, key=f"gen_upload_prodid_{gen_key_id}_{upload_counter}")
                    if col_u2.button("🚀 Upload", type="primary", use_container_width=True, key=f"gen_upload_btn_{gen_key_id}_{upload_counter}"):
                        if not s_shop or not s_token: st.error("Missing Shopify Secrets")
                        elif not u_prod_id: st.warning("Enter Product ID")
                        else:
                            with st.spinner("Uploading..."):
                                success, msg = add_single_image_to_shopify(s_shop, s_token, u_prod_id, st.session_state.current_generated_image, file_name=final_filename, alt_tag=final_alt)
                                if success: st.success(msg)
                                else: st.error(msg)

# === TAB RETOUCH ===
with tab_retouch:
    st.header("🎨 Retouch (via Gemini)")
    if "shopify_fetched_imgs" not in st.session_state: st.session_state.shopify_fetched_imgs = []
    rt_key_id = st.session_state.retouch_key_counter
    rt_c1, rt_c2 = st.columns([1, 1.2])
    
    with rt_c1:
        st.subheader("1. Input Images")
        with st.expander("🛍️ Import from Shopify", expanded=True):
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            if sh_secret_shop and sh_secret_token:
                st.success("✅ Shopify Connected")
                sh_imp_id = st.text_input("Product ID to Fetch", key=f"rt_imp_id_{rt_key_id}")
                c_fetch, c_clear = st.columns([2,1])
                if c_fetch.button("⬇️ Fetch Images", key=f"rt_fetch_btn_{rt_key_id}"):
                    if not sh_imp_id: st.warning("Enter ID")
                    else:
                        with st.spinner("Downloading..."):
                            imgs, err = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_imp_id)
                            if imgs:
                                st.session_state.shopify_fetched_imgs = imgs
                                # Save fetched Product ID and increment upload counter
                                st.session_state['rt_fetched_prod_id'] = sh_imp_id
                                if "rt_upload_counter" not in st.session_state: st.session_state.rt_upload_counter = 0
                                st.session_state.rt_upload_counter += 1
                                st.success(f"Loaded {len(imgs)} images!"); st.rerun()
                            else: st.error(err)
                if c_clear.button("❌ Clear", key=f"rt_clear_btn_{rt_key_id}"):
                    st.session_state.shopify_fetched_imgs = []
                    st.session_state.retouch_results = None
                    st.session_state.seo_name_result = None
                    st.session_state.rt_fetched_prod_id = ""
                    st.session_state.retouch_key_counter += 1  # Reset all widgets including Product ID
                    st.rerun()
            else: st.info("Set Secrets to use.")
        
        rt_imgs, source_type = [], ""
        if st.session_state.shopify_fetched_imgs:
            rt_imgs = st.session_state.shopify_fetched_imgs
            source_type = "Shopify"
            st.info(f"Using {len(rt_imgs)} images from Shopify")
            try:
                zip_orig = BytesIO()
                with zipfile.ZipFile(zip_orig, "w") as zf:
                    for i, img in enumerate(rt_imgs):
                        buf = BytesIO(); img.save(buf, format="JPEG", quality=95)
                        zf.writestr(f"original_{i+1}.jpg", buf.getvalue())
                st.download_button("💾 Download Originals (.zip)", data=zip_orig.getvalue(), file_name="originals.zip", mime="application/zip", key=f"rt_dl_orig_{rt_key_id}")
            except: pass
        else:
            rt_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png"], key=f"rt_up_{rt_key_id}")
            if rt_files: rt_imgs = [Image.open(f) for f in rt_files]; source_type = "Upload"
        if rt_imgs:
            with st.expander(f"📸 View Input ({len(rt_imgs)} images)", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(rt_imgs): cols[i%4].image(img, use_column_width=True)
        else: st.warning("Waiting for images...")

    with rt_c2:
        st.subheader("2. Prompt Settings")
        lib = st.session_state.library
        rt_cats = list(set(p.get('category','Other') for p in lib)) if lib else []
        default_cat_index = rt_cats.index("Retouch") if "Retouch" in rt_cats else 0
        rt_sel_cat = st.selectbox("Category", rt_cats, index=default_cat_index, key=f"rt_cat_{rt_key_id}") if rt_cats else None
        rt_filtered = [p for p in lib if p.get('category') == rt_sel_cat]
        if rt_filtered:
            rt_style = st.selectbox("Style", rt_filtered, format_func=lambda x: x.get('name','Unknown'), key=f"rt_style_{rt_key_id}")
            
            # Get current style id for dynamic key
            current_rt_style_id = rt_style.get('id', 'default')
            
            # Widget key for the text_area
            rt_prompt_widget_key = f"rt_prompt_display_{rt_key_id}_{current_rt_style_id}"
            
            # Key to track last variable values
            rt_vars_tracker_key = f"rt_vars_tracker_{rt_key_id}_{current_rt_style_id}"
            
            # Initialize with template if widget key not exists
            if rt_prompt_widget_key not in st.session_state:
                st.session_state[rt_prompt_widget_key] = rt_style.get('template','')
            
            # Variables input
            rt_vars = [v.strip() for v in rt_style.get('variables','').split(",") if v.strip()]
            
            if rt_vars:
                st.write("**Variables:**")
                cols_vars = st.columns(len(rt_vars)) if len(rt_vars) <= 3 else st.columns(3)
                rt_user_vals = {}
                for idx, v in enumerate(rt_vars):
                    with cols_vars[idx % len(cols_vars)]:
                        rt_user_vals[v] = st.text_input(v, key=f"rt_var_{v}_{rt_key_id}_{current_rt_style_id}")
                
                # Check if variables changed from last time
                last_rt_vars = st.session_state.get(rt_vars_tracker_key, {})
                rt_vars_changed = (rt_user_vals != last_rt_vars)
                
                # Only update prompt if variables changed
                if rt_vars_changed:
                    current_rt_prompt = rt_style.get('template','')
                    for k, val in rt_user_vals.items(): 
                        current_rt_prompt = current_rt_prompt.replace(f"{{{k}}}", val)
                    st.session_state[rt_prompt_widget_key] = current_rt_prompt
                    st.session_state[rt_vars_tracker_key] = rt_user_vals.copy()
            
            # Editable text area - value comes from session state key automatically
            rt_prompt_edit = st.text_area("✏️ Custom Instruction (edit if needed)", height=100, key=rt_prompt_widget_key)
            
            c_rt1, c_rt2 = st.columns([1, 1])
            run_retouch = c_rt1.button("🚀 Run Batch", type="primary", disabled=(not rt_imgs), key=f"rt_run_btn_{rt_key_id}")
            clear_retouch = c_rt2.button("🔄 Start Over", key=f"rt_startover_btn_{rt_key_id}")
            if clear_retouch:
                st.session_state.retouch_results = None; st.session_state.seo_name_result = None
                st.session_state.shopify_fetched_imgs = []; st.session_state.retouch_key_counter += 1
                if 'rt_upload_id' in st.session_state: del st.session_state['rt_upload_id']
                st.rerun()
            if run_retouch:
                if not gemini_key: st.error("Missing Gemini API Key!")
                else:
                    rt_temp_results = []
                    rt_pbar = st.progress(0)
                    for i, img in enumerate(rt_imgs):
                        with st.spinner(f"Processing #{i+1}..."):
                            gen_img_bytes, err = generate_image(gemini_key, [img], rt_prompt_edit)
                            rt_pbar.progress((i+1)/len(rt_imgs))
                            rt_temp_results.append(gen_img_bytes if gen_img_bytes else None)
                            if err: st.error(f"Failed #{i+1}: {err}")
                    st.session_state.retouch_results = rt_temp_results
                    st.success("Batch Complete!"); st.rerun()

    if st.session_state.retouch_results:
        st.divider(); st.subheader("🎨 Retouched Results")
        try:
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for i, res_bytes in enumerate(st.session_state.retouch_results):
                    if res_bytes: zf.writestr(f"retouched_{i+1}.jpg", res_bytes)
            st.download_button("📦 Download All (.zip)", data=zip_buf.getvalue(), file_name="retouched.zip", mime="application/zip", type="primary", key=f"rt_dl_all_{rt_key_id}")
        except: pass
        cols = st.columns(3)
        for i, res_bytes in enumerate(st.session_state.retouch_results):
            with cols[i%3]:
                st.write(f"**#{i+1}**")
                if res_bytes: st.image(res_bytes, use_column_width=True)
                else: st.error("Failed")
        st.markdown("---"); st.subheader("🚀 Upload to Shopify")
        with st.container(border=True):
            rt_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            rt_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            # Get Product ID from fetched value (updates on each Fetch)
            fetched_rt_prod_id = st.session_state.get("rt_fetched_prod_id", "")
            rt_upload_counter = st.session_state.get("rt_upload_counter", 0)
            col_rt_u1, col_rt_u2 = st.columns([2, 1])
            # Use upload_counter in key so widget refreshes when Fetch is clicked
            rt_prod_id = col_rt_u1.text_input("Product ID", value=fetched_rt_prod_id, key=f"rt_upload_id_{rt_key_id}_{rt_upload_counter}")
            if col_rt_u2.button("☁️ Upload & Replace", type="primary", use_container_width=True, key=f"rt_upload_btn_{rt_key_id}_{rt_upload_counter}"):
                if not rt_shop or not rt_token: st.error("Missing Secrets")
                elif not rt_prod_id: st.warning("Enter ID")
                elif not any(st.session_state.retouch_results): st.warning("No images")
                else:
                    with st.spinner("Uploading..."):
                        success, msg = upload_only_images_to_shopify(rt_shop, rt_token, rt_prod_id, st.session_state.retouch_results)
                        if success: st.success(msg); st.balloons()
                        else: st.error(msg)

    st.markdown("---"); st.subheader("🛍️ SEO Name & Slug Generator")
    current_text_model = st.session_state.get('selected_text_model', 'Gemini')
    st.caption(f"🤖 Using: **{current_text_model}**")
    target_images_for_seo = []
    if st.session_state.retouch_results and any(st.session_state.retouch_results):
        target_images_for_seo = [x for x in st.session_state.retouch_results if x]
        source_label = "Retouched"
    elif rt_imgs: target_images_for_seo = rt_imgs; source_label = source_type
    else: source_label = "None"
    c_seo1, c_seo2 = st.columns([1, 1])
    with c_seo1:
        user_product_desc = st.text_input("Description", placeholder="e.g., sterling silver bracelet", key=f"rt_seo_desc_{rt_key_id}")
        st.write(f"Source: {source_label}")
        if st.button("✨ Analyze", key=f"rt_seo_analyze_btn_{rt_key_id}"):
            if not target_images_for_seo: st.warning("No images.")
            elif not user_product_desc: st.warning("Enter description.")
            else:
                with st.spinner("Analyzing..."):
                    seo_json, seo_err = generate_seo_name_slug(gemini_key, claude_key, openai_key, current_text_model, target_images_for_seo, user_product_desc)
                    if seo_json:
                        res_dict = parse_json_response(seo_json)
                        if res_dict: st.session_state.seo_name_result = res_dict
                        else: st.error("Parse failed"); st.code(seo_json)
                    else: st.error(seo_err)
    with c_seo2:
        if st.session_state.seo_name_result:
            res = st.session_state.seo_name_result
            st.success("Done!")
            st.write("**Product Name:**"); st.text_input("Name", value=res.get("product_name", ""), label_visibility="collapsed", key=f"rt_res_name_{rt_key_id}")
            st.write("**URL Slug:**"); st.code(res.get("url_slug", ""))

# === TAB 2: BULK SEO ===
with tab2:
    st.header("🏷️ Bulk SEO Tags")
    current_text_model = st.session_state.get('selected_text_model', 'Gemini')
    st.caption(f"🤖 Using: **{current_text_model}**")
    bulk_key_id = st.session_state.bulk_key_counter
    bc1, bc2 = st.columns([1, 1.5])
    with bc1:
        bfiles = st.file_uploader("Upload Images", accept_multiple_files=True, key=f"bulk_up_{bulk_key_id}")
        bimgs = [Image.open(f) for f in bfiles] if bfiles else []
        if bimgs: st.success(f"{len(bimgs)} images")
    with bc2:
        burl = st.text_input("Product URL:", key=f"bulk_url_{bulk_key_id}")
        c_btn1, c_btn2 = st.columns([1, 1])
        run_batch = c_btn1.button("🚀 Run Batch", type="primary", disabled=(not bimgs), key=f"bulk_run_btn_{bulk_key_id}")
        if c_btn2.button("🔄 Start Over", key=f"bulk_startover_btn_{bulk_key_id}"):
            st.session_state.bulk_results = None; st.session_state.bulk_key_counter += 1; st.rerun()
        if run_batch:
            # Check API key based on selected model
            missing_key = False
            if current_text_model == "Gemini" and not gemini_key: missing_key = True
            elif current_text_model in CLAUDE_MODELS and not claude_key: missing_key = True
            elif current_text_model in OPENAI_MODELS and not openai_key: missing_key = True
            
            if missing_key: st.error("Missing API Key")
            elif not burl: st.error("Missing URL")
            else:
                pbar = st.progress(0); temp_results = []
                for i, img in enumerate(bimgs):
                    with st.spinner(f"Processing #{i+1}..."):
                        txt, err = generate_seo_for_existing_image(gemini_key, claude_key, openai_key, current_text_model, img, burl)
                        pbar.progress((i+1)/len(bimgs))
                        if txt:
                            d = parse_json_response(txt)
                            if isinstance(d, list) and d: d = d[0]
                            temp_results.append(d if isinstance(d, dict) else {"error": "Invalid format", "raw": txt})
                        else: temp_results.append({"error": err})
                st.session_state.bulk_results = temp_results; st.success("Done!"); st.rerun()
    if st.session_state.bulk_results and bimgs:
        st.divider()
        for i, res in enumerate(st.session_state.bulk_results):
            if i < len(bimgs):
                rc1, rc2 = st.columns([1, 3])
                with rc1: st.image(bimgs[i], width=150)
                with rc2:
                    if "error" in res: st.error(res.get('error')); st.code(res.get('raw', '')) if 'raw' in res else None
                    else: st.write("**File Name:**"); st.code(res.get('file_name', '')); st.write("**Alt Tag:**"); st.code(res.get('alt_tag', ''))
                st.divider()

# === TAB 3: WRITER ===
with tab3:
    st.header("📝 Product Writer")
    current_text_model = st.session_state.get('selected_text_model', 'Gemini')
    st.caption(f"🤖 Using: **{current_text_model}**")
    writer_key_id = st.session_state.writer_key_counter
    if "writer_shopify_imgs" not in st.session_state: st.session_state.writer_shopify_imgs = []
    text_area_key = f"w_raw_{writer_key_id}"
    c1, c2 = st.columns([1, 1.2])
    with c1:
        with st.expander("🛍️ Import from Shopify", expanded=True):
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            if sh_secret_shop and sh_secret_token:
                sh_writer_id = st.text_input("Product ID", key=f"writer_shopify_id_{writer_key_id}")
                col_w_fetch, col_w_clear = st.columns([2, 1])
                if col_w_fetch.button("⬇️ Fetch All", key=f"writer_fetch_btn_{writer_key_id}"):
                    if not sh_writer_id: st.warning("Enter ID")
                    else:
                        with st.spinner("Fetching..."):
                            imgs, _ = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_writer_id)
                            desc_html, _, _, _ = get_shopify_product_details(sh_secret_shop, sh_secret_token, sh_writer_id)
                            if imgs: st.session_state.writer_shopify_imgs = imgs
                            if desc_html: st.session_state[text_area_key] = remove_html_tags(desc_html)
                            # Save fetched Product ID and increment publish counter
                            st.session_state['writer_fetched_prod_id'] = sh_writer_id
                            if "writer_publish_counter" not in st.session_state: st.session_state.writer_publish_counter = 0
                            st.session_state.writer_publish_counter += 1
                            st.success("Loaded!"); st.rerun()
                if col_w_clear.button("❌ Clear", key=f"writer_clear_btn_{writer_key_id}"):
                    st.session_state.writer_shopify_imgs = []
                    st.session_state.writer_result = None
                    st.session_state.writer_fetched_prod_id = ""
                    st.session_state.writer_key_counter += 1  # Reset all widgets including Product ID
                    st.rerun()
        writer_imgs = st.session_state.writer_shopify_imgs if st.session_state.writer_shopify_imgs else []
        if not writer_imgs:
            files = st.file_uploader("Images (Optional)", type=["jpg", "png"], accept_multiple_files=True, key=f"w_img_{writer_key_id}")
            writer_imgs = [Image.open(f) for f in files] if files else []
        if writer_imgs:
            with st.expander(f"📸 Preview ({len(writer_imgs)} images)", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(writer_imgs): cols[i%4].image(img, use_column_width=True)
        raw = st.text_area("Paste Details:", height=300, key=text_area_key)
        wb1, wb2 = st.columns([1, 1])
        run_write = wb1.button("🚀 Generate Content", type="primary", key=f"writer_run_btn_{writer_key_id}")
        if wb2.button("🔄 Start Over", key=f"writer_startover_btn_{writer_key_id}"):
            st.session_state.writer_result = None; st.session_state.writer_shopify_imgs = []; st.session_state.writer_fetched_prod_id = ""; st.session_state.writer_key_counter += 1; st.rerun()
    with c2:
        if run_write:
            # Check API key based on selected model
            missing_key = False
            if selected_text_model == "Gemini" and not gemini_key: missing_key = True
            elif selected_text_model in CLAUDE_MODELS and not claude_key: missing_key = True
            elif selected_text_model in OPENAI_MODELS and not openai_key: missing_key = True
            
            if missing_key: st.error("Missing API Key")
            elif not raw: st.error("Missing details")
            else:
                with st.spinner(f"Writing with {current_text_model}..."):
                    # Fetch real store catalog for internal linking
                    catalog_text = ""
                    try:
                        with st.spinner("📦 Loading store catalog for internal links..."):
                            catalog = fetch_store_catalog("www.bikerringshop.com")
                            if catalog.get("collections") or catalog.get("products"):
                                catalog_text = format_catalog_for_prompt(catalog)
                    except: pass
                    json_txt, err = generate_full_product_content(gemini_key, claude_key, openai_key, current_text_model, writer_imgs, raw, catalog_text)
                    if json_txt:
                        d = parse_json_response(json_txt)
                        if isinstance(d, list) and d: d = d[0]
                        if isinstance(d, dict): st.session_state.writer_result = d; st.rerun()
                        else: st.code(json_txt)
                    else: st.error(err)
        if st.session_state.writer_result:
            d = st.session_state.writer_result
            st.subheader("Content Results")
            st.write("**H1:**"); st.code(d.get('product_title_h1', ''))
            st.write("**Slug:**"); st.code(d.get('url_slug', ''))
            st.write("**Meta Title:**"); st.code(d.get('meta_title', ''))
            st.write("**Meta Desc:**"); st.code(d.get('meta_description', ''))
            with st.expander("HTML Content"): st.code(d.get('html_content', ''), language="html")
            st.markdown(d.get('html_content', ''), unsafe_allow_html=True)
            st.divider(); st.subheader("🖼️ Image SEO")
            img_tags = d.get('image_seo', [])
            if writer_imgs:
                for i, img in enumerate(writer_imgs):
                    ic1, ic2 = st.columns([1, 3])
                    with ic1: st.image(img, width=120)
                    with ic2:
                        if i < len(img_tags):
                            item = img_tags[i]
                            st.write("**File:**"); st.code(clean_filename(item.get('file_name', '')) if isinstance(item, dict) else "N/A")
                            st.write("**Alt:**"); st.code(item.get('alt_tag', '') if isinstance(item, dict) else str(item))
                    st.divider()
            st.markdown("---"); st.subheader("🚀 Publish to Shopify")
            with st.container(border=True):
                secret_shop = st.secrets.get("SHOPIFY_SHOP_URL")
                secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN")
                s_shop, s_token, s_prod_id = None, None, None
                # Get Product ID from fetched value (updates on each Fetch)
                fetched_writer_id = st.session_state.get("writer_fetched_prod_id", "")
                writer_publish_counter = st.session_state.get("writer_publish_counter", 0)
                if secret_shop and secret_token:
                    col_info, col_input = st.columns([1, 1])
                    with col_info: st.success("✅ Credentials Loaded"); s_shop = secret_shop; s_token = secret_token
                    # Use publish_counter in key so widget refreshes when Fetch is clicked
                    with col_input: s_prod_id = st.text_input("Product ID", value=fetched_writer_id, key=f"writer_prod_id_{writer_key_id}_{writer_publish_counter}")
                else: 
                    st.warning("⚠️ Credentials Required")
                    c_x1, c_x2, c_x3 = st.columns(3)
                    s_shop = c_x1.text_input("Shop URL", key=f"writer_shop_{writer_key_id}")
                    s_token = c_x2.text_input("Token", type="password", key=f"writer_token_{writer_key_id}")
                    s_prod_id = c_x3.text_input("Product ID", value=fetched_writer_id, key=f"writer_prodid2_{writer_key_id}_{writer_publish_counter}")
                enable_img_upload = st.checkbox("📷 Upload Images", value=True, key=f"writer_imgchk_{writer_key_id}")
                if st.button("☁️ Update Product", type="primary", use_container_width=True, key=f"writer_update_btn_{writer_key_id}_{writer_publish_counter}"):
                    if not s_shop or not s_token or not s_prod_id: st.error("❌ Missing Data")
                    else:
                        with st.spinner("Updating..."):
                            success, msg = update_shopify_product_v2(s_shop, s_token, s_prod_id, st.session_state.writer_result, writer_imgs, enable_img_upload)
                            if success: st.success(msg); st.balloons()
                            else: st.error(msg)

# === TAB BATCH WRITER ===
with tab_batch:
    st.header("📝 Batch Product Writer")
    current_text_model = st.session_state.get('selected_text_model', 'Gemini')
    
    # Check Shopify credentials
    bw_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
    bw_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
    
    if not bw_shop or not bw_token:
        st.error("❌ Shopify credentials required. Set SHOPIFY_SHOP_URL and SHOPIFY_ACCESS_TOKEN in Secrets.")
    else:
        # --- TOP: Model selector + Load products ---
        bw_top1, bw_top2 = st.columns([1, 1])
        with bw_top1:
            all_models_batch = ["Gemini"] + list(CLAUDE_MODELS.keys()) + list(OPENAI_MODELS.keys())
            batch_model = st.selectbox("🤖 Model for Batch:", all_models_batch, 
                                       index=all_models_batch.index(current_text_model) if current_text_model in all_models_batch else 0,
                                       key="batch_model_select")
        with bw_top2:
            st.write("")  # spacer
            st.write("")
            if st.button("📦 Load Products from Shopify", type="primary", use_container_width=True, key="batch_load_btn"):
                load_progress = st.empty()
                load_status = st.empty()
                
                load_status.info("📂 Loading collections...")
                collections = get_shopify_all_collections(bw_shop, bw_token)
                load_status.info(f"✅ {len(collections)} collections loaded. Now loading products...")
                
                def progress_cb(count, page):
                    load_progress.progress(min(page / 25, 0.99), text=f"Loaded {count} products (page {page})...")
                
                products, err = get_shopify_all_products(bw_shop, bw_token, progress_callback=progress_cb)
                load_progress.empty()
                
                if err and not products:
                    load_status.error(f"Failed: {err}")
                else:
                    st.session_state.batch_products = products
                    st.session_state.batch_collections = collections
                    st.session_state.batch_results = {}
                    warn_msg = f" ⚠️ {err}" if err else ""
                    load_status.success(f"✅ Loaded {len(products)} products, {len(collections)} collections{warn_msg}")
                    st.rerun()
        
        if st.session_state.batch_products:
            products_df = st.session_state.batch_products
            
            st.divider()
            # --- FILTERS ---
            st.subheader("🔍 Filter Products")
            fc1, fc2, fc3 = st.columns(3)
            
            with fc1:
                search_term = st.text_input("Search by Name or SKU:", key="batch_search", placeholder="e.g. skull ring, SKU-001")
            
            with fc2:
                collection_options = ["All Collections"] + [c["title"] for c in st.session_state.batch_collections]
                selected_collection = st.selectbox("Product Collection:", collection_options, key="batch_collection_filter")
            
            with fc3:
                stock_filter = st.selectbox("Stock Filter:", ["All", "In Stock (> 0)", "Out of Stock (≤ 0)"], key="batch_stock_filter")
            
            # Apply filters
            filtered = products_df.copy()
            
            if search_term:
                term = search_term.lower()
                filtered = [p for p in filtered if term in p["title"].lower() or term in p.get("sku", "").lower() or term in p.get("all_skus", "").lower()]
            
            if selected_collection != "All Collections":
                # Need to fetch products for that collection
                sel_col = next((c for c in st.session_state.batch_collections if c["title"] == selected_collection), None)
                if sel_col:
                    col_products, _ = get_shopify_all_products(bw_shop, bw_token, collection_id=sel_col["id"], max_pages=5)
                    col_ids = {p["id"] for p in col_products}
                    filtered = [p for p in filtered if p["id"] in col_ids]
            
            if stock_filter == "In Stock (> 0)":
                filtered = [p for p in filtered if p["total_inventory"] > 0]
            elif stock_filter == "Out of Stock (≤ 0)":
                filtered = [p for p in filtered if p["total_inventory"] <= 0]
            
            st.caption(f"Showing **{len(filtered)}** of {len(products_df)} products")
            
            # --- PRODUCT TABLE WITH CHECKBOXES ---
            if filtered:
                # Select all / deselect
                sa_col1, sa_col2, sa_col3 = st.columns([1, 1, 4])
                select_all = sa_col1.button("☑️ Select All", key="batch_select_all")
                deselect_all = sa_col2.button("☐ Deselect All", key="batch_deselect_all")
                
                if select_all:
                    for p in filtered:
                        st.session_state[f"batch_chk_{p['id']}"] = True
                    st.rerun()
                if deselect_all:
                    for p in filtered:
                        st.session_state[f"batch_chk_{p['id']}"] = False
                    st.rerun()
                
                # Table header
                hdr_cols = st.columns([0.3, 0.5, 2.5, 1, 0.8, 1])
                hdr_cols[0].write("**✓**")
                hdr_cols[1].write("**Image**")
                hdr_cols[2].write("**Product Name**")
                hdr_cols[3].write("**SKU**")
                hdr_cols[4].write("**Stock**")
                hdr_cols[5].write("**Status**")
                st.markdown("<hr style='margin:2px 0'>", unsafe_allow_html=True)
                
                # Show products (paginated to avoid lag)
                page_size = 50
                total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
                if "batch_page" not in st.session_state: st.session_state.batch_page = 0
                current_page_items = filtered[st.session_state.batch_page * page_size : (st.session_state.batch_page + 1) * page_size]
                
                for p in current_page_items:
                    row_cols = st.columns([0.3, 0.5, 2.5, 1, 0.8, 1])
                    result_status = st.session_state.batch_results.get(p["id"])
                    
                    with row_cols[0]:
                        st.checkbox("", key=f"batch_chk_{p['id']}", label_visibility="collapsed")
                    with row_cols[1]:
                        if p.get("image_url"):
                            try: st.image(p["image_url"], width=40)
                            except: st.write("📷")
                        else: st.write("—")
                    with row_cols[2]:
                        st.write(f"**{p['title'][:50]}**" + ("..." if len(p['title']) > 50 else ""))
                        st.caption(f"ID: {p['id']}")
                    with row_cols[3]:
                        st.caption(p.get("sku", "—") or "—")
                    with row_cols[4]:
                        inv = p["total_inventory"]
                        if inv > 0: st.write(f"🟢 {inv}")
                        else: st.write(f"🔴 {inv}")
                    with row_cols[5]:
                        if result_status:
                            if result_status.get("success"):
                                st.write("✅ Done")
                            elif result_status.get("error"):
                                st.write("❌ Fail")
                            elif result_status.get("generating"):
                                st.write("⏳ ...")
                        else:
                            st.write("⬜ Pending")
                
                # Pagination
                if total_pages > 1:
                    pg_cols = st.columns([1, 2, 1])
                    if pg_cols[0].button("◀ Prev", disabled=(st.session_state.batch_page == 0), key="batch_prev"):
                        st.session_state.batch_page -= 1; st.rerun()
                    pg_cols[1].write(f"Page {st.session_state.batch_page + 1} of {total_pages}")
                    if pg_cols[2].button("Next ▶", disabled=(st.session_state.batch_page >= total_pages - 1), key="batch_next"):
                        st.session_state.batch_page += 1; st.rerun()
                
                st.divider()
                
                # --- BATCH ACTIONS ---
                selected_products = [p for p in filtered if st.session_state.get(f"batch_chk_{p['id']}", False)]
                st.write(f"**Selected: {len(selected_products)} products**")
                
                act_col1, act_col2, act_col3 = st.columns([1, 1, 1])
                
                gen_only = act_col1.button("🚀 Generate Content", type="primary", 
                                           disabled=(len(selected_products) == 0), key="batch_gen_btn")
                gen_and_update = act_col2.button("🚀 Generate & Update Shopify", type="primary",
                                                  disabled=(len(selected_products) == 0), key="batch_gen_update_btn")
                if act_col3.button("🔄 Clear Results", key="batch_clear_results"):
                    st.session_state.batch_results = {}
                    st.rerun()
                
                auto_update = gen_and_update  # Flag: auto-update to Shopify after gen
                
                if gen_only or gen_and_update:
                    if len(selected_products) == 0:
                        st.warning("No products selected")
                    else:
                        # Check API key
                        batch_missing_key = False
                        if batch_model == "Gemini" and not gemini_key: batch_missing_key = True
                        elif batch_model in CLAUDE_MODELS and not claude_key: batch_missing_key = True
                        elif batch_model in OPENAI_MODELS and not openai_key: batch_missing_key = True
                        
                        if batch_missing_key:
                            st.error(f"❌ Missing API Key for {batch_model}")
                        else:
                            # Fetch catalog once for internal linking
                            catalog_text = ""
                            try:
                                catalog = fetch_store_catalog("www.bikerringshop.com")
                                if catalog.get("collections") or catalog.get("products"):
                                    catalog_text = format_catalog_for_prompt(catalog)
                            except: pass
                            
                            progress_bar = st.progress(0)
                            status_container = st.container()
                            
                            for idx, prod in enumerate(selected_products):
                                pid = prod["id"]
                                st.session_state.batch_results[pid] = {"generating": True}
                                
                                with status_container:
                                    st.write(f"⏳ [{idx+1}/{len(selected_products)}] **{prod['title'][:60]}** ...")
                                
                                # Build input from existing product data
                                raw_input = prod.get("body_html", "") or ""
                                raw_input = remove_html_tags(raw_input) if raw_input else ""
                                raw_input = f"Product Name: {prod['title']}\nProduct Type: {prod.get('product_type', '')}\nSKU: {prod.get('sku', '')}\n\n{raw_input}"
                                
                                # Fetch product images
                                prod_imgs = []
                                try:
                                    imgs, _ = get_shopify_product_images(bw_shop, bw_token, pid)
                                    if imgs: prod_imgs = imgs
                                except: pass
                                
                                # Generate content
                                try:
                                    json_txt, err = generate_full_product_content(
                                        gemini_key, claude_key, openai_key, batch_model, 
                                        prod_imgs, raw_input, catalog_text
                                    )
                                    
                                    if json_txt:
                                        d = parse_json_response(json_txt)
                                        if isinstance(d, list) and d: d = d[0]
                                        if isinstance(d, dict):
                                            result_entry = {"success": True, "data": d}
                                            
                                            # Auto-update to Shopify if requested
                                            if auto_update:
                                                try:
                                                    ok, msg = update_shopify_description_only(bw_shop, bw_token, pid, d)
                                                    result_entry["updated"] = ok
                                                    result_entry["update_msg"] = msg
                                                except Exception as ue:
                                                    result_entry["updated"] = False
                                                    result_entry["update_msg"] = str(ue)
                                            
                                            st.session_state.batch_results[pid] = result_entry
                                        else:
                                            st.session_state.batch_results[pid] = {"error": "Parse failed", "raw": json_txt[:500]}
                                    else:
                                        st.session_state.batch_results[pid] = {"error": err or "Generation failed"}
                                except Exception as e:
                                    st.session_state.batch_results[pid] = {"error": str(e)}
                                
                                progress_bar.progress((idx + 1) / len(selected_products))
                                time.sleep(0.5)  # Small delay between API calls
                            
                            st.success(f"✅ Batch complete! {len(selected_products)} products processed.")
                            st.rerun()
                
                # --- RESULTS REVIEW ---
                if st.session_state.batch_results:
                    st.divider()
                    st.subheader("📊 Batch Results")
                    
                    success_count = sum(1 for r in st.session_state.batch_results.values() if r.get("success"))
                    fail_count = sum(1 for r in st.session_state.batch_results.values() if r.get("error"))
                    updated_count = sum(1 for r in st.session_state.batch_results.values() if r.get("updated"))
                    
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("✅ Generated", success_count)
                    mc2.metric("❌ Failed", fail_count)
                    mc3.metric("☁️ Updated to Shopify", updated_count)
                    
                    # Show detailed results
                    for pid, result in st.session_state.batch_results.items():
                        prod_info = next((p for p in products_df if p["id"] == pid), None)
                        if not prod_info: continue
                        
                        with st.expander(f"{'✅' if result.get('success') else '❌'} {prod_info['title'][:60]} (ID: {pid})", expanded=False):
                            if result.get("success"):
                                d = result["data"]
                                st.write("**H1:**", d.get("product_title_h1", ""))
                                st.write("**Meta Title:**", d.get("meta_title", ""))
                                st.write("**Meta Desc:**", d.get("meta_description", ""))
                                if result.get("updated"):
                                    st.success(f"Shopify: {result.get('update_msg', 'Updated')}")
                                elif result.get("update_msg"):
                                    st.error(f"Shopify: {result.get('update_msg')}")
                                
                                with st.expander("HTML Preview"):
                                    st.markdown(d.get("html_content", ""), unsafe_allow_html=True)
                                
                                # Manual update button if not auto-updated
                                if not result.get("updated"):
                                    if st.button(f"☁️ Update to Shopify", key=f"batch_manual_update_{pid}"):
                                        with st.spinner("Updating..."):
                                            ok, msg = update_shopify_description_only(bw_shop, bw_token, pid, d)
                                            if ok:
                                                st.session_state.batch_results[pid]["updated"] = True
                                                st.session_state.batch_results[pid]["update_msg"] = msg
                                                st.success(msg)
                                                st.rerun()
                                            else:
                                                st.error(msg)
                            elif result.get("error"):
                                st.error(f"Error: {result['error']}")
                                if result.get("raw"):
                                    st.code(result["raw"][:500])
            else:
                st.info("No products match your filters.")
        else:
            st.info("👆 Click **Load Products from Shopify** to get started.")

# === TAB 4: LIBRARY ===
with tab4:
    st.subheader("🛠️ Library Manager")
    target = st.session_state.edit_target
    
    # Use dynamic form key based on whether editing or adding
    form_key = f"lib_form_{target['id']}" if target else "lib_form_new"
    
    with st.form(form_key, clear_on_submit=True):
        st.write(f"**{'Edit: '+target['name'] if target else 'Add New'}**")
        c1, c2 = st.columns(2)
        
        # Use dynamic keys with target id to force refresh
        key_suffix = target['id'] if target else "new"
        n = c1.text_input("Name", value=target['name'] if target else "", key=f"lib_name_{key_suffix}")
        c = c2.text_input("Category", value=target['category'] if target else "", key=f"lib_cat_{key_suffix}")
        t = st.text_area("Template", value=target['template'] if target else "", key=f"lib_template_{key_suffix}")
        v = st.text_input("Vars (comma separated)", value=target['variables'] if target else "", key=f"lib_vars_{key_suffix}")
        u = st.text_input("Sample URL", value=target['sample_url'] if target else "", key=f"lib_url_{key_suffix}")
        
        cols = st.columns([1, 1, 3])
        save_btn = cols[0].form_submit_button("💾 Save", type="primary")
        cancel_btn = cols[1].form_submit_button("❌ Cancel") if target else False
        
        if save_btn:
            new = {"id": target['id'] if target else str(int(time.time())), "name": n, "category": c, "template": t, "variables": v, "sample_url": u}
            if target:
                for idx, item in enumerate(st.session_state.library):
                    if item['id'] == target['id']: st.session_state.library[idx] = new; break
            else: 
                st.session_state.library.append(new)
            save_prompts(st.session_state.library)
            st.session_state.edit_target = None
            st.success("✅ Saved!")
            st.rerun()
            
        if cancel_btn: 
            st.session_state.edit_target = None
            st.rerun()
    
    st.divider()
    st.write("**📚 Prompt Library:**")
    for i, p in enumerate(st.session_state.library):
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1, 4, 1, 1])
            if p.get("sample_url"):
                with c1: safe_st_image(p["sample_url"], width=50)
            else:
                c1.write("📝")
            c2.write(f"**{p.get('name')}** ({p.get('category', 'N/A')})")
            if c3.button("✏️ Edit", key=f"lib_edit_{i}"): 
                st.session_state.edit_target = p
                st.rerun()
            if c4.button("🗑️ Del", key=f"lib_del_{i}"): 
                st.session_state.library.pop(i)
                save_prompts(st.session_state.library)
                st.rerun()

# === TAB 5: MODELS ===
with tab5:
    st.subheader("📊 Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Current Configuration:**")
        current_model = st.session_state.get('selected_text_model', 'Gemini')
        model_info = {
            "Image Generation": "Gemini (gemini-3-pro-image-preview)", 
            "Text/SEO Model": current_model,
            "Gemini Text": MODEL_TEXT_GEMINI,
            "Gemini Fallback": MODEL_TEXT_GEMINI_FALLBACK
        }
        if current_model in CLAUDE_MODELS:
            model_info["Claude Model ID"] = CLAUDE_MODELS[current_model]
        if current_model in OPENAI_MODELS:
            model_info["OpenAI Model ID"] = OPENAI_MODELS[current_model]
        st.json(model_info)
        
        st.write("**Available Models:**")
        st.write("🔹 **Gemini** - Google AI (Free tier available)")
        st.write("🔹 **Claude Sonnet 4.5** - Anthropic (Balanced)")
        st.write("🔹 **Claude Opus 4.6** - Anthropic (Highest quality)")
        st.write("🔹 **GPT-5.2** - OpenAI (Flagship model)")
        
    with col2:
        st.write("**API Status:**")
        if gemini_key: st.success("✅ Gemini API Key: Configured")
        else: st.error("❌ Gemini API Key: Missing")
        if claude_key: st.success("✅ Claude API Key: Configured")
        else: st.warning("⚠️ Claude API Key: Not Set")
        if openai_key: st.success("✅ OpenAI API Key: Configured")
        else: st.warning("⚠️ OpenAI API Key: Not Set")
    st.divider()
    if st.button("📡 Scan Gemini Models", key="models_scan_btn"):
        if not gemini_key: st.error("No Key")
        else:
            with st.spinner("Scanning..."):
                m = list_available_models(gemini_key)
                if m:
                    gem = [x for x in m if "gemini" in x['name']]
                    st.success(f"Found {len(gem)} models")
                    st.dataframe(pd.DataFrame(gem)[['name','version','displayName']], use_container_width=True)
                else: st.error("Failed")














