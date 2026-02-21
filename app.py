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
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
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

**ALSO BANNED — Repetitive Contrast Phrases:**
These phrases appear in AI-generated jewelry content across many pages,
creating a detectable site-wide pattern. Google's Feb 2026 Core Update
flags repeated phrasing across product pages as "scaled AI content."
NEVER use any of these (or close variants):
> "cheap hollow castings cannot match"
> "cheap hollow castings simply cannot"
> "not like cheap hollow pieces"
> "unlike cheap alternatives"
> "that cheap alternatives can't replicate"
> "mass-produced alternatives can't match"
> "the kind of weight/heft that [cheap thing] cannot"
> "no stamped, hollow, or plated pieces" (once per site is fine — not in every description)

**ALSO BANNED — ALL Negative-Then-Positive Comparison Structures:**

This is a UNIVERSAL RULE, not a list of specific phrases to avoid.
The AI keeps finding new ways to write the same pattern. So instead of
banning specific wordings, here is the PRINCIPLE:

NEVER write ANY sentence or sentence pair where:
  1. You describe a negative experience with OTHER/GENERIC products, THEN
  2. You say THIS product is better/different/the opposite.

This applies regardless of how creatively you phrase it. ALL of these
are the SAME banned pattern:
> "Most [X] online [negative]. This one [positive]."
> "Some [X] [negative]. This one doesn't."
> "[Negative experience with generic products]. This [product] is the opposite."
> "You know the feeling when [negative experience]... This [product] [positive]."
> "Ever bought a [product] that [disappointing thing]? This is [different]."
> "Tired of [negative]? [Product] [solves it]."
> "If you've been burned by [negative], [product] [positive]."
> "[Generic products] tend to [flaw]. Not this one."

The KEY WORDS that signal this banned pattern — if your opening sentence
contains ANY of these referring to OTHER products, rewrite it:
"Most", "Some", "Many", "Other", "Typical", "Average", "Cheap",
"Unlike", "Tired of", "Ever bought", "You know the feeling",
"the opposite", "this one doesn't", "this one won't", "not this one"

BAD (all different words, ALL the same banned pattern):
- "Most signet rings look good in photos and feel like nothing on your
   finger. This gold fleur de lis ring is the opposite."
- "Some rings disappear the second you put your hand back on the
   handlebars. This baroque skull ring doesn't."
- "You know the feeling when a ring looks killer online… then shows up
   feather-light and weirdly sharp on the edges. This skull ring is
   the opposite experience."
- "Ever bought a chain bracelet that turned your wrist green after a
   week? This one won't."
- "Tired of rings that lose their finish? This band holds up."
↑ 5 different creative approaches to the SAME structure. All banned.

**INSTEAD — Start with THIS product. Never mention other products or
generic negative experiences. Describe what this product IS, not what
other products AREN'T:**

GOOD (each stands on its own — no comparison needed):
- "34 grams of solid 316L steel, lost-wax cast with hand-finished edges."
- "The knurled texture grips your skin — you feel this ring when you
   move your hand."
- "Sized to sit just below the knuckle on most index fingers."
- ".925 sterling silver with a rhodium-plated finish that resists tarnish."
- "The skull's jaw is articulated — it moves when you flex your finger."

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

> **Why this matters:** Google's February 2026 Core Update (following the
> December 2025 Core Update) continues to crack down on thin, one-sided
> AI content. One-sided praise reads as marketing copy.
> A balanced view signals genuine experience and survives core updates.

### [RULE 5 — WRITE LIKE A REAL PERSON]

Write as if you are the shop owner who handles these products every day,
talks to customers face-to-face, and genuinely cares about helping them
pick the right piece. Your content should read like a knowledgeable friend
giving honest advice — not like marketing copy or a product data sheet.

**Voice & Authenticity:**
- Write in first-hand voice. You've held this product, weighed it in
  your palm, tried it on. Describe what YOU noticed — not what a spec
  sheet says.
  SPEC SHEET: "This ring weighs 28 grams and is made of 316L stainless steel."
  REAL PERSON (many ways to say this — use a DIFFERENT one each time):
  - "28 grams of 316L steel. You notice it the second you pick it up."
  - "Thick band, solid cast. My kitchen scale read 28 grams."
  - "It's the kind of ring that makes your hand feel different when you're gripping handlebars."
  - "The shank alone is 4mm — wider than most wedding bands."
  ↑ These all describe weight/heft but with DIFFERENT sentence structures.
  NEVER re-use the same opening structure across multiple products.

**⚠️ CRITICAL — NO TEMPLATE OPENINGS:**
  When writing multiple product descriptions, you MUST vary how you
  start each one. If you find yourself writing "[number] grams of [metal]
  [sensory verb] in your palm..." for more than ONE product — STOP.
  That's a template, not authentic writing. Each product should open
  with a completely different angle:
  - Product A opens with a visual detail
  - Product B opens with how it feels on the finger
  - Product C opens with a specific design element
  - Product D opens with a practical observation
  - Product E opens with who it's for
  The opening sentence is the MOST visible pattern across pages.
  Vary it the most.

- Use contractions naturally: don't, it's, you'll, there's, won't, that's.
  Occasionally start a sentence with "And" or "But."
- Use em dashes — like this — to break your own thought mid-sentence.
  Humans interrupt themselves. AI doesn't.

**Sensory language — put the product in their hands:**
Online shoppers can't touch, smell, or feel your product. Your words
must fill that gap. Include at least 2 specific sensory details:
- Touch: texture, weight in hand, temperature, how it feels on skin
  "The knurled band grips your finger — you feel it when you twist your hand."
- Sight: specific visual details only someone holding it would notice
  "Under a loupe, you can see the individual feather lines on the owl's wings."
- Sound: click, snap, rattle, silence
  "The clasp clicks — one clean snap. No wiggle."
- Smell: if relevant (leather, metal, polish)
  "Faint mineral scent from the polishing compound. That's how you know
  it's fresh from the bench."

**Honest trade-offs & practical advice:**
Real product experts always mention something the customer should know —
not just the good stuff. Include at least 1:
- A sizing quirk: "Runs about half a size small — order up."
- A care tip: "Sterling tarnishes. Keep it in the pouch when you're not wearing it."
- A minor limitation: "The detail on the back is less defined than the front."
- A practical warning: "This one's heavy. If you're not used to chunky rings,
  start with something under 20 grams."
These aren't negatives — they BUILD trust. A description with zero
downsides reads like a brochure. A description with one honest caveat
reads like advice from someone who knows.

**Personal opinion — have a voice:**
Humans have opinions. Include 1 mildly opinionated statement per description:
  GOOD: "Personally, I'd size up half a step on this one."
  GOOD: "The matte finish looks better than the polished version."
  GOOD: "Best worn on the index or middle finger — it's too wide for the pinky."
  BAD: "This ring is available in multiple sizes to suit your preferences."
  ↑ This could be written about ANY ring. It says nothing. Avoid it.

**Natural sentence rhythm:**
Read your writing out loud. Does it sound like someone talking?
- Mix sentence lengths naturally. A long descriptive sentence followed
  by something short and blunt. "Heavy. Real heavy."
- Use sentence fragments when they feel right. Not every sentence
  needs a subject and a verb.
- Don't write every sentence in the same structure. If three sentences
  in a row all start with "The [noun] [verb]...", rewrite one.

**Storytelling over selling:**
Don't describe features — create a moment the reader can picture:
  SELLING: "This ring features an articulated jaw mechanism for added realism."
  STORYTELLING: "The skull's jaw swings open when you flex your knuckle.
  First time you notice it happening at a bar, you'll get a comment."
Let the customer imagine USING the product in their life, not just
reading about its specifications.

**Things that make writing feel artificial (avoid these):**
- Perfectly balanced paragraphs where every one is the same length.
- Repeating the same sentence structure across a paragraph.
- Summarizing what you just wrote. Humans don't recap their own thoughts.
- Using "whether you're [A] or [B]" more than once per description.
- Transition words in clusters: "Furthermore", "Moreover", "Additionally"
  — use "And", "Also", "Plus", or just start the next thought directly.
- Generic phrases that apply to any product: "perfect for any occasion",
  "makes a great gift", "sure to impress".

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

### [RULE 8 — TRANSACTIONAL SEARCH POSITIONING + AI MODE SIGNALS]

The people landing on this product page are searching with buying intent.
They're typing things like "best [category] for [use case]" or
"[product] vs [competitor category]."

Include ONE natural sentence that positions this product against its
category — not a specific competitor brand name.

Examples:
- "Most budget wireless earbuds sacrifice bass. This one doesn't."
- "Where other compact blenders struggle with ice, this handles
  frozen fruit without stalling."

Also include 1-2 "best for" micro-signals that AI systems can extract
for recommendation queries. These are short phrases embedded in natural
sentences that tell AI who this product is ideal for:
- "Best for daily wear" / "Best for riders who..." / "Best for men with larger hands"
Google AI Mode and AI Overviews use these "best for" signals to match
products with user queries like "which ring is best for everyday use?"

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

### [RULE 11 — AI CONTENT DETECTION & GEO COMPLIANCE (2026)]

Google's February 2026 Core Update (confirmed Feb 1, 2026) doubled down
on the December 2025 Core Update's crackdown on thin AI content.
Key enforcement in 2026:
- Thin, low-value AI-generated content is being algorithmically demoted.
- Sites demonstrating topical authority are rewarded — product pages must
  show deep product knowledge, not surface-level specs.
- Ecommerce pages saw 52% impact in Dec 2025; Feb 2026 further targets
  scaled AI product descriptions with repetitive patterns.

When writing multiple product descriptions (batch mode), you MUST:

1. **Vary sentence structures** — never use the same opening pattern,
   transition phrases, or paragraph structure across products.
2. **Use specific data over subjective adjectives** — LLMs are 30-40% more
   likely to cite sources with specific numbers than vague marketing claims.
   BAD: "This amazing ring is incredibly durable and beautifully crafted."
   GOOD: "28 grams of solid 316L steel — 2mm thicker than most mass-produced rings."
3. **Each product description must be unique enough** that it could NOT be
   swapped with another product and still make sense. Every sentence must
   contain details specific to THIS product.
4. **Information Gain** — every paragraph must add factual information
   the reader didn't have before. Zero filler sentences allowed.
5. **Topical authority signal** — include at least 1 sentence that shows
   knowledge BEYOND what's on the product label. A material comparison,
   a care tip, a sizing insight, or a construction detail that proves
   the writer actually understands the product category.

---

## OUTPUT STRUCTURE:

### [Hook — no H2 tag]

2-3 sentences. Open with a pain point or a scene.
Make the reader feel recognized before you sell anything.

> The product name + category (Main Keyword) must appear naturally
> within this section (Rule 7 + Rule 9).
> **GEO CRITICAL (2026):** The first 100 words of the product page are
> what AI systems (Google AI Overviews, AI Mode, ChatGPT, Perplexity)
> use to decide whether to cite or recommend your product.
> Within the Hook, naturally include: what the product IS (1 clear sentence),
> who it's for, and ONE key differentiator.
> This "definition-first" approach helps AI systems extract and recommend
> your product accurately in conversational shopping experiences.

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

**CRITICAL — ANTI-PATTERN RULE:**
Every product description MUST use DIFFERENT sentence structures in this section.
NEVER reuse these overused AI patterns:
- ❌ "If you like [X] but want [Y]..." (banned opening)
- ❌ "A lot of people who grab this end up..." (banned opening)
- ❌ "And if you want to browse more..." (banned opening)
- ❌ "If you're stacking/pairing rings..." (banned opening)
- ❌ Three paragraphs that all follow [context → product link → comparison] structure

Instead, vary your approach. Some alternatives:
- Start with the COMPLEMENTARY product first: "The [linked product] sits well
  next to this one — [reason]."
- Use a direct observation: "Sterling silver tarnishes. Brass doesn't.
  That's what makes [linked product] a good backup ring for daily wear."
- Ask a question: "Need something lower-profile for the office?
  [linked product] has the same build quality without the skull."
- Reference a use case: "For riding, most guys pair a heavy ring
  with [linked product] — keeps the look consistent without doubling the weight."
- State a fact: "This is one of five pieces we make in this brass alloy.
  [linked product] and [linked product] use the same material."

**CRITICAL — INTERNAL LINKING FROM REAL STORE DATA:**

You will receive REAL STORE CATALOG DATA at the end of this prompt
containing actual collections and products that exist on the store.
Each product entry includes [product_type] and {tags} to help you
find genuinely related items.
You MUST ONLY link to paths that appear in that catalog data.
NEVER invent or guess URLs — every href must come from the provided list.

**STEP-BY-STEP: How to choose related items (THIS IS THE MOST IMPORTANT PART):**

Before writing a single word in this section, you MUST complete this analysis:

1. **Identify THIS product's key attributes** from the description you just wrote:
   - Material (e.g., brass, sterling silver, stainless steel, gold)
   - Style/Theme (e.g., skull, gothic, Celtic, biker, Christian, tribal)
   - Product type (e.g., ring, pendant, bracelet, chain, necklace)
   - Price tier (budget, mid-range, premium)

2. **Scan the CATALOG DATA** and find items that match on at least 2 of these criteria:
   - **BEST match:** Same material + same style (e.g., another brass skull piece)
   - **GOOD match:** Same style + different product type (e.g., skull ring → skull pendant)
   - **GOOD match:** Same material + different style (e.g., brass skull → brass Celtic)
   - **OK match:** Same category collection (e.g., link to /collections/brass-rings)
   - **BAD match:** Random popular product with no connection to THIS product

   Use the [product_type] and {tags} in the catalog data to find matches.
   Search for THIS product's material and style keywords in the catalog titles and tags.

3. **Link MIX rule:** Include at least 1 specific product AND at least 1 collection.
   The product link shows a specific recommendation; the collection link
   gives the reader a browsing path.

4. **Now write** the 2-3 sentences using the matched items you found.

**EXAMPLE of the matching process (do NOT output this — just follow it):**
If you're writing about a "Brass Skull Ring with Red Eyes":
- Material = brass → scan catalog for "brass" in titles/tags
- Style = skull → scan catalog for "skull" in titles/tags
- Type = ring → complement with pendant, bracelet, or chain
- CHOOSE: a brass skull pendant (same material + style) ✅
- CHOOSE: /collections/skull-rings or /collections/brass-jewelry ✅
- DO NOT choose: a random sterling silver Celtic bracelet ❌

**How to write this section:**
1. First, complete the matching analysis above to find 2-3 genuinely
   related items from the CATALOG DATA.
2. Write each recommendation sentence with a DIFFERENT structure —
   no two sentences should open the same way or follow the same pattern.
3. Wrap the most natural phrase in each sentence with a link to the
   real catalog path. The linked phrase should flow seamlessly.

**Link format:**
<a href="[exact path from catalog]" title="[Product or Collection Title from catalog]" style="color:#1a3a6b; font-weight:600; text-decoration:underline;">[natural phrase from your sentence]</a>

**GOOD — each sentence uses a DIFFERENT structure:**
<p>Sterling silver darkens over time. Brass stays warm. <a href="/products/skull-cross-sterling-silver-wallet-chain" title="Skull Cross Sterling Silver Wallet Chain" style="color:#1a3a6b; font-weight:600; text-decoration:underline;">The skull cross wallet chain</a> is one of the few pieces that mixes both — and it pairs well with this ring.</p>
<p>Need something for the other hand? <a href="/collections/bracelets" title="Bracelets" style="color:#1a3a6b; font-weight:600; text-decoration:underline;">The bracelet section</a> has a dozen options in the same weight class.</p>

**BAD — repetitive AI pattern (all follow same structure):**
<p>If you like this ring but want something for your wrist, check out our bracelets.</p>
<p>A lot of people who grab this end up coming back for the wallet chain.</p>
<p>And if you want to browse more, our full rings collection has tons of options.</p>

**RULES:**
1. ONLY use paths from the provided REAL STORE CATALOG DATA.
   If no catalog data is provided, skip the links and write plain text recommendations.
2. Use PATH URLs only — start with / (e.g., /collections/... or /products/...).
3. Link 2-3 items total — mix of collections and products when possible.
4. The linked text must be a natural part of the sentence, NOT a standalone keyword.
5. The title attribute should match the real product/collection title from the catalog.
6. Choose DIFFERENT catalog items for each product — do not always link to
   the same 2-3 "popular" items. Pick items relevant to THIS product's material,
   style, and category. If this is a brass ring, link to other brass pieces.
   If this is a gothic pendant, link to gothic-themed items.

> This section creates internal links to related product/collection pages,
> which strengthens your site's crawlability and topical authority.
> Every link MUST point to a real, existing page from the store catalog.

---

### ## META (for CMS use — do not publish on page)

**Product Title (H1) — RULES:**
- Main Keyword MUST appear within the H1, ideally near the front.
- First letter MUST be capitalized. Use Title Case (capitalize major words).
- Clear, descriptive, keyword-rich product name.
- Keep under 70 characters for display consistency.
- Must NOT be identical to the Meta Title (they serve different purposes:
  H1 is for the page visitor, Meta Title is for the SERP).
- Include: [Product Name] + [Key Attribute] (material, style, or category)
- The H1 should tell a shopper exactly what this product IS at a glance.
  GOOD: "Skull Flame Ring — Heavy 316L Stainless Steel for Bikers"
  GOOD: "Gothic Cross Sterling Silver Pendant with Black Onyx"
  BAD: "Amazing Skull Ring Best Quality 2026" (vague, dated, keyword-stuffed)
  BAD: "Product #4521" (no description)
  BAD: "The Most Incredible Ring You've Ever Seen" (no product info)
- For jewelry/accessories, always include: [design/motif] + [material] + [product type]
  This helps Google Product schema AND AI systems categorize the item correctly.

**Meta Title — RULES:**
- MUST be under 60 characters (Google truncates at ~60 / ~580px pixel width).
  This is a HARD LIMIT — never exceed 60 characters.
- Main Keyword at the FRONT — Google gives more weight to leading words.
- First letter MUST be capitalized. Use Title Case for major words.
- Format: [Main Keyword — One Key Benefit] | Bikerringshop
- The benefit should be a concrete attribute (material, weight, style),
  not a generic claim ("best", "top quality", "amazing").
- Brand name "Bikerringshop" at the end after | — BUT ONLY if the total
  stays under 60 characters. If adding "| Bikerringshop" would push
  the title over 60 chars, DROP the brand entirely. The keyword and
  differentiator are more valuable for SEO than the brand name.
  Google already shows the site domain in SERP results.
  GOOD: "Skull Flame Ring — Heavy 316L Steel | Bikerringshop" (51 chars, brand fits)
  GOOD: "Gothic Cross Sterling Silver Pendant with Black Onyx" (52 chars, no brand — would be 70+ with brand)
  BAD: "Skull Flame Ring — Heavy 316L Stainless Steel for Bikers | Bikerringshop" (72 chars — OVER LIMIT)
  BAD: "Bikerringshop — Best Skull Rings Online" (brand first wastes prime space)
  BAD: "Skull Ring, Biker Ring, Gothic Ring, Silver Ring" (keyword list)

**Meta Description — RULES:**
- Under 155 characters. Aim for 140-155 chars.
- Main Keyword within the first 80 characters (visible on mobile SERP snippets).
- First letter MUST be capitalized (it's a sentence displayed in search results).
- Must COMPLEMENT the meta title, NOT repeat it word-for-word.
- Use the description to expand on the title's promise with specifics.
- Include: Main Keyword + 1 specific detail (weight, material, sensory detail).
- Optionally include the honest caveat — this stands out in SERPs and
  increases CTR because it reads as genuine, not promotional.
- No "shop now" or "buy today" CTAs — Google's 2026 algorithm devalues
  aggressive CTAs in meta descriptions. They waste characters and don't improve CTR.
- Use a DIFFERENT long-tail keyword variation than the one in the meta title.
  GOOD: "28 grams of solid 316L steel with hand-finished flame detail. Runs slightly large — check the size guide before ordering." (118 chars)
  BAD: "Shop our amazing skull ring! Best quality guaranteed. Buy now and get free shipping!" (CTA-heavy, generic)
  BAD: "Skull ring biker ring gothic ring stainless steel ring for men." (keyword-stuffed list)

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

Write like a knowledgeable friend who sells jewelry — someone who'll
tell you "this one's worth it" AND "skip a size up, it runs tight."
Confident, direct, honest. Not salesy. Not corporate.
If you wouldn't say it out loud to a customer standing at your counter,
don't write it.

## TARGET LENGTH:

480-600 words (body content, excluding table and meta section).

## READING LEVEL:

Grade 8-10. Clear, not dumbed down.

---

## CONTEXT FOR THE AI (do not output this section):

**Google Algorithm Context (2025-2026):**

- **February 2026 Core Update (LATEST — rolling out now):**
  Google confirmed a broad core update on February 1, 2026, targeting:
  (1) Thin, low-value AI-generated content — algorithmically demoted.
  (2) Topical authority rewarded — sites that cover a topic deeply
  and consistently rank higher. Product pages must show category expertise.
  (3) Crawl efficiency matters — Google noted 75% of crawling waste
  comes from faceted navigation and filtered URLs.
- **December 2025 Core Update** heavily rewarded e-commerce and retail brands
  while penalizing thin content. Ecommerce saw 52% impact rate. Thin category
  pages and product pages with generic manufacturer descriptions were hardest hit.
- E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)
  is the primary quality framework. "Experience" — first-hand product
  use — is the differentiator for product pages. E-E-A-T now applies
  to ALL verticals, not just YMYL content.
- Google improved AI content detection across both updates. Mass-produced
  unedited AI content lost 60-95% traffic. Content must show human insight,
  specific details, and balanced views. Repetitive patterns across products
  are flagged as scaled AI content.
- Product structured data (JSON-LD) remains critical for rich snippets.
  Core schema types — Product, Review, Breadcrumb, Organization —
  are confirmed as long-term priorities by Google.
- User engagement metrics (time on page, scroll depth, bounce rate)
  are stronger ranking signals than ever. Sites with LCP > 3s saw 23%
  more traffic loss.
- Content freshness signals matter — but product pages should use
  subtle time-anchors, not hard dates that expire.
- Zero-click searches (~60% of Google queries) mean product pages
  must provide enough structured data (via schema) to appear in
  rich results even when users don't click through.
- **GEO (Generative Engine Optimization) — 2026:**
  Google AI Overviews now appear in 30-45% of searches and reach 2 billion
  monthly users across 200+ countries. AI Mode allows conversation-to-checkout
  commerce. Product pages must be written so AI can extract: what the product IS,
  who it's for, key specs, and why it's different — all within the first 100 words.
  Structure and clarity beat storytelling for AI visibility.
  "Best for" signals (e.g., "best for daily wear", "best for riders who...")
  are specifically used by AI Mode to match products with comparison queries.
- **Agentic Commerce — 2026:**
  Google introduced Universal Commerce Protocol (UCP). AI agents now
  discover, evaluate, and recommend products based on structured data.
  Specific, measurable product attributes (material, weight, dimensions)
  are critical — vague adjectives are ignored by AI recommendation systems.
- **Visual Search & Google Lens — 2026:**
  Gen Z users now start 1 in 10 searches with Google Lens, with ~20%
  carrying commercial intent. Image SEO (file names + alt tags) is
  more important than ever — well-optimized images appear in visual
  search results and AI-powered shopping experiences.
- Google's meta title rewrite rate is increasing. Titles measured by
  pixel width (~580px), not character count. Meta descriptions are
  rewritten ~62% of the time but still matter for high-intent queries.

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
    { "file_name": "descriptive-name.jpg", "alt_tag": "Unique description of what this specific image shows" }
  ]
}

### URL SLUG RULES (for the url_slug field):
### Updated for Google's 2026 URL Best Practices

Google uses words in the URL as a lightweight ranking signal when first discovering
a page. A well-structured slug also improves CTR from SERPs and helps AI systems
(Google AI Mode, Perplexity, ChatGPT) understand page topic before reading content.

**Structure Rules:**
- Lowercase only — never use uppercase (case-sensitive servers treat them differently,
  causing duplicate content issues).
- Hyphens (-) only — never underscores (_). Google treats hyphens as word separators
  but underscores as word joiners.
- No special characters — no ?, %, #, &, @, or non-ASCII characters.
- No trailing slashes or file extensions.
- No stop words (a, an, the, of, for, and, in, on, with, to, is, by) unless
  removing them makes the slug confusing or unreadable.

**Content Rules:**
- Lead with the Main Keyword — place it at the beginning of the slug.
  Google weighs words at the start of the URL more heavily.
- Include product name + 1 key attribute (material, style, or category)
  that differentiates this product from similar ones.
- Keep it SHORT: 3-6 words (ideally under 60 characters).
  Google prefers shorter URLs — if two pages have identical metrics,
  the shorter URL wins as a tiebreaker.
- Must be UNIQUE — never produce a slug that could match another product.
  Include a differentiating attribute if the product name alone is too generic.
- Must match the page content — the slug should accurately describe what's
  on the page. Misleading slugs hurt bounce rate and trust.
- Never include dates, years, or version numbers — these expire and
  prevent long-term slug reuse.
- Never include prices or quantities — these change over time.

**Ecommerce Product Slug Format:**
[main-keyword]-[key-attribute]-[differentiator]

GOOD examples:
- skull-flame-stainless-steel-ring
- gold-plated-bishop-cross-ring
- heavy-chain-sterling-silver-bracelet
- celtic-knot-tungsten-wedding-band

BAD examples:
- product-98876451 (no description)
- the-amazing-best-skull-ring-for-men-2026 (stop words, date, too long)
- SKULL_Ring (uppercase, underscore)
- skull-ring (too generic — could match dozens of products)
- ring (no context at all)

### IMAGE SEO RULES (for the image_seo array):
### Updated for Google's 2026 Image SEO + Visual Search Best Practices

Each image MUST have a **unique** file_name and alt_tag.
Google's algorithm uses alt text alongside computer vision to understand images.
Repetitive patterns in AI-generated alt text can be flagged as duplicate content.

**2026 Visual Search Context:**
Google Lens now handles 1 in 10 searches by Gen Z, with ~20% carrying
commercial intent. Well-optimized product images (file names + alt tags)
appear in Google Lens visual search, Google Shopping, and AI-powered
"shop similar" experiences. Image SEO directly drives discovery and sales.

**File Name Rules:**
- Lowercase, hyphens only, end with .jpg
- DO NOT repeat the full product name in every file name.
  Use it once (first image), then lead with the VISUAL FOCUS of each image.
- Structure: [visual-focus]-[material-or-detail]-[angle-or-context].jpg
- Include product attributes (material, color, variant) in the filename
  to help search engines map images to catalog pages.
- Each file name should describe what makes THAT specific image different.
- Keep filenames descriptive but concise — 3-7 hyphenated words.

GOOD example (8 images of a bishop ring):
1. christian-crosier-bishop-ring-amethyst-top-view.jpg
2. gold-plated-cross-cutout-band-detail.jpg
3. amethyst-gemstone-crosier-setting-closeup.jpg
4. ribbed-gold-band-side-profile.jpg
5. openwork-cross-pattern-left-angle.jpg
6. bishop-ring-worn-on-hand-lifestyle.jpg
7. sterling-silver-base-interior-hallmark.jpg
8. crosier-bishop-ring-full-set-flat-lay.jpg

BAD example (all start the same = keyword stuffing, Google may see as spam):
1. christian-crosier-bishop-ring-angled-right-view.jpg
2. christian-crosier-bishop-ring-side-profile-left.jpg
3. christian-crosier-bishop-ring-side-profile-right.jpg
4. christian-crosier-bishop-ring-angled-left-view.jpg

**Alt Tag Rules:**
- Each alt tag must describe what is VISUALLY shown in that specific image.
- Keep under 125 characters (screen reader best practice per W3C/Google).
- DO NOT start every alt tag with the same product name prefix.
  Vary the opening: lead with the visual focus, the material detail,
  the angle, or the feature being highlighted.
- Include relevant keywords NATURALLY but DIFFERENTLY across images.
  Google explicitly warns: keyword stuffing in alt attributes results
  in a negative user experience and may cause the site to be seen as spam.
- Alt text must align with the surrounding page content for relevance signals.
- Never use "image of" or "picture of" — describe the content directly.
- Think of each alt tag as a unique sentence on the page:
  you wouldn't write the same paragraph 8 times.

GOOD alt tags (varied openings, unique descriptions, under 125 chars):
1. "Top view of the Christian crosier bishop ring with purple amethyst center stone"
2. "Gold-plated band detail showing openwork cross cutouts and ribbed texture"
3. "Close-up of the amethyst gemstone set in a crosier-shaped sterling silver bezel"
4. "Side profile highlighting the ribbed gold plating and layered band construction"

BAD alt tags (repetitive prefix = pattern Google flags as low-quality AI content):
1. "Christian crosier bishop ring angled right view showing amethyst..."
2. "Christian crosier bishop ring side profile left showing cross..."
3. "Christian crosier bishop ring side profile right displaying gold..."
"""

SEO_PROMPT_NAME_SLUG = """
You are an SEO expert with 10-15 years of experience. 
Analyze the provided product images and description. Generate:
1. An attractive, SEO-optimized Product Name.
2. A suitable, clean URL Slug (using hyphens).

**Product Name Rules:**
- Clear, descriptive, keyword-rich product name.
- Include the product type/category and 1-2 key attributes (material, style).
- Keep under 70 characters.

**URL Slug Rules (2026 SEO Best Practices):**
- Lowercase only, hyphens only (no underscores, no special characters).
- Lead with the main keyword at the beginning.
- Include product name + 1 key differentiating attribute (material, style, or category).
- Keep SHORT: 3-6 words, under 60 characters.
- Remove stop words (a, an, the, of, for, and, in, on, with) unless needed for clarity.
- Must be UNIQUE and specific enough to not match other products.
- Never include dates, prices, or version numbers.
- Format: [main-keyword]-[key-attribute]-[differentiator]

GOOD: skull-flame-stainless-steel-ring, gold-plated-bishop-cross-ring
BAD: product-12345, the-best-skull-ring-2026, SKULL_Ring, ring

User Input Description: "{user_desc}"

IMPORTANT: Return RAW JSON format ONLY (no markdown backticks).
Structure: {"product_name": "...", "url_slug": "..."}
"""

# === COLLECTION PAGE WRITER PROMPT ===
SEO_COLLECTION_WRITER_PROMPT = """
## ROLE:
You are a Senior E-commerce SEO Copywriter who specializes in writing
category and collection page content. Collection pages serve a dual purpose:
helping shoppers navigate products quickly AND ranking for broader,
high-volume category keywords. You write concise, value-driven copy
that strengthens SEO without cluttering the shopping experience.

## INPUT:
1. **Main Keyword:** {main_keyword}
2. **Collection URL:** {collection_url}

## CRITICAL RULES:

### [RULE 1 — COLLECTION PAGE ≠ BLOG POST ≠ PRODUCT PAGE]
- Be **short and focused** — shoppers browse products, not read articles.
  Keep total paragraph content to **150-300 words**.
- **Not compete with product pages** — target broader category keywords.
  Category pages target broad intent (e.g., "men's skull rings"),
  while product pages answer "is this the right item for me?"
- **Not compete with blog posts** — target transactional intent.
- **Support product discovery** — guide shoppers, don't distract.

### [RULE 2 — BAN LIST]
NEVER use: Delve, Elevate, Comprehensive, Cutting-edge, Unleash, Ultimate,
Testament, Precision-engineered, Game-changer, Furthermore, Moreover,
In conclusion, Seamlessly, Robust, Leverage, In today's world, Look no further,
It's worth noting, Revolutionize, State-of-the-art, Best-in-class,
Unparalleled, Groundbreaking, Next-level, Wide range of, Wide selection of,
Explore our collection, Browse our collection, Discover our collection.

**ALSO BANNED — Repetitive Contrast Phrases (site-wide pattern risk):**
These phrases appear in AI content across many pages. Using them on multiple
collection pages triggers Google's "scaled AI content" detection.
NEVER use: "cheap hollow castings cannot match", "unlike cheap alternatives",
"mass-produced alternatives can't match", "not like cheap hollow pieces",
"the kind of weight that [cheap thing] cannot", "no stamped, hollow, or plated".

**ALSO BANNED — ALL Negative-Then-Positive Comparison Structures:**
NEVER write ANY sentence where you describe a negative experience with
OTHER/GENERIC products and then say THIS collection is better/different.
This applies regardless of phrasing — "Most [X] online...", "You know
the feeling when...", "Tired of...", "Ever bought a...", "Unlike most..."
are ALL the same banned pattern. Start with what THIS collection IS,
not what other products AREN'T.

### [RULE 3 — KEYWORD ANALYSIS & INTEGRATION]
Using the Main Keyword, determine:
- **Main Keyword:** Use exactly as provided.
- **Secondary Keywords (2-3):** Variations and related category terms.
- **Long-tail Keywords (1-2):** More specific category searches.

Integration:
- Main Keyword in: H1, meta title, first paragraph (within first 2 sentences).
- Secondary Keywords: 1 time each in paragraphs.
- Long-tail Keywords: woven into paragraph 2.
All keywords must read naturally. Forced keywords are obvious — rewrite if stuffed.
Google's 2026 algorithm prioritizes meaning, context, and conversational relevance
over exact-match keywords. Write for intent, not for keyword density.

### [RULE 4 — WRITE FOR TRANSACTIONAL INTENT + DECISION SUPPORT]
People landing on collection pages are ready to shop, not research.

The copy should:
- ✅ Help shoppers understand what this collection offers.
- ✅ Highlight what makes these products different from generic alternatives.
- ✅ Mention key attributes (material, style, durability).
- ✅ Build just enough trust to keep them browsing.
- ✅ Include "best for" signals — who is this collection ideal for?
  (e.g., "best for riders who want heavy rings that survive the road")
  This helps both shoppers AND AI systems recommend your products.
- ❌ NOT educate at length. ❌ NOT tell brand story. ❌ NOT list products by name.
- ❌ NOT produce thin/generic content that could apply to any store.
  Google's spam policies explicitly target thin, mass-produced category pages.

### [RULE 5 — E-E-A-T FOR COLLECTION PAGES]
- 1 expertise signal: Show product category knowledge with a specific fact.
  Example: "316L stainless steel — the same grade used in marine
  and surgical equipment — so these won't turn your finger green."
- 1 audience understanding signal: Show you know who's buying.
  Example: "Built for riders who want rings that survive the road,
  not sit in a jewelry box."

### [RULE 6 — CONTENT STRUCTURE + GEO/AI CITATION OPTIMIZATION]
Write 2-3 short paragraphs:
- **Paragraph 1 (hook + definition):** Start with a clear, concise definition
  of what this collection IS and who it's for (1 sentence). This MUST be
  specific enough that Google AI Overview or AI Mode can extract and cite it
  as a direct answer. Then context, Main Keyword, what makes collection
  different. 2-3 sentences total.

  **IMPORTANT — VARY YOUR OPENING PATTERN:**
  The first sentence must define the collection, but DO NOT always use the
  same "[Collection name] are [definition]" structure. Google's Feb 2026
  Core Update flags repetitive sentence patterns across category pages as
  "scaled AI content." Rotate between these opening formats:

  FORMAT A (definition): "[Collection name] are [what they are]."
  FORMAT B (audience-first): "Built for [audience], [collection name] [key trait]."
  FORMAT C (material-lead): "Cast in [material], these [collection type] [purpose]."
  FORMAT D (statement): "Every [collection item] in this collection [unique fact]."
  FORMAT E (direct): "If you [need/want], [collection name] [delivers how]."
  FORMAT F (spec-highlight): "[Collection type] need [key quality] to [purpose] — these [specific spec]."
  FORMAT G (weight/spec-lead): "At [weight range] per piece, [collection name] [feel/impression]."
  FORMAT H (question): "Looking for [specific need]? [Collection name] [answer]."
  FORMAT I (craftsmanship): "Handcast in [location/method], each [item type] in this collection [detail]."
  FORMAT J (use-case): "Whether you're [activity 1] or [activity 2], [collection name] [benefit]."

  EXAMPLES (each uses a different format — this is what we want across collections):
  A: "Skull biker rings are heavy-duty statement rings built for riders
     who treat jewelry like gear, not decoration."
  B: "Built for veterans and active-duty service members, military rings
     carry real insignia detail in solid .925 sterling silver."
  C: "Cast in solid .925 sterling silver, these owl rings feature hand-carved
     feather detail that holds up to daily wear."
  D: "Every cross pendant in this collection is handcast in solid .925
     sterling silver — averaging 18 to 30 grams depending on design."
  E: "If you ride and your jewelry doesn't survive the road, these biker
     bracelets are built from the same 316L steel as your exhaust."
  F: "Celtic rings need weight to show off their knotwork — these run 20
     to 35 grams in solid .925 sterling silver, sized from 7 to 14."
  G: "At 25 to 45 grams per ring, these dragon rings have the kind of
     weight you notice the second you slide one on."
  H: "Looking for a ring that says something without saying a word?
     Freemason rings carry centuries of symbolism in solid sterling silver."
  I: "Handcast using lost-wax method, each gothic ring in this collection
     carries detail down to 0.5mm line work — visible in every skull and vine."
  J: "Whether you're commuting daily or riding cross-country, these chain
     bracelets handle sweat, rain, and road grime without tarnishing."

  BAD (every collection starts the same — triggers AI content pattern detection):
  "Cross rings are bold, faith-inspired bands..."
  "Skull rings are heavy-duty statement rings..."
  "Owl rings are handcrafted sterling silver..."
  "Military rings are symbol-heavy bands..."
  ↑ This pattern across 50 collections = flagged as scaled AI output.
- **Paragraph 2 (detail + "best for"):** Who it's for, key attributes,
  secondary/long-tail keywords. Include 1-2 specific "best for" statements
  that AI can extract (e.g., "best for daily wear," "best for riders who
  prefer silver over steel"). Include a concrete product fact — material,
  weight range, sizing, or construction method. 2-3 sentences.
- **Paragraph 3 (navigation):** Suggest related collections with internal links.
  Frame as decision support: "If you want [alternative], check [collection]."
  1-2 sentences.
Total: **150-300 words**.

### [RULE 7 — WRITE LIKE A REAL PERSON]
Your collection description should read like the shop owner wrote it
while leaning on the counter talking to a customer — not like a
marketing team drafted it in a meeting room.

**Voice:** Confident, direct, knowledgeable. Use contractions: don't,
it's, you'll, that's, won't. Start a sentence with "And" or "But"
when it feels natural.

**Sensory & first-hand details:**
Include at least 1 observation that only someone who handles these
products daily would make:
  GOOD: "The heavier pieces in this collection — anything over 30 grams —
  tend to fit snugger than you'd expect."
  GOOD: "Most of these run between 15 and 40 grams. You'll notice the
  weight the moment you pick one up."
  BAD: "This collection offers a wide variety of styles to suit every taste."
  ↑ Generic. Could be written about any collection anywhere.

**Honest & practical:**
Mention 1 practical detail that shows real product knowledge: a sizing
quirk, care tip, weight range, or material characteristic.

**Natural rhythm:**
Mix sentence lengths. A long descriptive sentence, then something short.
Don't write every sentence the same length or structure. Read it out
loud — if it sounds flat, rewrite it.

**Avoid artificial writing patterns:**
- Don't summarize what you just said at the end of a paragraph.
- Don't use "Furthermore", "Moreover", "Additionally" — use "And", "Plus",
  or just start the next thought.
- Don't write generic marketing phrases that could apply to any collection.

Write for the shopper standing in front of you, not for a search engine.

### [RULE 8 — 2026 ALGORITHM COMPLIANCE (Feb 2026 Core Update)]

**A. February 2026 Core Update (rolling out now):**
Google confirmed a broad core update on February 1, 2026, targeting:
1. Thin, low-value AI-generated content — every sentence must contain a
   concrete detail specific to THIS collection. No filler.
2. Topical authority — collection pages must demonstrate deep category knowledge,
   not surface-level descriptions. Include at least 1 material specification,
   construction detail, or measurable product attribute.
3. Crawl efficiency — faceted navigation and filtered URLs waste crawl budget.
   Collection content must be valuable enough to justify being crawled.

**B. December 2025 Core Update aftermath:**
1. **NO THIN CATEGORY PAGES:** Category pages with "limited or repetitive content"
   were among the hardest hit (52% of ecommerce pages impacted).
   Every collection MUST have unique, specific copy
   that could NOT be copy-pasted to another collection and still make sense.
2. **AI CONTENT DETECTION:** Google improved detection of "low-value patterns
   commonly associated with scaled, unreviewed AI output." Avoid:
   - Generic sentences that could apply to any jewelry store
   - Repetitive sentence structures across collections
   - Filler phrases that add word count but zero information
3. **INTENT PURITY:** Pages that mix informational + transactional intent
   lost rankings. Collection pages must be PURELY transactional.

**C. Google AI Mode & AI Overview Optimization (2026):**
AI Mode now surfaces category pages that help shoppers CHOOSE, not just browse.
1. **Definition-first structure:** Start paragraph 1 with a clear
   "[Category] are [definition]" sentence. AI systems extract this as the
   direct answer for "what are [category]?" queries.
2. **"Best for" signals:** Include specific "best for [use case]" phrases.
   AI Mode uses these to recommend products to users asking comparison
   questions like "which type of ring is best for daily wear?"
3. **Cite-worthy facts:** Include 1-2 specific, factual claims that AI
   can confidently cite (material grades, weight ranges, sizing info).
   Vague claims ("high quality") are never cited. Specific claims
   ("cast in 316L surgical-grade stainless steel, 28-42g per ring") are.
4. **Entity clarity:** Make it clear what category of products this is,
   who makes them, and who they're for. AI systems need entity context
   to confidently recommend and cite your page.

### [RULE 9 — INTERNAL LINKING]
Include 2-3 natural mentions of related collections within the content.
Frame as helpful navigation, not a link dump.

**CRITICAL — INTERNAL LINKS FROM REAL STORE DATA:**
You will receive REAL STORE CATALOG DATA at the end of this prompt.
You MUST ONLY link to paths that appear in that catalog data.
NEVER invent or guess URLs.

Link format:
<a href="[exact path from catalog]" title="[Collection Title from catalog]" style="color:#1a3a6b; font-weight:600; text-decoration:underline;">[natural phrase]</a>

**GOOD example:**
<p>If you're after something with less edge, our <a href="/collections/silver-rings" title="Silver Rings" style="color:#1a3a6b; font-weight:600; text-decoration:underline;">sterling silver rings</a> keep the same build quality in a cleaner style.</p>

**BAD example:**
<p>Check out our <a href="/collections/skull-rings">skull rings</a> for skull rings.</p>

RULES:
1. ONLY use paths from the REAL STORE CATALOG DATA provided.
2. Use PATH URLs only — start with / (e.g., /collections/...).
3. The title attribute must match the real collection title.
4. Anchor text must be a natural part of the sentence.
5. If no catalog data provided, write plain text without links.

## OUTPUT:

Return ONLY raw JSON (no markdown backticks) with this structure:

{
  "collection_title": "Collection H1 title — see H1 RULES below",
  "collection_description_html": "<div class='collection-description'><p>Paragraph 1...</p><p>Paragraph 2...</p><p>Paragraph 3 with internal links...</p></div>",
  "meta_title": "See META TITLE RULES below (under 60 chars)",
  "meta_description": "See META DESCRIPTION RULES below (under 155 chars)",
  "keyword_analysis": "Main: ... | Secondary: ..., ... | Long-tail: ..., ..."
}

### H1 (collection_title) RULES:
- Main Keyword MUST appear at the START of the H1.
- First letter MUST be capitalized. Use Title Case (capitalize major words).
- Clean, descriptive, human-readable.
- DO NOT stuff multiple keywords. One clear phrase.
- Under 70 characters.
- COLLECTION-LEVEL ONLY: The H1 describes the ENTIRE collection, not one product.
  Only mention attributes shared by MOST products (e.g., primary material).
  Never mention features that only some products have (e.g., specific gemstones,
  specific motifs that appear on only a few items).
- Format: [Main Keyword] — [Optional short qualifier using COMMON attributes]
  GOOD: "Owl Rings — Handcrafted Sterling Silver"  (material shared by most)
  GOOD: "Skull Biker Rings — Heavy Sterling Silver & Steel"
  BAD: "Owl Rings — Sterling Silver with Garnet Eyes"  (garnet eyes is only on SOME products)
  BAD: "Skull Rings with Red Gemstone Celtic Design"  (too specific to one product)

### META TITLE RULES:
- MUST be under 60 characters (Google truncates at ~60).
  This is a HARD LIMIT — never exceed 60 characters.
- Main Keyword at the FRONT — Google gives more weight to words at the start.
- First letter MUST be capitalized. Use Title Case for major words.
- Format: [Main Keyword] — [1 Differentiator] | Bikerringshop
- COLLECTION-LEVEL ONLY: The differentiator must apply to the ENTIRE collection.
  Use the PRIMARY material (from collection product data) or the main style/audience.
  NEVER use details specific to individual products (specific gemstones, specific
  design elements that only some products have).
- Brand name "Bikerringshop" at the end after | — BUT ONLY if the total
  stays under 60 characters. If adding "| Bikerringshop" would push
  the title over 60 chars, DROP the brand entirely. The keyword and
  differentiator are more valuable for SEO than the brand name.
  Google already shows the site domain in SERP results.
  GOOD: "Skull Biker Rings — Heavy Sterling Silver | Bikerringshop" (54 chars, brand fits)
  GOOD: "Handcrafted Sterling Silver Cross Pendants for Men" (50 chars, no brand — adding brand = 68+)
  BAD: "Owl Rings — Handcrafted .925 Sterling Silver Jewelry | Bikerringshop" (65 chars — OVER LIMIT)
  BAD: "Bikerringshop | The Best Skull Rings Collection Online" (brand first wastes space)
- The differentiator should be a concrete attribute (material, style, audience)
  not a generic claim ("best", "top quality", "amazing").

### META DESCRIPTION RULES:
- Under 155 characters. Aim for 140-155.
- Main Keyword within the first 80 characters (visible on mobile SERP).
- First letter MUST be capitalized (it's a sentence displayed in search results).
- Must answer: "Why should I click THIS collection?"
- COLLECTION-LEVEL ONLY: Describe what the WHOLE collection offers.
  Only mention materials, styles, and attributes shared by MOST products.
  If the collection product data shows 95% sterling silver, say "sterling silver" —
  do NOT add "with garnet eyes" or other details that only apply to a few products.
  Specific weight ranges and size ranges ARE fine (they describe the collection range).
- Include 1 specific differentiator (primary material, construction method, audience).
- End with a soft benefit or audience signal — NOT a "Shop now!" CTA.
  Google's 2026 algorithm devalues aggressive CTAs in meta descriptions.
  GOOD: "Handcrafted owl rings in solid .925 sterling silver. Built for daily wear — detailed feather carving, real weight, sizes 7-15." (under 155)
  BAD: "Owl rings with red garnet eyes in sterling silver. Shop now!" (garnet is product-specific, has CTA)
  BAD: "We have owl rings and bird rings and eagle rings for men and women." (keyword-stuffed list)

Return RAW JSON only. No explanations before or after.
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
        start_idx = cleaned.find(start_char)
        if start_idx == -1: continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start_idx, len(cleaned)):
            c = cleaned[i]
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
                    return json.loads(cleaned[start_idx:i+1])
                except: break
    
    # Step 4: Try to fix truncated JSON (AI response cut off)
    # Find first { and attempt to repair by closing open braces/brackets
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = cleaned.find(start_char)
        if start_idx == -1: continue
        fragment = cleaned[start_idx:]
        # Close any unclosed strings
        in_str = False
        esc = False
        for ch in fragment:
            if esc: esc = False; continue
            if ch == '\\' and in_str: esc = True; continue
            if ch == '"': in_str = not in_str
        if in_str:
            fragment += '"'
        # Remove trailing comma before closing
        fragment = re.sub(r',\s*$', '', fragment)
        # Count unclosed braces/brackets and close them
        open_braces = fragment.count('{') - fragment.count('}')
        open_brackets = fragment.count('[') - fragment.count(']')
        fragment += ']' * max(0, open_brackets)
        fragment += '}' * max(0, open_braces)
        try:
            return json.loads(fragment)
        except:
            # Step 5: Last resort — try to salvage partial fields
            try:
                # Fix common issues: trailing commas, unescaped newlines in strings
                fixed = re.sub(r',(\s*[}\]])', r'\1', fragment)
                return json.loads(fixed)
            except: pass
    
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
def update_shopify_image_seo_only(shop_url, access_token, product_id, image_seo_list, images_pil):
    """Re-upload images with new filenames and alt tags — no content/title/meta changes.
    Uses Shopify's product PUT with images array which replaces all images at once."""
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    
    if not images_pil:
        return False, "No images available to upload"
    
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    # Build images array with new filenames + alt tags (same method as update_shopify_product_v2)
    img_payloads = []
    for i, img in enumerate(images_pil):
        seo = image_seo_list[i] if i < len(image_seo_list) else {}
        img_payloads.append({
            "attachment": img_to_base64(img),
            "filename": seo.get("file_name", f"product-image-{i+1}.jpg"),
            "alt": seo.get("alt_tag", "")
        })
    
    # Only send id + images — no title, body_html, or metafields
    product_payload = {"id": product_id, "images": img_payloads}
    
    try:
        response = requests.put(url, json={"product": product_payload}, headers=headers, timeout=60)
        if response.status_code in [200, 201]:
            return True, f"✅ Re-uploaded {len(img_payloads)} images with new filenames & alt tags"
        return False, f"Shopify API Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

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
    """Fetch all custom + smart collections from Shopify admin with full pagination."""
    import urllib.parse
    all_collections = []
    for ctype in ["custom_collections", "smart_collections"]:
        cursor = None
        for _ in range(20):  # Max 20 pages per type = 5000 collections
            ep = f"{ctype}.json?limit=250"
            if cursor:
                ep = f"{ctype}.json?limit=250&page_info={cursor}"
            res, err = _shopify_admin_get(shop_url, access_token, ep)
            if not res: break
            items = res.json().get(ctype, [])
            if not items: break
            for c in items:
                col_type = "custom" if ctype == "custom_collections" else "smart"
                all_collections.append({"id": c["id"], "title": c.get("title", ""), "handle": c.get("handle", ""), "type": col_type, "body_html": c.get("body_html", "")})
            # Check for next page
            cursor = None
            link_header = res.headers.get("Link", "")
            if 'rel="next"' in link_header:
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        qp = urllib.parse.parse_qs(urllib.parse.urlparse(part.split(";")[0].strip().strip("<>")).query)
                        cursor = qp.get("page_info", [None])[0]
            if not cursor: break
            time.sleep(0.2)
    return all_collections

def get_shopify_products_page(shop_url, access_token, limit=250, page_info=None, collection_id=None):
    """Fetch one page of products from Shopify admin."""
    if page_info:
        # Cursor pagination — page_info already encodes the full query, just append it
        endpoint = f"products.json?limit={limit}&page_info={page_info}"
    elif collection_id:
        endpoint = f"collections/{collection_id}/products.json?limit={limit}"
    else:
        endpoint = f"products.json?limit={limit}"
    
    res, err = _shopify_admin_get(shop_url, access_token, endpoint, timeout=60)
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

def get_shopify_all_products(shop_url, access_token, collection_id=None, max_pages=50, progress_callback=None):
    """Fetch all products (paginated) from Shopify admin."""
    all_products = []
    cursor = None
    for page_num in range(max_pages):
        products, next_cursor, err = get_shopify_products_page(
            shop_url, access_token, limit=250, page_info=cursor, collection_id=collection_id if not cursor else None
        )
        if err and not products:
            if all_products:
                return all_products, f"Partial load ({len(all_products)} products). Stopped: {err}"
            return all_products, err
        all_products.extend(products)
        if progress_callback:
            progress_callback(len(all_products), page_num + 1)
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(0.5)
    return all_products, None

def update_shopify_description_only(shop_url, access_token, product_id, data):
    """Update only title, body_html, and meta fields — no images."""
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
                        "tags": ", ".join(p.get("tags", [])[:10]) if p.get("tags") else ""
                    })
                page += 1
            else: break
        except: break
    
    return catalog

def format_catalog_for_prompt(catalog, max_collections=50, max_products=150, product_context=""):
    """Format catalog data into a compact string for the AI prompt.
    Includes product tags and type to help AI match related items.
    
    If product_context is provided (the raw input description), 
    products are sorted by relevance to the current product first,
    so the AI sees the most related items even with a product limit.
    """
    lines = []
    
    # Always include ALL collections — they're small and critical for linking
    if catalog.get("collections"):
        lines.append("=== REAL COLLECTIONS (use these paths) ===")
        for c in catalog["collections"][:max_collections]:
            lines.append(f"- {c['path']}  →  \"{c['title']}\"")
    
    if catalog.get("products"):
        products = catalog["products"]
        
        # Smart filtering: if we know what product is being written,
        # prioritize related items by matching keywords
        if product_context and len(products) > max_products:
            context_lower = product_context.lower()
            
            # Extract key terms from the product being written
            # Common material/style keywords to match against
            match_terms = []
            material_keywords = ["brass", "sterling silver", "stainless steel", "gold", "silver", 
                                 "copper", "titanium", "tungsten", "bronze", "platinum", "pewter",
                                 "925", "316l", "plated", "two-tone", "rhodium"]
            style_keywords = ["skull", "gothic", "celtic", "biker", "viking", "tribal", "christian",
                              "cross", "dragon", "snake", "eagle", "lion", "wolf", "crown", "angel",
                              "demon", "masonic", "freemason", "templar", "steampunk", "punk",
                              "flame", "skeleton", "death", "pirate", "anchor", "nautical",
                              "buddha", "om", "zen", "hamsa", "evil eye", "pentagram", "norse",
                              "odin", "thor", "rune", "samurai", "japanese", "chinese"]
            type_keywords = ["ring", "pendant", "necklace", "bracelet", "chain", "earring",
                             "wallet chain", "cuff", "bangle", "charm", "brooch", "pin"]
            
            for term in material_keywords + style_keywords + type_keywords:
                if term in context_lower:
                    match_terms.append(term)
            
            if match_terms:
                def relevance_score(product):
                    """Score how related a product is to the current one."""
                    searchable = (product.get("title", "") + " " + product.get("tags", "") + " " + product.get("type", "")).lower()
                    score = 0
                    for term in match_terms:
                        if term in searchable:
                            # Material matches are worth more
                            if term in material_keywords: score += 3
                            # Style matches are valuable
                            elif term in style_keywords: score += 2
                            # Type matches help for cross-selling
                            elif term in type_keywords: score += 1
                    return score
                
                # Sort by relevance (highest first), then take top N
                scored = sorted(products, key=relevance_score, reverse=True)
                products = scored[:max_products]
            else:
                products = products[:max_products]
        else:
            products = products[:max_products]
        
        lines.append(f"\n=== REAL PRODUCTS ({len(products)} of {len(catalog['products'])} shown, sorted by relevance) ===")
        lines.append("Format: path → \"title\" [product_type] {tags}")
        for p in products:
            parts = [f"- {p['path']}  →  \"{p['title']}\""]
            if p.get('type'): parts.append(f"  [{p['type']}]")
            if p.get('tags'): parts.append(f"  {{{p['tags']}}}")
            lines.append("".join(parts))
    
    return "\n".join(lines)

# ============================================================
# --- CLAUDE API FUNCTION ---
# ============================================================
def call_claude_api(claude_key, prompt, img_pil_list=None, model_id="claude-sonnet-4-6"):
    """Call Claude API for Text/SEO tasks with optional image support"""
    url = "https://api.anthropic.com/v1/messages"
    headers = {"Content-Type": "application/json", "x-api-key": claude_key, "anthropic-version": "2023-06-01"}
    
    content = []
    if img_pil_list:
        for idx, img in enumerate(img_pil_list):
            content.append({"type": "text", "text": f"[IMAGE {idx+1} of {len(img_pil_list)}]"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_to_base64(img)}})
    content.append({"type": "text", "text": prompt})
    
    payload = {"model": model_id, "max_tokens": 8192, "messages": [{"role": "user", "content": content}]}
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=180)
            if res.status_code == 200:
                data = res.json()
                text_content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text": text_content += block.get("text", "")
                # Check if response was truncated
                stop_reason = data.get("stop_reason", "")
                if stop_reason == "max_tokens":
                    text_content += "}"  # Try to close truncated JSON
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
        for idx, img in enumerate(img_pil_list):
            content.append({"type": "text", "text": f"[IMAGE {idx+1} of {len(img_pil_list)}]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_to_base64(img)}"}})
    content.append({"type": "text", "text": prompt})
    
    payload = {
        "model": model_id, 
        "max_completion_tokens": 8192,  # Increased for product descriptions with many images
        "messages": [{"role": "user", "content": content}]
    }
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=180)
            if res.status_code == 200:
                data = res.json()
                choice = data.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content", "")
                # Check if response was truncated
                if choice.get("finish_reason") == "length":
                    text += "}"  # Try to close truncated JSON
                return text, None
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
                    st.session_state["_gemini_active_model"] = model
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
    if num_images > 0:
        prompt += f"\n\nNOTE: This product has {num_images} images. You do NOT need to generate image_seo — it will be handled separately. Return an EMPTY array for image_seo: \"image_seo\": []"
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
        for idx, img in enumerate(img_pil_list):
            parts.append({"text": f"[IMAGE {idx+1} of {len(img_pil_list)}]"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192, "responseMimeType": "application/json"}}
    return _call_gemini_text(gemini_key, payload, timeout=120)


def generate_image_seo_per_image(gemini_key, claude_key, openai_key, selected_model, img_pil, image_index, total_images, product_name, product_description_snippet, previous_filenames=None, previous_alts=None):
    """Generate SEO file_name and alt_tag for a SINGLE image.
    
    Sends ONE image at a time with product context so the AI:
    1. Cannot mix up image ordering (only 1 image per call)
    2. Uses product description to verify what it sees (cross-reference)
    3. Sees previous file_names/alt_tags to avoid repetition
    """
    prev_files_str = ""
    if previous_filenames:
        prev_files_str = "\n".join([f"  - Image {i+1}: {fn}" for i, fn in enumerate(previous_filenames)])
    prev_alts_str = ""
    if previous_alts:
        prev_alts_str = "\n".join([f"  - Image {i+1}: {at}" for i, at in enumerate(previous_alts)])
    
    prev_context = ""
    if prev_files_str or prev_alts_str:
        prev_context = f"""
**ALREADY USED (DO NOT REPEAT — you must use DIFFERENT wording):**
File names already assigned to previous images:
{prev_files_str if prev_files_str else "  (none yet — this is the first image)"}
Alt tags already assigned to previous images:
{prev_alts_str if prev_alts_str else "  (none yet — this is the first image)"}
"""
    
    prompt = f"""You are an SEO & Visual Content Specialist for Jewelry e-commerce.
Updated for Google's 2026 Image SEO + Visual Search Best Practices
(February 2026 Core Update & December 2025 Core Update compliance).

**Task:** Generate an SEO-optimized file_name and alt_tag for this ONE product image.

**Product Context (use this to VERIFY what you see in the image):**
- Product Name: {product_name}
- Description snippet: {product_description_snippet}
- This is image {image_index} of {total_images} for this product.
{prev_context}
**CRITICAL RULES:**

1. **LOOK at the image carefully.** Describe what you ACTUALLY SEE.
2. **CROSS-REFERENCE with the product name and description above.**
   If you see a symbol that could be ambiguous (e.g., it looks like a 
   question mark OR an ankh), CHECK the product name/description for clues.
   Trust the text data over your visual guess when visuals are ambiguous.
3. **DO NOT REPEAT** any file_name or alt_tag pattern from previous images listed above.
   Each image must have COMPLETELY DIFFERENT wording — not just a different suffix.

**FILE NAME RULES (Google 2026 Best Practices):**
- Lowercase, hyphens only, end with .jpg
- Structure: [visual-focus]-[material-or-detail]-[angle-or-context].jpg
- 3-7 hyphenated words. Concise but descriptive.
- For image 1 ONLY: include the full product name/identifier.
  For images 2+: DO NOT repeat the full product name. Lead with the
  VISUAL FOCUS unique to THIS image (the angle, detail, material, or feature).
- Include product attributes (material, color) to help search engines
  map images to catalog pages.
- Each file name must describe what makes THIS specific image DIFFERENT.

GOOD example (8 images — notice each LEADS with different focus):
1. christian-crosier-bishop-ring-amethyst-top-view.jpg  (image 1: full name)
2. gold-plated-cross-cutout-band-detail.jpg  (leads with visual detail)
3. amethyst-gemstone-crosier-setting-closeup.jpg  (leads with gemstone)
4. ribbed-gold-band-side-profile.jpg  (leads with texture)
5. openwork-cross-pattern-left-angle.jpg  (leads with pattern)
6. bishop-ring-worn-on-hand-lifestyle.jpg  (leads with context)
7. sterling-silver-base-interior-hallmark.jpg  (leads with interior)
8. crosier-bishop-ring-full-set-flat-lay.jpg  (leads with composition)

BAD example (all start the same = keyword stuffing, Google flags as spam):
1. christian-crosier-bishop-ring-angled-right-view.jpg
2. christian-crosier-bishop-ring-side-profile-left.jpg
3. christian-crosier-bishop-ring-side-profile-right.jpg

**ALT TAG RULES (Google 2026 + W3C Accessibility):**
- Describe what is VISUALLY shown in THIS specific image.
- Under 125 characters (screen reader best practice per W3C/Google).
- DO NOT start with the same product name prefix as previous images.
  Vary the opening: lead with the visual focus, the material detail,
  the angle, or the feature being highlighted in THIS image.
- Include relevant keywords NATURALLY but DIFFERENTLY from previous images.
  Google explicitly warns: keyword stuffing in alt attributes results
  in a negative user experience and may flag the site as spam.
- Never use "image of" or "picture of" — describe the content directly.

GOOD alt tags (varied openings — no two start the same way):
1. "Top view of the Christian crosier bishop ring with purple amethyst center stone"
2. "Gold-plated band detail showing openwork cross cutouts and ribbed texture"
3. "Close-up of the amethyst gemstone set in a crosier-shaped sterling silver bezel"
4. "Side profile highlighting the ribbed gold plating and layered band construction"

BAD alt tags (repetitive prefix = pattern Google flags as AI-generated spam):
1. "Christian crosier bishop ring angled right view showing amethyst..."
2. "Christian crosier bishop ring side profile left showing cross..."
3. "Christian crosier bishop ring side profile right displaying gold..."

**2026 VISUAL SEARCH CONTEXT:**
Google Lens now handles 1 in 10 searches with ~20% commercial intent.
Well-optimized product images appear in Google Lens visual search,
Google Shopping, and AI-powered "shop similar" experiences.
Diverse, specific file names and alt tags increase discovery across
multiple visual search queries — repetitive names only rank for one query.

Return RAW JSON only (no markdown backticks):
{{"file_name": "descriptive-name.jpg", "alt_tag": "Unique description of what this image shows"}}"""

    # Claude models
    if selected_model in CLAUDE_MODELS and claude_key:
        model_id = CLAUDE_MODELS[selected_model]
        return call_claude_api(claude_key, prompt, [img_pil], model_id=model_id)
    
    # OpenAI models
    if selected_model in OPENAI_MODELS and openai_key:
        model_id = OPENAI_MODELS[selected_model]
        return call_openai_api(openai_key, prompt, [img_pil], model_id=model_id)
    
    # Default: Gemini
    parts = [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}]
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json"}}
    return _call_gemini_text(gemini_key, payload, timeout=30)

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

def summarize_collection_products(shop_url, access_token, collection_id, max_products=100):
    """Fetch products in a specific collection and create a summary for AI context.
    Returns a text summary of materials, product types, and product names."""
    products, next_cursor, err = get_shopify_products_page(
        shop_url, access_token, limit=min(max_products, 250), collection_id=collection_id
    )
    if not products:
        return ""
    
    # Extract key info
    titles = []
    materials_found = {}
    product_types = {}
    
    material_keywords = [
        "sterling silver", ".925 silver", "925 silver", "silver",
        "stainless steel", "316l", "steel",
        "brass", "bronze", "copper",
        "gold", "gold plated", "gold-plated", "14k", "18k", "10k",
        "titanium", "tungsten", "pewter", "leather"
    ]
    
    for p in products:
        titles.append(p.get("title", ""))
        # Count product types
        ptype = p.get("product_type", "").strip()
        if ptype:
            product_types[ptype] = product_types.get(ptype, 0) + 1
        # Scan title + body for materials
        text = (p.get("title", "") + " " + (p.get("body_html", "") or "")).lower()
        for mat in material_keywords:
            if mat in text:
                # Normalize to common name
                if mat in [".925 silver", "925 silver"]:
                    mat_key = "sterling silver"
                elif mat == "316l":
                    mat_key = "stainless steel"
                else:
                    mat_key = mat
                materials_found[mat_key] = materials_found.get(mat_key, 0) + 1
    
    # Build summary
    lines = []
    lines.append(f"Total products in this collection: {len(products)}")
    
    if materials_found:
        # Sort by count descending
        sorted_mats = sorted(materials_found.items(), key=lambda x: -x[1])
        
        # Separate common vs rare materials
        common_mats = []
        rare_mats = []
        for name, count in sorted_mats:
            pct = int(count / len(products) * 100)
            if pct >= 20:
                common_mats.append(f"{name} ({count} products, {pct}%)")
            else:
                rare_mats.append(f"{name} ({count} products, {pct}%)")
        
        if common_mats:
            lines.append(f"COMMON materials (use these in title/meta/content): {', '.join(common_mats)}")
        if rare_mats:
            lines.append(f"RARE materials (do NOT feature in title/meta — only a few products): {', '.join(rare_mats)}")
        
        # Highlight primary material
        primary = sorted_mats[0]
        pct = int(primary[1] / len(products) * 100)
        if pct > 50:
            lines.append(f"⚠️ PRIMARY material: {primary[0]} ({pct}% of products) — use this as the main material in H1, meta title, and meta description")
    
    if product_types:
        sorted_types = sorted(product_types.items(), key=lambda x: -x[1])
        type_strs = [f"{name} ({count})" for name, count in sorted_types[:5]]
        lines.append(f"Product types: {', '.join(type_strs)}")
    
    # Show sample product names (first 15)
    sample = titles[:15]
    lines.append(f"Sample product names: {', '.join(sample)}")
    
    return "\n".join(lines)

def generate_collection_content(gemini_key, claude_key, openai_key, selected_model, main_keyword, collection_url, catalog_text="", collection_products_summary=""):
    """Generate SEO collection page content."""
    import random
    prompt = SEO_COLLECTION_WRITER_PROMPT.replace("{main_keyword}", main_keyword).replace("{collection_url}", collection_url)
    
    # Randomize opening format to prevent repetitive patterns across collections
    formats = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    chosen_format = random.choice(formats)
    format_labels = {
        "A": "definition — '[Collection] are [what they are].'",
        "B": "audience-first — 'Built for [audience], [collection] [trait].'",
        "C": "material-lead — 'Cast in [material], these [type] [purpose].'",
        "D": "statement — 'Every [item] in this collection [unique fact].'",
        "E": "direct — 'If you [need], [collection] [delivers].'",
        "F": "spec-highlight — '[Collection] need [quality] to [purpose] — these [spec].'",
        "G": "weight/spec-lead — 'At [spec], [collection] [impression].'",
        "H": "question — 'Looking for [need]? [Collection] [answer].'",
        "I": "craftsmanship — 'Handcast in [method], each [item] [detail].'",
        "J": "use-case — 'Whether you're [activity], [collection] [benefit].'",
    }
    prompt += f"\n\n⚠️ OPENING FORMAT INSTRUCTION: For THIS collection, use FORMAT {chosen_format} ({format_labels[chosen_format]}) for the first sentence of Paragraph 1. Do NOT use '[Collection name] are...' unless format A was assigned."
    if collection_products_summary:
        prompt += f"""

--- ACTUAL PRODUCTS IN THIS COLLECTION (use this data to write ACCURATE content) ---
The following is a summary of the REAL products currently in this collection.

CRITICAL RULES FOR USING THIS DATA:
1. Your H1, meta title, and meta description MUST only mention attributes
   shared by MOST products (marked as COMMON materials or PRIMARY material).
2. NEVER put product-specific details in H1/meta title/meta description.
   For example, if only 3 out of 30 products have "red garnet eyes", do NOT
   mention "garnet eyes" in the meta title — it misleads searchers.
3. Product-specific details (like specific gemstones, motifs found on only
   some items) may be mentioned briefly in the body description as variety examples,
   but NEVER in H1, meta title, or meta description.
4. Materials marked as RARE (under 20% of products) should NOT be featured
   as a primary material in any title or meta field.

{collection_products_summary}
--- END COLLECTION PRODUCTS ---"""
    if catalog_text:
        prompt += f"\n\n--- REAL STORE CATALOG DATA (for internal links) ---\n{catalog_text}\n--- END CATALOG DATA ---"
    
    # Claude models
    if selected_model in CLAUDE_MODELS and claude_key:
        model_id = CLAUDE_MODELS[selected_model]
        return call_claude_api(claude_key, prompt, None, model_id=model_id)
    
    # OpenAI models
    if selected_model in OPENAI_MODELS and openai_key:
        model_id = OPENAI_MODELS[selected_model]
        return call_openai_api(openai_key, prompt, None, model_id=model_id)
    
    # Default: Gemini (with fallback)
    parts = [{"text": prompt}]
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192, "responseMimeType": "application/json"}}
    return _call_gemini_text(gemini_key, payload)

def update_shopify_collection(shop_url, access_token, collection_id, data, collection_type="custom"):
    """Update a Shopify collection's title, body_html, and meta fields."""
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    
    endpoint_type = "custom_collections" if collection_type == "custom" else "smart_collections"
    url = f"https://{shop_url}/admin/api/2024-01/{endpoint_type}/{collection_id}.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    col_key = "custom_collection" if collection_type == "custom" else "smart_collection"
    payload = {
        col_key: {
            "id": collection_id,
            "title": data.get("collection_title", ""),
            "body_html": data.get("collection_description_html", ""),
            "metafields": [
                {"namespace": "global", "key": "title_tag", "value": data.get("meta_title", ""), "type": "single_line_text_field"},
                {"namespace": "global", "key": "description_tag", "value": data.get("meta_description", ""), "type": "multi_line_text_field"}
            ]
        }
    }
    
    try:
        response = requests.put(url, json=payload, headers=headers, timeout=30)
        if response.status_code in [200, 201]: return True, "✅ Collection Updated!"
        return False, f"Error {response.status_code}: {response.text[:300]}"
    except Exception as e: return False, str(e)

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
if "colwriter_result" not in st.session_state: st.session_state.colwriter_result = None
if "colwriter_collections" not in st.session_state: st.session_state.colwriter_collections = []

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
tab1, tab_retouch, tab2, tab3, tab_batch, tab_colwriter, tab4, tab5 = st.tabs(["✨ Gen Image", "🎨 Retouch", "🏷️ Bulk SEO", "📝 Writer", "📝 Batch Writer", "📂 Collection Writer", "📚 Library", "ℹ️ Models"])

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
                            # Clear previous results and image SEO edits on new fetch
                            st.session_state.writer_result = None
                            st.session_state.pop("writer_img_seo_edits", None)
                            st.session_state.pop("_writer_img_seo_fingerprint", None)
                            # Save fetched Product ID and increment publish counter
                            st.session_state['writer_fetched_prod_id'] = sh_writer_id
                            if "writer_publish_counter" not in st.session_state: st.session_state.writer_publish_counter = 0
                            st.session_state.writer_publish_counter += 1
                            st.success("Loaded!"); st.rerun()
                if col_w_clear.button("❌ Clear", key=f"writer_clear_btn_{writer_key_id}"):
                    st.session_state.writer_shopify_imgs = []
                    st.session_state.writer_result = None
                    st.session_state.writer_fetched_prod_id = ""
                    st.session_state.writer_key_counter += 1
                    st.session_state.pop("writer_img_seo_edits", None)
                    st.session_state.pop("_writer_img_seo_fingerprint", None)
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
        gen_mode = st.radio("Generation Mode:", 
            ["📝 Content + Image SEO", "📝 Content Only", "🖼️ Image SEO Only"],
            key=f"writer_gen_mode_{writer_key_id}", horizontal=True)
        wb1, wb2 = st.columns([1, 1])
        run_write = wb1.button("🚀 Generate", type="primary", key=f"writer_run_btn_{writer_key_id}")
        if wb2.button("🔄 Start Over", key=f"writer_startover_btn_{writer_key_id}"):
            st.session_state.writer_result = None; st.session_state.writer_shopify_imgs = []; st.session_state.writer_fetched_prod_id = ""; st.session_state.writer_key_counter += 1; st.session_state.pop("writer_img_seo_edits", None); st.session_state.pop("_writer_img_seo_fingerprint", None); st.rerun()
    with c2:
        if run_write:
            # Check API key based on selected model
            missing_key = False
            if selected_text_model == "Gemini" and not gemini_key: missing_key = True
            elif selected_text_model in CLAUDE_MODELS and not claude_key: missing_key = True
            elif selected_text_model in OPENAI_MODELS and not openai_key: missing_key = True
            
            if missing_key: st.error("Missing API Key")
            elif gen_mode == "🖼️ Image SEO Only" and not writer_imgs: st.error("No images — fetch a product or upload images first")
            elif gen_mode != "🖼️ Image SEO Only" and not raw: st.error("Missing details")
            else:
                # Clear previous image SEO edits
                st.session_state.pop("writer_img_seo_edits", None)
                st.session_state.pop("_writer_img_seo_fingerprint", None)
                
                if gen_mode == "🖼️ Image SEO Only":
                    # --- IMAGE SEO ONLY MODE ---
                    with st.spinner(f"Generating Image SEO with {current_text_model}..."):
                        product_name = raw[:100] if raw else "Product"
                        desc_snippet = raw[:300] if raw else ""
                        image_seo_results = []
                        prev_fnames = []
                        prev_alts = []
                        progress_bar = st.progress(0, text="🖼️ Generating Image SEO...")
                        for idx, img in enumerate(writer_imgs):
                            progress_bar.progress((idx + 1) / len(writer_imgs), text=f"🖼️ Image SEO {idx+1}/{len(writer_imgs)}...")
                            try:
                                img_json, img_err = generate_image_seo_per_image(
                                    gemini_key, claude_key, openai_key, current_text_model,
                                    img, idx + 1, len(writer_imgs), product_name, desc_snippet,
                                    previous_filenames=prev_fnames if prev_fnames else None,
                                    previous_alts=prev_alts if prev_alts else None
                                )
                                if img_json:
                                    img_d = parse_json_response(img_json)
                                    if isinstance(img_d, list) and img_d: img_d = img_d[0]
                                    if isinstance(img_d, dict):
                                        image_seo_results.append(img_d)
                                        prev_fnames.append(img_d.get("file_name", ""))
                                        prev_alts.append(img_d.get("alt_tag", ""))
                                    else:
                                        fallback = {"file_name": f"product-image-{idx+1}.jpg", "alt_tag": f"Product image {idx+1}"}
                                        image_seo_results.append(fallback)
                                        prev_fnames.append(fallback["file_name"])
                                        prev_alts.append(fallback["alt_tag"])
                                else:
                                    fallback = {"file_name": f"product-image-{idx+1}.jpg", "alt_tag": f"Product image {idx+1}"}
                                    image_seo_results.append(fallback)
                                    prev_fnames.append(fallback["file_name"])
                                    prev_alts.append(fallback["alt_tag"])
                            except Exception as img_e:
                                fallback = {"file_name": f"product-image-{idx+1}.jpg", "alt_tag": f"Product image {idx+1}"}
                                image_seo_results.append(fallback)
                                prev_fnames.append(fallback["file_name"])
                                prev_alts.append(fallback["alt_tag"])
                            time.sleep(0.3)
                        progress_bar.empty()
                        # Store as image_seo_only result (no content)
                        st.session_state.writer_result = {"_image_seo_only": True, "image_seo": image_seo_results}
                        st.rerun()
                else:
                    # --- CONTENT MODE (with or without Image SEO) ---
                    with st.spinner(f"Writing with {current_text_model}..."):
                        # Fetch real store catalog for internal linking
                        catalog_text = ""
                        try:
                            catalog = fetch_store_catalog("www.bikerringshop.com")
                            if catalog.get("collections") or catalog.get("products"):
                                catalog_text = format_catalog_for_prompt(catalog, product_context=raw)
                        except: pass
                        json_txt, err = generate_full_product_content(gemini_key, claude_key, openai_key, current_text_model, writer_imgs, raw, catalog_text)
                        # Show which Gemini model was actually used
                        if current_text_model == "Gemini" and json_txt:
                            active_m = st.session_state.get("_gemini_active_model", "")
                            if active_m:
                                st.toast(f"✅ Used: {active_m.replace('models/', '')}")
                        if json_txt:
                            d = parse_json_response(json_txt)
                            if isinstance(d, list) and d: d = d[0]
                            if isinstance(d, dict):
                                # --- PASS 2: Generate image_seo per-image (only if Content + Image SEO mode) ---
                                if writer_imgs and gen_mode == "📝 Content + Image SEO":
                                    product_name = d.get('product_title_h1', '') or raw[:100]
                                    desc_snippet = raw[:300]
                                    image_seo_results = []
                                    prev_fnames = []
                                    prev_alts = []
                                    progress_bar = st.progress(0, text="🖼️ Generating Image SEO...")
                                    for idx, img in enumerate(writer_imgs):
                                        progress_bar.progress((idx + 1) / len(writer_imgs), text=f"🖼️ Image SEO {idx+1}/{len(writer_imgs)}...")
                                        try:
                                            img_json, img_err = generate_image_seo_per_image(
                                                gemini_key, claude_key, openai_key, current_text_model,
                                                img, idx + 1, len(writer_imgs), product_name, desc_snippet,
                                                previous_filenames=prev_fnames if prev_fnames else None,
                                                previous_alts=prev_alts if prev_alts else None
                                            )
                                            if img_json:
                                                img_d = parse_json_response(img_json)
                                                if isinstance(img_d, list) and img_d: img_d = img_d[0]
                                                if isinstance(img_d, dict):
                                                    image_seo_results.append(img_d)
                                                    prev_fnames.append(img_d.get("file_name", ""))
                                                    prev_alts.append(img_d.get("alt_tag", ""))
                                                else:
                                                    fallback = {"file_name": f"product-image-{idx+1}.jpg", "alt_tag": f"Product image {idx+1}"}
                                                    image_seo_results.append(fallback)
                                                    prev_fnames.append(fallback["file_name"])
                                                    prev_alts.append(fallback["alt_tag"])
                                            else:
                                                fallback = {"file_name": f"product-image-{idx+1}.jpg", "alt_tag": f"Product image {idx+1}"}
                                                image_seo_results.append(fallback)
                                                prev_fnames.append(fallback["file_name"])
                                                prev_alts.append(fallback["alt_tag"])
                                        except Exception as img_e:
                                            fallback = {"file_name": f"product-image-{idx+1}.jpg", "alt_tag": f"Product image {idx+1}"}
                                            image_seo_results.append(fallback)
                                            prev_fnames.append(fallback["file_name"])
                                            prev_alts.append(fallback["alt_tag"])
                                        time.sleep(0.3)  # Rate limit safety
                                    progress_bar.empty()
                                    d["image_seo"] = image_seo_results
                                st.session_state.writer_result = d; st.rerun()
                            else:
                                st.error("⚠️ AI returned content but JSON parsing failed. This usually happens when the response was truncated (too long) or contained invalid characters. Try again — the AI may produce a cleaner output on retry.")
                                with st.expander("🔍 Raw AI Output (for debugging)", expanded=False):
                                    st.code(json_txt[:3000] if len(json_txt) > 3000 else json_txt)
                                # Attempt partial recovery — try to extract at least some fields
                                partial = {}
                                for field in ['url_slug', 'meta_title', 'meta_description', 'product_title_h1']:
                                    m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', json_txt)
                                    if m: partial[field] = m.group(1)
                                # Try to get html_content (may contain quotes)
                                m = re.search(r'"html_content"\s*:\s*"(.*?)(?:"\s*,\s*"image_seo|"\s*})', json_txt, re.DOTALL)
                                if m: partial['html_content'] = m.group(1).replace('\\"', '"').replace('\\n', '\n')
                                if partial and len(partial) >= 3:
                                    st.info(f"🔧 Partially recovered {len(partial)} fields. You can use these or regenerate.")
                                    st.session_state.writer_result = partial; st.rerun()
                        else: st.error(err)
        if st.session_state.writer_result:
            d = st.session_state.writer_result
            is_img_seo_only = d.get("_image_seo_only", False)
            
            if not is_img_seo_only:
                st.subheader("Content Results")
                # Show which model actually ran
                active_gemini = st.session_state.get("_gemini_active_model", "")
                if current_text_model == "Gemini" and active_gemini:
                    model_label = active_gemini.replace("models/", "")
                    st.caption(f"🤖 Generated by: **{model_label}**")
                elif current_text_model in CLAUDE_MODELS:
                    st.caption(f"🤖 Generated by: **{CLAUDE_MODELS[current_text_model]}**")
                elif current_text_model in OPENAI_MODELS:
                    st.caption(f"🤖 Generated by: **{OPENAI_MODELS[current_text_model]}**")
                st.write("**H1:**"); st.code(d.get('product_title_h1', ''))
                st.write("**Slug:**"); st.code(d.get('url_slug', ''))
                st.write("**Meta Title:**"); st.code(d.get('meta_title', ''))
                st.write("**Meta Desc:**"); st.code(d.get('meta_description', ''))
                with st.expander("HTML Content"): st.code(d.get('html_content', ''), language="html")
                st.markdown(d.get('html_content', ''), unsafe_allow_html=True)
            else:
                st.subheader("🖼️ Image SEO Only Results")
            st.divider(); st.subheader("🖼️ Image SEO")
            img_tags = d.get('image_seo', [])
            if writer_imgs:
                # Initialize or reinitialize editable image_seo from current result
                # Use a fingerprint to detect when result has changed
                current_seo_fingerprint = str([(t.get('file_name',''), t.get('alt_tag','')) for t in img_tags if isinstance(t, dict)])
                prev_fingerprint = st.session_state.get("_writer_img_seo_fingerprint", "")
                
                needs_reinit = (
                    "writer_img_seo_edits" not in st.session_state
                    or len(st.session_state.writer_img_seo_edits) != len(writer_imgs)
                    or current_seo_fingerprint != prev_fingerprint
                )
                
                if needs_reinit:
                    st.session_state.writer_img_seo_edits = []
                    for i in range(len(writer_imgs)):
                        if i < len(img_tags) and isinstance(img_tags[i], dict):
                            st.session_state.writer_img_seo_edits.append({
                                "file_name": clean_filename(img_tags[i].get('file_name', '')),
                                "alt_tag": img_tags[i].get('alt_tag', '')
                            })
                        else:
                            st.session_state.writer_img_seo_edits.append({
                                "file_name": f"product-image-{i+1}.jpg",
                                "alt_tag": f"Product image {i+1}"
                            })
                    st.session_state._writer_img_seo_fingerprint = current_seo_fingerprint
                
                img_seo_changed = False
                for i, img in enumerate(writer_imgs):
                    ic1, ic2 = st.columns([1, 3])
                    with ic1: st.image(img, width=120); st.caption(f"Image {i+1}")
                    with ic2:
                        new_fname = st.text_input(
                            f"File Name (Image {i+1}):", 
                            value=st.session_state.writer_img_seo_edits[i]["file_name"],
                            key=f"img_seo_fname_{writer_key_id}_{i}"
                        )
                        new_alt = st.text_input(
                            f"Alt Tag (Image {i+1}):", 
                            value=st.session_state.writer_img_seo_edits[i]["alt_tag"],
                            key=f"img_seo_alt_{writer_key_id}_{i}"
                        )
                        # Track changes
                        if new_fname != st.session_state.writer_img_seo_edits[i]["file_name"]:
                            st.session_state.writer_img_seo_edits[i]["file_name"] = new_fname
                            img_seo_changed = True
                        if new_alt != st.session_state.writer_img_seo_edits[i]["alt_tag"]:
                            st.session_state.writer_img_seo_edits[i]["alt_tag"] = new_alt
                            img_seo_changed = True
                    st.divider()
                
                # Apply edits button
                if st.button("💾 Apply Image SEO Edits", key=f"apply_img_seo_{writer_key_id}", type="secondary"):
                    st.session_state.writer_result["image_seo"] = [
                        {"file_name": e["file_name"], "alt_tag": e["alt_tag"]}
                        for e in st.session_state.writer_img_seo_edits
                    ]
                    st.success("✅ Image SEO updated!")
                    st.rerun()
                
                # Always sync edits to writer_result for publish
                d["image_seo"] = [
                    {"file_name": e["file_name"], "alt_tag": e["alt_tag"]}
                    for e in st.session_state.writer_img_seo_edits
                ]
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
                
                if is_img_seo_only:
                    # Image SEO Only — re-upload images with new filenames + alt tags
                    st.info("🖼️ Image SEO Only — will delete old images and re-upload with new filenames & alt tags (no content changes)")
                    if st.button("☁️ Re-upload Images with SEO", type="primary", use_container_width=True, key=f"writer_update_imgseo_btn_{writer_key_id}_{writer_publish_counter}"):
                        if not s_shop or not s_token or not s_prod_id: st.error("❌ Missing Data")
                        else:
                            with st.spinner("Updating image SEO..."):
                                success, msg = update_shopify_image_seo_only(s_shop, s_token, s_prod_id, d.get("image_seo", []), writer_imgs)
                                if success: st.success(msg); st.balloons()
                                else: st.error(msg)
                else:
                    # Full content update
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
                
                # Get total product count first
                total_count = "?"
                count_res, _ = _shopify_admin_get(bw_shop, bw_token, "products/count.json")
                if count_res:
                    total_count = count_res.json().get("count", "?")
                
                load_status.info(f"✅ {len(collections)} collections. Loading products (total: {total_count})...")
                
                def progress_cb(count, page):
                    load_progress.progress(min(page / 50, 0.99), text=f"Loaded {count} products (page {page})...")
                
                products, err = get_shopify_all_products(bw_shop, bw_token, progress_callback=progress_cb)
                load_progress.empty()
                
                if err and not products:
                    load_status.error(f"Failed: {err}")
                else:
                    st.session_state.batch_products = products
                    st.session_state.batch_collections = collections
                    st.session_state.batch_results = {}
                    # Clear collection filter cache
                    for k in list(st.session_state.keys()):
                        if k.startswith("_batch_col_cache_"): del st.session_state[k]
                    warn_msg = f" ⚠️ {err}" if err else ""
                    count_msg = f" (of {total_count} total)" if total_count != "?" else ""
                    load_status.success(f"✅ Loaded {len(products)}{count_msg} products, {len(collections)} collections{warn_msg}")
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
                sel_col = next((c for c in st.session_state.batch_collections if c["title"] == selected_collection), None)
                if sel_col:
                    cache_key = f"_batch_col_cache_{sel_col['id']}"
                    if cache_key not in st.session_state:
                        col_product_ids = set()
                        
                        # Method 1: Collects API — works for custom collections
                        collect_cursor = None
                        for _ in range(20):
                            ep = f"collects.json?collection_id={sel_col['id']}&limit=250"
                            if collect_cursor: ep += f"&page_info={collect_cursor}"
                            res, _ = _shopify_admin_get(bw_shop, bw_token, ep)
                            if not res: break
                            collects = res.json().get("collects", [])
                            if not collects: break
                            for c in collects: col_product_ids.add(str(c.get("product_id", "")))
                            lh = res.headers.get("Link", "")
                            collect_cursor = None
                            if 'rel="next"' in lh:
                                import urllib.parse
                                for part in lh.split(","):
                                    if 'rel="next"' in part:
                                        qp = urllib.parse.parse_qs(urllib.parse.urlparse(part.split(";")[0].strip().strip("<>")).query)
                                        collect_cursor = qp.get("page_info", [None])[0]
                            if not collect_cursor: break
                            time.sleep(0.2)
                        
                        # Method 2: If Collects returned nothing — smart collection
                        if not col_product_ids:
                            prod_cursor = None
                            for _ in range(20):
                                ep = f"collections/{sel_col['id']}/products.json?limit=250"
                                if prod_cursor: ep += f"&page_info={prod_cursor}"
                                res, _ = _shopify_admin_get(bw_shop, bw_token, ep)
                                if not res: break
                                prods = res.json().get("products", [])
                                if not prods: break
                                for p in prods: col_product_ids.add(str(p.get("id", "")))
                                lh = res.headers.get("Link", "")
                                prod_cursor = None
                                if 'rel="next"' in lh:
                                    import urllib.parse
                                    for part in lh.split(","):
                                        if 'rel="next"' in part:
                                            qp = urllib.parse.parse_qs(urllib.parse.urlparse(part.split(";")[0].strip().strip("<>")).query)
                                            prod_cursor = qp.get("page_info", [None])[0]
                                if not prod_cursor: break
                                time.sleep(0.2)
                        
                        st.session_state[cache_key] = col_product_ids
                    
                    filtered = [p for p in filtered if p["id"] in st.session_state[cache_key]]
            
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
                            catalog = None
                            try:
                                catalog = fetch_store_catalog("www.bikerringshop.com")
                            except: pass
                            
                            progress_bar = st.progress(0)
                            status_container = st.container()
                            
                            for idx, prod in enumerate(selected_products):
                                pid = prod["id"]
                                st.session_state.batch_results[pid] = {"generating": True}
                                
                                with status_container:
                                    # Show which model is being used
                                    if batch_model == "Gemini":
                                        active_m = st.session_state.get("_gemini_active_model", MODEL_TEXT_GEMINI).replace("models/", "")
                                        model_tag = f"Gemini (`{active_m}`)"
                                    elif batch_model in CLAUDE_MODELS:
                                        model_tag = f"{batch_model} (`{CLAUDE_MODELS[batch_model]}`)"
                                    elif batch_model in OPENAI_MODELS:
                                        model_tag = f"{batch_model} (`{OPENAI_MODELS[batch_model]}`)"
                                    else:
                                        model_tag = batch_model
                                    st.write(f"⏳ [{idx+1}/{len(selected_products)}] **{prod['title'][:60]}** — {model_tag}")
                                
                                # Build input from existing product data
                                raw_input = prod.get("body_html", "") or ""
                                raw_input = remove_html_tags(raw_input) if raw_input else ""
                                raw_input = f"Product Name: {prod['title']}\nProduct Type: {prod.get('product_type', '')}\nSKU: {prod.get('sku', '')}\n\n{raw_input}"
                                
                                # Generate content (text only — no images for batch speed)
                                try:
                                    # Smart catalog: filter by this product's context for relevant links
                                    catalog_text = ""
                                    if catalog and (catalog.get("collections") or catalog.get("products")):
                                        catalog_text = format_catalog_for_prompt(catalog, product_context=raw_input)
                                    
                                    json_txt, err = generate_full_product_content(
                                        gemini_key, claude_key, openai_key, batch_model, 
                                        None, raw_input, catalog_text
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
                                st.write("**Slug:**", d.get("url_slug", ""))
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

# === TAB COLLECTION WRITER ===
with tab_colwriter:
    st.header("📂 Collection Page Writer")
    
    cw_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
    cw_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
    current_text_model_cw = st.session_state.get('selected_text_model', 'Gemini')
    
    if not cw_shop or not cw_token:
        st.error("❌ Shopify credentials required.")
    else:
        # --- Model + Load Collections ---
        cw_c1, cw_c2 = st.columns([1, 1])
        with cw_c1:
            all_models_cw = ["Gemini"] + list(CLAUDE_MODELS.keys()) + list(OPENAI_MODELS.keys())
            cw_model = st.selectbox("🤖 Model:", all_models_cw,
                                     index=all_models_cw.index(current_text_model_cw) if current_text_model_cw in all_models_cw else 0,
                                     key="cw_model_select")
        with cw_c2:
            st.write(""); st.write("")
            if st.button("📦 Load Collections", type="primary", use_container_width=True, key="cw_load_btn"):
                with st.spinner("Loading collections from Shopify..."):
                    collections = get_shopify_all_collections(cw_shop, cw_token)
                    if collections:
                        st.session_state.colwriter_collections = collections
                        st.session_state.colwriter_result = None
                        st.success(f"✅ Loaded {len(collections)} collections")
                        st.rerun()
                    else:
                        st.error("Failed to load collections")
        
        if st.session_state.colwriter_collections:
            st.divider()
            
            # --- Collection Selector + Main Keyword ---
            col_options = [f"{c['title']}  (/collections/{c['handle']})" for c in st.session_state.colwriter_collections]
            selected_col_idx = st.selectbox("📁 Select Collection:", range(len(col_options)),
                                             format_func=lambda i: col_options[i], key="cw_col_select")
            selected_col = st.session_state.colwriter_collections[selected_col_idx]
            
            # Auto-clear result if user selected a different collection
            if st.session_state.colwriter_result:
                prev_col_id = st.session_state.colwriter_result.get("_col_id")
                if prev_col_id and str(prev_col_id) != str(selected_col["id"]):
                    st.session_state.colwriter_result = None
            
            # Build full URL
            store_domain = cw_shop.replace("https://", "").replace("http://", "").replace(".myshopify.com", "").strip()
            collection_full_url = f"https://www.bikerringshop.com/collections/{selected_col['handle']}"
            st.caption(f"🔗 URL: `{collection_full_url}`")
            
            # Show current description from Shopify
            current_body = selected_col.get("body_html", "")
            if current_body and current_body.strip():
                with st.expander("📄 Current Description (from Shopify)", expanded=False):
                    st.markdown(current_body, unsafe_allow_html=True)
                    plain_text = remove_html_tags(current_body).strip()
                    word_count = len(plain_text.split()) if plain_text else 0
                    st.caption(f"📏 {word_count} words | {len(plain_text)} characters")
            else:
                st.info("📄 No existing description — this collection is currently empty.")
            
            main_keyword = st.text_input("🔑 Main Keyword:", placeholder="e.g. skull biker rings, gothic silver pendants", key="cw_main_keyword")
            
            # --- Generate Button ---
            cw_btn1, cw_btn2 = st.columns([1, 1])
            run_gen = cw_btn1.button("🚀 Generate Collection Content", type="primary", key="cw_gen_btn",
                                      disabled=(not main_keyword))
            if cw_btn2.button("🔄 Start Over", key="cw_startover_btn"):
                st.session_state.colwriter_result = None
                st.rerun()
            
            if run_gen:
                if not main_keyword:
                    st.error("Please enter a Main Keyword")
                else:
                    # Check API key
                    cw_missing_key = False
                    if cw_model == "Gemini" and not gemini_key: cw_missing_key = True
                    elif cw_model in CLAUDE_MODELS and not claude_key: cw_missing_key = True
                    elif cw_model in OPENAI_MODELS and not openai_key: cw_missing_key = True
                    
                    if cw_missing_key:
                        st.error(f"❌ Missing API Key for {cw_model}")
                    else:
                        with st.spinner(f"Writing collection content with {cw_model}..."):
                            # Fetch real products in THIS collection for accurate content
                            collection_products_summary = ""
                            try:
                                collection_products_summary = summarize_collection_products(
                                    cw_shop, cw_token, selected_col["id"]
                                )
                                if collection_products_summary:
                                    st.toast(f"📦 Loaded product data from collection")
                            except Exception as cps_err:
                                pass  # Continue without product data
                            
                            # Fetch catalog for internal links
                            catalog_text = ""
                            try:
                                catalog = fetch_store_catalog("www.bikerringshop.com")
                                if catalog.get("collections") or catalog.get("products"):
                                    catalog_text = format_catalog_for_prompt(catalog, product_context=main_keyword)
                            except: pass
                            
                            json_txt, err = generate_collection_content(
                                gemini_key, claude_key, openai_key, cw_model,
                                main_keyword, collection_full_url, catalog_text,
                                collection_products_summary=collection_products_summary
                            )
                            
                            if json_txt:
                                d = parse_json_response(json_txt)
                                if isinstance(d, list) and d: d = d[0]
                                if isinstance(d, dict):
                                    # Store collection info with result
                                    d["_col_id"] = selected_col["id"]
                                    d["_col_type"] = selected_col.get("type", "custom")
                                    d["_col_handle"] = selected_col["handle"]
                                    st.session_state.colwriter_result = d
                                    # Show Gemini model used
                                    if cw_model == "Gemini":
                                        active_m = st.session_state.get("_gemini_active_model", "")
                                        if active_m: st.toast(f"✅ Used: {active_m.replace('models/', '')}")
                                    st.rerun()
                                else:
                                    st.error("Failed to parse response")
                                    st.code(json_txt[:1000])
                            else:
                                st.error(err)
            
            # --- Preview & Edit Results ---
            if st.session_state.colwriter_result:
                d = st.session_state.colwriter_result
                st.divider()
                st.subheader("📋 Preview & Edit")
                
                # Show model info
                active_gemini = st.session_state.get("_gemini_active_model", "")
                if cw_model == "Gemini" and active_gemini:
                    st.caption(f"🤖 Generated by: **{active_gemini.replace('models/', '')}**")
                elif cw_model in CLAUDE_MODELS:
                    st.caption(f"🤖 Generated by: **{CLAUDE_MODELS[cw_model]}**")
                elif cw_model in OPENAI_MODELS:
                    st.caption(f"🤖 Generated by: **{OPENAI_MODELS[cw_model]}**")
                
                # Editable fields
                edited_title = st.text_input("**Collection Title (H1):**",
                                              value=d.get("collection_title", ""), key="cw_edit_title")
                
                edited_meta_title = st.text_input(f"**Meta Title** ({len(d.get('meta_title', ''))} chars):",
                                                    value=d.get("meta_title", ""), key="cw_edit_meta_title")
                if len(edited_meta_title) > 60:
                    st.warning(f"⚠️ Meta title is {len(edited_meta_title)} chars (recommended: under 60)")
                
                edited_meta_desc = st.text_area(f"**Meta Description** ({len(d.get('meta_description', ''))} chars):",
                                                  value=d.get("meta_description", ""), height=80, key="cw_edit_meta_desc")
                if len(edited_meta_desc) > 155:
                    st.warning(f"⚠️ Meta description is {len(edited_meta_desc)} chars (recommended: under 155)")
                
                edited_html = st.text_area("**Collection Description (HTML):**",
                                            value=d.get("collection_description_html", ""), height=300, key="cw_edit_html")
                
                # HTML Preview
                with st.expander("🔍 HTML Preview", expanded=True):
                    st.markdown(edited_html, unsafe_allow_html=True)
                
                # Keyword analysis
                if d.get("keyword_analysis"):
                    with st.expander("🔍 Keyword Analysis"):
                        st.write(d.get("keyword_analysis", ""))
                
                # Word count
                import re as _re
                clean_text = _re.sub(r'<[^>]+>', '', edited_html)
                word_count = len(clean_text.split())
                if 150 <= word_count <= 300:
                    st.caption(f"📝 Word count: **{word_count}** ✅ (target: 150-300)")
                else:
                    st.caption(f"📝 Word count: **{word_count}** ⚠️ (target: 150-300)")
                
                st.divider()
                
                # --- Update to Shopify ---
                st.subheader("☁️ Update to Shopify")
                st.caption(f"Collection: **{d.get('_col_handle', '')}** (ID: {d.get('_col_id', '')})")
                
                if st.button("☁️ Update Collection on Shopify", type="primary", key="cw_update_btn"):
                    # Build updated data from edited fields
                    update_data = {
                        "collection_title": edited_title,
                        "collection_description_html": edited_html,
                        "meta_title": edited_meta_title,
                        "meta_description": edited_meta_desc,
                    }
                    with st.spinner("Updating collection on Shopify..."):
                        ok, msg = update_shopify_collection(
                            cw_shop, cw_token,
                            d["_col_id"], update_data,
                            collection_type=d.get("_col_type", "custom")
                        )
                        if ok:
                            st.success(msg)
                            st.balloons()
                        else:
                            st.error(msg)
        else:
            st.info("👆 Click **Load Collections** to get started.")

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
        st.write("🔹 **Claude Sonnet 4.6** - Anthropic (Best value, near-Opus performance)")
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














