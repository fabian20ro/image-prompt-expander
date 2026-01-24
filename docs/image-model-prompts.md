# Image Model Prompt Optimization

This document covers prompt optimization for different image generation models.

## FLUX.2 Klein (4B / 9B)

### Overview
FLUX.2 Klein is a smaller, faster variant of FLUX optimized for quick generation. It prefers natural language prompts over keyword lists.

### Prompt Structure
```
Subject → Setting → Details → Lighting → Atmosphere
```

Front-load important elements - the model prioritizes earlier content.

### Best Practices

**DO:**
- Write descriptive prose, complete sentences
- Describe lighting explicitly (source, quality, direction, temperature)
- Include atmosphere/mood at the end
- Be specific about materials, textures, colors

**DON'T:**
- Use comma-separated keyword lists
- Include filler text that doesn't add visual info
- Rely on style tags alone

### Example Prompts

**Good:**
> A ginger tabby cat curled contentedly on a worn velvet armchair. The cat's eyes are half-closed in peaceful slumber, one paw tucked beneath its chin. A crackling fireplace casts warm amber light across the scene, creating soft shadows on the faded floral wallpaper. The atmosphere is intimate and tranquil, evoking a sense of quiet domestic comfort.

**Bad:**
> cat, cozy, armchair, fireplace, warm lighting, peaceful, 4k, detailed, masterpiece

### Lighting Keywords That Work Well
- "Golden hour sunlight bathes..."
- "Soft diffused light from..."
- "Dramatic backlighting creates..."
- "Pale moonlight illuminates..."
- "Warm amber glow from..."

### Model-Specific Notes

**4B variant:**
- Faster but less detailed
- Keep prompts slightly shorter (2-3 sentences)
- Focus on main subject, less background detail

**9B variant:**
- More detailed output
- Can handle longer prompts (3-4 sentences)
- Better at complex scenes and multiple subjects

---

## Z-Image-Turbo

### Overview
Fast generation model optimized for speed. Handles both prose and semi-structured prompts.

### Prompt Structure
More flexible than FLUX - can use either:
1. Natural prose (like FLUX)
2. Structured format with categories

### Structured Format Example
```
Subject: A majestic dragon with emerald scales
Setting: Flying over snow-capped mountain peaks
Lighting: Golden sunset light, long shadows
Style: Fantasy illustration, detailed, epic scale
```

### Best Practices

**DO:**
- Keep prompts concise (1-2 sentences for turbo)
- Include style descriptors
- Specify aspect ratio if important

**DON'T:**
- Overload with detail (turbo prioritizes speed)
- Use conflicting style terms

### Example Prompts

**Good:**
> An emerald dragon soars over snow-capped peaks at sunset. Golden light illuminates its massive wings. Epic fantasy style, highly detailed.

**Good (structured):**
> Subject: Dragon flying over mountains
> Style: Fantasy art, dramatic lighting, detailed scales
> Mood: Epic, majestic

---

## Comparison Table

| Aspect | FLUX.2 Klein | Z-Image-Turbo |
|--------|--------------|---------------|
| Prompt Style | Prose preferred | Prose or structured |
| Optimal Length | 2-4 sentences | 1-2 sentences |
| Lighting | Explicit description | Keywords work |
| Speed | Medium | Fast |
| Detail Level | High (9B) / Medium (4B) | Medium |

---

## General Tips (All Models)

### Lighting Matters
All models respond strongly to lighting descriptions. Include:
- **Source**: sun, moon, fire, lamp, window
- **Quality**: soft, harsh, diffused, dramatic
- **Direction**: backlighting, side-lit, overhead
- **Temperature**: warm, cool, golden, blue

### Avoid Contradictions
- Don't mix incompatible times of day ("sunset" + "harsh midday sun")
- Keep style consistent ("photorealistic" + "anime style" = confused output)
- Weather should match lighting

### Subject Placement
- Main subject should appear early in prompt
- Background/setting comes after subject
- Atmosphere/mood is last

### Testing Prompts
1. Generate small batch (10-20) first
2. Check for consistent interpretation
3. Identify which elements are being ignored
4. Refine and regenerate
