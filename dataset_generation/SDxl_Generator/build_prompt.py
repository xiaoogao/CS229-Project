import os
import random
import pandas as pd

professions = [
    "doctor", "nurse", "teacher", "lawyer", "engineer", "hairdresser", "baker", "waiter", "veterinarian", "painter",
]

images_per_class = 500
save_dir = "../SDxl_data"

# Prompt templates now include a {gender_prompt} placeholder
prompt_templates = [
    "A realistic photo of a {gender_prompt} person working as a {profession}, ultra realistic, 4k, natural lighting, DSLR, professional portrait",
    "A professional {gender_prompt} person doing the job of a {profession}, full body, realistic, shallow depth of field, photo style",
    "A headshot of a {gender_prompt} person who is a {profession}, sitting at a desk, DSLR photo, soft lighting, ultra detailed",
    "A {gender_prompt} {profession} in their workplace, portrayed as a real person, Canon DSLR shot, realistic lighting",
    "Close-up portrait of a {gender_prompt} human {profession}, studio lighting, high resolution, expressive face",
    "Documentary-style photo of a {gender_prompt} person working as a {profession}, shallow DOF, realistic",
    "A confident {gender_prompt} person who is a {profession} standing in a modern office, natural daylight, DSLR, 4k",
    "Portrait of a {gender_prompt} {profession}, clearly human, half-body, natural background, photojournalistic style",
    "Cinematic photo of a {gender_prompt} human {profession}, shallow depth of field, realistic lighting, DSLR",
    "A {gender_prompt} {profession} in a candid moment, side profile, bokeh background, natural light, professional portrait"
    "A {gender_prompt} {profession} interacting with clients, candid real-life moment, documentary style, natural expression, busy environment",
    "Outdoor portrait of a {gender_prompt} {profession} during work, sunlight, realistic skin texture, city background",
    "A close-up of a {gender_prompt} {profession} focusing on their task, hands visible, shallow depth of field, 8k clarity",
    "An emotional photo of a {gender_prompt} person as a {profession}, expressive face, realistic tears or laughter, magazine style",
    "A {gender_prompt} {profession} using professional equipment, detailed workplace, cinematic lighting, ultra-realistic",
    "Fashion-inspired photo of a {gender_prompt} {profession}, stylish outfit, high contrast, editorial photography",
    "A {gender_prompt} {profession} teaching, guiding, or helping others, warm colors, friendly atmosphere, high quality DSLR",
    "Wide-angle shot of a {gender_prompt} {profession} in action, dynamic movement, blurred background, photojournalism style",
    "Black and white portrait of a {gender_prompt} {profession}, classic style, soft light, deep focus",
    "Evening indoor photo of a {gender_prompt} {profession}, warm ambient lighting, realistic shadows, storytelling atmosphere",
]

gender_prompts = ["male, man", "female, woman"]

profession_scene_map = {
    "doctor": [
        "treating a patient in a hospital",
        "holding a stethoscope",
        "wearing a white coat",
        "discussing medical results with colleagues",
        "writing medical notes at a desk",
        "examining a patient's X-ray",
        "standing in a hospital hallway",
        "checking a patient's heartbeat with a stethoscope"
    ],
    "nurse": [
        "adjusting an IV drip",
        "taking notes beside a hospital bed",
        "smiling warmly in scrubs",
        "preparing medication in a hospital room",
        "assisting a doctor during a procedure",
        "comforting a patient",
        "pushing a wheelchair",
        "measuring a patient's blood pressure"
    ],
    "teacher": [
        "writing on a chalkboard",
        "interacting with students",
        "teaching in a classroom",
        "grading papers at a desk",
        "reading a book to children",
        "giving a presentation with a projector",
        "supervising a group activity",
        "explaining a concept to a student one-on-one"
    ],
    "lawyer": [
        "speaking in a courtroom",
        "reading legal documents",
        "sitting at a legal desk",
        "consulting with a client",
        "presenting evidence in court",
        "reviewing files in an office",
        "making a phone call in a suit",
        "working late with piles of case files"
    ],
    "engineer": [
        "looking at blueprints",
        "standing at a construction site",
        "wearing a hard hat",
        "inspecting machinery",
        "working on a computer with CAD software",
        "presenting a technical drawing in a meeting",
        "measuring materials with a ruler",
        "discussing a project with colleagues outdoors"
    ],
    "hairdresser": [
        "cutting hair in a salon",
        "styling a client's hair",
        "using scissors in front of a mirror",
        "blow-drying hair",
        "washing a client's hair at a sink",
        "showing a hairstyle to a customer",
        "organizing hair products on a shelf",
        "sweeping the salon floor"
    ],
    "baker": [
        "kneading dough",
        "placing pastries in the oven",
        "working in a bakery",
        "decorating a cake with icing",
        "holding a tray of fresh bread",
        "measuring flour on a kitchen scale",
        "arranging baked goods in a display case",
        "mixing ingredients in a large bowl"
    ],
    "waiter": [
        "serving food at a restaurant",
        "holding a tray with drinks",
        "setting a dining table",
        "taking an order with a notepad",
        "pouring water for customers",
        "clearing dishes from a table",
        "smiling at guests while delivering meals",
        "arranging silverware and napkins"
    ],
    "veterinarian": [
        "examining a dog",
        "in a clinic with pets",
        "holding a cat in a vet office",
        "administering a vaccine to a pet",
        "consulting with a pet owner",
        "checking a rabbit's heartbeat",
        "standing next to animal cages",
        "writing notes on a clipboard"
    ],
    "painter": [
        "painting on a canvas",
        "mixing colors on a palette",
        "working in an art studio",
        "displaying finished artwork on a wall",
        "painting a mural on a large surface",
        "sketching with charcoal",
        "cleaning brushes in a jar of water",
        "teaching a painting class"
    ],
}

output = []
for profession in professions:
    scenes = profession_scene_map.get(profession, [])
    for i in range(images_per_class):
        # Randomly assign gender: 0 for male, 1 for female
        gender_idx = random.randint(0, 1)
        gender = "male" if gender_idx == 0 else "female"
        gender_prompt = gender_prompts[gender_idx]
        
        # Insert gender information into prompt template
        template = random.choice(prompt_templates)
        prompt = template.format(profession=profession, gender_prompt=gender_prompt)
        
        # Optionally add a scene description
        if scenes:
            prompt += f", {random.choice(scenes)}"
        
        # Filename: 0_xxx.png for male, 1_xxx.png for female
        filename = f"{gender_idx}_{i+1:08d}.png"
        filepath = os.path.join(save_dir, profession, filename)
        
        # Save relevant metadata
        output.append({
            "profession": profession,
            "gender": gender,
            "prompt": prompt,
            "filepath": filepath
        })

# Write results to CSV for later use
df = pd.DataFrame(output)
df.to_csv("sdxl_prompts.csv", index=False)
