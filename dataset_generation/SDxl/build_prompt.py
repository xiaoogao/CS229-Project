import os
import random
import pandas as pd

# 职业列表（每个职业生成多样 prompt，无需关键词组合）
professions = [
    "doctor", "nurse", "teacher", "lawyer", "engineer", "hairdresser", "baker", "waiter",
    "bartender", "architect", "accountant", "paralegal", "veterinarian", "painter", "broker",
    "clerk", "counsellor", "manager", "physician"
]

# 每类生成数量
images_per_class = 200
save_dir = "./SDxl_data"

# 多样化 prompt 模板
prompt_templates = [
    "A realistic photo of a person working as a {profession}, ultra realistic, 4k, natural lighting, DSLR, professional portrait",
    "A professional person doing the job of a {profession}, full body, realistic, shallow depth of field, photo style",
    "A headshot of a person who is a {profession}, sitting at a desk, DSLR photo, soft lighting, ultra detailed",
    "A {profession} in their workplace, portrayed as a real person, Canon DSLR shot, realistic lighting",
    "Close-up portrait of a human {profession}, studio lighting, high resolution, expressive face",
    "Documentary-style photo of a person working as a {profession}, shallow DOF, realistic",
    "A confident person who is a {profession} standing in a modern office, natural daylight, DSLR, 4k",
    "Portrait of a {profession}, clearly human, half-body, natural background, photojournalistic style",
    "Cinematic photo of a human {profession}, shallow depth of field, realistic lighting, DSLR",
    "A {profession} in a candid moment, side profile, bokeh background, natural light, professional portrait"
]

profession_scene_map = {
    "doctor": ["treating a patient in a hospital", "holding a stethoscope", "wearing a white coat"],
    "nurse": ["adjusting an IV", "taking notes beside a hospital bed", "smiling warmly in scrubs"],
    "teacher": ["writing on a chalkboard", "interacting with students", "teaching in a classroom"],
    "lawyer": ["speaking in a courtroom", "reading legal documents", "sitting at a legal desk"],
    "engineer": ["looking at blueprints", "standing at a construction site", "wearing a hard hat"],
    "hairdresser": ["cutting hair in a salon", "styling a client", "using scissors in front of a mirror"],
    "baker": ["kneading dough", "placing pastries in the oven", "working in a bakery"],
    "waiter": ["serving food at a restaurant", "holding a tray with drinks", "setting a dining table"],
    "bartender": ["mixing cocktails", "cleaning glasses", "working behind the bar"],
    "architect": ["presenting building plans", "discussing models with colleagues", "sketching designs"],
    "accountant": ["typing at a computer", "reviewing spreadsheets", "working in an office cubicle"],
    "paralegal": ["organizing case files", "reading legal books", "assisting a lawyer"],
    "veterinarian": ["examining a dog", "in a clinic with pets", "holding a cat in a vet office"],
    "painter": ["painting on a canvas", "mixing colors on a palette", "working in an art studio"],
    "broker": ["talking on the phone", "monitoring stock charts", "working in a financial office"],
    "clerk": ["sorting files", "working at a front desk", "checking documents"],
    "counsellor": ["talking to a client", "taking notes in a therapy session", "sitting in a quiet office"],
    "manager": ["leading a team meeting", "reviewing project plans", "standing at a whiteboard"],
    "physician": ["holding a clipboard", "talking to a patient", "wearing a stethoscope"]
}

# 构造所有样本
output = []
for profession in professions:
    scenes = profession_scene_map.get(profession, [])
    for i in range(images_per_class):
        template = random.choice(prompt_templates)
        prompt = template.format(profession=profession)
        
        # 随机选一个场景描述（如有）
        if scenes:
            prompt += f", {random.choice(scenes)}"

        filepath = os.path.join(save_dir, profession, f"{i+1:08d}.png")
        output.append({
            "profession": profession,
            "prompt": prompt,
            "filepath": filepath
        })

df = pd.DataFrame(output)
df.to_csv("sdxl_prompts.csv", index=False)
