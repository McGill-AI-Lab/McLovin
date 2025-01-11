from random import randint
import random
import json
import JSON_manipulation as Jm
import pandas as pd
import time
import os
import google.generativeai as genai

#Api Holder
from dotenv import load_dotenv

import pandas.errors

LIM_DAY_API = 1300
LIM_MIN_API = 15

hobbies_JSON = 'utils/ProfileGenerator/json_prompts/hobbies.json'
phys_traits_JSON = 'utils/ProfileGenerator/json_prompts/phys_traits.json'
pers_traits_JSON = 'utils/ProfileGenerator/json_prompts/pers_traits.json'

dict_data = {
    "gender": [],
    "major": [],
    "hobbies": [],
    "Attractive phys traits": [],
    "Attractive pers traits": [],
    "bios": []
}

majors = [
    "Business and Economics",
    "Education",
    "Engineering",
    "Environment",
    "Health Sciences",
    "Humanities and Social Sciences",
    "Languages",
    "Math, Computing and technology",
    "Music",
    "Politics and Law",
    "Pure and Applied sciences"
]


with open("utils/ProfileGenerator/json_prompts/prompts.json") as file:

    #Getting each prompt as an f string
    data = json.load(file)
    instruction_prompts = [f'{prompt}' for prompt in data['prompts']]
    prompts_count = len(data['prompts'])


def major_generator(lst):
    """
    list -> string
    Generates a random major for a user based on the list of majors provided
    """
    major = lst[random.randint(0, len(majors) - 1)]
    dict_data['major'].append(major)
    return major


def hobby_generator(major):
    """
    None -> string
    Generates a string of hobbies which will be associated to a user
    """
    hobby_num = randint(1, 6)
    user_hobbies = []
    i = 1
    while i <= hobby_num:
        with open(hobbies_JSON) as file:
            data = json.load(file)

            # less than 4 gets specific hobby, greater than 4 generates general hobby
            x = randint(0, 10)
            if x <= 4:
                hobby = data[major][randint(0, len(data[major]) - 1)]
            else:
                hobby = data['General'][randint(0, len(data['General']) - 1)]

            user_hobbies.append(hobby)
            i += 1
    hobby_prompt = ", ".join(user_hobbies)
    dict_data['hobbies'].append(hobby_prompt)

    return hobby_prompt


def gender_generator():
    genders = ["female", "male"]
    probabilities = [0.49, 0.48]
    selected_gender = random.choices(genders, probabilities, k=1)
    dict_data['gender'].append(selected_gender[0])

    return selected_gender[0]


'''
def gender_of_interest(self_gender):
    sexualities = ["heterosexual", "homosexual"]
    probabilities = [0.9, 0.04]

    selected_sexuality = random.choices(sexualities, probabilities, k=1)

    if selected_sexuality[0] == "heterosexual":
        GOI = 'female' if self_gender == "male" else "male"

    elif selected_sexuality[0] == "homosexual":
        GOI = self_gender

    #elif selected_sexuality == "bisexual":
        #GOI = self_gender, 'female' if self_gender == "male" else "female"

    #else:
        #GOI = None

    dict_data['Attracted to'].append(GOI)

    return GOI
'''

def phys_trait_generator(self_gender):
    trait_num = randint(1, 3)
    self_traits = []
    i = 1
    while i <= trait_num:
        with open(phys_traits_JSON) as file:
            data = json.load(file)
            trait = data[self_gender][randint(0, len(data[self_gender]) - 1)]
            self_traits.append(trait)
        i+=1

    str_phys_traits = ', '.join(self_traits)
    dict_data['Attractive phys traits'].append(str_phys_traits)
    return str_phys_traits


def pers_trait_generator(self_gender):
    trait_num = randint(1, 3)
    self_traits = []
    i = 1
    while i <= trait_num:
        with open(pers_traits_JSON) as file:
            data = json.load(file)
            trait = data[self_gender][randint(0, len(data[self_gender]) - 1)]
            self_traits.append(trait)
        i+=1

    str_pers_traits = ', '.join(self_traits)
    dict_data['Attractive pers traits'].append(str_pers_traits)
    return str_pers_traits

def generate_bio(major, gender, hobby_prompt, phys_trait, pers_trait):

    #Formatting the instructions for the text generator
    instructions = f'{instruction_prompts[randint(0,prompts_count - 1)]}'
    formatted_instructions = instructions.format(gender=gender,
                                                 major=major,
                                                 hobby_prompt=hobby_prompt,
                                                 phys_trait=phys_trait,
                                                 pers_trait=pers_trait
                                                 )

    #Setting up the text generator
    load_dotenv()
    API = os.getenv("GEMINI_KEY")
    genai.configure(api_key=API)
    GenModel = genai.GenerativeModel("gemini-1.5-flash", system_instruction=formatted_instructions)

    #Generating bios
    print(f'Bio for {major} major')
    response = GenModel.generate_content("Write one bio",
                                        safety_settings = {
                                            'HATE': 'BLOCK_NONE',
                                            'HARASSMENT': 'BLOCK_NONE',
                                            'SEXUAL': 'BLOCK_NONE', # ooookay dory
                                            'DANGEROUS': 'BLOCK_NONE'
                                        },
                                        generation_config=genai.types.GenerationConfig(
                                            max_output_tokens = 100,
                                            temperature= 1.0,
                                        ),
                                        )

    formatted_response = str(response.text).rsplit('.',1)[0] + '.'
    print(formatted_response)
    dict_data['bios'].append(formatted_response)

def main():
    # limited at 15 calls per minute
    start_time = time.time()
    Jm.reset_count()

    while Jm.load_count() < LIM_DAY_API:

        major = major_generator(majors)
        hobbies = hobby_generator(major)
        gender = gender_generator()
        #GOI = gender_of_interest(gender)
        phys_traits = phys_trait_generator(gender)
        pers_traits = pers_trait_generator(gender)

        generate_bio(major, gender, hobbies, phys_traits, pers_traits)

        #saving after every 10 iterations
        if Jm.load_count()%10 == 0 and Jm.load_count() != 0:
            print("Saving 10 iterations")
            csv_file = 'profiles.csv'
            df = pd.DataFrame(dict_data)
            if os.path.exists(csv_file):
                try:
                    existing_df = pd.read_csv(csv_file)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.drop_duplicates(inplace= True)
                    combined_df.to_csv(csv_file, index=False)

                except pandas.errors.ParserError:
                    print(f"Error parsing {csv_file}")

            else:
                df = pd.DataFrame(dict_data)
                df.to_csv(csv_file, index=False)

        Jm.increment_count()

        # maximum of 15 api calls per minute
        if Jm.load_count() % LIM_MIN_API == 0 and Jm.load_count() != 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                print("15 iterations done. Waiting for next minute")
                time.sleep(60 - elapsed_time)
            else:
                pass
            start_time = time.time()

main()
