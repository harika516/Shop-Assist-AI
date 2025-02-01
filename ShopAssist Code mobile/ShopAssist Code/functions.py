import openai
import ast
import re
import pandas as pd
import json

# Ensure you have the latest openai library installed
# You can upgrade it using:
# pip install --upgrade openai

def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"
    example_user_req = {'storage capacity': 'high','camera quality': 'high','Performance and speed': 'low','battery life': 'high','display quality': 'high','Budget': '150000'}

    system_message = f"""

    You are an intelligent mobile gadget expert and your goal is to find the best mobile for a user.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('storage capacity','camera quality','Performance and speed','battery life','display quality','Budget') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this {{'storage capacity ': 'values','camera quality': 'values','Performance and speed': 'values','battery life': 'values','display quality': 'values','Budget': 'values'}}
    The values for all keys, except 'budget', should be 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    The value for 'budget' should be a numerical value extracted from the user's response.
    The values currently in the dictionary are only representative values.

    {delimiter}Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
    - The values for all keys, except 'Budget', should strictly be either 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    - The value for 'budget' should be a numerical value extracted from the user's response.
    - 'Budget' value needs to be greater than or equal to 15000 INR. If the user says less than that, please mention that there are no Mobiles in that range.
    - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    {delimiter} Thought 1: Ask a question to understand the user's profile and requirements. \n
    If their primary use for the laptop is unclear. Ask another question to comprehend their needs.
    You are trying to fill the values of all the keys ('storage capacity','camera quality','Performance and speed','battery life','display quality','Budget') in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, proceed to the next step. Otherwise, rephrase the question to capture their profile. \n{delimiter}

    {delimiter}Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

    {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions. {delimiter}

    Follow the above chain of thoughts and only output the final updated python dictionary. \n


    {delimiter} Here is 2 sample conversations between the user and assistant:


User: Hi, I'm looking for a new smartphone.
Assistant: Great! Let's find the perfect smartphone for you. Can you tell me a bit about what you're looking for in a smartphone?
User: I need a smartphone that can handle demanding tasks and has a high-quality display for better visuals. I also use a lot of graphics-intensive applications like gaming and video streaming.
Assistant: Understood. Could you tell me more about the specific tasks you do on your smartphone? Do you mainly focus on gaming, video streaming, or both?
User: I primarily use my smartphone for gaming, but I also watch a lot of videos and stream content.
Assistant: Got it. Gaming and streaming require a smartphone with a powerful processor and a high-resolution display for an immersive experience. Do you have any preferences in terms of brand or operating system?
User: I'm open to different brands, but I prefer smartphones with a clean and user-friendly interface.
Assistant: That's helpful to know. Do you have any specific gaming requirements, such as support for high-framerate gaming or gaming accessories compatibility?
User: Yes, I'm interested in high-framerate gaming and compatibility with gaming accessories like controllers.
Assistant:"Could you kindly let me know your budget for the mobile? This will help me find options that fit within your price range while meeting the specified requirements."
    User: "my max budget is 50,000  inr"
    Assistant: "{example_user_req}"
{delimiter}

User: Hi, I'm interested in buying a new smartphone.
Assistant: Sure thing! Let's find the perfect smartphone for you. Can you share a bit about what you're looking for in terms of features and functionality?
User: I need a smartphone that can have good batery life  and has a great camera for photography.
Assistant: Got it. Are there any specific tasks or activities you'll be using the smartphone for? For example, are you into photography, gaming, or productivity apps?
User: I mainly use my smartphone for photography and social media, but I also enjoy gaming occasionally.
Assistant: That's helpful to know. For photography, a smartphone with a high-resolution camera, advanced features like optical image stabilization, and good low-light performance would be ideal. And for gaming, you'd want a device with a powerful processor and a high-refresh-rate display. Do you have a preferred operating system or brand?
User: I'm open to both iOS and Android, but I've had good experiences with Apple brand in the past.
Assistant: Great! We'll explore options from both operating systems and consider Apple brand smartphones. Regarding your budget, do you have a specific price range in mind?
User: I'm willing to invest in a high-quality smartphone, so my budget is flexible. I'd say around 75K would be my maximum.
Assistant: "{example_user_req}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation

def get_chat_model_completions(messages):
    response = openai.chat.completions.create(
  model="gpt-3.5-turbo",
    messages=messages,
    temperature=0,
    max_tokens=300
)

    return response.choices[0].message.content
def moderation_check(user_input):
    # Call the moderation API with the new model
    response = openai.moderations.create(
        model="omni-moderation-latest",
        input=user_input
    )
    
    # Extract the first result from the response
    moderation_output = response.results[0]
    
    # Check if the content is flagged
    if moderation_output.flagged:
        return "Flagged"
    else:
        return "Not Flagged"



import openai

def intent_confirmation_layer(response_assistant):
    prompt = f"""
    You are a senior evaluator who has an eye for detail.
    You are provided an input dictionary. You need to evaluate if the dictionary contains the following keys:
    'storage capacity', 'camera quality', 'performance and speed', 'battery life', 'display quality', and 'budget'.

    Next, check if the values for all keys are correctly filled:
    - The values for 'storage capacity', 'camera quality', 'performance and speed', 'battery life', and 'display quality' should be one of: 'low', 'medium', or 'high'.
    - The value for 'budget' should be a numeric string (e.g., '80000').

    If the dictionary meets these conditions, output 'Yes'. Otherwise, output 'No'.

    Here is the input: {response_assistant}
    Only output a one-word string: Yes or No.
    """
    #print("Prompt Sent to LLM:\n", prompt)  # Debugging print

    messages = [
        {"role": "system", "content": "You are a strict evaluator checking dictionary correctness."},
        {"role": "user", "content": prompt}
    ]
    
    confirmation = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=5  # We only expect "Yes" or "No"
    )
    
    confirmation_text = confirmation.choices[0].message.content.strip()
    print("LLM Output:", confirmation_text)  # Debugging print
    
    return "Yes" if confirmation_text.lower() == "yes" else "No"



def dictionary_present(response):
    delimiter = "####"
    user_req = {'storage capacity': 'medium','camera quality': 'high','Performance and speed': 'high','battery life': 'medium','display quality': 'high','Budget': '50000'}
    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract and return only the python dictionary from the input.
            The output should match the format as {user_req}.
            The output should contain the exact keys and values as present in the input.

            Here are some sample input output pairs for better understanding:
            {delimiter}
            input: - storage capacity: low - camera quality: high - Performance and speed: high - battery life: low - display quality: medium - Budget: 20,000 INR
            output: {{'storage capacity': 'low','camera quality': 'high','Performance and speed': 'high','battery life': 'low','display quality': 'medium','Budget': '20000'}}

            input: {{'storage capacity':     'low','camera quality':      'high','Performance and speed':     'high','battery life':     'low','display quality':
               'medium','Budget':     '90000'}}
            output: {{'storage capacity': 'low','camera quality': 'high','Performance and speed': 'high','battery life': 'low','display quality': 'medium','Budget': '90000'}}

            input: Here is your user profile 'storage capacity': 'low','camera quality': 'high','Performance and speed': 'high','battery life': 'low','display quality': 'medium','Budget': '35000'
            output: {{'storage capacity': 'low','camera quality': 'high','Performance and speed': 'high','battery life': 'low','display quality': 'medium','Budget': '35000'}}
            {delimiter}

            task: following is the input text you want to extract the python dictionary from it.

            {response}

            your task is to find the python dictionary from the input text.
            expected output: output should only have the python dictionary from the given text. please have a look in to the provided input , output  examples
      for your reference.

      {delimiter}
      Guidelines: Do not give any coding to extract the dictionary from text. I want only the dictionary with key, value pairs.
       Do not give any extra text bedore and after the dictionary with key-value pairs.
       Do not use coding to extract the dictionary from text.
       {delimiter}
       expected output:  extracted dictionary like string
            {delimiter}
            """
    response_completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000,
        temperature=0.0
    )

    

    # Extract and return the content
    dictionary_content = response_completion.choices[0].message.content.strip()
    return dictionary_content


def extract_dictionary_from_string(string):
    regex_pattern = r"\{[^{}]+\}"

    dictionary_matches = re.findall(regex_pattern, string)

    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        dictionary_string = dictionary_string.lower()

        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        try:
            dictionary = ast.literal_eval(dictionary_string)
            return dictionary
        except (ValueError, SyntaxError):
            # Handle cases where the dictionary string is not properly formatted
            return {}
    else:
        return {}

def getTop3(filtered_mobiles, user_requirements):
    mappings = {
        'low': 0,
        'medium': 1,
        'high': 2
    }

    filtered_mobiles['Score'] = 0

    for index, row in filtered_mobiles.iterrows():
        mobile_values = row['mobile_feature']
        score = 0

        for key, user_value in user_requirements.items():
            if key.lower() == 'budget':
                continue

            mobile_value = mobile_values.get(key.lower(), None)
            if mobile_value is not None:
                mobile_mapping = mappings.get(mobile_value.lower(), -1)
                user_mapping = mappings.get(user_value.lower(), -1)

                if mobile_mapping >= user_mapping:
                    score += 1

        filtered_mobiles.loc[index, 'Score'] = score

    top_mobiles = filtered_mobiles.sort_values('Score', ascending=False).head(3)
    return top_mobiles.to_json(orient='records')

def compare_laptops_with_user(user_req_string):
   
    user_requirements=user_req_string
    print("user requirements",user_requirements)
    print(type(user_requirements))
    #print(user_requirements )
    budget = int(user_requirements.get('budget', '0').replace(',', '').split()[0])
    print("budget is ",budget)
   
    df=pd.read_excel("mobile_dataNew.xlsx")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df['mobile_feature'] = df['mobile_feature'].apply(lambda x: extract_dictionary_from_string(x))
    filtered_mobiles = df.copy()
    #filtered_mobiles['Price(INR)'] = filtered_mobiles['Price(INR)'].str.replace(',','').astype(int)
    filtered_mobiles = filtered_mobiles[filtered_mobiles['Price(INR)'] <= budget].copy()
   
    mappings = {
        'low': 0,
        'medium': 1,
        'high': 2
    }
    # Create 'Score' column in the DataFrame and initialize to 0
    filtered_mobiles['Score'] = 0
    #getTop3(filtered_mobiles, user_requirements)
    top_mobiles = getTop3(filtered_mobiles, user_requirements)
    #print(top_mobiles)
    return top_mobiles

def extract_dictionary_from_string(string):
    regex_pattern = r"\{[^{}]+\}"

    dictionary_matches = re.findall(regex_pattern, string)

    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        dictionary_string = dictionary_string.lower()

        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        dictionary = ast.literal_eval(dictionary_string)
    return dictionary

'''def compare_laptops_with_user(user_req_string):
    df = pd.read_excel('ShopAssist Code mobile\ShopAssist Code\mobile_dataNew.xlsx')
    user_requirements = extract_dictionary_from_string(user_req_string)
    budget_str = user_requirements.get('budget', '0')
    
    # Extract numerical budget value
    budget_match = re.search(r'\d+', budget_str.replace(',', ''))
    budget = int(budget_match.group()) if budget_match else 0

    # Ensure budget is at least 25000 INR
    if budget < 25000:
        return json.dumps([])  # No laptops available in this range

    filtered_laptops = laptop_df.copy()
    filtered_laptops['Price'] = filtered_laptops['Price'].str.replace(',', '').astype(int)
    filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= budget].copy()

    mappings = {
        'low': 0,
        'medium': 1,
        'high': 2
    }
    # Create 'Score' column in the DataFrame and initialize to 0
    filtered_laptops['Score'] = 0
    for index, row in filtered_laptops.iterrows():
        user_product_match_str = row['laptop_feature']
        laptop_values = extract_dictionary_from_string(user_product_match_str)
        score = 0

        for key, user_value in user_requirements.items():
            if key.lower() == 'budget':
                continue  # Skip budget comparison
            laptop_value = laptop_values.get(key, None)
            if laptop_value is None:
                continue  # Skip if the key is not present
            laptop_mapping = mappings.get(laptop_value.lower(), -1)
            user_mapping = mappings.get(user_value.lower(), -1)
            if laptop_mapping >= user_mapping:
                # If the laptop value is greater than or equal to the user value, increment the score by 1
                score += 1

        filtered_laptops.loc[index, 'Score'] = score

    # Sort the laptops by score in descending order and return the top 3 products
    top_laptops = filtered_laptops.drop('laptop_feature', axis=1)
    top_laptops = top_laptops.sort_values('Score', ascending=False).head(3)

    return top_laptops.to_json(orient='records')'''


def recommendation_validation(laptop_recommendation):
    try:
        data = json.loads(laptop_recommendation)
    except json.JSONDecodeError:
        return []

    validated_reco = [item for item in data if item.get('Score', 0) > 2]
    return validated_reco


def initialize_conv_reco(products):
    if not products:
        system_message = "There are no mobiles that match your requirements."
    else:
        system_message = f"""
        You are an intelligent mobile gadget expert and you are tasked with the objective to \
        solve the user queries about any product from the catalogue: {products}.\
        You should keep the user profile in mind while answering the questions.\
    
        Start with a brief summary of each mobile in the following format, in decreasing order of price of mobiles:
        1. <mobile Name> : <Major specifications of the mobile>, <Price in Rs>
        2. <mobile Name> : <Major specifications of the mobile>, <Price in Rs>
        """
    conversation = [{"role": "system", "content": system_message}]
    return conversation
