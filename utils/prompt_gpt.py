import requests
import os
import numpy as np
import base64
from utils.api_key import api_key

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}


def get_answer(content):
    return int(content.split('Best Result:')[-1].split('Confidence:')[0].strip(' :*.'))

def get_view(content):
    return int(content.split('Best View:')[-1].strip(' :*.'))

def get_stage(content):
    part = content.split('Current Stage:')[-1]
    stage = part.strip().splitlines()[0]
    return int(stage)

def get_grasp(content):
    answer = content.split('Grasp:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_release(content):
    answer = content.split('Release:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_success(content):
    answer = content.split('Satisfied:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_close_gripper(content):
    answer = content.split('Keep Gripper Closed:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_subgoals(content):
    print(content)

    subgoal_list = []
    goal_id = 0
    content = content.split('Sub Goals:')[-1]
    while True:
        goal_id += 1
        goal = content.split(f'{goal_id}.')[-1].split(f'{goal_id + 1}.')[0].strip(' :*.\n"')
        subgoal_list.append(goal)
        if f'{goal_id + 1}.' not in content:
            break

    return subgoal_list

def get_names(content):
    print(content)

    content = content.split('Objects:')[-1]

    name_list = []
    name_id = 1
    while True:
        if f'{name_id}.' not in content:
            break
        name = content.split(f'{name_id}.')[-1].split(f'{name_id + 1}.')[0].strip(' :*.\n"')
        name_list.append(name)
        name_id += 1

    return name_list

def get_description_list(content, num_results):
    description_list = []
    for idx in range(1, num_results + 1):
        start = content.find(f'Description {idx}:')
        end = content.find(f'Description {idx + 1}:')
        if idx == num_results:
            end = len(content)
        if start == -1 or end == -1:
            print('Description not found')
            return None
        description = content[start:end]
        description_list.append(description)
    return description_list


def get_action(content):
    return content.split('Best Action:')[-1].strip(' :*."')


def simple_generate_response(results: list, system_prompt: str, history = [], grasping = False, model: str = 'gpt-4.1'):
    usr_content = []

    if grasping:
        usr_content.append({"type": "text", "text": f'The gripper is grasping the object now.'})

    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the obervation of future result {idx + 1}:"})
        image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def simple_select_view(images: list, system_prompt: str, history = None, examples = None, model: str = 'gpt-4.1',
                      temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if examples is not None:
        usr_content.append({"type": "text", "text": f"First we will show you some examples."})
        for idx, example in enumerate(examples):
            usr_content.append({"type": "text", "text": f"This is one example:"})
            usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
            for view_id, example_image in enumerate(example[1]):
                usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {view_id + 1} in this example:"})
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
            usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

        usr_content.append({"type": "text", "text": f"Below are the real observations you need to handle."})

    for idx, image in enumerate(images):
        usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {idx + 1}:"})
        concatenated_image = image[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def select_stage(images: list, system_prompt: str, grasping=None, history = None, examples = None, model: str = 'gpt-4.1',
                      temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if examples is not None:
        usr_content.append({"type": "text", "text": f"First we will show you some examples."})
        for idx, example in enumerate(examples):
            usr_content.append({"type": "text", "text": f"This is one example:"})
            usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
            for view_id, example_image in enumerate(example[1]):
                usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {view_id + 1} in this example:"})
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
            usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

        usr_content.append({"type": "text", "text": f"Below are the real observations you need to handle."})

    usr_content.append({"type": "text", "text": f"These are the image obervations of the current state from different views:"})
    if grasping is not None:
        if grasping:
            usr_content.append({"type": "text", "text": f"The gripper is grasping something now."})
        else:
            usr_content.append({"type": "text", "text": f"The gripper is not grasping anything now."})

    for idx, image in enumerate(images):
        concatenated_image = image[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_success(images: list, system_prompt: str, grasping=None, history = None, examples = None, model: str = 'gpt-4.1',
                      temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if examples is not None:
        usr_content.append({"type": "text", "text": f"First we will show you some examples."})
        for idx, example in enumerate(examples):
            usr_content.append({"type": "text", "text": f"This is one example:"})
            usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
            for view_id, example_image in enumerate(example[1]):
                usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {view_id + 1} in this example:"})
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
            usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

        usr_content.append({"type": "text", "text": f"Below are the real observations you need to handle."})

    usr_content.append({"type": "text", "text": f"These are the image obervations of the current state from different views:"})
    if grasping is not None:
        if grasping:
            usr_content.append({"type": "text", "text": f"The gripper is grasping something now."})
        else:
            usr_content.append({"type": "text", "text": f"The gripper is not grasping anything now."})

    for idx, image in enumerate(images):
        concatenated_image = image[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_subgoals(image, system_prompt: str, model: str = 'gpt-4.1',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if image is not None:
        if isinstance(image, list):
            usr_content.append({"type": "text", "text": f"These are the image obervations of the initial state:"})
            for img in image:
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}})
        else:
            usr_content.append({"type": "text", "text": f"This is the image obervation of the initial state:"})
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"Please break down the goal into sub-goals for robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_grasp(image, system_prompt: str, model: str = 'gpt-4.1',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    usr_content.append({"type": "text", "text": f"These are the image obervations after the grasping the object:"})
    for img in image:
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}})
    # usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"Please tell whether grasping this object align with the goal of the robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_release(image, system_prompt: str, model: str = 'gpt-4.1',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    # careful! not compatible with single image
    usr_content.append({"type": "text", "text": f"These are the image obervations after the releasing the object:"})
    for idx, img in enumerate(image):
        concatenated_image = img[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    # image = image[0]
    # usr_content.append({"type": "text", "text": f"This is the image obervation after the releasing the object:"})
    # usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"Please tell whether releasing this object align with the goal of the robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content


def generate_close_gripper(system_prompt: str, model: str = 'gpt-4.1',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    usr_content.append({"type": "text", "text": f'Do you think the robot should keep the gripper closed during the whole process?'})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    content = response['choices'][0]['message']['content']

    return content


def prompt_helper(group_id, queue, prompt, system_prompt, grasping=False):
    try_time = 0
    change = None
    answer = None
    while change is None and try_time < 5:
        try_time += 1
        try:
            content = simple_generate_response(prompt, system_prompt, grasping=grasping)
            answer = get_answer(content)
            change = True

        except Exception as e:
            print('catched', e)
            pass
    
    if change is None:
        print('Warning: failed to match format')
        answer = 1

    queue.put((group_id, answer, content))


def prompt_release_helper(release_id, queue, prompt, system_prompt):
    try_time = 0
    change = None
    release = None
    while change is None and try_time < 5:
        try_time += 1
        try:
            content = generate_release(prompt, system_prompt)
            
            release = get_release(content)
            change = True

        except Exception as e:
            print('catched', e)
            pass
    
    if change is None:
        print('Warning: failed to match format')
        release = False

    queue.put((release_id, release, content))


def generate_segment_names(system_prompt: str, image, instruction: str, model: str = 'gpt-4.1',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    usr_content.append({"type": "text", "text": f"The instruction of the task is: {instruction}"})
    # image = results[0][0]
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    # for idx, result in enumerate(results[1:]):
    #     usr_content.append({"type": "text", "text": f"This is the image containing multi-view observations of result {idx + 1}."})
    #     if motion_name_list is not None:
    #         usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
    #     concatenated_image = result[0]
    #     usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    # print(response)
    content = response['choices'][0]['message']['content']
    # print(content)

    return content
