
import openai
# from autogen.agentchat.contrib.teachable_agent import TeachableAgent
import os
import autogen
# from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, config_list_openai_aoai
from memgpt.autogen.memgpt_agent import create_autogen_memgpt_agent
from dotenv import load_dotenv
load_dotenv()
from memgpt.persistence_manager import InMemoryStateManager, InMemoryStateManagerWithPreloadedArchivalMemory, InMemoryStateManagerWithFaiss
import memgpt.humans.humans as humans

# openai.api_key = os.getenv("OPENAI_API_KEY")


# config_list = config_list_from_json(
#   env_or_file="OAI_CONFIG_LIST",
#   filter_dict={
#         "model": [
#             "gpt-3.5-turbo",
#             "gpt-3.5-turbo-16k",
#             "gpt-3.5-turbo-16k-0613",
#             "gpt-3.5-turbo-0301",
#             "chatgpt-35-turbo-0301",
#             "gpt-35-turbo-v0301",]
        
#     },
# )

config_list = [{
  "api_type" : "open_ai",
  "api_base" : "https://xr2dvy694ryym6-5001.proxy.runpod.net/v1",
  "api_key"  : "sk-11111111111111111111111111111111111111111111111",
  "model"    : "airoboros-l2-70b-2.1"
}]

USE_MEMGPT = True
persistence_manager = InMemoryStateManager()
llm_config = {
  "config_list": config_list, 
  "seed": 42,
  "request_timeout":300,
  "temperature":0.5
  
  }

david_ji = humans.get_human_text(key="david_ji")

# The user agent

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin.",
    code_execution_config=False,
    human_input_mode="TERMINATE"
)

executor = autogen.AssistantAgent(
  name="Executor",
  llm_config=llm_config,
  system_message="Executor. Write down a social post to Instagram, Youtube and Facebook based on content provided by Editor to file. The file is named as timestamp",
  code_execution_config={"last_n_messages": 2, "work_dir": "post"},
)



planner = autogen.AssistantAgent(
    name="Planner",
    system_message='''Planner. Suggest the idea for the tasks. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan need the involve of creator, writer and editor.
Explain the approved plan first. Be clear which step is performed by an Creator, and which step is performed by a Writer and Editor.

''',
    llm_config=llm_config,
)

creator = autogen.AssistantAgent(
    name="Creator",
    system_message='''Creator. Suggest the idea for the tasks. Based on the target audience, topic of interest provided by david, suggest the idea for social post.
    Revive the idea based on the feed back of david.
    Explain the approved idea to Writer and Editor.
''',
    llm_config=llm_config,
)

editor = autogen.AssistantAgent(
    name="Editor",
    system_message='''Editor. Based on the feedback, instruction of david, edit the content of post provided by writer until admin approval.
    Explain and send the approved to executor 
''',
    llm_config=llm_config,
)

writer = autogen.AssistantAgent(
  name="Writer",
  llm_config=llm_config,
  system_message='''Writer. Develop a social post to Instagram, Youtube and Facebook based on approved idea provided by planner.
  Explain the post to the davia and editor to help them edit your post.
  ''',
)
if not USE_MEMGPT:
  crisis = autogen.AssistantAgent(
    name="Research Agent",
    system_message="Research web and video platforms for content related to the generated topics.",
    llm_config=llm_config
  )
 
else:

  crisis = create_autogen_memgpt_agent(
    "david",
    persona_description=david_ji,
    user_description= f'''You are participating in a group chat with user_admin ({user_proxy.name}), executor ({executor.name}), planner ({planner.name}), writer ({writer.name}) and editor ({editor.name}).
    You will provide your topic of interest and target audience to creator to help him suggest idea for social post.
    Based on your persona and you interest, choose an idea from creator to send to writer to develop a post content.
    Get the post content developed by writer. Listen to writer for the post explanation.
    Based on the post content and explanation from writer and your persona, revive and give feedback and instruction to editor to edit the post.
    ''',
    # interface_kwargs={"debug": True},
    model="dolphin-2.1-mistral-7b",
    persistence_manager=persistence_manager,
    # model="gpt-3.5-turbo"
    # model="airoboros-l2-70b-2.1"
  )

groupchat = autogen.GroupChat(agents=[user_proxy, planner, creator, crisis, writer,  editor, executor], messages=[], max_round=100)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Begin the group chat with a message from the user
user_proxy.initiate_chat(
    manager,
    message="I want to have the social media post that can reach and engage milions of people.",
)

