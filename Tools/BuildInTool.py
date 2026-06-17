#duckduck go search engine

from langchain_community.tools import DuckDuckGoSearchRun

search=DuckDuckGoSearchRun()
result=search.invoke("what is the date and time")
print(result)
print("name of the tool: ",search.name)
print("Description of the tool: ",search.description)
print("Argument passed to the tool: ",search.args)
print("Schema of about our tool sending to LLM: ", search.args_schema.model_json_schema()) # to see schema of out tool

# To exacute shell commands
from langchain_community.tools import ShellTool
shell=ShellTool()

command=shell.invoke("whoami")
print(command)

print("name of the tool: ",shell.name)
print("Description of the tool: ",shell.description)
print("Argument passed to the tool: ",shell.args)
print("Schema of about our tool sending to LLM: ", shell.args_schema.model_json_schema()) # to see schema of out tool


